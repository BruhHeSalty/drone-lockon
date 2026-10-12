"""Offline person-mode regressions: no model download, camera, or drone."""
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import config
import main
from controller import Controller, HOVER
from vision import AsyncPersonTracker, PersonTracker, NOT_FOUND


def target(**changes):
    return dict(dict(found=True, x=.8, y=.2, w=.1, h=.3, conf=.9), **changes)


class PersonTests(unittest.TestCase):
    def test_person_commands_never_translate_and_yaw_is_capped(self):
        ctrl = Controller(person_mode=True)
        for x in (0., .2, .5, .8, 1.):
            for y in (0., .5, 1.):
                cmd, _ = ctrl.update(target(x=x, y=y))
                self.assertEqual([cmd[k] for k in ('forward', 'right', 'up')], [0, 0, 0])
                self.assertLessEqual(abs(cmd['yaw']), .15)
                self.assertEqual(np.sign(cmd['yaw']), np.sign(x - .5))

    def test_close_lost_invalid_and_uncertain_people_hover(self):
        ctrl = Controller(person_mode=True)
        for value in (target(w=.4), target(h=.8), dict(found=False), target(conf=.59),
                      target(x=float('nan')), target(w=0), target(x=1.1)):
            self.assertEqual(ctrl.update(value)[0], HOVER)

    def test_lower_yaw_cap_is_respected(self):
        with patch.object(config, 'MAX_YAW', .04):
            self.assertEqual(Controller(person_mode=True).update(target())[0]['yaw'], .04)

    def test_selection_survives_leaving_frame_and_reacquires_new_id(self):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker.allow_live_yaw = True
        tracker._tracks = {}
        tracker._selected_id = 7
        tracker._require_click = True
        tracker._consistent = 3
        tracker._pending_id = 7
        tracker._reference = np.ones(3)
        tracker._previous_live = None
        tracker._visible = True
        tracker._last_confirmed = time.monotonic()
        tracker.tracker_config = "botsort.yaml"
        tracker.person_class = 0
        tracker.imgsz = 640
        tracker.device = "cpu"
        boxes = Mock()
        boxes.id.cpu.return_value.tolist.return_value = [8]
        boxes.xyxy.cpu.return_value.tolist.return_value = [[10, 10, 30, 60]]
        boxes.conf.cpu.return_value.tolist.return_value = [.9]
        tracker.model = Mock()
        tracker.model.track.return_value = [types.SimpleNamespace(boxes=None)]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        reference = tracker._reference
        with patch('vision.time.monotonic', return_value=tracker._last_confirmed + 3600):
            for _ in range(5):
                result = tracker.find(frame)
                self.assertFalse(result['found'])
                self.assertEqual(Controller(person_mode=True).update(result)[0], HOVER)
        self.assertEqual(tracker._selected_id, 7)
        self.assertIs(tracker._reference, reference)
        self.assertIn('remembered', tracker.status)
        tracker.model.track.return_value = [types.SimpleNamespace(boxes=boxes)]
        with patch.object(tracker, '_appearance', return_value=np.ones(3)):
            self.assertFalse(tracker.find(frame)['found'])
            self.assertFalse(tracker.find(frame)['found'])
            self.assertTrue(tracker.find(frame)['found'])
        self.assertEqual(tracker._selected_id, 8)
        self.assertIs(tracker._reference, reference)

    def test_slow_detection_does_not_block_and_result_age_includes_inference(self):
        entered, release = threading.Event(), threading.Event()
        detector = Mock()
        detector._tracks = {}
        detector.status = "Testing"
        def detect(frame):
            entered.set()
            release.wait(2)
            return target()
        detector.find.side_effect = detect
        worker = AsyncPersonTracker(detector)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        try:
            self.assertFalse(worker.find(frame)['found'])
            self.assertTrue(entered.wait(1))
            # The second UI read returns while inference remains blocked.
            self.assertFalse(worker.find(frame)['found'])
            with worker._condition:
                worker._result = (1., target())
            with patch('vision.time.monotonic', return_value=1.31):
                self.assertFalse(worker.find(frame)['found'])
            worker.set_color_from_pixel(frame, 5, 5)
            self.assertIsNone(worker._click)  # No fresh visible box to select.
        finally:
            worker.close()
            release.set()
            worker._thread.join(1)
        self.assertFalse(worker._thread.is_alive())

    def test_live_person_loop_starts_off_turns_only_and_lands_without_video(self):
        video, link, worker = Mock(), Mock(), Mock()
        video.kind = 'drone'
        video.fresh = True
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        video.read.side_effect = [frame, frame, None, None]
        worker.find.return_value = target()
        with patch.object(config, 'VIDEO_SOURCE', 'drone'), \
             patch.object(config, 'CONTROL_PORT', 8800), \
             patch.object(main, 'PersonTracker'), \
             patch.object(main, 'AsyncPersonTracker', return_value=worker), \
             patch.object(main, 'VideoSource', return_value=video), \
             patch.object(main, 'DroneLink', return_value=link), \
             patch.object(main.voice, 'announce'), \
             patch.object(main.hud, 'draw', return_value=frame) as draw, \
             patch.object(main.cv2, 'namedWindow'), \
             patch.object(main.cv2, 'setMouseCallback'), \
             patch.object(main.cv2, 'imshow'), \
             patch.object(main.cv2, 'setWindowTitle'), \
             patch.object(main.cv2, 'waitKey', side_effect=[ord('a'), -1, ord(' '), ord('q')]), \
             patch.object(main.cv2, 'getWindowProperty', return_value=1), \
             patch.object(main.cv2, 'destroyAllWindows'):
            main.main(person=True)
        link.send.assert_called_once_with(dict(yaw=.15, up=0., forward=0., right=0.))
        self.assertEqual(link.method_calls[1][0], 'hover')  # First frame: autopilot off.
        link.land.assert_called_once()
        link.close.assert_called_once()
        worker.close.assert_called_once()
        # Dropped camera frames still render the controls/status, but they must
        # not retain a visible lock or an old steering command.
        self.assertEqual(draw.call_count, 4)
        for call in draw.call_args_list[2:]:
            self.assertFalse(call.args[1]['found'])
            self.assertEqual(call.args[2], 'LOST')
            self.assertEqual(call.args[3], HOVER)
        self.assertEqual([c.kwargs['video_fresh'] for c in worker.draw_candidates.call_args_list],
                         [True, True, False, False])

    def test_capture_delay_counts_toward_detection_expiry(self):
        detector = Mock()
        with patch('vision.threading.Thread'):
            worker = AsyncPersonTracker(detector)
        with patch('vision.time.monotonic', return_value=10.):
            worker._result = (9.9, target())
            self.assertFalse(worker.find(np.zeros((10, 10, 3)), frame_time=9.69)['found'])
        self.assertIsNone(worker._pending)

    def test_click_selects_snapshot_without_replaying_an_old_frame(self):
        detector = Mock()
        detector._appearance.return_value = np.ones(3)
        with patch('vision.threading.Thread'):
            worker = AsyncPersonTracker(detector)
        displayed = np.zeros((100, 100, 3), dtype=np.uint8)
        worker._result = (time.monotonic(), dict(NOT_FOUND))
        worker._snapshot = (displayed, {7: target()}, 'Select')
        worker.draw_candidates(displayed.copy())
        self.assertIs(worker.set_color_from_pixel(displayed, 80, 20), True)
        self.assertEqual(worker._click[0], 7)
        self.assertEqual(worker._generation, 1)
        self.assertFalse(worker._result[1]['found'])
        detector.find.assert_not_called()

    def test_click_uses_displayed_box_when_worker_has_already_moved_on(self):
        detector = Mock()
        detector._appearance.return_value = np.ones(3)
        with patch('vision.threading.Thread'):
            worker = AsyncPersonTracker(detector)
        displayed = np.zeros((100, 100, 3), dtype=np.uint8)
        worker._result = (time.monotonic(), dict(NOT_FOUND))
        worker._snapshot = (displayed, {7: target()}, 'Select')
        worker.draw_candidates(displayed.copy())
        # Inference finishes between imshow and the mouse callback. Its person
        # is elsewhere; that must not change the meaning of the visible box.
        worker._snapshot = (displayed.copy(), {8: target(x=.2)}, 'Select')
        worker._result = (time.monotonic(), dict(NOT_FOUND))
        worker.set_color_from_pixel(displayed, 80, 20)
        self.assertEqual(worker._click[0], 7)
        self.assertIs(detector._appearance.call_args.args[0], displayed)
        self.assertIn('Selected person 7', worker._click_notice)
        self.assertFalse(worker._result[1]['found'])

    def test_fresh_worker_result_cannot_make_an_expired_display_clickable(self):
        detector = Mock()
        with patch('vision.threading.Thread'):
            worker = AsyncPersonTracker(detector)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        worker._result = (10., dict(NOT_FOUND))
        worker._snapshot = (frame, {7: target()}, 'Select')
        with patch('vision.time.monotonic', return_value=10.1):
            worker.draw_candidates(frame.copy())
        worker._result = (10.4, dict(NOT_FOUND))
        with patch('vision.time.monotonic', return_value=10.4):
            self.assertFalse(worker.set_color_from_pixel(frame, 80, 20))
        self.assertIsNone(worker._click)
        detector._appearance.assert_not_called()

    def test_video_gap_clears_clickable_overlay_even_with_a_recent_detection(self):
        detector = Mock()
        with patch('vision.threading.Thread'):
            worker = AsyncPersonTracker(detector)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        worker._result = (time.monotonic(), dict(NOT_FOUND))
        worker._snapshot = (frame, {7: target()}, 'Select')
        worker.draw_candidates(frame.copy())
        self.assertIsNotNone(worker._displayed)
        worker.draw_candidates(frame.copy(), video_fresh=False)
        worker.set_color_from_pixel(frame, 80, 20)
        self.assertIsNone(worker._displayed)
        self.assertIsNone(worker._click)
        detector._appearance.assert_not_called()

    def test_same_clothing_elsewhere_does_not_break_selected_id(self):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker.allow_live_yaw = True
        tracker._reference = np.ones(3)
        tracker._selected_id = 7
        tracker._visible = True
        tracker._tracks = {7: target(x=.2), 8: target(x=.8)}
        with patch.object(tracker, '_appearance', return_value=np.ones(3)):
            self.assertEqual(tracker._choose_match(np.zeros((100, 100, 3))), 7)
            tracker._visible = False  # Once lost, same clothes are ambiguous.
            self.assertIsNone(tracker._choose_match(np.zeros((100, 100, 3))))
            tracker._visible = True
            tracker._tracks[8] = target(x=.2)
            self.assertIsNone(tracker._choose_match(np.zeros((100, 100, 3))))

    def test_weak_detection_hovers_then_requires_three_strong_frames(self):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker.allow_live_yaw = True
        tracker.tracker_config, tracker.person_class = 'botsort.yaml', 0
        tracker.imgsz, tracker.device = 640, 'cpu'
        tracker._require_click = True
        tracker._tracks = {}
        tracker._selected_id = 7
        tracker._consistent = 3
        tracker._pending_id = 7
        tracker._reference = np.ones(3)
        tracker._previous_live = None
        tracker._visible = True
        tracker._last_confirmed = time.monotonic()
        boxes = Mock()
        boxes.id.cpu.return_value.tolist.return_value = [7]
        boxes.xyxy.cpu.return_value.tolist.return_value = [[10, 10, 30, 60]]
        boxes.conf.cpu.return_value.tolist.return_value = [.3]
        tracker.model = Mock()
        tracker.model.track.return_value = [types.SimpleNamespace(boxes=boxes)]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertFalse(tracker.find(frame)['found'])
        self.assertEqual(tracker._selected_id, 7)
        self.assertEqual(tracker.model.track.call_args.kwargs['conf'], .1)
        boxes.conf.cpu.return_value.tolist.return_value = [.9]
        with patch.object(tracker, '_choose_match', return_value=7):
            self.assertFalse(tracker.find(frame)['found'])
            self.assertFalse(tracker.find(frame)['found'])
            self.assertTrue(tracker.find(frame)['found'])
        boxes.conf.cpu.return_value.tolist.return_value = [.3]
        with patch('vision.time.monotonic', return_value=tracker._last_confirmed + .51):
            self.assertFalse(tracker.find(frame)['found'])
        self.assertEqual(tracker._selected_id, 7)
        self.assertIsNotNone(tracker._reference)

    def test_reacquisition_rejects_wrong_appearance_and_weak_candidates(self):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker._reference = np.array([1., 0.])
        tracker._selected_id = 7
        tracker._visible = False
        tracker._tracks = {8: target()}
        frame = np.zeros((100, 100, 3))
        with patch.object(tracker, '_appearance', return_value=np.array([0., 1.])):
            self.assertIsNone(tracker._choose_match(frame))
        tracker._tracks[8] = target(conf=.3)
        with patch.object(tracker, '_appearance', return_value=np.array([1., 0.])):
            self.assertIsNone(tracker._choose_match(frame))
        self.assertEqual(tracker._selected_id, 7)
        np.testing.assert_array_equal(tracker._reference, [1., 0.])

    def test_explicit_reselection_replaces_remembered_target(self):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker._selected_id = 7
        tracker._reference = np.array([1., 0.])
        tracker.select_track(8, target(), np.array([0., 1.]))
        self.assertEqual(tracker._selected_id, 8)
        self.assertEqual(tracker._consistent, 0)
        np.testing.assert_array_equal(tracker._reference, [0., 1.])

    def test_invalid_model_settings_fail_before_loading_model(self):
        with self.assertRaises(ValueError):
            PersonTracker(imgsz=641)
        with self.assertRaises(ValueError):
            PersonTracker(tracker='unknown.yaml')


class AppearanceLightingTests(unittest.TestCase):
    @staticmethod
    def feature(hue, saturation=160):
        hsv = np.full((100, 100, 3), (hue, saturation, 160), dtype=np.uint8)
        return PersonTracker._appearance(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
                                         target(x=.5, y=.5, w=1., h=1.))

    def test_neighboring_hues_are_closer_than_a_different_outfit(self):
        # Tiny color changes must not cost as much as blue-to-green changes.
        # Include red on both sides of OpenCV's circular hue boundary.
        for reference, neighbor, different in ((9, 10, 90), (179, 0, 60)):
            with self.subTest(hue=reference):
                anchor = self.feature(reference)
                near = np.linalg.norm(anchor - self.feature(neighbor))
                far = np.linalg.norm(anchor - self.feature(different))
                self.assertLess(near, .75 * far)
                self.assertGreater(far / np.sqrt(2), PersonTracker.MAX_APPEARANCE_DISTANCE)

    def test_similar_hue_does_not_erase_saturation_difference(self):
        distance = np.linalg.norm(self.feature(110, 60) - self.feature(110, 200)) / np.sqrt(2)
        self.assertGreater(distance, PersonTracker.MAX_APPEARANCE_DISTANCE)

    def test_exposure_change_preserves_relative_body_appearance(self):
        gradient = np.tile(np.linspace(60, 220, 100, dtype=np.uint8), (100, 1))
        image = cv2.merge((gradient, gradient, gradient))
        darker = (image.astype(np.float32) * .5).astype(np.uint8)
        box = target(x=.5, y=.5, w=1., h=1.)
        distance = np.linalg.norm(PersonTracker._appearance(image, box)
                                  - PersonTracker._appearance(darker, box)) / np.sqrt(2)
        self.assertLess(distance, .1)

    def test_flat_dark_and_clipped_crops_remain_finite(self):
        for brightness in (0, 255):
            image = np.full((100, 100, 3), brightness, dtype=np.uint8)
            feature = PersonTracker._appearance(image, target(x=.05, y=.05, w=.3, h=.3))
            self.assertTrue(np.isfinite(feature).all())
        self.assertIsNone(PersonTracker._appearance(image, target(w=0)))


class AppearanceMemoryTests(unittest.TestCase):
    def setUp(self):
        clock = patch('vision.time.monotonic', return_value=10.)
        clock.start()
        self.addCleanup(clock.stop)
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.tracker = tracker = PersonTracker.__new__(PersonTracker)
        tracker.allow_live_yaw = True
        tracker._tracks = {}
        tracker.tracker_config, tracker.person_class = 'botsort.yaml', 0
        tracker.imgsz, tracker.device = 640, 'cpu'
        self.boxes = Mock()
        self.boxes.id.cpu.return_value.tolist.return_value = [7]
        self.boxes.xyxy.cpu.return_value.tolist.return_value = [[10, 10, 30, 60]]
        self.boxes.conf.cpu.return_value.tolist.return_value = [.9]
        tracker.model = Mock()
        tracker.model.track.return_value = [types.SimpleNamespace(boxes=self.boxes)]
        tracker._appearance = Mock(return_value=np.array([1., 0.]))
        tracker.select_track(7, target(x=.2, y=.35, w=.2, h=.5), np.array([1., 0.]))

    def test_gradual_pose_change_keeps_lock_without_overwriting_click(self):
        tracker = self.tracker
        anchor = tracker._reference.copy()
        for index, angle in enumerate(np.linspace(0., .9, 60)):
            tracker._appearance.return_value = np.array([np.cos(angle), np.sin(angle)])
            result = tracker.find(self.frame)
            self.assertEqual(result['found'], index >= 2)
        np.testing.assert_array_equal(tracker._reference, anchor)
        # The final view differs too much from the initial click alone.
        distance = np.linalg.norm(tracker._appearance.return_value - anchor) / np.sqrt(2)
        self.assertGreater(distance, tracker.MAX_APPEARANCE_DISTANCE)
        self.assertGreater(tracker._recent_reference[1], .5)

    def test_wrong_appearance_cannot_teach_the_same_id(self):
        tracker = self.tracker
        for _ in range(3):
            tracker.find(self.frame)
        recent = tracker._recent_reference.copy()
        tracker._appearance.return_value = np.array([0., 1.])
        for _ in range(5):
            self.assertFalse(tracker.find(self.frame)['found'])
        np.testing.assert_array_equal(tracker._recent_reference, recent)

    def test_reacquisition_can_use_recent_view_or_original_click_after_loss(self):
        tracker = self.tracker
        for angle in np.linspace(0., .9, 60):
            tracker._appearance.return_value = np.array([np.cos(angle), np.sin(angle)])
            tracker.find(self.frame)
        for new_id, appearance in ((8, tracker._appearance.return_value), (9, np.array([1., 0.]))):
            recent = tracker._recent_reference.copy()
            tracker.model.track.return_value = [types.SimpleNamespace(boxes=None)]
            self.assertFalse(tracker.find(self.frame)['found'])
            tracker.model.track.return_value = [types.SimpleNamespace(boxes=self.boxes)]
            self.boxes.id.cpu.return_value.tolist.return_value = [new_id]
            tracker._appearance.return_value = appearance
            for expected in (False, False, True):
                self.assertEqual(tracker.find(self.frame)['found'], expected)
                np.testing.assert_array_equal(tracker._recent_reference, recent)
            self.assertEqual(tracker._selected_id, new_id)

    def test_loss_weak_detection_and_new_position_freeze_learning(self):
        tracker = self.tracker
        for _ in range(3):
            tracker.find(self.frame)
        recent = tracker._recent_reference.copy()
        tracker._appearance.return_value = np.array([.98, .2])
        self.boxes.conf.cpu.return_value.tolist.return_value = [.3]
        self.assertFalse(tracker.find(self.frame)['found'])
        np.testing.assert_array_equal(tracker._recent_reference, recent)
        tracker.model.track.return_value = [types.SimpleNamespace(boxes=None)]
        self.assertFalse(tracker.find(self.frame)['found'])
        np.testing.assert_array_equal(tracker._recent_reference, recent)
        tracker.model.track.return_value = [types.SimpleNamespace(boxes=self.boxes)]
        self.boxes.conf.cpu.return_value.tolist.return_value = [.9]
        for expected in (False, False, True):
            self.assertEqual(tracker.find(self.frame)['found'], expected)
            np.testing.assert_array_equal(tracker._recent_reference, recent)
        # A sudden position jump must fail without adapting appearance.
        self.boxes.xyxy.cpu.return_value.tolist.return_value = [[65, 10, 85, 60]]
        self.assertFalse(tracker.find(self.frame)['found'])
        np.testing.assert_array_equal(tracker._recent_reference, recent)

    def test_second_person_prevents_learning_even_while_selected_id_is_visible(self):
        tracker = self.tracker
        for _ in range(3):
            tracker.find(self.frame)
        recent = tracker._recent_reference.copy()
        tracker._appearance.return_value = np.array([.98, .2])
        self.boxes.id.cpu.return_value.tolist.return_value = [7, 8]
        self.boxes.xyxy.cpu.return_value.tolist.return_value = [[10, 10, 30, 60], [65, 10, 85, 60]]
        self.boxes.conf.cpu.return_value.tolist.return_value = [.9, .9]
        self.assertTrue(tracker.find(self.frame)['found'])
        np.testing.assert_array_equal(tracker._recent_reference, recent)
        tracker._visible = False
        self.assertFalse(tracker.find(self.frame)['found'])  # Ambiguous after loss.

    def test_reselection_discards_old_recent_appearance(self):
        self.tracker._recent_reference = np.array([.8, .6])
        self.tracker.select_track(8, target(), np.array([0., 1.]))
        np.testing.assert_array_equal(self.tracker._reference, [0., 1.])
        np.testing.assert_array_equal(self.tracker._recent_reference, [0., 1.])
        self.assertEqual(self.tracker._consistent, 0)


if __name__ == '__main__':
    unittest.main()
