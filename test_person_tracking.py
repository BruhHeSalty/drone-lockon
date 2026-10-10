"""Offline person-mode regressions: no model download, camera, or drone."""
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

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

    def test_live_tracking_never_reacquires_another_id(self):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker.allow_live_yaw = True
        tracker._tracks = {}
        tracker._selected_id = 7
        tracker._require_click = True
        tracker._consistent = 3
        tracker._pending_id = 7
        tracker._reference = np.ones(3)
        tracker._previous_live = None
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
        tracker.model.track.return_value = [types.SimpleNamespace(boxes=boxes)]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(tracker, '_choose_match', return_value=8):
            self.assertFalse(tracker.find(frame)['found'])
        self.assertIsNone(tracker._selected_id)
        self.assertIsNone(tracker._reference)
        self.assertFalse(tracker.find(frame)['found'])

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
             patch.object(main.hud, 'draw', return_value=frame), \
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
        worker.set_color_from_pixel(displayed, 80, 20)
        self.assertEqual(worker._click[0], 7)
        self.assertEqual(worker._generation, 1)
        self.assertFalse(worker._result[1]['found'])
        detector.find.assert_not_called()

    def test_same_clothing_elsewhere_does_not_break_selected_id(self):
        tracker = PersonTracker.__new__(PersonTracker)
        tracker.allow_live_yaw = True
        tracker._reference = np.ones(3)
        tracker._selected_id = 7
        tracker._tracks = {7: target(x=.2), 8: target(x=.8)}
        with patch.object(tracker, '_appearance', return_value=np.ones(3)):
            self.assertEqual(tracker._choose_match(np.zeros((100, 100, 3))), 7)
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
        self.assertIsNone(tracker._selected_id)

    def test_invalid_model_settings_fail_before_loading_model(self):
        with self.assertRaises(ValueError):
            PersonTracker(imgsz=641)
        with self.assertRaises(ValueError):
            PersonTracker(tracker='unknown.yaml')


if __name__ == '__main__':
    unittest.main()
