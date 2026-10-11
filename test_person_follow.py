"""Following logic and app wiring without cameras, network, or flight."""
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import numpy as np

import config
import main
from controller import Controller, HOVER


def person(**changes):
    return dict(dict(found=True, x=.5, y=.5, w=.2, h=.4, conf=.9), **changes)


class FollowTests(unittest.TestCase):
    def setUp(self):
        self.ctrl = Controller(person_mode=True, follow_mode=True)
        self.clock = patch('controller.time.monotonic', return_value=10.)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)

    def moving(self):
        self.assertTrue(self.ctrl.arm_follow(person()))
        small = person(w=.1, h=.2)
        self.assertEqual(self.ctrl.update(small)[0]['forward'], 0)
        self.now.return_value += .5
        cmd, state = self.ctrl.update(small)
        self.assertGreater(cmd['forward'], 0)
        self.assertEqual(state, 'APPROACHING')
        return cmd

    def test_explicit_arming_and_receding_target_required(self):
        self.assertEqual(self.ctrl.update(person(w=.1, h=.2))[0], HOVER)
        self.assertTrue(self.ctrl.arm_follow(person()))
        for _ in range(10):
            self.now.return_value += 1
            self.assertEqual(self.ctrl.update(person())[0], HOVER)
        cmd = self.moving()
        self.assertLessEqual(cmd['forward'], .08)
        self.assertEqual(cmd['up'], 0)
        self.assertEqual(cmd['right'], 0)

    def test_loss_uncertainty_invalid_and_close_target_stop_immediately(self):
        for target in (dict(found=False), person(conf=.59), person(x=float('nan')),
                       person(w=.31), person(h=.66), person(w=0)):
            with self.subTest(target=target):
                self.moving()
                self.assertEqual(self.ctrl.update(target)[0], HOVER)
                self.assertEqual(self.ctrl.update(person(w=.1, h=.2))[0]['forward'], 0)

    def test_alignment_and_both_dimensions_are_required(self):
        for target in (person(x=.7, w=.1, h=.2), person(y=.85, w=.1, h=.2),
                       person(y=.05, w=.1, h=.2), person(w=.1), person(h=.2)):
            with self.subTest(target=target):
                self.moving()
                self.assertEqual(self.ctrl.update(target)[0]['forward'], 0)
                self.assertEqual(self.ctrl.update(person(w=.1, h=.2))[0]['forward'], 0)

    def test_lower_forward_cap_and_return_to_starting_gap(self):
        with patch.object(config, 'MAX_FORWARD', .025):
            self.assertLessEqual(self.moving()['forward'], .025)
        self.assertEqual(self.ctrl.update(person())[0], HOVER)
        self.ctrl.disarm_follow()
        self.now.return_value += 10
        self.assertEqual(self.ctrl.update(person(w=.1, h=.2))[0], HOVER)

    def test_cannot_arm_on_close_partial_or_uncertain_person(self):
        for target in (person(w=.31), person(h=.7), person(x=.02), person(conf=.59), dict(found=False)):
            self.assertFalse(self.ctrl.arm_follow(target))
        self.assertFalse(Controller(person_mode=True).arm_follow(person()))
        with self.assertRaises(ValueError):
            Controller(follow_mode=True)

    def test_app_starts_off_and_space_stops_follow_preview(self):
        video, link, worker = Mock(), Mock(), Mock()
        video.kind, video.fresh = 'drone', True
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        ticks = iter([10., 11., 11.5, 12., 12.5])
        def read():
            self.now.return_value = next(ticks)
            return frame
        video.read.side_effect = read
        worker.find.side_effect = [person()] + [person(w=.1, h=.2)] * 4
        with patch.object(config, 'CONTROL_PORT', None), \
             patch.object(main, 'PersonTracker'), \
             patch.object(main, 'AsyncPersonTracker', return_value=worker), \
             patch.object(main, 'VideoSource', return_value=video), \
             patch.object(main, 'DroneLink', return_value=link), \
             patch.object(main.voice, 'announce'), \
             patch.object(main.cv2, 'namedWindow'), \
             patch.object(main.cv2, 'setMouseCallback'), \
             patch.object(main.cv2, 'imshow'), \
             patch.object(main.cv2, 'setWindowTitle'), \
             patch.object(main.cv2, 'waitKey', side_effect=[ord('a'), -1, -1, ord(' '), ord('q')]), \
             patch.object(main.cv2, 'getWindowProperty', return_value=1), \
             patch.object(main.cv2, 'destroyAllWindows'):
            main.main(follow=True)
        self.assertEqual(link.method_calls[1][0], 'hover')
        self.assertEqual(link.send.call_count, 3)
        self.assertEqual(link.send.call_args_list[0].args[0], HOVER)
        self.assertGreater(link.send.call_args_list[1].args[0]['forward'], 0)
        for call in link.send.call_args_list:
            self.assertEqual(call.args[0]['right'], 0)
            self.assertEqual(call.args[0]['up'], 0)
        link.land.assert_called_once()
        link.close.assert_called_once()

    def test_reselection_and_a_in_same_ui_cycle_cannot_arm_an_old_target(self):
        video, link, worker = Mock(), Mock(), Mock()
        video.kind, video.fresh = 'drone', True
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        video.read.return_value = frame
        worker.find.return_value = person()
        callbacks = []
        keys = iter([ord('a'), ord('q')])
        def keypress(_):
            key = next(keys)
            if key == ord('a'):
                callbacks[0](main.cv2.EVENT_LBUTTONDOWN, 320, 180, None, None)
            return key
        with ExitStack() as stack:
            for context in (
                patch.object(config, 'CONTROL_PORT', None),
                patch.object(main, 'PersonTracker'),
                patch.object(main, 'AsyncPersonTracker', return_value=worker),
                patch.object(main, 'Controller', return_value=self.ctrl),
                patch.object(main, 'VideoSource', return_value=video),
                patch.object(main, 'DroneLink', return_value=link),
                patch.object(main.voice, 'announce'),
                patch.object(main.cv2, 'namedWindow'),
                patch.object(main.cv2, 'setMouseCallback', side_effect=lambda _, cb: callbacks.append(cb)),
                patch.object(main.cv2, 'imshow'),
                patch.object(main.cv2, 'setWindowTitle'),
                patch.object(main.cv2, 'waitKey', side_effect=keypress),
                patch.object(main.cv2, 'getWindowProperty', return_value=1),
                patch.object(main.cv2, 'destroyAllWindows'),
            ):
                stack.enter_context(context)
            main.main(follow=True)
        worker.set_color_from_pixel.assert_called_once()
        self.assertIsNone(self.ctrl._follow_reference)
        link.send.assert_not_called()

    def test_live_follow_rejected_before_any_hardware_or_model_is_opened(self):
        with patch.object(config, 'CONTROL_PORT', 8800), \
             patch.object(main, 'VideoSource') as video, \
             patch.object(main, 'DroneLink') as link, \
             patch.object(main, 'PersonTracker') as tracker:
            with self.assertRaisesRegex(RuntimeError, 'Live follow is disabled'):
                main.main(follow=True)
        video.assert_not_called()
        link.assert_not_called()
        tracker.assert_not_called()

    def run_click_flow(self, actions, targets, accepted=True):
        """Drive the UI without a camera, real network, or background inference."""
        video, link, worker = Mock(), Mock(), Mock()
        video.kind, video.fresh = 'drone', True
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        worker.set_color_from_pixel.return_value = accepted
        callbacks, sent, shown = [], [], []
        step = [-1]

        def read():
            step[0] += 1
            self.now.return_value = 10. + step[0]
            worker.find.return_value = targets[step[0]]
            return frame

        def keypress(_):
            action = actions[step[0]]
            if action == 'click':
                callbacks[0](main.cv2.EVENT_LBUTTONDOWN, 320, 180, None, None)
                return -1
            return ord(action) if action else -1

        def draw(frame, target, state, command, autopilot, fps):
            shown.append((step[0], dict(command), autopilot))
            return frame.copy()

        video.read.side_effect = read
        link.send.side_effect = lambda cmd: sent.append((step[0], dict(cmd)))
        with ExitStack() as stack:
            for context in (
                patch.object(config, 'CONTROL_PORT', None),
                patch.object(config, 'VIDEO_SOURCE', 'drone'),
                patch.object(main, 'PersonTracker'),
                patch.object(main, 'AsyncPersonTracker', return_value=worker),
                patch.object(main, 'Controller', return_value=self.ctrl),
                patch.object(main, 'VideoSource', return_value=video),
                patch.object(main, 'DroneLink', return_value=link),
                patch.object(main.voice, 'announce'),
                patch.object(main.hud, 'draw', side_effect=draw),
                patch.object(main.cv2, 'namedWindow'),
                patch.object(main.cv2, 'setMouseCallback', side_effect=lambda _, cb: callbacks.append(cb)),
                patch.object(main.cv2, 'imshow'),
                patch.object(main.cv2, 'setWindowTitle'),
                patch.object(main.cv2, 'waitKey', side_effect=keypress),
                patch.object(main.cv2, 'getWindowProperty', return_value=1),
                patch.object(main.cv2, 'destroyAllWindows'),
            ):
                stack.enter_context(context)
            main.main(follow=True)
        return link, sent, shown

    def test_preview_click_waits_for_confirmation_before_following(self):
        targets = [dict(found=False)] * 4 + [person(), person(w=.1, h=.2), person(w=.1, h=.2), person()]
        link, sent, shown = self.run_click_flow(
            ['t', 'click', None, None, None, None, ' ', 'q'], targets)
        self.assertEqual([step for step, _ in sent], [4, 5, 6])
        self.assertEqual(sent[0][1], HOVER)
        self.assertEqual(sent[1][1], HOVER)  # forward confirmation dwell
        self.assertGreater(sent[2][1]['forward'], 0)
        self.assertEqual(sent[2][1]['right'], 0)
        for step, command, enabled in shown:
            if step < 4 or step == 7:
                self.assertFalse(enabled)
                self.assertEqual(command, HOVER)
        link.takeoff.assert_called_once()
        link.land.assert_called_once()

    def test_cancelled_or_rejected_click_never_starts_later(self):
        for cancel in ('h', 'H', ' ', 'k', 't', 'a'):
            with self.subTest(cancel=cancel):
                self.ctrl.disarm_follow()
                _, sent, _ = self.run_click_flow(
                    ['click', cancel, None, 'q'], [dict(found=False)] * 2 + [person()] * 2)
                self.assertEqual(sent, [])
                self.assertIsNone(self.ctrl._follow_reference)
        _, sent, _ = self.run_click_flow(['click', None, 'q'], [person()] * 3, accepted=None)
        self.assertEqual(sent, [])


if __name__ == '__main__':
    unittest.main()
