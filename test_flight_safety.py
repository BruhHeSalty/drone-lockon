"""Offline flight-safety regression tests. No camera, socket, or drone is opened."""
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

import config
import drone
import main
from controller import Controller


class FakeSocket:
    def __init__(self):
        self.fail = False
        self.sent = []

    def sendto(self, packet, address):
        if self.fail:
            raise OSError('simulated closed socket')
        self.sent.append((packet, address))
        return 1


class FakeAdapter:
    def __init__(self, *args, **kwargs):
        self.sock = FakeSocket()
        self._target_ports = (8800, 8801)
        self.drone_ip = 'test-drone'

    def build_control_packet(self, sticks):
        packet = vars(sticks).copy()
        for name in ('takeoff_flag', 'land_flag', 'stop_flag', 'calibration_flag'):
            setattr(sticks, name, False)
        return packet

    def set_socket(self, sock):
        self.sock = sock

    def stop(self):
        pass


class FlightTests(unittest.TestCase):
    def setUp(self):
        module = types.ModuleType('protocols.wifi_uav_rc_protocol_adapter')
        module.WifiUavRcProtocolAdapter = FakeAdapter
        for context in (
            patch.dict('sys.modules', {'protocols.wifi_uav_rc_protocol_adapter': module}),
            patch.object(drone, '_load_turbodrone'),
            patch.object(drone, '_shared_sock', None),
            patch.object(config, 'CONTROL_PORT', 8800),
            patch.object(drone.threading, 'Thread'),
        ):
            context.start()
            self.addCleanup(context.stop)
        clock = patch.object(drone.time, 'monotonic', return_value=10.0)
        self.clock = clock.start()
        self.addCleanup(clock.stop)
        self.link = drone.DroneLink(flight_test=True)

    def ready(self):
        self.link._video_time = 10.0

    def last_packet(self):
        return self.link.rc.sock.sent[-1][0]

    def test_takeoff_requires_fresh_video(self):
        self.link.takeoff()
        self.assertIsNone(self.link._takeoff_at)
        self.ready()
        self.link.takeoff()
        self.assertEqual(self.link._takeoff_at, 10.0)

    def test_one_recent_frame_allows_takeoff_without_one_second_wait(self):
        self.link.note_frame()
        self.link.takeoff()
        self.assertEqual(self.link._takeoff_at, 10.0)

    def test_single_stale_frame_does_not_allow_takeoff(self):
        self.link.note_frame()
        self.clock.return_value = 10.4
        self.link.takeoff()
        self.assertIsNone(self.link._takeoff_at)

    def test_takeoff_requests_directly_without_calibration_and_hovers(self):
        self.ready()
        self.link.takeoff()
        self.link._tick()
        self.assertTrue(self.last_packet()['takeoff_flag'])
        self.assertFalse(self.last_packet()['calibration_flag'])
        self.assertTrue(self.link.flying)
        self.link._tick()
        self.assertFalse(self.last_packet()['takeoff_flag'])

    def test_space_cancels_pending_takeoff(self):
        self.ready()
        self.link.takeoff()
        self.link.land()
        self.assertIsNone(self.link._takeoff_at)
        self.link._tick()
        self.assertFalse(self.last_packet()['takeoff_flag'])
        self.assertFalse(self.last_packet()['land_flag'])

    def test_video_loss_cancels_takeoff_and_neutralizes(self):
        self.ready()
        self.link.takeoff()
        self.link.video_lost()
        self.link._tick()
        self.assertIsNone(self.link._takeoff_at)
        self.assertFalse(self.last_packet()['takeoff_flag'])
        self.assertEqual(self.last_packet()['pitch'], 128)

    def test_stalled_ui_cancels_takeoff(self):
        self.ready()
        self.link.takeoff()
        self.clock.return_value = 12.0
        self.link._tick()
        self.assertIsNone(self.link._takeoff_at)
        self.assertFalse(self.link.flying)

    def test_flight_mode_never_sends_tracking_motion(self):
        self.ready()
        self.link.flying = True
        self.link.send(dict(yaw=1, up=1, forward=1, right=1))
        self.link._tick()
        for axis in ('yaw', 'pitch', 'throttle', 'roll'):
            self.assertEqual(self.last_packet()[axis], 128)

    def test_manual_axes_are_capped_and_expire(self):
        self.ready()
        self.link.flying = True
        for axis, stick in (('up', 'throttle'), ('forward', 'pitch'), ('right', 'roll'), ('yaw', 'yaw')):
            for direction in (-1, 1):
                with self.subTest(axis=axis, direction=direction):
                    self.assertTrue(self.link.manual_nudge(axis, direction))
                    self.link._tick()
                    expected = direction * self.link.MANUAL_LIMITS[axis]
                    self.assertEqual(self.last_packet()[stick], drone._stick(expected))
                    self.assertEqual(self.link.manual_command()[axis], expected)
                    for other in ('throttle', 'pitch', 'roll', 'yaw'):
                        if other != stick:
                            self.assertEqual(self.last_packet()[other], 128)
        self.clock.return_value = 10.21
        self.link._tick()
        for axis in ('throttle', 'pitch', 'roll', 'yaw'):
            self.assertEqual(self.last_packet()[axis], 128)
        self.assertEqual(self.link.manual_command(), drone.HOVER)

    def test_manual_respects_lower_configured_caps(self):
        self.ready()
        self.link.flying = True
        with patch.object(config, 'MAX_FORWARD', 0.05):
            self.link.manual_nudge('right', 1)
            self.link._tick()
        self.assertEqual(self.last_packet()['roll'], drone._stick(0.05))

    def test_manual_blocked_on_ground_during_takeoff_and_without_video(self):
        self.ready()
        self.assertFalse(self.link.manual_nudge('up', 1))
        self.link.flying = True
        self.link._hold_until = 13.0
        self.assertFalse(self.link.manual_nudge('up', 1))
        self.link._hold_until = 0
        self.clock.return_value = 10.4
        self.assertFalse(self.link.manual_nudge('up', 1))

    def test_hover_video_loss_land_and_kill_cancel_manual(self):
        for action in ('hover', 'video_lost', 'land', 'kill'):
            with self.subTest(action=action):
                self.ready()
                self.link.flying = True
                self.assertTrue(self.link.manual_nudge('forward', 1))
                getattr(self.link, action)()
                self.link._tick()
                self.assertEqual(self.last_packet()['pitch'], 128)
                self.assertEqual(self.link.manual_command(), drone.HOVER)

    def test_manual_not_available_in_tracking_demo(self):
        self.ready()
        self.link.flying = True
        self.link.flight_test = False
        self.assertFalse(self.link.manual_nudge('forward', 1))

    def test_stale_commands_hover_outside_flight_mode(self):
        self.ready()
        self.link.flight_test = False
        self.link.flying = True
        self.link.send(dict(yaw=.2, up=.2, forward=.2, right=.2))
        self.clock.return_value = 10.4
        self.link._video_time = 10.4
        self.link._tick()
        self.assertEqual(self.last_packet()['pitch'], 128)

    def test_land_survives_known_local_failure_but_not_repeated_on_success(self):
        self.link.flying = True
        self.link.land()
        self.link.rc.sock.fail = True
        self.link._tick()
        self.assertTrue(self.link.sticks.land_flag)
        self.assertTrue(self.link._control_fault)
        self.link.rc.sock.fail = False
        self.link._tick()
        self.assertTrue(self.last_packet()['land_flag'])
        self.link._tick()
        self.assertFalse(self.last_packet()['land_flag'])

    def test_partial_send_does_not_repeat_toggle(self):
        self.link.flying = True
        self.link.land()
        real_send = self.link.rc.sock.sendto
        def partial(packet, address):
            if address[1] == 8801:
                raise OSError('second port failed')
            return real_send(packet, address)
        self.link.rc.sock.sendto = partial
        self.link._tick()
        self.assertFalse(self.link.sticks.land_flag)

    def test_kill_clears_land_and_blocks_new_takeoff(self):
        self.link.flying = True
        self.link.land()
        self.link.kill()
        self.link._tick()
        self.assertTrue(self.last_packet()['stop_flag'])
        self.assertFalse(self.last_packet()['land_flag'])
        self.ready()
        self.link.takeoff()
        self.assertIsNone(self.link._takeoff_at)

    def test_close_cancels_takeoff_even_if_tick_runs_before_land(self):
        self.ready()
        self.link.takeoff()
        self.clock.return_value = 11.0
        self.link._video_time = 11.0
        original_land = self.link.land
        def land_after_tick():
            self.link._tick()
            original_land()
        with patch.object(self.link, 'land', side_effect=land_after_tick), patch.object(drone.time, 'sleep'):
            self.link.close()
        self.assertIsNone(self.link._takeoff_at)
        self.assertFalse(self.last_packet()['takeoff_flag'])

    def test_lost_target_commands_hover_immediately(self):
        ctrl = Controller()
        ctrl.update(dict(found=True, x=.5, y=.5, w=.1, h=.1, conf=1))
        cmd, _ = ctrl.update(dict(found=False))
        self.assertTrue(all(value == 0 for value in cmd.values()))

    def test_plain_live_launch_blocked_before_hardware(self):
        with patch.object(main, 'VideoSource') as video:
            with self.assertRaises(RuntimeError):
                main.main()
            video.assert_not_called()

    def test_main_lands_before_video_release_on_exception(self):
        calls = []
        video, link = Mock(), Mock()
        video.kind = 'drone'
        video.read.side_effect = RuntimeError('simulated video failure')
        video.release.side_effect = lambda: calls.append('video release')
        link.close.side_effect = lambda: calls.append('land/close link')
        with patch.object(config, 'VIDEO_SOURCE', 'drone'), \
             patch.object(main, 'VideoSource', return_value=video), \
             patch.object(main, 'DroneLink', return_value=link), \
             patch.object(main.cv2, 'namedWindow'), \
             patch.object(main.cv2, 'setMouseCallback'), \
             patch.object(main.cv2, 'destroyAllWindows'):
            with self.assertRaises(RuntimeError):
                main.main(flight_test=True)
        self.assertEqual(calls, ['land/close link', 'video release'])

    def test_stale_drone_frame_displayed_but_not_fresh_for_controls(self):
        video = drone.DroneVideo.__new__(drone.DroneVideo)
        video.adapter = Mock()
        video.adapter.is_running.return_value = True
        video.adapter.get_frame.return_value = types.SimpleNamespace(timestamp=1, data=b'old')
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(drone.time, 'time', return_value=2), \
             patch.object(drone.cv2, 'imdecode', return_value=image):
            ok, result = video.read()
            self.assertTrue(ok)
            self.assertIs(result, image)
            self.assertFalse(video.fresh)

    def test_main_emergency_keys_work_without_video_and_autopilot_stays_off(self):
        video, link, tracker = Mock(), Mock(), Mock()
        link.manual_command.return_value = dict(drone.HOVER)
        video.kind = 'drone'
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        video.read.side_effect = [frame, frame, None, None]
        with patch.object(config, 'VIDEO_SOURCE', 'drone'), \
             patch.object(main, 'VideoSource', return_value=video), \
             patch.object(main, 'DroneLink', return_value=link), \
             patch.object(main, 'ColorTracker', return_value=tracker), \
             patch.object(main.voice, 'announce') as announce, \
             patch.object(main.cv2, 'namedWindow'), \
             patch.object(main.cv2, 'setMouseCallback'), \
             patch.object(main.cv2, 'imshow'), \
             patch.object(main.cv2, 'setWindowTitle'), \
             patch.object(main.cv2, 'waitKey', side_effect=[ord('a'), ord('t'), ord(' '), ord('q')]), \
             patch.object(main.cv2, 'getWindowProperty', return_value=1), \
             patch.object(main.cv2, 'destroyAllWindows'):
            main.main(flight_test=True)
        tracker.find.assert_not_called()
        announce.assert_not_called()
        link.send.assert_not_called()
        link.manual_nudge.assert_called_once_with('right', -1)
        link.takeoff.assert_called_once()
        link.land.assert_called_once()
        link.video_lost.assert_not_called()
        self.assertEqual(link.hover.call_count, 1)  # T explicitly neutralizes, missing frames do not.
        link.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
