"""Offline flight-safety regression tests. No camera, socket, or drone is opened."""
import types
import os
import json
import tempfile
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
        self.capabilities = types.SimpleNamespace(transport='uav_dual_port')

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
            # Exercise legacy control logic only with the fake adapter/socket/thread below.
            patch.object(drone, 'LIVE_FLIGHT_BLOCK_REASON', None),
            patch.dict('sys.modules', {'protocols.wifi_uav_rc_protocol_adapter': module}),
            patch.object(drone, '_load_turbodrone'),
            patch.object(drone, '_shared_sock', None),
            patch.object(drone, '_shared_control_link', None),
            patch.object(config, 'CONTROL_PORT', 8800),
            patch.object(drone.threading, 'Thread'),
            patch.object(drone, 'FlightRecorder'),
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

    def test_yaw_only_transport_rejects_translation_caps_turning_and_holds_on_loss(self):
        # Restore an active restriction to exercise the explicit yaw-only path.
        # setUp has already replaced every socket, adapter and sender thread.
        with patch.object(drone, 'LIVE_FLIGHT_BLOCK_REASON', 'unrestricted blocked'):
            link = drone.DroneLink(yaw_only=True)
        desired = dict(yaw=1., forward=1., right=-1., up=1.)
        link.note_frame()
        link.send(desired)
        link._tick()
        self.assertEqual(link.rc.sock.sent[-1][0]['yaw'], 128)  # T is required.
        link.takeoff()
        link._tick()
        self.assertEqual(link.rc.sock.sent[-1][0]['yaw'], 128)  # Settling hold.
        self.clock.return_value = 13.1
        link.note_frame()
        for yaw, expected in ((1., .15), (-1., -.15), (0., 0.), (float('nan'), 0.), (float('inf'), 0.)):
            link.send(dict(desired, yaw=yaw))
            link._tick()
            packet = link.rc.sock.sent[-1][0]
            self.assertEqual(packet['yaw'], drone._stick(expected))
            self.assertEqual([packet[a] for a in ('pitch', 'throttle', 'roll')], [128] * 3)
            ack = link._video_ack_payload(bytes(40))
            self.assertEqual(ack[20:24], bytes([drone._stick(expected), 128, 128, 128]))
        with patch.object(config, 'MAX_YAW', .05):
            link.send(desired)
            link._tick()
            self.assertEqual(link.rc.sock.sent[-1][0]['yaw'], drone._stick(.05))
        link.video_lost()
        link._tick()
        self.assertEqual(link.rc.sock.sent[-1][0]['yaw'], 128)
        self.assertFalse(link.manual_nudge('forward', 1))

    def test_short_attempt_is_suspected_failure_not_claimed_airtime(self):
        self.ready()
        self.link.takeoff()
        self.link._tick()
        self.clock.return_value = 17.
        self.link.land()
        summaries = [c for c in self.link._recorder.record.call_args_list if c.args[0] == 'attempt_end']
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].kwargs['duration_s'], 7.)
        self.assertEqual(summaries[0].kwargs['outcome'], 'suspected_failed')
        self.assertFalse(summaries[0].kwargs['airborne_verified'])
        self.link.kill()
        self.assertEqual(len([c for c in self.link._recorder.record.call_args_list if c.args[0] == 'attempt_end']), 1)

    def test_long_attempt_is_not_assumed_successful(self):
        self.ready()
        self.link.takeoff()
        self.link._tick()
        self.clock.return_value = 45.
        self.link.land()
        summary = [c for c in self.link._recorder.record.call_args_list if c.args[0] == 'attempt_end'][0]
        self.assertEqual(summary.kwargs['outcome'], 'unverified')
        self.assertFalse(summary.kwargs['airborne_verified'])

    def test_control_journal_records_neutral_on_loss_and_socket_failure(self):
        self.link.flight_test = False
        self.link.flying = True
        self.ready()
        self.link.send(dict(yaw=0., up=0., forward=.08, right=0.))
        self.link._tick()
        self.link.video_lost()
        self.link._tick()
        packets = [c.kwargs for c in self.link._recorder.record.call_args_list if c.args[0] == 'control_packet']
        self.assertEqual(packets[0]['sticks']['pitch'], 138)
        self.assertEqual(packets[-1]['command'], drone.HOVER)
        self.link.rc.sock.fail = True
        self.link._tick()
        packet = [c.kwargs for c in self.link._recorder.record.call_args_list if c.args[0] == 'control_packet'][-1]
        self.assertFalse(packet['locally_accepted'])
        self.assertIn('simulated', packet['error'])

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

    def test_takeoff_discards_old_motion_and_readiness_waits_for_hold(self):
        self.link.flight_test = False
        self.ready()
        self.link.send(dict(yaw=-.15, up=0., forward=.08, right=-.1))
        self.link.takeoff()
        self.assertEqual(self.link._cmd, drone.HOVER)
        self.link._tick()
        self.assertIn('takeoff hold', self.link.tracking_block_reason())
        for now in (10., 11., 12.99):
            self.clock.return_value = now
            self.link.note_frame()
            self.link.send(dict(yaw=-.15, up=0., forward=.08, right=-.1))
            self.link._tick()
            for axis in ('yaw', 'pitch', 'throttle', 'roll'):
                self.assertEqual(self.last_packet()[axis], 128)
        self.clock.return_value = 13.01
        self.link.note_frame()
        self.link.hover()
        self.assertIsNone(self.link.tracking_block_reason())
        self.link._tick()
        self.assertEqual(self.last_packet()['roll'], 128)
        self.link.video_lost()
        self.assertIn('video', self.link.tracking_block_reason())
        self.link._control_fault = True
        self.assertIn('unavailable', self.link.tracking_block_reason())

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

    def test_follow_forward_reaches_stick_packet_and_video_loss_neutralizes(self):
        # Fake adapter/socket only: verify the new controller's forward output
        # has an actual control path, and retains the video-loss stop.
        self.link = drone.DroneLink(flight_test=False)
        self.ready()
        self.link.flying = True
        self.link.send(dict(yaw=0., up=0., forward=.08, right=0.))
        self.link._tick()
        self.assertEqual(self.last_packet()['pitch'], drone._stick(.08))
        for axis in ('yaw', 'throttle', 'roll'):
            self.assertEqual(self.last_packet()[axis], 128)
        self.link.video_lost()
        self.link._tick()
        self.assertEqual(self.last_packet()['pitch'], 128)

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

    def test_held_keys_never_send_full_sticks_and_expire_on_ui_stall(self):
        self.ready()
        self.link.flying = True
        self.assertTrue(self.link.manual_input(dict(forward=1, right=-1, up=1, yaw=-1)))
        self.link._tick()
        self.assertEqual(self.last_packet()['pitch'], 153)  # 20% stick input
        self.assertEqual(self.last_packet()['roll'], 103)
        self.assertEqual(self.last_packet()['throttle'], 153)
        self.assertEqual(self.last_packet()['yaw'], 103)
        self.assertEqual(self.link.manual_command()['forward'], .20)
        self.assertEqual(self.link.manual_command()['up'], .20)
        self.clock.return_value = 10.21
        self.link._tick()
        self.assertEqual(self.last_packet()['pitch'], 128)
        self.assertEqual(self.last_packet()['throttle'], 128)

    def test_manual_release_fault_loss_and_lower_caps(self):
        self.ready()
        self.link.flying = True
        with patch.object(config, 'MAX_FORWARD', .01):
            self.link.manual_input(dict(forward=1, right=1, up=0, yaw=0))
            self.assertEqual(self.link.manual_command()['forward'], .01)
            self.assertEqual(self.link.manual_command()['right'], .01)
        self.link.manual_input(drone.HOVER)
        self.assertEqual(self.link.manual_command(), drone.HOVER)
        self.link.video_lost()
        self.assertFalse(self.link.manual_input(dict(forward=1, right=0, up=0, yaw=0)))
        self.assertEqual(self.link._manual_cmd, drone.HOVER)
        self.ready()
        self.link._control_fault = True
        self.assertFalse(self.link.manual_input(dict(forward=1, right=0, up=0, yaw=0)))

    def test_manual_invalid_input_neutralizes_and_preview_needs_no_socket(self):
        self.ready()
        self.link.flying = True
        self.link.manual_input(dict(forward=1, right=0, up=0, yaw=0))
        with self.assertRaises(ValueError):
            self.link.manual_input(dict(forward=float('nan'), right=0, up=0, yaw=0))
        self.assertEqual(self.link.manual_command(), drone.HOVER)
        with patch.object(config, 'CONTROL_PORT', None):
            preview = drone.DroneLink(flight_test=True)
        preview.manual_input(dict(forward=1, right=0, up=0, yaw=0))
        self.assertEqual(preview.manual_command()['forward'], .20)
        self.assertFalse(hasattr(preview, 'rc'))

    def test_manual_tracking_display_never_enables_autopilot(self):
        video, link, tracker, window = Mock(), Mock(), Mock(), Mock()
        video.kind, video.fresh = 'drone', True
        video.read.return_value = np.zeros((360, 640, 3), dtype=np.uint8)
        tracker.find.return_value = dict(found=True, x=.8, y=.5, w=.1, h=.2, conf=.95)
        link.manual_command.return_value = dict(drone.HOVER)
        window.key.side_effect = [-1, ord('1'), ord(' '), 27]
        window.axes.return_value = dict(forward=1, right=0, yaw=-1, up=0)
        window.input_status.return_value = {'keys': ['w', 'q'], 'blocked_reason': None}
        link.manual_status.return_value = 'READY: manual input enabled (hover unverified)'
        from contextlib import ExitStack
        with ExitStack() as stack:
            for context in (
                patch.object(config, 'VIDEO_SOURCE', 'drone'),
                patch.object(main, 'VideoSource', return_value=video),
                patch.object(main, 'DroneLink', return_value=link),
                patch.object(main, 'PersonTracker'),
                patch.object(main, 'AsyncPersonTracker', return_value=tracker),
                patch.object(main, 'ManualWindow', return_value=window),
                patch.object(main.voice, 'announce', side_effect=lambda *args, **kw: self.assertTrue(kw.get('visual_only'))),
            ):
                stack.enter_context(context)
            main.main(manual=True)
        self.assertEqual(tracker.find.call_count, 4)
        link.send.assert_not_called()
        self.assertEqual(link.manual_input.call_args_list[0].args[0]['yaw'], -1)
        self.assertEqual(link.manual_input.call_args_list[-1].args[0], drone.HOVER)
        link.record_event.assert_any_call('operator_flight_label', label='steady', source='operator', airborne_verified=False)
        link.land.assert_called_once()
        link.close.assert_called_once()
        window.close.assert_called_once()

    def test_held_manual_keys_survive_frame_gap_but_release_and_stale_video_stop(self):
        video, tracker, window = Mock(), Mock(), Mock()
        video.kind, video.fresh = 'drone', True
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        tracker.find.return_value = dict(found=False, x=0, y=0, w=0, h=0, conf=0)
        ticks = iter([10., 10.1, 10.15, 10.2, 10.31, 10.32])

        def read():
            self.clock.return_value = next(ticks)
            return frame if self.clock.return_value == 10. else None

        video.read.side_effect = read
        window.key.side_effect = [-1, -1, -1, -1, -1, 27]
        forward = dict(forward=1, right=0, yaw=0, up=0)
        window.axes.side_effect = [forward, forward, dict(drone.HOVER), forward]
        window.input_status.return_value = {'keys': ['w'], 'blocked_reason': None}
        self.link.flying = True
        packets = []
        manual_input = self.link.manual_input

        def send_input(axes):
            result = manual_input(axes)
            self.link._tick()
            packets.append(self.last_packet()['pitch'])
            return result

        from contextlib import ExitStack
        with ExitStack() as stack:
            for context in (
                patch.object(config, 'VIDEO_SOURCE', 'drone'),
                patch.object(main, 'VideoSource', return_value=video),
                patch.object(main, 'DroneLink', return_value=self.link),
                patch.object(main, 'PersonTracker'),
                patch.object(main, 'AsyncPersonTracker', return_value=tracker),
                patch.object(main, 'ManualWindow', return_value=window),
                patch.object(main.voice, 'announce'),
                patch.object(self.link, 'manual_input', side_effect=send_input),
                patch.object(self.link, 'close'),
            ):
                stack.enter_context(context)
            main.main(manual=True)
        self.assertEqual(packets, [153, 153, 128, 153, 128, 128])
        self.assertEqual(tracker.find.call_count, 1)
        self.assertEqual(self.link._video_time, 10.)  # Gaps never refresh freshness.
        self.assertFalse(self.link.video_ready())

    def test_manual_respects_lower_configured_caps(self):
        self.ready()
        self.link.flying = True
        with patch.object(config, 'MAX_FORWARD', 0.05):
            self.link.manual_nudge('right', 1)
            self.link._tick()
        self.assertEqual(self.last_packet()['roll'], drone._stick(0.05))

    def test_blocked_key_requests_are_logged_during_takeoff_hold(self):
        self.ready()
        self.link.takeoff()
        self.link._tick()
        axes = dict(forward=1, right=0, yaw=0, up=0)
        self.assertFalse(self.link.manual_input(axes))
        self.assertIn('3.0s remaining', self.link.manual_status())
        calls = [c.kwargs for c in self.link._recorder.record.call_args_list if c.args[0] == 'manual_input']
        self.assertEqual(calls[-1]['requested_axes'], axes)
        self.assertEqual(calls[-1]['command'], drone.HOVER)
        self.assertIn('takeoff hold', calls[-1]['blocked_reason'])
        self.link.manual_input(axes)
        self.assertEqual(len([c for c in self.link._recorder.record.call_args_list if c.args[0] == 'manual_input']), len(calls))
        self.clock.return_value = 13.1
        self.link.note_frame()
        self.assertTrue(self.link.manual_input(axes))
        self.assertIn('READY', self.link.manual_status())
        self.link.land()
        self.assertIn('session ended', self.link.manual_status())

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

    def test_close_cancels_queued_takeoff_and_submits_stop_immediately(self):
        self.ready()
        self.link.takeoff()
        self.clock.return_value = 11.0
        self.link._video_time = 11.0
        with patch.object(self.link, 'land') as land, patch.object(drone.time, 'sleep'):
            self.link.close()
        land.assert_not_called()
        self.assertIsNone(self.link._takeoff_at)
        self.assertFalse(self.last_packet()['takeoff_flag'])
        self.assertFalse(self.last_packet()['land_flag'])
        self.assertTrue(self.last_packet()['stop_flag'])

    def test_failed_takeoff_then_close_never_submits_land_toggle(self):
        # Reproduce the incident: local send succeeds but aircraft ignores T.
        # The old implementation assumed airborne and sent LAND on close.
        self.ready()
        self.link.takeoff()
        self.link._tick()
        self.assertTrue(self.link.flying)  # Still unverified request bookkeeping.
        sent_before_close = len(self.link.rc.sock.sent)
        with patch.object(drone.time, 'sleep'):
            self.link.close()
        for packet, _ in self.link.rc.sock.sent[sent_before_close:]:
            self.assertTrue(packet['stop_flag'])
            self.assertFalse(packet['takeoff_flag'])
            self.assertFalse(packet['land_flag'])
            self.assertEqual([packet[a] for a in ('yaw', 'pitch', 'throttle', 'roll')], [128] * 4)

    def test_close_overrides_pending_land_calibration_and_motion(self):
        self.link.flying = True
        self.link.land()
        self.link.sticks.calibration_flag = True
        self.link.sticks.flip_flag = True
        self.link._calib_left = 20
        self.link._cmd = dict(yaw=.15, up=.15, forward=.1, right=.1)
        with patch.object(drone.time, 'sleep'):
            self.link.close()
        packet = self.last_packet()
        self.assertTrue(packet['stop_flag'])
        for flag in ('takeoff_flag', 'land_flag', 'calibration_flag', 'flip_flag'):
            self.assertFalse(packet[flag])
        self.assertEqual(self.link._cmd, drone.HOVER)

    def test_close_without_takeoff_sends_stop_once_and_is_idempotent(self):
        with patch.object(drone.time, 'sleep'):
            self.link.close()
            first_count = len(self.link.rc.sock.sent)
            self.link.close()
        self.assertEqual(len(self.link.rc.sock.sent), first_count)
        self.assertTrue(self.last_packet()['stop_flag'])
        self.assertFalse(self.last_packet()['land_flag'])

    def test_close_reports_no_submission_when_socket_failed(self):
        self.link.rc.sock.fail = True
        with patch.object(drone.time, 'sleep'):
            self.link.close()
        reports = [c.kwargs for c in self.link._recorder.record.call_args_list
                   if c.args[0] == 'shutdown_motor_stop']
        self.assertEqual(reports[0]['locally_submitted_packets'], 0)
        self.assertFalse(reports[0]['physical_stop_verified'])

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

    def test_main_closes_control_before_video_release_on_exception(self):
        calls = []
        video, link = Mock(), Mock()
        video.kind = 'drone'
        video.read.side_effect = RuntimeError('simulated video failure')
        video.release.side_effect = lambda: calls.append('video release')
        link.close.side_effect = lambda: calls.append('stop/close link')
        with patch.object(config, 'VIDEO_SOURCE', 'drone'), \
             patch.object(main, 'VideoSource', return_value=video), \
             patch.object(main, 'DroneLink', return_value=link), \
             patch.object(main.cv2, 'namedWindow'), \
             patch.object(main.cv2, 'setMouseCallback'), \
             patch.object(main.cv2, 'destroyAllWindows'):
            with self.assertRaises(RuntimeError):
                main.main(flight_test=True)
        self.assertEqual(calls, ['stop/close link', 'video release'])

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


@unittest.skipUnless(os.path.isdir(os.path.expanduser(config.TURBODRONE_PATH)),
                     'optional Turbodrone checkout is not installed')
class WirePacketTests(unittest.TestCase):
    def setUp(self):
        # Every encoder test below supplies FakeSocket and suppresses sender threads.
        guard = patch.object(drone, 'LIVE_FLIGHT_BLOCK_REASON', None)
        guard.start()
        self.addCleanup(guard.stop)

    def test_shutdown_wire_packets_are_motor_stop_never_takeoff_land(self):
        drone._load_turbodrone()
        from protocols.wifi_uav_rc_protocol_adapter import WifiUavRcProtocolAdapter
        sock = FakeSocket()
        adapter = WifiUavRcProtocolAdapter(shared_sock=sock, variant='uav')
        with patch('protocols.wifi_uav_rc_protocol_adapter.WifiUavRcProtocolAdapter', return_value=adapter), \
             patch.object(config, 'CONTROL_PORT', 8800), \
             patch.object(drone, '_shared_sock', None), \
             patch.object(drone, '_shared_control_link', None), \
             patch.object(drone.threading, 'Thread'), \
             patch.object(drone, 'FlightRecorder'), \
             patch.object(drone.time, 'monotonic', return_value=10.):
            link = drone.DroneLink()
            link.note_frame()
            link.takeoff()
            link._tick()  # Locally accepted takeoff with no aircraft acknowledgement.
            sock.sent.clear()

            def flush(_):
                # First stop was sent synchronously before the flush wait.
                self.assertEqual(sock.sent[0][0][24], 0x02)
                for _ in range(link.STOP_PACKETS):
                    link._tick()

            with patch.object(drone.time, 'sleep', side_effect=flush):
                link.close()
            packets = [p for p, address in sock.sent if address[1] == 8800]
            self.assertEqual(sum(p[24] == 0x02 for p in packets), link.STOP_PACKETS)
            self.assertEqual(link._stop_packets_submitted, link.STOP_PACKETS)
            for p in packets:
                self.assertIn(p[24], (0x00, 0x02))
                self.assertEqual(p[20:24], bytes([128] * 4))
                checksum = 0
                for b in p[20:36]:
                    checksum ^= b
                self.assertEqual(p[36], checksum)

    def test_native_video_requests_match_rc_and_clear_on_release_loss_and_land(self):
        drone._load_turbodrone()
        from protocols.wifi_uav_rc_protocol_adapter import WifiUavRcProtocolAdapter
        from utils.wifi_uav_packets import build_native_ack_packet, build_ack_slot
        from contextlib import ExitStack
        for variant in ('uav', 'fld'):
            with self.subTest(variant=variant), ExitStack() as stack:
                sock = FakeSocket()
                adapter = WifiUavRcProtocolAdapter(shared_sock=sock, variant=variant)
                for context in (
                    patch('protocols.wifi_uav_rc_protocol_adapter.WifiUavRcProtocolAdapter', return_value=adapter),
                    patch.object(config, 'CONTROL_PORT', 8800),
                    patch.object(drone, '_shared_sock', sock),
                    patch.object(drone, '_shared_control_link', None),
                    patch.object(drone.threading, 'Thread'),
                    patch.object(drone, 'FlightRecorder'),
                ):
                    stack.enter_context(context)
                clock = stack.enter_context(patch.object(drone.time, 'monotonic', return_value=10.))
                link = drone.DroneLink(flight_test=True)
                link.note_frame()
                link.takeoff()
                link._tick()
                clock.return_value = 13.1
                link.note_frame()
                link.manual_input(dict(yaw=-1, forward=1, up=1, right=1))
                link._tick()
                rc_axes = sock.sent[-1][0][20:24]
                request = build_native_ack_packet(123, [build_ack_slot(123, 3)])

                class Protocol:
                    def _safe_sendto(self, payload, port, label):
                        sock.sendto(payload, ('test-drone', port))
                        return True

                class Video(drone._LatestFrameVideo, Protocol):
                    pass

                video = Video()
                video._sock = sock

                def ack_axes():
                    video._safe_sendto(request, 8800, 'test ACK')
                    packet = sock.sent[-1][0]
                    self.assertEqual(packet[:20], request[:20])
                    self.assertEqual(packet[37:], request[37:])
                    self.assertEqual(packet[24], 0)  # Never replay a one-key action.
                    checksum = 0
                    for value in packet[20:36]:
                        checksum ^= value
                    self.assertEqual(packet[36], checksum)
                    return packet[20:24]

                self.assertEqual(ack_axes(), rc_axes)
                link._tick()  # Interleave an RC send with the camera ACKs.
                self.assertEqual(ack_axes(), rc_axes)
                self.assertNotEqual(rc_axes, bytes([128] * 4))
                link.manual_input(drone.HOVER)
                self.assertEqual(ack_axes(), bytes([128] * 4))  # No sender tick needed after release.
                link.manual_input(dict(yaw=0, forward=1, up=0, right=0))
                clock.return_value += .21
                self.assertEqual(ack_axes(), bytes([128] * 4))
                link.note_frame()
                link.manual_input(dict(yaw=0, forward=1, up=0, right=0))
                link.video_lost()
                self.assertEqual(ack_axes(), bytes([128] * 4))
                link.note_frame()
                link.manual_input(dict(yaw=0, forward=1, up=0, right=0))
                link._control_fault = True
                self.assertEqual(ack_axes(), bytes([128] * 4))
                link._control_fault = False
                link.land()
                self.assertEqual(ack_axes(), bytes([128] * 4))
                self.assertTrue(link.sticks.land_flag)  # ACK does not consume the pending landing action.
                link._tick()
                self.assertEqual(sock.sent[-1][0][24], 1)  # Landing action survives sequencing.

    def test_real_encoder_takeoff_hover_forward_and_loss_with_fake_socket(self):
        # Exercise the actual installed packet encoder, with every socket and
        # sender thread replaced. This does not communicate with a drone.
        drone._load_turbodrone()
        from protocols.wifi_uav_rc_protocol_adapter import WifiUavRcProtocolAdapter
        sock = FakeSocket()
        adapter = WifiUavRcProtocolAdapter(shared_sock=sock, variant='uav')
        with patch('protocols.wifi_uav_rc_protocol_adapter.WifiUavRcProtocolAdapter', return_value=adapter), \
             patch.object(config, 'CONTROL_PORT', 8800), \
             patch.object(drone, '_shared_sock', None), \
             patch.object(drone, '_shared_control_link', None), \
             patch.object(drone.threading, 'Thread'), \
             patch.object(drone, 'FlightRecorder'), \
             patch.object(drone.time, 'monotonic', return_value=10.) as clock:
            link = drone.DroneLink()
            link.note_frame()
            link.takeoff()
            for now in (10., 11., 12.99, 13.01, 20.):
                clock.return_value = now
                link.note_frame()
                link.hover()
                link._tick()
                packet = sock.sent[-1][0]
                self.assertEqual(packet[20:24], bytes([128] * 4))
                self.assertEqual(packet[24], 1 if now == 10. else 0)
                self.assertEqual(packet[25], 2)  # headless remains off
            link.send(dict(yaw=0., up=0., forward=.08, right=0.))
            link._tick()
            self.assertEqual(sock.sent[-1][0][20:24], bytes([128, 138, 128, 128]))
            link.video_lost()
            link._tick()
            self.assertEqual(sock.sent[-1][0][20:24], bytes([128] * 4))
            link.land()
            link._tick()
            self.assertEqual(sock.sent[-1][0][24], 1)


class JournalTests(unittest.TestCase):
    def test_json_lines_survive_close_with_timestamps_and_diagnostic_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = drone.FlightRecorder(directory)
            recorder.record('takeoff_requested', airborne_verified=False)
            recorder.record('hover_key', position_hold_verified=False)
            recorder.close()
            recorder.close()
            rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows], ['takeoff_requested', 'hover_key', 'journal_end'])
            self.assertTrue(all('utc' in r and 'elapsed_s' in r for r in rows))
            self.assertFalse(rows[0]['airborne_verified'])
            self.assertEqual(rows[-1]['dropped_events'], 0)
            self.assertFalse(recorder._thread.is_alive())


class ManualKeyboardTests(unittest.TestCase):
    def setUp(self):
        self.window = main.ManualWindow.__new__(main.ManualWindow)
        self.window.root = Mock()
        self.window.held = set()
        self.window.actions = []
        self.window.inhibited = self.window.closed = False

    def press(self, *keys):
        for key in keys:
            self.window._press(types.SimpleNamespace(keysym=key))

    def release(self, *keys):
        for key in keys:
            self.window._release(types.SimpleNamespace(keysym=key))

    def test_axes_modifiers_releases_and_opposite_keys(self):
        self.press('w', 'a', 'q', 'Shift_L')
        self.assertEqual(self.window.axes(), dict(forward=1, right=-1, yaw=-1, up=1))
        self.assertEqual(self.window.key(), -1)  # Q does not quit; A does not arm.
        self.press('s', 'd', 'e', 'Control_L')
        self.assertEqual(self.window.axes(), drone.HOVER)
        self.release('s', 'd', 'e', 'Control_L', 'w', 'a', 'q', 'Shift_L')
        self.assertEqual(self.window.axes(), drone.HOVER)

    def test_neutral_and_focus_loss_cancel_held_inputs(self):
        self.press('w', 'h')
        self.assertEqual(self.window.axes(), drone.HOVER)
        self.release('h')
        self.press('w')  # OS repeat cannot restart a held input after H.
        self.assertEqual(self.window.axes(), drone.HOVER)
        self.release('w')
        self.press('w')
        self.assertEqual(self.window.axes()['forward'], 1)
        self.window._focus_out()
        self.assertEqual(self.window.axes(), drone.HOVER)
        self.assertEqual(self.window.key(), ord('h'))
        self.press('w')
        self.window.root.focus_displayof.return_value = None
        self.assertEqual(self.window.axes(), drone.HOVER)

    def test_emergency_actions_override_queued_takeoff(self):
        self.press('t', 'l')
        self.assertEqual(self.window.key(), ord(' '))
        self.assertEqual(self.window.key(), -1)
        self.assertEqual(self.window.axes(), drone.HOVER)
        self.release('t', 'l')
        self.press('t', 'space', 'k')
        self.assertEqual(self.window.key(), ord('k'))
        self.assertEqual(self.window.key(), -1)
        self.release('t', 'space', 'k')
        self.press('k')
        self.window._focus_out()
        self.assertEqual(self.window.key(), ord('k'))


if __name__ == '__main__':
    unittest.main()
