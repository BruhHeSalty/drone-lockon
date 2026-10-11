"""Concurrent ACK regression checks, with no sockets, drone, or model."""
import threading
import types
import unittest
import json
import tempfile
from unittest.mock import Mock, patch

import drone

from drone import _LatestFrameVideo


class VideoTransportTests(unittest.TestCase):
    def test_inspection_saves_bounded_replies_without_a_flight_controller(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as directory:
                clock = [0.]
                video = Mock()
                reply = b'\x01\x65' + b'X' * 2000
                video.adapter.get_packets.return_value = [b'\x93\x01jpeg', reply, reply]

                def read():
                    clock[0] += 1.
                    if interrupted:
                        raise KeyboardInterrupt
                    return True, types.SimpleNamespace(shape=(360, 640, 3))

                video.read.side_effect = read
                with patch.object(drone, 'DroneVideo', return_value=video), \
                     patch.object(drone, 'DroneLink') as controller, \
                     patch.object(drone, '_shared_control_link', None), \
                     patch.object(drone.time, 'monotonic', side_effect=lambda: clock[0]):
                    path = drone.inspect_video_link(seconds=2, directory=directory)
                controller.assert_not_called()
                video.release.assert_called_once()
                report = json.loads(path.read_text())
                self.assertNotIn('error', report)
                self.assertEqual(report['decoded_frames'], 0 if interrupted else 2)
                self.assertEqual(report.get('interrupted', False), interrupted)
                self.assertEqual(len(report['nonvideo_samples']), 1)
                sample = report['nonvideo_samples'][0]
                self.assertTrue(sample['truncated'])
                self.assertEqual(len(bytes.fromhex(sample['packet_hex'])), 1024)
                self.assertNotIn('jpeg', path.read_text())

    def test_fragment_reception_cannot_mutate_a_set_during_ack_iteration(self):
        reading, release, received, receiving = (threading.Event() for _ in range(4))

        class Protocol:
            def __init__(self):
                self.fragments = {0, 1}

            def _build_ack_slots(self, seq):
                result = []
                for fragment in self.fragments:
                    reading.set()
                    if not release.wait(2):
                        raise TimeoutError('test did not release ACK reader')
                    result.append(fragment)
                return sorted(result)

            def handle_payload(self, payload):
                self.fragments.add(payload)
                received.set()

        class Adapter(_LatestFrameVideo, Protocol):
            pass

        adapter = Adapter()
        errors, replies = [], []

        def acknowledge():
            try:
                replies.append(adapter._build_ack_slots(1))
            except Exception as exc:
                errors.append(exc)

        def receive():
            receiving.set()
            try:
                adapter.handle_payload(2)
            except Exception as exc:
                errors.append(exc)

        ack = threading.Thread(target=acknowledge, daemon=True)
        rx = threading.Thread(target=receive, daemon=True)
        ack.start()
        try:
            self.assertTrue(reading.wait(1))
            rx.start()
            self.assertTrue(receiving.wait(1))
            self.assertFalse(received.wait(.05), 'fragment changed while ACK reader was iterating')
        finally:
            release.set()
            ack.join(2)
            if rx.ident is not None:
                rx.join(2)
        self.assertFalse(ack.is_alive())
        self.assertFalse(rx.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(replies, [[0, 1]])
        self.assertEqual(adapter._build_ack_slots(1), [0, 1, 2])

    def test_completed_frame_can_request_ack_without_deadlocking(self):
        class Protocol:
            def __init__(self):
                self.frames = []

            def _build_ack_slots(self, seq):
                return [seq]

            def handle_payload(self, frame):
                self.frames.append(self._build_ack_slots(frame.frame_id))
                return frame

        class Adapter(_LatestFrameVideo, Protocol):
            pass

        adapter = Adapter()
        first = types.SimpleNamespace(frame_id=1, timestamp=1.)
        latest = types.SimpleNamespace(frame_id=2, timestamp=1.)
        errors = []

        def receive():
            try:
                adapter.handle_payload(first)
                adapter.handle_payload(latest)
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=receive, daemon=True)
        worker.start()
        worker.join(2)
        self.assertFalse(worker.is_alive(), 'recursive ACK request deadlocked')
        self.assertEqual(errors, [])
        self.assertEqual(adapter.frames, [[1], [2]])
        self.assertIs(adapter.get_frame(timeout=0), latest)
        self.assertEqual(adapter.frames_skipped, 1)
        self.assertIsNone(adapter.get_frame(timeout=0))


if __name__ == '__main__':
    unittest.main()
