"""Offline distance-view tests. All flight links and video sources are mocked."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import config
import distance_check
import flight_distance as display
import main
from vision import AsyncPersonTracker


def target():
    return dict(found=True, x=.8, y=.5, w=.1, h=.3, conf=.95)


def observation(generation=0):
    return dict(frame=np.zeros((360,640,3),np.uint8), target=target(), tracks={1:target()},
                status='Selected person confirmed', source_frame_time=time.monotonic(), generation=generation)


class FlightDistanceTests(unittest.TestCase):
    def test_pose_runs_off_thread_and_old_selection_cannot_return(self):
        entered, release = threading.Event(), threading.Event()
        estimator = Mock(profile={})
        def estimate(frame, person, stamp):
            if not entered.is_set():
                entered.set()
                release.wait(2)
                return dict(feet=17., reason='old selection', keypoints=None)
            return dict(feet=10., reason='new selection', keypoints=None)
        estimator.estimate.side_effect = estimate
        worker = display.AsyncDistance(estimator)
        try:
            self.assertIsNone(worker.update(observation())['feet'])
            self.assertTrue(entered.wait(1))
            new = observation(1)
            # This returns while the first inference is still blocked.
            self.assertIsNone(worker.update(new)['feet'])
            release.set()
            with worker._condition:
                self.assertTrue(worker._condition.wait_for(lambda: worker._result is not None, timeout=1))
                self.assertEqual(worker._result[1]['feet'], 10.)
            self.assertEqual(worker.update(new)['feet'], 10.)
            self.assertIsNone(worker.update(None)['feet'])
        finally:
            release.set()
            worker.close()

    def test_pose_failure_and_stale_frames_return_unknown(self):
        estimator = Mock(profile={})
        estimator.estimate.side_effect = RuntimeError('pose failed')
        worker = display.AsyncDistance(estimator)
        try:
            obs = observation()
            worker.update(obs)
            with worker._condition:
                self.assertTrue(worker._condition.wait_for(lambda: worker._result is not None, timeout=1))
            self.assertIsNone(worker.update(obs)['feet'])
            obs['source_frame_time'] -= 1
            self.assertIsNone(worker.update(obs)['feet'])
        finally:
            worker.close()

    def test_snapshot_pairs_detection_with_its_original_frame_and_invalidates_clicks(self):
        with patch('vision.threading.Thread'):
            tracker = AsyncPersonTracker(Mock())
        obs = observation()
        tracker._result = (obs['source_frame_time'], obs['target'])
        tracker._snapshot = (obs['frame'], obs['tracks'], obs['status'])
        result = tracker.distance_observation()
        self.assertIs(result['frame'], obs['frame'])
        self.assertEqual(result['source_frame_time'], obs['source_frame_time'])
        self.assertIsNotNone(tracker._displayed)
        self.assertIsNone(tracker.distance_observation(video_fresh=False))
        self.assertIsNone(tracker._displayed)

    def test_click_mapping_excludes_panels(self):
        frame = observation()['frame']
        self.assertEqual(display.source_click(frame,480,420), (320,180))
        self.assertIsNone(display.source_click(frame,480,149))
        self.assertIsNone(display.source_click(frame,480,690))

    def run_view(self, recording_factory, keys):
        obs = observation()
        video, link, tracker, estimator = Mock(), Mock(), Mock(), Mock()
        video.kind, video.fresh, video.frame_time = 'drone', True, obs['source_frame_time']
        video.read.return_value = obs['frame']
        tracker.find.return_value = obs['target']
        tracker.distance_observation.return_value = obs
        estimator.profile = {'reference_feet':19}
        estimator.update.return_value = dict(feet=10.2, reason='Experimental torso estimate', keypoints=None)
        link.tracking_block_reason.return_value = None  # Simulated settled takeoff.
        with patch.object(config,'CONTROL_PORT',8800), \
             patch.object(config,'VIDEO_SOURCE','drone'), \
             patch.object(main,'VideoSource',return_value=video), \
             patch.object(main,'DroneLink',return_value=link), \
             patch.object(main,'PersonTracker'), \
             patch.object(main,'AsyncPersonTracker',return_value=tracker), \
             patch.object(display,'AsyncDistance',return_value=estimator), \
             patch.object(distance_check,'Recording',side_effect=recording_factory), \
             patch.object(main.voice,'announce'), \
             patch.object(main.hud,'draw') as old_hud, \
             patch.object(main.cv2,'namedWindow'), \
             patch.object(main.cv2,'setMouseCallback'), \
             patch.object(main.cv2,'imshow'), \
             patch.object(main.cv2,'waitKey',side_effect=list(map(ord,keys))), \
             patch.object(main.cv2,'getWindowProperty',return_value=1), \
             patch.object(main.cv2,'destroyAllWindows'):
            main.main(person=True,distance_view=True)
        old_hud.assert_not_called()
        estimator.close.assert_called_once()
        link.close.assert_called_once()
        return link

    def test_t_records_raw_distance_view_timing_and_controls_without_lockon_hud(self):
        real_recording = distance_check.Recording
        with tempfile.TemporaryDirectory() as directory:
            def factory(_, *args):
                return real_recording(directory, *args)
            link = self.run_view(factory, ['t','a',' ','m','r','q'])
            link.takeoff.assert_called_once()
            link.land.assert_called_once()
            rows = [json.loads(l) for l in next(Path(directory).glob('*.jsonl')).read_text().splitlines()]
            self.assertEqual(rows[0]['event'],'recording_started')
            self.assertEqual(rows[1]['event'],'flight_test_metadata')
            self.assertEqual(rows[2]['event'],'takeoff_key')
            frames = [r for r in rows if r['event']=='frame']
            self.assertEqual(len(frames),4)
            self.assertEqual(frames[0]['result']['feet'],10.2)
            for row in frames:
                self.assertIn('source_frame_time',row)
                self.assertIn('session_seconds',row)
                self.assertEqual([row['command'][k] for k in ('up','forward','right')],[0,0,0])
            self.assertEqual(sum(r['event']=='measurement_marker' for r in rows),1)
            for suffix, shape in (('_raw.mp4',(360,640)),('_hud.mp4',(790,960))):
                cap = cv2.VideoCapture(str(next(Path(directory).glob('*'+suffix))))
                ok, frame = cap.read(); cap.release()
                self.assertTrue(ok)
                self.assertEqual(frame.shape[:2],shape)

    def test_disk_error_stops_recording_but_does_not_disable_landing_key(self):
        recorder = Mock()
        recorder.write.side_effect = OSError('disk full')
        link = self.run_view(lambda *args: recorder, ['t',' ','q'])
        link.takeoff.assert_called_once()
        link.land.assert_called_once()
        recorder.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
