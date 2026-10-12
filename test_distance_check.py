import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np

import config
import distance_check as check


def target():
    return dict(found=True,x=.5,y=.5,w=.4,h=.8,conf=.95)


def pose_result(duplicate=False,cropped=False):
    points=np.zeros((17,3),np.float32)
    for i,x,y in ((5,.4,.2),(6,.6,.2),(11,.42,.6),(12,.58,.6)):
        points[i]=[x*640,y*360,.95]
    if cropped:
        points[11,1]=359
    tensor=Mock()
    tensor.cpu.return_value.numpy.return_value=points
    boxes=[[.3,.1,.7,.9]]*(2 if duplicate else 1)
    return SimpleNamespace(boxes=SimpleNamespace(xyxyn=np.array(boxes),conf=[.95]*len(boxes)),
                           keypoints=SimpleNamespace(data=[tensor]*len(boxes)))


class DistanceCheckTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.zeros((360,640,3),np.uint8)
        self.model=Mock()
        self.model.predict.return_value=[pose_result()]
        self.estimator=check.DistanceEstimator(model=self.model)

    def test_uses_saved_measured_reference(self):
        with patch.object(check.time,'monotonic',return_value=10.1):
            r=self.estimator.estimate(self.frame,target(),10.)
        self.assertAlmostEqual(r['feet'],19*.22946203500032425/.4,places=4)

    def test_lost_or_stale_never_reuses_previous_distance(self):
        with patch.object(check.time,'monotonic',return_value=10.1):
            self.assertIsNotNone(self.estimator.estimate(self.frame,target(),10.)['feet'])
            self.assertIsNone(self.estimator.estimate(self.frame,dict(found=False),10.)['feet'])
            self.assertIsNone(self.estimator.estimate(self.frame,target(),9.)['feet'])
            self.assertIsNone(self.estimator.estimate(self.frame,target(),11.)['feet'])

    def test_inference_that_finishes_late_returns_unknown(self):
        with patch.object(check.time,'monotonic',side_effect=[10.1,10.4]):
            self.assertIsNone(self.estimator.estimate(self.frame,target(),10.)['feet'])

    def test_ambiguous_pose_or_cropped_joint_returns_unknown(self):
        for result in (pose_result(duplicate=True),pose_result(cropped=True)):
            self.model.predict.return_value=[result]
            with patch.object(check.time,'monotonic',return_value=10.1):
                self.assertIsNone(self.estimator.estimate(self.frame,target(),10.)['feet'])

    def test_different_aspect_ratio_is_not_assumed_calibrated(self):
        with patch.object(check.time,'monotonic',return_value=10.1):
            r=self.estimator.estimate(np.zeros((480,640,3),np.uint8),target(),10.)
        self.assertIsNone(r['feet'])
        self.model.predict.assert_not_called()

    def test_recording_preserves_frame_measurement_timing_and_marker(self):
        result=dict(feet=14.1,reason='experimental',keypoints=None)
        view=check.make_view(self.frame,target(),{},result,4.2,True)
        with tempfile.TemporaryDirectory() as directory:
            recorder=check.Recording(directory,self.frame,view,self.estimator.profile,4.)
            base=recorder.base
            recorder.write(self.frame,view,session_seconds=4.2,result=result)
            self.assertEqual(recorder.mark(4.2,result),1)
            recorder.close();recorder.close()
            rows=[json.loads(line) for line in Path(str(base)+'.jsonl').read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows],['recording_started','frame','measurement_marker','recording_stopped'])
            self.assertEqual(rows[1]['result']['feet'],14.1)
            self.assertEqual(rows[2]['session_seconds'],4.2)
            self.assertIsNone(rows[2]['measured_feet'])
            for suffix in ('_raw.mp4','_hud.mp4'):
                cap=cv2.VideoCapture(str(base)+suffix)
                self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT),1)
                ok,_=cap.read();self.assertTrue(ok);cap.release()

    def test_recording_failure_releases_both_writers(self):
        first,second=Mock(),Mock()
        first.isOpened.return_value=False
        with tempfile.TemporaryDirectory() as directory,patch.object(check.cv2,'VideoWriter',side_effect=[first,second]):
            with self.assertRaisesRegex(RuntimeError,'Cannot open'):
                check.Recording(directory,self.frame,self.frame,{},0)
        first.release.assert_called_once();second.release.assert_called_once()
        recorder=check.Recording.__new__(check.Recording)
        recorder.raw,recorder.hud=Mock(),Mock()
        recorder.frames=recorder.markers=0
        recorder.journal=Mock(closed=False)
        recorder.journal.write.side_effect=OSError('simulated disk full')
        with self.assertRaises(OSError):
            recorder.close()
        recorder.raw.release.assert_called_once()
        recorder.hud.release.assert_called_once()
        recorder.journal.close.assert_called_once()

    def test_grounded_mode_records_marks_and_never_creates_flight_link(self):
        video,tracker,estimator=Mock(),Mock(),Mock()
        video.read.return_value=self.frame
        video.fresh=True;video.frame_time=10.
        tracker.find.return_value=target();tracker._tracks={1:target()}
        estimator.profile={}
        estimator.estimate.return_value=dict(feet=14.1,reason='experimental',keypoints=None)
        callbacks=[]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config,'CONTROL_PORT',8800), \
             patch.object(check,'DistanceEstimator',return_value=estimator), \
             patch.object(check,'PersonTracker',return_value=tracker), \
             patch.object(check,'VideoSource',return_value=video), \
             patch('drone.DroneLink') as link, \
             patch.object(check.cv2,'namedWindow'), \
             patch.object(check.cv2,'setMouseCallback',side_effect=lambda _,cb:callbacks.append(cb)), \
             patch.object(check.cv2,'imshow'), \
             patch.object(check.cv2,'waitKey',side_effect=[ord('r'),ord('m'),ord('t'),ord('a'),ord(' '),ord('r'),ord('q')]), \
             patch.object(check.cv2,'getWindowProperty',return_value=1), \
             patch.object(check.cv2,'destroyAllWindows'):
            check.run(output=directory)
            self.assertIsNone(config.CONTROL_PORT)
            link.assert_not_called()
            video.release.assert_called_once()
            rows=[json.loads(line) for line in next(Path(directory).glob('*.jsonl')).read_text().splitlines()]
            self.assertEqual(sum(r['event']=='measurement_marker' for r in rows),1)
            self.assertEqual(rows[-1]['frames'],5)


if __name__=='__main__':
    unittest.main()
