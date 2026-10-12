"""Grounded camera-only distance checking. No DroneLink or flight controls.

R records annotated video, raw frames and a timestamped JSONL journal. M adds
a measurement marker. Recorded video is silent and uses fixed 15fps playback;
the visible session clock and journal retain actual observation times.
"""
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

import config
from body_range import approximate_range_feet, torso_measurement
from controller import Controller
from drone import VideoSource
from preview_body_range import overlap
from vision import PersonTracker, NOT_FOUND

PROFILE = Path(__file__).parent/'calibration_data/20261012165212471.json'
WINDOW = 'Distance check - camera only'


def unknown(reason):
    return dict(feet=None, reason=reason, keypoints=None)


class DistanceEstimator:
    """Display-only range from a measured reference; no model training or steering."""
    MAX_AGE = .3

    def __init__(self, profile=PROFILE, model=None):
        self.profile=json.loads(Path(profile).read_text())
        self.reference_height=self.profile['reference_torso_height']
        self.reference_feet=self.profile['reference_feet']
        if not all(math.isfinite(v) and v>0 for v in (self.reference_height,self.reference_feet)):
            raise ValueError('Distance reference must contain finite positive measurements')
        if model is None:
            cache=Path(sys.prefix)/'share/drone-lockon'
            weights=cache/'yolo11n-pose.pt'
            if not weights.is_file():
                raise FileNotFoundError(f'Local pose weights required: {weights}')
            os.environ.setdefault('YOLO_CONFIG_DIR',str(cache))
            os.environ.setdefault('YOLO_AUTOINSTALL','false')
            from ultralytics import YOLO
            model=YOLO(str(weights))
            if model.task!='pose':
                raise ValueError('Expected COCO pose weights')
            model.predict(np.zeros((360,640,3),np.uint8),imgsz=640,device='cpu',verbose=False)
        self.model=model

    def estimate(self, frame, target, frame_time):
        if not Controller._valid_person(target):
            return unknown('Selected person unconfirmed')
        age=time.monotonic()-frame_time
        if not 0<=age<self.MAX_AGE:
            return unknown('Video or analysis delayed')
        height,width=frame.shape[:2]
        # Reference was recorded at 1920x1080; changed image crops need calibration.
        if abs(width/height-16/9)>.02:
            return unknown('Camera aspect ratio differs from reference')
        result=self.model.predict(frame,imgsz=640,device='cpu',verbose=False)[0]
        box=[target['x']-target['w']/2,target['y']-target['h']/2,
             target['x']+target['w']/2,target['y']+target['h']/2]
        matches=sorted([(overlap(box,b),i) for i,b in enumerate(result.boxes.xyxyn.tolist())
                        if float(result.boxes.conf[i])>=.6],reverse=True)
        if not matches or matches[0][0]<.5 or (len(matches)>1 and matches[0][0]-matches[1][0]<.15):
            return unknown('Torso association unclear')
        points=result.keypoints.data[matches[0][1]].cpu().numpy().copy()
        points[:,0]/=width
        points[:,1]/=height
        measurement=torso_measurement(points.tolist())
        feet=approximate_range_feet(measurement,self.reference_height,self.reference_feet)
        # Inference itself can make a fresh input stale. Never display that old
        # estimate as current just because it passed the check before inference.
        if time.monotonic()-frame_time>=self.MAX_AGE:
            return unknown('Video or analysis delayed')
        if feet is None:
            return unknown('Shoulders or hips cropped / unclear')
        return dict(feet=feet,reason='Experimental torso estimate',keypoints=points.tolist())


class Recording:
    """Keep source frames and analysis times so later measurements can be audited."""
    def __init__(self, directory, raw, view, profile, session_seconds):
        directory=Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        self.base=directory/f'distance_{time.time_ns()}'
        self.raw_shape=raw.shape
        self.view_shape=view.shape
        self.frames=0
        self.markers=0
        self.raw=self.hud=self.journal=None
        try:
            self.raw=cv2.VideoWriter(str(self.base)+'_raw.mp4',cv2.VideoWriter_fourcc(*'mp4v'),15,raw.shape[1::-1])
            self.hud=cv2.VideoWriter(str(self.base)+'_hud.mp4',cv2.VideoWriter_fourcc(*'mp4v'),15,view.shape[1::-1])
            if not self.raw.isOpened() or not self.hud.isOpened():
                raise RuntimeError('Cannot open distance recordings')
            self.journal=Path(str(self.base)+'.jsonl').open('x',buffering=1)
            self.event('recording_started',session_seconds=session_seconds,profile=profile,
                       playback_fps=15,audio=False,note='Use session_seconds, not playback time, for measurement timing. No flight commands.')
        except Exception:
            self.close()
            raise
        print(f'[distance] Recording: {self.base.resolve()}_hud.mp4 (+ raw.mp4 and .jsonl)',flush=True)

    def event(self, kind, **values):
        self.journal.write(json.dumps(dict(event=kind,**values),allow_nan=False)+'\n')

    def write(self, raw, view, **values):
        if raw.shape!=self.raw_shape or view.shape!=self.view_shape:
            raise ValueError('Frame size changed during recording; restart recording')
        self.raw.write(raw)
        self.hud.write(view)
        self.event('frame',index=self.frames,**values)
        self.frames+=1

    def mark(self, session_seconds, result):
        # M is a bookmark, not ground truth. The operator may have walked back
        # to the keyboard; measured distances belong to confirmed stationary intervals.
        self.markers+=1
        self.event('measurement_marker',marker=self.markers,frame_index=self.frames-1,
                   session_seconds=session_seconds,estimated_feet=result['feet'],
                   reason=result['reason'],measured_feet=None)
        return self.markers

    def close(self):
        for writer in (self.raw,self.hud):
            if writer is not None:
                writer.release()
        if self.journal is not None and not self.journal.closed:
            try:
                self.event('recording_stopped',frames=self.frames,markers=self.markers)
            finally:
                self.journal.close()


def make_view(frame, target, tracks, result, seconds, recording=False, notice=''):
    width=960
    height=round(frame.shape[0]*width/frame.shape[1])
    view=np.full((height+200,width,3),20,np.uint8)
    view[140:140+height]=cv2.resize(frame,(width,height))
    def text(s,y,scale=.6,color=(235,235,235)):
        cv2.putText(view,s,(14,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    text('GROUNDED DISTANCE CHECK | NO FLIGHT CONTROLS',27,.66,(80,220,255))
    text(f"DISTANCE: about {result['feet']:.1f} ft" if result['feet'] is not None else 'DISTANCE: UNKNOWN',65,.95)
    text(result['reason'],94,.58)
    text('Same-person torso reference | Angle/posture changes can affect the estimate',120,.51)
    for box in tracks.values():
        if box['conf']<.25:
            continue
        x,y,w,h=(box[k] for k in ('x','y','w','h'))
        cv2.rectangle(view,(round((x-w/2)*width),140+round((y-h/2)*height)),
                      (round((x+w/2)*width),140+round((y+h/2)*height)),(150,150,150),1)
    if target.get('found'):
        x,y,w,h=(target[k] for k in ('x','y','w','h'))
        cv2.rectangle(view,(round((x-w/2)*width),140+round((y-h/2)*height)),
                      (round((x+w/2)*width),140+round((y+h/2)*height)),(80,240,90),2)
    if result['keypoints'] is not None:
        kp=result['keypoints']
        for a,b in ((5,6),(5,11),(6,12),(11,12)):
            cv2.line(view,(round(kp[a][0]*width),140+round(kp[a][1]*height)),
                     (round(kp[b][0]*width),140+round(kp[b][1]*height)),(255,220,70),2)
    text(f"Time {seconds:.1f}s | {'RECORDING' if recording else 'Not recording'} | {notice}",height+165,.55,(80,220,255))
    text('Click yourself | R record/stop | M measurement marker | Q/Esc quit | No audio',height+189,.49)
    return view


def run(person_options=None, output='distance_recordings', profile=PROFILE):
    # A fresh process never constructs DroneLink. Video requests are still needed
    # to receive the camera stream, but flight keys/sender are absent here.
    config.CONTROL_PORT=None
    estimator=DistanceEstimator(profile)
    tracker=PersonTracker(allow_live_yaw=True,**(person_options or {}))
    video=None
    recorder=None
    last_frame=None
    last_result=unknown('Click yourself to start')
    target=dict(NOT_FOUND)
    started=time.monotonic()
    notice=''

    def on_click(event,x,y,flags,param):
        nonlocal target,last_result
        if event!=cv2.EVENT_LBUTTONDOWN or last_frame is None:
            return
        h,w=last_frame.shape[:2]
        display_h=round(h*960/w)
        if not (0<=x<960 and 140<=y<140+display_h):
            return
        target=dict(NOT_FOUND)
        last_result=unknown('Confirming selection')
        tracker.set_color_from_pixel(last_frame,int(x*w/960),int((y-140)*h/display_h))

    try:
        video=VideoSource()
        cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(WINDOW,on_click)
        print('[distance] Keep drone grounded/motors off. Click yourself; R records; M marks a phone reading. No audio or flight controls.',flush=True)
        while True:
            frame=video.read()
            frame_time=video.frame_time if frame is not None else None
            fresh=frame is not None and video.fresh
            if fresh:
                last_frame=frame
                target=tracker.find(frame)
                last_result=estimator.estimate(frame,target,frame_time)
            else:
                last_frame=None  # Missing video must not accept clicks on old boxes.
                target=dict(NOT_FOUND)
                last_result=unknown('Waiting for fresh video')
            seconds=time.monotonic()-started
            display=frame if frame is not None else np.zeros((540,960,3),np.uint8)
            view=make_view(display,target,tracker._tracks if fresh else {},last_result,seconds,recorder is not None,notice)
            if recorder is not None and frame is not None:
                recorder.write(frame,view,session_seconds=seconds,source_frame_time=frame_time,
                               video_fresh=bool(fresh),target=target,result=last_result)
            cv2.imshow(WINDOW,view)
            key=cv2.waitKey(1)&0xFF
            if ord('A')<=key<=ord('Z'):
                key+=32
            if key in (ord('q'),27) or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                break
            if key==ord('r'):
                if recorder is not None:
                    recorder.close();recorder=None
                    notice='Recording saved'
                    print('[distance] Recording saved',flush=True)
                elif fresh:
                    recorder=Recording(output,frame,view,estimator.profile,seconds)
                    notice=''
                else:
                    notice='Wait for video before recording'
            elif key==ord('m'):
                if recorder is not None:
                    marker=recorder.mark(seconds,last_result)
                    notice=f'Marker {marker}: note your iPhone reading'
                    print(f'[distance] Marker {marker} at {seconds:.1f}s; estimate={last_result["feet"]}',flush=True)
                else:
                    notice='Press R before adding a marker'
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            try:
                if video is not None:
                    video.release()
            finally:
                cv2.destroyAllWindows()
