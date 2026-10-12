"""Distance display for yaw-only tests. Estimates never enter flight control.

Pose inference runs on the exact frame matched by the person worker. A single
pending job is replaced by newer data; controls never wait for pose inference.
"""
import threading
import time

import cv2
import numpy as np

from controller import Controller
from distance_check import DistanceEstimator, unknown

WIDTH = 960
HEADER = 150
FOOTER = 100


class AsyncDistance:
    def __init__(self, estimator=None):
        # Warm the model before main opens a drone connection.
        self.estimator = estimator if estimator is not None else DistanceEstimator()
        self.profile = self.estimator.profile
        self._condition = threading.Condition()
        self._pending = None
        self._epoch = 0
        self._selection = None
        self._last_submit = float('-inf')
        self._result = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _reset_locked(self):
        self._epoch += 1
        self._pending = self._result = None
        self._last_submit = float('-inf')

    def reset(self):
        with self._condition:
            self._reset_locked()

    def update(self, observation):
        now = time.monotonic()
        with self._condition:
            if (observation is None or not Controller._valid_person(observation['target'])
                    or not 0 <= now-observation['source_frame_time'] < DistanceEstimator.MAX_AGE):
                self._reset_locked()
                return unknown('Waiting for a fresh, confirmed selection')
            if observation['generation'] != self._selection:
                self._reset_locked()
                self._selection = observation['generation']
            stamp = observation['source_frame_time']
            if stamp > self._last_submit and not self._closed:
                self._last_submit = stamp
                self._pending = (self._epoch, stamp, observation['frame'].copy(), dict(observation['target']))
                self._condition.notify()
            if self._result is None:
                return unknown('Measuring visible torso')
            stamp, result = self._result
            age = now-stamp
            if not 0 <= age < DistanceEstimator.MAX_AGE:
                return unknown('Distance analysis delayed')
            return dict(result, source_frame_time=stamp, age_s=age)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                epoch, stamp, frame, target = self._pending
                self._pending = None
            try:
                result = self.estimator.estimate(frame, target, stamp)
            except Exception:
                # A display-only model failure must not close the flight UI.
                result = unknown('Distance analysis unavailable')
            with self._condition:
                if not self._closed and epoch == self._epoch:
                    self._result = (stamp, dict(result, measured_target=target))
                    self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._reset_locked()
            self._condition.notify_all()
        self._thread.join(timeout=.5)


def source_click(frame, x, y):
    h, w = frame.shape[:2]
    display_h = round(h * WIDTH / w)
    if not (0 <= x < WIDTH and HEADER <= y < HEADER+display_h):
        return None
    return int(x*w/WIDTH), int((y-HEADER)*h/display_h)


def make_view(frame, observation, result, seconds, autopilot, command, live, recording, fresh):
    h, w = frame.shape[:2]
    display_h = round(h*WIDTH/w)
    view = np.full((HEADER+display_h+FOOTER, WIDTH, 3), 20, np.uint8)
    view[HEADER:HEADER+display_h] = cv2.resize(frame, (WIDTH, display_h))

    def text(label, y, size=.55, color=(235,235,235)):
        cv2.putText(view, label, (14,y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    text('LIVE TURN-ONLY TEST' if live else 'DISTANCE + TURN PREVIEW | NO FLIGHT COMMANDS', 25, .65)
    feet = result.get('feet')
    text(f'DISTANCE: about {feet:.1f} ft' if feet is not None else 'DISTANCE: UNKNOWN', 65, 1.)
    text(result['reason'], 94)
    text('Experimental: camera angle/posture affect distance. Display only.', 119, .5)
    text(f"AUTO {'ON' if autopilot else 'OFF'} | YAW {command['yaw']:+.2f} | "
         f"{'VIDEO FRESH' if fresh else 'VIDEO UNAVAILABLE'} | {seconds:.1f}s", 141, .5, (80,220,255))
    if observation is not None:
        for tid, target in observation['tracks'].items():
            if target['conf'] < .25:
                continue
            x,y,bw,bh = (target[k] for k in ('x','y','w','h'))
            x1,y1 = round((x-bw/2)*WIDTH), HEADER+round((y-bh/2)*display_h)
            x2,y2 = round((x+bw/2)*WIDTH), HEADER+round((y+bh/2)*display_h)
            cv2.rectangle(view,(x1,y1),(x2,y2),(90,210,210),1)
            cv2.putText(view, f'person {tid}', (max(0,x1),max(HEADER+12,y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,.4,(90,210,210),1)
        status = observation['status']
    else:
        status = 'Waiting for fresh person detection'
    bottom = HEADER+display_h
    text(('RECORDING | ' if recording else 'Not recording | ')+status[:100], bottom+22, .48)
    text('T takeoff + record | Click yourself | A after steady hover: turn only', bottom+45, .48)
    text('R record/stop | M marker | H tracking off | SPACE landing request', bottom+67, .48)
    text('K or Q/close: CUT MOTORS - airborne drone drops' if live else 'Q/close: exit preview',
         bottom+89, .5, (80,160,255))
    return view
