"""Offline detector/tracker timing probe; never opens a camera or flight link.

Without --source, uses Ultralytics' bundled bus image, NOT drone footage.
Counts are detections, not accuracy scores; use labeled held-out video for accuracy.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

import config
from vision import PersonTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='Local image or recorded video (no live camera)')
    parser.add_argument('--models', nargs='+', default=['yolo11n.pt'])
    parser.add_argument('--trackers', nargs='+', choices=['botsort', 'bytetrack'], default=['botsort'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--output', help='Write JSON results here')
    args = parser.parse_args()
    if not 5 <= args.frames <= 300:
        parser.error('--frames must be between 5 and 300')
    config.CONTROL_PORT = None
    if args.source:
        source = Path(args.source).expanduser()
    else:
        import importlib.util
        source = Path(importlib.util.find_spec('ultralytics').origin).parent / 'assets' / 'bus.jpg'
    if not source.is_file():
        parser.error(f'Local image/video not found: {source}')
    frames = []
    image = cv2.imread(str(source))
    if image is not None:
        frames = [image] * args.frames
    else:
        cap = cv2.VideoCapture(str(source))
        try:
            for _ in range(args.frames):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
    if not frames:
        parser.error('Source contains no readable frames')

    results = []
    for model in args.models:
        for tracker_name in args.trackers:
            for degraded in (False, True):
                tracker = PersonTracker(allow_live_yaw=True, model=model, device=args.device,
                                        imgsz=args.imgsz, tracker=tracker_name + '.yaml')
                # Prime tracker initialization separately from steady-state timing.
                tracker.find(frames[0])
                durations, counts = [], []
                for frame in frames:
                    if degraded:
                        h, w = frame.shape[:2]
                        small = cv2.resize(frame, (max(1, w // 3), max(1, h // 3)))
                        small = cv2.GaussianBlur(small, (3, 3), 0)
                        ok, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 35])
                        if not ok:
                            raise RuntimeError('JPEG test encoding failed')
                        frame = cv2.resize(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), (w, h))
                    start = time.perf_counter()
                    tracker.find(frame)
                    durations.append((time.perf_counter() - start) * 1000)
                    counts.append(sum(t['conf'] >= .6 for t in tracker._tracks.values()))
                row = dict(model=model, tracker=tracker_name, device=tracker.device,
                           imgsz=args.imgsz, degraded=degraded, frames=len(durations),
                           median_ms=round(float(np.median(durations)), 1),
                           p95_ms=round(float(np.percentile(durations, 95)), 1),
                           over_300ms=sum(ms >= 300 for ms in durations),
                           mean_confident_detections=round(float(np.mean(counts)), 2))
                results.append(row)
                print(json.dumps(row))
    report = dict(source=str(source), note='Timing and detection counts only; not an accuracy or flight test.',
                  results=results)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
