"""Offline comparison using saved tracking results and visible torso landmarks.

This program has no drone/control connection. It illustrates approximate range
from a user-estimated reference, never authorizes motion or trains a model.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from body_range import torso_measurement, approximate_range_feet


def overlap(a, b):
    # How much do two boxes cover the same area? 0 = none, 1 = a perfect match.
    area = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    return area / max(1e-8, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def selected_box(row):
    # Turn the chosen box's middle and size into its four edges.
    if not row['found']:
        return None
    t = row['tracks'][str(row['selected_id'])]
    return [t['x']-t['w']/2, t['y']-t['h']/2, t['x']+t['w']/2, t['y']+t['h']/2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--before', type=Path, required=True, help='Original full-frame evaluation JSON')
    parser.add_argument('--after', type=Path, required=True, help='Candidate full-frame evaluation JSON')
    parser.add_argument('--weights', type=Path, required=True, help='Local pretrained COCO pose weights')
    parser.add_argument('--reference-feet', type=float, help='Optional approximate user reference; omit to inspect landmarks without distance labels')
    parser.add_argument('--reference-start', type=float, default=0)
    parser.add_argument('--reference-end', type=float, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (not np.isfinite([args.reference_start, args.reference_end]).all()
            or not 0 <= args.reference_start < args.reference_end
            or (args.reference_feet is not None and (not np.isfinite(args.reference_feet) or args.reference_feet <= 0))):
        parser.error('Use finite reference times and a positive reference distance')
    if not args.weights.is_file() or not args.video.is_file():
        parser.error('Video and model weights must be local files')
    inputs = {p.resolve() for p in (args.video, args.before, args.after, args.weights)}
    if args.output.resolve() in inputs or args.output.with_suffix('.json').resolve() in inputs:
        parser.error('Output must not overwrite an input file')
    old, new = [json.loads(path.read_text()) for path in (args.before, args.after)]
    for data in (old, new):
        if data['summary']['file'] != args.video.name:
            parser.error('Tracking results do not match the source filename')
    cap = cv2.VideoCapture(str(args.video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0 or len(old['rows']) != count or len(new['rows']) != count:
        cap.release()
        parser.error('Video and tracking frame counts must match')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(sys.prefix)/'share/drone-lockon'))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    if model.task != 'pose':
        cap.release()
        parser.error('Weights must be a COCO pose model with 17 keypoints')
    measured = []
    # First pass: match body points to the chosen person in each saved picture.
    try:
        for index, row in enumerate(new['rows']):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode frame {index}')
            keypoints, measurement = None, None
            box = selected_box(row)
            if box is not None:
                result = model.predict(frame, imgsz=640, device='cpu', verbose=False)[0]
                candidates = sorted([(overlap(box, b), i) for i, b in enumerate(result.boxes.xyxyn.tolist())
                                     if float(result.boxes.conf[i]) >= .6], reverse=True)
                if candidates and candidates[0][0] >= .5 and (len(candidates) == 1 or candidates[0][0]-candidates[1][0] >= .15):
                    keypoints = result.keypoints.data[candidates[0][1]].cpu().numpy()
                    keypoints[:, 0] /= frame.shape[1]
                    keypoints[:, 1] /= frame.shape[0]
                    keypoints = keypoints.tolist()
                    measurement = torso_measurement(keypoints)
            measured.append(dict(seconds=index/fps, measurement=measurement, keypoints=keypoints))
            if index % 200 == 0:
                print(f'Pose review {index}/{count}', flush=True)
    finally:
        cap.release()
    samples = [r['measurement']['torso_height'] for r in measured if r['measurement'] is not None
               and args.reference_start <= r['seconds'] < args.reference_end]
    if args.reference_feet is not None and len(samples) < 10:
        raise ValueError('Reference interval has fewer than 10 usable torso measurements')
    reference_height = float(np.median(samples)) if samples and args.reference_feet is not None else None
    # Use the middle sample as the starting size so one odd picture matters less.
    for r in measured:
        r['approximate_feet'] = (approximate_range_feet(r['measurement'], reference_height, args.reference_feet)
                                 if reference_height is not None else None)
    report = dict(source=str(args.video), reference_feet=args.reference_feet,
                  reference_kind='user_estimate' if args.reference_feet is not None else None,
                  reference_interval=[args.reference_start, args.reference_end] if args.reference_feet is not None else None,
                  reference_samples=len(samples) if args.reference_feet is not None else 0,
                  reference_height=reference_height, torso_frames=sum(r['measurement'] is not None for r in measured),
                  note='Experimental same-person size cue only; posture/camera changes affect it. No flight commands. Missing joints produce no estimate, never a stale distance.', rows=measured)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    cap = cv2.VideoCapture(str(args.video))
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 520))
    # Second pass: put the old and new results side by side in a review movie.
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Cannot open output video')
    try:
        for index, measurement in enumerate(measured):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode frame {index}')
            panels = []
            for candidate, data in enumerate((old, new)):
                row = data['rows'][index]
                panel = np.full((520, 640, 3), (25, 20, 17), dtype=np.uint8)
                panel[120:480] = cv2.resize(frame, (640, 360))
                box = selected_box(row)
                color = (80, 240, 90) if box is not None else (80, 220, 255)
                if box is not None:
                    cv2.rectangle(panel, (int(box[0]*640), 120+int(box[1]*360)),
                                  (int(box[2]*640), 120+int(box[3]*360)), color, 2)
                title = 'BEFORE: SMALL DETECTOR' if not candidate else 'AFTER: STRONGER DETECTOR + TORSO REVIEW'
                cv2.putText(panel, title, (10,25), cv2.FONT_HERSHEY_SIMPLEX, .48, (235,235,235), 1, cv2.LINE_AA)
                cv2.putText(panel, 'TARGET CONFIRMED' if box else 'TARGET UNCONFIRMED', (10,52),
                            cv2.FONT_HERSHEY_SIMPLEX, .57, color, 1, cv2.LINE_AA)
                if not candidate:
                    size_stop = box is not None and (box[2]-box[0] >= .30 or box[3]-box[1] >= .65)
                    detail = ('Old size rule: TARGET REACHED' if size_stop else
                              'Old box-size stop: clear' if box else 'Neutral: target uncertain')
                elif measurement['measurement'] is not None:
                    detail = (f"Torso-size estimate: about {measurement['approximate_feet']:.0f} ft*"
                              if measurement['approximate_feet'] is not None else
                              'Torso landmarks visible | Distance NOT calibrated')
                    kp = measurement['keypoints']
                    for a, b in ((5,6),(5,11),(6,12),(11,12)):
                        cv2.line(panel, (int(kp[a][0]*640),120+int(kp[a][1]*360)),
                                 (int(kp[b][0]*640),120+int(kp[b][1]*360)), (255,220,70), 2)
                else:
                    detail = 'Distance UNKNOWN: torso landmarks unclear'
                cv2.putText(panel, detail, (10,79), cv2.FONT_HERSHEY_SIMPLEX, .48, (80,220,255), 1, cv2.LINE_AA)
                cv2.putText(panel, 'OFFLINE ONLY | No flight commands', (10,105), cv2.FONT_HERSHEY_SIMPLEX, .48, (210,210,210), 1, cv2.LINE_AA)
                footer = (f"{measurement['seconds']:.1f}s | *Reference estimated at {args.reference_feet:g} ft; not measured depth"
                          if args.reference_feet is not None else
                          f"{measurement['seconds']:.1f}s | Camera angle changes apparent size; no distance assumed")
                cv2.putText(panel, footer, (10,504), cv2.FONT_HERSHEY_SIMPLEX, .42, (210,210,210), 1, cv2.LINE_AA)
                panels.append(panel)
            writer.write(np.hstack(panels))
    finally:
        cap.release()
        writer.release()
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)
    print(args.output)


if __name__ == '__main__':
    main()
