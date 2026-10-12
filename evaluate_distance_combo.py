"""Offline comparison on saved tracking results; never sends flight commands.

Evaluates the reference clip at every frame and other clips every fifth frame.
Uses local pretrained models and reuses existing pose reports where available.
The 12ft reference is approximate; the 3ft endpoint is held out of calibration.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from body_range import torso_measurement
from distance_combo import size_depth, ground_contact, ground_depth, combine_estimates
from preview_body_range import overlap, selected_box


def face_measurement(frame, box, keypoints, detector):
    # Look for one clear face belonging to the selected person's box.
    # Require face detector AND reliable pose nose/eyes on the selected person.
    if box is None or keypoints is None:
        return None
    if any(keypoints[i][2] < .8 for i in (0, 1, 2)):
        return None
    small = cv2.resize(frame, (960, 540))
    x0, y0 = max(0, int(box[0]*960)), max(0, int(box[1]*540))
    x1, y1 = min(960, int(box[2]*960)), min(540, int(box[3]*540))
    if x1 <= x0 or y1 <= y0:
        return None
    detector.setInputSize((960,540))
    _, faces = detector.detect(small)
    if faces is None:
        return None
    candidates = []
    for detection in faces:
        x,y,w,h = map(float,detection[:4])
        b = [x/960, y/540, (x+w)/960, (y+h)/540]
        if not all(.02 < v < .98 for v in b):
            continue
        if not (box[0] <= (b[0]+b[2])/2 <= box[2] and box[1] <= (b[1]+b[3])/2 <= box[3]):
            continue
        if all(b[0] <= keypoints[i][0] <= b[2] and b[1] <= keypoints[i][1] <= b[3]
               for i in (0,1,2)):
            candidates.append(dict(width=float(w/960), box=b,
                                   anchor=[(b[0]+b[2])/2, (b[1]+b[3])/2]))
    return candidates[0] if len(candidates) == 1 else None


def extract(video, tracking, cached, model, cascade, stride):
    # Read saved pictures and gather clues: face size, torso size, and feet position.
    # A cache is earlier saved work we can reuse instead of doing it again.
    cap = cv2.VideoCapture(str(video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0 or len(tracking['rows']) != count:
        raise ValueError('Video and tracking counts must match')
    if tracking['summary']['file'] != video.name:
        raise ValueError('Tracking filename must match video')
    if cached and (Path(cached['source']).name != video.name or len(cached['rows']) != count):
        raise ValueError('Pose cache must match video')
    rows = []
    try:
        for index, track in enumerate(tracking['rows']):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode {video.name} frame {index}')
            if index % stride:
                continue
            box = selected_box(track)
            kp = None
            if box is not None:
                if cached:
                    kp = cached['rows'][index]['keypoints']
                else:
                    result = model.predict(frame, imgsz=640, device='cpu', verbose=False)[0]
                    matches = sorted([(overlap(box,b),i) for i,b in enumerate(result.boxes.xyxyn.tolist())
                                      if float(result.boxes.conf[i]) >= .6], reverse=True)
                    if matches and matches[0][0] >= .5 and (len(matches)==1 or matches[0][0]-matches[1][0] >= .15):
                        points = result.keypoints.data[matches[0][1]].cpu().numpy()
                        points[:,0] /= frame.shape[1]
                        points[:,1] /= frame.shape[0]
                        kp = points.tolist()
            target = track['tracks'][str(track['selected_id'])] if box else dict(found=False)
            torso = torso_measurement(kp) if kp is not None else None
            face = face_measurement(frame, box, kp, cascade)
            contact = ground_contact(target,kp)
            rows.append(dict(frame=index, seconds=index/fps, found=box is not None,
                             box=box, keypoints=kp, torso=torso, face=face, contact=contact))
            if index % 200 == 0:
                print(f'{video.name}: {index}/{count}', flush=True)
    finally:
        cap.release()
    return dict(source=str(video), fps=fps, total_frames=count, stride=stride, rows=rows)


def annotate(data, references, is_reference):
    # Add rough distance guesses to the saved results; missing clues stay unknown.
    for r in data['rows']:
        face = size_depth(r['face']['width'], references['face'],12) if r['face'] else None
        torso = size_depth(r['torso']['torso_height'],references['torso'],12) if r['torso'] else None
        # Supplied height/tilt are approximate and intrinsics are unknown.
        # Never turn those into an invented calibrated plane.
        ground, reason = ground_depth(r['contact'], None, None, r['seconds'])
        r.update(face_feet=face, torso_feet=torso, ground_feet=ground,
                 ground_reason=reason, combined=combine_estimates(r['found'],face,torso,ground))
    data['reference_transfer'] = 'same clip' if is_reference else 'unverified across camera angles'
    data['summary'] = dict(samples=len(data['rows']),
        confirmed=sum(r['found'] for r in data['rows']),
        face=sum(r['face'] is not None for r in data['rows']),
        torso=sum(r['torso'] is not None for r in data['rows']),
        foot_contact_candidates=sum(r['contact'] is not None for r in data['rows']),
        ground_estimates=sum(r['ground_feet'] is not None for r in data['rows']),
        combined_estimates=sum(r['combined']['feet'] is not None for r in data['rows']),
        reasons=dict(Counter(r['combined']['reason'] for r in data['rows'])))


def render(data, output):
    # Make a side-by-side movie so a person can inspect the two methods.
    cap = cv2.VideoCapture(data['source'])
    writer = cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),
                            data['fps']/data['stride'],(1280,560))
    if not writer.isOpened():
        raise RuntimeError('Cannot open comparison output')
    try:
        next_index = 0
        for r in data['rows']:
            while next_index <= r['frame']:
                ok,frame = cap.read()
                next_index += 1
                if not ok:
                    raise RuntimeError('Video decode failed')
            panels=[]
            for after in (False,True):
                panel=np.full((560,640,3),20,np.uint8)
                panel[150:510]=cv2.resize(frame,(640,360))
                def label(text,y,color=(235,235,235),scale=.51):
                    cv2.putText(panel,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
                label('AFTER: FACE + TORSO + GROUND CHECK' if after else 'BEFORE: TORSO ONLY',25)
                value=r['combined']['feet'] if after else r['torso_feet']
                label(f'Approx. {value:.1f} ft*' if value is not None else 'DISTANCE UNKNOWN',55,(80,220,255),.7)
                face_status = 'no reference' if data['calibration']['sizes']['face'] is None else 'calibrated'
                label(('Source: '+r['combined']['source']+' | Face: '+face_status)
                      if after else 'Same 12 ft starting reference',80,scale=.47)
                label(('Ground: unavailable (angle not calibrated)' if r['contact'] else 'Ground: feet not reliably visible')
                      if after else 'Needs visible shoulders and hips',105,(220,220,220),.47)
                label('EXPERIMENTAL | OFFLINE | No flight commands',130,(180,200,240),.46)
                if r['box']:
                    b=r['box']
                    cv2.rectangle(panel,(int(b[0]*640),150+int(b[1]*360)),
                                  (int(b[2]*640),150+int(b[3]*360)),(80,240,90),2)
                if r['torso']:
                    kp=r['keypoints']
                    for a,b in ((5,6),(5,11),(6,12),(11,12)):
                        cv2.line(panel,(int(kp[a][0]*640),150+int(kp[a][1]*360)),
                                 (int(kp[b][0]*640),150+int(kp[b][1]*360)),(255,220,70),2)
                if after and r['face']:
                    b=r['face']['box']
                    cv2.rectangle(panel,(int(b[0]*640),150+int(b[1]*360)),
                                  (int(b[2]*640),150+int(b[3]*360)),(220,80,240),2)
                label(f"{r['seconds']:.1f}s | *Approximate reference, not measured clearance",532,scale=.43)
                label('Angle transfer unverified' if data['reference_transfer']!='same clip' else 'End ~3 ft estimate held out of calibration',552,scale=.42)
                panels.append(panel)
            writer.write(np.hstack(panels))
    finally:
        cap.release()
        writer.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos',type=Path,required=True)
    parser.add_argument('--evaluations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    model_dir=Path(sys.prefix)/'share/drone-lockon'
    os.environ.setdefault('YOLO_CONFIG_DIR',str(model_dir))
    os.environ.setdefault('YOLO_AUTOINSTALL','false')
    from ultralytics import YOLO
    weights=model_dir/'yolo11n-pose.pt'
    if not weights.is_file():
        raise ValueError('Local pretrained pose weights required')
    model=YOLO(str(weights))
    face_weights=model_dir/'face_detection_yunet_2023mar.onnx'
    if not face_weights.is_file():
        raise ValueError('Local OpenCV YuNet face weights required')
    cascade=cv2.FaceDetectorYN.create(str(face_weights),'',(960,540),.9,.3,5000)
    reference='20261011222411553'
    stems=[reference,'20261011165245073','20261011165024895','20261011164938939','20261011182901040']
    cache_names={reference:'cropped-body-before-after.json','20261011165245073':'airborne-torso-review.json'}
    reports=[]
    for stem in stems:
        track=json.loads((args.evaluations/f'{stem}-stronger-full-baseline.json').read_text())
        cached=json.loads((args.evaluations/cache_names[stem]).read_text()) if stem in cache_names else None
        data=extract(args.videos/f'{stem}.MOV',track,cached,model,cascade,1 if stem==reference else 5)
        if stem==reference:
            reference_samples={k:[r[k]['width' if k=='face' else 'torso_height'] for r in data['rows']
                                if r['seconds']<5 and r[k] is not None] for k in ('face','torso')}
            references={k:float(np.median(v)) if len(v)>=10 else None for k,v in reference_samples.items()}
        annotate(data,references,stem==reference)
        data['calibration']=dict(reference_video=reference,feet=12,kind='user estimate',interval=[0,5],
                                sizes=references,samples={k:len(v) for k,v in reference_samples.items()},
                                camera_height_feet=[4+5/12,5],tilt='slightly down, unknown angle',
                                fov_degrees=60,fov_axis='unknown',ground_calibration=None)
        (args.output/f'{stem}-combo.json').write_text(json.dumps(data,allow_nan=False))
        render(data,args.output/f'{stem}-before-after.mp4')
        reports.append(dict(video=stem,**data['summary']))
        if stem==reference:
            # Keep this later part out of setup, so we can check it afterward.
            heldout=[r for r in data['rows'] if 20<=r['seconds']<25]
            reports[-1]['held_out_rough_3ft']=dict(samples=len(heldout),estimates=[r['combined']['feet'] for r in heldout if r['combined']['feet'] is not None])
        print(json.dumps(reports[-1]),flush=True)
        (args.output/'summary.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':
    main()
