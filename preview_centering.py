"""Render offline before/after centering from saved selected-person detections."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import config
from centering_preview import CenteringPreview
from controller import Controller, HOVER


def direction(yaw):
    return 'TURN LEFT' if yaw<0 else 'TURN RIGHT' if yaw>0 else 'NEUTRAL'


def replay(video, tracking, output):
    data=json.loads(tracking.read_text())
    cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS)
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps<=0 or count!=len(data['rows']) or data['summary']['file']!=video.name:
        raise ValueError('Video and saved tracking must match')
    old,new=Controller(person_mode=True),CenteringPreview()
    enabled=False
    log=[]
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,560))
    if not writer.isOpened():
        raise RuntimeError('Cannot open preview output')
    try:
        for index,row in enumerate(data['rows']):
            ok,frame=cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode frame {index}')
            seconds=index/fps
            target=row['tracks'][str(row['selected_id'])] if row['found'] else dict(found=False)
            if seconds>=1 and Controller._valid_person(target):
                enabled=True
            before,state=old.update(target)
            if not enabled:
                before=dict(HOVER)
            # User identified this clip as nearly straight-down camera footage.
            view_supported=video.stem!='20261011165245073'
            after=new.update(target,enabled=enabled,view_supported=view_supported)
            assert after['command']['forward']==after['command']['up']==after['command']['right']==0
            assert abs(after['command']['yaw'])<=.15
            if not enabled or not Controller._valid_person(target):
                assert after['command']==HOVER and after['desired_yaw']==0
            if Controller._valid_person(target):
                if target['w']>=config.STOP_SIZE or target['h']>=.65:
                    assert after['command']==HOVER
                if after['desired_yaw']:
                    assert after['desired_yaw']*(target['x']-.5)>0
            log.append(dict(frame=index,seconds=seconds,enabled=enabled,target=target,
                            before=before,after=after))
            panels=[]
            for candidate in (False,True):
                panel=np.full((560,640,3),20,np.uint8)
                panel[150:510]=cv2.resize(frame,(640,360))
                def text(value,y,color=(235,235,235),scale=.51):
                    cv2.putText(panel,value,(10,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
                text('AFTER: CENTERING PROTOTYPE' if candidate else 'BEFORE: CURRENT CONTROLLER',24)
                yaw=after['desired_yaw'] if candidate else before['yaw']
                text(f'Request: {direction(yaw)} ({abs(yaw)*100:.1f}% stick)',54,(80,220,255),.64)
                cmd=after['command'] if candidate else before
                text('Allowed output: '+direction(cmd['yaw']),80,(80,240,90) if cmd['yaw'] else (80,220,255))
                explanation=after['reason'] if candidate else (
                    'controls off' if not enabled else 'target unconfirmed' if not row['found'] else
                    'image-size stop' if (target['w']>=config.STOP_SIZE or target['h']>=.65) else
                    'turn toward selected person' if before['yaw'] else 'inside center band')
                text(explanation,104,scale=.48)
                text('OFFLINE | Forward / climb / sideways: 0',130,scale=.46)
                for x in (.44,.5,.56):
                    cv2.line(panel,(round(x*640),150),(round(x*640),510),(170,170,170),1)
                if row['found']:
                    t=target
                    x,y,w,h=t['x'],t['y'],t['w'],t['h']
                    cv2.rectangle(panel,(round((x-w/2)*640),150+round((y-h/2)*360)),
                                  (round((x+w/2)*640),150+round((y+h/2)*360)),(80,240,90),2)
                    cv2.circle(panel,(round(x*640),150+round(y*360)),5,(80,240,90),-1)
                    if yaw:
                        cv2.arrowedLine(panel,(320,480),(420 if yaw>0 else 220,480),(80,220,255),3,tipLength=.2)
                text(f'{seconds:.1f}s | Selected-person position, no predicted drone motion',531,scale=.42)
                text('Size stop still enforced | Turns shown are requests, not actual flight',552,scale=.40)
                panels.append(panel)
            writer.write(np.hstack(panels))
    finally:
        cap.release()
        writer.release()
    summary=dict(video=video.name,frames=len(log),fps=fps,
                 camera_assumption='forward-facing' if view_supported else 'near-downward; yaw centering blocked',
                 old_output=dict(Counter(direction(r['before']['yaw']) for r in log)),
                 new_requested=dict(Counter(direction(r['after']['desired_yaw']) for r in log)),
                 new_output=dict(Counter(direction(r['after']['command']['yaw']) for r in log)),
                 turns_held=sum(bool(r['after']['desired_yaw']) and r['after']['command']==HOVER for r in log),
                 inputs_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (tracking,Path(__file__).with_name('centering_preview.py'),Path(__file__).with_name('controller.py'),Path(__file__).with_name('config.py'))})
    output.with_suffix('.json').write_text(json.dumps(dict(summary=summary,rows=log),allow_nan=False))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos',type=Path,required=True)
    parser.add_argument('--evaluations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for stem in ('20261011222411553','20261011182901040','20261011165245073','20261011165024895','20261011164938939'):
        summary=replay(args.videos/f'{stem}.MOV',args.evaluations/f'{stem}-stronger-full-baseline.json',
                       args.output/f'{stem}-centering-before-after.mp4')
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
