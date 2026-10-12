"""Measure selected-person boxes from offline evaluation JSON, without flight I/O.

Example: python calibrate_tracking.py TRACKS.json --start 40 --end 50
Add --distance-feet only for an interval at one distance. Use --distance-kind
estimated for approximate distances. Output is a report, never a control setting.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from controller import Controller


def measure(rows, start, end, distance_feet=None, distance_kind='measured'):
    # Each row describes one saved picture. Inspect only the chosen time slice.
    if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
        raise ValueError('Use finite times with 0 <= start < end')
    if distance_feet is not None and (not math.isfinite(distance_feet) or distance_feet <= 0):
        raise ValueError('Distance must be finite and positive')
    if distance_kind not in ('measured', 'estimated'):
        raise ValueError('Distance kind must be measured or estimated')
    selected = [r for r in rows if start <= r['seconds'] < end]
    if not selected:
        raise ValueError('No frames in the requested interval')
    valid, complete = [], []
    # Keep clear detections and fully visible bodies in separate piles.
    close = 0
    ctrl = Controller(person_mode=True)
    for row in selected:
        if not row['found']:
            continue
        target = row['tracks'].get(str(row['selected_id']))
        if target is None or not ctrl._valid_person(target):
            continue
        valid.append(target)
        _, state = ctrl.update(target)
        close += state == 'REACHED'
        if ctrl._fully_visible(target):
            complete.append(target)

    def stats(targets):
        # Summarize small, middle, and large box sizes instead of trusting one picture.
        return {axis: dict(zip(('p10', 'median', 'p90'),
                              (float(v) for v in np.percentile([t[axis] for t in targets], [10, 50, 90]))))
                for axis in ('w', 'h')} if targets else None

    return dict(start_seconds=start, end_seconds=end, frames=len(selected),
                confirmed_frames=len(valid), complete_body_frames=len(complete),
                close_stop_frames=close, all_confirmed_box_fractions=stats(valid),
                complete_body_box_fractions=stats(complete),
                user_supplied_distance_feet=distance_feet,
                distance_kind=distance_kind if distance_feet is not None else None,
                user_supplied_measured_distance_feet=distance_feet if distance_kind == 'measured' else None,
                status=('DISTANCE_NOT_MEASURED' if distance_feet is None else
                        'ESTIMATED_REFERENCE_ONLY' if distance_kind == 'estimated' else 'MEASUREMENTS_ONLY'),
                note='No thresholds changed. A distance label assumes the same person, camera, zoom, pose and distance throughout this interval. This report does not establish stopping distance or validate flight.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tracks', type=Path, help='Full-frame evaluation JSON containing rows')
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--end', type=float, required=True)
    parser.add_argument('--distance-feet', type=float, help='User-supplied subject distance over this interval')
    parser.add_argument('--distance-kind', choices=('measured', 'estimated'), default='measured')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.tracks.read_text())
        result = measure(data['rows'], args.start, args.end, args.distance_feet, args.distance_kind)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    result['source'] = str(args.tracks.resolve())
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    # Save or print a readable report; this does not change the flight settings.
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
