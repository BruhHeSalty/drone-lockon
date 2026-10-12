"""Offline size/ground distance experiments. Never imported by flight control.

Size uses reference_depth * reference_image_size / current_image_size.
Ground uses a calibrated camera ray intersecting a ground plane, then projects
up the person's vertical line to the SAME image anchor as the size estimate.
Unknown camera pose is not replaced by an invented plane.
"""
import math

import numpy as np


def size_depth(size, reference_size, reference_feet):
    # Compare picture size with an earlier size at a supplied distance.
    # If the feature looks twice as big, the rough distance becomes half as much.
    values = (size, reference_size, reference_feet)
    if any(v is None or not math.isfinite(v) or v <= 0 for v in values):
        return None
    return reference_feet * reference_size / size


def ground_contact(target, keypoints):
    """Candidate foot contact; no distance is inferred from it alone.

    Require both reliable ankles and an uncropped box bottom close to the
    ankles. Box bottom is still a heuristic, not a measured sole position.
    """
    if not target.get('found') or keypoints is None or len(keypoints) != 17:
        return None
    ankles = [keypoints[i] for i in (15, 16)]
    # Both ankles must be clear before we even guess where the feet meet the floor.
    if any(len(p) != 3 or not np.isfinite(p).all() or p[2] < .8
           or not (.02 < p[0] < .98 and .02 < p[1] < .98) for p in ankles):
        return None
    bottom = target['y'] + target['h']/2
    ankle_y = max(p[1] for p in ankles)
    if not .02 < bottom < .98 or not -.01 <= bottom-ankle_y <= .06:
        return None
    return (sum(p[0] for p in ankles)/2, bottom)


def ground_depth(contact, anchor, calibration, seconds):
    """Return camera-axis feet at the body anchor, plus explicit failure reason.

    Calibration is scoped to a known valid time interval. Camera axes are
    right/down/forward; plane is normal.dot(point) = offset_feet, with the
    normal directed toward the ground. Intrinsics use normalized image units.
    Per-frame pose or a verified fixed pose is required for moving footage.
    """
    if contact is None:
        return None, 'feet not reliably visible'
    if calibration is None:
        return None, 'camera height/tilt and ground plane not calibrated'
    if not calibration.get('pose_verified', False):
        return None, 'camera pose not verified for this frame'
    try:
        if not calibration['valid_from'] <= seconds <= calibration['valid_until']:
            return None, 'ground calibration expired'
        normal = np.asarray(calibration['normal'], dtype=float)
        fx, fy = calibration['fx_norm'], calibration['fy_norm']
        offset = calibration['offset_feet']
        cx, cy = calibration.get('cx_norm', .5), calibration.get('cy_norm', .5)
        values = [*normal, fx, fy, offset, cx, cy, *contact, *anchor]
        if normal.shape != (3,) or not np.isfinite(values).all() or min(fx, fy, offset) <= 0:
            return None, 'invalid ground calibration'
        norm = float(np.linalg.norm(normal))
        if abs(norm-1) > .001:
            return None, 'ground normal must be a unit vector'
        ray = np.array([(contact[0]-cx)/fx, (contact[1]-cy)/fy, 1.])
        # Imagine a straight line from the camera through the feet in the picture.
        # Find where that line meets the floor described by the supplied setup.
        denom = float(normal @ ray)
        if denom <= .05:
            return None, 'ground ray too close to horizon or points away'
        foot = ray * (offset / denom)
        body_ray = np.array([(anchor[0]-cx)/fx, (anchor[1]-cy)/fy, 1.])
        # A point on the body lies above the foot along the world's vertical.
        matrix = np.column_stack((body_ray, normal))
        solution, _, rank, _ = np.linalg.lstsq(matrix, foot, rcond=None)
        depth, body_height = map(float, solution)
        residual = float(np.linalg.norm(matrix @ solution-foot))
        if rank < 2 or depth <= 0 or not 0 <= body_height <= 7 or residual > .35:
            return None, 'body anchor and ground contact do not agree geometrically'
        return depth, None
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None, 'invalid ground calibration'


def combine_estimates(found, face, torso, ground=None, ground_anchor=None):
    """Use a fresh compatible estimate; disagreement returns unknown.

    The 30% agreement gate is an experimental rejection rule, not an accuracy
    bound. These cues share calibration and do not independently validate it.
    """
    # Compare the guesses. When they disagree too much, say "unknown".
    # Agreement is just a check; it does not prove that a guess is correct.
    if not found:
        return dict(feet=None, source='unknown', reason='selected person not confirmed')
    values = {k:v for k,v in [('face', face), ('torso', torso)]
              if v is not None and math.isfinite(v) and v > 0}
    if len(values) == 2 and max(values.values())/min(values.values()) > 1.3:
        return dict(feet=None, source='unknown', reason='face and torso disagree')
    source = 'face' if 'face' in values else 'torso' if 'torso' in values else None
    if ground is not None and math.isfinite(ground) and ground > 0:
        if source is not None and ground_anchor != source:
            return dict(feet=None, source='unknown', reason='ground and size refer to different points')
        if source is not None and max(values[source], ground)/min(values[source], ground) > 1.3:
            return dict(feet=None, source='unknown', reason='ground and size disagree')
        if source is not None:
            return dict(feet=values[source], source=source+' + ground check', reason='approximate, not flight calibrated')
        return dict(feet=ground, source='ground', reason='approximate ground geometry')
    if source is not None:
        return dict(feet=values[source], source=source, reason='approximate size only; no ground check')
    return dict(feet=None, source='unknown', reason='no calibrated visible feature')
