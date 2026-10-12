"""Experimental torso-size measurements for offline or grounded camera review.

Feet may be cropped, but both shoulders and hips must be confidently visible.
Apparent torso length is a relative size cue, not a depth sensor. No controller
imports this module; approximate distance labels must never enable flight.
"""
import math

# A torso is the middle of the body, between shoulders and hips.
# Measure its size in an image. This helper never tells the drone to fly.

def torso_measurement(keypoints):
    """Accept normalized COCO (x, y, confidence) points or return None.

    Reject offscreen/weak joints, tiny torsos and substantial lean. Confident
    predictions can still be inferred for occluded joints.
    These checks do not prove correct pose or identity; associate with the
    selected person's box separately before calling this function.
    """
    if len(keypoints) != 17:
        return None
    joints = [keypoints[i] for i in (5, 6, 11, 12)]
    # The finder numbers body points. These four are the two shoulders and two hips.
    if any(len(p) != 3 or not all(math.isfinite(float(v)) for v in p)
           or not (.02 < p[0] < .98 and .02 < p[1] < .98 and .8 <= p[2] <= 1)
           for p in joints):
        return None
    ls, rs, lh, rh = joints
    # Average each pair, then measure the picture gap from shoulders down to hips.
    shoulder_y, hip_y = (ls[1]+rs[1])/2, (lh[1]+rh[1])/2
    height = hip_y - shoulder_y
    if height < .06 or min(lh[1]-ls[1], rh[1]-rs[1]) < .04:
        return None
    if abs((ls[0]+rs[0]-lh[0]-rh[0])/2) > height*.5:
        # Leaning changes the visible size, so do not use a strongly tilted body.
        return None
    if abs(ls[1]-rs[1]) > height*.3 or abs(lh[1]-rh[1]) > height*.3:
        return None
    return dict(torso_height=height, shoulder_y=shoulder_y, hip_y=hip_y)


def approximate_range_feet(measurement, reference_height, reference_feet):
    """Same-subject/camera/upright-pose illustration, not a flight distance."""
    if measurement is None:
        return None
    values = (measurement['torso_height'], reference_height, reference_feet)
    if not all(math.isfinite(v) and v > 0 for v in values):
        return None
    # Under the same viewing conditions, half the picture size suggests twice
    # the distance. Posture or camera changes can spoil this rough guess.
    return reference_feet * reference_height / measurement['torso_height']
