# INTERFACES — paste this into your AI before every coding session

We're building a drone that locks onto a target in its camera view and flies
toward it, then STOPS and hovers about 1–2 m away. Drone: Contixo F19
(app: "WiFi UAV"). Language: Python 3.10+, OpenCV.

## Rules for the AI
- Only edit the file you own (see table). Don't rewrite other people's files.
- Don't change the data formats below without telling the whole team.
- Keep it simple. Hackathon code, not production code.

## Who owns what
| File            | Owner    | Job                                                    |
|-----------------|----------|--------------------------------------------------------|
| `drone.py`      | Person 1 | Get video frames from the drone, send commands to it   |
| `vision.py`     | Person 2 | Find the target in a frame                             |
| `hud.py`        | Person 3 | Draw the GTA-style lock-on overlay                     |
| `controller.py` | Person 4 | Turn target position into flight commands + safety     |
| `main.py`       | Person 1 | Wires everything together (small, rarely changes)      |
| `config.py`     | Everyone | Shared settings — tell the team when you change one    |

## Data formats

### Frame
A normal OpenCV image: `numpy` array, shape `(height, width, 3)`, BGR colors.

### Target  (vision.py → controller.py, hud.py)
```python
{
    "found":  bool,   # is the target visible this frame?
    "x":      float,  # center of target, 0.0 = left edge,  1.0 = right edge
    "y":      float,  # center of target, 0.0 = top edge,   1.0 = bottom edge
    "w":      float,  # target width  as a fraction of frame width  (0–1)
    "h":      float,  # target height as a fraction of frame height (0–1)
    "conf":   float,  # 0–1, how sure we are
}
```
If `found` is False, the other numbers don't matter.

### Command  (controller.py → drone.py)
```python
{
    "yaw":     float,  # -1 = turn left full,  +1 = turn right full
    "up":      float,  # -1 = descend full,    +1 = climb full
    "forward": float,  # -1 = back full,       +1 = forward full
    "right":   float,  # -1 = left full,       +1 = right full (strafe)
}
```
All zeros = hover in place. `drone.py` converts these to whatever bytes the
drone actually wants.

### State  (controller.py → hud.py)
One of these strings: `"SEARCHING"`, `"LOCKED"`, `"APPROACHING"`, `"REACHED"`, `"LOST"`.

## Safety rules (non-negotiable)
- Target lost → command all zeros (hover). Never keep flying blind.
- Target big enough in frame (`config.STOP_SIZE`) → stop forward motion. Never fly into a person.
- Speeds are capped by `config.MAX_*`. Keep them low.
- Autopilot starts OFF. Pressing SPACE always sends emergency land.
