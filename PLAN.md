# The Plan — what we're building

## The idea (one sentence)
A drone that you tell what to look for — a person, a car, a dog — and it finds it
with its camera, does a GTA-style lock-on, says out loud what it found like Jarvis
("Target acquired. Person, gray shirt."), then flies toward it and stops a safe
distance away.

## The drone
Contixo F19 — 143 g, ~10 min battery, 1080p camera that streams over its own WiFi
(app: "WiFi UAV"), auto-hover. Laptop does all the thinking; the drone is just
eyes and motors.

## How it works
```
Drone camera ──WiFi──> Laptop
                         ├─ vision:     find the chosen target (YOLO)
                         ├─ hud:        GTA lock-on animation on screen
                         ├─ voice:      "Target acquired. Person, gray shirt."
                         └─ controller: turn toward it, fly closer, stop
Laptop ──WiFi──> Drone (flight commands — the control hack)
```

## The demo (what judges see)
1. We say/press what to look for: "person".
2. The drone camera feed is on the big screen. Brackets slide in and lock on — red.
3. Laptop speaks: "Target acquired. Person, gray shirt."
4. Autopilot on: the drone turns to face the person and flies toward them.
5. It stops ~1–2 m away and hovers. Screen flashes "TARGET REACHED."

## Build in levels — each one is a working demo on its own
| Level | What works | Needs the control hack? |
|---|---|---|
| 1 | Lock-on HUD + voice on the laptop webcam | No |
| 2 | Same thing on the live drone camera | No |
| 3 | Drone turns to face the target by itself | Yes |
| 4 | Drone flies toward it and stops | Yes |
| Stretch | Smart description from Claude/OpenAI ("gray hoodie, holding a phone"), voice commands | No |

If the control hack fails, Level 2 is still a solid demo: a hand-flown drone with
a live AI lock-on system.

## Who does what
| Person | File | Job |
|---|---|---|
| 1 | `drone.py` | Drone video to the laptop + the control hack |
| 2 | `vision.py` | YOLO: pick a category, lock on, tell its color |
| 3 | `hud.py` + `voice.py` | The GTA animation and the Jarvis voice |
| 4 | `controller.py` | Smooth, safe flying + the pitch |

## Big unknown — test first
Does the WiFi UAV app fly the drone with the remote OFF?
- **Yes** → the control hack is possible (decompile the app, copy the joystick packets).
- **No** → we stay at Level 2 and polish it.

## Safety (non-negotiable)
- Autopilot starts OFF. SPACE = land.
- Lost target = hover. Close enough = stop. Never fly into a person.
- Test with props off first. Save battery — test by hand when possible.
