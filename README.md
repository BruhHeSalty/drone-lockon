# drone-lockon

A drone that locks onto a target in its camera view (GTA-style), flies toward
it, and stops about 1–2 m away. Built on a Contixo F19 ("WiFi UAV" app).

## Setup (everyone)
```bash
git clone https://github.com/BruhHeSalty/drone-lockon
cd drone-lockon
pip install -r requirements.txt
python test_pipeline.py      # should say "All checks passed"
python main.py               # opens your webcam with the lock-on HUD
```
Hold up something brightly colored, **click on it** in the window, and the HUD
locks on. Nothing is sent to the drone yet; commands are only printed.

## Who does what
Read **INTERFACES.md** first, and paste it into your AI every time you start.

| Person | File | First goal |
|---|---|---|
| 1 | `drone.py` | Drone video on the laptop, then crack the control packets |
| 2 | `vision.py` | Reliable target detection on recorded drone video |
| 3 | `hud.py` | Make the lock-on screen look awesome |
| 4 | `controller.py` | Smooth, safe movement; tune with `config.py` |

Only edit your own file. Commit small, pull often.

## Testing without the drone
Record a short clip (walk the drone toward a target, or use a phone), save it as
`test_video.mp4`, and set `VIDEO_SOURCE = "file"` in `config.py`.

## Safety
- Autopilot starts OFF (press `A`). `SPACE` = emergency land.
- Lost target → hover. Target close → stop. Never fly at people.
- Test with props off first. Battery is about 10 minutes, so test by hand as much as possible.
