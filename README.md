# drone-lockon

A drone that locks onto a target in its camera view (GTA-style), flies toward
it, and stops about 1–2 m away. Built on a Contixo F19 ("WiFi UAV" app).

## Setup (everyone)
```bash
git clone https://github.com/BruhHeSalty/drone-lockon
cd drone-lockon
pip install -r requirements.txt
python test_pipeline.py      # should say "All checks passed"
```

**No drone with you?** In `config.py` set `VIDEO_SOURCE = "webcam"` and
`CONTROL_PORT = None`, then run `python main.py`. Hold up something brightly
colored, **click on it** in the window, and the HUD locks on. Commands are only
printed, nothing is sent.

On a Mac, allow camera access the first time (System Settings → Privacy &
Security → Camera → turn on Terminal), then fully quit Terminal (Cmd+Q) and reopen.

## Drone setup (live video + flying)
`drone.py` talks to the drone using [Turbodrone](https://github.com/marshallrichards/turbodrone),
an open-source project that already reverse-engineered the "WiFi UAV" protocol.
Its code is **not** in this repo; download it once (no extra installs needed):
```bash
git clone https://github.com/marshallrichards/turbodrone ~/turbodrone
```
Then:
1. Turn on the drone (props off for the first test) and join its WiFi on your laptop.
2. Run `python main.py`. The drone camera shows up within a few seconds.

Your laptop has no internet while on the drone's WiFi, so pull/install first.
Close any Turbodrone web app first; only one program can talk to the drone at a time.

## Who does what
Read **INTERFACES.md** first, and paste it into your AI every time you start.

| Person | File | First goal |
|---|---|---|
| 1 | `drone.py` | Done: drone video + control (via Turbodrone). Next: tune flying |
| 2 | `vision.py` | Reliable target detection on recorded drone video |
| 3 | `hud.py` | Make the lock-on screen look awesome |
| 4 | `controller.py` | Smooth, safe movement; tune with `config.py` |

Only edit your own file. Commit small, pull often.

## Testing without the drone
Record a short clip (walk the drone toward a target, or use a phone), save it as
`test_video.mp4`, and set `VIDEO_SOURCE = "file"` in `config.py`.

## Keys
| Key | Does |
|---|---|
| click | Pick the target color |
| `T` | Take off (hovers for 3 s) |
| `A` | Autopilot on/off (starts OFF) |
| `SPACE` | Land |
| `K` | Kill motors right away (drone drops) |
| `R` | Record the HUD to a video file |
| `Q` | Quit (lands first if flying) |

## Safety
- Autopilot starts OFF (press `A`). `SPACE` = land, `K` = kill motors.
- The drone only moves by itself after you press `T` *and* turn autopilot on.
- Lost target → hover. Target close → stop. Never fly at people.
- Test with props off first. Battery is about 10 minutes, so test by hand as much as possible.
