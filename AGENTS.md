# Instructions for AI coding agents (Codex, Claude, Cursor)

Read INTERFACES.md before doing anything. It defines the data formats every file uses.

- Only edit the file the user says they own (drone.py, vision.py, hud.py, controller.py). Don't rewrite other files.
- Don't change the Target / Command / State formats in INTERFACES.md.
- Keep it simple — hackathon code. Python 3.10+, OpenCV.
- Safety rules are non-negotiable: lost target = hover, target close = stop, autopilot starts OFF, SPACE = land. Never fly at people.
- After changes, run `python test_pipeline.py` and make sure it still passes.

## Cursor Cloud specific instructions

- `python` is `/usr/local/bin/python` (a symlink to `python3`). Install dependencies with `sudo python -m pip install --break-system-packages -r requirements.txt` (`opencv-python` and `numpy`).
- `python test_pipeline.py` is the check that the vision and controller pipeline works. It needs no camera, display, or drone, and it should print `All checks passed`.
- `python main.py` opens an OpenCV window named `drone-lockon`. There is no webcam here, and `VIDEO_SOURCE` defaults to `"webcam"`, so that window receives frames only when a camera is attached or `VIDEO_SOURCE` is `"file"` and `test_video.mp4` exists. While `CONTROL_PORT` is `None`, flight commands are printed and not sent.
- The window needs a display (`DISPLAY`). A Qt message about a missing font directory on startup does not affect the HUD.
