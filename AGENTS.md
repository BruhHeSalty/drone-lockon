# Instructions for AI coding agents (Codex, Claude, Cursor)

Read PLAN.md (what we are building) and INTERFACES.md (data formats) before doing anything. It defines the data formats every file uses.

- Don't change the Target / Command / State formats in INTERFACES.md.
- Keep it simple — hackathon code. Python 3.10+, OpenCV.
- Safety rules are non-negotiable: lost target = hover, target close = stop, autopilot starts OFF, SPACE = land. Never fly at people.
- After changes, run `python test_pipeline.py` and make sure it still passes.
