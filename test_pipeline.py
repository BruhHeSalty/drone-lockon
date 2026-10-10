"""Quick check that vision + controller work together, no drone or camera needed.

Run:  python test_pipeline.py
It fakes a green square that starts small and off to the side, then grows.
"""
import numpy as np
import cv2

from controller import Controller
from vision import ColorTracker


def fake_frame(cx, cy, size, w=640, h=480):
    img = np.full((h, w, 3), 40, np.uint8)
    half = size // 2
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 0), -1)
    return img


def main():
    tracker = ColorTracker(hsv=(60, 255, 255))
    ctrl = Controller()

    # Off to the right and small -> should turn right, not go forward yet
    t = tracker.find(fake_frame(560, 240, 40))
    cmd, state = ctrl.update(t)
    assert t["found"] and cmd["yaw"] > 0 and cmd["forward"] == 0 and state == "LOCKED", (cmd, state)

    # Centered and small -> should fly forward
    t = tracker.find(fake_frame(320, 240, 40))
    cmd, state = ctrl.update(t)
    assert cmd["forward"] > 0 and state == "APPROACHING", (cmd, state)

    # Centered and big -> should stop
    t = tracker.find(fake_frame(320, 240, 260))
    cmd, state = ctrl.update(t)
    assert cmd["forward"] == 0 and state == "REACHED", (cmd, state)

    # Gone -> hover, and LOST after a few frames
    empty = np.full((480, 640, 3), 40, np.uint8)
    for _ in range(10):
        cmd, state = ctrl.update(tracker.find(empty))
    assert all(v == 0 for v in cmd.values()) and state == "LOST", (cmd, state)

    print("All checks passed: turns toward target, approaches, stops, hovers when lost.")


if __name__ == "__main__":
    main()
