"""OWNER: Person 3 — the GTA-style lock-on overlay. This is what judges remember.

Ideas to add: lock-on beep sound, pulsing reticle, distance estimate,
"TARGET REACHED" flash, a recording indicator.
"""
import time

import cv2

WHITE = (255, 255, 255)
RED = (40, 40, 255)
GREEN = (80, 255, 80)
YELLOW = (0, 220, 255)

STATE_COLORS = {
    "SEARCHING": WHITE,
    "LOCKED": RED,
    "APPROACHING": RED,
    "REACHED": GREEN,
    "LOST": YELLOW,
}


def _corner_box(img, x0, y0, x1, y1, color, length=18, thick=2):
    """Draw only the corners of a box — looks like a targeting bracket."""
    for (cx, cy, dx, dy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thick)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thick)


def draw(frame, target, state, command, autopilot, fps=0.0):
    img = frame.copy()
    h, w = img.shape[:2]
    color = STATE_COLORS.get(state, WHITE)

    # Center crosshair
    cx, cy = w // 2, h // 2
    cv2.circle(img, (cx, cy), 22, color, 1)
    cv2.line(img, (cx - 32, cy), (cx - 12, cy), color, 1)
    cv2.line(img, (cx + 12, cy), (cx + 32, cy), color, 1)
    cv2.line(img, (cx, cy - 32), (cx, cy - 12), color, 1)
    cv2.line(img, (cx, cy + 12), (cx, cy + 32), color, 1)

    # Target bracket
    if target["found"]:
        tx, ty = int(target["x"] * w), int(target["y"] * h)
        bw, bh = int(target["w"] * w), int(target["h"] * h)
        pulse = 4 if state in ("LOCKED", "APPROACHING") and int(time.time() * 4) % 2 else 0
        _corner_box(img, tx - bw // 2 - pulse, ty - bh // 2 - pulse,
                    tx + bw // 2 + pulse, ty + bh // 2 + pulse, color)
        cv2.line(img, (cx, cy), (tx, ty), color, 1)

    # Status text
    label = {"LOCKED": "TARGET LOCKED", "APPROACHING": "LOCKED - CLOSING IN",
             "REACHED": "TARGET REACHED", "LOST": "TARGET LOST",
             "SEARCHING": "SEARCHING..."}.get(state, state)
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Autopilot status + controls
    ap = "AUTOPILOT: ON" if autopilot else "AUTOPILOT: OFF"
    cv2.putText(img, ap, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                RED if autopilot else WHITE, 2)
    cv2.putText(img, "click=pick target  A=autopilot  SPACE=LAND  Q=quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    # Command readout
    cmd = f"yaw {command['yaw']:+.2f}  up {command['up']:+.2f}  fwd {command['forward']:+.2f}"
    cv2.putText(img, cmd, (w - 360, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(img, f"{fps:.0f} fps", (w - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    if state == "REACHED":
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), GREEN, 6)

    return img
