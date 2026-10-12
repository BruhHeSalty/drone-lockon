"""OWNER: Person 3 — readable flight status and steady GTA-style lock-on.

Only renders the supplied state; never changes targets or flight commands.
The area below the header stays free for main.py's mode/video warnings.
"""
import math
import time

import cv2
import config

# HUD means the words and marks drawn on top of the camera picture.
# OpenCV colors are three numbers in blue, green, red order (0 to 255 each).
WHITE = (236, 242, 247)
RED = (95, 100, 255)
GREEN = (155, 231, 102)
YELLOW = (95, 211, 255)
MUTED = (166, 179, 190)
PANEL = (22, 19, 16)
CYAN = (240, 211, 95)

STATE_COLORS = {
    "SEARCHING": WHITE,
    "LOCKED": RED,
    "APPROACHING": RED,
    "REACHED": GREEN,
    "LOST": YELLOW,
}
STATE_LABELS = {
    "SEARCHING": ("SEARCHING", "Awaiting target"),
    "LOCKED": ("TARGET LOCKED", "Target in view"),
    "APPROACHING": ("TRACKING TARGET", "Approach requested"),
    "REACHED": ("TARGET REACHED", "Stop threshold reached"),
    "LOST": ("TARGET LOST", "Target unavailable"),
}


class _LockAnimation:
    """Presentation-only memory. Never returned to the tracker or controller."""
    # Make the aiming marks slide smoothly and fade instead of blinking away.
    # These pretty marks are only for the screen; they never decide movement.
    HOLD = 0.40
    FADE = 0.25
    ACQUIRE = 0.85

    def __init__(self):
        self.target = None
        self.shape = None
        self.last_seen = self.last_draw = None
        self.progress = 0.0
        self.was_found = False
        self.color = RED

    def update(self, target, state, shape, now):
        # Remember where we drew last time, then move a little toward the new spot.
        expired = self.last_seen is None or now - self.last_seen > self.HOLD + self.FADE
        dt = max(0.0, now - self.last_draw) if self.last_draw is not None else 0.0
        if self.shape != shape or expired:
            self.target = None
        self.shape, self.last_draw = shape, now
        if target['found']:
            jumped = self.target is not None and math.hypot(
                target['x'] - self.target['x'], target['y'] - self.target['y']) > 0.3
            if self.target is None or jumped:
                self.target = dict(target)
                self.progress = 0.0
            else:
                # Time-based easing behaves the same at different camera rates.
                alpha = 1.0 - math.exp(-dt / 0.075)
                for key in ('x', 'y', 'w', 'h', 'conf'):
                    self.target[key] += alpha * (target[key] - self.target[key])
                if self.was_found:
                    self.progress = min(1.0, self.progress + min(dt, 0.15) / self.ACQUIRE)
            self.last_seen = now
            self.color = GREEN if state == 'REACHED' else RED
        self.was_found = bool(target['found'])
        age = now - self.last_seen if self.last_seen is not None else float('inf')
        opacity = max(0.0, min(1.0, 1.0 - (age - self.HOLD) / self.FADE))
        return self.target, opacity


_lock = _LockAnimation()


def reset_lock():
    """Clear visual memory on a new session or explicit target selection."""
    global _lock
    _lock = _LockAnimation()


def _text(img, text, x, y, scale=0.45, color=WHITE, thick=1, max_width=None):
    """Fit text to its own column instead of colliding with adjacent data."""
    if max_width is not None:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
        scale *= min(1.0, max(1, max_width) / max(1, width))
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _panel(img, x0, y0, x1, y1):
    # Put a dark patch behind the words so bright video does not hide them.
    region = img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if region.size:
        shade = region.copy()
        shade[:] = PANEL
        cv2.addWeighted(shade, 0.86, region, 0.14, 0, dst=region)


def _line(img, start, end, color, thick=1):
    # A dark outline keeps fine aiming marks visible against bright video.
    cv2.line(img, start, end, PANEL, thick + 2, cv2.LINE_AA)
    cv2.line(img, start, end, color, thick, cv2.LINE_AA)


def draw(frame, target, state, command, autopilot, fps=0.0):
    """Return an annotated copy at the original resolution (and click coordinates)."""
    # Draw on a copy so the finder still gets a clean camera picture.
    img = frame.copy()
    h, w = img.shape[:2]
    color = STATE_COLORS.get(state, WHITE)
    # Fixed-height chrome aligns with the overlays and manual footer in main.py.
    unit = min(1.0, w / 640, h / 300)
    pad = max(4, int(14 * unit))
    header = max(1, int(56 * unit))
    footer = max(1, int(88 * unit))
    bottom = h - footer
    cx, cy = w // 2, h // 2
    radius = max(4, int(12 * unit))
    outer = radius + max(4, int(9 * unit))

    # A neutral, open reticle preserves visibility of the object at frame center.
    for start, end in (((cx - outer, cy), (cx - radius, cy)),
                       ((cx + radius, cy), (cx + outer, cy)),
                       ((cx, cy - outer), (cx, cy - radius)),
                       ((cx, cy + radius), (cx, cy + outer))):
        _line(img, start, end, WHITE)
    cv2.circle(img, (cx, cy), 2, PANEL, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 1, WHITE, -1, cv2.LINE_AA)

    visual_target, opacity = _lock.update(target, state, (h, w), time.monotonic())
    if visual_target is not None and opacity > 0:
        before_reticle = img.copy() if opacity < 1 else None
        tx, ty = int(visual_target['x'] * w), int(visual_target['y'] * h)
        # Four square corners travel inward and close into a solid lock square.
        # This is a reticle around the center, not a detector bounding rectangle.
        half = max(18 * unit, min(visual_target['w'] * w, visual_target['h'] * h) * 0.32)
        half = min(half, min(w, h) * 0.16)
        ease = (1.0 - _lock.progress) ** 3
        spread = half + min(w, h) * 0.20 * ease
        x0, x1 = round(tx - spread), round(tx + spread)
        y0, y1 = round(ty - spread), round(ty + spread)
        reticle_color = tuple(round(a + (b - a) * _lock.progress)
                              for a, b in zip(YELLOW, _lock.color))
        arm = round(half * (0.45 + 0.55 * _lock.progress))
        # Unlike detector brackets, these arms meet as acquisition completes.
        for x, y, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                              (x0, y1, 1, -1), (x1, y1, -1, -1)):
            _line(img, (x, y), (x + dx * arm, y), reticle_color, 2)
            _line(img, (x, y), (x, y + dy * arm), reticle_color, 2)
        conf = max(0.0, min(1.0, visual_target['conf']))
        label = ('LAST KNOWN' if not target['found'] else
                 f"LOCK {conf:.0%}" if _lock.progress >= 1 else 'ACQUIRING')
        label_scale = 0.4 * unit
        label_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)[0][0] + 12
        label_h = max(12, int(24 * unit))
        lx = max(0, min(x0, w - label_w))
        # Keep clear of mode warnings (y=70..130) and the command panel.
        ly = max(140, min(y1 + 8, bottom - label_h - 5))
        if ly + label_h < bottom - 4:
            _panel(img, lx, ly, lx + label_w, ly + label_h)
            _text(img, label, lx + 6, ly + label_h - 7, label_scale, reticle_color)
        if before_reticle is not None:
            cv2.addWeighted(img, opacity, before_reticle, 1 - opacity, 0, dst=img)

    _panel(img, 0, 0, w, header)
    _panel(img, 0, bottom, w, h)
    cv2.line(img, (0, header - 1), (w - 1, header - 1), MUTED, 1)
    cv2.line(img, (0, bottom), (w - 1, bottom), MUTED, 1)

    split = int(w * 0.61)
    title, detail = STATE_LABELS.get(state, (state, "Tracking status"))
    if visual_target is not None and opacity > 0:
        title = 'TARGET REACHED' if state == 'REACHED' else 'TARGET TRACKING'
        color = _lock.color
        if not target['found']:
            detail = 'No detection / last known position'
    if state == "APPROACHING" and not autopilot:
        detail = "Preview / autopilot off"
    cv2.rectangle(img, (pad, int(13 * unit)),
                  (pad + max(2, int(3 * unit)), int(42 * unit)), color, -1)
    _text(img, title, pad + 12 * unit, 25 * unit, 0.62 * unit, color, 1,
          split - pad - 24 * unit)
    _text(img, detail, pad + 12 * unit, 44 * unit, 0.36 * unit, MUTED, 1,
          split - pad - 24 * unit)
    # Leave the upper-right corner free for main.py's recording indicator.
    ap_width = w - split - max(36, pad)
    _text(img, "AUTOPILOT ON" if autopilot else "AUTOPILOT OFF", split, 24 * unit,
          0.49 * unit, YELLOW if autopilot else WHITE, 1, ap_width)
    _text(img, "Automatic control on" if autopilot else "Automatic control off", split, 43 * unit,
          0.35 * unit, MUTED, 1, ap_width)

    # Signed, zero-centered command meters, including strafe.
    # These bars show requested movement, not measured speed. Middle means zero.
    cell_w = (w - 2 * pad) / 4
    for index, (key, label) in enumerate((("yaw", "YAW"), ("up", "UP"),
                                         ("forward", "FWD"), ("right", "STRAFE"))):
        left = int(pad + index * cell_w)
        right = int(pad + (index + 1) * cell_w - 12 * unit)
        value = max(-1.0, min(1.0, command[key]))
        _text(img, f"{label} {value:+.2f}", left, bottom + 21 * unit,
              0.4 * unit, WHITE, 1, right - left)
        middle = (left + right) // 2
        meter_y = int(bottom + 32 * unit)
        cv2.line(img, (left, meter_y), (right, meter_y), MUTED, 1)
        endpoint = middle + int(value * (right - left) / 2)
        if endpoint != middle:
            cv2.line(img, (middle, meter_y), (endpoint, meter_y), CYAN, 3, cv2.LINE_AA)
        cv2.line(img, (middle, meter_y - 3), (middle, meter_y + 3), WHITE, 1)

    land_w = int(126 * unit)
    land_top = int(bottom + 48 * unit)
    cv2.rectangle(img, (pad, land_top), (pad + land_w, h - max(5, int(10 * unit))), RED, -1)
    _text(img, "SPACE  LAND", pad + 8 * unit, h - 20 * unit, 0.45 * unit, PANEL,
          max_width=land_w - 16 * unit)
    hint_x = pad + land_w + 14 * unit
    fps_width = int(72 * unit)
    quit_hint = ('Q-close CUT MOTORS' if config.CONTROL_PORT is not None else 'Q quit preview')
    _text(img, f"CLICK pick / A auto / {quit_hint}", hint_x, h - 21 * unit,
          0.38 * unit, WHITE, max_width=w - pad - fps_width - hint_x - 10 * unit)
    # FPS means frames per second: how many pictures the loop handles each second.
    _text(img, f"{max(0.0, fps):.0f} FPS", w - pad - fps_width, h - 21 * unit,
          0.4 * unit, MUTED, max_width=fps_width)
    return img
