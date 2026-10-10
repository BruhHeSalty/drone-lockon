"""OWNER: Person 2 — find the target in a frame.

Starts with color tracking (fast, reliable, no AI model needed).
Stretch goal: swap in YOLO to lock onto object types (person, bottle, etc).
Must return a Target dict (see INTERFACES.md).
"""
import cv2
import numpy as np

import config

NOT_FOUND = {"found": False, "x": 0.5, "y": 0.5, "w": 0.0, "h": 0.0, "conf": 0.0}


class ColorTracker:
    def __init__(self, hsv=config.TARGET_HSV, tol=config.HSV_TOLERANCE):
        self.hsv = np.array(hsv, dtype=int)
        self.tol = np.array(tol, dtype=int)

    def set_color_from_pixel(self, frame, px, py, radius=5):
        """Pick the target color by clicking on it."""
        h, w = frame.shape[:2]
        x0, x1 = max(px - radius, 0), min(px + radius, w)
        y0, y1 = max(py - radius, 0), min(py + radius, h)
        patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        self.hsv = np.median(patch.reshape(-1, 3), axis=0).astype(int)
        print(f"[vision] target color set to HSV {tuple(self.hsv)}")

    def _mask(self, hsv_frame):
        lo = self.hsv - self.tol
        hi = self.hsv + self.tol
        sv_lo = np.clip(lo[1:], 0, 255)
        sv_hi = np.clip(hi[1:], 0, 255)
        # Hue wraps around at 180 (red sits on both ends)
        if lo[0] < 0 or hi[0] > 179:
            m1 = cv2.inRange(hsv_frame, np.array([0, *sv_lo]), np.array([hi[0] % 180, *sv_hi]))
            m2 = cv2.inRange(hsv_frame, np.array([lo[0] % 180, *sv_lo]), np.array([179, *sv_hi]))
            mask = cv2.bitwise_or(m1, m2)
        else:
            mask = cv2.inRange(hsv_frame, np.array([lo[0], *sv_lo]), np.array([hi[0], *sv_hi]))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def find(self, frame):
        """Return a Target dict for the biggest blob of the target color."""
        fh, fw = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._mask(hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return dict(NOT_FOUND)

        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest) / (fw * fh)
        if area < config.MIN_TARGET_AREA:
            return dict(NOT_FOUND)

        x, y, w, h = cv2.boundingRect(biggest)
        fill = cv2.contourArea(biggest) / max(w * h, 1)  # solid blobs = more confident
        return {
            "found": True,
            "x": (x + w / 2) / fw,
            "y": (y + h / 2) / fh,
            "w": w / fw,
            "h": h / fh,
            "conf": float(min(1.0, fill)),
        }
