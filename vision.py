"""OWNER: Person 2 — find the target in a frame.

Color tracking for the drone; optional local YOLO person tracking for dry runs.
Run `python vision.py --person` for the Mac webcam person-tracking demo.
Must return a Target dict (see INTERFACES.md).
"""
import time

import cv2
import numpy as np

import config

NOT_FOUND = {"found": False, "x": 0.5, "y": 0.5, "w": 0.0, "h": 0.0, "conf": 0.0}


class ColorTracker:
    CONFIRM_FRAMES = 3
    LOST_TIMEOUT_S = 1.0
    MAX_CENTER_STEP = 0.08  # fraction of frame width/height per frame
    MAX_SIZE_RATIO = 1.5
    AMBIGUITY_MARGIN = 0.15

    def __init__(self, hsv=config.TARGET_HSV, tol=config.HSV_TOLERANCE):
        self.hsv = np.array(hsv, dtype=int)
        self.tol = np.array(tol, dtype=int)
        self._selected = False
        self._previous = None
        self._consistent = 0
        self._last_seen = 0.0

    def set_color_from_pixel(self, frame, px, py, radius=5):
        """Select the colored object under the cursor; confirm it in find()."""
        h, w = frame.shape[:2]
        if not (0 <= px < w and 0 <= py < h) or radius < 1:
            return
        self._selected = True
        self._previous = None
        self._consistent = 0
        x0, x1 = max(px - radius, 0), min(px + radius, w)
        y0, y1 = max(py - radius, 0), min(py + radius, h)
        patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        self.hsv = np.median(patch.reshape(-1, 3), axis=0).astype(int)
        candidates, mask = self._candidates(frame)
        for contour, target in candidates:
            if mask[py, px] and cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0:
                self._previous = target
                self._last_seen = time.monotonic()
                break
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

    def _candidates(self, frame):
        """Return sufficiently large colored blobs and their normalized boxes."""
        fh, fw = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._mask(hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area / (fw * fh) < config.MIN_TARGET_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((contour, {
                "found": True,
                "x": (x + w / 2) / fw,
                "y": (y + h / 2) / fh,
                "w": w / fw,
                "h": h / fh,
                "conf": float(min(1.0, area / max(w * h, 1))),
            }))
        return candidates, mask

    def _match_score(self, target):
        previous = self._previous
        # Limit displacement in both frame units and object-size units.
        dx = abs(target["x"] - previous["x"]) / min(self.MAX_CENTER_STEP, previous["w"])
        dy = abs(target["y"] - previous["y"]) / min(self.MAX_CENTER_STEP, previous["h"])
        distance = (dx * dx + dy * dy) ** 0.5
        ratios = [max(target[k] / previous[k], previous[k] / target[k]) for k in ("w", "h")]
        if distance > 1.0 or max(ratios) > self.MAX_SIZE_RATIO:
            return None
        overlap_w = max(0.0, min(target["x"] + target["w"] / 2, previous["x"] + previous["w"] / 2)
                        - max(target["x"] - target["w"] / 2, previous["x"] - previous["w"] / 2))
        overlap_h = max(0.0, min(target["y"] + target["h"] / 2, previous["y"] + previous["h"] / 2)
                        - max(target["y"] - target["h"] / 2, previous["y"] - previous["h"] / 2))
        intersection = overlap_w * overlap_h
        union = target["w"] * target["h"] + previous["w"] * previous["h"] - intersection
        return distance + sum(r - 1 for r in ratios) + (1 - intersection / union)

    def find(self, frame):
        """Before selection, detect color as before. After clicking, keep that lock.

        Uncertain/expired locks return NOT_FOUND so the controller hovers.
        A lock that expires never falls back to selecting the biggest blob.
        """
        now = time.monotonic()
        if self._selected:
            if self._previous is None or now - self._last_seen >= self.LOST_TIMEOUT_S:
                self._previous = None
                self._consistent = 0
                return dict(NOT_FOUND)

        candidates, _ = self._candidates(frame)
        if not self._selected:
            if not candidates:
                return dict(NOT_FOUND)
            return max(candidates, key=lambda item: cv2.contourArea(item[0]))[1]

        matches = []
        for _, target in candidates:
            score = self._match_score(target)
            if score is not None:
                matches.append((score, target))
        matches.sort(key=lambda item: item[0])
        if not matches or (len(matches) > 1 and matches[1][0] - matches[0][0] < self.AMBIGUITY_MARGIN):
            self._consistent = 0
            return dict(NOT_FOUND)

        self._previous = matches[0][1]
        self._last_seen = now
        self._consistent += 1
        if self._consistent < self.CONFIRM_FRAMES:
            return dict(NOT_FOUND)
        return dict(self._previous)


class PersonTracker:
    """YOLO + ByteTrack person IDs, restricted to dry-run control.

    Remember the selected person's clothing appearance even after their track
    ID disappears. Reacquire only an unambiguous appearance match confirmed
    across several frames. This is an appearance heuristic, not identity proof.
    """

    CONFIRM_FRAMES = 3
    MAX_APPEARANCE_DISTANCE = 0.30
    APPEARANCE_MARGIN = 0.08

    def __init__(self):
        self._check_dry_run()
        # Keep optional dependencies/model caches out of the color-only path.
        import os
        import sys
        from pathlib import Path

        cache = Path(sys.prefix) / "share" / "drone-lockon"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        try:
            from ultralytics import YOLO
            import lap  # noqa: F401 -- ByteTrack dependency; fail before opening camera
        except ImportError as exc:
            raise RuntimeError(
                "Person mode needs local dependencies: python -m pip install ultralytics lap"
            ) from exc
        self.model = YOLO(str(cache / "yolo11n.pt"))
        self._tracks = {}
        self._selected_id = None
        self._require_click = False
        self._consistent = 0
        self._pending_id = None
        self._reference = None
        print("[vision] PERSON mode (dry run). Stand in view; click to select/reselect a person.")

    @staticmethod
    def _check_dry_run():
        if config.CONTROL_PORT is not None:
            raise RuntimeError("Person tracking requires CONTROL_PORT = None; never fly at people.")

    def set_color_from_pixel(self, frame, px, py, radius=5):
        """Keep the existing click interface, but select a person ID, not color."""
        self._check_dry_run()
        h, w = frame.shape[:2]
        if not (0 <= px < w and 0 <= py < h):
            return
        x, y = px / w, py / h
        matches = [track_id for track_id, target in self._tracks.items()
                   if abs(x - target["x"]) <= target["w"] / 2
                   and abs(y - target["y"]) <= target["h"] / 2]
        if len(matches) != 1:
            print("[vision] Click inside one visible person; keeping the previous selection.")
            return
        self._selected_id = matches[0]
        self._reference = self._appearance(frame, self._tracks[self._selected_id])
        self._require_click = True
        self._consistent = 0
        self._pending_id = None
        print(f"[vision] selected person track {self._selected_id}; appearance remembered")

    @staticmethod
    def _appearance(frame, target):
        """Coarse clothing-color signature, independent of crop resolution.

        Trim box edges to reduce background; retain upper/lower color layout.
        Low-saturation pixels use brightness bins instead of unstable hue.
        """
        height, width = frame.shape[:2]
        x1 = max(0, int((target["x"] - target["w"] * 0.35) * width))
        x2 = min(width, int((target["x"] + target["w"] * 0.35) * width))
        y1 = max(0, int((target["y"] - target["h"] * 0.40) * height))
        y2 = min(height, int((target["y"] + target["h"] * 0.40) * height))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = cv2.resize(frame[y1:y2, x1:x2], (48, 96))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        features = []
        for region, weight in ((hsv, 0.5), (hsv[:48], 0.25), (hsv[48:], 0.25)):
            colored = ((region[:, :, 1] >= 40) & (region[:, :, 2] >= 30)).astype(np.uint8)
            color_hist = cv2.calcHist([region], [0, 1], colored, [18, 4], [0, 180, 0, 256])
            gray_hist = cv2.calcHist([region], [2], 1 - colored, [8], [0, 256])
            hist = np.concatenate((color_hist.ravel(), gray_hist.ravel()))
            features.append(np.sqrt(weight * hist / max(float(hist.sum()), 1.0)))
        return np.concatenate(features)

    def _choose_match(self, frame):
        if self._reference is None:
            return None
        matches = []
        for track_id, target in self._tracks.items():
            appearance = self._appearance(frame, target)
            if appearance is not None:
                distance = float(np.linalg.norm(appearance - self._reference) / np.sqrt(2))
                matches.append((distance, track_id))
        matches.sort()
        if not matches or matches[0][0] > self.MAX_APPEARANCE_DISTANCE:
            return None
        if len(matches) > 1 and matches[1][0] - matches[0][0] < self.APPEARANCE_MARGIN:
            return None
        return matches[0][1]

    def find(self, frame):
        self._check_dry_run()
        # Use a persistent tracker to associate people across frames and scale changes.
        result = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0],
            conf=0.35, imgsz=640, device="cpu", verbose=False,
        )[0]
        self._tracks = {}
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            height, width = frame.shape[:2]
            for coords, track_id, confidence in zip(
                boxes.xyxy.cpu().tolist(), boxes.id.cpu().tolist(), boxes.conf.cpu().tolist()
            ):
                x1, y1, x2, y2 = coords
                x1, x2 = np.clip([x1, x2], 0, width)
                y1, y2 = np.clip([y1, y2], 0, height)
                if x2 <= x1 or y2 <= y1 or confidence < 0.45:
                    continue
                self._tracks[int(track_id)] = {
                    "found": True, "x": float((x1 + x2) / (2 * width)),
                    "y": float((y1 + y2) / (2 * height)),
                    "w": float((x2 - x1) / width), "h": float((y2 - y1) / height),
                    "conf": float(confidence),
                }

        if self._reference is None and not self._require_click and len(self._tracks) == 1:
            self._selected_id = next(iter(self._tracks))
            self._reference = self._appearance(frame, self._tracks[self._selected_id])

        match_id = self._choose_match(frame)
        if match_id is None:
            self._consistent = 0
            self._pending_id = None
            return dict(NOT_FOUND)
        if match_id != self._pending_id:
            self._pending_id = match_id
            self._consistent = 0
        self._consistent += 1
        if self._consistent < self.CONFIRM_FRAMES:
            return dict(NOT_FOUND)
        if match_id != self._selected_id:
            print(f"[vision] Reacquired selected appearance as person track {match_id}")
        self._selected_id = match_id
        return dict(self._tracks[match_id])


def _person_webcam_demo():
    """Reuse the existing HUD/keys with a webcam and a dry-run DroneLink."""
    import argparse

    parser = argparse.ArgumentParser(description="Local webcam person tracking; no drone commands.")
    parser.add_argument("--person", action="store_true", required=True)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    args = parser.parse_args()
    # Set these before importing main: VideoSource captures defaults at import time.
    config.VIDEO_SOURCE = "webcam"
    config.WEBCAM_INDEX = args.camera
    config.CONTROL_PORT = None
    tracker = PersonTracker()  # Load model before main opens the camera.
    import main

    main.ColorTracker = lambda: tracker
    main.main()


if __name__ == "__main__":
    _person_webcam_demo()
