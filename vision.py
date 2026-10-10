"""OWNER: Person 2 — find the target in a frame.

Color tracking and YOLO person tracking, including explicit live yaw-only use.
Run `python vision.py --person` for the Mac webcam person-tracking demo.
Must return a Target dict (see INTERFACES.md).
"""
import time
import threading

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
    """YOLO person IDs with camera-motion compensation for yaw-only control.

    Remember the selected person's clothing appearance even after their track
    ID disappears. Reacquire only an unambiguous appearance match confirmed
    across several frames in demos. Live mode never changes the selected ID;
    weak detections briefly retain it while hovering, and ID loss requires a click.
    This is an appearance heuristic, not identity proof.
    """

    CONFIRM_FRAMES = 3
    MAX_APPEARANCE_DISTANCE = 0.30
    APPEARANCE_MARGIN = 0.08
    WEAK_TRACK_TIMEOUT = 0.5

    def __init__(self, allow_live_yaw=False, model="yolo11n.pt", device="auto",
                 imgsz=640, tracker="botsort.yaml"):
        if imgsz < 320 or imgsz > 1280 or imgsz % 32:
            raise ValueError("Person image size must be a multiple of 32 between 320 and 1280")
        if tracker not in ("botsort.yaml", "bytetrack.yaml"):
            raise ValueError("Choose botsort.yaml or bytetrack.yaml")
        self.allow_live_yaw = allow_live_yaw
        self._check_control_mode()
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
            import torch
            import lap  # noqa: F401 -- ByteTrack dependency; fail before opening camera
        except ImportError as exc:
            raise RuntimeError(
                "Person mode needs local dependencies: python -m pip install ultralytics lap"
            ) from exc
        model_path = Path(model).expanduser()
        if not model_path.is_file():
            if model not in ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt"):
                raise FileNotFoundError(f"Person model not found: {model}")
            model_path = cache / model
        self.model = YOLO(str(model_path))
        person_classes = [key for key, name in self.model.names.items() if name == "person"]
        if len(person_classes) != 1:
            raise ValueError("Person model must have exactly one class named 'person'")
        self.person_class = person_classes[0]
        self.device = ("0" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
        self.imgsz = imgsz
        self.tracker_config = tracker
        self._tracks = {}
        self._selected_id = None
        self._require_click = allow_live_yaw
        self._consistent = 0
        self._pending_id = None
        self._reference = None
        self._previous_live = None
        self._last_confirmed = 0.0
        self.status = "Click a person to select"
        # First inference and accelerator initialization happen before flight connects.
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), imgsz=imgsz,
                           device=self.device, classes=[self.person_class], verbose=False)
        print(f"[vision] {model_path.name}, {tracker}, {self.device}, size {imgsz}. Click a person.")

    def _check_control_mode(self):
        if config.CONTROL_PORT is not None and not self.allow_live_yaw:
            raise RuntimeError("Live person tracking requires the explicit yaw-only mode.")

    def set_color_from_pixel(self, frame, px, py, radius=5):
        """Keep the existing click interface, but select a person ID, not color."""
        self._check_control_mode()
        h, w = frame.shape[:2]
        if not (0 <= px < w and 0 <= py < h):
            return
        x, y = px / w, py / h
        matches = [track_id for track_id, target in self._tracks.items()
                   if target["conf"] >= 0.6
                   and abs(x - target["x"]) <= target["w"] / 2
                   and abs(y - target["y"]) <= target["h"] / 2]
        if len(matches) != 1:
            print("[vision] Click inside one visible person; keeping the previous selection.")
            return
        track_id = matches[0]
        self.select_track(track_id, self._tracks[track_id], self._appearance(frame, self._tracks[track_id]))

    def select_track(self, track_id, target, appearance):
        """Select from a displayed detection snapshot, without replaying old frames."""
        self._selected_id = track_id
        self._reference = appearance
        self._require_click = True
        self._consistent = 0
        self._pending_id = None
        self._previous_live = dict(target)
        self._last_confirmed = time.monotonic()
        self.status = "Confirming selected person"
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
        if self.allow_live_yaw:
            candidate = self._tracks.get(self._selected_id)
            if candidate is None:
                self._failure_reason = "Selected person missing - click to select"
                return None
            appearance = self._appearance(frame, candidate)
            if appearance is None:
                return None
            distance = float(np.linalg.norm(appearance - self._reference) / np.sqrt(2))
            if distance > self.MAX_APPEARANCE_DISTANCE:
                self._failure_reason = "Appearance uncertain - click to select"
                return None
            # Similar clothes elsewhere in the picture do not invalidate this ID.
            # Overlapping people do: we cannot confidently attribute the pixels.
            for track_id, other in self._tracks.items():
                if track_id == self._selected_id:
                    continue
                overlap_x = max(0., min(candidate["x"] + candidate["w"] / 2, other["x"] + other["w"] / 2)
                                - max(candidate["x"] - candidate["w"] / 2, other["x"] - other["w"] / 2))
                overlap_y = max(0., min(candidate["y"] + candidate["h"] / 2, other["y"] + other["h"] / 2)
                                - max(candidate["y"] - candidate["h"] / 2, other["y"] - other["h"] / 2))
                if overlap_x * overlap_y / min(candidate["w"] * candidate["h"], other["w"] * other["h"]) > .2:
                    self._failure_reason = "People overlap - select when clear"
                    return None
            return self._selected_id
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
        self._check_control_mode()
        self._failure_reason = "Lock lost - click to select"
        # Use a persistent tracker to associate people across frames and scale changes.
        result = self.model.track(
            frame, persist=True, tracker=self.tracker_config, classes=[self.person_class],
            # Weak detections help association; flight still requires >= 0.6.
            conf=0.1, imgsz=self.imgsz, device=self.device, verbose=False,
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
                if not np.isfinite([x1, y1, x2, y2, confidence]).all():
                    continue
                if x2 <= x1 or y2 <= y1 or confidence < 0.1:
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

        if self.allow_live_yaw and self._selected_id is not None:
            current = self._tracks.get(self._selected_id)
            if time.monotonic() - self._last_confirmed > self.WEAK_TRACK_TIMEOUT:
                self._reference = None
                self._failure_reason = "Lock expired - click to select"
            elif current is not None and current["conf"] < 0.6:
                # Preserve only the same ID briefly, but never steer on a weak box.
                self._consistent = 0
                self._pending_id = None
                self.status = "Low confidence - holding"
                return dict(NOT_FOUND)

        match_id = self._choose_match(frame)
        if self.allow_live_yaw:
            # Never transfer a live lock to a different person after occlusion.
            candidate = self._tracks.get(match_id)
            previous = self._previous_live
            jumped = candidate is not None and previous is not None and (
                abs(candidate["x"] - previous["x"]) > 0.10
                or abs(candidate["y"] - previous["y"]) > 0.10
                or any(max(candidate[k], previous[k]) / min(candidate[k], previous[k]) > 1.5
                       for k in ("w", "h"))
            )
            if match_id != self._selected_id or candidate is None or candidate["conf"] < 0.6 or jumped:
                if jumped:
                    self._failure_reason = "Position changed too far - click to select"
                self._reference = None
                self._selected_id = None
                self._previous_live = None
                match_id = None
            else:
                self._previous_live = dict(candidate)
                self._last_confirmed = time.monotonic()
        if match_id is None:
            self.status = self._failure_reason if self._require_click else "Searching"
            self._consistent = 0
            self._pending_id = None
            return dict(NOT_FOUND)
        if match_id != self._pending_id:
            self._pending_id = match_id
            self._consistent = 0
        self._consistent += 1
        if self._consistent < self.CONFIRM_FRAMES:
            self.status = "Confirming selected person"
            return dict(NOT_FOUND)
        if match_id != self._selected_id:
            print(f"[vision] Reacquired selected appearance as person track {match_id}")
        self._selected_id = match_id
        self.status = "Selected person confirmed"
        return dict(self._tracks[match_id])


class AsyncPersonTracker:
    """One latest-frame worker; stale detections never become flight commands."""

    MAX_AGE = 0.3

    def __init__(self, tracker):
        self.tracker = tracker
        self._condition = threading.Condition()
        self._pending = None
        self._click = None
        self._generation = 0
        self._result = (0.0, dict(NOT_FOUND))
        self._snapshot = (None, {}, "Waiting for detection")
        self._last_submit = float("-inf")
        self._inference_ms = 0.0
        self._closed = False
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_color_from_pixel(self, frame, x, y):
        with self._condition:
            stamp, _ = self._result
            detected_frame, tracks, _ = self._snapshot
            if detected_frame is None or not 0 <= time.monotonic() - stamp < self.MAX_AGE:
                return
            h, w = frame.shape[:2]
            matches = [(track_id, t) for track_id, t in tracks.items()
                       if t["conf"] >= .6 and abs(x / w - t["x"]) <= t["w"] / 2
                       and abs(y / h - t["y"]) <= t["h"] / 2]
            if len(matches) != 1:
                return
            track_id, target = matches[0]
            appearance = self.tracker._appearance(detected_frame, target)
            self._click = (track_id, dict(target), appearance)
            self._generation += 1
            self._result = (0.0, dict(NOT_FOUND))
            if self._pending is not None:
                stamp, pending_frame, _ = self._pending
                self._pending = (stamp, pending_frame, self._generation)

    def find(self, frame, frame_time=None):
        now = time.monotonic()
        stamp = now if frame_time is None else frame_time
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Person detection failed") from self._error
            if not 0 <= now - stamp < self.MAX_AGE:
                return dict(NOT_FOUND)
            if stamp > self._last_submit:
                self._last_submit = stamp
                self._pending = (stamp, frame.copy(), self._generation)
                self._condition.notify()
            stamp, target = self._result
            return dict(target if 0 <= now - stamp < self.MAX_AGE else NOT_FOUND)

    def draw_candidates(self, view):
        """Show selectable detections and measured delay without changing Target."""
        with self._condition:
            stamp, _ = self._result
            _, tracks, status = self._snapshot
            inference_ms = self._inference_ms
        age = time.monotonic() - stamp
        fresh = 0 <= age < self.MAX_AGE
        if fresh:
            h, w = view.shape[:2]
            for track_id, target in tracks.items():
                x1, y1 = int((target["x"] - target["w"] / 2) * w), int((target["y"] - target["h"] / 2) * h)
                x2, y2 = int((target["x"] + target["w"] / 2) * w), int((target["y"] + target["h"] / 2) * h)
                color = (0, 220, 220) if target["conf"] >= .6 else (130, 130, 130)
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 1)
                cv2.putText(view, f"person {track_id} {target['conf']:.0%}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        else:
            status = "Detection delayed - holding"
        cv2.putText(view, f"{status} | inference {inference_ms:.0f} ms", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 220), 1)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                stamp, frame, generation = self._pending
                click, self._click = self._click, None
                self._pending = None
            try:
                started = time.monotonic()
                if click is not None:
                    self.tracker.select_track(*click)
                target = self.tracker.find(frame)
                with self._condition:
                    if generation == self._generation:
                        self._result = (stamp, target)
                        self._snapshot = (frame, dict(self.tracker._tracks), self.tracker.status)
                        self._inference_ms = (time.monotonic() - started) * 1000
            except Exception as exc:
                with self._condition:
                    self._error = exc
                    self._result = (0.0, dict(NOT_FOUND))
                return

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=0.5)


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
