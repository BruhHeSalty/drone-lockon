"""OWNER: Person 2 — find the target in a frame.

Color tracking and YOLO person tracking for explicit person-control modes.
Run `python vision.py --person` for the Mac webcam person-tracking demo.
Must return a Target dict (see INTERFACES.md).

Drone frames get color-noise and luminance cleanup before detection; webcam frames do not.
DRONE_VISION_PREPROCESS=off disables it for comparison; =on enables it on a
recorded drone clip. This does not restore detail lost to clipping/compression.
Clicking a weak candidate remembers the selection, but steering still requires
three detections at >= 0.60 confidence and a result younger than 0.3 seconds.
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
    """YOLO person IDs with camera-motion compensation for person control.

    Remember the selected person's clothing appearance even after their track
    ID disappears. Reacquire only an unambiguous appearance match confirmed
    across several fresh frames. Selection persists until another person is clicked
    or the app closes; uncertain or missing observations always command hover.
    This is an appearance heuristic, not identity proof.
    """

    CONFIRM_FRAMES = 3
    MAX_APPEARANCE_DISTANCE = 0.30
    APPEARANCE_MARGIN = 0.08
    CONTINUITY_TIMEOUT = 0.5
    SELECT_CONF = 0.25
    CONTROL_CONF = 0.60

    def __init__(self, allow_live_yaw=False, model="yolo11n.pt", device="auto",
                 imgsz=640, tracker="botsort.yaml", preprocess="auto"):
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

        self._preprocess_mode = os.environ.get("DRONE_VISION_PREPROCESS", preprocess).lower()
        if self._preprocess_mode not in ("auto", "on", "off"):
            raise ValueError("DRONE_VISION_PREPROCESS must be auto, on or off")
        self._drone_video = config.VIDEO_SOURCE == "drone"
        self._clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

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
        self._visible = False
        self.status = "Click a person to select"
        # First inference and accelerator initialization happen before flight connects.
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), imgsz=imgsz,
                           device=self.device, classes=[self.person_class], verbose=False)
        cleanup = self._preprocess_mode == "on" or (self._preprocess_mode == "auto" and self._drone_video)
        print(f"[vision] {model_path.name}, {tracker}, {self.device}, size {imgsz}, "
              f"image cleanup {'ON' if cleanup else 'OFF'}. Click a person box; weak boxes wait for confirmation.")

    def _prepare_frame(self, frame):
        """Color denoising/contrast correction at native size, without sharpening.

        Only the detector sees this copy. Original BGR pixels and coordinates
        stay available for the HUD and appearance matching. One inference pass
        per frame keeps the existing age deadline meaningful.
        """
        mode = getattr(self, "_preprocess_mode", "off")
        if mode == "off" or (mode == "auto" and not self._drone_video):
            return frame
        # The drone clip has strong colored speckles and edge halos. Filtering
        # Y alone leaves those chroma artifacts for YOLO to interpret as edges.
        # Smooth all three channels first; keep this out of the webcam path.
        # This is spatial filtering only: no old frames or invented detail.
        denoised = cv2.GaussianBlur(frame, (5, 5), 1.2)
        ycc = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
        luminance = cv2.GaussianBlur(ycc[:, :, 0], (3, 3), 0.6)
        low, high = np.percentile(luminance[::4, ::4], (5, 95))
        if high - low < 16:
            return frame  # No usable contrast: do not amplify nearly flat noise.
        gain = min(1.6, max(1.0, 220.0 / (high - low)))
        stretched = np.clip((luminance.astype(np.float32) - low) * gain + 16, 0, 255).astype(np.uint8)
        local = self._clahe.apply(luminance)
        ycc[:, :, 0] = cv2.addWeighted(stretched, 0.7, local, 0.3, 0)
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    def _check_control_mode(self):
        if config.CONTROL_PORT is not None and not self.allow_live_yaw:
            raise RuntimeError("Live person tracking requires an explicit person-control mode.")

    def set_color_from_pixel(self, frame, px, py, radius=5):
        """Keep the existing click interface, but select a person ID, not color."""
        self._check_control_mode()
        h, w = frame.shape[:2]
        if not (0 <= px < w and 0 <= py < h):
            return
        x, y = px / w, py / h
        matches = [track_id for track_id, target in self._tracks.items()
                   if target["conf"] >= self.SELECT_CONF
                   and abs(x - target["x"]) <= target["w"] / 2
                   and abs(y - target["y"]) <= target["h"] / 2]
        if len(matches) != 1:
            print("[vision] Click inside one visible person; keeping the previous selection.")
            return
        track_id = matches[0]
        self.select_track(track_id, self._tracks[track_id], self._appearance(frame, self._tracks[track_id]))

    def select_track(self, track_id, target, appearance):
        """Select from a displayed detection snapshot, without replaying old frames."""
        if appearance is None:
            self.status = "Selection has no usable appearance - click another clear box"
            return
        self._selected_id = track_id
        self._reference = appearance
        self._require_click = True
        self._consistent = 0
        self._pending_id = track_id
        self._previous_live = dict(target)
        self._last_confirmed = time.monotonic()
        self._visible = True  # The click anchors continuity; output still needs confirmation.
        self.status = "Confirming selected person"
        print(f"[vision] selected person track {self._selected_id}; appearance remembered")

    @staticmethod
    def _appearance(frame, target):
        """Clothing color plus exposure-normalized greyscale body layout.

        Trim box edges to reduce background; retain upper/lower color layout.
        Washout reduces hue's weight continuously; relative luminance and coarse
        spatial structure stay useful without relying on saturated clothing.
        """
        height, width = frame.shape[:2]
        x1 = max(0, int((target["x"] - target["w"] * 0.35) * width))
        x2 = min(width, int((target["x"] + target["w"] * 0.35) * width))
        y1 = max(0, int((target["y"] - target["h"] * 0.40) * height))
        y2 = min(height, int((target["y"] + target["h"] * 0.40) * height))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = cv2.resize(frame[y1:y2, x1:x2], (48, 96), interpolation=cv2.INTER_AREA)
        crop = cv2.GaussianBlur(crop, (3, 3), 0.6)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        low, high = np.percentile(gray, (5, 95))
        relative = np.clip((gray - low) / max(high - low, 24.0), 0, 1)
        normalized = np.round(relative * 255).astype(np.uint8)
        features = []
        for rows, weight in ((slice(None), 0.5), (slice(0, 48), 0.25), (slice(48, None), 0.25)):
            region = hsv[rows]
            colored = (region[:, :, 1] >= 40).astype(np.uint8)
            color_hist = cv2.calcHist([region], [0, 1], colored, [18, 4], [0, 180, 0, 256])
            gray_hist = cv2.calcHist([normalized[rows]], [0], None, [8], [0, 256])
            color_weight = 0.6 * min(float(region[:, :, 1].mean()) / 80.0, 1.0)
            if not color_hist.any():
                color_weight = 0.0
            color_hist = color_weight * color_hist.ravel() / max(float(color_hist.sum()), 1.0)
            gray_hist = (1 - color_weight) * gray_hist.ravel() / max(float(gray_hist.sum()), 1.0)
            features.append(np.sqrt(0.7 * weight * np.concatenate((color_hist, gray_hist))))
        layout = cv2.resize(relative, (6, 12), interpolation=cv2.INTER_AREA).ravel()
        features.append(np.sqrt(0.3) * layout / max(float(np.linalg.norm(layout)), 1e-6))
        return np.concatenate(features)

    def _overlapping(self, track_id):
        candidate = self._tracks[track_id]
        for other_id, other in self._tracks.items():
            if other_id == track_id or other["conf"] < .25:
                continue
            overlap_x = max(0., min(candidate["x"] + candidate["w"] / 2, other["x"] + other["w"] / 2)
                            - max(candidate["x"] - candidate["w"] / 2, other["x"] - other["w"] / 2))
            overlap_y = max(0., min(candidate["y"] + candidate["h"] / 2, other["y"] + other["h"] / 2)
                            - max(candidate["y"] - candidate["h"] / 2, other["y"] - other["h"] / 2))
            if overlap_x * overlap_y / min(candidate["w"] * candidate["h"], other["w"] * other["h"]) > .2:
                return True
        return False

    def _choose_match(self, frame):
        if self._reference is None:
            return None
        current = self._tracks.get(self._selected_id)
        if current is not None and current["conf"] < self.CONTROL_CONF:
            self._failure_reason = "Target remembered - low confidence, holding"
            return None
        matches = []
        for track_id, target in self._tracks.items():
            appearance = self._appearance(frame, target)
            if appearance is not None:
                distance = float(np.linalg.norm(appearance - self._reference) / np.sqrt(2))
                matches.append((distance, track_id))
        matches.sort()
        # During uninterrupted tracking, retain the existing ID even when another
        # person elsewhere wears similar clothes. After loss, compare everyone.
        if self._visible and current is not None:
            selected = next((distance for distance, track_id in matches if track_id == self._selected_id), None)
            if selected is not None and selected <= self.MAX_APPEARANCE_DISTANCE:
                if self._overlapping(self._selected_id):
                    self._failure_reason = "Target remembered - people overlap, holding"
                    return None
                return self._selected_id
        if not matches or matches[0][0] > self.MAX_APPEARANCE_DISTANCE:
            self._failure_reason = ("Appearance differs - click a box to select it" if matches
                                    else "Target remembered - waiting for a clear match")
            return None
        if len(matches) > 1 and matches[1][0] - matches[0][0] < self.APPEARANCE_MARGIN:
            self._failure_reason = "Target remembered - similar people, holding"
            return None
        best_id = matches[0][1]
        if self._tracks[best_id]["conf"] < self.CONTROL_CONF or self._overlapping(best_id):
            self._failure_reason = "Target remembered - uncertain match, holding"
            return None
        return best_id

    def find(self, frame):
        self._check_control_mode()
        self._failure_reason = "Target remembered - out of view, holding"
        # Use a persistent tracker to associate people across frames and scale changes.
        detector_frame = self._prepare_frame(frame)
        result = self.model.track(
            detector_frame, persist=True, tracker=self.tracker_config, classes=[self.person_class],
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

        if time.monotonic() - self._last_confirmed > self.CONTINUITY_TIMEOUT:
            # A gap expires motion continuity, never the user's selection.
            self._visible = False
            self._consistent = 0
            self._pending_id = None
            self._previous_live = None

        match_id = self._choose_match(frame)
        if match_id is not None:
            candidate = self._tracks.get(match_id)
            previous = self._previous_live if self._pending_id == match_id else None
            jumped = candidate is not None and previous is not None and (
                abs(candidate["x"] - previous["x"]) > 0.10
                or abs(candidate["y"] - previous["y"]) > 0.10
                or any(max(candidate[k], previous[k]) / min(candidate[k], previous[k]) > 1.5
                       for k in ("w", "h"))
            )
            if candidate is None or candidate["conf"] < self.CONTROL_CONF or jumped:
                if jumped:
                    self._failure_reason = "Target remembered - confirming new position"
                match_id = None
            else:
                self._previous_live = dict(candidate)
                self._last_confirmed = time.monotonic()
        if match_id is None:
            if self._reference is not None:
                self.status = self._failure_reason
            elif any(t["conf"] >= self.SELECT_CONF for t in self._tracks.values()):
                self.status = "Click a person box - weak boxes wait for confidence"
            else:
                self.status = "No selectable person detection - waiting"
            self._consistent = 0
            self._pending_id = None
            self._previous_live = None
            self._visible = False
            return dict(NOT_FOUND)
        if match_id != self._pending_id:
            self._pending_id = match_id
            self._consistent = 0
        self._consistent += 1
        if self._consistent < self.CONFIRM_FRAMES:
            self.status = f"Confirming selected person {self._consistent}/{self.CONFIRM_FRAMES}"
            return dict(NOT_FOUND)
        if match_id != self._selected_id:
            print(f"[vision] Reacquired selected appearance as person track {match_id}")
        self._selected_id = match_id
        self._visible = True
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
        self._displayed = None
        self._selected_id = None
        self._last_submit = float("-inf")
        self._inference_ms = 0.0
        self._click_notice = ""
        self._click_notice_until = 0.0
        self._closed = False
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_color_from_pixel(self, frame, x, y):
        """Return True only when this click accepts one fresh person box."""
        with self._condition:
            # The worker may already have moved on since draw_candidates().
            # Select the box the user actually saw, with its original timestamp.
            displayed = self._displayed
            if displayed is None:
                self._click_notice = "Click ignored: no fresh detection shown"
                self._click_notice_until = time.monotonic() + 2.0
                return
            stamp, (detected_frame, tracks, _), generation = displayed
            if (detected_frame is None or generation != self._generation
                    or not 0 <= time.monotonic() - stamp < self.MAX_AGE):
                self._click_notice = "Click ignored: no fresh detection yet"
                self._click_notice_until = time.monotonic() + 2.0
                return
            h, w = frame.shape[:2]
            matches = [(track_id, t) for track_id, t in tracks.items()
                       if t["conf"] >= PersonTracker.SELECT_CONF and abs(x / w - t["x"]) <= t["w"] / 2
                       and abs(y / h - t["y"]) <= t["h"] / 2]
            if len(matches) != 1:
                self._click_notice = "Click inside one person box" if not matches else "Click ambiguous: boxes overlap"
                self._click_notice_until = time.monotonic() + 2.0
                return
            track_id, target = matches[0]
            appearance = self.tracker._appearance(detected_frame, target)
            if appearance is None:
                self._click_notice = "Click ignored: person crop is empty"
                self._click_notice_until = time.monotonic() + 2.0
                return
            self._click_notice = f"Selected person {track_id} - waiting for 3 clear frames"
            self._click_notice_until = time.monotonic() + 2.0
            self._click = (track_id, dict(target), appearance)
            self._generation += 1
            self._displayed = None
            self._selected_id = track_id
            self._result = (0.0, dict(NOT_FOUND))
            if self._pending is not None:
                stamp, pending_frame, _ = self._pending
                self._pending = (stamp, pending_frame, self._generation)
            return True

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

    def draw_candidates(self, view, video_fresh=True):
        """Show selectable detections and measured delay without changing Target."""
        with self._condition:
            stamp, _ = self._result
            _, tracks, status = self._snapshot
            selected_id = self._selected_id
            inference_ms = self._inference_ms
            click_notice = self._click_notice if time.monotonic() < self._click_notice_until else ""
            age = time.monotonic() - stamp
            fresh = video_fresh and 0 <= age < self.MAX_AGE
            self._displayed = (stamp, self._snapshot, self._generation) if fresh else None
        if fresh:
            h, w = view.shape[:2]
            for track_id, target in tracks.items():
                x1, y1 = int((target["x"] - target["w"] / 2) * w), int((target["y"] - target["h"] / 2) * h)
                x2, y2 = int((target["x"] + target["w"] / 2) * w), int((target["y"] + target["h"] / 2) * h)
                selectable = target["conf"] >= PersonTracker.SELECT_CONF
                strong = target["conf"] >= PersonTracker.CONTROL_CONF
                color = (0, 220, 220) if strong else (0, 165, 255) if selectable else (130, 130, 130)
                label = "person" if strong else "candidate" if selectable else "weak"
                if track_id == selected_id:
                    label = "SELECTED"
                    color = (0, 220, 0) if strong else (0, 165, 255)
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 1)
                cv2.putText(view, f"{label} {track_id} {target['conf']:.0%}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        else:
            status = "Target remembered - video delayed, holding" if selected_id is not None else "Detection delayed - holding"
        best = max((t["conf"] for t in tracks.values()), default=0.) if fresh else 0.
        detail = click_notice or f"Boxes {len(tracks) if fresh else 0} | best {best:.0%} | steering: 60% + 3 frames"
        if selected_id is not None:
            status = f"Selection {selected_id}: {status}"
        # Drone frames can be only 320 pixels wide; do not clip the reason a
        # target cannot be selected or confirmed off the right edge.
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, .4
        lines = []
        for message in (status, f"Inference {inference_ms:.0f} ms | {detail}"):
            line = ""
            for word in message.split():
                joined = f"{line} {word}".strip()
                if line and cv2.getTextSize(joined, font, scale, 1)[0][0] > view.shape[1] - 20:
                    lines.append(line)
                    line = word
                else:
                    line = joined
            lines.append(line)
        for y, line in zip(range(110, view.shape[0] - 5, 16), lines):
            cv2.putText(view, line, (10, y), font, scale, (0, 220, 220), 1)

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
                        self._selected_id = self.tracker._selected_id
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
