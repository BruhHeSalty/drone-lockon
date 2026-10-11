"""OWNER: Person 4 — turn a Target into a Command, plus all the safety logic.

Simple proportional control:
  target left of center  -> yaw left
  target above center    -> climb
  target small           -> fly forward
  target big enough      -> STOP (hover). Never ram the target.
  target lost            -> hover
"""
import math
import time

import config

HOVER = {"yaw": 0.0, "up": 0.0, "forward": 0.0, "right": 0.0}


def _clamp(v, limit):
    return max(-limit, min(limit, v))


def _deadband(err):
    return 0.0 if abs(err) < config.CENTER_DEADBAND else err


class Controller:
    FOLLOW_FORWARD_CAP = 0.08
    FOLLOW_SETTLE_S = 0.4

    def __init__(self, person_mode=False, follow_mode=False):
        if follow_mode and not person_mode:
            raise ValueError("Follow mode requires selected-person tracking")
        self.person_mode = person_mode
        self.follow_mode = follow_mode
        self._follow_reference = None
        self._far_since = None
        self.lost_count = 0
        self.state = "SEARCHING"

    @staticmethod
    def _valid_person(target):
        return bool(target.get("found")) and all(
            isinstance(target.get(k), (int, float)) and math.isfinite(target[k])
            and 0 <= target[k] <= 1 for k in ("x", "y", "w", "h", "conf")
        ) and target["conf"] >= 0.6 and min(target["w"], target["h"]) > 0

    @staticmethod
    def _fully_visible(target):
        return all(target[c] - target[s] / 2 > .02 and target[c] + target[s] / 2 < .98
                   for c, s in (("x", "w"), ("y", "h")))

    def disarm_follow(self):
        self._follow_reference = None
        self._far_since = None

    def follow_block_reason(self, target):
        if not self.follow_mode:
            return "follow mode is off"
        if not self._valid_person(target):
            return "waiting for the selected person to be confirmed"
        if target["w"] >= config.STOP_SIZE or target["h"] >= .65:
            return "selected person is at the close-stop size"
        if not self._fully_visible(target):
            return "waiting for the whole person box to be inside the frame"
        return None

    def arm_follow(self, target):
        """Remember apparent size at the operator's chosen starting gap.

        This is not a measurement in metres or obstacle avoidance. Re-arming
        or selecting someone else always requires a new reference.
        """
        self.disarm_follow()
        if self.follow_block_reason(target) is not None:
            return False
        self._follow_reference = (target["w"], target["h"])
        return True

    def _follow_forward(self, target):
        if self._follow_reference is None:
            return 0.0
        # Cropping, turning sideways, or a changing box shape can resemble
        # receding motion. Require a complete box, alignment, and shrinkage
        # in BOTH dimensions. Loss/uncertainty always resets the dwell timer.
        width, height = self._follow_reference
        scale = max(target["w"] / width, target["h"] / height)
        if (not self._fully_visible(target) or abs(target["x"] - .5) > .10
                or abs(target["y"] - .5) > .25 or scale >= .85):
            self._far_since = None
            return 0.0
        now = time.monotonic()
        if self._far_since is None:
            self._far_since = now
        if now - self._far_since < self.FOLLOW_SETTLE_S:
            return 0.0
        return min(max(0.0, config.MAX_FORWARD), self.FOLLOW_FORWARD_CAP, (.85 - scale) * .3)

    def update(self, target):
        """Return (command, state)."""
        if self.person_mode:
            if not self._valid_person(target):
                self._far_since = None
                self.state = "LOST"
                return dict(HOVER), self.state
            if target["w"] >= config.STOP_SIZE or target["h"] >= 0.65:
                self._far_since = None
                self.state = "REACHED"
                return dict(HOVER), self.state
            cmd = dict(HOVER)
            cmd["yaw"] = _clamp(_deadband(target["x"] - 0.5) * 2 * config.YAW_GAIN,
                                max(0.0, min(config.MAX_YAW, 0.15)))
            self.state = "LOCKED"
            if self.follow_mode:
                cmd["forward"] = self._follow_forward(target)
                if cmd["forward"] > 0:
                    self.state = "APPROACHING"
            return cmd, self.state
        if not target["found"]:
            self.lost_count += 1
            if self.state != "SEARCHING" and self.lost_count > config.LOST_FRAMES:
                self.state = "LOST"
            return dict(HOVER), self.state

        self.lost_count = 0

        # Error from frame center: positive x_err = target is to the right
        x_err = _deadband(target["x"] - 0.5)
        y_err = _deadband(0.5 - target["y"])  # positive = target is above center

        yaw = _clamp(x_err * 2 * config.YAW_GAIN, config.MAX_YAW)
        up = _clamp(y_err * 2 * config.UP_GAIN, config.MAX_UP)

        size = target["w"]
        centered = abs(x_err) < 0.15
        if size >= config.STOP_SIZE:
            forward = 0.0
            self.state = "REACHED"
        elif not centered:
            forward = 0.0           # turn to face it first, then go
            self.state = "LOCKED"
        else:
            # Slow down as it gets closer
            forward = (config.STOP_SIZE - size) / config.STOP_SIZE * config.FORWARD_GAIN
            forward = _clamp(forward, config.MAX_FORWARD)
            self.state = "APPROACHING"

        return {"yaw": yaw, "up": up, "forward": forward, "right": 0.0}, self.state
