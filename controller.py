"""OWNER: Person 4 — turn a Target into a Command, plus all the safety logic.

Simple proportional control:
  target left of center  -> yaw left
  target above center    -> climb
  target small           -> fly forward
  target big enough      -> STOP (hover). Never ram the target.
  target lost            -> hover
"""
import config

HOVER = {"yaw": 0.0, "up": 0.0, "forward": 0.0, "right": 0.0}


def _clamp(v, limit):
    return max(-limit, min(limit, v))


def _deadband(err):
    return 0.0 if abs(err) < config.CENTER_DEADBAND else err


class Controller:
    def __init__(self):
        self.lost_count = 0
        self.state = "SEARCHING"

    def update(self, target):
        """Return (command, state)."""
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
