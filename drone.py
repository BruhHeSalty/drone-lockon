"""OWNER: Person 1 — talk to the drone (Contixo F19, "WiFi UAV" app).

Two jobs:
  1. VIDEO:   get frames from the drone's camera
  2. CONTROL: send flight commands

Both reuse the reverse-engineered WiFi UAV protocol from Turbodrone
(https://github.com/marshallrichards/turbodrone), cloned at config.TURBODRONE_PATH.
Only its plain-Python protocol files are imported, so no extra installs.

Without the drone everything still runs:
  - VIDEO_SOURCE = "file" / "webcam" in config.py
  - CONTROL_PORT = None in config.py -> commands are only PRINTED (dry run)
"""
import os
import sys
import threading
import time

import cv2
import numpy as np

import config

HOVER = {"yaw": 0.0, "up": 0.0, "forward": 0.0, "right": 0.0}

# Set by DroneVideo; DroneLink sends its packets on the same UDP socket.
_shared_sock = None


def _load_turbodrone():
    path = os.path.expanduser(config.TURBODRONE_PATH)
    if not os.path.isdir(path):
        raise RuntimeError(
            f"Turbodrone not found at {path}. Run:\n"
            "  git clone https://github.com/marshallrichards/turbodrone ~/turbodrone"
        )
    if path not in sys.path:
        sys.path.append(path)


# ---------------------------------------------------------------- VIDEO
class VideoSource:
    """Gives frames from a file, the webcam, or the drone."""

    def __init__(self, kind=config.VIDEO_SOURCE):
        self.kind = kind
        if kind == "file":
            self.cap = cv2.VideoCapture(config.VIDEO_FILE)
        elif kind == "webcam":
            self.cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        elif kind == "drone":
            self.cap = DroneVideo()
        else:
            raise ValueError(f"unknown VIDEO_SOURCE {kind!r}")

    def read(self):
        """Return a BGR frame, or None if there isn't one."""
        ok, frame = self.cap.read()
        if not ok and self.kind == "file":
            # Loop the recorded video so testing never stops
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class DroneVideo:
    """Live video from the drone over UDP. Behaves like cv2.VideoCapture."""

    def __init__(self):
        _load_turbodrone()
        from protocols.wifi_uav_video_protocol import WifiUavVideoProtocolAdapter
        self._adapter_cls = WifiUavVideoProtocolAdapter
        self.adapter = None
        self._connect()

    def _connect(self):
        global _shared_sock
        if self.adapter is not None:
            self.adapter.stop()
        self.adapter = self._adapter_cls(
            config.DRONE_IP, config.VIDEO_PORT, config.VIDEO_PORT,
            variant=config.WIFI_UAV_VARIANT,
        )
        self.adapter.start()
        _shared_sock = self.adapter.get_receiver_socket()

    def read(self):
        if not self.adapter.is_running():
            print("[drone] no video — reconnecting (is the Mac on the drone's WiFi?)")
            self._connect()
        frame = self.adapter.get_frame(timeout=0.5)
        if frame is None:
            return False, None
        img = cv2.imdecode(np.frombuffer(frame.data, np.uint8), cv2.IMREAD_COLOR)
        return img is not None, img

    def release(self):
        self.adapter.stop()


# -------------------------------------------------------------- CONTROL
def _stick(value):
    """-1..1 -> 0..255 with 128 = center."""
    return int(max(0, min(255, round(128 + value * 127))))


class _Sticks:
    """The fields Turbodrone's packet builder reads."""

    def __init__(self):
        self.yaw = self.pitch = self.throttle = self.roll = 128
        self.takeoff_flag = self.land_flag = self.stop_flag = False
        self.calibration_flag = self.flip_flag = self.headless_flag = False
        self.camera_tilt_state = 0
        self.speed_index = 2  # 100%; config.MAX_* already caps speed


class DroneLink:
    """Send Commands (see INTERFACES.md) to the drone.

    While CONTROL_PORT is None this is a dry run: it prints instead of sending.
    Otherwise a background thread streams stick packets at 80 Hz, because the
    drone expects a steady stream. Sticks stay centered (hover) unless we took
    off ourselves, and whenever send() hasn't been called recently.
    """

    RATE_HZ = 80
    STALE_S = 0.3          # no send() for this long -> hover
    TAKEOFF_HOLD_S = 3.0   # hover while the drone climbs after takeoff
    STOP_PACKETS = 40      # repeat motor-stop for ~0.5 s so it can't get lost

    def __init__(self):
        self.dry_run = config.CONTROL_PORT is None
        self.flying = False
        self._last_print = 0.0
        if self.dry_run:
            print("[drone] DRY RUN — commands are printed, not sent")
            return

        _load_turbodrone()
        from protocols.wifi_uav_rc_protocol_adapter import WifiUavRcProtocolAdapter
        self.rc = WifiUavRcProtocolAdapter(
            config.DRONE_IP, config.CONTROL_PORT, variant=config.WIFI_UAV_VARIANT
        )
        self.sticks = _Sticks()
        self._cmd = HOVER
        self._cmd_time = 0.0
        self._hold_until = 0.0
        self._stop_left = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            now = time.time()
            with self._lock:
                cmd = self._cmd
                if not self.flying or now < self._hold_until or now - self._cmd_time > self.STALE_S:
                    cmd = HOVER
                self.sticks.yaw = _stick(cmd["yaw"])
                self.sticks.pitch = _stick(cmd["forward"])
                self.sticks.throttle = _stick(cmd["up"])
                self.sticks.roll = _stick(cmd["right"])
                if self._stop_left > 0:
                    self.sticks.stop_flag = True
                    self._stop_left -= 1
                packet = self.rc.build_control_packet(self.sticks)
            if _shared_sock is not None and self.rc.sock is not _shared_sock:
                self.rc.set_socket(_shared_sock)
            self.rc.send_control_packet(packet)
            time.sleep(1 / self.RATE_HZ)

    def send(self, cmd):
        if self.dry_run:
            now = time.time()
            if now - self._last_print > 0.5:
                print(f"[drone] would send {cmd}")
                self._last_print = now
            return
        with self._lock:
            self._cmd = cmd
            self._cmd_time = time.time()

    def hover(self):
        self.send(HOVER)

    def takeoff(self):
        print("[drone] TAKEOFF")
        if self.dry_run or self.flying:
            return
        with self._lock:
            self.sticks.takeoff_flag = True
            self.flying = True
            self._hold_until = time.time() + self.TAKEOFF_HOLD_S

    def land(self):
        """SPACE. Takeoff and land are the same toggle bit on this drone, so it
        is only sent while flying — otherwise it would take off instead."""
        print("[drone] LAND")
        if self.dry_run or not self.flying:
            return
        with self._lock:
            self.sticks.land_flag = True
            self.flying = False

    def kill(self):
        """EMERGENCY: motors off immediately. The drone drops."""
        print("[drone] KILL MOTORS")
        if self.dry_run:
            return
        with self._lock:
            self._stop_left = self.STOP_PACKETS
            self.flying = False

    def close(self):
        if self.dry_run:
            return
        if self.flying:
            self.land()
            time.sleep(0.3)
        self._running = False
        self._thread.join(timeout=1.0)
        self.rc.stop()
