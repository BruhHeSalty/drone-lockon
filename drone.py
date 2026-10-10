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

    def __init__(self, kind=None):
        kind = config.VIDEO_SOURCE if kind is None else kind
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

    @property
    def fresh(self):
        return self.cap.fresh if self.kind == "drone" else True

    @property
    def frame_time(self):
        """Local monotonic receive time, including time already spent in video queues."""
        return self.cap.frame_time if self.kind == "drone" else time.monotonic()


class DroneVideo:
    """Live video from the drone over UDP. Behaves like cv2.VideoCapture."""

    def __init__(self):
        _load_turbodrone()
        from protocols.wifi_uav_video_protocol import WifiUavVideoProtocolAdapter
        self._adapter_cls = WifiUavVideoProtocolAdapter
        self.adapter = None
        self.fresh = False
        self.frame_time = 0.0
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
        self.fresh = False
        if not self.adapter.is_running():
            print("[drone] no video — reconnecting (is the laptop on the drone's WiFi?)")
            self._connect()
        frame = self.adapter.get_frame(timeout=0.05)
        if frame is None:
            return False, None
        # Display late frames too, but never treat them as fresh control feedback.
        age = time.time() - frame.timestamp
        self.frame_time = time.monotonic() - max(0.0, age)
        img = cv2.imdecode(np.frombuffer(frame.data, np.uint8), cv2.IMREAD_COLOR)
        self.fresh = img is not None and -0.1 <= age <= 0.3
        return img is not None, img

    def release(self):
        global _shared_sock
        _shared_sock = None
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
    CALIBRATE_PACKETS = 20 # repeat gyro-calibrate for ~0.25 s
    SHUTDOWN_S = 0.75
    MANUAL_PULSE_S = 0.20  # key events must keep arriving; never latch a direction
    MANUAL_LIMITS = {"forward": 0.12, "right": 0.12, "up": 0.15, "yaw": 0.15}

    def __init__(self, flight_test=False):
        self.dry_run = config.CONTROL_PORT is None
        self.flight_test = flight_test
        self.flying = False
        self._closed = False
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
        self._manual_cmd = dict(HOVER)
        self._manual_until = 0.0
        self._hold_until = 0.0
        self._stop_left = 0
        self._calib_left = 0
        self._takeoff_at = None   # request consumed by the next sender tick
        self._video_time = None
        self._used_takeoff = False
        self._control_fault = False
        self._last_error = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            self._tick()
            time.sleep(1 / self.RATE_HZ)

    def _video_ready(self, now):
        return (self._video_time is not None
                and 0 <= now - self._video_time <= self.STALE_S)

    def note_frame(self):
        """Called only after receiving a fresh drone frame."""
        if self.dry_run:
            return
        now = time.monotonic()
        with self._lock:
            self._video_time = now

    def video_lost(self):
        if self.dry_run:
            return
        with self._lock:
            self._video_time = None
            self._takeoff_at = None
            self._calib_left = 0
            self._cmd = dict(HOVER)
            self._manual_until = 0.0

    def _transmit(self, packet):
        """Report local socket acceptance; this is NOT a drone acknowledgement.

        Upstream send_control_packet swallows socket failures, which would lose
        our one-shot landing flag. Send on the same adapter socket/target ports
        here so known local failures can preserve that intent.
        """
        accepted = False
        for port in self.rc._target_ports:
            try:
                self.rc.sock.sendto(packet, (self.rc.drone_ip, port))
                accepted = True
            except OSError:
                pass
        return accepted

    def _tick(self):
        now = time.monotonic()
        try:
            with self._lock:
                if self._takeoff_at is not None and (self._closed or self._control_fault or not self._video_ready(now)):
                    self._takeoff_at = None
                    self._calib_left = 0
                    print("[drone] Takeoff cancelled: fresh video required")
                if self._takeoff_at is not None and now >= self._takeoff_at:
                    self._takeoff_at = None
                    self.sticks.takeoff_flag = True
                    self.flying = True
                    self._used_takeoff = True
                    self._hold_until = now + self.TAKEOFF_HOLD_S
                cmd = self._cmd
                if self.flight_test:
                    cmd = self._manual_cmd if now < self._manual_until else HOVER
                if (self._closed or self._control_fault or not self.flying or now < self._hold_until
                        or now - self._cmd_time > self.STALE_S or not self._video_ready(now)):
                    cmd = HOVER
                self.sticks.yaw = _stick(cmd["yaw"])
                self.sticks.pitch = _stick(cmd["forward"])
                self.sticks.throttle = _stick(cmd["up"])
                self.sticks.roll = _stick(cmd["right"])
                if self._stop_left > 0:
                    self.sticks.stop_flag = True
                    self._stop_left -= 1
                if self._calib_left > 0:
                    self.sticks.calibration_flag = True
                    self._calib_left -= 1
                if _shared_sock is not None and self.rc.sock is not _shared_sock:
                    self.rc.set_socket(_shared_sock)
                landing = self.sticks.land_flag
                taking_off = self.sticks.takeoff_flag
                packet = self.rc.build_control_packet(self.sticks)
                accepted = self._transmit(packet)
                if not accepted:
                    self._control_fault = True
                    # Retry LAND only after a known local send failure. Never
                    # repeat a successfully submitted takeoff/land toggle.
                    self.sticks.land_flag = landing
                    if taking_off:
                        self.flying = False
                    self._cmd = dict(HOVER)
                    self._takeoff_at = None
                    if now - self._last_error >= 1.0:
                        print("[drone] Control send failed; physical response is unknown")
                        self._last_error = now
        except Exception as exc:
            with self._lock:
                self._takeoff_at = None
                self._cmd = dict(HOVER)
                self._control_fault = True
                self.sticks.takeoff_flag = False
            if now - self._last_error >= 1.0:
                print(f"[drone] Control error: {exc}")
                self._last_error = now

    def send(self, cmd):
        if self.dry_run:
            now = time.time()
            if now - self._last_print > 0.5:
                print(f"[drone] would send {cmd}")
                self._last_print = now
            return
        with self._lock:
            self._cmd = dict(HOVER if self.flight_test else cmd)
            self._cmd_time = time.monotonic()

    def hover(self):
        if not self.dry_run:
            with self._lock:
                self._manual_until = 0.0
        self.send(HOVER)

    def manual_nudge(self, axis, direction):
        """One low-speed axis for 0.2 seconds; repeat key events to extend.

        Only available in flight-test mode after takeoff settling. These are
        relative stick inputs, not altitude/angle setpoints or position hold.
        """
        if axis not in self.MANUAL_LIMITS or direction not in (-1, 1):
            raise ValueError("Manual input requires a known axis and direction -1 or 1")
        if not self.flight_test:
            return False
        if self.dry_run:
            print(f"[drone] DRY RUN manual {axis}: {direction:+d}")
            return False
        now = time.monotonic()
        with self._lock:
            if (self._closed or self._control_fault or not self.flying
                    or now < self._hold_until or not self._video_ready(now)):
                print("[drone] Manual input blocked: need airborne state, settled takeoff and live video")
                return False
            configured = {"forward": config.MAX_FORWARD, "right": config.MAX_FORWARD,
                          "up": config.MAX_UP, "yaw": config.MAX_YAW}
            limit = max(0.0, min(self.MANUAL_LIMITS[axis], configured[axis]))
            self._manual_cmd = dict(HOVER)
            self._manual_cmd[axis] = direction * limit
            self._manual_until = now + self.MANUAL_PULSE_S
            self._cmd_time = now
            return True

    def manual_command(self):
        """Current requested manual sticks for the HUD (not measured motion)."""
        if self.dry_run:
            return dict(HOVER)
        now = time.monotonic()
        with self._lock:
            if (self.flight_test and self.flying and not self._closed and not self._control_fault
                    and now >= self._hold_until and now < self._manual_until and self._video_ready(now)):
                return dict(self._manual_cmd)
        return dict(HOVER)

    def _ground_action_block(self, now):
        if self._closed:
            return "program is closing"
        if self._control_fault:
            return "control socket failed; reconnect and restart while grounded"
        if self.flying or self._takeoff_at is not None:
            return "takeoff already requested"
        if self._used_takeoff:
            return "this session already requested takeoff/stop; confirm grounded, then restart"
        if not self._video_ready(now):
            return "drone video is missing or stale"
        return None

    def preflight(self):
        """Optional P key: calibrate on a flat surface with motors off."""
        if self.dry_run:
            print("[drone] DRY RUN — would calibrate gyro")
            return
        with self._lock:
            reason = self._ground_action_block(time.monotonic())
            if reason:
                print(f"[drone] Calibration blocked: {reason}")
                return
            self._calib_left = self.CALIBRATE_PACKETS
            print("[drone] Calibrating gyro")

    def takeoff(self):
        """T. Request takeoff directly, without automatic calibration or delay."""
        if self.dry_run:
            print("[drone] DRY RUN — would request takeoff")
            return
        with self._lock:
            now = time.monotonic()
            reason = self._ground_action_block(now)
            if reason:
                print(f"[drone] Takeoff blocked: {reason}")
                return
            self._calib_left = 0
            self._takeoff_at = now
            print("[drone] Takeoff requested")

    def land(self):
        """SPACE. Takeoff and land are the same toggle bit on this drone, so it
        is only sent while flying — otherwise it would take off instead."""
        print("[drone] LAND")
        if self.dry_run:
            return
        with self._lock:
            self._takeoff_at = None
            self._calib_left = 0
            self.sticks.takeoff_flag = False
            self._cmd = dict(HOVER)
            self._manual_until = 0.0
            if not self.flying:
                return
            self.sticks.land_flag = True
            self.flying = False

    def kill(self):
        """EMERGENCY: motors off immediately. The drone drops."""
        print("[drone] KILL MOTORS")
        if self.dry_run:
            return
        with self._lock:
            self._takeoff_at = None
            self._calib_left = 0
            self.sticks.takeoff_flag = self.sticks.land_flag = False
            self._cmd = dict(HOVER)
            self._manual_until = 0.0
            self._stop_left = self.STOP_PACKETS
            self._used_takeoff = True
            self.flying = False

    def close(self):
        if self.dry_run or self._closed:
            return
        with self._lock:
            self._closed = True
        self.land()  # also cancels a scheduled takeoff while still on the ground
        # Keep the shared video socket alive until this flush is finished.
        time.sleep(self.SHUTDOWN_S)
        self._running = False
        self._thread.join(timeout=1.0)
        if self.sticks.land_flag:
            print("[drone] WARNING: landing packet could not be submitted; physical landing is unconfirmed")
        self.rc.stop()
