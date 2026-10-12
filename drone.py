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
import logging
import json
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import config

# Zero on every control means "do not ask for movement." It does not prove
# the drone is staying in one place; we cannot measure that here.
HOVER = {"yaw": 0.0, "up": 0.0, "forward": 0.0, "right": 0.0}

# Incident 2026-10-12: closing after an unconfirmed takeoff sent the shared
# takeoff/land toggle and the operator reported an unexpected launch. Neither
# sendto() nor self.flying establishes the physical state. Shutdown now requests
# motor stop, but SPACE still uses the ambiguous toggle. The operator requested
# resuming manual tests and yaw-only tracking; unrestricted control stays blocked.
LIVE_FLIGHT_BLOCK_REASON = (
    "Unrestricted live flight is disabled after an unexpected takeoff on program close. "
    "The current takeoff/land toggle cannot guarantee landing without flight-state feedback. "
    "Only explicit --flight-test or turn-only --person modes are available."
)


def require_safe_live_control(*, flight_test=False, yaw_only=False):
    if flight_test and yaw_only:
        raise ValueError('Choose manual flight test or yaw-only control, not both')
    if (config.CONTROL_PORT is not None and LIVE_FLIGHT_BLOCK_REASON
            and flight_test is not True and yaw_only is not True):
        raise RuntimeError(LIVE_FLIGHT_BLOCK_REASON)

# Set by DroneVideo; DroneLink sends its packets on the same UDP socket.
_shared_sock = None
_shared_control_link = None

def _load_turbodrone():
    # Find the helper library that knows how to write messages the drone understands.
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
        # Choose our eyes: a saved movie, the laptop camera, or the drone camera.
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


class _LatestFrameVideo:
    """Adapter mixin: publish complete frames without waiting for the consumer.

    Turbodrone's receive loop queues handle_payload()'s return value with a
    blocking put. Publish here and return None to bypass that queue, while
    retaining its packet assembly, acknowledgements, watchdog and socket.
    """

    def __init__(self, *args, **kwargs):
        # Upstream starts its watchdog during __init__. Create this first so
        # receive/ACK callbacks always share a lock, including during startup.
        self._ack_lock = threading.RLock()
        self._frame_ready = threading.Condition()
        self._latest_frame = None
        self.frames_received = 0
        self.frames_skipped = 0
        self._stats_at = time.monotonic()
        self._stats_received = 0
        self._last_frame_at = None
        self._max_frame_gap = 0.0
        super().__init__(*args, **kwargs)

    def handle_payload(self, payload):
        # Pictures arrive as little pieces. Let the helper put them together.
        # This mutates received_fragments; the watchdog concurrently iterates
        # that set to build ACKs. Completion also requests an ACK recursively,
        # so this must be reentrant rather than a plain Lock.
        with self._ack_lock:
            frame = super().handle_payload(payload)
        if frame is not None:
            now = time.monotonic()
            with self._frame_ready:
                if self._latest_frame is not None:
                    # Replace an unread picture with the newer one to avoid falling behind.
                    self.frames_skipped += 1
                self._latest_frame = frame
                self.frames_received += 1
                if self._last_frame_at is not None:
                    self._max_frame_gap = max(self._max_frame_gap, now - self._last_frame_at)
                self._last_frame_at = now
                self._frame_ready.notify()
        return None

    def _build_ack_slots(self, seq):
        with self._ack_lock:
            return super()._build_ack_slots(seq)

    def _safe_sendto(self, payload, port, label):
        # An ACK says "got your video." It also carries joystick numbers, so
        # those numbers must agree with the movement request we are sending.
        # Native video ACKs also contain RC bytes. Sending the upstream
        # neutral default between movement packets fights keyboard input.
        link = _shared_control_link
        if (link is not None and self._sock is _shared_sock
                and len(payload) >= 38 and payload[:2] == b'\xef\x02'
                and payload[16:20] == b'\x14\x00\x66\x14'):
            # Serialize with H/release/land: never send an ACK built from a
            # command snapshot taken before a concurrent stop request.
            with link._lock:
                payload = link._video_ack_payload(payload)
                accepted = super()._safe_sendto(payload, port, label)
                now = time.monotonic()
                signature = (payload[20:26], accepted)
                if signature != link._last_ack_log or now - link._ack_log_time >= .1:
                    link.record_event('video_ack_control', sticks=list(payload[20:24]),
                                      packet_hex=payload.hex(), port=port, locally_accepted=accepted)
                    link._last_ack_log, link._ack_log_time = signature, now
                return accepted
        return super()._safe_sendto(payload, port, label)

    def get_frame(self, timeout=1.0):
        with self._frame_ready:
            if not self._frame_ready.wait_for(lambda: self._latest_frame is not None, timeout):
                return None
            frame = self._latest_frame
            self._latest_frame = None
            self._first_frame = False
            now = time.monotonic()
            elapsed = now - self._stats_at
            if elapsed >= 5.0:
                stats = ((self.frames_received - self._stats_received) / elapsed,
                         self.frames_received, self.frames_skipped,
                         1000 * max(0.0, time.time() - frame.timestamp),
                         1000 * self._max_frame_gap)
                self._stats_at = now
                self._stats_received = self.frames_received
                self._max_frame_gap = 0.0
            else:
                stats = None
        # Logging stays outside the handoff lock so it cannot hold up reception.
        if stats is not None:
            logging.getLogger(__name__).info(
                "[video] receive=%.1f fps frames=%d skipped=%d local_age=%.0f ms max_gap=%.0f ms",
                *stats,
            )
        return frame


class DroneVideo:
    """Live video from the drone over UDP. Behaves like cv2.VideoCapture."""

    def __init__(self):
        _load_turbodrone()
        from protocols.wifi_uav_video_protocol import WifiUavVideoProtocolAdapter
        class LatestFrameAdapter(_LatestFrameVideo, WifiUavVideoProtocolAdapter):
            pass

        self._adapter_cls = LatestFrameAdapter
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
        # The picture's age matters: seeing where something WAS is not enough.
        self.frame_time = time.monotonic() - max(0.0, age)
        img = cv2.imdecode(np.frombuffer(frame.data, np.uint8), cv2.IMREAD_COLOR)
        self.fresh = img is not None and -0.1 <= age <= 0.3
        return img is not None, img

    def release(self):
        global _shared_sock
        _shared_sock = None
        self.adapter.stop()


# -------------------------------------------------------------- CONTROL
class FlightRecorder:
    """Append-only diagnostic journal; disk I/O stays off the control thread."""
    # Keep a notebook of what the program tried. A helper writes it to disk so
    # a slow disk does not make the movement-message loop wait.

    def __init__(self, directory="flight_logs"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"flight_{time.time_ns()}.jsonl"
        self._file = self.path.open("x", encoding="utf-8", buffering=1)
        self._queue = queue.Queue(maxsize=4096)
        self._started = time.monotonic()
        self._closed = False
        self._failed = False
        self.dropped = 0
        self._thread = threading.Thread(target=self._write, daemon=True)
        self._thread.start()
        print(f"[drone] Flight-attempt log: {self.path.resolve()}")

    def record(self, event, **fields):
        if self._closed or self._failed:
            return
        row = dict(fields, event=event, utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_s=round(time.monotonic() - self._started, 4))
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            while True:
                row = self._queue.get()
                if row is None:
                    break
                self._file.write(json.dumps(row, allow_nan=False) + "\n")
        except (OSError, ValueError, TypeError) as exc:
            self._failed = True
            print(f"[drone] Flight log failed: {exc}")
        finally:
            self._file.close()

    def close(self):
        if self._closed:
            return
        self.record("journal_end", dropped_events=self.dropped)
        self._closed = True
        if not self._failed:
            try:
                self._queue.put(None, timeout=.5)
            except queue.Full:
                print("[drone] Flight log writer stalled; log may be incomplete")
        self._thread.join(timeout=1.)
        if self._thread.is_alive():
            print("[drone] Flight log still flushing; final rows may be incomplete")


def _stick(value):
    """-1..1 -> 0..255 with 128 = center."""
    # Translate our small movement number into the drone's joystick number.
    return int(max(0, min(255, round(128 + value * 127))))


class _Sticks:
    """The fields Turbodrone's packet builder reads."""
    # Drone names: pitch = forward/back, throttle = up/down, roll = slide sideways.

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
    drone expects a steady stream. Sticks stay centered unless we requested
    takeoff ourselves, and whenever send() hasn't been called recently.
    Centered sticks are not a verified position/altitude hold on this airframe.
    """

    RATE_HZ = 80
    STALE_S = 0.3          # no send() for this long -> hover
    TAKEOFF_HOLD_S = 3.0   # neutral sticks after takeoff; stable hover is unmeasured
    STOP_PACKETS = 40      # repeat motor-stop for ~0.5 s so it can't get lost
    CALIBRATE_PACKETS = 20 # repeat gyro-calibrate for ~0.25 s
    SHUTDOWN_S = 0.75
    MANUAL_PULSE_S = 0.20  # key events must keep arriving; never latch a direction
    MANUAL_LIMITS = {"forward": 0.12, "right": 0.12, "up": 0.15, "yaw": 0.15}
    KEYBOARD_LIMITS = {"forward": 0.20, "right": 0.20, "up": 0.20, "yaw": 0.20}

    def __init__(self, flight_test=False, yaw_only=False):
        global _shared_control_link
        require_safe_live_control(flight_test=flight_test, yaw_only=yaw_only)
        self.dry_run = config.CONTROL_PORT is None
        self.flight_test = flight_test
        self.yaw_only = yaw_only
        self.flying = False
        self._closed = False
        self._last_print = 0.0
        self._recorder = None
        self._manual_cmd = dict(HOVER)
        self._manual_until = 0.0
        self._last_manual_request = None
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
        self._stop_packets_submitted = 0
        self._calib_left = 0
        self._takeoff_at = None   # request consumed by the next sender tick
        self._video_time = None
        self._used_takeoff = False
        self._control_fault = False
        self._last_error = 0.0
        self._send_error = None
        self._last_control_log = None
        self._control_log_time = float("-inf")
        self._attempt_at = None
        self._attempt_count = 0
        self._last_packet_log = None
        self._packet_log_time = float("-inf")
        self._last_ack_log = None
        self._ack_log_time = float('-inf')
        try:
            self._recorder = FlightRecorder()
        except OSError:
            self.rc.stop()
            raise
        self.record_event("session_start", variant=config.WIFI_UAV_VARIANT,
                          flight_test=flight_test, airborne_verified=False,
                          rate_hz=self.RATE_HZ, speed_index=self.sticks.speed_index,
                          command_sequence='upstream',
                          note="Times measure command attempts, not time in the air")
        print('[drone] Control transport: original Turbodrone packet numbering')
        self._lock = threading.Lock()
        _shared_control_link = self
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def record_event(self, event, **fields):
        if self._recorder is not None:
            fields.setdefault('attempt', self._attempt_count)
            self._recorder.record(event, **fields)

    def _finish_attempt(self, reason):
        if self._attempt_at is None:
            return
        duration = max(0., time.monotonic() - self._attempt_at)
        outcome = "suspected_failed" if duration < 30 else "unverified"
        self.record_event("attempt_end", attempt=self._attempt_count,
                          duration_s=round(duration, 3), ended_by=reason,
                          outcome=outcome, airborne_verified=False,
                          classification_reason="under_30_seconds" if duration < 30 else "no_airborne_telemetry")
        print(f"[drone] Attempt {self._attempt_count}: {duration:.1f}s, {outcome}; actual airtime unknown")
        self._attempt_at = None

    def _loop(self):
        # Keep sending like a heartbeat; the drone expects regular messages.
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
        # No fresh eyes: cancel pending takeoff and clear movement requests.
        if self.dry_run:
            return
        with self._lock:
            self.record_event("video_lost")
            if self._takeoff_at is not None:
                self._finish_attempt("video_lost_before_takeoff")
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
        errors = []
        for port in self.rc._target_ports:
            try:
                self.rc.sock.sendto(packet, (self.rc.drone_ip, port))
                accepted = True
            except OSError as exc:
                errors.append(f"UDP {port}: {exc}")
        self._send_error = "; ".join(errors) if not accepted else None
        return accepted

    def _effective_command(self, now):
        """Caller holds _lock; same decision for RC and embedded video controls."""
        # Last safety gate: old instructions, old pictures, or a broken link
        # must produce zero movement requests, even if an earlier step said go.
        cmd = self._cmd
        if self.flight_test:
            cmd = self._manual_cmd if now < self._manual_until else HOVER
        if (self._closed or self._control_fault or not self.flying or now < self._hold_until
                or now - self._cmd_time > self.STALE_S or not self._video_ready(now)):
            return HOVER
        if self.yaw_only:
            # Final transport gate, also used by video ACKs: tracker/controller
            # mistakes cannot request translation or altitude in person mode.
            yaw = cmd.get('yaw', 0.)
            cap = config.MAX_YAW
            if not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
                return HOVER
            cap = max(0., min(.15, cap)) if isinstance(cap, (int, float)) and math.isfinite(cap) else 0.
            return dict(HOVER, yaw=max(-cap, min(cap, yaw)))
        return cmd

    def _video_ack_payload(self, payload):
        """Replace only the embedded sticks/checksum, never repeat action flags."""
        cmd = self._effective_command(time.monotonic())
        axes = ('right', 'forward', 'up', 'yaw') if self.rc.capabilities.transport == 'fld_compat' else ('yaw', 'forward', 'up', 'right')
        packet = bytearray(payload)
        packet[20:24] = bytes(_stick(cmd[axis]) for axis in axes)
        packet[24] = 0  # Takeoff/land/stop/calibration remain owned by the RC sender.
        packet[36] = 0
        for value in packet[20:36]:
            packet[36] ^= value
        return bytes(packet)

    def _tick(self):
        # Send one heartbeat: check safety, pack the joystick numbers, and send.
        now = time.monotonic()
        try:
            with self._lock:
                if self._takeoff_at is not None and (self._closed or self._control_fault or not self._video_ready(now)):
                    self._takeoff_at = None
                    self._calib_left = 0
                    self._finish_attempt("takeoff_cancelled")
                    print("[drone] Takeoff cancelled: fresh video required")
                if self._takeoff_at is not None and now >= self._takeoff_at:
                    self._takeoff_at = None
                    self.sticks.takeoff_flag = True
                    self.flying = True
                    self._used_takeoff = True
                    self._hold_until = now + self.TAKEOFF_HOLD_S
                cmd = self._effective_command(now)
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
                stopping = self.sticks.stop_flag
                calibrating = self.sticks.calibration_flag
                packet = self.rc.build_control_packet(self.sticks)
                accepted = self._transmit(packet)
                if stopping and accepted:
                    self._stop_packets_submitted += 1
                signature = (tuple(cmd.values()), accepted, taking_off, landing, stopping, calibrating, self._control_fault)
                if signature != self._last_packet_log or now - self._packet_log_time >= .1:
                    self.record_event("control_packet", command=dict(cmd),
                                      sticks=dict(yaw=self.sticks.yaw, pitch=self.sticks.pitch,
                                                  throttle=self.sticks.throttle, roll=self.sticks.roll),
                                      packet_hex=packet.hex() if isinstance(packet, bytes) else None,
                                      locally_accepted=accepted, error=self._send_error,
                                      takeoff=taking_off, land=landing,
                                      motor_stop=stopping, calibrating=calibrating,
                                      video_age_s=None if self._video_time is None else now - self._video_time,
                                      takeoff_hold=now < self._hold_until,
                                      airborne_verified=False)
                    self._last_packet_log = signature
                    self._packet_log_time = now
                # Log the locally submitted controls, not detector wishes or
                # measured motion. Rate-limit moving-stick changes.
                phase = ("TAKEOFF HOLD" if now < self._hold_until and self.flying
                         else "ACTIVE" if self.flying else "GROUND/LAND")
                controls = (self.sticks.yaw, self.sticks.pitch,
                            self.sticks.throttle, self.sticks.roll)
                snapshot = (phase, controls)
                phase_changed = self._last_control_log is None or phase != self._last_control_log[0]
                if accepted and snapshot != self._last_control_log and (phase_changed or now - self._control_log_time >= 1):
                    print(f"[drone] {phase}: submitted Y/P/T/R={controls} (128=neutral; motion unmeasured)")
                    self._last_control_log = snapshot
                    self._control_log_time = now
                if not accepted:
                    if not self._control_fault:
                        self.record_event("control_fault", error=self._send_error)
                    self._control_fault = True
                    # Retry LAND only after a known local send failure. Never
                    # repeat a successfully submitted takeoff/land toggle.
                    self.sticks.land_flag = landing
                    if taking_off:
                        self.flying = False
                    self._cmd = dict(HOVER)
                    self._takeoff_at = None
                    if now - self._last_error >= 1.0:
                        print(f"[drone] Control send failed ({self._send_error}); physical response is unknown. "
                              "Movement disabled for this session; land with the physical controller "
                              "and restart only when grounded.")
                        self._last_error = now
        except Exception as exc:
            with self._lock:
                self._takeoff_at = None
                self._cmd = dict(HOVER)
                self._control_fault = True
                self.sticks.takeoff_flag = False
            self.record_event("control_fault", error=str(exc))
            if now - self._last_error >= 1.0:
                print(f"[drone] Control error: {exc}")
                self._last_error = now

    def send(self, cmd):
        # Save the newest request for the sending helper. In dry run, just print it.
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
        if self.dry_run:
            self._manual_cmd = dict(HOVER)
            self._manual_until = 0.0
        if not self.dry_run:
            with self._lock:
                self._manual_until = 0.0
        self.send(HOVER)

    def manual_input(self, axes):
        """Refresh capped held-key input; expiry also stops a stalled UI.

        These are normalized stick requests, not angles, speeds or altitude.
        The tracker cannot call this through the ordinary send() path.
        """
        # Held keys must keep checking in. If the window freezes, their request expires.
        if set(axes) != set(HOVER) or any(not isinstance(v, (int, float))
                                        or not math.isfinite(v) or not -1 <= v <= 1 for v in axes.values()):
            self.hover()
            raise ValueError('Manual axes must contain four finite values in -1..1')
        if not self.flight_test:
            return False
        configured = {'forward': config.MAX_FORWARD, 'right': config.MAX_FORWARD,
                      'up': config.MAX_UP, 'yaw': config.MAX_YAW}
        command = {axis: axes[axis] * max(0., min(self.KEYBOARD_LIMITS[axis], configured[axis]))
                   for axis in HOVER}
        now = time.monotonic()
        if self.dry_run:
            self._manual_cmd = command
            self._manual_until = now + self.MANUAL_PULSE_S
            return True
        with self._lock:
            reason = self._movement_block_reason(now)
            ready = reason is None
            effective = command if ready else dict(HOVER)
            signature = (tuple(axes[axis] for axis in HOVER), reason)
            if signature != self._last_manual_request:
                self.record_event('manual_input', requested_axes=dict(axes), command=effective,
                                  ready=ready, blocked_reason=reason)
                if reason is not None and any(axes.values()):
                    print(f'[drone] Keys received, movement blocked: {reason}')
                self._last_manual_request = signature
            self._manual_cmd = effective
            self._manual_until = now + self.MANUAL_PULSE_S if ready else 0.
            self._cmd_time = now
            return ready

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
            return dict(self._manual_cmd if time.monotonic() < self._manual_until else HOVER)
        now = time.monotonic()
        with self._lock:
            if (self.flight_test and self.flying and not self._closed and not self._control_fault
                    and now >= self._hold_until and now < self._manual_until and self._video_ready(now)):
                return dict(self._manual_cmd)
        return dict(HOVER)

    def tracking_block_reason(self):
        """Local readiness only; the drone does not report a stable hover."""
        if self.dry_run:
            return None
        with self._lock:
            return self._movement_block_reason(time.monotonic())

    def video_ready(self):
        """Whether the last fresh video receipt is within the existing timeout."""
        if self.dry_run:
            return False
        with self._lock:
            return self._video_ready(time.monotonic())

    def _movement_block_reason(self, now):
        """Caller holds _lock. A reason explains why requested keys send zero."""
        if self._closed or self._control_fault:
            return 'control link unavailable; movement disabled'
        if not self.flying:
            return ('session ended: confirm grounded, then restart' if self._used_takeoff
                    else 'takeoff not requested; press T')
        if now < self._hold_until:
            return 'takeoff hold: neutral sticks for 3 seconds'
        if not self._video_ready(now):
            return 'waiting for fresh drone video'
        return None

    def manual_status(self):
        if self.dry_run:
            return 'PREVIEW: no flight commands'
        with self._lock:
            now = time.monotonic()
            reason = self._movement_block_reason(now)
            if self.flying and now < self._hold_until and not self._closed and not self._control_fault:
                return f'TAKEOFF HOLD: {self._hold_until - now:.1f}s remaining'
            return reason or 'READY: manual input enabled (hover unverified)'

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
        # This asks for takeoff; it cannot tell us whether the drone really lifted off.
        if self.dry_run:
            print("[drone] DRY RUN — would request takeoff")
            return
        with self._lock:
            now = time.monotonic()
            reason = self._ground_action_block(now)
            if reason:
                self.record_event("takeoff_blocked", reason=reason)
                print(f"[drone] Takeoff blocked: {reason}")
                return
            self._calib_left = 0
            self._cmd = dict(HOVER)
            self._cmd_time = now
            self._manual_cmd = dict(HOVER)
            self._manual_until = 0.0
            self._takeoff_at = now
            self._attempt_at = now
            self._attempt_count += 1
            self.record_event("takeoff_requested", attempt=self._attempt_count, airborne_verified=False)
            print("[drone] Takeoff requested")

    def land(self):
        """Legacy SPACE toggle request; physical landing is NOT guaranteed.

        self.flying records a request, not measured airborne state. This is
        unsafe after a failed/unconfirmed takeoff: the toggle can launch instead.
        Restricted tests are enabled; this protocol/state issue remains unresolved.
        """
        print("[drone] LAND")
        if self.dry_run:
            return
        with self._lock:
            self.record_event("land_requested")
            self._finish_attempt("land_requested")
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
            self._request_motor_stop_locked('motor_stop_requested')

    def _request_motor_stop_locked(self, reason):
        """Cancel other actions atomically. Caller holds _lock."""
        self.record_event("motor_stop_requested", reason=reason)
        self._finish_attempt(reason)
        self._takeoff_at = None
        self._calib_left = 0
        self.sticks.takeoff_flag = self.sticks.land_flag = False
        self.sticks.calibration_flag = self.sticks.flip_flag = False
        self.sticks.camera_tilt_state = 0
        self._cmd = dict(HOVER)
        self._manual_cmd = dict(HOVER)
        self._manual_until = 0.0
        self._stop_left = self.STOP_PACKETS
        self._stop_packets_submitted = 0
        self._used_takeoff = True
        self.flying = False

    def close(self):
        global _shared_control_link
        if self.dry_run or self._closed:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # User-selected exit policy: CUT MOTORS, never send the takeoff/land
            # toggle based on assumed flight state. An airborne drone will drop.
            self._request_motor_stop_locked('program_close')
        print('[drone] EXIT: EMERGENCY MOTOR STOP — airborne drone will drop')
        self._tick()  # Submit immediately, even if the background sender stalled.
        # Keep the shared video socket alive until this flush is finished.
        time.sleep(self.SHUTDOWN_S)
        self._running = False
        self._thread.join(timeout=1.0)
        self.record_event('shutdown_motor_stop',
                          locally_submitted_packets=self._stop_packets_submitted,
                          unsent_repetitions=self._stop_left, physical_stop_verified=False)
        if not self._stop_packets_submitted:
            print('[drone] WARNING: motor-stop could not be submitted; physical stop is unconfirmed')
        try:
            self.rc.stop()
        finally:
            self.record_event("session_end", attempts=self._attempt_count,
                              outcome="no_takeoff_requested" if not self._attempt_count else "see_attempts",
                              airborne_verified=False)
            self._recorder.close()
            if _shared_control_link is self:
                _shared_control_link = None


def inspect_video_link(seconds=15, directory='flight_logs'):
    """Collect existing video-link replies without creating a flight controller.

    Stream setup/ACKs still go out, with the upstream neutral stick template.
    This does not query new firmware commands or establish RC compatibility.
    """
    if not math.isfinite(seconds) or not 1 <= seconds <= 60:
        raise ValueError('Inspection duration must be between 1 and 60 seconds')
    if _shared_control_link is not None:
        raise RuntimeError('Close the active control link before inspecting video')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'link_inspect_{time.time_ns()}.json'
    report = dict(drone_ip=config.DRONE_IP, variant=config.WIFI_UAV_VARIANT,
                  duration_requested_s=seconds, decoded_frames=0, frame_sizes=[],
                  packet_prefix_counts={}, nonvideo_samples=[],
                  note='Video setup and neutral ACKs only; flight control compatibility unverified')
    seen = set()
    video = None
    started = time.monotonic()
    print('[inspect] Keep the drone grounded and close other drone-control programs.')
    print('[inspect] Reading video replies; no takeoff or movement requests.')
    try:
        video = DroneVideo()
        while time.monotonic() - started < seconds:
            for payload in video.adapter.get_packets():
                prefix = payload[:2].hex()
                counts = report['packet_prefix_counts']
                counts[prefix] = counts.get(prefix, 0) + 1
                # JPEG fragments are large and contain camera imagery. Keep
                # counts only; collect bounded, unique nonvideo replies.
                if payload[:2] != b'\x93\x01':
                    sample = payload[:1024].hex()
                    if sample not in seen and len(seen) < 64:
                        seen.add(sample)
                        report['nonvideo_samples'].append(dict(
                            elapsed_s=round(time.monotonic() - started, 3),
                            length=len(payload), packet_hex=sample,
                            truncated=len(payload) > 1024))
            ok, frame = video.read()
            if ok:
                report['decoded_frames'] += 1
                size = [frame.shape[1], frame.shape[0]]
                if size not in report['frame_sizes']:
                    report['frame_sizes'].append(size)
    except KeyboardInterrupt:
        report['interrupted'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if video is not None:
                video.release()
        finally:
            report['elapsed_s'] = round(time.monotonic() - started, 3)
            with path.open('x', encoding='utf-8') as output:
                json.dump(report, output, indent=2)
            print(f'[inspect] Saved: {path.resolve()}')
            print(f"[inspect] Decoded frames: {report['decoded_frames']}; "
                  f"nonvideo samples: {len(report['nonvideo_samples'])}")
            if 'error' in report:
                print(f"[inspect] {report['error']}")
            elif not report['nonvideo_samples']:
                print('[inspect] No nonvideo replies captured; control format remains unknown.')
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Grounded drone video-link diagnostics')
    parser.add_argument('--inspect', action='store_true', required=True,
                        help='Save video-link reply samples without creating a flight controller')
    parser.add_argument('--seconds', type=float, default=15, help='Capture duration, 1–60 seconds')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 60:
        parser.error('--seconds must be between 1 and 60')
    inspect_video_link(args.seconds)
