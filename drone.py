"""OWNER: Person 1 — talk to the drone (Contixo F19, "WiFi UAV" app).

Two jobs:
  1. VIDEO:   get frames from the drone's camera
  2. CONTROL: send flight commands (this is the hack)

Until both work, everything still runs:
  - video falls back to a recorded file or the webcam (set in config.py)
  - commands are only PRINTED (dry run), nothing is sent to the drone

Leads:
  - Video: WiFi UAV drones stream on UDP port 8800 after you send a start packet.
    Someone already decoded it: https://github.com/JadanPoll/DroneUAVHack
  - Control: first check the WiFi UAV app's on-screen joysticks fly the drone
    (remote OFF). If yes, decompile the Android APK with jadx and find the code
    that sends joystick packets. Similar drone done this way (E88):
    https://cie.uoc.ac.in/projects/hack_a_drone/
"""
import socket
import time

import cv2

import config


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
    """TODO (Person 1): live video from the drone over UDP.

    Must behave like cv2.VideoCapture: read() -> (ok, frame), release().
    Steps:
      1. Connect laptop to the drone's WiFi.
      2. Send the start packet to (DRONE_IP, VIDEO_PORT) — see DroneUAVHack repo.
      3. Receive UDP chunks, rebuild each JPEG frame, decode with cv2.imdecode.
    """

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        # TODO: send start packet, then assemble frames in read()

    def read(self):
        return False, None  # TODO

    def release(self):
        self.sock.close()


# -------------------------------------------------------------- CONTROL
class DroneLink:
    """Send Commands (see INTERFACES.md) to the drone.

    While CONTROL_PORT is None this is a dry run: it prints instead of sending.
    """

    def __init__(self):
        self.dry_run = config.CONTROL_PORT is None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_print = 0.0
        if self.dry_run:
            print("[drone] DRY RUN — commands are printed, not sent")

    def send(self, cmd):
        if self.dry_run:
            now = time.time()
            if now - self._last_print > 0.5:
                print(f"[drone] would send {cmd}")
                self._last_print = now
            return
        packet = self._encode(cmd)
        self.sock.sendto(packet, (config.DRONE_IP, config.CONTROL_PORT))

    def hover(self):
        self.send({"yaw": 0.0, "up": 0.0, "forward": 0.0, "right": 0.0})

    def land(self):
        """EMERGENCY. TODO: capture the app's land / stop button packet."""
        print("[drone] LAND")
        if not self.dry_run:
            pass  # TODO: send the real land packet here

    def _encode(self, cmd):
        """TODO (Person 1): turn -1..1 values into the drone's packet bytes.

        Cheap drones usually use 0–255 per stick with 128 = center:
            stick = int(128 + value * 127)
        plus a header byte, a checksum, and a footer. Find the exact format in
        the decompiled app.
        """
        raise NotImplementedError("control packet format not found yet")

    def close(self):
        self.sock.close()
