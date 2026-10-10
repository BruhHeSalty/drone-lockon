"""Runs the whole thing: video -> vision -> controller -> HUD -> drone.

Live first-flight check: python main.py --flight-test (tracking disabled).
Tracking demos require config.CONTROL_PORT = None.

Controls (in the video window):
  click   pick the target color
  P       preflight only: calibrate gyro (drone flat, motors off)
  T       calibrate gyro, then take off (then hovers)
  A       autopilot on/off (starts OFF)
  SPACE   emergency land (also turns autopilot off)
  K       KILL motors — drone drops (also turns autopilot off)
  R       start/stop recording the HUD to a video file
  Q       quit
"""
import time

import cv2
import numpy as np

import config
from controller import Controller
from drone import DroneLink, VideoSource
import hud
import voice
from vision import ColorTracker

WINDOW = "drone-lockon"


def main(flight_test=False):
    live = config.CONTROL_PORT is not None
    if live and (not flight_test or config.VIDEO_SOURCE != "drone"):
        raise RuntimeError(
            "Live control requires: python main.py --flight-test (drone video only). "
            "Use CONTROL_PORT = None for tracking demos."
        )
    video = link = recorder = None
    autopilot = False
    last_frame = None
    fps, last_t = 0.0, time.monotonic()

    def on_click(event, x, y, flags, param):
        if not flight_test and event == cv2.EVENT_LBUTTONDOWN and last_frame is not None:
            tracker.set_color_from_pixel(last_frame, x, y)

    try:
        video = VideoSource()
        tracker = ColorTracker()
        controller = Controller()
        link = DroneLink(flight_test=flight_test)
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_click)
        if flight_test:
            print("[main] FLIGHT TEST: T = takeoff/hover, SPACE = land, K = motor stop, Q = land/quit")
            print("[main] Tracking/autopilot disabled. One takeoff per session; restart only after confirming landing.")

        while True:
            frame = video.read()
            if frame is None:
                last_frame = None
                autopilot = False
                link.video_lost()
                link.hover()
                view = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(view, "NO FRESH VIDEO - HOVER REQUESTED", (15, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(view, "SPACE: land   K: motor stop   Q: quit", (15, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
            else:
                last_frame = frame
                if video.kind == "drone":
                    link.note_frame()
                if flight_test:
                    target = {"found": False, "x": 0.5, "y": 0.5, "w": 0.0, "h": 0.0, "conf": 0.0}
                else:
                    target = tracker.find(frame)
                command, state = controller.update(target)
                if not flight_test:
                    voice.announce(state, frame, target)
                if autopilot:
                    link.send(command)
                else:
                    link.hover()
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
                last_t = now
                view = hud.draw(frame, target, state, command, autopilot, fps)
                if flight_test:
                    cv2.putText(view, "FLIGHT TEST: T hover / SPACE land / K stop", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if recorder is not None:
                    recorder.write(view)
                    cv2.circle(view, (view.shape[1] - 20, 20), 8, (0, 0, 255), -1)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF

            # Emergency keys remain available when the video is missing.
            if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord(" "):
                autopilot = False
                link.land()
            elif key == ord("k"):
                autopilot = False
                link.kill()
            elif key == ord("a"):
                if flight_test:
                    print("[main] Autopilot disabled in flight-test mode")
                elif frame is not None:
                    autopilot = not autopilot
                    print(f"[main] autopilot {'ON' if autopilot else 'OFF'}")
                    if not autopilot:
                        link.hover()
            elif key == ord("p") and frame is not None:
                link.preflight()
            elif key == ord("t") and frame is not None:
                autopilot = False
                link.hover()
                link.takeoff()
            elif key == ord("r") and frame is not None:
                if recorder is None:
                    h, w = frame.shape[:2]
                    name = f"run_{int(time.time())}.mp4"
                    recorder = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                    print(f"[main] recording to {name}")
                else:
                    recorder.release()
                    recorder = None
                    print("[main] recording stopped")
    finally:
        # LAND must be flushed before video.release() closes its shared socket.
        try:
            if link is not None:
                link.close()
        finally:
            try:
                if video is not None:
                    video.release()
            finally:
                if recorder is not None:
                    recorder.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drone lock-on demos and isolated flight test")
    parser.add_argument("--flight-test", action="store_true",
                        help="Live takeoff/hover/landing test; disables tracking and autopilot")
    args = parser.parse_args()
    main(flight_test=args.flight_test)
