"""Runs the whole thing: video -> vision -> controller -> HUD -> drone.

Live first-flight check: python main.py --flight-test (tracking disabled).
Live person framing: python main.py --person (yaw only; click to select).
Offline person demo: python main.py --person --dry-run --webcam.

Controls (in the video window):
  click   pick the target color
  P       preflight only: calibrate gyro (drone flat, motors off)
  T       request takeoff directly (then hovers)
  A       autopilot on/off (starts OFF)
  flight-test only: W/S pitch, A/D roll, E/C throttle, J/L yaw, H hover
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
from drone import DroneLink, VideoSource, HOVER
import hud
import voice
from vision import ColorTracker, PersonTracker, AsyncPersonTracker

WINDOW = "drone-lockon"
MANUAL_KEYS = {
    ord("w"): ("forward", 1), ord("s"): ("forward", -1),
    ord("a"): ("right", -1), ord("d"): ("right", 1),
    ord("e"): ("up", 1), ord("c"): ("up", -1),
    ord("j"): ("yaw", -1), ord("l"): ("yaw", 1),
}


def main(flight_test=False, person=False, person_options=None):
    live = config.CONTROL_PORT is not None
    if flight_test and person:
        raise ValueError("Choose flight-test or person mode, not both")
    if live and (not (flight_test or person) or config.VIDEO_SOURCE != "drone"):
        raise RuntimeError(
            "Live control requires --flight-test or --person (drone video only). "
            "Use CONTROL_PORT = None for tracking demos."
        )
    video = link = recorder = raw_recorder = tracker = None
    autopilot = False
    last_frame = None
    last_camera = None
    fps, last_t = 0.0, time.monotonic()

    def on_click(event, x, y, flags, param):
        nonlocal autopilot
        if not flight_test and event == cv2.EVENT_LBUTTONDOWN and last_frame is not None:
            if person:
                autopilot = False
                link.hover()
            tracker.set_color_from_pixel(last_frame, x, y)

    try:
        # Load the model before opening a flight connection.
        tracker = AsyncPersonTracker(PersonTracker(allow_live_yaw=True, **(person_options or {}))) if person else ColorTracker()
        video = VideoSource()
        controller = Controller(person_mode=person)
        link = DroneLink(flight_test=flight_test)
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_click)
        if flight_test:
            print("[main] FLIGHT TEST: T = takeoff/hover, SPACE = land, K = motor stop, Q = land/quit")
            print("[main] Tracking/autopilot disabled. One takeoff per session; restart only after confirming landing.")
            print("[main] Manual taps: W/S forward/back, A/D left/right, E/C climb/descend, J/L turn, H hover")
        if person:
            print("[main] PERSON: click to select, T takeoff, A yaw tracking, H hover, SPACE land")
            print("[main] Turn only. Selection is remembered out of view; uncertain matches hover until reacquired.")

        while True:
            frame = video.read()
            fresh = frame is not None and video.fresh
            if frame is None:
                last_frame = None
                if person:
                    link.hover()
                # A skipped frame is not a lost detection. Keep the display
                # and let the sender's expiry/watchdog handle prolonged gaps.
                view = (last_camera.copy() if last_camera is not None
                        else np.zeros((480, 640, 3), dtype=np.uint8))
                cv2.setWindowTitle(WINDOW, WINDOW + " (video unavailable)")
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
            else:
                last_frame = frame if fresh else None
                last_camera = frame.copy()
                cv2.setWindowTitle(WINDOW, WINDOW if fresh else WINDOW + " (delayed video)")
                if video.kind == "drone":
                    if fresh:
                        link.note_frame()
                if flight_test or not fresh:
                    target = {"found": False, "x": 0.5, "y": 0.5, "w": 0.0, "h": 0.0, "conf": 0.0}
                else:
                    target = tracker.find(frame, frame_time=video.frame_time) if person else tracker.find(frame)
                command, state = controller.update(target) if fresh else (dict(HOVER), "SEARCHING")
                if flight_test:
                    command = link.manual_command()
                if person and not fresh:
                    link.hover()
                if not flight_test and fresh:
                    voice.announce(state, frame, target)
                if autopilot and fresh:
                    link.send(command)
                elif not flight_test and fresh:
                    link.hover()
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
                last_t = now
                view = hud.draw(frame, target, state, command, autopilot, fps)
                if person:
                    tracker.draw_candidates(view)
                    cv2.putText(view, "PERSON: click select | A turn only | H hover | SPACE land", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                if flight_test:
                    cv2.putText(view, "FLIGHT TEST: T hover / SPACE land / K stop", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(view, "W/S pitch  A/D roll  E/C throttle  J/L yaw  H hover", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.putText(view, f"Manual roll {command['right']:+.2f} (short taps)", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                if recorder is not None:
                    raw_recorder.write(frame)
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
            elif (flight_test or person) and key == ord("h"):
                autopilot = False
                link.hover()
            elif flight_test and key in MANUAL_KEYS:
                if fresh:
                    link.manual_nudge(*MANUAL_KEYS[key])
            elif key == ord("a"):
                if flight_test:
                    print("[main] Autopilot disabled in flight-test mode")
                elif autopilot or (fresh and (not person or target["found"])):
                    autopilot = not autopilot
                    print(f"[main] autopilot {'ON' if autopilot else 'OFF'}")
                    if not autopilot:
                        link.hover()
            elif key == ord("p"):
                link.preflight()
            elif key == ord("t"):
                autopilot = False
                link.hover()
                link.takeoff()
            elif key == ord("r") and frame is not None:
                if recorder is None:
                    h, w = frame.shape[:2]
                    name = f"run_{int(time.time())}.mp4"
                    recorder = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                    raw_name = name.replace('.mp4', '_raw.mp4')
                    raw_recorder = cv2.VideoWriter(raw_name, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                    if not recorder.isOpened() or not raw_recorder.isOpened():
                        recorder.release()
                        raw_recorder.release()
                        recorder = raw_recorder = None
                        print("[main] Recording failed: could not open video files")
                    else:
                        print(f"[main] recording HUD: {name}, clean frames: {raw_name} (15 fps playback)")
                else:
                    recorder.release()
                    raw_recorder.release()
                    recorder = raw_recorder = None
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
                if raw_recorder is not None:
                    raw_recorder.release()
                if person and tracker is not None:
                    tracker.close()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drone lock-on demos and isolated flight test")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--flight-test", action="store_true",
                        help="Live takeoff/manual/landing test; disables tracking and autopilot")
    modes.add_argument("--person", action="store_true", help="Selected-person tracking with yaw only")
    parser.add_argument("--dry-run", action="store_true", help="Disable all outgoing flight commands")
    parser.add_argument("--webcam", action="store_true", help="Use the laptop camera (requires --dry-run)")
    parser.add_argument("--video", help="Replay a recording (requires --dry-run)")
    parser.add_argument("--model", default="yolo11n.pt", help="Person model name or local weights path")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA device index")
    parser.add_argument("--imgsz", type=int, default=640, help="Person inference size, multiple of 32 (320-1280)")
    parser.add_argument("--tracker", choices=("botsort", "bytetrack"), default="botsort")
    args = parser.parse_args()
    if args.webcam and not args.dry_run:
        parser.error("--webcam requires --dry-run")
    if args.video and (not args.dry_run or args.webcam):
        parser.error("--video requires --dry-run and cannot be combined with --webcam")
    if args.dry_run:
        config.CONTROL_PORT = None
    if args.webcam:
        config.VIDEO_SOURCE = "webcam"
    if args.video:
        config.VIDEO_SOURCE, config.VIDEO_FILE = "file", args.video
    main(flight_test=args.flight_test, person=args.person, person_options=dict(
        model=args.model, device=args.device, imgsz=args.imgsz, tracker=args.tracker + ".yaml"))
