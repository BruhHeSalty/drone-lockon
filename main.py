"""Runs the whole thing: video -> vision -> controller -> HUD -> drone.

Controls (in the video window):
  click   pick the target color
  A       autopilot on/off (starts OFF)
  SPACE   emergency land (also turns autopilot off)
  R       start/stop recording the HUD to a video file
  Q       quit
"""
import time

import cv2

import config
from controller import Controller
from drone import DroneLink, VideoSource
import hud
from vision import ColorTracker

WINDOW = "drone-lockon"


def main():
    video = VideoSource()
    tracker = ColorTracker()
    controller = Controller()
    link = DroneLink()

    autopilot = False
    recorder = None
    last_frame = None
    fps, last_t = 0.0, time.time()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and last_frame is not None:
            tracker.set_color_from_pixel(last_frame, x, y)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)

    try:
        while True:
            frame = video.read()
            if frame is None:
                link.hover()
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
            last_frame = frame

            target = tracker.find(frame)
            command, state = controller.update(target)

            if autopilot:
                link.send(command)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
            last_t = now

            view = hud.draw(frame, target, state, command, autopilot, fps)
            if recorder is not None:
                recorder.write(view)
                cv2.circle(view, (view.shape[1] - 20, 20), 8, (0, 0, 255), -1)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                autopilot = not autopilot
                print(f"[main] autopilot {'ON' if autopilot else 'OFF'}")
                if not autopilot:
                    link.hover()
            elif key == ord(" "):
                autopilot = False
                link.land()
            elif key == ord("r"):
                if recorder is None:
                    h, w = view.shape[:2]
                    name = f"run_{int(time.time())}.mp4"
                    recorder = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
                    print(f"[main] recording to {name}")
                else:
                    recorder.release()
                    recorder = None
                    print("[main] recording stopped")
    finally:
        link.hover()
        if recorder is not None:
            recorder.release()
        video.release()
        link.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
