"""Runs the whole thing: video -> vision -> controller -> HUD -> drone.

Live first-flight check: python main.py --flight-test (tracking disabled).
Manual control + tracking display: python main.py --manual.
Live person framing: python main.py --person (yaw only; click to select).
Selected-person following: python main.py --follow (yaw + limited forward).
Offline person demo: python main.py --person --dry-run --webcam.

Controls (in the video window):
  click   select target (in --follow, also requests following after confirmation)
  P       preflight only: calibrate gyro (drone flat, motors off)
  T       request takeoff directly (then neutral sticks)
  A       autopilot on/off (starts OFF)
  flight-test only: W/S pitch, A/D roll, E/C throttle, J/L yaw, H neutral
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
from vision import ColorTracker, PersonTracker, AsyncPersonTracker, NOT_FOUND

WINDOW = "drone-lockon"
MANUAL_KEYS = {
    ord("w"): ("forward", 1), ord("s"): ("forward", -1),
    ord("a"): ("right", -1), ord("d"): ("right", 1),
    ord("e"): ("up", 1), ord("c"): ("up", -1),
    ord("j"): ("yaw", -1), ord("l"): ("yaw", 1),
}


class ManualWindow:
    """Tk supplies key-up and modifier events that HighGUI does not expose."""
    ACTIONS = {'t': ord('t'), 'l': ord(' '), 'space': ord(' '),
               'k': ord('k'), 'h': ord('h'), 'escape': 27, 'r': ord('r'),
               '1': ord('1'), '2': ord('2'), '3': ord('3')}

    def __init__(self, on_click):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title('Manual control — tracking display only')
        self.root.resizable(False, False)
        self.label = tk.Label(self.root, borderwidth=0, takefocus=True)
        self.label.pack()
        self.held = set()
        self.actions = []
        self.inhibited = False
        self.closed = False
        self.root.bind('<KeyPress>', self._press)
        self.root.bind('<KeyRelease>', self._release)
        self.root.bind('<FocusOut>', self._focus_out)
        self.label.bind('<Button-1>', lambda e: on_click(cv2.EVENT_LBUTTONDOWN, e.x, e.y, None, None))
        self.root.protocol('WM_DELETE_WINDOW', self._quit)
        self.label.focus_set()

    def _press(self, event):
        key = event.keysym.lower()
        if key in self.held:
            return
        self.held.add(key)
        if key in self.ACTIONS:
            action = self.ACTIONS[key]
            self.actions.append(action)
            if action in (ord('t'), ord(' '), ord('k'), ord('h'), 27):
                self.inhibited = True

    def _release(self, event):
        self.held.discard(event.keysym.lower())
        if not self.held:
            self.inhibited = False

    def _focus_out(self, event=None):
        self.held.clear()
        self.actions = [action for action in self.actions if action in (ord('k'), ord(' '), 27)]
        self.actions.append(ord('h'))
        self.inhibited = False

    def _quit(self):
        self.closed = True
        self.held.clear()

    def axes(self):
        if self.closed or self.inhibited or self.root.focus_displayof() is None:
            return dict(HOVER)
        keys = self.held
        return dict(forward=int('w' in keys) - int('s' in keys),
                    right=int('d' in keys) - int('a' in keys),
                    yaw=int('e' in keys) - int('q' in keys),
                    up=int(bool(keys & {'shift_l', 'shift_r', 'up'}))
                    - int(bool(keys & {'control_l', 'control_r', 'down'})))

    def input_status(self):
        reason = ('window closed' if self.closed else
                  'focus this window' if self.root.focus_displayof() is None else
                  'release all keys after T/H/land/stop' if self.inhibited else None)
        return {'keys': sorted(self.held), 'blocked_reason': reason}

    def show(self, frame):
        ok, ppm = cv2.imencode('.ppm', frame)
        if not ok:
            raise RuntimeError('Could not render manual-control frame')
        self.photo = self.tk.PhotoImage(data=ppm.tobytes(), format='PPM')
        self.label.configure(image=self.photo)
        self.root.update()

    def key(self):
        if self.closed:
            return 27
        # Emergency actions take priority over queued takeoff or recording.
        for priority in (ord('k'), ord(' '), 27, ord('h')):
            if priority in self.actions:
                self.actions.clear()
                return priority
        return self.actions.pop(0) if self.actions else -1

    def close(self):
        self.root.destroy()


def main(flight_test=False, person=False, person_options=None, follow=False, manual=False):
    if manual and (flight_test or follow):
        raise ValueError('Choose one control mode')
    person = person or follow or manual
    manual_controls = flight_test or manual
    live = config.CONTROL_PORT is not None
    if flight_test and person:
        raise ValueError("Choose flight-test or person mode, not both")
    if follow and live:
        raise RuntimeError(
            "Live follow is disabled after reported uncontrolled motion/altitude loss. "
            "Neutral sticks have not been verified to hold this drone. "
            "Use --follow --dry-run for tracking preview; do not repeat the live follow launch."
        )
    if live and (not (flight_test or person) or config.VIDEO_SOURCE != "drone"):
        raise RuntimeError(
            "Live control requires --flight-test, --person or --follow (drone video only). "
            "Use CONTROL_PORT = None for tracking demos."
        )
    video = link = recorder = raw_recorder = tracker = ui = None
    autopilot = False
    follow_pending = False
    follow_wait_reason = None
    last_tracking_status = None
    last_frame = None
    last_camera = None
    target = dict(NOT_FOUND)
    fps, last_t = 0.0, time.monotonic()
    recorded_frames = 0
    last_keyboard_status = None
    manual_hint = 'Focus the window; T requests takeoff'

    def on_click(event, x, y, flags, param):
        nonlocal autopilot, target, follow_pending, follow_wait_reason
        if not flight_test and event == cv2.EVENT_LBUTTONDOWN and last_frame is not None:
            if person:
                autopilot = False
                follow_pending = False
                follow_wait_reason = None
                controller.disarm_follow()
                target = dict(NOT_FOUND)  # A click must invalidate this loop's old lock too.
                link.hover()
            accepted = tracker.set_color_from_pixel(last_frame, x, y)
            if person:
                link.record_event("selection_click", accepted=accepted is True, x=x, y=y)
            if follow and accepted is True:
                follow_pending = True
                print("[main] Follow requested by click; holding until takeoff hold and person confirmation finish. H cancels.")

    try:
        # Load the model before opening a flight connection.
        tracker = AsyncPersonTracker(PersonTracker(allow_live_yaw=True, **(person_options or {}))) if person else ColorTracker()
        video = VideoSource()
        controller = Controller(person_mode=person, follow_mode=follow)
        if manual:
            ui = ManualWindow(on_click)
        link = DroneLink(flight_test=manual_controls)
        if not manual:
            cv2.namedWindow(WINDOW)
            cv2.setMouseCallback(WINDOW, on_click)
        if manual:
            print('[main] MANUAL: W/S forward/back, A/D strafe, Q/E rotate; Shift/Ctrl or arrows up/down')
            print('[main] T takeoff; L/SPACE land; K stop motors; H neutral; Esc land/quit. Tracking cannot steer.')
            print('[main] Outcome labels: 1 steady, 2 unstable, 3 crash (operator reports only; no automatic learning)')
            print('[main] After T, release the keys and wait for the 3-second TAKEOFF HOLD countdown to reach READY.')
            print('[main] Person confirmation is visual only; manual movement does not require a target lock.')
        if flight_test:
            print("[main] FLIGHT TEST: T = takeoff/hover, SPACE = land, K = motor stop, Q = land/quit")
            print("[main] Tracking/autopilot disabled. One takeoff per session; restart only after confirming landing.")
            print("[main] Manual taps: W/S forward/back, A/D left/right, E/C climb/descend, J/L turn, H neutral")
        if person and not manual:
            if follow:
                print("[main] FOLLOW: T takeoff/neutral hold, then click person to start following; H neutral, SPACE land")
                print("[main] Image-size following only: no measured distance or obstacle avoidance. No reverse, strafe or climb.")
            else:
                print("[main] PERSON: click to select, T takeoff, A yaw tracking, H neutral, SPACE land")
                print("[main] Turn only. Selection is remembered out of view; uncertain matches hover until reacquired.")
            if not live:
                print("[main] DRY RUN: camera/selection preview only; no flight commands")

        while True:
            frame = video.read()
            fresh = frame is not None and video.fresh
            # A 50 ms read timeout can occur between healthy camera frames.
            # Held manual input uses the sender's existing 300 ms freshness
            # deadline; detector-driven steering still requires a fresh frame.
            manual_video_ready = fresh or (manual and live and frame is None and link.video_ready())
            if follow and not fresh:
                controller.update(NOT_FOUND)  # A video gap resets forward confirmation immediately.
            if frame is None:
                last_frame = None
                if person and not (manual and manual_video_ready):
                    link.hover()
                # Keep rendering the HUD on the last camera image. Returning
                # the bare image here made every video gap erase the overlay.
                # No previous target or command is reused for flight control.
                display_frame = (last_camera if last_camera is not None
                                 else np.zeros((480, 640, 3), dtype=np.uint8))
                target, command = dict(NOT_FOUND), dict(HOVER)
                if manual and manual_video_ready:
                    command = link.manual_command()
                state = "LOST" if person else "SEARCHING"
                if not manual:
                    cv2.setWindowTitle(WINDOW, WINDOW + " (video unavailable)")
            else:
                last_frame = frame if fresh else None
                last_camera = frame.copy()
                if not manual:
                    cv2.setWindowTitle(WINDOW, WINDOW if fresh else WINDOW + " (delayed video)")
                if video.kind == "drone":
                    if fresh:
                        link.note_frame()
                if flight_test or not fresh:
                    target = {"found": False, "x": 0.5, "y": 0.5, "w": 0.0, "h": 0.0, "conf": 0.0}
                else:
                    target = tracker.find(frame, frame_time=video.frame_time) if person else tracker.find(frame)
                if follow_pending:
                    reason = (link.tracking_block_reason() if live else None)
                    reason = reason or (controller.follow_block_reason(target) if fresh else "waiting for fresh drone video")
                    if reason is None and controller.arm_follow(target):
                        follow_pending = False
                        follow_wait_reason = None
                        autopilot = True
                        print("[main] Follow ON: starting gap recorded; forward follows when the person moves farther away")
                    elif reason != follow_wait_reason:
                        follow_wait_reason = reason
                        print(f"[main] Follow waiting: {reason}; H cancels")
                command, state = controller.update(target) if fresh else (dict(HOVER), "LOST" if person else "SEARCHING")
                if manual_controls:
                    command = link.manual_command()
                if person and not fresh:
                    link.hover()
                if not flight_test and fresh:
                    if manual:
                        voice.announce(state, frame, target, visual_only=True)
                    else:
                        voice.announce(state, frame, target)
                if autopilot and fresh:
                    link.send(command)
                elif not manual_controls and fresh:
                    link.hover()
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
                last_t = now
                display_frame = frame

            # One render path for both received frames and video gaps.
            if not manual_controls and (not autopilot or (live and link.tracking_block_reason() is not None)):
                command = dict(HOVER)  # Do not show movement wishes while controls are held neutral.
            tracking_status = (state, bool(target['found']), bool(fresh), autopilot, follow_pending)
            if tracking_status != last_tracking_status:
                link.record_event("tracking_state", state=state, target=dict(target), video_fresh=bool(fresh),
                                  autopilot=autopilot, follow_pending=follow_pending)
                last_tracking_status = tracking_status
            view = hud.draw(display_frame, target, state, command, autopilot, fps)
            if person:
                tracker.draw_candidates(view, video_fresh=fresh)
                mode_hint = ("PERSON: click select | A turn only | H neutral | SPACE land" if live
                             else "DRY RUN: no flight commands | click a person box")
                if follow:
                    mode_hint = ("FOLLOW: T takeoff | click to follow | H neutral | SPACE land" if live
                                 else "FOLLOW PREVIEW: click person to follow | H cancels")
                    if follow_pending:
                        message = "HOLD: " + (follow_wait_reason or "waiting for confirmation") + " | H cancels"
                        width = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)[0][0]
                        scale = min(.4, .4 * (view.shape[1] - 20) / max(1, width))
                        cv2.putText(view, message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                    scale, (0, 255, 255), 1)
                if manual:
                    mode_hint = 'MANUAL: W/S pitch A/D strafe Q/E yaw | Shift/Ctrl altitude'
                hint_width = cv2.getTextSize(mode_hint, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)[0][0]
                hint_scale = min(.5, .5 * (view.shape[1] - 20) / max(1, hint_width))
                cv2.putText(view, mode_hint, (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, hint_scale, (0, 255, 255), 2)
            if not fresh:
                video_hint = ('WAITING FOR NEXT FRAME' if manual and manual_video_ready else
                              'VIDEO UNAVAILABLE' if frame is None else 'VIDEO DELAYED')
                cv2.putText(view, video_hint, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            if flight_test:
                cv2.putText(view, "FLIGHT TEST: T hover / SPACE land / K stop", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(view, "W/S pitch  A/D roll  E/C throttle  J/L yaw  H neutral", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.putText(view, f"Manual roll {command['right']:+.2f} (short taps)", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if recorder is not None:
                if frame is not None:
                    raw_recorder.write(frame)
                    link.record_event('video_frame', index=recorded_frames, source_frame_time=video.frame_time)
                    recorded_frames += 1
                    if not manual:
                        recorder.write(view)
                cv2.circle(view, (view.shape[1] - 20, 20), 8, (0, 0, 255), -1)
            if manual:
                # Replace the autopilot shortcut legend; Q now rotates left.
                if fresh:
                    feedback = (f"INPUT pitch {command['forward']:+.2f} roll {command['right']:+.2f} "
                                f"yaw {command['yaw']:+.2f} up {command['up']:+.2f}")
                    cv2.putText(view, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                .4, (255, 255, 255), 1)
                view[-88:] = 0
                status_width = cv2.getTextSize(manual_hint, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)[0][0]
                scale = min(.4, .4 * (view.shape[1] - 20) / max(status_width, 1))
                cv2.putText(view, manual_hint, (10, view.shape[0] - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 1)
                for y, text in ((view.shape[0] - 48, 'MANUAL ONLY | T takeoff | L/SPACE land | K motor stop'),
                                (view.shape[0] - 30, 'H neutral | Esc quit | R record | arrows also control altitude'),
                                (view.shape[0] - 12, 'Label flight: 1 steady / 2 unstable / 3 crash')):
                    cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                if recorder is not None and frame is not None:
                    recorder.write(view)
                ui.show(view)
                key = ui.key()
                keyboard_status = ui.input_status()
                if keyboard_status != last_keyboard_status:
                    link.record_event('keyboard_state', **keyboard_status)
                    last_keyboard_status = keyboard_status
                axes = ui.axes() if manual_video_ready and key == -1 else dict(HOVER)
                link.manual_input(axes)
                reason = keyboard_status['blocked_reason'] or link.manual_status()
                if not manual_video_ready:
                    reason = 'VIDEO UNAVAILABLE: movement disabled'
                manual_hint = 'Keys: ' + (' '.join(keyboard_status['keys']) or 'none') + ' | ' + reason
            else:
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
            if ord('A') <= key <= ord('Z'):
                key += ord('a') - ord('A')

            # Emergency keys remain available when the video is missing.
            if (manual and key == 27) or (not manual and
                    (key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1)):
                break
            elif key == ord(" "):
                autopilot = False
                follow_pending = False
                controller.disarm_follow()
                link.land()
            elif key == ord("k"):
                autopilot = False
                follow_pending = False
                controller.disarm_follow()
                link.kill()
            elif (flight_test or person) and key == ord("h"):
                autopilot = False
                follow_pending = False
                controller.disarm_follow()
                link.hover()
                link.record_event("hover_key", position_hold_verified=False)
                print("[main] H received: tracking OFF, sticks centered; position/altitude hold is unverified")
            elif flight_test and key in MANUAL_KEYS:
                if fresh:
                    link.manual_nudge(*MANUAL_KEYS[key])
            elif key == ord("a"):
                if manual_controls:
                    print("[main] Autopilot disabled in flight-test mode")
                elif follow_pending:
                    follow_pending = False
                    controller.disarm_follow()
                    link.hover()
                    print("[main] Follow request cancelled")
                elif autopilot or (fresh and (not person or target["found"])):
                    if follow and not autopilot:
                        reason = (link.tracking_block_reason() if live else None) or controller.follow_block_reason(target)
                        if reason is not None:
                            print(f"[main] Follow not enabled: {reason}")
                            continue
                        controller.arm_follow(target)
                    autopilot = not autopilot
                    print(f"[main] autopilot {'ON' if autopilot else 'OFF'}")
                    if not autopilot:
                        controller.disarm_follow()
                        link.hover()
                else:
                    print("[main] Tracking unavailable: need fresh video and a confirmed selected person")
            elif key == ord("p"):
                link.preflight()
            elif manual and key in (ord('1'), ord('2'), ord('3')):
                label = {ord('1'): 'steady', ord('2'): 'unstable', ord('3'): 'crash'}[key]
                link.record_event('operator_flight_label', label=label, source='operator', airborne_verified=False)
                print(f'[main] Flight labelled {label}; this does not adjust flight controls')
            elif key == ord("t"):
                autopilot = False
                follow_pending = False
                controller.disarm_follow()
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
                        recorded_frames = 0
                        link.record_event('recording_started', hud_file=name, raw_file=raw_name, playback_fps=15)
                        print(f"[main] recording HUD: {name}, clean frames: {raw_name} (15 fps playback)")
                else:
                    recorder.release()
                    raw_recorder.release()
                    recorder = raw_recorder = None
                    link.record_event('recording_stopped', frames=recorded_frames)
                    print("[main] recording stopped")
    except Exception as exc:
        if link is not None:
            link.record_event("program_error", error=str(exc))
        raise
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
                if ui is not None:
                    ui.close()
                else:
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drone lock-on demos and isolated flight test")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--manual', action='store_true', help='Held-key manual control with person tracking display only')
    modes.add_argument("--flight-test", action="store_true",
                        help="Live takeoff/manual/landing test; disables tracking and autopilot")
    modes.add_argument("--person", action="store_true", help="Selected-person tracking with yaw only")
    modes.add_argument("--follow", action="store_true", help="Follow preview only: requires --dry-run while hover failure is investigated")
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
    main(flight_test=args.flight_test, person=args.person, follow=args.follow, manual=args.manual, person_options=dict(
        model=args.model, device=args.device, imgsz=args.imgsz, tracker=args.tracker + ".yaml"))
