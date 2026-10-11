# drone-lockon

Camera tracking and a lock-on HUD for the Contixo F19 (WiFi UAV app).
The laptop processes video and can send flight commands. First verify video
with commands disabled. Never fly at people.

## Windows: get the drone video working first

Use **PowerShell**. Install Git and Python 3.10+ if missing. Run downloads while
connected to normal internet, before switching to the drone Wi-Fi.

### 1. Get the current project

For a new checkout:

```powershell
git clone https://github.com/BruhHeSalty/drone-lockon.git
cd drone-lockon
```

If you already have it, open PowerShell in that project folder and run:

```powershell
git pull --ff-only
```

If Git reports local changes or a divergent branch, keep your changes and
resolve that first; do not reset your work just to update.

### 2. Install the basic video dependencies

Create your own Windows environment. Do not copy a Mac's `.venv` directory.
These commands call Python directly, so PowerShell activation permissions
are not needed.

```powershell
py -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-win\Scripts\python.exe test_pipeline.py
```

Expected: `All checks passed`. This checks vision/controller logic, not drone
connectivity. If `py` is not recognized, install Python with its launcher,
reopen PowerShell, and retry. No YOLO model or API key is needed for drone video.

### 3. Download the separate drone protocol code

```powershell
git clone https://github.com/marshallrichards/turbodrone.git "$env:USERPROFILE/turbodrone"
```

If that checkout already exists, update it instead:

```powershell
git -C "$env:USERPROFILE/turbodrone" pull --ff-only
```

Our `config.py` expects `TURBODRONE_PATH = "~/turbodrone/backend"`, which expands
to your Windows home directory. Verify the actual configured path and imports:

```powershell
.\.venv-win\Scripts\python.exe -c "import os, config; p = os.path.expanduser(config.TURBODRONE_PATH); print(p); print('Video adapter exists:', os.path.isfile(os.path.join(p, 'protocols', 'wifi_uav_video_protocol.py')))"
.\.venv-win\Scripts\python.exe -c "from drone import _load_turbodrone; _load_turbodrone(); from protocols.wifi_uav_video_protocol import WifiUavVideoProtocolAdapter; print('Drone video imports OK')"
```

Expected: `Video adapter exists: True` and `Drone video imports OK`. If you
cloned elsewhere, point `TURBODRONE_PATH` at that checkout's **backend** folder
(use forward slashes in the Python string). Tell the team if you change config.
Only the protocol modules are used; you do not need Turbodrone's web app running.

### 4. Connect to the drone

1. Power on the drone on a flat surface, with props removed for this first test.
2. Connect the Windows laptop to the drone's Wi-Fi and stay connected even
   when Windows says **No internet**. Keep automatic IP/DHCP settings.
3. Close the phone's WiFi UAV app and disconnect the phone and other laptops
   from the drone while testing. Close other drone-control programs.
4. Run `ipconfig` and check the Wi-Fi adapter. This project's configured drone
   IP is `192.168.169.1`; a typical matching laptop address is `192.168.169.x`.
   An address beginning `169.254` indicates DHCP did not provide an address.
   Reconnect/restart the drone in that case. Compare the actual network with
   `config.DRONE_IP` rather than guessing a static address.

### 5. Launch drone video with flight commands disabled

Paste this entire line in PowerShell in the project folder:

```powershell
.\.venv-win\Scripts\python.exe -c "import config; config.VIDEO_SOURCE = 'drone'; config.CONTROL_PORT = None; import main; main.main()"
```

Look for `[drone] DRY RUN` and a window showing the drone camera. It still sends
video-protocol requests, but does not send flight-control commands. This launch
does not change saved settings. Press **Q** in the video window to quit.

`python vision.py --person` uses the **laptop webcam**, not the drone camera.
Plain `python main.py` refuses live control. Use the dry-run command above while
troubleshooting video; live control requires `--flight-test` or explicit yaw-only
`--person` mode described below.

### If video still does not appear

| Symptom | Next check |
|---|---|
| `No module named cv2` / `numpy` | Install requirements with the same `.venv-win\Scripts\python.exe` used to launch. |
| `Turbodrone not found` / `No module named protocols` | Repeat step 3; the configured path must end in `backend`. |
| `No module named voice` | Pull the complete current repo, including `voice.py`; do not copy only `main.py`. |
| `[voice] elevenlabs not installed` / missing API key | Voice is optional. These messages do not prevent video. |
| Repeated `no video — reconnecting` or a blank feed | Check Wi-Fi/IP, other connected apps/devices, firewall, and the protocol settings below. |
| Socket permission/`WinError 10013` | Check Windows Firewall and any managed security software; close other drone apps. |
| Webcam appears | Use the step 5 drone-video command, not `--person`. |

**Windows Firewall:** open Windows Security → Firewall & network protection →
Allow an app through firewall. Allow the Python executable used by this project
for the active drone network's profile. A drone network may be classified as
Public, so a Private-only exception may not apply. Keep the firewall enabled;
use an app-specific exception. If unsure which interpreter is running:

```powershell
.\.venv-win\Scripts\python.exe -c "import sys; print(sys.executable); print(sys._base_executable)"
```

Check VPN/security software if it blocks local-network traffic. For a managed
laptop, ask its administrator about the blocked app rather than disabling policy.

Current project protocol settings are `DRONE_IP = "192.168.169.1"`,
`VIDEO_PORT = 8800`, and `WIFI_UAV_VARIANT = "uav"`. Our local Turbodrone adapter
probes UDP 8800/8801 in this mode and receives on an OS-assigned local port.
A TCP port test does not verify this UDP video stream. Do not assume an inbound
rule for local port 8800 alone covers it.

To capture a more useful error log (still with flight controls disabled):

```powershell
.\.venv-win\Scripts\python.exe -u -c "import logging; logging.basicConfig(level=logging.DEBUG); import config; config.VIDEO_SOURCE = 'drone'; config.CONTROL_PORT = None; import main; main.main()" 2>&1 | Tee-Object -FilePath "$env:TEMP/drone-video-log.txt"
```

Share the error lines, Wi-Fi IPv4 address, configured drone IP, and whether
video works on the same drone with a phone or Mac. Test the phone separately,
then close its app and disconnect it before retrying Windows. Windows hardware
connectivity has not yet been verified by this project session.

References: [Python virtual environments](https://docs.python.org/3/library/venv.html),
[Microsoft firewall app exceptions](https://support.microsoft.com/en-us/windows/security/firewall/risks-of-allowing-apps-through-windows-firewall),
[Turbodrone upstream](https://github.com/marshallrichards/turbodrone).

## Mac setup and drone video

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python test_pipeline.py
git clone https://github.com/marshallrichards/turbodrone ~/turbodrone
```

For an existing Turbodrone checkout, use `git -C ~/turbodrone pull --ff-only`
instead of cloning again. Join the drone Wi-Fi, then launch video only:

```bash
python -c 'import config; config.VIDEO_SOURCE = "drone"; config.CONTROL_PORT = None; import main; main.main()'
```

## First computer-controlled takeoff/manual/landing test

This is a restricted flight-test mode, not autonomous target following. Confirm
the video-only setup above works first. Run these offline checks before connecting:

Windows PowerShell:

```powershell
.\.venv-win\Scripts\python.exe test_pipeline.py
.\.venv-win\Scripts\python.exe test_flight_safety.py
```

Mac (with `.venv` active):

```bash
python test_pipeline.py
python test_flight_safety.py
```

Then connect to the drone Wi-Fi. With the drone on a flat surface and props
removed, launch:

Windows PowerShell:

```powershell
.\.venv-win\Scripts\python.exe main.py --flight-test
```

Mac:

```bash
python main.py --flight-test
```

This uses the saved drone address and control port; `CONTROL_PORT` must be a
live port (currently 8800). If the console says `DRY RUN`, flight commands are
disabled. Do not change this while the program is running.

- Wait for the live camera feed to appear. Focus the video window. A recent frame
  is required, but there is no one-second continuous-video wait.
- `T` requests takeoff directly, without automatic calibration or a preflight delay.
  Verify props-off responses first. The program commands neutral sticks for
  three seconds after takeoff, then permits the manual controls below. It does
  not steer toward a target; autopilot is disabled in this mode.
- `P` remains available for optional gyro calibration on a flat surface with motors off.
- `SPACE` cancels a pending takeoff or requests landing if this program has
  requested takeoff. `Q` or closing the window also cancels/requests landing.
- `K` requests immediate motor stop; an airborne drone will drop. It is not a
  normal landing control.
- After the three-second takeoff settling period, use **short keyboard taps**
  with the video window focused:

  | Keys | Manual control |
  |---|---|
  | `W` / `S` | Pitch: forward / backward |
  | `A` / `D` | Roll: left / right |
  | `E` / `C` | Throttle: climb / descend |
  | `J` / `L` | Yaw: turn left / right |
  | `H` | Immediately request neutral sticks / hover |

  Each key event requests one axis for at most 0.2 seconds, then automatically
  returns to neutral. A new direction replaces the previous direction; simultaneous
  multi-axis keyboard flight is not supported. Holding a key relies on OS key
  repeat and may produce gaps, so start with taps. Pitch/roll are capped at 12%
  of stick range and throttle/yaw at 15%, or the corresponding lower config cap.
  These are relative stick inputs, not a commanded angle, altitude or speed.
  Missing/stale video, landing, motor stop, and shutdown inhibit manual movement.
  Verify axis directions on the actual drone with props off before airborne use.
- One takeoff is allowed per session. After landing or a motor stop, physically
  confirm the drone is grounded before restarting. Do not start this program
  while the drone is already airborne under another controller.
- A single missed frame does not issue hover or cancel manual input. Late frames
  are displayed too; the window title indicates delayed or unavailable video.
  Only a recently received frame refreshes the control watchdog. After 0.3 seconds
  without fresh video, the sender cancels pending takeoff and uses neutral commands.
  Manual inputs independently expire after 0.2 seconds without another key event.
  An explicit lost-target detection in tracking demos immediately produces neutral commands.
  No forward movement or turning search is commanded while a target is missing.

After the props-off checks, a supervised low hover in a clear area away from
people is a separate hardware test. Neutral sticks are not GPS position hold:
the drone can drift. Keep the normal controller available, but do not assume
it can override Wi-Fi commands until that handoff has been verified props-off.

The protocol exposes a takeoff/land **toggle**, not a confirmed landing state.
The program sends a landing toggle before closing the shared video socket and
preserves it after a known local socket-send failure. It does not repeat a toggle
after local submission because that could trigger the opposite action. UDP
submission does not prove receipt or landing; a total Wi-Fi outage can prevent
all commands from reaching the drone. Verify physical response and stop testing
if it differs from the requested action. Offline tests do not certify flight safety.

## Webcam tests without a drone

For the existing color tracker, use the same dry-run command as above with
`config.VIDEO_SOURCE = "webcam"`. Click a distinctly colored object to select it.
On Mac, allow camera access for the terminal/IDE running Python when prompted.

For **person tracking**, install the optional packages while online:

Windows PowerShell:

```powershell
.\.venv-win\Scripts\python.exe -m pip install ultralytics lap
.\.venv-win\Scripts\python.exe vision.py --person
```

Mac, with `.venv` active:

```bash
python -m pip install ultralytics lap
python vision.py --person
```

The first person-mode launch downloads `yolo11n.pt`; start it once while online.
It runs locally after download and needs no API key. If the wrong webcam opens,
add `--camera 1`. Person mode always uses dry-run controls.

A single visible person is selected automatically, or click a specific person.
The tracker remembers clothing appearance after the person leaves the frame,
reports no target while uncertain, and requires three consistent matching
frames to reacquire. Similar clothes, occlusion, or lighting changes can confuse
this heuristic; it is not identity recognition. Click again to change selection.

The color tracker keeps its separate behavior: after clicking, it rejects
unlikely jumps and requires another click after a one-second loss.

## Optional voice

On Mac, drone-camera mode uses built-in `say` speech: no internet, ElevenLabs
package, or API key is required. Callouts play through the Mac's audio output.
Voice runs in the background and does not overlap or queue old state changes.

Other video sources use ElevenLabs when available, with Mac speech as a
fallback. For cloud voice, install `elevenlabs` in the same environment and set
`ELEVENLABS_API_KEY` locally. Do not commit keys. Cloud voice needs internet,
which the drone Wi-Fi generally does not provide. Missing voice dependencies
do not stop the video loop; offline speech currently requires macOS `say`.

## Keys

Focus the video window for keyboard controls. In dry-run mode, flight actions
only print messages; they do not control the drone.

| Key | Does |
|---|---|
| click | Select target color, or a person in person mode |
| `P` | Calibrate gyro on a flat surface with motors off |
| `T` | Request takeoff directly (holds neutral commands for 3 seconds) |
| `A` | Autopilot on/off (starts OFF) |
| `W/S`, `A/D` | Flight-test only: manual pitch / roll (`A` is left roll here) |
| `E/C`, `J/L` | Flight-test only: manual throttle / yaw |
| `H` | Flight-test only: cancel manual input and request hover |
| `SPACE` | Land and turn autopilot off |
| `K` | Kill motors immediately (an airborne drone drops) |
| `R` | Record the HUD to a video file |
| `Q` | Quit (requests landing if this program considers the drone flying) |

## Team and safety

Read **INTERFACES.md** before changing code. Keep Target / Command / State
formats unchanged. Only edit your assigned code unless the team agrees otherwise.

| Person | File | Responsibility |
|---|---|---|
| 1 | `drone.py`, `main.py` | Video, control protocol, main loop |
| 2 | `vision.py` | Detection and tracking |
| 3 | `hud.py`, `voice.py` | Display and voice |
| 4 | `controller.py` | Movement logic and safety |

- Autopilot starts OFF. SPACE requests landing; K stops motors.
- Lost target → neutral/hover commands immediately. Never keep flying blind.
- Close target → stop forward motion. Image size does not guarantee 1–2 metres.
- `--person` turns only; it never commands approach or altitude changes.
  `--follow` is a separate experimental mode with limited forward movement,
  described below. Never fly into a person; box size is not a distance sensor.
- Verify video first, with props off. Live flight behavior needs separate testing.

## Person tracking on the drone feed

`main.py --person` uses YOLO and BoT-SORT camera-motion compensation. It combines
the selected track ID, clothing appearance, position/size continuity and confidence.
Click a yellow person box, then press `A` to enable yaw tracking. Autopilot starts
off. `T` requests takeoff, `H` cancels tracking and requests neutral sticks, and
`SPACE` requests landing. There are no forward, sideways or altitude commands
in this mode. Neutral sticks can still drift; physical flight testing is required.

### Following is currently preview-only

**Live `--follow` is disabled after reports of uncontrolled forward movement
and altitude loss.** It exits before loading the model or opening the camera
or control connection. The earlier live launch command must not be used for
another flight. Centered sticks have not been verified to hold this F19 in
position or at altitude; a three-second delay cannot establish stable hover.

Tracking/following logic can still be previewed without RC commands:

```bash
.venv/bin/python -u main.py --follow --dry-run --model yolo11s.pt --imgsz 640
```

Click a person box. After confirmation, its current width and height become
the reference gap. The preview requests forward input only when both box
dimensions shrink by at least 15% for 0.4 seconds and the person is centred.
The 0.08 cap is a normalized stick input, **not a measured speed or angle**.
It does not guarantee gentle motion, altitude retention, or stopping distance.
`H` cancels selection-based control and requests neutral sticks; it is not a
position-hold controller. Uppercase and lowercase letter shortcuts work.
`SPACE` continues to request landing.

Before re-enabling live following, compare the original app's known-working
neutral/takeoff/control packets with ours, with props removed. Verify the
packet layout, axis order/sign, neutral values, and throttle/mode flags.
The current program has no measured altitude, velocity, or airborne-state
feedback. Do not guess compensating throttle or trim from detector boxes.

### Manual keyboard control with tracking display

`--manual` keeps person detection and click selection visible while giving
movement control only to the keyboard. Selecting a person never enables
autopilot. Offline target acquired/lost callouts also work in manual mode;
they describe visual tracking, not hovering or approaching. It uses a Tk
window (included in this Mac Python installation) to
receive key releases and standalone modifier keys.

Preview without flight commands:

```bash
.venv/bin/python -u main.py --manual --dry-run --webcam --model yolo11s.pt
```

With the drone camera, omit `--webcam`. Omitting `--dry-run` enables real
manual commands. The reported hover/altitude-loss fault remains unresolved;
manual input does not repair the protocol or provide position hold.

| Key | Manual action |
| --- | --- |
| W / S | Forward / back |
| A / D | Slide left / right |
| Q / E | Rotate left / right (Q no longer quits) |
| Shift / Control | Climb / descend; Up/Down arrows are alternatives |
| T | Request takeoff, followed by a three-second neutral interval |
| L or SPACE | Request landing |
| K | Stop motors; the drone drops |
| H | Cancel held movement and request neutral sticks |
| Escape | Request landing and quit |
| R | Start/stop HUD and clean video recordings |
| 1 / 2 / 3 | Label the attempt steady / unstable / crash |

Hold movement keys to request input; release to return that axis to neutral.
Opposing keys cancel. All four manual keyboard axes are capped at 0.20
(20% stick travel, or lower configured limits). These are stick fractions, not measured
angles or speeds. There is no guarantee of stable hover or altitude retention.
Losing window focus clears held input; sender-side input expires after 0.2s
without a refresh. H requires releasing held keys before movement can restart.
Missing/stale video and control-link faults also suppress movement. Landing
and motor-stop actions remain available without video.

The footer shows held keys, the takeoff countdown, and the reason movement is
blocked (such as window focus or stale video). After T, release all keys and
wait for `READY: manual input enabled` before requesting movement. READY means
software permits input; it does not confirm hover or the drone's response.
Person-confirmation messages describe the tracking display and never gate
manual keys. A brief wait for the next camera frame preserves held-key input
within the sender's existing 0.3-second video timeout. Releasing a key still
centers its axis immediately; missing frames never refresh that timeout.

The `INPUT` row shows capped pitch, roll, yaw and throttle requests; these are
not measured movement. Video acknowledgement packets embed stick commands
too. They now use the same current input and expiry checks as the RC sender,
instead of interleaving neutral values with key inputs. Action flags such as
takeoff/land are never copied into video acknowledgements. Physical response
to this transport change is unverified. Manual keyboard strength is configured
separately by `DroneLink.KEYBOARD_LIMITS`.

RC and video ACK headers retain Turbodrone's original packet numbering.
The experimental shared command counter was reverted after runs with it
received no fresh video. Startup reports `original Turbodrone packet numbering`;
the journal marks this as `command_sequence: upstream`. Compatibility of a
different numbering scheme has not been established on this F19.

Operator labels are stored with the attempt number in the journal. R recordings
are linked by filename and per-frame timestamps, allowing video to be compared
with command and outcome labels. This collects review data; it does not train
a controller or automatically raise limits. The known WiFi-UAV status format
mentions flight-state, battery and flags; usable attitude/gyro/altitude telemetry
has not been verified for this F19 or wired into this program.

### Flight-attempt logs

Live control sessions automatically write `flight_logs/flight_<time>.jsonl`.
The terminal prints the path. These local files are ignored by Git; no logs
are uploaded automatically. Dry runs do not create flight-attempt logs.

Each line is JSON with a UTC timestamp and elapsed session time. Events cover
session start/end, takeoff requests and rejections, selection/tracking changes,
H presses, landing/motor-stop requests, and local socket failures. Control
records include the effective command, stick bytes, raw packet hex, and video
age. Changed control/flag values are logged immediately; unchanged packets are
sampled every 0.1 seconds. `video_ack_control` events separately record sampled
video ACK packets and their embedded stick values while live controls run.
`keyboard_state` records held keys and window-input blocks. `manual_input`
records requested axes, the effective command, and any movement-block reason,
including key requests during the takeoff hold.
Disk writes run on a separate thread; the final row reports dropped log events.

An attempt lasts from an accepted takeoff request until a land/stop request or
cancellation, **not measured time airborne**. Attempts shorter than 30 seconds
are labelled `suspected_failed`; longer attempts remain `unverified`. A session
without a takeoff request is labelled `no_takeoff_requested`. A file without
`journal_end` is incomplete, for example after a forced exit; it must not be
counted as a successful flight. These flags help review attempts but cannot
prove a crash, takeoff, hover, or successful landing.

### Person preview and tracking details

Start with the drone camera and **flight commands disabled** (Mac example):

```bash
.venv/bin/python main.py --person --dry-run
```

Add `--webcam` for the laptop camera. Remove `--dry-run` only for supervised
hardware checks after the existing props-off procedure. On Windows use
`.\.venv-win\Scripts\python.exe` in place of `.venv/bin/python`.

- The display shows person IDs, confidence and measured detection time. Yellow
  and amber boxes are selectable; gray boxes are too uncertain to select.
  Clicking uses the displayed box and reports whether selection was accepted.
  Amber selections still need three detections at 60% or higher to confirm.
  A high-confidence box for another person does not replace your selection:
  click that box explicitly. Stale or overlapping boxes cannot be selected.
- `--dry-run` displays a no-flight-commands banner. It previews selection and
  tracking; neither takeoff nor autopilot keys send flight commands.
- Selection stays remembered when confidence drops or the person leaves view,
  until you click a different person or close the app. The display says
  "Target remembered" while their position is unknown; neutral sticks are sent.
- On return, the tracker compares clothing appearance and requires three
  consistent, confident observations before resuming, even if the tracker assigns
  a new ID. Overlapping or similarly dressed people keep the drone hovering.
  Appearance matching is a heuristic and can confuse similar people; it does
  not guarantee the same person after they leave view. Click to correct a match.
- The 0.3-second detection deadline includes local video queue age and inference.
  Slow results cannot steer. This is local receive age, not measured camera exposure age.
- `R` saves both `run_...mp4` (HUD) and `run_..._raw.mp4` (clean frames for evaluation).
  Files use fixed 15 fps playback, so their timing does not measure live latency.

Model settings are command-line options, leaving shared config unchanged:

```bash
.venv/bin/python main.py --person --dry-run --model yolo11s.pt --device auto --imgsz 640
.venv/bin/python main.py --person --dry-run --tracker bytetrack
.venv/bin/python main.py --person --dry-run --video run_EXAMPLE_raw.mp4
```

The default is `yolo11n.pt`, size 640, BoT-SORT. `--device auto` selects available
CUDA, Apple MPS, or CPU. The model warms up before opening a flight connection.
Named YOLO11 n/s/m weights download on first use; do that while connected to the
internet. A local model path is also accepted and must contain a `person` class.
Larger models or input sizes need measurement before choosing them for live use.

Run the offline timing probe without a camera or drone:

```bash
.venv/bin/python benchmark_vision.py --models yolo11n.pt yolo11s.pt --trackers botsort bytetrack
.venv/bin/python benchmark_vision.py --source run_EXAMPLE_raw.mp4 --frames 100 --output /tmp/vision-results.json
.venv/bin/python -m unittest test_person_tracking.py test_flight_safety.py
.venv/bin/python test_pipeline.py
```

Without `--source`, the probe uses Ultralytics' bundled bus photo. It compares
original and synthetically blurred/downsampled/compressed input, reports median
and p95 processing time and detection counts. Counts are **not accuracy scores**;
this does not establish performance on the drone. Use separate labeled drone
recordings to measure missed people and wrong-person switches before changing defaults.

Tracker reference: [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/).
