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
troubleshooting video; live control requires the explicit flight-test mode below.

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

## First computer-controlled takeoff/hover/landing test

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

- Wait for at least one second of fresh video. Focus the video window.
- `T` requests gyro calibration, waits about one second, then requests takeoff.
  Verify props-off responses first. The program commands neutral sticks after
  takeoff; it does not steer toward a target. `A` is disabled in this mode.
- `SPACE` cancels a pending takeoff or requests landing if this program has
  requested takeoff. `Q` or closing the window also cancels/requests landing.
- `K` requests immediate motor stop; an airborne drone will drop. It is not a
  normal landing control.
- One takeoff is allowed per session. After landing or a motor stop, physically
  confirm the drone is grounded before restarting. Do not start this program
  while the drone is already airborne under another controller.
- Stale/missing video cancels pending takeoff and sends neutral/hover commands.
  Loss of a target in tracking demos also immediately produces neutral commands.
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

`voice.py` is included and imports without the ElevenLabs package. Voice is
optional; its missing-package/key messages do not stop the video loop.
To enable it separately, install `elevenlabs` in the same environment and set
`ELEVENLABS_API_KEY` locally. Do not commit keys. Cloud voice needs internet,
which the drone Wi-Fi generally does not provide.

## Keys

Focus the video window for keyboard controls. In dry-run mode, flight actions
only print messages; they do not control the drone.

| Key | Does |
|---|---|
| click | Select target color, or a person in person mode |
| `P` | Calibrate gyro on a flat surface with motors off |
| `T` | Calibrate, then take off (holds neutral commands for 3 seconds) |
| `A` | Autopilot on/off (starts OFF) |
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
- Never fly at people. Person mode is a webcam/dry-run demo.
- Verify video first, with props off. Live flight behavior needs separate testing.
