"""Shared settings. Tell the team if you change something here."""

# ---- Video source ----
# "file"   -> play a recorded video (good for building without the drone)
# "webcam" -> laptop webcam (good for quick tests)
# "drone"  -> live drone feed (drone.py must implement it)
VIDEO_SOURCE = "webcam"
VIDEO_FILE = "test_video.mp4"
WEBCAM_INDEX = 0

# ---- Target color (HSV). Click the target in the window to set it live. ----
TARGET_HSV = (60, 200, 200)      # default: bright green
HSV_TOLERANCE = (12, 80, 80)     # how far from the color still counts
MIN_TARGET_AREA = 0.002          # ignore blobs smaller than this fraction of the frame

# ---- Controller ----
STOP_SIZE = 0.30        # stop moving forward when target is 30% of frame width
CENTER_DEADBAND = 0.05  # don't turn if target is within 5% of center
YAW_GAIN = 1.2
UP_GAIN = 1.0
FORWARD_GAIN = 1.0

# Speed caps (0–1). Start LOW. Raise slowly once it behaves.
MAX_YAW = 0.35
MAX_UP = 0.30
MAX_FORWARD = 0.25

# Frames the target can disappear before we call it LOST and hover
LOST_FRAMES = 5

# ---- Drone network (Person 1 fills these in after the hack) ----
DRONE_IP = "192.168.169.1"   # guess — check what IP your laptop gets on the drone WiFi
VIDEO_PORT = 8800            # WiFi UAV video, per others' reverse engineering
CONTROL_PORT = None          # unknown until the control hack is done
