"""Shared settings. Tell the team if you change something here."""

# Think of this file as the program's settings knobs.
# Other files read these values to agree on colors, movement limits, and Wi-Fi.
# ---- Video source ----
# "file"   -> play a recorded video (good for building without the drone)
# "webcam" -> laptop webcam (good for quick tests)
# "drone"  -> live drone feed (drone.py must implement it)
VIDEO_SOURCE = "drone"
VIDEO_FILE = "test_video.mp4"
WEBCAM_INDEX = 0

# ---- Target color (HSV). Click the target in the window to set it live. ----
# HSV describes the color, how colorful it is, and how bright it is.
TARGET_HSV = (60, 200, 200)      # default: bright green
HSV_TOLERANCE = (12, 80, 80)     # how far from the color still counts
MIN_TARGET_AREA = 0.002          # ignore blobs smaller than this fraction of the frame

# ---- Controller ----
STOP_SIZE = 0.30        # stop moving forward when target is 30% of frame width
CENTER_DEADBAND = 0.05  # don't turn if target is within 5% of center
# Gains say how strongly to turn, climb, or move forward when a change is needed.
YAW_GAIN = 1.2
UP_GAIN = 1.0
FORWARD_GAIN = 1.0

# Speed caps (0–1). Start LOW. Raise slowly once it behaves.
MAX_YAW = 0.35
MAX_UP = 0.30
MAX_FORWARD = 0.25

# A frame is one camera picture. This waits before showing the LOST label;
# movement is stopped as soon as the target disappears.
LOST_FRAMES = 5

# ---- Drone network ----
# The IP is the drone's Wi-Fi address; ports are doors for different messages.
DRONE_IP = "192.168.169.1"
VIDEO_PORT = 8800
CONTROL_PORT = 8800          # None = dry run: commands are printed, nothing is sent
WIFI_UAV_VARIANT = "uav"     # the F19 only streams video in Turbodrone's "uav" mode
TURBODRONE_PATH = "~/turbodrone/backend"
