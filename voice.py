"""OWNER: Person 3 — Jarvis-style voice callouts using ElevenLabs TTS.

Speaks "Target acquired" (with a color description) when the lock-on first
engages, and "Target lost" when the lock drops. Runs TTS in a background
thread so it never blocks the video loop.

Setup:
  pip install elevenlabs
  export ELEVENLABS_API_KEY="your-key-here"
"""
import os
import threading
import time

_eleven = None
_voice_id = None


def _init():
    """Lazy-load the ElevenLabs client on first use."""
    global _eleven, _voice_id
    if _eleven is not None:
        return True
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import play
    except ImportError:
        print("[voice] elevenlabs not installed — pip install elevenlabs")
        return False

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("[voice] set ELEVENLABS_API_KEY to enable voice")
        return False

    _eleven = ElevenLabs(api_key=key)
    # Default voice — change this to any ElevenLabs voice ID you like.
    # "Adam" is a good deep/calm option. Browse voices at elevenlabs.io.
    _voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Adam
    print(f"[voice] ElevenLabs ready (voice: {_voice_id})")
    return True


def _speak(text):
    """Generate and play TTS audio (blocking — call from a thread)."""
    try:
        from elevenlabs import play
        audio = _eleven.text_to_speech.convert(
            text=text,
            voice_id=_voice_id,
            model_id="eleven_turbo_v2",
        )
        play(audio)
    except Exception as e:
        print(f"[voice] TTS error: {e}")


def _speak_async(text):
    """Fire-and-forget TTS in a daemon thread."""
    t = threading.Thread(target=_speak, args=(text,), daemon=True)
    t.start()


# --- State tracking for the main loop ---

_last_state = None
_last_speak_time = 0.0
_MIN_INTERVAL = 3.0  # don't spam callouts faster than this


def _color_name(frame, target):
    """Get a rough color name from the target's center region."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    cx, cy = int(target["x"] * w), int(target["y"] * h)
    r = max(5, int(min(target["w"] * w, target["h"] * h) * 0.15))
    x0, x1 = max(cx - r, 0), min(cx + r, w)
    y0, y1 = max(cy - r, 0), min(cy + r, h)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return ""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    med = np.median(hsv.reshape(-1, 3), axis=0)
    hue, sat, val = med[0], med[1], med[2]

    if val < 40:
        return "black"
    if sat < 30:
        return "white" if val > 180 else "gray"
    if hue < 10 or hue > 168:
        return "red"
    if hue < 25:
        return "orange"
    if hue < 35:
        return "yellow"
    if hue < 80:
        return "green"
    if hue < 105:
        return "cyan"
    if hue < 135:
        return "blue"
    if hue < 160:
        return "purple"
    return "red"


def announce(state, frame=None, target=None):
    """Call every frame from main.py. Speaks on state transitions."""
    global _last_state, _last_speak_time

    if state == _last_state:
        return
    prev = _last_state
    _last_state = state

    if not _init():
        return

    now = time.time()
    if now - _last_speak_time < _MIN_INTERVAL:
        return

    text = None
    if state == "LOCKED" and prev in ("SEARCHING", "LOST", None):
        color = ""
        if frame is not None and target is not None and target.get("found"):
            color = _color_name(frame, target)
        if color:
            text = f"Target acquired. {color.capitalize()} object."
        else:
            text = "Target acquired."
    elif state == "REACHED":
        text = "Target reached. Holding position."
    elif state == "LOST" and prev in ("LOCKED", "APPROACHING", "REACHED"):
        text = "Target lost. Scanning."

    if text:
        _last_speak_time = now
        print(f"[voice] {text}")
        _speak_async(text)
