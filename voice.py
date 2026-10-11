"""OWNER: Person 3 — Jarvis-style voice callouts using ElevenLabs TTS.

Speaks "Target acquired" (with a color description) when the lock-on first
engages, and "Target lost" when the lock drops. Runs TTS in a background
thread so it never blocks the video loop.

Setup:
  pip install elevenlabs
  export ELEVENLABS_API_KEY="your-key-here"
"""
import os
import random
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
        import io
        import sounddevice as sd
        import soundfile as sf

        audio = _eleven.text_to_speech.convert(
            text=text,
            voice_id=_voice_id,
            model_id="eleven_turbo_v2",
            output_format="pcm_24000",
        )
        # audio is a generator of bytes — collect it
        raw = b"".join(audio)
        import numpy as np
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(samples, samplerate=24000, blocking=True)
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


def _describe_target(frame, target):
    """Short spoken description: position and distance only."""
    tx = target["x"]
    tw = target["w"]

    if tx < 0.33:
        pos = "left side"
    elif tx > 0.66:
        pos = "right side"
    else:
        pos = "dead ahead"

    if tw > 0.4:
        dist = "close range"
    elif tw > 0.15:
        dist = "mid range"
    else:
        dist = "long range"

    return f"{pos}, {dist}"


# --- Voice line pools ---

_ACQUIRED_LINES = [
    "Target acquired. {desc}.",
    "Lock on confirmed. I see {desc}.",
    "Target in sight. {desc}.",
    "Contact. {desc}.",
    "Got a visual. {desc}.",
    "Tracking. {desc}.",
]

_APPROACHING_LINES = [
    "Closing in on target.",
    "Moving to intercept.",
    "Approaching target. Hold steady.",
    "En route to target.",
    "Advancing on target position.",
]

_REACHED_LINES = [
    "Target reached. Holding position.",
    "We're on top of it. Hovering.",
    "Arrived at target. Maintaining distance.",
    "Target in range. Holding steady.",
    "Right on target. Standing by.",
]

_LOST_LINES = [
    "Target lost. Scanning.",
    "I've lost visual. Searching the area.",
    "Target dropped off. Looking around.",
    "Contact lost. Standing by.",
    "No visual. Sweeping the area.",
]

_REACQUIRED_LINES = [
    "Target reacquired. {desc}.",
    "I see it again. {desc}.",
    "Back on track. {desc}.",
    "Visual restored. {desc}.",
]


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

    desc = ""
    if frame is not None and target is not None and target.get("found"):
        desc = _describe_target(frame, target)

    text = None
    if state in ("LOCKED", "APPROACHING") and prev in ("SEARCHING", None):
        line = random.choice(_ACQUIRED_LINES)
        text = line.format(desc=desc) if desc else line.replace(". {desc}", "").replace("{desc}", "target")
    elif state in ("LOCKED", "APPROACHING") and prev == "LOST":
        line = random.choice(_REACQUIRED_LINES)
        text = line.format(desc=desc) if desc else line.replace(". {desc}", "").replace("{desc}", "target")
    elif state == "APPROACHING" and prev in ("LOCKED", "REACHED"):
        text = random.choice(_APPROACHING_LINES)
    elif state == "REACHED":
        text = random.choice(_REACHED_LINES)
    elif state == "LOST" and prev in ("LOCKED", "APPROACHING", "REACHED"):
        text = random.choice(_LOST_LINES)

    if text:
        _last_speak_time = now
        print(f"[voice] {text}")
        _speak_async(text)
