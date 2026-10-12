"""OWNER: Person 3 — offline Mac / ElevenLabs voice callouts.

Speaks "Target acquired" (with position and size descriptions) when the lock-on first
engages, and "Target lost" when the lock drops. Runs TTS in a background
thread so it never blocks the video loop. Drone video uses macOS speech without
internet or an API key. Other sources use ElevenLabs with a Mac fallback.

Optional cloud voice setup:
  pip install elevenlabs
  export ELEVENLABS_API_KEY="your-key-here"
"""
import os
import random
import shutil
import subprocess
import threading
import time

import config

# This is the announcer: turn status changes into short spoken messages.
_eleven = None
_voice_id = None
_speech_lock = threading.Lock()


def _init():
    """Lazy-load the ElevenLabs client on first use."""
    global _eleven, _voice_id
    if _eleven is not None:
        return True
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        print("[voice] ElevenLabs not installed; trying offline speech")
        return False

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("[voice] No ElevenLabs key; trying offline speech")
        return False

    _eleven = ElevenLabs(api_key=key, timeout=3.0)
    # Default voice — change this to any ElevenLabs voice ID you like.
    # "Adam" is a good deep/calm option. Browse voices at elevenlabs.io.
    _voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Adam
    print(f"[voice] ElevenLabs ready (voice: {_voice_id})")
    return True


def _speak_local(text):
    """macOS speech works on the drone's Wi-Fi, without a cloud request."""
    say = shutil.which("say")
    if say is None:
        return False
    try:
        subprocess.run([say, text], check=True, timeout=10,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[voice] Offline speech failed ({type(exc).__name__})")
        return False


def _speak(text):
    """Generate/play speech in the worker; never require internet on drone Wi-Fi."""
    if config.VIDEO_SOURCE == "drone" and shutil.which("say"):
        _speak_local(text)
        return
    try:
        if not _init():
            _speak_local(text)
            return
        import sounddevice as sd

        audio = _eleven.text_to_speech.convert(
            text=text,
            voice_id=_voice_id,
            model_id="eleven_turbo_v2",
            output_format="pcm_24000",
            request_options={"max_retries": 0, "timeout_in_seconds": 3},
        )
        # audio is a generator of bytes — collect it
        raw = b"".join(audio)
        import numpy as np
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(samples, samplerate=24000, blocking=True)
    except Exception as exc:
        print(f"[voice] Cloud speech unavailable ({type(exc).__name__}); trying offline speech")
        _speak_local(text)


def _speak_async(text):
    """One speaker, no queue of stale callouts and no overlapping playback."""
    # Only one voice talks at a time. A helper talks while the video keeps running.
    if not _speech_lock.acquire(blocking=False):
        return False

    def run():
        try:
            _speak(text)
        finally:
            _speech_lock.release()

    try:
        threading.Thread(target=run, daemon=True).start()
    except Exception:
        _speech_lock.release()
        raise
    return True


# --- State tracking for the main loop ---

_last_state = None
_last_speak_time = 0.0
_pending_text = None
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
        # These words come from box size, not a real distance measurement.
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


def announce(state, frame=None, target=None, visual_only=False):
    """Call every frame from main.py. Speaks on state transitions."""
    # Say something when the situation changes, not for every camera picture.
    global _last_state, _pending_text

    # Manual mode describes the camera only. A large box is not proof of
    # arrival, hovering, or physical motion toward the person.
    if visual_only and state in ('APPROACHING', 'REACHED'):
        state = 'LOCKED'

    if state == _last_state:
        _flush_pending()
        return
    prev = _last_state
    _last_state = state

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
        text = 'Visual target lost.' if visual_only else random.choice(_LOST_LINES)

    # Keep only the current state's callout. A short-lived state must not
    # suppress a later stable lock merely because the speaker was busy.
    _pending_text = text
    _flush_pending()


def _flush_pending():
    # Wait for a pause between messages, then try to say the newest one.
    global _last_speak_time, _pending_text
    now = time.monotonic()
    if _pending_text is not None and now - _last_speak_time >= _MIN_INTERVAL:
        if _speak_async(_pending_text):
            _last_speak_time = now
            print(f"[voice] {_pending_text}")
            _pending_text = None
