# ai_server/voice_control.py
from __future__ import annotations

import json
import os
import signal
import sys
import time

_running = True


def _handle_sigterm(signum, frame):
    global _running
    _running = False


def load_config() -> dict:
    raw = os.environ.get("AI_MODE_CONFIG_JSON", "{}")
    try:
        return json.loads(raw)
    except Exception:
        return {}


def main():
    global _running
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    cfg = load_config()
    print("[VOICE_CONTROL] started")
    print("[VOICE_CONTROL] config:", cfg)

    # TODO: Buraya voice pipeline (Whisper/Vosk vs) + robot control gelecek

    while _running:
        time.sleep(0.2)

    print("[VOICE_CONTROL] stopping gracefully...")
    sys.exit(0)


if __name__ == "__main__":
    main()