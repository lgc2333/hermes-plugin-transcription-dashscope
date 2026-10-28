#!/usr/bin/env python3
"""End-to-end smoke check against the real DashScope endpoint.

    python scripts/smoke.py AUDIO [--live] [--language zh]

File mode exercises ``transcribe()`` — the path voice messages and desktop fallbacks take.
``--live`` replays the same audio as 16 kHz PCM in 100 ms frames through ``open_stream_session()``,
printing partials as they arrive, which is what dictation does with a microphone.

Run it with the interpreter Hermes uses (it needs ``websockets`` and Hermes' own package):
``/opt/hermes/.venv/bin/python scripts/smoke.py clip.mp3 --live``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "transcription_dashscope"

if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

provider_module = importlib.import_module(f"{PACKAGE}.provider")
PCM_CHUNK = 3200  # 100 ms of 16 kHz mono s16le


def _pcm(path: str) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", "-"],
        check=True, capture_output=True, timeout=300,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="audio file ffmpeg can read")
    parser.add_argument("--live", action="store_true", help="also replay the audio through the live session")
    parser.add_argument("--language", default=None, help="language hint, e.g. zh")
    args = parser.parse_args()

    provider = provider_module.DashScopeProvider()
    print(f"provider={provider.name} model={provider.default_model()} available={provider.is_available()}", flush=True)
    if not provider.is_available():
        print("no API key: set HERMES_CUSTOM_DASHSCOPE_API_KEY (or DASHSCOPE_API_KEY)", file=sys.stderr)
        return 2

    started = time.monotonic()
    result = provider.transcribe(args.audio, language=args.language)
    print(f"file: success={result['success']} ({time.monotonic() - started:.2f}s)", flush=True)
    print(f"file transcript: {result.get('transcript') or result.get('error')}", flush=True)
    if not result["success"]:
        return 1

    if not args.live:
        return 0

    session = provider.open_stream_session(language=args.language)
    pcm = _pcm(args.audio)
    seen = ""
    started = time.monotonic()
    for offset in range(0, len(pcm), PCM_CHUNK):
        session.push_audio(pcm[offset:offset + PCM_CHUNK])
        partial = session.partial_transcript()
        if partial != seen:
            seen = partial
            print(f"  [{time.monotonic() - started:5.2f}s] partial: {partial}", flush=True)
        time.sleep(0.1)  # pace it like a live microphone
    live = session.finalize()
    print(f"live: success={live['success']} ({time.monotonic() - started:.2f}s)", flush=True)
    print(f"live transcript: {live.get('transcript') or live.get('error')}", flush=True)
    return 0 if live["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
