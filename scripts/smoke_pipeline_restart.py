"""Smoke test: rapid start/stop/start/stop of the pipeline.

This is the exact crash repro pattern from the BEX64 / 0xc0000409 we
investigated: spin the pipeline up, immediately tear it down, repeat.

We don't have audio hardware in CI, so we monkey-patch AudioCapture to
push fake PCM chunks instead of opening a real recorder. The point is
to exercise the teardown ordering (capture.stop() before ASR aclose,
no zombie thread refs left behind, Vosk reset routed through the
worker queue).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_translator import audio_capture as ac_mod
from rt_translator.audio_capture import AudioCapture
from rt_translator.config import AppConfig
from rt_translator.device_picker import DeviceInfo
from rt_translator.events import LocalPreviewDelta, TranscriptFinal
from rt_translator.pipeline import run_pipeline
from rt_translator.providers.asr.local_vosk import VoskLocalASR


class _FakeAudioCapture(AudioCapture):
    """Drop-in AudioCapture that pushes 16 kHz silence chunks every 50 ms."""

    def start(self, loop):  # type: ignore[override]
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("FakeAudioCapture already running")
        self._thread = None
        self._loop = loop
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._fake_run, name="rt-audio-capture", daemon=True
        )
        self._thread.start()

    def _fake_run(self):
        # 50ms @ 16kHz mono int16 = 1600 samples = 3200 bytes
        silence = b"\x00\x00" * 1600
        while not self._stop_event.is_set():
            self._enqueue(silence)
            time.sleep(0.05)


class _NullSink:
    async def run(self, queue):
        while True:
            await queue.get()


async def one_cycle(cfg, device, duration_s: float):
    sink = _NullSink()
    task = asyncio.create_task(run_pipeline(cfg, device, sink=sink), name="pipe")
    await asyncio.sleep(duration_s)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def main():
    os.environ["RT_TRANSLATOR_VOSK_AUTO_DOWNLOAD"] = "0"
    cfg = AppConfig()
    cfg.asr.provider = "vosk_local"
    cfg.mode = "ascii_only" if hasattr(cfg, "mode") else cfg.mode

    device = DeviceInfo(source="mic", name="Fake", id="fake-id", is_default=False)

    ac_mod.AudioCapture = _FakeAudioCapture  # monkey-patch the symbol the pipeline imports

    print("=== rapid cycle: 5 start/stop iterations ===", flush=True)
    for i in range(5):
        print(f"  cycle {i+1}", flush=True)
        await one_cycle(cfg, device, duration_s=0.5)
        await asyncio.sleep(0.1)
    print("=== done; no native crash. ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
