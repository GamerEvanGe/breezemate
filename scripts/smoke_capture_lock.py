"""Smoke test: cross-instance AudioCapture lock.

Verifies the new process-wide guard prevents two recorder threads from
running at once, AND that the guard releases properly when the old
thread eventually exits (no permanent lockout).
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_translator.audio_capture import AudioCapture
from rt_translator.config import AudioConfig
from rt_translator.device_picker import DeviceInfo


# Subclass that doesn't actually open a recorder; the "run" just
# blocks on a controllable event so we can simulate "thread refuses
# to exit on stop()".
class _ControlledCapture(AudioCapture):
    def __init__(self, *args, hang_event=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._hang_event = hang_event

    def _run(self):  # type: ignore[override]
        # Pretend we're stuck in a slow WASAPI record() call: we
        # ignore _stop_event until _hang_event is released.
        if self._hang_event is not None:
            self._hang_event.wait()
        # Once unblocked, drain quickly.
        while not self._stop_event.is_set():
            time.sleep(0.01)


def make_capture(hang_event=None):
    cfg = AudioConfig()
    dev = DeviceInfo(source="mic", name="fake", id="x", is_default=False)
    q: asyncio.Queue[bytes] = asyncio.Queue()
    return _ControlledCapture(cfg, dev, q, hang_event=hang_event)


def main():
    loop = asyncio.new_event_loop()

    print("Scenario 1: clean start/stop x3 -- should succeed", flush=True)
    for i in range(3):
        cap = make_capture()
        cap.start(loop)
        cap.stop()
        print(f"  cycle {i+1} ok", flush=True)

    print("", flush=True)
    print("Scenario 2: stuck thread blocks the next start", flush=True)
    hang = threading.Event()
    cap1 = make_capture(hang_event=hang)
    cap1.start(loop)

    # Trigger stop -- with hang held, the thread will not exit within
    # the 10 s join, so stop() should keep the global lock held.
    t0 = time.monotonic()
    cap1.stop()
    elapsed = time.monotonic() - t0
    print(f"  cap1.stop() returned after {elapsed:.2f}s (expected ~10s)", flush=True)
    assert elapsed >= 9.5, "stop() should have waited the full join timeout"

    # Now attempt to start a NEW capture instance -- should be refused.
    cap2 = make_capture()
    refused = False
    try:
        cap2.start(loop)
    except RuntimeError as e:
        refused = True
        print(f"  cap2.start() correctly refused: {e}", flush=True)
    assert refused, "cap2.start() should have raised while cap1 thread is alive"

    print("", flush=True)
    print("Scenario 3: release the stuck thread -> next start succeeds", flush=True)
    hang.set()
    # Give the zombie a moment to wind down.
    if cap1._thread is not None:
        cap1._thread.join(timeout=2.0)
    cap3 = make_capture()
    cap3.start(loop)
    print("  cap3.start() ok after zombie exited", flush=True)
    cap3.stop()

    loop.close()
    print("", flush=True)
    print("=== all scenarios passed ===", flush=True)


if __name__ == "__main__":
    main()
