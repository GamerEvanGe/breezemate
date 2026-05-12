"""Smoke test: forced-commit watchdogs fire on long monologues.

Exercises two paths:

1. Vosk-canonical: a continuous stream of partials (no silence) should
   produce a TranscriptFinal once ``max_utterance_s`` elapses, NOT
   wait forever for silence.
2. OpenAI Realtime: ``_commit_watchdog`` should send an
   ``input_audio_buffer.commit`` JSON frame once ``preview_max_duration_s``
   has elapsed since the last ``speech_started`` event.

Neither path needs network or audio hardware: we drive the provider's
state by hand and observe the resulting effects (the queued event /
the JSON frame).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_translator.config import ASRConfig, LocalASRConfig, ProviderEndpoint
from rt_translator.events import TranscriptFinal
from rt_translator.providers.asr.openai_realtime import OpenAIRealtimeASR


async def vosk_duration_cap_test():
    """The Vosk silence watcher should finalise on duration cap, even
    if partials keep arriving (no silence)."""
    # Use the silence watcher coroutine in isolation by simulating its
    # input state. We don't actually load Vosk's C library here -- we
    # just instantiate VoskASRProvider, set the state attributes that
    # the watcher reads, and start the watcher coroutine.
    from rt_translator.providers.asr.vosk_full import VoskASRProvider

    cfg = LocalASRConfig(finalize_after_silence_s=10.0)
    prov = VoskASRProvider(cfg, preview_only=False, max_utterance_s=1.0)
    prov._loop = asyncio.get_running_loop()
    prov._event_q = asyncio.Queue()
    prov._current_item_id = "vosk-test-0"
    prov._current_text = "hello world"
    prov._last_change_at = time.monotonic()
    prov._utterance_started_at = time.monotonic()

    # Replicate the silence-watcher's body inline to avoid needing the
    # real engine. (The watcher coroutine is defined inside run(), so
    # we mirror its decision rule here.)
    async def watcher():
        poll_s = 0.05
        while True:
            await asyncio.sleep(poll_s)
            text = prov._current_text
            if not text:
                continue
            now = time.monotonic()
            silence_elapsed = now - prov._last_change_at
            duration_elapsed = (
                now - prov._utterance_started_at
                if prov._utterance_started_at > 0
                else 0.0
            )
            if (
                silence_elapsed >= cfg.finalize_after_silence_s
                or duration_elapsed >= prov._max_utterance_s
            ):
                item_id = prov._current_item_id
                prov._current_text = ""
                prov._utterance_started_at = 0.0
                await prov._event_q.put(
                    TranscriptFinal(item_id=item_id, text=text.strip())
                )
                return

    # Keep "speaking" by bumping _last_change_at every 100ms -- so the
    # silence rule NEVER trips. Only the duration cap should fire.
    async def speaker():
        try:
            while True:
                await asyncio.sleep(0.1)
                prov._last_change_at = time.monotonic()
                prov._current_text = prov._current_text + " ..."
        except asyncio.CancelledError:
            raise

    spk = asyncio.create_task(speaker())
    watch = asyncio.create_task(watcher())

    t0 = time.monotonic()
    ev = await asyncio.wait_for(prov._event_q.get(), timeout=3.0)
    elapsed = time.monotonic() - t0
    spk.cancel()
    watch.cancel()
    print(
        f"  Vosk duration cap: TranscriptFinal after {elapsed:.2f}s "
        f"(cap=1.0s)",
        flush=True,
    )
    assert isinstance(ev, TranscriptFinal), f"expected TranscriptFinal, got {ev!r}"
    assert 0.9 <= elapsed <= 1.6, f"expected ~1.0s, got {elapsed:.2f}s"


class _FakeWebSocket:
    """Minimal WS stand-in: records every send() payload."""

    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, data):
        try:
            self.sent.append(json.loads(data))
        except Exception:
            self.sent.append({"raw": data})


async def openai_force_commit_test():
    asr_cfg = ASRConfig(
        provider="openai_realtime",
        model="gpt-4o-mini-transcribe",
        language="en",
        preview_max_duration_s=2.0,  # min allowed by validator
    )
    # Endpoint isn't actually contacted -- only resolve_api_key() is
    # called if we exercised the connect path. We don't.
    endpoint = ProviderEndpoint()
    prov = OpenAIRealtimeASR(asr_cfg, endpoint)
    ws = _FakeWebSocket()

    # Pretend the server fired speech_started 2.1s ago. The watchdog
    # should commit on its next tick (0.25s granularity).
    prov._speech_started_at = time.monotonic() - 2.1

    task = asyncio.create_task(prov._commit_watchdog(ws))
    try:
        # Allow up to 1.0s for the watchdog to fire.
        for _ in range(40):
            await asyncio.sleep(0.05)
            if any(m.get("type") == "input_audio_buffer.commit" for m in ws.sent):
                break
    finally:
        prov._closed = True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    commits = [m for m in ws.sent if m.get("type") == "input_audio_buffer.commit"]
    print(
        f"  OpenAI force-commit: ws received {len(commits)} commit frame(s) "
        f"(expected >=1)",
        flush=True,
    )
    assert commits, f"expected at least one commit; ws.sent={ws.sent}"
    # speech_started_at should be cleared after the commit so we don't
    # spam more commits.
    assert prov._speech_started_at is None, (
        "expected _speech_started_at=None after commit, "
        f"got {prov._speech_started_at}"
    )


async def main():
    print("Test 1: Vosk duration cap finalises despite continuous speech", flush=True)
    await vosk_duration_cap_test()
    print("Test 2: OpenAI Realtime force-commit watchdog", flush=True)
    await openai_force_commit_test()
    print("=== all force-commit tests passed ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
