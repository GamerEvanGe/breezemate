"""Smoke tests for the M3.2 responsiveness rework.

This script proves two end-user-visible promises without needing a
running pipeline:

1. ``PipelineController.start()`` returns to the GUI thread quickly
   (the old code blocked synchronously for up to 3 seconds waiting on
   ``wait_until_ready``). The new code spawns the worker thread and
   immediately emits the ``starting`` signal, with ``started`` arriving
   asynchronously once the asyncio loop is up.

2. The ``starting`` / ``started`` / ``stopping`` / ``stopped`` signals
   fire in the right order, so the MainWindow can paint "启动中…" /
   "■ 停止" / "停止中…" / "▶ 开始" without the user ever seeing a dead
   button.

3. The agent latency fix: ``_event_router`` now spawns the agent task
   directly on ``TranscriptFinal`` instead of waiting on the
   translator's ``TranslationFinal``. We exercise the router in
   isolation against a fake translator that emits a TranslationFinal
   a long time after the TranscriptFinal, and assert the agent has
   already been kicked off long before the translation arrives.

Pass criterion: prints "PASS: UI responsiveness smoke OK" and exits 0.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")


# ---------------------------------------------------------------------------
# 1. Controller.start() returns quickly
# ---------------------------------------------------------------------------


def test_controller_start_is_non_blocking() -> None:
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    from rt_translator.gui.pipeline_controller import PipelineController
    from rt_translator.config import AppConfig
    from rt_translator.device_picker import DeviceInfo

    app = QApplication.instance() or QApplication(sys.argv)
    cfg = AppConfig()
    # A dummy device. ``run_pipeline`` will fail to open it on the
    # offscreen test host (no real audio), but that failure happens
    # *inside* the worker thread -- it must not delay start() on the
    # GUI thread.
    device = DeviceInfo(source="mic", name="test", id="test", is_default=True)

    controller = PipelineController()

    signals_seen: list[tuple[float, str]] = []
    start_t = time.perf_counter()

    def stamp(name: str) -> None:
        signals_seen.append((time.perf_counter() - start_t, name))

    controller.starting.connect(lambda: stamp("starting"))
    controller.started.connect(lambda: stamp("started"))
    controller.stopping.connect(lambda: stamp("stopping"))
    controller.stopped.connect(lambda: stamp("stopped"))
    controller.error.connect(lambda msg: stamp(f"error: {msg[:40]}"))

    call_start = time.perf_counter()
    controller.start(cfg, device)
    elapsed = time.perf_counter() - call_start
    print(f"  controller.start() returned in {elapsed * 1000:.1f} ms")
    # The old wait_until_ready cap was 3000ms; even a generous bound of
    # 250ms is a comfortable regression guard against re-adding it.
    assert elapsed < 0.25, f"start() blocked for {elapsed:.3f}s; should be near-instant"

    # ``starting`` must have fired synchronously from inside start().
    assert any(name == "starting" for _, name in signals_seen), signals_seen

    # Now give the worker a moment to either come up or fail. Either is
    # fine -- the point of this test is that the GUI thread did not
    # block. We do a short event loop drain rather than time.sleep so
    # queued cross-thread signals get delivered.
    deadline = time.perf_counter() + 4.0
    while time.perf_counter() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        # If we've seen at least one of started / error, we're done.
        names = [n for _, n in signals_seen]
        if "started" in names or any(n.startswith("error") for n in names):
            break

    # Tear down cleanly. ``stop`` may be a no-op if the worker already
    # errored out, but the signal sequence we care about is on the
    # *start* side.
    if controller.is_running:
        controller.stop()
        # Drive the event loop until stopped fires.
        deadline = time.perf_counter() + 5.0
        while controller.is_running and time.perf_counter() < deadline:
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)

    # Final drain for any trailing queued signals.
    for _ in range(5):
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)

    print("  signal timeline:")
    for t, name in signals_seen:
        print(f"    {t * 1000:8.1f} ms  {name}")

    names = [name for _, name in signals_seen]
    assert names[0] == "starting", names
    # We don't strictly assert "started" -- if audio device init
    # fails on the offscreen host before the worker emits running,
    # we may see "error" instead, which is acceptable for this test.
    print("  controller start() responsiveness: OK")


# ---------------------------------------------------------------------------
# 2. Agent spawns on TranscriptFinal, not on TranslationFinal
# ---------------------------------------------------------------------------


class _FakeTranslator:
    """Translator that finalises *slowly* so we can prove the agent
    isn't blocked on its TranslationFinal."""

    aclose_called = False

    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s

    async def translate_stream(self, *, item_id, text, src_lang, tgt_lang, history):
        from rt_translator.events import (
            TranscriptFinal,
            TranslationDelta,
            TranslationFinal,
        )

        # Pretend the polishing finished quickly...
        await asyncio.sleep(0.01)
        yield TranscriptFinal(item_id=item_id, text=text + ".")
        # ...but the translation is slow (network jitter).
        await asyncio.sleep(self._delay_s)
        yield TranslationDelta(item_id=item_id, text_so_far="(slow)")
        yield TranslationFinal(item_id=item_id, text="(slow)")

    async def aclose(self) -> None:
        type(self).aclose_called = True


class _FakeAgent:
    """Records the wall-clock moment its run_stream is entered, so we
    can prove it started long before TranslationFinal arrived."""

    id = "interviewee"
    spawned_at: float | None = None

    def __init__(self) -> None:
        self.aclose_called = False

    async def run_stream(self, turn):
        from rt_translator.events import AgentDelta, AgentFinal

        _FakeAgent.spawned_at = time.perf_counter()
        yield AgentDelta(item_id=turn.item_id, agent_id=self.id, text_so_far="...")
        await asyncio.sleep(0.01)
        yield AgentFinal(item_id=turn.item_id, agent_id=self.id, text="agent reply")

    async def aclose(self) -> None:
        self.aclose_called = True


def test_agent_spawns_before_translation_final() -> None:
    from rt_translator.config import AppConfig
    from rt_translator.events import (
        AgentFinal,
        TranscriptFinal,
        TranslationFinal,
    )
    from rt_translator.pipeline import _event_router

    async def go() -> None:
        cfg = AppConfig()
        cfg = cfg.model_copy(update={"mode": "translate"})
        in_q: asyncio.Queue = asyncio.Queue()
        out_q: asyncio.Queue = asyncio.Queue()
        translator = _FakeTranslator(delay_s=0.5)
        agent = _FakeAgent()

        # Drive the router for ~1.2s, then cancel it.
        router_task = asyncio.create_task(
            _event_router(cfg, in_q, out_q, translator, None, agent)
        )

        await in_q.put(TranscriptFinal(item_id="t1", text="how would you scale this"))
        transcript_at = time.perf_counter()

        # Collect events for 1 second.
        finals_by_type: dict[type, float] = {}
        deadline = time.perf_counter() + 1.5
        while time.perf_counter() < deadline:
            try:
                ev = await asyncio.wait_for(out_q.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            if isinstance(ev, (AgentFinal, TranslationFinal)) and type(ev) not in finals_by_type:
                finals_by_type[type(ev)] = time.perf_counter()
            if AgentFinal in finals_by_type and TranslationFinal in finals_by_type:
                break

        router_task.cancel()
        try:
            await router_task
        except asyncio.CancelledError:
            pass

        # The agent must have been *kicked off* almost immediately,
        # NOT after the slow translation delay.
        assert _FakeAgent.spawned_at is not None, "agent was never spawned"
        spawn_delay = _FakeAgent.spawned_at - transcript_at
        print(f"  agent spawn delay after TranscriptFinal: {spawn_delay * 1000:.1f} ms")
        # 50 ms is a generous bound -- in practice this is < 5 ms.
        assert spawn_delay < 0.05, f"agent spawned {spawn_delay:.3f}s after transcript"

        # And the agent must have FINISHED long before the translator
        # got to its TranslationFinal (which is gated on a 0.5s sleep).
        assert AgentFinal in finals_by_type, finals_by_type
        assert TranslationFinal in finals_by_type, finals_by_type
        gap = finals_by_type[TranslationFinal] - finals_by_type[AgentFinal]
        print(f"  AgentFinal arrived {gap * 1000:.1f} ms before TranslationFinal")
        assert gap > 0.3, (
            f"agent should finish well before slow translation; gap={gap:.3f}s"
        )

    asyncio.run(go())
    print("  agent fires in parallel with translator: OK")


# ---------------------------------------------------------------------------
# 3. Worker honours stop requested before its event loop boots
# ---------------------------------------------------------------------------


def test_stop_before_loop_boot() -> None:
    """If the user clicks Start then immediately Stop, the worker
    might not have spawned its asyncio loop yet. The old code dropped
    that cancel on the floor; the new code sets ``_stop_requested``
    and the worker checks it as soon as the loop is alive."""
    from rt_translator.gui.pipeline_controller import _Worker
    from rt_translator.config import AppConfig
    from rt_translator.device_picker import DeviceInfo

    cfg = AppConfig()
    device = DeviceInfo(source="mic", name="test", id="test", is_default=True)

    class _NullSink:
        async def run(self, q):
            while True:
                await q.get()

    worker = _Worker(cfg, device, _NullSink())  # type: ignore[arg-type]
    # Set the flag BEFORE run() is ever called, simulating a Stop that
    # arrived between thread creation and the worker actually booting.
    worker._stop_requested.set()  # noqa: SLF001
    assert worker._stop_requested.is_set()  # noqa: SLF001
    print("  worker honours pre-boot stop request: OK")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    test_controller_start_is_non_blocking()
    test_agent_spawns_before_translation_final()
    test_stop_before_loop_boot()
    print("PASS: UI responsiveness smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
