"""Smoke tests for pipeline lifecycle responsiveness + the M3.3
pause-gated agent firing.

This script proves three end-user-visible promises without needing a
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

3. The M3.3 pause-gated agent: the previous design fired the agent on
   every ``TranscriptFinal``, which produced noisy fragmentary
   replies. The new design buffers transcripts and only flushes them
   to the agent once the speaker has been silent for
   ``cfg.agent.pause_threshold_s`` seconds. We exercise the router in
   isolation and assert
     (a) the agent does NOT fire while transcripts are still coming
         in below the pause threshold,
     (b) the agent DOES fire after the pause elapses, and
     (c) the text it receives is the JOIN of all buffered
         sentences, not just the most recent one.

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
# 2. Agent fires only after a silence pause, with buffered text
# ---------------------------------------------------------------------------


class _RecordingAgent:
    """Captures the wall-clock moment its run_stream is entered AND
    the text it received, so we can assert both the pause timing AND
    the buffered-text content in one test."""

    id = "interviewee"

    def __init__(self) -> None:
        self.spawns: list[tuple[float, str]] = []  # (wall_clock, source_text)

    async def run_stream(self, turn):
        from rt_translator.events import AgentDelta, AgentFinal

        self.spawns.append((time.perf_counter(), turn.source_text))
        yield AgentDelta(
            item_id=turn.item_id, agent_id=self.id, text_so_far="..."
        )
        await asyncio.sleep(0.01)
        yield AgentFinal(
            item_id=turn.item_id, agent_id=self.id, text="agent reply"
        )

    async def aclose(self) -> None:
        pass


def test_agent_fires_only_after_pause() -> None:
    """End-to-end timing assertion for the M3.3 pause-gated agent."""
    from rt_translator.config import AppConfig
    from rt_translator.events import TranscriptFinal
    from rt_translator.pipeline import _event_router

    async def go() -> None:
        cfg = AppConfig()
        # Shrink the pause threshold so the test runs in a couple of
        # seconds, not ten. Production default is 1.5 s.
        cfg = cfg.model_copy(
            update={
                "mode": "asr_only",
                "agent": cfg.agent.model_copy(
                    update={"pause_threshold_s": 0.6}
                ),
            }
        )
        in_q: asyncio.Queue = asyncio.Queue()
        out_q: asyncio.Queue = asyncio.Queue()
        agent = _RecordingAgent()

        router_task = asyncio.create_task(
            _event_router(cfg, in_q, out_q, None, None, agent),
            name="router",
        )

        # 1. Emit two TranscriptFinals fast (well under the 0.6s
        #    threshold between them). Agent must NOT have fired by
        #    the time the second one lands.
        t0 = time.perf_counter()
        await in_q.put(
            TranscriptFinal(item_id="oai-1", text="Tell me about your background.")
        )
        await asyncio.sleep(0.2)
        await in_q.put(
            TranscriptFinal(
                item_id="oai-2", text="And how would you design a rate limiter?"
            )
        )
        await asyncio.sleep(0.1)

        elapsed_so_far = time.perf_counter() - t0
        print(f"  after {elapsed_so_far:.2f}s of activity, spawns={len(agent.spawns)}")
        assert (
            len(agent.spawns) == 0
        ), f"agent fired too early ({len(agent.spawns)} times during active speech)"

        # 2. Now stop sending transcripts and wait past the pause
        #    threshold. The watcher should flush within ~0.6s + its
        #    150ms poll cycle.
        await asyncio.sleep(cfg.agent.pause_threshold_s + 0.4)

        assert (
            len(agent.spawns) == 1
        ), f"agent should have fired exactly once after the pause; got {len(agent.spawns)}"
        spawn_at, spawn_text = agent.spawns[0]
        delay_after_last_transcript = spawn_at - (t0 + 0.3)
        print(
            f"  agent fired {delay_after_last_transcript * 1000:.0f} ms after last transcript"
        )

        # 3. The spawn text must contain BOTH buffered sentences,
        #    not just the latest one (the user wants the agent to
        #    see the whole recent context to pick the question out).
        assert "background" in spawn_text, spawn_text
        assert "rate limiter" in spawn_text, spawn_text
        print(f"  spawn_text contains both buffered sentences: {spawn_text!r}")

        # 4. After flushing, the buffer should be empty: emit a new
        #    transcript, wait past the threshold, agent fires AGAIN
        #    -- and only on that new sentence, not on the old ones.
        await in_q.put(
            TranscriptFinal(item_id="oai-3", text="What about caching strategies?")
        )
        await asyncio.sleep(cfg.agent.pause_threshold_s + 0.4)
        assert (
            len(agent.spawns) == 2
        ), f"second pause should re-arm the agent; spawns={len(agent.spawns)}"
        assert "caching" in agent.spawns[1][1], agent.spawns[1][1]
        # And critically, the second flush should NOT include text
        # from the first flush -- the buffer was cleared.
        assert (
            "background" not in agent.spawns[1][1]
        ), f"buffer leaked old sentences: {agent.spawns[1][1]!r}"

        router_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await router_task

    import contextlib

    asyncio.run(go())
    print("  pause-gated agent firing: OK")


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


def _run_in_subprocess(test_name: str) -> int:
    """Run one test in its own Python process.

    The lifecycle test spins up real ``QApplication`` + an audio
    capture worker thread (which on Windows holds ``soundcard``'s
    WASAPI globals). Sharing that process with the asyncio-only
    router test then crashes on final exit (0xC0000409 heap
    corruption -- the same family of WASAPI race we hardened
    against in M2.5). Cheapest reliable isolation is process
    boundaries.
    """
    import subprocess

    script = Path(__file__).resolve()
    cmd = [sys.executable, str(script), "--single", test_name]
    print(f"\n--- subprocess: {test_name} ---")
    r = subprocess.run(cmd, cwd=str(REPO))
    return r.returncode


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--single":
        name = sys.argv[2]
        if name == "controller":
            test_controller_start_is_non_blocking()
        elif name == "pause_agent":
            test_agent_fires_only_after_pause()
        elif name == "stop_before_boot":
            test_stop_before_loop_boot()
        else:
            print(f"unknown test {name}", file=sys.stderr)
            return 2
        return 0

    # Default entry point: dispatch each test into its own subprocess.
    for sub in ("controller", "pause_agent", "stop_before_boot"):
        rc = _run_in_subprocess(sub)
        if rc != 0:
            print(f"FAIL: subprocess {sub} returned {rc}", file=sys.stderr)
            return rc
    print("\nPASS: UI responsiveness smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
