"""QObject wrapper that runs the asyncio pipeline in a worker QThread.

Lifecycle:

    controller = PipelineController()
    controller.connect_status.connect(...)
    controller.transcript_delta.connect(...)
    ...
    controller.start(cfg, device)   # spawns the worker thread
    # ... pipeline runs, signals flow ...
    controller.stop()               # cancels the asyncio task, joins thread
    controller.start(cfg2, device2) # may be called again with new config

The asyncio pipeline lives entirely in the worker thread. Qt signal
emission is thread-safe (queued across threads automatically), so the
GUI thread never directly touches asyncio primitives. This avoids the
complexity / extra dependency of ``qasync``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from ..config import AppConfig
from ..device_picker import DeviceInfo
from ..pipeline import run_pipeline
from .signal_sink import SignalSink, SinkSignals

log = logging.getLogger(__name__)


class _Worker(QObject):
    """Bottom half: owns the asyncio event loop in its own thread."""

    finished = Signal()
    error = Signal(str)
    # Emitted from the worker thread once the asyncio loop is up *and*
    # the main task has been scheduled. The PipelineController relays
    # this as ``started`` to the GUI. This lets us tell the user
    # "pipeline is actually booting" instead of "I'm about to start
    # booting it" -- the former is what the user cares about.
    running = Signal()

    def __init__(
        self,
        cfg: AppConfig,
        device: DeviceInfo,
        sink: SignalSink,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._device = device
        self._sink = sink
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None
        # Set as soon as the loop is alive and ``_main_task`` is
        # assigned. Currently only used internally as a race-detector;
        # no GUI code blocks on it any more (see ``request_stop``).
        self._ready = threading.Event()
        # Set by ``request_stop`` so a Stop click that lands BEFORE
        # the worker has finished booting is still honoured -- ``run``
        # checks this flag right after creating the main task and
        # cancels straight away if it has been set in the meantime.
        # This is what lets us drop the old blocking ``wait_until_ready``
        # call from the GUI thread.
        self._stop_requested = threading.Event()

    def run(self) -> None:
        """Slot connected to ``QThread.started``. Runs until the main
        task finishes (graceful) or raises (error). Always emits
        ``finished`` exactly once on exit."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._main_task = loop.create_task(
                run_pipeline(self._cfg, self._device, sink=self._sink),
                name="rt-gui-pipeline",
            )
            self._ready.set()
            # Race: if Stop was clicked between PipelineController.start()
            # returning and us getting here, ``request_stop`` may have
            # tried to schedule a cancel against a still-None loop and
            # silently dropped it. Catch that case explicitly.
            if self._stop_requested.is_set():
                self._main_task.cancel()
            # Tell the GUI we're really up. Auto-connection across
            # threads is queued, so this is safe even though we're on
            # the worker thread.
            self.running.emit()
            loop.run_until_complete(self._main_task)
        except asyncio.CancelledError:
            log.info("Pipeline worker cancelled cleanly.")
        except Exception as e:
            log.exception("Pipeline worker crashed")
            self.error.emit(str(e))
        finally:
            try:
                # Drain any leftover tasks so the loop closes cleanly.
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                log.debug("Cleanup of pending tasks failed", exc_info=True)
            try:
                loop.close()
            except Exception:
                pass
            self._loop = None
            self._main_task = None
            self._ready.clear()
            self._stop_requested.clear()
            self.finished.emit()

    def request_stop(self) -> None:
        """Thread-safe cancellation. Called from the GUI thread.

        Always sets ``_stop_requested`` first so a Stop click that
        lands before ``run`` has assigned ``_loop`` is still honoured
        when the loop comes up moments later.
        """
        self._stop_requested.set()
        loop = self._loop
        task = self._main_task
        if loop is None or task is None:
            return
        # call_soon_threadsafe is the documented bridge from "any thread"
        # to "code that runs inside the loop".
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            # Loop already closed: nothing to do.
            pass


class PipelineController(QObject):
    """Public API exposed to GUI widgets.

    Wire the signals up *once*, then start/stop as many times as you
    like. ``is_running`` mirrors whether a worker thread is active.
    """

    transcript_delta = Signal(str, str)
    transcript_final = Signal(str, str)
    translation_delta = Signal(str, str)
    translation_final = Signal(str, str)
    connection_status = Signal(str, str)
    preview_delta = Signal(str, str)
    preview_reset = Signal(str)
    # M3 agent events: (item_id, agent_id, text).
    agent_delta = Signal(str, str, str)
    agent_final = Signal(str, str, str)
    agent_skipped = Signal(str, str, str)
    # High-level lifecycle, easier to bind UI elements to than the
    # internal QThread.started/finished.
    #
    # * ``starting`` fires synchronously from ``start()`` (the click
    #   handler is still running) so the UI can immediately flip the
    #   button to "启动中…" without waiting on the worker.
    # * ``started`` fires from the worker thread once the asyncio
    #   loop is up and the main pipeline task is scheduled.
    # * ``stopping`` mirrors ``starting`` for the Stop click.
    # * ``stopped`` fires once the worker thread has fully torn down.
    starting = Signal()
    started = Signal()
    stopping = Signal()
    stopped = Signal()
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, cfg: AppConfig, device: DeviceInfo) -> None:
        if self.is_running:
            log.warning("PipelineController.start called while already running; ignoring")
            return

        sink = SignalSink(
            SinkSignals(
                transcript_delta=self.transcript_delta,
                transcript_final=self.transcript_final,
                translation_delta=self.translation_delta,
                translation_final=self.translation_final,
                connection_status=self.connection_status,
                preview_delta=self.preview_delta,
                preview_reset=self.preview_reset,
                agent_delta=self.agent_delta,
                agent_final=self.agent_final,
                agent_skipped=self.agent_skipped,
            )
        )

        thread = QThread()
        worker = _Worker(cfg, device, sink)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        worker.error.connect(self.error)
        # Cross-thread queued connection: the worker fires ``running``
        # from inside its own thread, Qt re-marshals it onto the GUI
        # thread before our ``started`` signal emits. This is what
        # lets us drop the old ``wait_until_ready`` blocking wait.
        worker.running.connect(self.started)

        self._thread = thread
        self._worker = worker

        # Tell the GUI we're transitioning *before* spawning the thread
        # so the click feels instant. The actual pipeline boot happens
        # asynchronously after thread.start().
        self.starting.emit()
        thread.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        worker = self._worker
        if worker is not None:
            # Same trick on the way out: tell the UI we're stopping now,
            # so the button can flip to "停止中…" before the (potentially
            # multi-second) teardown begins.
            self.stopping.emit()
            worker.request_stop()

    def _on_thread_finished(self) -> None:
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.deleteLater()
        self.stopped.emit()

    def shutdown(self, wait_ms: int = 5000) -> None:
        """Synchronously stop and join. Call from QApplication aboutToQuit."""
        if not self.is_running:
            return
        self.stop()
        thread = self._thread
        if thread is not None:
            thread.wait(wait_ms)
