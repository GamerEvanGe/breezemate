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
        self._ready = threading.Event()

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """Block the caller until the worker's event loop is running.

        Needed so ``request_stop`` can be safely scheduled even if the
        user immediately clicks Stop after Start.
        """
        return self._ready.wait(timeout=timeout)

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
            self.finished.emit()

    def request_stop(self) -> None:
        """Thread-safe cancellation. Called from the GUI thread."""
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
    # High-level lifecycle, easier to bind UI elements to than the
    # internal QThread.started/finished.
    started = Signal()
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

        self._thread = thread
        self._worker = worker

        thread.start()
        # Wait briefly for the loop to spin up. If it never becomes
        # ready, the worker probably crashed during setup -- the error
        # signal will fire shortly and stop() becomes a no-op.
        worker.wait_until_ready(timeout=3.0)
        self.started.emit()

    def stop(self) -> None:
        if not self.is_running:
            return
        worker = self._worker
        if worker is not None:
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
