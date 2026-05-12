"""Drain the pipeline ``display_q`` and re-emit each event as a Qt signal.

Used by the GUI's ``PipelineController`` as a drop-in replacement for the
CLI's ``ConsoleSink``. Qt signals are thread-safe under
``Qt.QueuedConnection``, so it's fine that this runs inside the asyncio
worker thread while listeners live on the GUI thread.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..events import (
    ConnectionStatus,
    LocalPreviewDelta,
    LocalPreviewReset,
    PipelineEvent,
    TranscriptDelta,
    TranscriptFinal,
    TranslationDelta,
    TranslationFinal,
)

log = logging.getLogger(__name__)


@dataclass
class SinkSignals:
    """Lightweight container of the Qt signals we emit.

    Defined as plain attributes so this module imports cleanly even when
    PySide6 is not installed (CLI-only environments). The caller wires up
    actual ``Signal`` instances at construction time.
    """

    transcript_delta: object  # Signal(str item_id, str text)
    transcript_final: object  # Signal(str item_id, str text)
    translation_delta: object  # Signal(str item_id, str text_so_far)
    translation_final: object  # Signal(str item_id, str text)
    connection_status: object  # Signal(str state, str detail)
    preview_delta: object  # Signal(str item_id, str text)  - local Vosk partial
    preview_reset: object  # Signal(str item_id) - drop a stale preview row


class SignalSink:
    """Implements the ``EventSink`` Protocol from ``pipeline``."""

    def __init__(self, signals: SinkSignals) -> None:
        self.signals = signals

    async def run(self, queue: "asyncio.Queue[PipelineEvent]") -> None:
        try:
            while True:
                ev = await queue.get()
                self._dispatch(ev)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("SignalSink crashed")
            raise

    def _dispatch(self, ev: PipelineEvent) -> None:
        if isinstance(ev, TranscriptDelta):
            self.signals.transcript_delta.emit(ev.item_id, ev.text)
        elif isinstance(ev, TranscriptFinal):
            self.signals.transcript_final.emit(ev.item_id, ev.text)
        elif isinstance(ev, TranslationDelta):
            self.signals.translation_delta.emit(ev.item_id, ev.text_so_far)
        elif isinstance(ev, TranslationFinal):
            self.signals.translation_final.emit(ev.item_id, ev.text)
        elif isinstance(ev, LocalPreviewDelta):
            self.signals.preview_delta.emit(ev.item_id, ev.text)
        elif isinstance(ev, LocalPreviewReset):
            self.signals.preview_reset.emit(ev.item_id)
        elif isinstance(ev, ConnectionStatus):
            self.signals.connection_status.emit(ev.state, ev.detail)
