"""Vosk-based offline streaming ASR provider.

Two operating modes
-------------------
``preview_only=False`` (canonical mode):
    Vosk drives both the live preview row AND the canonical sentence
    text used for translation. ``TranscriptFinal`` is emitted on
    silence-based sentence boundaries. This is what you get when the
    user picks "本地 Vosk" as the ASR backend.

``preview_only=True`` (preview-only mode):
    Vosk emits ONLY ``LocalPreviewDelta`` events -- never
    ``TranscriptFinal``. Used as a companion engine alongside a
    higher-latency cloud ASR (OpenAI Realtime) so the user still sees
    words-as-they-are-spoken without the network round-trip, while
    the canonical sentence comes from the cloud. The pipeline calls
    ``reset_preview()`` to start a fresh utterance whenever the cloud
    ASR commits a final.

How a sentence becomes a sentence (canonical mode)
--------------------------------------------------
Vosk's own VAD emits a "segment ended" event on EVERY pause (~300ms),
which would chop a single utterance into many fragments and
fire one translation per fragment -- bad UX. Instead we:

1. Keep Vosk in "accumulate" mode: every fragment Vosk finalises is
   appended to an internal running buffer; the user sees the whole
   buffer (committed fragments + current partial) in the preview row.
2. Watch for *true* silence: if the running text stops changing for
   ``finalize_after_silence_s`` seconds, we treat that as the real
   sentence boundary, emit a single ``TranscriptFinal`` with the
   accumulated text, then reset Vosk so the next utterance starts
   from scratch.

Effectively the preview row is the user's live feedback ("words appear
as I speak"); ``TranscriptFinal`` only fires when the speaker actually
pauses. That gives the translator coherent sentence-sized chunks while
keeping perceived latency at zero.

Trade-offs vs OpenAI Realtime ASR
---------------------------------
* Free, offline, China-friendly -- zero network round-trips, zero
  per-minute cost.
* Quality is lower (Vosk's small models trade accuracy for speed).
  Punctuation is largely missing; rare words get guessed.
* Sentence boundaries depend on the speaker actually pausing. For
  back-to-back narration the user may want to lower
  ``finalize_after_silence_s`` so the translation keeps up.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from typing import AsyncIterator, Optional

from ...config import LocalASRConfig
from ...events import (
    ConnectionStatus,
    LocalPreviewDelta,
    TranscriptFinal,
)
from ..base import ASREvent
from .local_vosk import VoskLocalASR
from .vosk_model import is_model_present, model_path

log = logging.getLogger(__name__)


class VoskASRProvider:
    """``StreamingASRProvider`` backed by a local Vosk model.

    Parameters
    ----------
    cfg:
        Local ASR config (model id, silence cutoff, ...).
    preview_only:
        When True, Vosk emits ``LocalPreviewDelta`` events only and
        never ``TranscriptFinal``. The silence-watcher coroutine is
        not started at all. Used when this provider runs alongside a
        higher-quality canonical ASR (OpenAI Realtime) -- the cloud
        owns sentence boundaries and translation; we just contribute
        the instant preview text.
    """

    def __init__(
        self, cfg: LocalASRConfig, preview_only: bool = False
    ) -> None:
        self._cfg = cfg
        self._preview_only = preview_only
        self._engine: Optional[VoskLocalASR] = None
        self._feeder_task: Optional[asyncio.Task] = None
        self._finalizer_task: Optional[asyncio.Task] = None
        self._event_q: Optional[asyncio.Queue[ASREvent]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # State driven by the Vosk worker thread, read by the
        # silence-watcher coroutine running on the asyncio loop.
        # Plain attributes are fine: int/float writes are atomic in
        # CPython and we only ever care about the latest value.
        self._current_text: str = ""
        self._last_change_at: float = 0.0
        self._current_item_id: str = ""

    # --- StreamingASRProvider --------------------------------------

    async def run(
        self, audio_queue: "asyncio.Queue[bytes]"
    ) -> AsyncIterator[ASREvent]:
        if not is_model_present(self._cfg.model):
            yield ConnectionStatus(
                state="error",
                detail=(
                    f"Vosk 模型 {self._cfg.model} 未下载。"
                    "请在设置 → 语音识别 中点击下载。"
                ),
            )
            return

        self._loop = asyncio.get_running_loop()
        self._event_q = asyncio.Queue()
        self._current_item_id = self._new_item_id()
        self._current_text = ""
        self._last_change_at = time.monotonic()

        self._engine = VoskLocalASR(
            model_path=model_path(self._cfg.model),
            min_partial_chars=self._cfg.min_partial_chars,
        )

        # Vosk worker -> our state. Callbacks run on the Vosk thread;
        # ``_push_threadsafe`` is the only way to talk to the asyncio
        # side. We update plain attributes here and let the
        # silence-watcher coroutine flush them when appropriate.
        def _on_partial(text: str) -> None:
            # ``text`` is the FULL preview (committed segments +
            # current partial) thanks to accumulate=True. Empty
            # strings happen briefly after Vosk emits a segment_end;
            # don't reset our timer in that case or we'd never
            # finalise during a real pause.
            self._current_text = text
            if text:
                self._last_change_at = time.monotonic()
            # Tagging the preview with the *current* item id lets the
            # GUI promote the same row to the canonical entry when
            # the matching TranscriptFinal arrives later -- no
            # row-swap juggling needed downstream.
            ev = LocalPreviewDelta(item_id=self._current_item_id, text=text)
            self._push_threadsafe(ev)

        self._engine.start(on_partial=_on_partial, accumulate=True)

        # Block briefly for model load. Small models load in ~1-3s on
        # a modern laptop CPU; large ones can take 10-20s.
        ready = await asyncio.get_running_loop().run_in_executor(
            None, lambda: self._engine.wait_until_ready(timeout=30.0)
        )
        if not ready:
            yield ConnectionStatus(
                state="error",
                detail="Vosk 模型加载失败。请检查模型文件完整性。",
            )
            return

        yield ConnectionStatus(
            state="connected",
            detail=f"Vosk 本地模型已加载 ({self._cfg.model})",
        )

        # Audio uplink: bounded queue -> Vosk feed. Runs as its own
        # task so the event iterator doesn't block on I/O.
        async def _feeder() -> None:
            try:
                while True:
                    chunk = await audio_queue.get()
                    if not chunk:
                        continue
                    eng = self._engine
                    if eng is not None:
                        eng.feed(chunk)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Vosk audio feeder crashed")

        # Silence watcher: emit TranscriptFinal when the partial
        # stops changing for ``finalize_after_silence_s`` seconds.
        async def _silence_watcher() -> None:
            poll_s = max(0.05, min(0.2, self._cfg.finalize_after_silence_s / 5))
            try:
                while True:
                    await asyncio.sleep(poll_s)
                    text = self._current_text
                    if not text:
                        continue
                    if (time.monotonic() - self._last_change_at) < self._cfg.finalize_after_silence_s:
                        continue
                    # Sentence boundary: snapshot text, reset state,
                    # emit the canonical final, then nuke Vosk's
                    # internal recognizer so the next utterance
                    # starts at a clean baseline.
                    item_id = self._current_item_id
                    finalized_text = text.strip()
                    self._current_text = ""
                    self._current_item_id = self._new_item_id()
                    self._last_change_at = time.monotonic()
                    eng = self._engine
                    if eng is not None:
                        try:
                            eng.reset()
                        except Exception:
                            log.debug("Vosk reset after finalize raised", exc_info=True)
                    if finalized_text:
                        assert self._event_q is not None
                        await self._event_q.put(
                            TranscriptFinal(item_id=item_id, text=finalized_text)
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Vosk silence watcher crashed")

        self._feeder_task = asyncio.create_task(_feeder(), name="vosk-feeder")
        if not self._preview_only:
            # Canonical mode owns sentence boundaries; preview-only
            # mode lets the cloud ASR own them and would otherwise
            # double-emit finals.
            self._finalizer_task = asyncio.create_task(
                _silence_watcher(), name="vosk-silence"
            )

        try:
            while True:
                ev = await self._event_q.get()
                yield ev
        finally:
            await self._teardown()

    # --- preview-only coordination ---------------------------------

    def reset_preview(self) -> None:
        """Start a fresh preview utterance (preview-only mode only).

        Called by the pipeline when the canonical (cloud) ASR commits
        a TranscriptFinal. We:

        * Bump the preview ``item_id`` so the next batch of
          ``LocalPreviewDelta`` events lands in a brand-new subtitle
          row instead of mutating the now-finalised one.
        * Reset Vosk's internal recogniser so the accumulator starts
          empty -- otherwise the next preview would still show the
          text that just got finalised by the cloud, doubling up.

        Safe to call from any thread; the Vosk engine has its own
        internal lock and the attribute writes are atomic in CPython.
        """
        self._current_text = ""
        self._current_item_id = self._new_item_id()
        self._last_change_at = time.monotonic()
        engine = self._engine
        if engine is None:
            return
        try:
            engine.reset()
        except Exception:
            log.debug("Vosk reset during preview reset raised", exc_info=True)

    def current_preview_item_id(self) -> str:
        """Return the ``item_id`` of the preview row currently being
        built (so the pipeline can emit a matching ``LocalPreviewReset``
        before bumping to a fresh utterance)."""
        return self._current_item_id

    async def aclose(self) -> None:
        await self._teardown()

    # --- internals --------------------------------------------------

    def _push_threadsafe(self, ev: ASREvent) -> None:
        loop = self._loop
        q = self._event_q
        if loop is None or q is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(q.put(ev), loop)
        except RuntimeError:
            # Loop closed during shutdown.
            pass

    @staticmethod
    def _new_item_id() -> str:
        # Short UUID prefix is enough to disambiguate within a session.
        return "vosk-" + uuid.uuid4().hex[:8]

    async def _teardown(self) -> None:
        for task_attr in ("_feeder_task", "_finalizer_task"):
            t = getattr(self, task_attr, None)
            if t is not None:
                t.cancel()
                with contextlib.suppress(BaseException):
                    await t
                setattr(self, task_attr, None)
        engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                log.debug("Vosk engine stop raised", exc_info=True)
            self._engine = None
