"""Local Vosk-based streaming ASR for sub-second word-level previews.

Runs entirely in its own background thread. Audio chunks come in via
``feed()`` (called by ``AudioCapture`` from the recorder thread); rough
partial transcripts go out via the ``on_partial`` callback (typically a
function that schedules an event onto the asyncio loop).

This is *not* a replacement for the OpenAI Realtime ASR -- the partials
are fuzzy and lack punctuation. They exist purely so the user gets
something on screen within ~100ms of speaking, while OpenAI's polished
transcript is still being produced ~1-2s later.

Design notes:

* We don't trust Vosk's own segmentation: ``AcceptWaveform`` returning
  True (which Vosk uses to say "this segment is done") gets translated
  into a ``LocalPreviewReset`` so the UI can clear the preview row,
  but the canonical sentence boundary always comes from OpenAI.
* Dedupe partials: Vosk emits the same partial repeatedly until it
  changes, which would churn the UI. We only forward changed text.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)


# Sample rate / channel layout fed to ``feed()``. Must match what
# ``audio_capture`` produces (16 kHz mono PCM16).
EXPECTED_SAMPLE_RATE = 16_000


class VoskLocalASR:
    """Threaded Vosk wrapper with a partial-only output contract.

    Lifecycle:

        asr = VoskLocalASR(model_path)
        asr.start(on_partial=..., on_segment_end=...)
        # ... feed PCM16 bytes via asr.feed(chunk) ...
        asr.stop()
    """

    def __init__(
        self,
        model_path: Path,
        min_partial_chars: int = 2,
        queue_max: int = 200,
    ) -> None:
        self._model_path = Path(model_path)
        self._min_partial_chars = max(1, int(min_partial_chars))
        # queue_max is in CHUNKS, not bytes. With 50ms chunks at 16kHz
        # that's 10s of buffered audio max. Beyond that we drop on the
        # producer side so the recorder thread never blocks.
        self._audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=queue_max)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._on_partial: Optional[Callable[[str], None]] = None
        self._on_segment_end: Optional[Callable[[str], None]] = None
        self._accumulate: bool = True

        # Sentence-level accumulation. Vosk has its own VAD and will
        # emit "segment finalised" mid-sentence on short pauses (e.g.
        # between clauses). If we cleared the preview row on every
        # segment, the user would see the text disappear at every
        # comma. Instead we keep the finalised segments in
        # ``_committed_segments`` and append the current ``_partial``
        # for display purposes. The preview row only clears when the
        # *canonical* sentence boundary arrives from OpenAI (which
        # triggers ``reset()`` from the pipeline).
        self._committed_segments: list[str] = []
        self._partial: str = ""
        self._last_emitted: str = ""

        # Loaded inside the worker thread to keep start() snappy.
        self._model = None
        self._recognizer = None
        self._ready_event = threading.Event()
        self._init_error: Optional[BaseException] = None

    # ------------------------------------------------------------------ Public

    def start(
        self,
        on_partial: Callable[[str], None],
        on_segment_end: Optional[Callable[[str], None]] = None,
        accumulate: bool = True,
    ) -> None:
        """Start the worker thread.

        Two usage modes, picked by ``accumulate``:

        * ``accumulate=True`` (default, used by the preview row): the
          worker keeps a running buffer of all Vosk-finalised segments
          plus the current partial, and ``on_partial(full_text)`` is
          called with the COMPLETE preview string. ``on_segment_end``
          is ignored. The preview only clears when ``reset()`` is
          called (typically when OpenAI's canonical final arrives).

        * ``accumulate=False`` (used by the full-ASR provider):
          ``on_partial(text)`` is the in-progress text for the
          CURRENT segment only -- short, like a running guess. When
          Vosk's VAD finalises a segment, ``on_segment_end(text)``
          fires with that segment's canonical text. The caller is
          responsible for stitching segments into sentences.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("VoskLocalASR already running")
        self._on_partial = on_partial
        self._on_segment_end = on_segment_end
        self._accumulate = bool(accumulate)
        self._stop_event.clear()
        self._ready_event.clear()
        self._init_error = None
        self._committed_segments = []
        self._partial = ""
        self._last_emitted = ""
        self._thread = threading.Thread(
            target=self._run, name="rt-vosk-local-asr", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Wake up the worker if it's blocked on queue.get.
        try:
            self._audio_q.put_nowait(None)
        except queue.Full:
            pass
        t = self._thread
        if t is not None:
            t.join(timeout=3.0)
        self._thread = None
        self._model = None
        self._recognizer = None

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Block until the model has finished loading. Returns False on
        timeout or worker initialisation error."""
        ok = self._ready_event.wait(timeout=timeout)
        return ok and self._init_error is None

    def feed(self, pcm16_bytes: bytes) -> None:
        """Push audio from the capture thread. Drops on overflow."""
        if not pcm16_bytes or self._stop_event.is_set():
            return
        try:
            self._audio_q.put_nowait(pcm16_bytes)
        except queue.Full:
            # Dropping a chunk only affects local-preview accuracy; the
            # canonical OpenAI stream gets its own copy.
            log.debug("Vosk feed queue full; dropping %d byte chunk", len(pcm16_bytes))

    # ------------------------------------------------------------------ Worker

    def _run(self) -> None:
        try:
            # vosk import is local so it's not paid by users who never
            # turn on local ASR.
            import vosk
        except ImportError as e:
            self._init_error = e
            self._ready_event.set()
            log.error("vosk not installed: %s", e)
            return

        # Vosk prints to stderr by default which would garble the
        # console; silence it once at startup.
        try:
            vosk.SetLogLevel(-1)
        except Exception:
            pass

        try:
            log.info("Loading Vosk model from %s ...", self._model_path)
            self._model = vosk.Model(str(self._model_path))
            self._recognizer = vosk.KaldiRecognizer(self._model, EXPECTED_SAMPLE_RATE)
            # Enable word-level confidence/timing in results; harmless
            # for our use case and useful when debugging.
            try:
                self._recognizer.SetWords(True)
            except Exception:
                pass
        except Exception as e:
            self._init_error = e
            self._ready_event.set()
            log.exception("Failed to construct Vosk recognizer: %s", e)
            return

        self._ready_event.set()
        log.info("Vosk recognizer ready (model=%s)", self._model_path.name)

        try:
            self._loop()
        except Exception:
            log.exception("Vosk worker crashed")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                # Sentinel from stop().
                break
            try:
                self._process_chunk(chunk)
            except Exception:
                log.exception("Vosk processing error on chunk of %d bytes", len(chunk))

    def _process_chunk(self, chunk: bytes) -> None:
        rec = self._recognizer
        if rec is None:
            return

        if rec.AcceptWaveform(chunk):
            # Vosk's own VAD just finalised a segment.
            try:
                seg = json.loads(rec.Result())
                seg_text = (seg.get("text") or "").strip()
            except Exception:
                seg_text = ""

            if not self._accumulate:
                # ASR mode: fire the per-segment callback with the
                # canonical text and reset partial state. We do NOT
                # buffer across segments here -- the caller stitches
                # segments into sentences as it sees fit.
                self._partial = ""
                if seg_text and self._on_segment_end is not None:
                    try:
                        self._on_segment_end(seg_text)
                    except Exception:
                        log.debug("on_segment_end callback raised", exc_info=True)
                return

            # Preview mode: append to the running buffer, keep
            # displaying it, only clear on explicit ``reset()``.
            if seg_text:
                self._committed_segments.append(seg_text)
                # Safety cap: if OpenAI's transcript_final never
                # arrives (e.g. connection issues), the committed
                # list would grow forever. Keep only the last few so
                # the preview row stays roughly one-sentence-long.
                if len(self._committed_segments) > 6:
                    self._committed_segments = self._committed_segments[-6:]
            self._partial = ""
            self._emit_if_changed()
            return

        # In-progress partial. Read, dedupe, emit.
        try:
            raw = rec.PartialResult()
            obj = json.loads(raw)
        except Exception:
            return
        text = (obj.get("partial") or "").strip()
        # The dedupe gate applies only to the active partial. Once
        # text accumulates into committed_segments it's kept regardless.
        if text == self._partial:
            return
        # Reject very-short noise partials, but only when there's no
        # prior committed text -- otherwise a brief silence between
        # clauses would freeze the display while we wait for the
        # partial to grow back past min_partial_chars.
        if len(text) < self._min_partial_chars and not self._committed_segments:
            return
        self._partial = text

        if not self._accumulate:
            # ASR mode: just emit the current segment's partial.
            cb = self._on_partial
            if cb is not None:
                try:
                    cb(text)
                except Exception:
                    log.debug("on_partial callback raised", exc_info=True)
            return

        self._emit_if_changed()

    def _build_display(self) -> str:
        parts = [s for s in self._committed_segments if s]
        if self._partial:
            parts.append(self._partial)
        return " ".join(parts).strip()

    def _emit_if_changed(self) -> None:
        display = self._build_display()
        if display == self._last_emitted:
            return
        self._last_emitted = display
        cb = self._on_partial
        if cb is None:
            return
        try:
            cb(display)
        except Exception:
            log.debug("on_partial callback raised", exc_info=True)

    # ------------------------------------------------------------------ Misc

    def reset(self) -> None:
        """Re-create the recognizer to drop accumulated state.

        Called by the pipeline whenever the canonical (OpenAI) side
        has just finalised a sentence. This:

        * drops Vosk's internal partial-buffer state so the next
          chunk starts a fresh segment,
        * clears our own ``_committed_segments`` / ``_partial`` so
          the preview row's text doesn't carry over into the next
          sentence,
        * resets the dedupe baseline.
        """
        # Always clear our accumulation, even if the underlying model
        # never loaded -- this still gives the UI a clean slate.
        self._committed_segments = []
        self._partial = ""
        self._last_emitted = ""

        if self._model is None:
            return
        try:
            import vosk

            self._recognizer = vosk.KaldiRecognizer(self._model, EXPECTED_SAMPLE_RATE)
            try:
                self._recognizer.SetWords(True)
            except Exception:
                pass
        except Exception:
            log.exception("Vosk recognizer reset failed")
