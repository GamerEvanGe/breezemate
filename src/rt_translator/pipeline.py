"""Pipeline orchestration: wire audio -> ASR(s) -> [translation] -> display.

Two ASR backends are supported:

* ``vosk_local`` -- the embedded Vosk recogniser. Single-engine path:
  Vosk drives both the live preview row and the canonical sentence
  text used for translation.
* ``openai_realtime`` -- OpenAI's Realtime transcription API.
  Dual-engine path: OpenAI owns the canonical TranscriptDelta /
  TranscriptFinal stream (so the translation gets clean, polished
  text) while a *preview-only* Vosk instance runs in parallel and
  emits ``LocalPreviewDelta`` events so the user still sees words
  appear in real time without waiting on the cloud round-trip.

In the dual-engine case the audio capture pipeline fans out each PCM
chunk to BOTH ASR providers, and the pipeline coordinates them: every
canonical ``TranscriptFinal`` from OpenAI emits a ``LocalPreviewReset``
for the current preview row and asks the Vosk provider to start a
fresh utterance, so the preview row doesn't get left behind as a
stale ghost over the locked-in sentence.

Translation always goes out to whichever OpenAI-compatible chat
endpoint the user picked.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from typing import Optional, Protocol

from .audio_capture import AudioCapture
from .config import AppConfig
from .console_sink import ConsoleSink
from .device_picker import DeviceInfo
from .events import (
    LocalPreviewReset,
    PipelineEvent,
    TranscriptDelta,
    TranscriptFinal,
    TranslationFinal,
)
from .providers.asr.openai_realtime import OpenAIRealtimeASR
from .providers.asr.vosk_full import VoskASRProvider
from .providers.base import LLMTranslator, StreamingASRProvider
from .providers.llm.openai_compatible import OpenAICompatibleTranslator

log = logging.getLogger(__name__)


class EventSink(Protocol):
    """Anything that drains the display queue. Implemented by
    ``console_sink.ConsoleSink`` (CLI) and ``gui.signal_sink.SignalSink``
    (GUI)."""

    async def run(self, queue: "asyncio.Queue[PipelineEvent]") -> None: ...


def _build_canonical_asr(cfg: AppConfig) -> StreamingASRProvider:
    """Return the ASR backend whose ``TranscriptFinal`` drives translation."""
    if cfg.asr.provider == "openai_realtime":
        return OpenAIRealtimeASR(cfg.asr, cfg.openai_endpoint())
    if cfg.asr.provider not in ("vosk_local", ""):
        log.warning(
            "Unknown ASR provider %r; falling back to vosk_local.",
            cfg.asr.provider,
        )
    return VoskASRProvider(cfg.local_asr)


def _build_preview_asr(cfg: AppConfig) -> Optional[VoskASRProvider]:
    """Return the optional preview-only Vosk engine.

    Only used when the canonical ASR is a *cloud* backend (OpenAI
    Realtime). For ``vosk_local`` the canonical Vosk already drives
    the preview row, so a second engine would just be wasted CPU.
    """
    if cfg.asr.provider == "openai_realtime":
        return VoskASRProvider(cfg.local_asr, preview_only=True)
    return None


def _build_translator(cfg: AppConfig) -> LLMTranslator:
    # Every supported translator today speaks the OpenAI Chat
    # Completions protocol, so the choice of provider profile (openai /
    # deepseek / groq / openrouter / ollama / ...) is just a different
    # base_url + key. No code path forks per provider.
    return OpenAICompatibleTranslator(cfg.translator, cfg.translator_endpoint())


async def _asr_runner(
    provider: StreamingASRProvider,
    audio_q: "asyncio.Queue[bytes]",
    event_q: "asyncio.Queue[PipelineEvent]",
    name: str,
) -> None:
    try:
        async for ev in provider.run(audio_q):
            await event_q.put(ev)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("%s ASR runner crashed", name)


async def _audio_fanout(
    src: "asyncio.Queue[bytes]",
    dests: list["asyncio.Queue[bytes]"],
) -> None:
    """Copy every chunk from ``src`` into every destination queue.

    Uses ``put_nowait`` + silent drop on QueueFull. The canonical ASR
    is sensitive to back-pressure (a stalled WS send shouldn't pause
    the recorder); dropping a 50 ms chunk is preferable to a stalled
    capture thread.
    """
    try:
        while True:
            chunk = await src.get()
            for q in dests:
                try:
                    q.put_nowait(chunk)
                except asyncio.QueueFull:
                    # Slow consumer; dropping this chunk is the lesser
                    # evil compared to back-pressuring the recorder.
                    pass
    except asyncio.CancelledError:
        raise


async def _event_router(
    cfg: AppConfig,
    in_q: "asyncio.Queue[PipelineEvent]",
    out_q: "asyncio.Queue[PipelineEvent]",
    translator: Optional[LLMTranslator],
    preview_asr: Optional[VoskASRProvider],
) -> None:
    history: deque[tuple[str, str]] = deque(maxlen=cfg.translator.context_window)
    pending: dict[str, asyncio.Task] = {}
    # Tracks the last cloud item id we forwarded. When it changes
    # (= the cloud started a fresh utterance) we wipe the now-stale
    # local preview row and bump Vosk to a new preview item_id so
    # subsequent partials don't accidentally fall back into the
    # finalised row. Only used in dual-engine mode.
    last_cloud_item_id: Optional[str] = None

    async def translate_one(item_id: str, raw_text: str) -> None:
        try:
            assert translator is not None
            # The translator may emit a polished version of the source
            # (TranscriptFinal) before the translation deltas. We
            # remember it so the history pair stored for subsequent
            # turns is (polished_source, translation) rather than
            # (raw_ASR_source, translation) -- this keeps the
            # multi-turn context coherent for the LLM. Falls back to
            # the raw text if the model never emits a polished final.
            polished_source = raw_text
            async for tev in translator.translate_stream(
                item_id=item_id,
                text=raw_text,
                src_lang=cfg.asr.language or "en",
                tgt_lang=cfg.translator.target_lang,
                history=list(history),
            ):
                await out_q.put(tev)
                if isinstance(tev, TranscriptFinal):
                    polished_source = tev.text or raw_text
                elif isinstance(tev, TranslationFinal):
                    history.append((polished_source, tev.text))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Translation task crashed for item_id=%s", item_id)
        finally:
            pending.pop(item_id, None)

    try:
        while True:
            ev = await in_q.get()

            # In dual-engine mode (cloud canonical + Vosk preview):
            # whenever the cloud starts emitting events for a NEW
            # utterance, the in-progress local preview becomes a
            # stale duplicate of what the cloud is about to render in
            # its own polished row. Detect that on the very first
            # delta (or final) of each item_id, wipe the preview, and
            # tell Vosk to start a fresh utterance.
            if (
                preview_asr is not None
                and isinstance(ev, (TranscriptDelta, TranscriptFinal))
                and not ev.item_id.startswith("vosk-")
                and ev.item_id != last_cloud_item_id
            ):
                last_cloud_item_id = ev.item_id
                preview_id = preview_asr.current_preview_item_id()
                if preview_id:
                    await out_q.put(LocalPreviewReset(item_id=preview_id))
                preview_asr.reset_preview()

            await out_q.put(ev)

            if isinstance(ev, TranscriptFinal):
                if cfg.mode == "translate" and translator is not None:
                    # Fire-and-forget; multiple translations can stream in
                    # parallel which keeps the sink responsive even if one
                    # is slow.
                    task = asyncio.create_task(
                        translate_one(ev.item_id, ev.text),
                        name=f"translate-{ev.item_id}",
                    )
                    pending[ev.item_id] = task
    except asyncio.CancelledError:
        for t in pending.values():
            t.cancel()
        for t in pending.values():
            with contextlib.suppress(BaseException):
                await t
        raise


async def run_pipeline(
    cfg: AppConfig,
    device: DeviceInfo,
    sink: Optional[EventSink] = None,
) -> None:
    """Run the full pipeline. Returns when cancelled.

    ``sink`` is the event drain. If omitted, a CLI ``ConsoleSink`` is
    used. The GUI passes its own SignalSink so events fan out to Qt
    signals instead.
    """
    # Queues:
    #
    # capture_q          -- raw PCM16 chunks from AudioCapture.
    # canonical_audio_q  -- subset destined for the canonical ASR.
    # preview_audio_q    -- subset destined for the preview ASR
    #                       (only present when the canonical is cloud).
    # asr_event_q        -- canonical ASR's events (TranscriptDelta /
    #                       TranscriptFinal / ConnectionStatus). These
    #                       are what the event_router consumes and may
    #                       trigger translation tasks for.
    # display_q          -- final fan-out to the sink. Preview events
    #                       go here directly (they're not routed) so
    #                       the UI sees them with the lowest possible
    #                       latency.
    capture_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    canonical_audio_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    asr_event_q: asyncio.Queue[PipelineEvent] = asyncio.Queue()
    display_q: asyncio.Queue[PipelineEvent] = asyncio.Queue()

    canonical_asr = _build_canonical_asr(cfg)
    preview_asr = _build_preview_asr(cfg)
    preview_audio_q: Optional[asyncio.Queue[bytes]] = None
    if preview_asr is not None:
        preview_audio_q = asyncio.Queue(maxsize=200)

    translator = _build_translator(cfg) if cfg.mode == "translate" else None

    if sink is None:
        sink = ConsoleSink(
            mode=cfg.mode,
            display_cfg=cfg.display,
            source_label=f"{device.source}:{device.name}",
            target_lang=cfg.translator.target_lang,
        )

    capture = AudioCapture(
        cfg.audio,
        device,
        capture_q,
        status_queue=display_q,
    )
    capture.start(asyncio.get_running_loop())

    fanout_dests = [canonical_audio_q]
    if preview_audio_q is not None:
        fanout_dests.append(preview_audio_q)

    tasks: list[asyncio.Task] = [
        asyncio.create_task(
            _audio_fanout(capture_q, fanout_dests), name="audio-fanout"
        ),
        asyncio.create_task(
            _asr_runner(canonical_asr, canonical_audio_q, asr_event_q, "canonical"),
            name="asr-canonical",
        ),
        asyncio.create_task(
            _event_router(cfg, asr_event_q, display_q, translator, preview_asr),
            name="router",
        ),
        asyncio.create_task(sink.run(display_q), name="sink"),
    ]
    if preview_asr is not None and preview_audio_q is not None:
        # Preview events bypass the router (they don't need translation
        # or coordination) and go straight to the display queue.
        tasks.append(
            asyncio.create_task(
                _asr_runner(preview_asr, preview_audio_q, display_q, "preview"),
                name="asr-preview",
            )
        )

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        log.info("Shutting down pipeline...")
    finally:
        # Teardown order matters. We must stop the *audio capture
        # thread* before letting any of the ASR engines (Vosk's own
        # worker thread; OpenAI's WS sender) tear down -- otherwise
        # the capture thread keeps pushing PCM chunks into queues /
        # callbacks that point at half-freed C objects, and on
        # Windows + WASAPI a stuck recorder.record() call can leave
        # a zombie thread behind that races the next pipeline's
        # recorder. That race is exactly what produced the BEX64 /
        # 0xc0000409 crashes we used to see on quick stop/start
        # cycles. Sequence:
        #
        #   1. Cancel the asyncio tasks (router/sink/fanout/...).
        #      This stops them from doing more work BUT does not
        #      interrupt the C-level audio thread.
        #   2. Stop the audio capture thread (blocking join). After
        #      this returns, no more chunks enter the pipeline.
        #   3. aclose() the ASR engines + translator. They now own
        #      a quiescent audio stream, so their internal threads
        #      can join without risk of UAF.
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(BaseException):
                await t

        # capture.stop() is a synchronous join on the recorder
        # thread; running it in a thread executor frees the asyncio
        # loop to keep pumping during the join (some ASR aclose
        # paths await on the same loop). It also lets us put a hard
        # ceiling on how long the loop blocks here.
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, capture.stop)
        except Exception:
            log.exception("AudioCapture.stop raised during teardown")

        await canonical_asr.aclose()
        if preview_asr is not None:
            await preview_asr.aclose()
        if translator is not None:
            await translator.aclose()
        log.info("Pipeline stopped.")
