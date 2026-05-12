"""OpenAI Realtime API streaming ASR provider.

Connects via WebSocket to ``wss://api.openai.com/v1/realtime?intent=transcription``,
configures a server-VAD-driven transcription session, streams 16 kHz mono
PCM16 audio chunks (resampled to 24 kHz on the way out, since the Realtime
API only accepts 24 kHz pcm16), and yields TranscriptDelta / TranscriptFinal
events as the model commits partial / full transcriptions.

Roles in the larger pipeline
----------------------------
This is the *canonical* ASR backend: its ``TranscriptFinal`` events drive
the translation stage and the eventually-stored polished sentence text.
A second, *preview* ASR (the local Vosk recogniser) runs in parallel and
emits ``LocalPreviewDelta`` events so the subtitle overlay can show the
user words-as-they-are-spoken without waiting for OpenAI's network
round-trip. The two are coordinated by the pipeline: whenever this
provider commits a ``TranscriptFinal``, the pipeline asks the preview
provider to start a fresh utterance so the preview row does not get
left behind as a stale ghost.

Reconnect strategy
------------------
On any network error / closed websocket, we emit a ``ConnectionStatus``
"reconnecting" event and back off exponentially (capped at MAX_BACKOFF_S)
until either we reconnect or the caller cancels the task. Audio that
arrives while we are disconnected is dropped silently -- buffering it
would only cause the next session to play minutes of stale audio.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
from typing import AsyncIterator, Optional

import numpy as np
import soxr
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from ...config import ASRConfig, ProviderEndpoint
from ...events import ConnectionStatus, TranscriptDelta, TranscriptFinal
from ..base import ASREvent

log = logging.getLogger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
INITIAL_BACKOFF_S = 0.5
MAX_BACKOFF_S = 8.0

# Server -> client event types we care about.
EVENT_DELTA = "conversation.item.input_audio_transcription.delta"
EVENT_FINAL = "conversation.item.input_audio_transcription.completed"
EVENT_SPEECH_STARTED = "input_audio_buffer.speech_started"
EVENT_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
EVENT_COMMITTED = "input_audio_buffer.committed"
EVENT_ERROR = "error"

# Liveness / housekeeping events we silently drop. They're useful for
# debugging WS-level issues (which is why DEBUG_WS logs them) but they
# don't influence what the UI shows.
_LIVENESS_EVENTS = frozenset(
    {
        EVENT_SPEECH_STARTED,
        EVENT_SPEECH_STOPPED,
        EVENT_COMMITTED,
        "transcription_session.created",
        "transcription_session.updated",
        "session.created",
        "session.updated",
        "conversation.item.created",
        "input_audio_buffer.cleared",
    }
)

DEBUG_WS = os.environ.get("RT_TRANSLATOR_DEBUG_WS", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)


class OpenAIRealtimeASR:
    """``StreamingASRProvider`` backed by OpenAI's Realtime transcription endpoint.

    The instance is single-use across reconnect cycles -- one ``run()``
    call drives the WS connection forever until the consumer cancels or
    ``aclose()`` is invoked.

    Per-utterance text accumulation
    -------------------------------
    OpenAI's ``...transcription.delta`` events carry only the *new* text
    since the previous delta (a few characters at a time). Downstream
    components (the subtitle window in particular) expect cumulative
    text in ``TranscriptDelta.text``, so we maintain a tiny
    ``item_id -> running_text`` dict and emit the full running text on
    every delta. The dict is pruned when the matching ``...completed``
    event arrives.
    """

    def __init__(
        self,
        asr_cfg: ASRConfig,
        endpoint: ProviderEndpoint,
    ) -> None:
        self._asr = asr_cfg
        self._endpoint = endpoint
        self._closed = False
        self._ws = None  # active connection (set inside run loop)
        self._accum: dict[str, str] = {}

    # --- StreamingASRProvider --------------------------------------

    async def run(
        self, audio_queue: "asyncio.Queue[bytes]"
    ) -> AsyncIterator[ASREvent]:
        try:
            api_key = self._endpoint.resolve_api_key()
        except RuntimeError as e:
            yield ConnectionStatus(state="error", detail=str(e))
            return
        if not api_key:
            yield ConnectionStatus(
                state="error",
                detail="OpenAI Realtime 需要 OPENAI_API_KEY，请在设置中填写。",
            )
            return

        # Server expects 24 kHz pcm16. Our capture pipeline produces
        # 16 kHz, so we keep a streaming resampler that's reused across
        # reconnects (its internal buffer is per-utterance and
        # harmless across short disconnects).
        resampler = soxr.ResampleStream(16_000, 24_000, 1, dtype="int16")

        backoff = INITIAL_BACKOFF_S
        attempt = 0
        while not self._closed:
            attempt += 1
            yield ConnectionStatus(
                state="connecting" if attempt == 1 else "reconnecting",
                detail=f"OpenAI Realtime · {self._asr.model or 'gpt-4o-mini-transcribe'}",
            )
            log.info("Connecting to OpenAI Realtime (attempt %d)", attempt)

            try:
                async with ws_connect(
                    REALTIME_URL,
                    additional_headers={
                        "Authorization": f"Bearer {api_key}",
                        "OpenAI-Beta": "realtime=v1",
                    },
                    max_size=None,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    self._ws = ws
                    await self._configure(ws)
                    yield ConnectionStatus(
                        state="connected",
                        detail=f"OpenAI Realtime ({self._asr.model or 'gpt-4o-mini-transcribe'})",
                    )
                    backoff = INITIAL_BACKOFF_S

                    sender = asyncio.create_task(
                        self._send_loop(ws, audio_queue, resampler),
                        name="oai-rt-sender",
                    )
                    try:
                        async for raw in ws:
                            try:
                                msg = json.loads(raw)
                            except Exception:
                                log.debug("Non-JSON WS frame: %r", raw[:120])
                                continue
                            if DEBUG_WS:
                                log.debug("WS event: %s", msg.get("type"))
                            ev = self._dispatch(msg)
                            if ev is not None:
                                yield ev
                    finally:
                        sender.cancel()
                        with contextlib.suppress(BaseException):
                            await sender
            except ConnectionClosed as e:
                if self._closed:
                    return
                log.warning("Realtime WS closed: %s", e)
                yield ConnectionStatus(state="reconnecting", detail=f"WS closed: {e}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._closed:
                    return
                log.exception("OpenAI Realtime connection failed")
                yield ConnectionStatus(
                    state="error", detail=f"{type(e).__name__}: {e}"
                )
            finally:
                self._ws = None

            if self._closed:
                return

            await asyncio.sleep(backoff)
            backoff = min(MAX_BACKOFF_S, backoff * 1.7)

    async def aclose(self) -> None:
        self._closed = True
        ws = self._ws
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
            self._ws = None

    # --- internals --------------------------------------------------

    async def _configure(self, ws) -> None:
        """Send the one-shot ``transcription_session.update`` that tells
        the server what model + VAD + audio format we want."""
        session: dict = {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": self._asr.model or "gpt-4o-mini-transcribe",
            },
            "turn_detection": {
                "type": "server_vad",
                # 0.5 is the documented default; lower = more sensitive,
                # higher = ignore quiet speech. Stay with the default --
                # ASR quality regressions from VAD tuning aren't worth
                # the marginal latency win.
                "threshold": 0.5,
                # Helps the model anchor partial words at the start of
                # an utterance. 300 ms is the documented sweet spot.
                "prefix_padding_ms": 300,
                # How long the speaker has to actually stop before the
                # server emits ``...completed``. Lower = more responsive
                # but more sentence chops; higher = waits for true
                # silence. 200 ms = quick handoff to the translator.
                "silence_duration_ms": 200,
            },
        }
        if self._asr.language:
            session["input_audio_transcription"]["language"] = self._asr.language

        msg = {"type": "transcription_session.update", "session": session}
        await ws.send(json.dumps(msg))
        log.debug("Sent transcription_session.update for model=%s lang=%s",
                  session["input_audio_transcription"]["model"],
                  session["input_audio_transcription"].get("language"))

    async def _send_loop(
        self,
        ws,
        audio_queue: "asyncio.Queue[bytes]",
        resampler: soxr.ResampleStream,
    ) -> None:
        """Audio uplink: pop 16 kHz PCM16 chunks, resample to 24 kHz,
        base64-encode, push as ``input_audio_buffer.append`` frames.

        Errors here are intentionally swallowed: a broken send means
        the WS is going down anyway and the surrounding ``async for``
        will pick up the ConnectionClosed and trigger reconnect.
        """
        try:
            while True:
                chunk16 = await audio_queue.get()
                if not chunk16:
                    continue
                arr16 = np.frombuffer(chunk16, dtype=np.int16)
                if arr16.size == 0:
                    continue
                # soxr expects shape (n,) or (n, channels); mono int16
                # works directly. ``last=False`` keeps internal state
                # for the next chunk -- crucial for streaming quality.
                arr24 = resampler.resample_chunk(arr16, last=False)
                if arr24.size == 0:
                    continue
                payload = arr24.astype(np.int16).tobytes()
                b64 = base64.b64encode(payload).decode("ascii")
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.append", "audio": b64})
                )
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            return
        except Exception:
            log.exception("Realtime audio sender crashed")

    def _dispatch(self, msg: dict) -> Optional[ASREvent]:
        etype = msg.get("type", "")

        if etype == EVENT_DELTA:
            item_id = msg.get("item_id", "")
            delta = msg.get("delta", "")
            if not item_id or delta is None:
                return None
            full_id = f"oai-{item_id}"
            cumulative = self._accum.get(full_id, "") + delta
            self._accum[full_id] = cumulative
            return TranscriptDelta(item_id=full_id, text=cumulative)

        if etype == EVENT_FINAL:
            item_id = msg.get("item_id", "")
            transcript = msg.get("transcript", "")
            if not item_id:
                return None
            full_id = f"oai-{item_id}"
            # Prefer the server's authoritative ``transcript`` string;
            # fall back to whatever we've been accumulating if it's
            # empty (rare, but seen on some short utterances).
            text = (transcript or self._accum.get(full_id, "")).strip()
            self._accum.pop(full_id, None)
            if not text:
                return None
            return TranscriptFinal(item_id=full_id, text=text)

        if etype == EVENT_ERROR:
            err = msg.get("error", {}) if isinstance(msg.get("error"), dict) else {}
            code = err.get("code") or err.get("type") or "error"
            detail = err.get("message", "")
            # Server occasionally complains the buffer was too small to
            # commit (e.g. silence right after VAD triggered). It's
            # benign -- the next utterance proceeds normally -- so just
            # drop it to keep the status bar quiet.
            if code in {"input_audio_buffer_commit_empty"}:
                log.debug("Ignoring benign %s: %s", code, detail)
                return None
            log.warning("Realtime error %s: %s", code, detail)
            return ConnectionStatus(state="error", detail=f"{code}: {detail}")

        if etype in _LIVENESS_EVENTS:
            return None

        if etype:
            log.debug("Unhandled WS event: %s", etype)
        return None
