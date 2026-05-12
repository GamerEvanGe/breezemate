"""Provider Protocols.

These define the contracts that future ASR / translator backends must
implement. M1 ships one of each (OpenAI Realtime + OpenAI chat), but the
shape is locked down here so M2 can drop in faster-whisper, Deepgram,
Anthropic, etc. without touching the pipeline.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Protocol, Union, runtime_checkable

from ..events import (
    TranscriptDelta,
    TranscriptFinal,
    TranslationDelta,
    TranslationFinal,
    ConnectionStatus,
)


ASREvent = Union[TranscriptDelta, TranscriptFinal, ConnectionStatus]
TranslationEvent = Union[TranslationDelta, TranslationFinal]
# The translator may also emit TranscriptDelta / TranscriptFinal when
# it polishes the raw ASR output (adds punctuation, fixes obvious
# recognition slips) -- those overwrite the source-text line of the
# matching subtitle entry on the way to the translation. The pipeline
# event router and every sink already know how to handle them.
TranslatorOutput = Union[
    TranslationDelta,
    TranslationFinal,
    TranscriptDelta,
    TranscriptFinal,
]


@runtime_checkable
class StreamingASRProvider(Protocol):
    """A duplex streaming ASR backend.

    Implementations read raw 16 kHz mono int16 PCM byte chunks from the
    given queue and yield ASR events as they happen. We use a Queue (not
    an AsyncIterator) so reconnection logic can spawn fresh consumer
    tasks without worrying about an iterator that was closed by a
    cancellation in a previous attempt.
    """

    async def run(
        self, audio_queue: "asyncio.Queue[bytes]"
    ) -> AsyncIterator[ASREvent]:
        ...

    async def aclose(self) -> None:
        ...


@runtime_checkable
class LLMTranslator(Protocol):
    """A streaming text translator backed by a chat-completion-style LLM."""

    async def translate_stream(
        self,
        item_id: str,
        text: str,
        src_lang: str,
        tgt_lang: str,
        history: list[tuple[str, str]],
    ) -> AsyncIterator[TranslatorOutput]:
        ...

    async def aclose(self) -> None:
        ...
