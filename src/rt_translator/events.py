"""Event dataclasses that flow between pipeline stages.

Every utterance carries an ``item_id`` so the UI can update the right row
when partial deltas arrive and translations stream in after the fact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass(frozen=True)
class TranscriptDelta:
    """Incremental partial transcript for an in-progress utterance."""

    item_id: str
    text: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TranscriptFinal:
    """Final transcript for a completed utterance (server VAD detected end)."""

    item_id: str
    text: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TranslationDelta:
    """Incremental translation token stream for a finalised utterance."""

    item_id: str
    text_so_far: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TranslationFinal:
    """Complete translation for a finalised utterance."""

    item_id: str
    text: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class LocalPreviewDelta:
    """Word-level rough transcript produced by the *local* ASR (Vosk).

    Now carries the ``item_id`` of the sentence Vosk is currently
    building toward. The UI uses this to materialise (or update) the
    same subtitle entry that will later become canonical when the
    matching ``TranscriptFinal`` arrives -- no separate "preview row"
    widget any more; preview text lives in the main scrolling area
    just like any other entry, but rendered with the preview styling
    until finalised.
    """

    item_id: str
    text: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class LocalPreviewReset:
    """The local ASR abandoned its in-flight preview (e.g. silence
    without any speech).

    Carries the ``item_id`` of the preview entry to drop so the UI
    can clear it without disturbing finalised neighbours. Currently
    unused by the Vosk provider (which finalises through
    ``TranscriptFinal`` instead) but kept so future ASR backends can
    cancel a stale partial without leaving an orphan row on screen.
    """

    item_id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ConnectionStatus:
    """ASR provider connection state change. Drives the UI status indicator."""

    state: Literal["connecting", "connected", "reconnecting", "disconnected", "error"]
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


PipelineEvent = Union[
    TranscriptDelta,
    TranscriptFinal,
    TranslationDelta,
    TranslationFinal,
    LocalPreviewDelta,
    LocalPreviewReset,
    ConnectionStatus,
]
