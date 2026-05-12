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


# ---------------------------------------------------------------------- Agents
#
# M3 introduces an optional second LLM stream that consumes the
# finalised transcript+translation pair and produces auxiliary content
# (vocabulary glosses, idiom explanations, interview-style answers,
# etc.). Its output flows into its own floating window via these three
# events, all keyed by the SAME ``item_id`` as the upstream transcript
# so the UI can group them with the sentence that triggered them.


@dataclass(frozen=True)
class AgentDelta:
    """Incremental token stream of an agent reply for one transcript turn."""

    item_id: str
    agent_id: str  # e.g. "supplement" / "interviewee"
    text_so_far: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class AgentFinal:
    """Completed agent reply for one transcript turn."""

    item_id: str
    agent_id: str
    text: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class AgentSkipped:
    """The agent looked at the turn and decided not to produce output.

    Lets the UI optionally show a "(skipped)" placeholder or, more
    commonly, just suppress the row entirely. Used heavily by the
    interview agent, which only responds when the speaker asked a
    real question.
    """

    item_id: str
    agent_id: str
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


PipelineEvent = Union[
    TranscriptDelta,
    TranscriptFinal,
    TranslationDelta,
    TranslationFinal,
    LocalPreviewDelta,
    LocalPreviewReset,
    ConnectionStatus,
    AgentDelta,
    AgentFinal,
    AgentSkipped,
]
