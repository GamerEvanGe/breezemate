"""Map ``AgentConfig.mode`` strings to ``Agent`` subclasses.

Adding a new agent flavour: drop a new module under ``agents/``, then
register it in ``_AGENT_TYPES`` below. The pipeline + GUI surfaces
pick the new entry up automatically.
"""

from __future__ import annotations

from typing import Optional, Type

from ..config import AgentConfig, ProviderEndpoint
from .base import Agent
from .context import ContextStore
from .interviewee import IntervieweeAgent
from .supplement import SupplementAgent


_AGENT_TYPES: dict[str, Type[Agent]] = {
    SupplementAgent.id: SupplementAgent,
    IntervieweeAgent.id: IntervieweeAgent,
}


def list_agent_modes() -> list[tuple[str, str]]:
    """Return ``(id, human_label)`` pairs for the settings UI."""
    return [(cls.id, cls.label) for cls in _AGENT_TYPES.values()]


def build_agent(
    cfg: AgentConfig,
    endpoint: ProviderEndpoint,
    context: Optional[ContextStore] = None,
) -> Agent:
    """Instantiate the agent matching ``cfg.mode``.

    Unknown modes fall back to ``SupplementAgent`` with a warning -- we
    never raise here so a corrupted YAML can't keep the user out of the
    UI; they'll just see the default agent until they pick a new one
    from the settings dialog.
    """
    cls = _AGENT_TYPES.get(cfg.mode, SupplementAgent)
    context_text = context.combined if context is not None else ""
    return cls(cfg=cfg, endpoint=endpoint, context=context_text)
