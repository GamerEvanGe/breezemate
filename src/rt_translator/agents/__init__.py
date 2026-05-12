"""M3 agent layer.

An *agent* is an optional second LLM stream that observes every
finalised transcript turn (and its translation, when translate mode is
on) and writes its own auxiliary content into a dedicated floating
window. Two agents ship with M3:

* ``supplement`` -- explains proper nouns / idioms / fixed
  collocations / grammar points that appear in the turn. Great for
  language learners following a lecture or podcast.
* ``interviewee`` -- watches for interview-style questions in the
  transcript and answers them in first person, as if the user were the
  interviewee. The user's CV / project docs go in via the context-file
  upload so the agent can ground its answers.

Adding a new agent = drop a new module here, subclass ``Agent``, and
register it in ``registry.py``. The pipeline, signal_sink and UI do
not need to know about specific agent ids -- they just plumb
``AgentDelta`` / ``AgentFinal`` events whose ``agent_id`` matches
whatever the user picked.
"""

from .base import Agent, AgentInput
from .context import ContextStore
from .registry import build_agent, list_agent_modes

__all__ = [
    "Agent",
    "AgentInput",
    "ContextStore",
    "build_agent",
    "list_agent_modes",
]
