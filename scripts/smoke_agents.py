"""Smoke tests for the M3 agent layer.

Exercises three independent pieces without needing a live LLM:

1. ``ContextStore`` happily loads .txt / .md / .pdf / .docx files,
   stitches them into a single combined string, and applies the
   global char cap correctly.
2. ``Agent.run_stream`` produces ``AgentDelta`` then ``AgentFinal``
   for normal output, and produces ``AgentSkipped`` when the LLM
   prints the ``<<<SKIP>>>`` sentinel. We monkey-patch the OpenAI
   client to avoid hitting the network.
3. ``AgentWindow`` builds, accepts a row for an AgentDelta, drops it
   on AgentSkipped, and keeps a different row on AgentFinal. Run
   headless via ``QT_QPA_PLATFORM=offscreen``.

Pass criterion: prints "PASS: agents smoke OK" and exits 0.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import AsyncIterator

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")


# -------------------------- 1. context loader ---------------------------------


def test_context_store() -> None:
    from rt_translator.agents.context import ContextStore, SUPPORTED_EXTENSIONS

    assert SUPPORTED_EXTENSIONS == (".txt", ".md", ".pdf", ".docx")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        a = tmp_p / "a.txt"
        a.write_text("Alpha facts about project X.\nMetric: 99.9% uptime.", encoding="utf-8")
        b = tmp_p / "b.md"
        b.write_text("# Title\nBullet 1\nBullet 2", encoding="utf-8")
        c = tmp_p / "huge.txt"
        c.write_text("X" * 5000, encoding="utf-8")

        store = ContextStore([a, b, c], max_total_chars=200)
        combined = store.combined
        assert "Alpha facts about project X" in combined, combined
        assert "Bullet 1" in combined, combined
        # huge.txt should be truncated (or omitted) once the budget runs out.
        assert "XXXX" not in combined[:200] or combined.endswith(
            "[(truncated to fit context budget)]"
        ), combined[-200:]
        assert len(combined) <= 600  # 200 chars + headers + truncation markers

        # Files-only signal in sections.
        sections = store.sections
        names = [s.display_name for s in sections]
        assert "a.txt" in names and "b.md" in names, names

    print("  ContextStore: OK")


# -------------------------- 2. agent streaming --------------------------------


class _FakeDelta:
    def __init__(self, text: str) -> None:
        self.content = text


class _FakeChoice:
    def __init__(self, text: str) -> None:
        self.delta = _FakeDelta(text)


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.choices = [_FakeChoice(text)]


class _FakeStream:
    def __init__(self, parts: list[str]) -> None:
        self._parts = list(parts)

    def __aiter__(self) -> "AsyncIterator[_FakeChunk]":
        return self._gen()

    async def _gen(self) -> "AsyncIterator[_FakeChunk]":
        for p in self._parts:
            yield _FakeChunk(p)
            await asyncio.sleep(0)


class _FakeCompletions:
    def __init__(self, parts: list[str]) -> None:
        self._parts = parts

    async def create(self, **kwargs):  # noqa: D401 - matches openai SDK shape
        return _FakeStream(self._parts)


class _FakeChat:
    def __init__(self, parts: list[str]) -> None:
        self.completions = _FakeCompletions(parts)


class _FakeOpenAI:
    def __init__(self, parts: list[str]) -> None:
        self.chat = _FakeChat(parts)

    async def close(self) -> None:
        pass


async def _run_agent_with_parts(parts: list[str]):
    from rt_translator.agents.supplement import SupplementAgent
    from rt_translator.agents.base import AgentInput
    from rt_translator.config import AgentConfig, ProviderEndpoint

    cfg = AgentConfig(enabled=True, provider="openai", model="gpt-4o-mini")
    endpoint = ProviderEndpoint(base_url="http://localhost", auth_required=False)
    agent = SupplementAgent(cfg, endpoint, context="")
    agent._client = _FakeOpenAI(parts)  # type: ignore[assignment]
    turn = AgentInput(
        item_id="oai-1", source_text="Hello world.", translation="你好世界。", src_lang="en"
    )
    events: list = []
    async for ev in agent.run_stream(turn):
        events.append(ev)
    return events


def test_agent_streaming_normal() -> None:
    from rt_translator.events import AgentDelta, AgentFinal

    events = asyncio.run(_run_agent_with_parts(["• Vocab: hello ", "-- greeting"]))
    deltas = [e for e in events if isinstance(e, AgentDelta)]
    finals = [e for e in events if isinstance(e, AgentFinal)]
    assert deltas, "expected at least one AgentDelta"
    assert len(finals) == 1, f"expected exactly one AgentFinal, got {finals}"
    assert "greeting" in finals[0].text, finals[0].text
    print("  agent streaming (normal): OK")


def test_agent_streaming_skipped() -> None:
    from rt_translator.events import AgentSkipped

    events = asyncio.run(_run_agent_with_parts(["<<<SKIP>>>"]))
    skipped = [e for e in events if isinstance(e, AgentSkipped)]
    assert skipped, f"expected an AgentSkipped event, got {events}"
    print("  agent streaming (skipped): OK")


# -------------------------- 3. agent window UI --------------------------------


def test_agent_window() -> None:
    from PySide6.QtWidgets import QApplication

    from rt_translator.config import AgentWindowConfig
    from rt_translator.gui.agent_window import AgentWindow

    app = QApplication.instance() or QApplication(sys.argv)
    win = AgentWindow(AgentWindowConfig())
    win.set_agent_label("补充讲解")

    win.on_transcript_final("oai-1", "Hello world, how are you?")
    win.on_agent_delta("oai-1", "supplement", "• Greeting")
    app.processEvents()
    assert "oai-1" in win._entries, list(win._entries)

    # A skipped turn should leave no row, even if a delta was speculative.
    win.on_transcript_final("oai-2", "Random small talk")
    win.on_agent_delta("oai-2", "supplement", "• partial")
    win.on_agent_skipped("oai-2", "supplement", "no notable items")
    app.processEvents()
    assert "oai-2" not in win._entries, list(win._entries)

    # Final lands -> body stays, state flips to "final".
    win.on_agent_final("oai-1", "supplement", "• Greeting: a casual hello.")
    app.processEvents()
    entry = win._entries["oai-1"]
    assert entry.state() == "final", entry.state()

    win.close()
    print("  AgentWindow: OK")


# ------------------------------ main ------------------------------------------


def main() -> int:
    test_context_store()
    test_agent_streaming_normal()
    test_agent_streaming_skipped()
    test_agent_window()
    print("PASS: agents smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
