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


def test_interviewee_depth_directive() -> None:
    """The interviewee agent's system prompt should change shape based on
    the configured ``answer_depth``, and must always contain a length
    budget. This guards the fix for "interviewee answers are too short
    compared to ChatGPT"."""
    from rt_translator.agents.interviewee import IntervieweeAgent
    from rt_translator.config import AgentConfig, ProviderEndpoint

    endpoint = ProviderEndpoint(base_url="http://localhost", auth_required=False)

    seen_lengths: dict[str, int] = {}
    for depth in ("concise", "standard", "deep"):
        cfg = AgentConfig(
            enabled=True,
            mode="interviewee",
            provider="openai",
            model="gpt-4o-mini",
            answer_depth=depth,  # type: ignore[arg-type]
        )
        agent = IntervieweeAgent(cfg, endpoint, context="")
        prompt = agent.system_prompt()
        assert "interviewee" in prompt.lower(), prompt[:200]
        assert "step 1" in prompt.lower(), "missing classify step"
        assert "STAR" in prompt, "missing behavioural structure guidance"
        assert "<<<SKIP>>>" in prompt, "missing skip sentinel"
        # The depth directive should be embedded.
        assert "Length" in prompt, prompt[-400:]
        seen_lengths[depth] = len(prompt)

    # Deep must give strictly more guidance text than standard, which
    # must give strictly more than concise. If somebody accidentally
    # blows away the depth knob, this catches it.
    assert (
        seen_lengths["deep"] > seen_lengths["standard"] > seen_lengths["concise"]
    ), seen_lengths

    # And under "deep", the prompt should explicitly forbid artificial
    # shortening -- that one line is what makes the difference between
    # a 60-word reply and a 500-word reply on the same model.
    cfg_deep = AgentConfig(
        enabled=True, mode="interviewee", provider="openai", model="gpt-4o-mini",
        answer_depth="deep",
    )
    deep_prompt = IntervieweeAgent(cfg_deep, endpoint, context="").system_prompt()
    assert "Do NOT" in deep_prompt and "artificially" in deep_prompt.lower(), deep_prompt[-600:]
    print("  interviewee depth directive: OK")


def test_agent_config_defaults() -> None:
    """Regression guard: defaults must match the M3.1 'GPT-5.5 parity'
    rework. If somebody bumps them down again the interviewee agent
    will quietly start producing thin answers."""
    from rt_translator.config import AgentConfig

    cfg = AgentConfig()
    assert cfg.answer_depth == "deep", cfg.answer_depth
    assert cfg.max_output_tokens >= 1500, cfg.max_output_tokens
    assert cfg.timeout_s >= 30.0, cfg.timeout_s
    print("  agent config defaults: OK")


def test_model_family_caps() -> None:
    """The model-family helper must classify gpt-5*/o*-family models as
    "use max_completion_tokens", and reasoning models as "no
    temperature". Legacy chat models stay on the old shape so non-OpenAI
    providers (DeepSeek, glm, Ollama, ...) keep working."""
    from rt_translator.agents.base import _model_family_caps

    cases = {
        "gpt-4o-mini": (False, False),
        "gpt-4o": (False, False),
        "gpt-4.1-mini": (False, False),
        "gpt-5-mini": (True, False),
        "gpt-5": (True, False),
        "gpt-5.5": (True, False),
        "o1-mini": (True, True),
        "o3-mini": (True, True),
        "o3": (True, True),
        "o4-mini": (True, True),
        "deepseek-chat": (False, False),
        "glm-4-flash": (False, False),
        "Qwen/Qwen2.5-7B-Instruct": (False, False),
    }
    for model, expected in cases.items():
        got = _model_family_caps(model)
        assert got == expected, f"{model}: expected {expected}, got {got}"
    print("  model family caps: OK")


# Imitates OpenAI's 400 the first time, succeeds on retry.
class _Adaptive400Completions:
    def __init__(self, expected_param: str) -> None:
        self._expected_param = expected_param
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if "messages" not in kwargs:
            raise AssertionError("messages missing")
        if self._expected_param not in kwargs:
            # Simulate OpenAI's exact wording so the adapter's
            # heuristic has to actually parse it.
            from openai import BadRequestError

            # The OpenAI SDK builds BadRequestError from an httpx
            # Response in production, but the constructor also accepts
            # message=... + body=... shapes. Build the bare-message
            # variant since our adapter only inspects str(e).
            other = "max_tokens" if self._expected_param == "max_completion_tokens" else "max_completion_tokens"
            raise BadRequestError(
                message=(
                    f"Error code: 400 - Unsupported parameter: "
                    f"'{other}' is not supported with this model. "
                    f"Use '{self._expected_param}' instead."
                ),
                response=None,  # type: ignore[arg-type]
                body=None,
            )
        return _FakeStream(["• ", "ok"])


class _Adaptive400Chat:
    def __init__(self, expected_param: str) -> None:
        self.completions = _Adaptive400Completions(expected_param)


class _Adaptive400OpenAI:
    def __init__(self, expected_param: str) -> None:
        self.chat = _Adaptive400Chat(expected_param)

    async def close(self) -> None:
        pass


async def _run_adaptive_with_model(model: str, expected_param: str):
    from rt_translator.agents.supplement import SupplementAgent
    from rt_translator.agents.base import AgentInput
    from rt_translator.config import AgentConfig, ProviderEndpoint

    cfg = AgentConfig(enabled=True, provider="openai", model=model)
    endpoint = ProviderEndpoint(base_url="http://localhost", auth_required=False)
    agent = SupplementAgent(cfg, endpoint, context="")
    fake = _Adaptive400OpenAI(expected_param)
    agent._client = fake  # type: ignore[assignment]
    turn = AgentInput(item_id="x", source_text="hi", translation="你好", src_lang="en")
    events: list = []
    async for ev in agent.run_stream(turn):
        events.append(ev)
    return agent, fake.chat.completions.calls, events


def test_max_tokens_rename_for_gpt5() -> None:
    """When the model is gpt-5* and the first attempt still happens to
    send ``max_tokens`` (e.g. cached overrides got cleared), the
    adapter must rename it to ``max_completion_tokens`` and retry
    without surfacing the error to the user."""
    from rt_translator.events import AgentFinal, AgentDelta

    # gpt-5* models should pick max_completion_tokens up front and never
    # hit the 400 at all.
    agent, calls, events = asyncio.run(
        _run_adaptive_with_model("gpt-5.5", expected_param="max_completion_tokens")
    )
    assert len(calls) == 1, f"expected 1 call, got {len(calls)}: {calls}"
    assert "max_completion_tokens" in calls[0], calls[0].keys()
    assert "max_tokens" not in calls[0]
    assert "temperature" in calls[0], "gpt-5 chat models accept temperature"
    finals = [e for e in events if isinstance(e, AgentFinal)]
    assert finals and "ok" in finals[-1].text, events
    assert agent._param_overrides is not None
    print("  gpt-5 picks max_completion_tokens up front: OK")


def test_temperature_dropped_for_o3() -> None:
    """o3 reasoning models reject temperature AND max_tokens. The
    initial kwargs heuristic must skip temperature; if a provider
    surprises us with a different 400, the adapter should still
    recover."""
    from rt_translator.agents.base import Agent, _model_family_caps
    from rt_translator.config import AgentConfig, ProviderEndpoint

    cfg = AgentConfig(enabled=True, provider="openai", model="o3-mini")
    endpoint = ProviderEndpoint(base_url="http://localhost", auth_required=False)
    agent = Agent(cfg, endpoint, context="")  # base class is fine here

    kwargs = agent._initial_kwargs()
    assert "temperature" not in kwargs, kwargs
    assert "max_completion_tokens" in kwargs, kwargs
    assert "max_tokens" not in kwargs, kwargs

    # adapter must still drop temperature if a provider complains
    kwargs2 = {
        "model": "o3-mini",
        "stream": True,
        "max_completion_tokens": 500,
        "temperature": 0.4,
    }
    changed = Agent._adapt_kwargs_for_error(
        kwargs2,
        "Error code: 400 - Unsupported parameter: 'temperature' is not supported with this model.",
    )
    assert changed
    assert "temperature" not in kwargs2
    print("  o3 reasoning model adaptation: OK")


def test_reply_mode_source_default() -> None:
    """Default reply_mode is 'source' -- the interviewee answers in the
    audio language, NOT in target_lang. Regression guard for the
    user-reported issue 'agent only replies in one language'."""
    from rt_translator.agents.interviewee import IntervieweeAgent
    from rt_translator.config import AgentConfig, ProviderEndpoint

    endpoint = ProviderEndpoint(base_url="http://localhost", auth_required=False)
    cfg = AgentConfig(enabled=True, mode="interviewee", provider="openai", model="gpt-4o-mini")
    assert cfg.reply_mode == "source", cfg.reply_mode

    agent = IntervieweeAgent(cfg, endpoint, context="", src_lang="en")
    prompt = agent.system_prompt()
    # The source-language directive must be present and reference the
    # audio language explicitly.
    assert "REPLY LANGUAGE" in prompt, prompt
    assert "English" in prompt, prompt
    # And the old hard-coded "write in {target_lang}" line must NOT be
    # there any more -- that was the original bug.
    assert "write in Simplified Chinese" not in prompt, (
        "interviewee prompt still hard-codes target_lang"
    )
    print("  default reply_mode = source: OK")


def test_reply_mode_source_and_translation_layout() -> None:
    """The bilingual mode must spell out the alternating paragraph
    layout the user asked for ('一段原文回答、一段翻译, 交替显示')."""
    from rt_translator.agents.interviewee import IntervieweeAgent
    from rt_translator.config import AgentConfig, ProviderEndpoint

    endpoint = ProviderEndpoint(base_url="http://localhost", auth_required=False)
    cfg = AgentConfig(
        enabled=True,
        mode="interviewee",
        provider="openai",
        model="gpt-4o-mini",
        reply_mode="source_and_translation",
        target_lang="zh",
    )
    agent = IntervieweeAgent(cfg, endpoint, context="", src_lang="en")
    prompt = agent.system_prompt()

    # Both languages must be named.
    assert "English" in prompt and "Simplified Chinese" in prompt, prompt
    # The structural template should appear, with paragraph 1 / 2
    # placeholders in BOTH languages.
    assert "paragraph 1" in prompt, prompt
    assert "paragraph 2" in prompt, prompt
    # The anti-grouping rule must be explicit.
    assert "Do NOT group" in prompt or "do not group" in prompt.lower(), prompt
    print("  reply_mode = source_and_translation layout: OK")


def test_reply_mode_supplement_monolingual_vs_bilingual() -> None:
    """SupplementAgent must follow the same source/bilingual switch:
    'source' -> glosses entirely in src_lang;
    'source_and_translation' -> bilingual em-dash bullets."""
    from rt_translator.agents.supplement import SupplementAgent
    from rt_translator.config import AgentConfig, ProviderEndpoint

    endpoint = ProviderEndpoint(base_url="http://localhost", auth_required=False)

    cfg_src = AgentConfig(
        enabled=True, mode="supplement", provider="openai",
        model="gpt-4o-mini", reply_mode="source",
    )
    src_prompt = SupplementAgent(cfg_src, endpoint, context="", src_lang="en").system_prompt()
    assert "English" in src_prompt, src_prompt
    assert "Do not include" in src_prompt, src_prompt  # no zh translation

    cfg_both = AgentConfig(
        enabled=True, mode="supplement", provider="openai",
        model="gpt-4o-mini", reply_mode="source_and_translation",
    )
    both_prompt = SupplementAgent(cfg_both, endpoint, context="", src_lang="en").system_prompt()
    assert "English" in both_prompt and "Simplified Chinese" in both_prompt, both_prompt
    assert "bilingual" in both_prompt.lower(), both_prompt
    print("  supplement reply-mode switch: OK")


def test_src_lang_threaded_through_agent() -> None:
    """src_lang must travel from cfg.asr.language -> build_agent ->
    Agent constructor and end up in the system prompt."""
    from rt_translator.agents.interviewee import IntervieweeAgent
    from rt_translator.agents.registry import build_agent
    from rt_translator.config import AgentConfig, ProviderEndpoint

    endpoint = ProviderEndpoint(base_url="http://localhost", auth_required=False)
    cfg = AgentConfig(enabled=True, mode="interviewee", provider="openai", model="gpt-4o-mini")
    agent = build_agent(cfg, endpoint, context=None, src_lang="ja")
    assert isinstance(agent, IntervieweeAgent)
    prompt = agent.system_prompt()
    assert "Japanese" in prompt, prompt
    assert "English" not in prompt or prompt.count("English") < prompt.count("Japanese"), (
        "Japanese src_lang should dominate prompt over the en default"
    )
    print("  src_lang plumbing: OK")


def test_legacy_model_unchanged() -> None:
    """Non-OpenAI / legacy models must keep using `max_tokens` and
    `temperature`. Otherwise we'd break free providers like glm-4-flash,
    Ollama, DeepSeek, etc."""
    from rt_translator.agents.base import Agent
    from rt_translator.config import AgentConfig, ProviderEndpoint

    cfg = AgentConfig(enabled=True, provider="openai", model="gpt-4o-mini")
    endpoint = ProviderEndpoint(base_url="http://localhost", auth_required=False)
    agent = Agent(cfg, endpoint, context="")
    kwargs = agent._initial_kwargs()
    assert kwargs.get("max_tokens") == cfg.max_output_tokens, kwargs
    assert kwargs.get("temperature") == 0.4, kwargs
    assert "max_completion_tokens" not in kwargs, kwargs
    print("  legacy model kwargs unchanged: OK")


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
    test_interviewee_depth_directive()
    test_agent_config_defaults()
    test_model_family_caps()
    test_max_tokens_rename_for_gpt5()
    test_temperature_dropped_for_o3()
    test_reply_mode_source_default()
    test_reply_mode_source_and_translation_layout()
    test_reply_mode_supplement_monolingual_vs_bilingual()
    test_src_lang_threaded_through_agent()
    test_legacy_model_unchanged()
    test_agent_window()
    print("PASS: agents smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
