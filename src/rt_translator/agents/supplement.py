"""Supplement agent.

For each finalised sentence, add short, *useful* commentary aimed at a
language learner: vocabulary glosses for hard words, idioms and fixed
collocations, brief grammar notes, and culture / domain background
when a proper noun appears. The goal is "fill in the gaps the
translation glossed over".

Output style is structured but compact -- usually two or three short
bullet lines -- so it fits the agent overlay without taking over the
screen. The agent is expected to skip turns that don't need any
unpacking (small talk, simple statements) by printing the
``<<<SKIP>>>`` sentinel.
"""

from __future__ import annotations

from .base import Agent, AgentInput

_LANG_FRIENDLY = {
    "zh": "Simplified Chinese",
    "zh-CN": "Simplified Chinese",
    "zh-TW": "Traditional Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}


def _lang(code: str) -> str:
    return _LANG_FRIENDLY.get(code, code)


class SupplementAgent(Agent):
    id = "supplement"
    label = "补充讲解"

    def system_prompt(self) -> str:
        ctx = super().system_prompt()
        tgt = _lang(self.cfg.target_lang)
        body = (
            "You are the 'supplement' agent in a real-time subtitle app. "
            "For every transcript turn the user shows you, decide whether "
            "anything in the line would benefit a language learner who is "
            "already reading the literal translation. If yes, write a "
            f"very short note in {tgt}. If the sentence is plain everyday "
            "speech with no notable vocabulary / idiom / cultural item, "
            "respond with EXACTLY the sentinel <<<SKIP>>> and nothing else.\n"
            "\n"
            "When you DO respond, output at most three short bullet lines, "
            "each starting with '• ' (a bullet then a space). Allowed bullet "
            "kinds, in priority order:\n"
            "  • Vocab / collocation: pick at most TWO non-trivial items "
            "from the line. Give: source term -- short gloss (under ~15 "
            f"words). Example in {tgt}.\n"
            "  • Idiom / fixed expression: source idiom -- literal meaning "
            "AND idiomatic meaning. Skip if the translation already nailed "
            "the idiom obviously.\n"
            "  • Proper noun / cultural reference: who/what it is in one "
            "short clause, only if relevant to understanding the sentence.\n"
            "  • Grammar note: only when the source uses a structure that "
            "would trip up a learner (subjunctive, inversion, ellipsis...). "
            "One short line max.\n"
            "\n"
            "Hard rules:\n"
            "  * Never repeat the translation itself. The user already has it.\n"
            "  * Never produce more than three bullets total.\n"
            "  * Never invent etymology / quotations you're not sure about.\n"
            "  * No markdown headers, no preamble, no closing remark.\n"
            "  * Output language: " + tgt + ".\n"
        )
        return f"{ctx}\n{body}" if ctx else body


__all__ = ["SupplementAgent"]
