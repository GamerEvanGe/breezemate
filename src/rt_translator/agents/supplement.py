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
        # "Where the gloss lives" tracks the new reply-language mode:
        #   * source                  -> monolingual gloss in the
        #                                audio language (e.g. English
        #                                explanation of an English term)
        #   * source_and_translation  -> bilingual gloss: source-language
        #                                gloss + same gloss translated
        #                                into target_lang, separated by
        #                                " — " on the same bullet so the
        #                                bullet density stays manageable.
        src = _lang(self._src_lang)
        tgt = _lang(self.cfg.target_lang)
        bilingual = self.cfg.reply_mode == "source_and_translation"
        depth = self.cfg.answer_depth
        # Bullet ceiling scales with depth -- "deep" lets the agent
        # write a fuller study note, while "concise" keeps it to a
        # quick gloss.
        if depth == "concise":
            max_bullets = 2
            vocab_quota = "ONE"
            length_hint = "under ~15 words"
        elif depth == "standard":
            max_bullets = 3
            vocab_quota = "at most TWO"
            length_hint = "under ~20 words"
        else:  # deep
            max_bullets = 5
            vocab_quota = "at most THREE"
            length_hint = "under ~30 words; include collocations and register"
        if bilingual:
            # Bilingual bullets: each gloss is written as
            #   "<source-language gloss> — <target-language gloss>"
            # so the user can see both languages at a glance without
            # doubling the bullet count.
            gloss_lang_clause = (
                f"in **{src}** (the audio language), followed by " 
                f"' — ' and the same idea in **{tgt}**"
            )
            example_clause = f"a tiny example in {src} (no translation needed)"
            output_lang_rule = (
                f"Each bullet is bilingual: '<{src} text> — <{tgt} text>'. "
                f"Do not split the bilingual halves onto separate bullets."
            )
        else:
            gloss_lang_clause = f"in **{src}** (the audio language)"
            example_clause = f"a tiny example in {src}"
            output_lang_rule = (
                f"All bullets are written in **{src}**. Do not include "
                f"a {tgt} translation."
            )

        body = (
            "You are the 'supplement' agent in a real-time subtitle app. "
            "For every transcript turn the user shows you, decide whether "
            "anything in the line would benefit a language learner who is "
            "already reading the literal translation. If yes, write a "
            f"compact study note {gloss_lang_clause}. If the sentence is "
            "plain everyday speech with no notable vocabulary / idiom / "
            "cultural item, respond with EXACTLY the sentinel "
            "<<<SKIP>>> and nothing else.\n"
            "\n"
            f"When you DO respond, output at most {max_bullets} short "
            "bullet lines, each starting with '• ' (a bullet then a "
            "space). Allowed bullet kinds, in priority order:\n"
            f"  • Vocab / collocation: pick {vocab_quota} non-trivial "
            f"items from the line. Give: source term -- short gloss "
            f"({length_hint}). Add {example_clause}.\n"
            "  • Idiom / fixed expression: source idiom -- literal meaning "
            "AND idiomatic meaning. Skip if the translation already nailed "
            "the idiom obviously.\n"
            "  • Proper noun / cultural reference: who/what it is in one "
            "short clause, only if relevant to understanding the sentence.\n"
            "  • Grammar note: only when the source uses a structure that "
            "would trip up a learner (subjunctive, inversion, ellipsis...). "
            "One short line max.\n"
            + (
                "  • Pronunciation / register tip: at most one line, only "
                "if the line contains a tricky stress pattern, a homophone, "
                "or a formal/informal register clash worth flagging.\n"
                if depth == "deep"
                else ""
            )
            + "\n"
            "Hard rules:\n"
            "  * Never repeat the translation itself. The user already has it.\n"
            f"  * Never produce more than {max_bullets} bullets total.\n"
            "  * Never invent etymology / quotations you're not sure about.\n"
            "  * No markdown headers, no preamble, no closing remark.\n"
            f"  * {output_lang_rule}\n"
        )
        return f"{ctx}\n{body}" if ctx else body


__all__ = ["SupplementAgent"]
