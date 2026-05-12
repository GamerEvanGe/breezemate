"""Streaming polisher + translator over an OpenAI-compatible chat API.

Single LLM call produces TWO outputs, separated by a sentinel:

    <polished source text>
    <<<TRANSLATION>>>
    <target-language translation>

Both halves stream back to the UI:

* The polished half overwrites the raw Vosk ASR text inside the same
  subtitle row via ``TranscriptDelta`` -> ``TranscriptFinal`` events.
  Polishing is "add punctuation, fix glaring word errors with the
  smallest possible edit; never paraphrase". Result: the user sees
  the same words they spoke, but with the periods, commas and
  capitals filled in.
* The translation half flows as ``TranslationDelta`` -> ``TranslationFinal``
  unchanged, so existing translation-rendering code keeps working.

Works with any provider that speaks the OpenAI Chat Completions protocol:
official OpenAI, DeepSeek, Qwen, Kimi, local Ollama (``/v1`` endpoint), etc.
Swapping providers is purely a config change -- no code modifications.

Reliability:
* Per-request timeout with a single retry. If both attempts fail we still
  emit a ``TranslationFinal`` placeholder so the UI doesn't hang.
* If the model misbehaves and never emits the sentinel, we fall back to
  treating the entire output as the translation (leaving the raw ASR text
  visible as the source). Safe degradation, not a hard error.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from openai import AsyncOpenAI
from openai import APIError, APITimeoutError

from ...config import ProviderEndpoint, TranslatorConfig
from ...events import TranscriptDelta, TranscriptFinal, TranslationDelta, TranslationFinal
from ..base import TranslatorOutput

log = logging.getLogger(__name__)


_LANG_NAMES = {
    "zh": "Simplified Chinese",
    "zh-CN": "Simplified Chinese",
    "zh-TW": "Traditional Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "uk": "Ukrainian",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "hi": "Hindi",
}


# Literal sentinel the LLM is instructed to print between the polished
# source half and the translation half. Chosen to be visually distinct
# AND extremely unlikely to occur in natural prose. We do NOT use a
# bare "---" or "###" because those show up too often in real ASR
# output (URLs, code, em-dashes transcribed as triple-hyphens, etc.).
_TRANSLATION_SENTINEL = "<<<TRANSLATION>>>"


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get(code, code)


def _system_prompt(src_lang: str, tgt_lang: str) -> str:
    return (
        f"You are a real-time bilingual subtitle assistant. Each user "
        f"message is raw ASR text in {_lang_name(src_lang)} -- typically "
        "missing punctuation and capitalisation, sometimes with small "
        "recognition errors. Produce TWO outputs separated by a literal "
        f"line containing only {_TRANSLATION_SENTINEL}.\n"
        "\n"
        f"Part 1 -- polished {_lang_name(src_lang)} (the user's own words "
        "with punctuation):\n"
        "  * Add appropriate punctuation (periods, commas, question marks, "
        "    quotation marks) and capitalise where natural.\n"
        "  * Preserve the speaker's words exactly. Do NOT rephrase, "
        "    summarise, or change vocabulary.\n"
        "  * Only fix a word if you are sure the ASR mis-recognised it AND "
        "    the original is ungrammatical. Use the SMALLEST possible edit "
        "    (a single character or word).\n"
        "  * Keep proper nouns, brand names, code identifiers, file paths, "
        "    URLs and numbers in their original form.\n"
        "\n"
        f"Part 2 -- {_lang_name(tgt_lang)} translation:\n"
        "  * Natural, conversational, spoken-style. Translate meaning, not "
        "    word-for-word.\n"
        "  * Match the speaker's tone (formal / casual / technical).\n"
        "  * If the input is a partial / mid-sentence fragment, translate "
        "    it as-is without trying to complete it.\n"
        "\n"
        "Output strictly in this format, with no quotes, no markdown, no "
        "headers, no commentary:\n"
        "\n"
        f"<polished {_lang_name(src_lang)} text>\n"
        f"{_TRANSLATION_SENTINEL}\n"
        f"<{_lang_name(tgt_lang)} translation>"
    )


class OpenAICompatibleTranslator:
    def __init__(self, cfg: TranslatorConfig, endpoint: ProviderEndpoint) -> None:
        self.cfg = cfg
        # ``AsyncOpenAI`` refuses to build a client when ``api_key`` is
        # falsy -- it raises the "missing credentials" error before any
        # HTTP traffic is attempted. For keyless endpoints (Ollama, LM
        # Studio, Pollinations, ...) ``resolve_api_key`` correctly
        # returns "" but we still have to feed *something* to the SDK.
        # The server simply ignores whatever placeholder we pass.
        api_key = endpoint.resolve_api_key() or "no-key-required"
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint.base_url,
            timeout=cfg.timeout_s,
            max_retries=0,  # we do our own retry once
        )

    async def aclose(self) -> None:
        try:
            await self._client.close()
        except Exception:
            pass

    async def translate_stream(
        self,
        item_id: str,
        text: str,
        src_lang: str,
        tgt_lang: str,
        history: list[tuple[str, str]],
    ) -> AsyncIterator[TranslatorOutput]:
        messages = [{"role": "system", "content": _system_prompt(src_lang, tgt_lang)}]
        for src, tgt in history[-self.cfg.context_window :]:
            # Replay each historic turn in the same sentinel-delimited
            # format the system prompt asks for, so the LLM has
            # concrete examples of the layout it should produce.
            messages.append({"role": "user", "content": src})
            messages.append(
                {
                    "role": "assistant",
                    "content": f"{src}\n{_TRANSLATION_SENTINEL}\n{tgt}",
                }
            )
        messages.append({"role": "user", "content": text})

        for attempt in (1, 2):
            try:
                async for ev in self._stream_once(item_id, messages, text):
                    yield ev
                return
            except (APITimeoutError, asyncio.TimeoutError) as e:
                log.warning("Translation timeout (attempt %d): %s", attempt, e)
            except APIError as e:
                log.warning("Translation API error (attempt %d): %s", attempt, e)
            except Exception as e:
                log.warning("Translation unexpected error (attempt %d): %s", attempt, e)
            if attempt == 2:
                yield TranslationFinal(item_id=item_id, text="[翻译失败]")

    async def _stream_once(
        self,
        item_id: str,
        messages: list[dict],
        raw_source: str,
    ) -> AsyncIterator[TranslatorOutput]:
        """Stream the model's response and split it on ``_TRANSLATION_SENTINEL``.

        We can't emit the polished prefix immediately on every chunk:
        the sentinel could be straddling two chunks (e.g. the model
        emits ``"...sentence.<<<TRA"`` then ``"NSLATION>>>\\nhello"``).
        So we hold back the last ``len(sentinel)`` characters of the
        polished section until either (a) the sentinel materialises
        and we know exactly where the split is, or (b) the stream
        ends without a sentinel (degraded path).

        Once the sentinel is found, ``TranscriptFinal`` fires with the
        canonical polished text, and everything after the sentinel
        is forwarded as TranslationDelta events.
        """
        stream = await self._client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            stream=True,
            temperature=0.2,
        )

        sentinel = _TRANSLATION_SENTINEL
        sentinel_len = len(sentinel)

        accumulated = ""             # everything received so far
        polished_emitted = ""        # source we've already pushed as TranscriptDelta
        polished_done = False        # have we seen the sentinel?
        translation_so_far = ""      # part after the sentinel
        translation_emitted = False  # have we yielded at least one delta?

        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except (IndexError, AttributeError):
                delta = ""
            if not delta:
                continue
            accumulated += delta

            if not polished_done:
                sentinel_idx = accumulated.find(sentinel)
                if sentinel_idx >= 0:
                    # Sentinel arrived this chunk: split, emit, switch
                    # streams to translation mode.
                    polished_full = accumulated[:sentinel_idx].rstrip()
                    if polished_full and polished_full != polished_emitted:
                        yield TranscriptDelta(item_id=item_id, text=polished_full)
                    yield TranscriptFinal(item_id=item_id, text=polished_full)
                    polished_done = True
                    polished_emitted = polished_full
                    translation_so_far = accumulated[sentinel_idx + sentinel_len :].lstrip()
                    if translation_so_far:
                        translation_emitted = True
                        yield TranslationDelta(
                            item_id=item_id, text_so_far=translation_so_far
                        )
                else:
                    # No sentinel yet -- but the very tail of
                    # `accumulated` might be the start of one straddled
                    # across chunks. Hold back the last sentinel-length
                    # characters until we either see the rest of the
                    # sentinel or get enough additional text to prove
                    # it's not coming.
                    safe_end = max(0, len(accumulated) - sentinel_len)
                    safe_text = accumulated[:safe_end].rstrip()
                    if safe_text and safe_text != polished_emitted:
                        polished_emitted = safe_text
                        yield TranscriptDelta(item_id=item_id, text=safe_text)
            else:
                # In translation mode: re-derive the post-sentinel
                # slice each time and emit a cumulative delta.
                sentinel_idx = accumulated.find(sentinel)
                translation_so_far = accumulated[sentinel_idx + sentinel_len :].lstrip()
                if translation_so_far:
                    translation_emitted = True
                    yield TranslationDelta(
                        item_id=item_id, text_so_far=translation_so_far
                    )

        # End of stream. Three possible terminal states:
        if polished_done:
            yield TranslationFinal(
                item_id=item_id, text=translation_so_far.strip()
            )
            return

        # The model never emitted a sentinel. Two sub-cases:
        # 1. The output looks like a translation only -- assume it is,
        #    leave the raw ASR text visible as the source. This matches
        #    the legacy pre-polishing behaviour.
        # 2. The output is empty -- emit a clear placeholder.
        stripped = accumulated.strip()
        if stripped:
            log.warning(
                "LLM omitted the %s sentinel; treating entire output as translation.",
                sentinel,
            )
            if not translation_emitted:
                yield TranslationDelta(item_id=item_id, text_so_far=stripped)
            yield TranslationFinal(item_id=item_id, text=stripped)
        else:
            yield TranslationFinal(item_id=item_id, text="[翻译失败]")
