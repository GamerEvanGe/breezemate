"""Configuration loading: YAML file + .env + secrets.json, with pydantic
validation.

Resolution order for every field (highest priority first):
    1. CLI flags / GUI state (handled in cli.py / gui/)
    2. config.yaml values
    3. config.example.yaml defaults (bundled)
    4. Hard-coded defaults below

For API keys specifically, see ``secrets.py`` for the additional
resolution order between secrets.json and environment variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


Mode = Literal["asr_only", "translate"]
AudioSource = Literal["loopback", "mic"]


class AudioConfig(BaseModel):
    source: Optional[AudioSource] = None
    device_name: Optional[str] = None
    chunk_ms: int = Field(default=50, ge=10, le=200)


class ASRConfig(BaseModel):
    """Top-level ASR config.

    Two backends are supported:

    * ``vosk_local`` -- the embedded offline Vosk recogniser. Free,
      offline, decent quality on small models. ``ASRConfig.model``
      mirrors ``cfg.local_asr.model`` (the actual Vosk model id).
    * ``openai_realtime`` -- OpenAI's Realtime transcription API.
      Higher quality, networked, billed per minute. ``ASRConfig.model``
      is the transcription model name (``gpt-4o-mini-transcribe``,
      ``gpt-4o-transcribe``). When this is chosen the pipeline still
      runs Vosk as a *preview-only* engine in parallel so the user
      keeps the words-as-they-are-spoken UX even though OpenAI owns
      the canonical transcript.

    The Vosk-specific knobs (model id, silence cutoff, ...) always
    live in ``cfg.local_asr`` regardless of which backend is canonical
    -- that struct is also what the preview engine reads.
    """

    provider: Literal["vosk_local", "openai_realtime"] = "vosk_local"
    # Currently-selected model. For ``vosk_local`` this is the Vosk
    # model id; for ``openai_realtime`` it's the transcription model
    # name. The settings dialog keeps these in sync with whichever
    # backend is active.
    model: str = "vosk-model-small-en-us-0.15"
    language: Optional[str] = "en"


class LocalASRConfig(BaseModel):
    """Settings for the embedded Vosk offline recogniser.

    Vosk drives BOTH the live preview row (showing accumulating word
    fragments at sub-second latency) AND the canonical sentence text
    that gets sent to the translator. There is no longer a separate
    "cloud ASR" path; the single ``finalize_after_silence_s`` knob
    below replaces the old OpenAI Realtime VAD parameters.
    """

    # Vosk model id (also the folder name under ``vosk-models/``).
    # See ``providers/asr/vosk_model.py`` for the catalog and the
    # supported languages.
    model: str = "vosk-model-small-en-us-0.15"
    # Minimum partial-text length before we emit it to the UI. Filters
    # the very-noisy single-character partials Vosk produces at the
    # start of every utterance.
    min_partial_chars: int = Field(default=2, ge=1, le=20)
    # Sentence boundary detector. When the running Vosk partial hasn't
    # changed for this many seconds we treat it as "speaker paused",
    # emit a canonical TranscriptFinal, and reset the recogniser. The
    # preview row keeps the in-progress text visible during the wait
    # so a 1s pause doesn't make the screen go blank.
    #
    # Lower = chops sentences earlier (more responsive translations,
    # but commas inside a sentence may become breaks). Higher = waits
    # longer for the speaker to truly stop talking.
    finalize_after_silence_s: float = Field(default=1.0, ge=0.2, le=10.0)


class TranslatorConfig(BaseModel):
    # ``provider`` is the *id* of a ProviderEndpoint inside
    # ``AppConfig.providers``. Defaults to ``openai`` so we stay
    # backwards-compatible with M1 configs.
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    target_lang: str = "zh"
    context_window: int = Field(default=3, ge=0, le=10)
    timeout_s: float = Field(default=5.0, gt=0)


class ProviderEndpoint(BaseModel):
    """An OpenAI-protocol HTTP endpoint plus the env-var name where its
    API key is stored.

    The actual key is *not* stored here in plain text; it's looked up at
    runtime from ``rt_translator.secrets.SecretStore`` (which checks the
    encrypted-at-rest-by-OS secrets.json first, then the process env).
    """

    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    # If False, ``resolve_api_key`` returns an empty string instead of
    # raising. Used for local providers like Ollama / LM Studio that
    # accept any (or no) auth header.
    auth_required: bool = True

    def resolve_api_key(self) -> str:
        # Import locally to avoid a circular import with secrets.py at
        # module load time (secrets.py imports from device_picker).
        from .secrets import get_secret_store

        key = get_secret_store().get(self.api_key_env)
        if not key and self.auth_required:
            raise RuntimeError(
                f"No API key found for {self.api_key_env}. "
                "Set it in the GUI Settings dialog, drop it in "
                ".env, or export it as an environment variable."
            )
        return key


class DisplayConfig(BaseModel):
    max_rows: int = Field(default=8, ge=2, le=30)
    refresh_hz: int = Field(default=20, ge=4, le=60)


class SubtitleWindowConfig(BaseModel):
    """Floating subtitle overlay settings (GUI only)."""

    # Window geometry. None means "use screen-bottom-center default on
    # first launch and then remember user's drag/resize after that".
    x: Optional[int] = None
    y: Optional[int] = None
    width: int = Field(default=900, ge=200, le=4096)
    # Default height fits ~3-4 finalised rows + 1 live row at the
    # bundled font sizes. The user can drag-resize at any time and we
    # persist whatever they pick.
    height: int = Field(default=260, ge=60, le=2048)
    # Background opacity 0..1. Controls ONLY the rounded dark plate
    # painted behind the text; text alpha is independent (see below).
    background_opacity: float = Field(default=0.55, ge=0.0, le=1.0)
    # Text opacity 0..1. Multiplies into every text colour (source
    # transcript, translation, live preview) before it is handed to the
    # stylesheet. Decoupled from ``background_opacity`` so the user can
    # have, say, a fully-transparent plate but still bright readable
    # text -- or a dark plate with subtler text.
    text_opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    font_family: str = "Microsoft YaHei"
    asr_font_size_pt: int = Field(default=16, ge=8, le=72)
    translation_font_size_pt: int = Field(default=20, ge=8, le=72)
    asr_color: str = "#cfd8dc"
    translation_color: str = "#ffffff"
    # Colour for the live "currently being recognised" sentence shown
    # in the bottom preview row (driven by either the local Vosk ASR
    # or, when local is off, OpenAI's transcript_delta stream).
    preview_color: str = "#80deea"
    # Always-on-top toggle. Off is mostly useful when debugging.
    always_on_top: bool = True
    # When True, mouse clicks pass through the overlay to the app behind
    # it. Toggleable from the tray / overlay context menu.
    click_through: bool = False
    # How many recent utterances stay on screen. New utterances push
    # the oldest off the top (visually the rows scroll up). Range 1..12.
    max_visible_entries: int = Field(default=4, ge=1, le=12)
    # Fixed pixel gap between two adjacent finalised sentences. We
    # don't tie this to font size on purpose: large fonts already
    # imply tall lines, and an additional font-scaled gap looked
    # excessive in testing. Tweak this if you want denser / sparser
    # subtitles.
    row_spacing_px: int = Field(default=4, ge=0, le=40)
    # Slide-in animation duration in ms (set 0 to disable animation).
    slide_animation_ms: int = Field(default=320, ge=0, le=2000)


class AppConfig(BaseModel):
    mode: Mode = "translate"
    audio: AudioConfig = Field(default_factory=AudioConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    local_asr: LocalASRConfig = Field(default_factory=LocalASRConfig)
    translator: TranslatorConfig = Field(default_factory=TranslatorConfig)
    # Keyed by provider id (e.g. ``openai``, ``groq``, ``ollama``).
    providers: dict[str, ProviderEndpoint] = Field(
        default_factory=lambda: {"openai": ProviderEndpoint()}
    )
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    subtitle_window: SubtitleWindowConfig = Field(default_factory=SubtitleWindowConfig)

    @field_validator("providers")
    @classmethod
    def _ensure_openai(cls, v: dict[str, ProviderEndpoint]) -> dict[str, ProviderEndpoint]:
        if "openai" not in v:
            v["openai"] = ProviderEndpoint()
        return v

    @model_validator(mode="after")
    def _normalise_asr_model(self) -> "AppConfig":
        """Keep ``asr.model`` consistent with the chosen backend.

        Both ASR backends store their "currently selected model" in
        ``asr.model`` so the main window can show a one-line summary,
        but the canonical source-of-truth differs:

        * For ``vosk_local`` it's ``local_asr.model``.
        * For ``openai_realtime`` it's whatever transcription model the
          user picked in the settings dialog.

        If the saved config gets into a confused state (e.g. the user
        toggles backend in the dialog but the persisted YAML still
        names the other backend's model) we silently reconcile here.
        """
        if self.asr.provider == "vosk_local":
            if not self.asr.model or not self.asr.model.startswith("vosk-"):
                self.asr.model = self.local_asr.model
        elif self.asr.provider == "openai_realtime":
            if not self.asr.model or self.asr.model.startswith("vosk-"):
                # Pick a sensible default if nothing better is saved.
                self.asr.model = "gpt-4o-mini-transcribe"
        return self

    def openai_endpoint(self) -> ProviderEndpoint:
        """Endpoint profile for the OpenAI provider (used by Realtime ASR
        whenever it's selected). The OpenAI key may be set in the
        translator panel too, so this just looks the profile up rather
        than asking the user to re-enter it."""
        return self.providers.get("openai") or ProviderEndpoint()

    def translator_endpoint(self) -> ProviderEndpoint:
        """Endpoint for the translation provider profile, falling back to OpenAI."""
        return self.providers.get(self.translator.provider) or self.openai_endpoint()


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load AppConfig from YAML if provided, else return defaults."""
    if path is None:
        return AppConfig()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig.model_validate(data)


def default_config_path() -> Path:
    """User-writable config path used by the GUI to persist edits.

    The CLI still accepts an explicit ``--config <path>``; this is the
    fallback for when no path is given and the GUI's Settings dialog
    needs *somewhere* to save.
    """
    from .paths import appdata_dir

    return appdata_dir() / "config.yaml"


def save_config(cfg: AppConfig, path: Optional[Path] = None) -> Path:
    """Dump AppConfig to YAML. Returns the path actually written."""
    target = path or default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.model_dump(mode="json", exclude_none=False)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return target
