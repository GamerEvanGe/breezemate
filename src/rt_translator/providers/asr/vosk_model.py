"""On-demand downloader for Vosk speech recognition models.

Vosk ships its models as ZIP archives on alphacephei.com. We download
once to ``%APPDATA%\\rt-translator\\vosk-models\\<name>\\`` and reuse on
every subsequent launch.

We deliberately ONLY catalog the "small" CPU models (~30-80 MB each).
Alphacephei's "full" models (~1-2 GB) consistently underperform the
small ones on real-world subtitle workloads in our testing -- they
load 10x slower, eat far more RAM, and the extra context buys little
because we punt punctuation and grammar fixes to the downstream LLM
polishing step anyway. Hand-curated single-tier catalog keeps the
settings UI simple ("pick a language, hit Download") and avoids the
"why is the bigger model worse" footgun.

Functions are blocking on purpose. Call them from a background thread
(the GUI does this from a QThread) so the main / UI loop stays responsive.
"""

from __future__ import annotations

import io
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from ...paths import appdata_dir

log = logging.getLogger(__name__)


# Catalog. Key is the on-disk folder name (== the zip's top-level dir);
# the dict carries:
#   url            -- canonical Alphacephei download URL
#   size_mb        -- approximate compressed size, used for the
#                     "are you sure you want to download N MB" prompt
#   language       -- ISO 639-1 / locale-ish short code, the *language*
#                     selector in the settings dialog groups by this
#   language_label -- human-readable label in the native script
#   quality        -- short tag shown in the model dropdown
_KNOWN_MODELS: dict[str, dict[str, str]] = {
    # ----- English -----
    "vosk-model-small-en-us-0.15": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "size_mb": "42",
        "language": "en",
        "language_label": "English",
        "quality": "small",
    },
    # ----- Chinese -----
    "vosk-model-small-cn-0.22": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
        "size_mb": "42",
        "language": "zh",
        "language_label": "中文 (普通话)",
        "quality": "small",
    },
    # ----- Japanese -----
    "vosk-model-small-ja-0.22": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip",
        "size_mb": "48",
        "language": "ja",
        "language_label": "日本語",
        "quality": "small",
    },
    # ----- Russian -----
    "vosk-model-small-ru-0.22": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
        "size_mb": "45",
        "language": "ru",
        "language_label": "Русский",
        "quality": "small",
    },
    # ----- French -----
    "vosk-model-small-fr-0.22": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
        "size_mb": "41",
        "language": "fr",
        "language_label": "Français",
        "quality": "small",
    },
    # ----- German -----
    "vosk-model-small-de-0.15": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
        "size_mb": "45",
        "language": "de",
        "language_label": "Deutsch",
        "quality": "small",
    },
    # ----- Spanish -----
    "vosk-model-small-es-0.42": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
        "size_mb": "39",
        "language": "es",
        "language_label": "Español",
        "quality": "small",
    },
    # ----- Italian -----
    "vosk-model-small-it-0.22": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip",
        "size_mb": "48",
        "language": "it",
        "language_label": "Italiano",
        "quality": "small",
    },
    # ----- Portuguese -----
    "vosk-model-small-pt-0.3": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
        "size_mb": "31",
        "language": "pt",
        "language_label": "Português",
        "quality": "small",
    },
    # ----- Korean -----
    "vosk-model-small-ko-0.22": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip",
        "size_mb": "82",
        "language": "ko",
        "language_label": "한국어",
        "quality": "small",
    },
    # ----- Vietnamese -----
    "vosk-model-small-vn-0.4": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-vn-0.4.zip",
        "size_mb": "32",
        "language": "vi",
        "language_label": "Tiếng Việt",
        "quality": "small",
    },
    # ----- Hindi -----
    "vosk-model-small-hi-0.22": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
        "size_mb": "42",
        "language": "hi",
        "language_label": "हिन्दी",
        "quality": "small",
    },
    # ----- Turkish -----
    "vosk-model-small-tr-0.3": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip",
        "size_mb": "35",
        "language": "tr",
        "language_label": "Türkçe",
        "quality": "small",
    },
    # ----- Polish -----
    "vosk-model-small-pl-0.22": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip",
        "size_mb": "50",
        "language": "pl",
        "language_label": "Polski",
        "quality": "small",
    },
    # ----- Ukrainian -----
    "vosk-model-small-uk-v3-small": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-uk-v3-small.zip",
        "size_mb": "75",
        "language": "uk",
        "language_label": "Українська",
        "quality": "small",
    },
}


DEFAULT_MODEL_ID = "vosk-model-small-en-us-0.15"

# Display order for the language dropdown. Keep "popular for our user
# base" near the top; everything else falls back to alphabetical by code.
_LANGUAGE_ORDER = (
    "en",
    "zh",
    "ja",
    "ko",
    "ru",
    "fr",
    "de",
    "es",
    "it",
    "pt",
    "pl",
    "uk",
    "tr",
    "vi",
    "hi",
)


def models_dir() -> Path:
    target = appdata_dir() / "vosk-models"
    target.mkdir(parents=True, exist_ok=True)
    return target


def model_path(model_id: str) -> Path:
    """Path on disk where this model SHOULD live (may not exist yet)."""
    return models_dir() / model_id


def is_model_present(model_id: str) -> bool:
    """A model dir is "present" when it has the bare-minimum files
    Vosk needs to construct a ``Model``.

    We check the presence of the ``am/`` and ``conf/`` subdirs since
    those are universal across vosk models; a missing tarball / partial
    extract typically leaves them out."""
    root = model_path(model_id)
    if not root.is_dir():
        return False
    return (root / "am").is_dir() and (root / "conf").is_dir()


def list_known_models() -> dict[str, dict[str, str]]:
    """Return a *copy* of the model catalog for UI display."""
    return {k: dict(v) for k, v in _KNOWN_MODELS.items()}


def available_languages() -> list[tuple[str, str]]:
    """Distinct ``(language_code, language_label)`` pairs, in display order.

    ``language_label`` is the user-facing string written in the native
    script (e.g. "中文 (普通话)" for ``zh``). Codes with multiple models
    only appear once; the label of the *first* matching model wins.
    """
    seen: dict[str, str] = {}
    for meta in _KNOWN_MODELS.values():
        code = meta.get("language", "")
        if not code or code in seen:
            continue
        seen[code] = meta.get("language_label", code)

    def sort_key(code: str) -> tuple[int, str]:
        try:
            return (_LANGUAGE_ORDER.index(code), code)
        except ValueError:
            return (len(_LANGUAGE_ORDER), code)

    return [(c, seen[c]) for c in sorted(seen.keys(), key=sort_key)]


def models_for_language(lang_code: str) -> dict[str, dict[str, str]]:
    """Subset of the catalog whose ``language`` field matches ``lang_code``.

    Today every language has exactly one entry (the small model), but
    we keep the function returning a dict so the settings dialog can
    still drive a combobox -- and so the catalog can grow back to
    multi-tier later without a UI rewrite.
    """
    return {
        mid: dict(meta)
        for mid, meta in _KNOWN_MODELS.items()
        if meta.get("language") == lang_code
    }


def language_of(model_id: str) -> Optional[str]:
    """Return the language code of a known model, or None if unknown."""
    meta = _KNOWN_MODELS.get(model_id)
    return meta.get("language") if meta else None


def download_model(
    model_id: str,
    progress: Optional[Callable[[int, int], None]] = None,
    force: bool = False,
) -> Path:
    """Download + extract the given Vosk model.

    ``progress(downloaded_bytes, total_bytes)`` is called periodically
    if provided; ``total_bytes`` may be 0 if the server omits
    Content-Length. Returns the on-disk model directory.

    Raises on download / extraction failure.
    """
    if model_id not in _KNOWN_MODELS:
        raise ValueError(
            f"Unknown vosk model id: {model_id}. "
            f"Known: {', '.join(_KNOWN_MODELS)}"
        )

    target = model_path(model_id)
    if target.exists() and not force and is_model_present(model_id):
        log.debug("Vosk model %s already present at %s", model_id, target)
        return target

    url = _KNOWN_MODELS[model_id]["url"]
    log.info("Downloading Vosk model %s from %s", model_id, url)

    # Stream into memory; every model we ship is <100 MB so RAM is
    # not a concern, and streaming to a temp file would need cleanup
    # paths on every error.
    req = urllib.request.Request(url, headers={"User-Agent": "BreezeMate"})
    buf = io.BytesIO()
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        chunk_size = 64 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            buf.write(chunk)
            downloaded += len(chunk)
            if progress is not None:
                try:
                    progress(downloaded, total)
                except Exception:
                    log.debug("Vosk download progress callback raised", exc_info=True)

    log.info("Vosk model %s downloaded (%.1f MB), extracting...", model_id, downloaded / 1e6)

    # Wipe a half-extracted previous attempt before unpacking afresh.
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    models_dir().mkdir(parents=True, exist_ok=True)

    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        # The zip is structured as ``<model_id>/...``. Extract into
        # models_dir() and ZipFile will recreate that top-level folder
        # automatically.
        zf.extractall(models_dir())

    if not is_model_present(model_id):
        raise RuntimeError(
            f"Vosk model {model_id} extracted but doesn't look complete. "
            f"Check {target} on disk."
        )

    log.info("Vosk model ready at %s", target)
    return target


def ensure_model(
    model_id: str,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download if missing, otherwise return the existing path."""
    if is_model_present(model_id):
        return model_path(model_id)
    return download_model(model_id, progress=progress)
