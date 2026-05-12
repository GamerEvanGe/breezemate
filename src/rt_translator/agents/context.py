"""Load user-uploaded reference files into a single context string.

Supported formats (decided per file by extension):

  .txt / .md       -> plain text, UTF-8 with fallback
  .pdf             -> pypdf.PdfReader -> per-page text join
  .docx            -> python-docx -> paragraph join

Hard-coded safeguards:

* Per-file byte cap (default 8 MB) so a stray binary doesn't get
  decoded as garbage.
* Aggregate character cap (the agent config's ``max_context_chars``)
  enforced AFTER concatenation; we trim from the end of each file in
  insertion order so newer uploads keep their head.
* Each file is wrapped with a clearly-named header so the LLM can
  refer back to it.

PDF / DOCX dependencies are imported lazily so the rest of BreezeMate
still works in environments where they happen to be missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

# Anything larger than this is almost certainly not a "context note"
# and would just burn tokens. Adjust per file as needed.
_PER_FILE_BYTES_LIMIT = 8 * 1024 * 1024

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx")


@dataclass(frozen=True)
class ContextSection:
    """One loaded reference file."""

    path: Path
    text: str
    chars: int
    truncated: bool = False
    error: Optional[str] = None

    @property
    def display_name(self) -> str:
        return self.path.name


class ContextStore:
    """Loads + caches the user-uploaded reference files.

    Holds raw per-file text plus a single concatenated string that
    agents stitch directly into their system prompt. The store is
    cheap to rebuild (single-digit megabytes typically); we just
    recreate it whenever the user adds / removes a file rather than
    juggling cache invalidation.
    """

    def __init__(self, paths: Iterable[str | Path], max_total_chars: int) -> None:
        self._max_total_chars = max(0, int(max_total_chars or 0))
        self._sections: list[ContextSection] = []
        self._combined: str = ""
        self._rebuild([Path(p) for p in paths])

    @property
    def sections(self) -> list[ContextSection]:
        return list(self._sections)

    @property
    def combined(self) -> str:
        """Single string suitable for splicing into an agent system prompt.

        Format: a small header per file followed by its content,
        separated by blank lines. Empty when no files are loaded.
        """
        return self._combined

    def _rebuild(self, paths: list[Path]) -> None:
        if self._max_total_chars == 0:
            self._sections = []
            self._combined = ""
            return

        sections: list[ContextSection] = []
        budget = self._max_total_chars
        for p in paths:
            section = _load_one(p)
            if section is None:
                continue
            # Trim per file to leave budget for the next ones.
            if len(section.text) > budget:
                trimmed = section.text[: max(0, budget)]
                section = ContextSection(
                    path=section.path,
                    text=trimmed,
                    chars=len(trimmed),
                    truncated=True,
                    error=section.error,
                )
            sections.append(section)
            budget -= section.chars
            if budget <= 0:
                break
        self._sections = sections

        if not sections:
            self._combined = ""
            return

        parts: list[str] = []
        for s in sections:
            header = f"--- {s.display_name} ---"
            tail = "" if not s.truncated else "\n[(truncated to fit context budget)]"
            parts.append(f"{header}\n{s.text.strip()}{tail}")
        self._combined = "\n\n".join(parts)


def _load_one(path: Path) -> Optional[ContextSection]:
    if not path.exists() or not path.is_file():
        log.warning("Context file missing: %s", path)
        return ContextSection(path=path, text="", chars=0, error="missing")
    try:
        size = path.stat().st_size
    except OSError as e:
        log.warning("Stat failed for %s: %s", path, e)
        return ContextSection(path=path, text="", chars=0, error=str(e))
    if size > _PER_FILE_BYTES_LIMIT:
        log.warning(
            "Skipping %s (%.1f MB > %.0f MB cap)",
            path,
            size / 1_048_576,
            _PER_FILE_BYTES_LIMIT / 1_048_576,
        )
        return ContextSection(path=path, text="", chars=0, error="too large")

    suffix = path.suffix.lower()
    try:
        if suffix in (".txt", ".md", ""):
            text = _read_text(path)
        elif suffix == ".pdf":
            text = _read_pdf(path)
        elif suffix == ".docx":
            text = _read_docx(path)
        else:
            log.warning("Unsupported context extension: %s", path)
            return ContextSection(path=path, text="", chars=0, error="unsupported")
    except Exception as e:  # noqa: BLE001 -- best-effort loader
        log.exception("Failed to load context file %s", path)
        return ContextSection(path=path, text="", chars=0, error=str(e))

    cleaned = text.strip()
    return ContextSection(path=path, text=cleaned, chars=len(cleaned))


def _read_text(path: Path) -> str:
    # UTF-8 first, then a few common Windows fallbacks. Anything weirder
    # than that is the user's problem.
    for encoding in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "pypdf is not installed; cannot read PDF context files. "
            "Run `uv add pypdf` or install BreezeMate's PDF extras."
        ) from e
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - some pages are unparseable
            continue
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    try:
        import docx  # type: ignore  # provided by python-docx
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "python-docx is not installed; cannot read .docx context files."
        ) from e
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs if p.text)
