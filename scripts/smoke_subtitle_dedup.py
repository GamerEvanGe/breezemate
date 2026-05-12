"""Smoke test for the dual-engine subtitle de-duplication.

Goal: when OpenAI Realtime is the canonical ASR and Vosk runs as a
preview, the floating window must show ONE row per spoken sentence,
not two (one for Vosk's noisy partial + one for the cloud's polished
text covering the same audio).

We drive ``SubtitleWindow`` directly with the same slot signals the
pipeline emits, in the same order. The headless Qt test:

* Vosk emits a preview ``vosk-1``        -> 1 row.
* Cloud emits its first delta ``oai-1``  -> Vosk row dropped via the
                                           ``preview_reset`` signal,
                                           cloud row created, dedup
                                           flag flips ON.
* Vosk emits a fresh ``vosk-2`` partial  -> SUPPRESSED (would otherwise
                                           show as a duplicate under the
                                           cloud row).
* Cloud emits ``oai-1`` final            -> dedup flag flips OFF.
* Vosk emits a ``vosk-3`` preview        -> shown again (next sentence).
* Cloud emits ``oai-2`` delta            -> Vosk row dropped, cloud row
                                           created, dedup flag ON.

Pass criterion: at every checkpoint we see the expected entry set
(no doubled rows).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from rt_translator.config import SubtitleWindowConfig  # noqa: E402
from rt_translator.gui.subtitle_window import SubtitleWindow  # noqa: E402


def _ids(win: SubtitleWindow) -> list[str]:
    return list(win._entry_order)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    cfg = SubtitleWindowConfig()
    win = SubtitleWindow(cfg)

    # ---- sentence 1 -----------------------------------------------------
    win.on_preview_delta("vosk-1", "hello")
    win.on_preview_delta("vosk-1", "hello world")
    app.processEvents()
    assert _ids(win) == ["vosk-1"], _ids(win)
    assert win._cloud_in_progress is False

    # Cloud's first delta lands. The pipeline router would emit
    # LocalPreviewReset for "vosk-1" immediately before this; we
    # mirror that ordering.
    win.on_preview_reset("vosk-1")
    win.on_transcript_delta("oai-1", "Hello")
    app.processEvents()
    assert _ids(win) == ["oai-1"], _ids(win)
    assert win._cloud_in_progress is True

    # Vosk resumed under a fresh utterance id. THIS is the event that
    # used to create the visible duplicate. With the fix it must be
    # silently dropped.
    win.on_preview_delta("vosk-2", "hello world")
    win.on_preview_delta("vosk-2", "hello world how are you")
    app.processEvents()
    assert _ids(win) == ["oai-1"], _ids(win)
    assert win._cloud_in_progress is True

    # Cloud finishes the sentence.
    win.on_transcript_delta("oai-1", "Hello, world. How are you")
    win.on_transcript_final("oai-1", "Hello, world. How are you?")
    app.processEvents()
    assert _ids(win) == ["oai-1"], _ids(win)
    assert win._cloud_in_progress is False

    # ---- sentence 2 -----------------------------------------------------
    # User pauses then resumes. Vosk previews the NEXT sentence; we
    # want this one to show, since no cloud sentence is in flight.
    win.on_preview_delta("vosk-3", "fine")
    app.processEvents()
    assert _ids(win) == ["oai-1", "vosk-3"], _ids(win)

    # Cloud overtakes the new sentence: vosk-3 must be dropped (existing
    # router-driven path), oai-2 created, dedup flag flips ON.
    win.on_preview_reset("vosk-3")
    win.on_transcript_delta("oai-2", "Fine")
    app.processEvents()
    assert _ids(win) == ["oai-1", "oai-2"], _ids(win)
    assert win._cloud_in_progress is True

    # And the leaked late Vosk partial is again suppressed.
    win.on_preview_delta("vosk-4", "fine thanks")
    app.processEvents()
    assert _ids(win) == ["oai-1", "oai-2"], _ids(win)

    # ---- vosk-canonical mode (no cloud) --------------------------------
    # Reset and run a quick check that pure-Vosk flow is untouched: no
    # event has a cloud id, so the dedup flag must never trip and
    # previews must remain visible.
    win.clear()
    assert win._cloud_in_progress is False
    win.on_preview_delta("vosk-A", "good")
    win.on_preview_delta("vosk-A", "good morning")
    win.on_transcript_final("vosk-A", "good morning")
    win.on_preview_delta("vosk-B", "everyone")
    app.processEvents()
    assert _ids(win) == ["vosk-A", "vosk-B"], _ids(win)
    assert win._cloud_in_progress is False

    print("PASS: subtitle dedup works in dual-engine and Vosk-canonical modes")
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
