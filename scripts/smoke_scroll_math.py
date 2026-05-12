"""Smoke test: subtitle window stepped-scroll math under a long preview.

Validates two things:

1. After our heightForWidth fix, a _SubtitleEntry containing a long
   word-wrapped paragraph reports a height-for-width that matches what
   QLabel actually paints -- i.e. many hundreds of pixels for a
   multi-line paragraph, not the ~20px single-line fallback.
2. With that correct height, _RowsViewport.target_offset() steps in
   vh/3 increments and the bottom of content always lands at 2/3 of
   the viewport, never above it (which was the user's "text all goes
   past the top" complaint).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtWidgets import QApplication, QWidget

from rt_translator.config import SubtitleWindowConfig
from rt_translator.gui.subtitle_window import _RowsViewport, _SubtitleEntry


LOREM = (
    "this is the kind of long live-preview text that vosk produces when "
    "a speaker just keeps talking without pausing for breath and it has "
    "to wrap across many lines inside the floating subtitle overlay "
    "because the speaker really truly just will not stop and the "
    "preview keeps growing and growing and growing until the duration "
    "cap finally fires or the user gives up listening and stops the "
    "pipeline themselves"
) * 2


def main():
    app = QApplication.instance() or QApplication([])
    cfg = SubtitleWindowConfig(width=600, height=200)

    host = QWidget()
    host.resize(cfg.width, cfg.height)

    viewport = _RowsViewport(row_spacing=cfg.row_spacing_px, parent=host)
    viewport.setGeometry(0, 0, cfg.width, cfg.height)
    viewport.show()  # required for heightForWidth to work properly

    # Add a single long preview entry.
    entry = _SubtitleEntry(
        item_id="preview", translate_mode=False, cfg=cfg, parent=host, state="preview"
    )
    entry.update_en(LOREM)
    viewport.inner_layout().addWidget(entry)
    app.processEvents()

    width = viewport.width()
    entry_hfw = entry.heightForWidth(width)
    layout_hfw = viewport.inner_layout().heightForWidth(width)
    inner_h = viewport.inner_natural_height()
    print(
        f"viewport width={width} | entry heightForWidth={entry_hfw} | "
        f"layout heightForWidth={layout_hfw} | inner_natural_height={inner_h}"
    )

    # Sanity: the entry must report a non-trivial wrapped height, NOT
    # the single-line fallback (~20-40 px).
    assert entry_hfw > 80, (
        f"_SubtitleEntry.heightForWidth({width})={entry_hfw} is unreasonably "
        "small -- heightForWidth override probably isn't firing."
    )
    # The viewport's inner_natural_height should reflect the same.
    assert inner_h >= entry_hfw - 10, (
        f"inner_natural_height={inner_h} much smaller than entry height={entry_hfw}"
    )

    # Now drive target_offset() and verify the rule. The first call
    # with inner_h >> vh should trigger a step that places content
    # bottom at exactly 2/3 of vh (within a few px of rounding).
    vh = viewport.height()
    target = viewport.target_offset()
    bottom_in_vp = inner_h - target
    expected_bottom = (2 * vh) // 3
    print(
        f"vh={vh} | target_offset={target} | bottom_in_vp={bottom_in_vp} | "
        f"expected_bottom (~2/3 vh)={expected_bottom}"
    )
    # Allow a 2 px rounding tolerance.
    assert abs(bottom_in_vp - expected_bottom) <= 2, (
        f"after one slide, content bottom should sit at ~2*vh/3 = {expected_bottom}; "
        f"actually got bottom_in_vp={bottom_in_vp}."
    )
    # Crucially, target must NOT push content off the top of the
    # viewport. Visible content top edge = max(0, target). If target
    # >= inner_h the viewport would be empty, which is what the user
    # observed before the fix.
    assert target < inner_h, (
        f"target_offset={target} >= inner_h={inner_h}: viewport would be empty."
    )

    print("=== scroll math OK ===")


if __name__ == "__main__":
    main()
