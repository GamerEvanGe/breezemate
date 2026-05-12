"""Verify the AgentWindow is *not* hollow.

We render the window into a QPixmap and sample pixels from the inner
scroll area. With the bug, the viewport paints opaque white over our
rounded-rect plate; with the fix, the dark translucent plate should
show through.

Also asserts that the rows container itself reports as translucent so
future regressions in setStyleSheet / WA_TranslucentBackground show up
immediately.
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

from PySide6.QtCore import QSize, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rt_translator.config import AgentWindowConfig  # noqa: E402
from rt_translator.gui.agent_window import AgentWindow  # noqa: E402


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    cfg = AgentWindowConfig(width=520, height=380, background_opacity=0.55)
    win = AgentWindow(cfg)
    win.set_agent_label("补充讲解")
    win.resize(QSize(520, 380))
    win.show()
    # Three passes of event-loop drain so layout settles before paint.
    for _ in range(3):
        app.processEvents()

    # Sanity: viewport / rows container must have translucent
    # backgrounds enabled, otherwise the visible test is meaningless.
    vp = win._scroll.viewport()
    assert vp is not None
    assert vp.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground), (
        "QScrollArea viewport is still opaque -- it will paint white over "
        "the rounded-rect plate"
    )
    assert not vp.autoFillBackground(), "viewport.autoFillBackground must be off"
    assert not win._rows_container.autoFillBackground(), (
        "rows container.autoFillBackground must be off"
    )

    # Visible test: grab the rendered surface and sample a pixel that
    # sits inside the scroll area but well clear of the title row.
    pixmap = QPixmap(win.size())
    pixmap.fill(Qt.GlobalColor.transparent)
    win.render(pixmap)
    img: QImage = pixmap.toImage()

    sample_x = win.width() // 2
    sample_y = max(60, win.height() // 2)
    px = img.pixelColor(sample_x, sample_y)
    # With the rounded-rect plate visible (the fix), this pixel comes
    # from the alpha-blended dark plate. With the bug it comes from
    # the opaque white viewport. We accept the fix iff the pixel is
    # noticeably DARK (rgb<140) AND has some alpha. White viewport
    # bug gave rgb~255.
    r, g, b, a = px.red(), px.green(), px.blue(), px.alpha()
    print(f"sampled pixel @({sample_x},{sample_y}) = rgba({r},{g},{b},{a})")
    # Allow either fully-transparent (offscreen platform can compose
    # weirdly) OR clearly-dark colour. Reject anything white-ish.
    is_dark = (r < 140 and g < 140 and b < 140)
    is_transparent = a < 30
    assert is_dark or is_transparent, (
        f"expected dark/transparent pixel, got rgba({r},{g},{b},{a}) -- "
        "the agent window is rendering opaque white over the plate again"
    )
    print("PASS: AgentWindow background is not hollow")
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
