"""Frameless, always-on-top floating window for agent output.

Sibling of ``SubtitleWindow`` but with a much simpler model:

* One row per transcript turn (keyed by ``item_id``).
* Header line shows the agent label (``[补充讲解]`` / ``[面试者建议]``) and
  the source sentence that triggered it.
* Body line streams the agent's reply. While the reply is still
  streaming we render it in ``streaming_color`` (yellow by default);
  once ``AgentFinal`` lands we flip it to ``body_color`` (light teal
  by default).
* ``AgentSkipped`` either deletes the row outright (if it was a
  speculative one already showing partial text) or never creates one
  in the first place.

The window deliberately re-implements its own small scroll viewport
rather than re-using ``SubtitleWindow._RowsViewport`` because the agent
content scrolls differently (it's longer-form prose, not one-sentence
captions, so we want simple "append + scrollbar appears when needed"
behaviour rather than the stepped 1/3-of-viewport slide subtitles use).

Geometry, colours and opacity are persisted into ``AppConfig.agent_window``.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QContextMenuEvent,
    QGuiApplication,
    QMouseEvent,
    QPainter,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMenu,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..config import AgentWindowConfig

log = logging.getLogger(__name__)


_HARD_MAX_ENTRIES = 12


def _hex_with_alpha(hex_color: str, alpha: float) -> str:
    """Return an ``rgba(...)`` CSS colour with the requested alpha applied
    on top of a hex like ``#80cbc4``. Used by both header and body so
    the per-window text-opacity slider tints everything together.
    """
    c = QColor(hex_color)
    a = int(round(max(0.0, min(1.0, alpha)) * 255))
    return f"rgba({c.red()},{c.green()},{c.blue()},{a})"


def _make_wrapping_label(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lbl.setTextFormat(Qt.TextFormat.PlainText)
    lbl.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
    return lbl


class _AgentEntry(QFrame):
    """One agent reply row: header + body."""

    STATES = ("streaming", "final")

    def __init__(
        self,
        item_id: str,
        agent_label: str,
        source_excerpt: str,
        cfg: AgentWindowConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._item_id = item_id
        self._state: str = "streaming"
        self.setObjectName("AgentEntry")
        self.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._header = _make_wrapping_label(
            f"[{agent_label}] {source_excerpt}".strip()
        )
        self._body = _make_wrapping_label("")
        layout.addWidget(self._header)
        layout.addWidget(self._body)

        self.apply_styling(cfg)

    def item_id(self) -> str:
        return self._item_id

    def state(self) -> str:
        return self._state

    def set_state(self, state: str, cfg: AgentWindowConfig) -> None:
        if state not in self.STATES:
            return
        self._state = state
        self.apply_styling(cfg)

    def update_body(self, text: str) -> None:
        self._body.setText(text)

    def update_header(self, agent_label: str, source_excerpt: str) -> None:
        self._header.setText(f"[{agent_label}] {source_excerpt}".strip())

    def apply_styling(self, cfg: AgentWindowConfig) -> None:
        family = cfg.font_family
        header_size = cfg.heading_font_size_pt
        body_size = cfg.body_font_size_pt
        alpha = cfg.text_opacity
        header_color = _hex_with_alpha(cfg.heading_color, alpha)
        body_hex = cfg.streaming_color if self._state == "streaming" else cfg.body_color
        body_color = _hex_with_alpha(body_hex, alpha)

        self._header.setStyleSheet(
            f"color: {header_color}; "
            f"font-family: '{family}'; "
            f"font-size: {header_size}pt; "
            f"font-weight: 600;"
        )
        self._body.setStyleSheet(
            f"color: {body_color}; "
            f"font-family: '{family}'; "
            f"font-size: {body_size}pt; "
        )

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        lay = self.layout()
        if lay is None:
            return super().heightForWidth(width)
        h = lay.heightForWidth(max(1, int(width)))
        if h <= 0:
            return super().heightForWidth(width)
        return h


class AgentWindow(QWidget):
    """Floating overlay showing recent agent replies."""

    settings_changed = Signal(AgentWindowConfig)

    def __init__(
        self, cfg: AgentWindowConfig, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._agent_label: str = ""

        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.NoDropShadowWindowHint
        )
        if cfg.always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 10, 16, 10)
        outer.setSpacing(2)
        outer.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.setMinimumSize(220, 80)

        # Title bar with the active mode label, so the user can tell
        # the two overlays apart at a glance.
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        self._title_label = QLabel("Agent")
        self._title_label.setStyleSheet(
            "color: rgba(255,255,255,180); font-weight: 600; font-size: 11pt;"
        )
        title_row.addWidget(self._title_label, 1)
        outer.addLayout(title_row)

        # Scrolling region. We pick a real QScrollArea here (unlike the
        # stepped-slide viewport in SubtitleWindow) because agent
        # replies are paragraphs of prose, not one-second sentences,
        # and users want to be able to scroll back and re-read.
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._scroll.viewport().setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground, True
        )
        self._scroll.setStyleSheet("QScrollArea { background: transparent; }")

        self._rows_container = QWidget()
        self._rows_container.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground, True
        )
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(cfg.row_spacing_px)
        self._rows_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self._rows_layout.addStretch(1)
        self._scroll.setWidget(self._rows_container)
        outer.addWidget(self._scroll, 1)

        # Resize grip in the bottom-right corner.
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(16, 16)
        grip_row.addWidget(
            self._grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom
        )
        outer.addLayout(grip_row)

        self._entries: dict[str, _AgentEntry] = {}
        self._entry_order: list[str] = []
        self._source_excerpts: dict[str, str] = {}

        self._set_click_through(cfg.click_through)
        self._apply_geometry()
        self._drag_origin: Optional[QPoint] = None

    # ------------------------------------------------------------------ Public API

    def set_agent_label(self, label: str) -> None:
        """Update the title bar + every existing row header to use ``label``.

        Called when the user picks a new agent mode in settings.
        """
        self._agent_label = label
        self._title_label.setText(label)
        for item_id, entry in self._entries.items():
            entry.update_header(label, self._source_excerpts.get(item_id, ""))

    def apply_config(self, cfg: AgentWindowConfig) -> None:
        self._cfg = cfg
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.NoDropShadowWindowHint
        )
        if cfg.always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        was_visible = self.isVisible()
        self.setWindowFlags(flags)
        if was_visible:
            self.show()
        self._set_click_through(cfg.click_through)
        self._rows_layout.setSpacing(cfg.row_spacing_px)
        for entry in self._entries.values():
            entry.apply_styling(cfg)
        self.resize(cfg.width, cfg.height)
        self._apply_geometry()
        self.update()

    def on_transcript_final(self, item_id: str, text: str) -> None:
        """Remember the source text per item so the row header can show
        it as soon as the first agent delta lands. We don't create the
        row here -- that happens on the first AgentDelta -- because
        Skipped turns should leave no visual trace.
        """
        self._source_excerpts[item_id] = _excerpt(text)

    def on_agent_delta(self, item_id: str, agent_id: str, text_so_far: str) -> None:
        if not text_so_far:
            return
        entry = self._entries.get(item_id)
        if entry is None:
            entry = self._add_entry(item_id)
        if entry.state() != "streaming":
            entry.set_state("streaming", self._cfg)
        entry.update_body(text_so_far)
        self._scroll_to_bottom()

    def on_agent_final(self, item_id: str, agent_id: str, text: str) -> None:
        if not text:
            # Treat empty final as a skip.
            self.on_agent_skipped(item_id, agent_id, "empty")
            return
        entry = self._entries.get(item_id)
        if entry is None:
            entry = self._add_entry(item_id)
        entry.update_body(text)
        entry.set_state("final", self._cfg)
        self._scroll_to_bottom()

    def on_agent_skipped(self, item_id: str, agent_id: str, reason: str) -> None:
        entry = self._entries.pop(item_id, None)
        if entry is None:
            return
        if item_id in self._entry_order:
            self._entry_order.remove(item_id)
        self._rows_layout.removeWidget(entry)
        entry.setParent(None)
        entry.deleteLater()

    def clear(self) -> None:
        for item_id in list(self._entry_order):
            entry = self._entries.pop(item_id, None)
            if entry is not None:
                self._rows_layout.removeWidget(entry)
                entry.setParent(None)
                entry.deleteLater()
        self._entry_order.clear()
        self._source_excerpts.clear()

    # ------------------------------------------------------------------ Internals

    def _add_entry(self, item_id: str) -> _AgentEntry:
        excerpt = self._source_excerpts.get(item_id, "")
        entry = _AgentEntry(
            item_id=item_id,
            agent_label=self._agent_label or "Agent",
            source_excerpt=excerpt,
            cfg=self._cfg,
            parent=self._rows_container,
        )
        # Insert just before the trailing stretch so newest is at the bottom.
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, entry)
        self._entries[item_id] = entry
        self._entry_order.append(item_id)
        self._enforce_max_entries()
        return entry

    def _enforce_max_entries(self) -> None:
        limit = max(1, min(_HARD_MAX_ENTRIES, self._cfg.max_visible_entries))
        while len(self._entry_order) > limit:
            oldest = self._entry_order.pop(0)
            self._source_excerpts.pop(oldest, None)
            widget = self._entries.pop(oldest, None)
            if widget is not None:
                self._rows_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())

    def _apply_geometry(self) -> None:
        c = self._cfg
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(c.width, c.height)
            return
        avail: QRect = screen.availableGeometry()
        x = c.x if c.x is not None else avail.x() + avail.width() - c.width - 40
        y = c.y if c.y is not None else avail.y() + 80
        x = max(avail.x(), min(x, avail.x() + avail.width() - c.width))
        y = max(avail.y(), min(y, avail.y() + avail.height() - c.height))
        self.setGeometry(x, y, c.width, c.height)

    def _set_click_through(self, enabled: bool) -> None:
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)

    # ------------------------------------------------------------------ Painting

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        alpha = int(round(max(0.0, min(1.0, self._cfg.background_opacity)) * 255))
        bg = QColor(0, 0, 0, alpha)
        painter.setBrush(bg)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 14, 14)

    # ------------------------------------------------------------------ Input

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_origin is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            new_top_left = event.globalPosition().toPoint() - self._drag_origin
            self.move(new_top_left)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_origin is not None:
            self._drag_origin = None
            self._persist_geometry()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.hide()
            event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._persist_geometry()

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802
        menu = QMenu(self)
        act_hide = QAction("隐藏 Agent 浮窗", self)
        act_hide.triggered.connect(self.hide)
        menu.addAction(act_hide)

        act_clear = QAction("清空 Agent", self)
        act_clear.triggered.connect(self.clear)
        menu.addAction(act_clear)

        menu.addSeparator()

        act_click_through = QAction("鼠标穿透", self, checkable=True)
        act_click_through.setChecked(self._cfg.click_through)
        act_click_through.toggled.connect(self._toggle_click_through)
        menu.addAction(act_click_through)

        act_on_top = QAction("置顶显示", self, checkable=True)
        act_on_top.setChecked(self._cfg.always_on_top)
        act_on_top.toggled.connect(self._toggle_always_on_top)
        menu.addAction(act_on_top)

        menu.exec(event.globalPos())

    def _toggle_click_through(self, on: bool) -> None:
        self._cfg = self._cfg.model_copy(update={"click_through": on})
        self._set_click_through(on)
        self.settings_changed.emit(self._cfg)

    def _toggle_always_on_top(self, on: bool) -> None:
        self._cfg = self._cfg.model_copy(update={"always_on_top": on})
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.NoDropShadowWindowHint
        )
        if on:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        was_visible = self.isVisible()
        self.setWindowFlags(flags)
        if was_visible:
            self.show()
        self.settings_changed.emit(self._cfg)

    def _persist_geometry(self) -> None:
        g = self.geometry()
        self._cfg = self._cfg.model_copy(
            update={"x": g.x(), "y": g.y(), "width": g.width(), "height": g.height()}
        )
        self.settings_changed.emit(self._cfg)


def _excerpt(text: str, max_chars: int = 60) -> str:
    """Trim a long transcript line to a short header preview."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"
