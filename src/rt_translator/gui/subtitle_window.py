"""Frameless, always-on-top floating subtitle overlay.

Renders a *scrolling* list of recent utterances. Each utterance is a
``_SubtitleEntry`` row showing the source transcript on top and (when
translation mode is on) the translation below.

Three visual states per entry:

* ``preview``  -- Vosk is still streaming partials for this utterance.
                  Rendered with ``preview_color`` + italic so the user
                  reads it as "draft text". No translation row visible.
* ``active``   -- ``TranscriptFinal`` has landed (Vosk locked the
                  sentence). Switches to the regular ASR colour and
                  the translator's polished-original + translation
                  deltas now stream into this same entry.
* ``history``  -- A newer utterance has shown up; this row is no
                  longer being updated. Dimmed to match the older
                  rows above.

There is NO separate "preview row" widget below the viewport any
more; the in-progress utterance is always the bottom-most entry in
the scrolling area, and behaves identically to history rows for
sliding / clipping. When Vosk finalises, the same widget stays in
place -- only its styling and text source change.

Behaviour:

* A new utterance (new ``item_id`` from the pipeline) is appended at the
  bottom. The previous "active" entry becomes a finalised history row.
* When the number of rows exceeds ``max_visible_entries`` (default 4),
  the oldest row is removed -- visually the whole list scrolls up one
  line. No QScrollArea is used; we just delete the top widget.
* Per-utterance updates keep mutating the same row's labels until that
  utterance is replaced by a newer one.

Interaction:

* Left-click + drag       -> move the window
* Right-click             -> context menu (opacity, font, click-through, hide)
* Drag from bottom-right  -> resize (Qt.SizeGripHandle)
* Double-click            -> hide (re-show from the main window's tray icon)

The overlay's geometry is ALWAYS the user-set size. We pin the outer
layout's size constraint to ``SetNoConstraint`` so a single un-wrappable
word can't force Qt to grow the window on its own -- only an explicit
drag of the size-grip changes the size.

Geometry, opacity, font sizes and history depth are persisted into
``AppConfig.subtitle_window``.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import (
    Property,
    QAbstractAnimation,
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
    Signal,
)
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
    QSizeGrip,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..config import SubtitleWindowConfig

log = logging.getLogger(__name__)


# Cap on how many history rows we ever keep alive. Larger values make
# the window taller; we cap so the layout cannot grow unbounded even if
# the user resizes the window very large.
_HARD_MAX_ENTRIES = 12


EntryState = str  # one of: "preview", "active", "history"


class _SubtitleEntry(QFrame):
    """One subtitle row: source transcript + optional translation.

    Holds a stable ``item_id`` so the parent can route deltas to the
    right entry without re-creating it. Styling is re-applied via
    ``apply_styling`` so the overlay's "Settings -> font size" etc.
    propagates to every row, including history rows already on screen.

    The row goes through three states over its lifetime:

    * ``preview``  -- Vosk is still streaming partials; rendered with
                      ``preview_color`` + italic. No translation row
                      visible even in translate mode (there is no
                      polished text to translate yet).
    * ``active``   -- Vosk finalised the sentence. Switches to the
                      regular ASR colour; the translation row appears
                      and starts filling with deltas from the LLM.
    * ``history``  -- Superseded by a newer utterance; dimmed.
    """

    def __init__(
        self,
        item_id: str,
        translate_mode: bool,
        cfg: SubtitleWindowConfig,
        parent: Optional[QWidget] = None,
        state: EntryState = "preview",
    ) -> None:
        super().__init__(parent)
        self.item_id = item_id
        self._translate_mode = translate_mode
        # Newly created entries default to "preview" -- live ASR
        # partials feed into ``update_en`` and the row paints with
        # preview styling until the matching TranscriptFinal lands.
        self._state: EntryState = state
        # Internal cached texts so we can swap mode (asr_only / translate)
        # without losing what's already been transcribed.
        self._en_text = ""
        self._cn_text = ""

        self.setFrameShape(QFrame.Shape.NoFrame)
        # Horizontal: Preferred -- the frame fills whatever width the
        # parent layout offers and grows/shrinks with it. Vertical:
        # *Preferred*, NOT MinimumExpanding. We want the entry to claim
        # exactly its wrapped natural height and nothing more. Combined
        # with AlignTop on the parent layout, this keeps the gap
        # between consecutive sentences EXACTLY ``row_spacing_px``
        # regardless of how tall the overlay window happens to be. With
        # MinimumExpanding, Qt would split any leftover vertical space
        # between entries and the gap would visually drift when the
        # user resizes the window.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        # EN and CN must read as ONE visual unit ("a sentence"), so
        # zero internal spacing and zero internal margins. Inter-row
        # spacing is handled by the parent rows_layout.
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Don't let this child layout poke back at our parent's geometry
        # (we manage the inner widget's size manually in _RowsViewport).
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        self._en_label = _make_wrapping_label()
        self._cn_label = _make_wrapping_label()

        layout.addWidget(self._en_label)
        layout.addWidget(self._cn_label)
        # Translation label is created visible-when-translate-mode, but
        # in "preview" state we hide it regardless -- a partial source
        # has no translation yet so an empty CN row would just waste
        # vertical space.
        self._update_cn_visibility()

        self.apply_styling(cfg)

    # ------------------------------------------------------------------ API

    def update_en(self, text: str) -> None:
        self._en_text = text
        self._en_label.setText(text)

    def update_cn(self, text: str) -> None:
        self._cn_text = text
        self._cn_label.setText(text)

    def current_text(self) -> str:
        """Latest source text held by this entry. Used by the parent
        window when promoting a preview row to a TranscriptFinal."""
        return self._en_text

    def set_translate_mode(self, on: bool) -> None:
        self._translate_mode = on
        self._update_cn_visibility()

    def set_state(self, state: EntryState, cfg: SubtitleWindowConfig) -> None:
        """Switch this row's lifecycle state and re-apply styling."""
        if self._state == state:
            return
        self._state = state
        self._update_cn_visibility()
        self.apply_styling(cfg)

    def state(self) -> EntryState:
        return self._state

    def _update_cn_visibility(self) -> None:
        # Hide the translation row whenever we're still in the preview
        # phase OR translate-mode is off; everything else shows it
        # (active entries fill it with deltas, history entries display
        # the locked-in translation).
        self._cn_label.setVisible(
            self._translate_mode and self._state != "preview"
        )

    def apply_styling(self, cfg: SubtitleWindowConfig) -> None:
        # Three-way visual state. Active rows get the configured
        # colours at full intensity; history rows are softened so
        # they read as "already done"; preview rows use the dedicated
        # preview_color + italic so the user can tell at a glance
        # that what they're seeing is still being recognised.
        en_italic = False
        if self._state == "preview":
            en_hex = cfg.preview_color
            cn_hex = cfg.translation_color  # unused while preview hides CN
            cn_weight = 600
            en_italic = True
        elif self._state == "active":
            en_hex = cfg.asr_color
            cn_hex = cfg.translation_color
            cn_weight = 600
        else:  # "history"
            en_hex = _dim_color(cfg.asr_color, 0.6)
            cn_hex = _dim_color(cfg.translation_color, 0.75)
            cn_weight = 500

        # Text alpha is controlled by ``cfg.text_opacity`` and is
        # *independent* of the rounded-plate's background opacity --
        # we paint rgba() into the stylesheet so Qt blends the text
        # against the plate at the user-configured strength.
        en_rgba = _rgba_from_hex(en_hex, cfg.text_opacity)
        cn_rgba = _rgba_from_hex(cn_hex, cfg.text_opacity)

        self._en_label.setStyleSheet(
            f"color: {en_rgba}; "
            f"background: transparent; "
            f"font-family: '{cfg.font_family}'; "
            f"font-size: {cfg.asr_font_size_pt}pt;"
            + (" font-style: italic;" if en_italic else "")
        )
        self._cn_label.setStyleSheet(
            f"color: {cn_rgba}; "
            f"background: transparent; "
            f"font-family: '{cfg.font_family}'; "
            f"font-size: {cfg.translation_font_size_pt}pt; "
            f"font-weight: {cn_weight};"
        )


def _make_wrapping_label(text: str = "") -> QLabel:
    """Create a QLabel that actually wraps + grows in height correctly.

    QLabel + ``setWordWrap(True)`` alone is famously broken inside
    nested layouts: the label keeps reporting a single-line height,
    which causes the row below it (e.g. the translation) to overlap.
    The fix is a combination of:

    * horizontal size policy = MinimumExpanding -- so the label
      claims as much width as the parent layout offers but never
      pushes the window wider when text shrinks;
    * vertical size policy = MinimumExpanding with heightForWidth
      enabled -- so the layout asks the label "given this width,
      how tall do you need?" and reserves that height for it.
    * ``setMinimumWidth(0)`` so the label doesn't refuse to shrink
      when the user drags the window narrower.

    Without this, dragging the right edge inwards keeps the labels at
    their original width and the window won't reflow.
    """
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
    lbl.setMinimumWidth(0)
    lbl.setTextFormat(Qt.TextFormat.PlainText)
    # Horizontal: MinimumExpanding so the label claims the full row
    # width and re-wraps when the user drags the overlay narrower.
    # Vertical: *Preferred* (not MinimumExpanding). The label asks for
    # exactly heightForWidth(width) and won't try to absorb leftover
    # vertical space -- crucial so the inter-line gap inside a wrapped
    # paragraph stays at the font's natural leading rather than being
    # padded out when the overlay window is taller than the content.
    policy = QSizePolicy(
        QSizePolicy.Policy.MinimumExpanding,
        QSizePolicy.Policy.Preferred,
    )
    policy.setHeightForWidth(True)
    lbl.setSizePolicy(policy)
    return lbl


def _dim_color(hex_rgb: str, factor: float) -> str:
    """Multiply RGB channels by ``factor`` (0..1) and return ``#rrggbb``.

    Falls back to the input string if parsing fails, so a stray colour
    name in the config doesn't crash the overlay."""
    try:
        c = QColor(hex_rgb)
        if not c.isValid():
            return hex_rgb
        r = max(0, min(255, int(c.red() * factor)))
        g = max(0, min(255, int(c.green() * factor)))
        b = max(0, min(255, int(c.blue() * factor)))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_rgb


def _rgba_from_hex(hex_rgb: str, alpha: float) -> str:
    """Convert ``#rrggbb`` plus an alpha in ``[0, 1]`` to a Qt
    stylesheet-compatible ``rgba(r, g, b, a)`` string.

    Used to apply ``SubtitleWindowConfig.text_opacity`` to every text
    colour at render time without mutating the persisted hex values --
    so the user can toggle text alpha back to 1.0 and recover the
    original colours exactly.
    """
    try:
        c = QColor(hex_rgb)
        if not c.isValid():
            # Fall back: Qt will accept the original string and just
            # ignore opacity. Better than crashing the overlay.
            return hex_rgb
        a = max(0.0, min(1.0, float(alpha)))
        return f"rgba({c.red()}, {c.green()}, {c.blue()}, {a:.3f})"
    except Exception:
        return hex_rgb


class _RowsViewport(QWidget):
    """Clipped scroll-region that holds all subtitle rows.

    The viewport itself is a fixed-rect window; inside it lives a
    single ``_inner`` QWidget whose vertical position is moved up by
    ``scrollOffset`` pixels. Because we move ONE widget by ONE
    integer, every visible row shifts in lockstep -- the relative
    distance between rows is mathematically preserved.

    Scrolling rule ("stepped slide")
    --------------------------------
    The user asked for a very specific scroll model:

    * While content is shorter than the viewport: NO scroll at all.
      Rows just stack from the top.
    * The moment a new row pushes the bottom of the stack down to (or
      below) the viewport's bottom edge, the whole stack slides up
      ONCE by ``viewport_height / 3`` pixels. This leaves a 1/3-tall
      blank stripe at the bottom that the next few rows can fill.
    * When the bottom of the stack reaches the viewport bottom again,
      another 1/3 slide happens. Repeat.

    So ``target_offset()`` is *stepped*, not continuous: between two
    1/3-VP threshold crossings the target is constant and no animation
    fires. Per-token text growth that doesn't cross a threshold does
    not cause any motion at all -- exactly the behaviour the user
    requested ("文本一次性向上滑动").

    The step count is held as instance state (``_target_offset``)
    rather than recomputed from inner height each time, because
    history rows can be evicted from the top of the stack -- in which
    case we want the visible rows to stay pixel-locked, not to slide
    DOWN as if a slide had been undone.
    """

    def __init__(self, row_spacing: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Important: clip child painting to our rect so rows that have
        # scrolled above the viewport top edge don't leak into the
        # window's top padding area.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.MinimumExpanding,
        )

        self._inner = QWidget(self)
        self._inner.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # The inner widget needs to be wider than 0 to be useful; we
        # resize it to match the viewport width on every resizeEvent.
        self._inner.setGeometry(0, 0, 1, 1)
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(row_spacing)
        self._inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Don't let the inner layout propagate its sizeHint back up
        # to the outer SubtitleWindow -- the viewport's geometry is
        # managed manually in _sync_inner_geometry, and we don't want
        # a single wide row to force the parent window to grow.
        self._inner_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        # Current scroll position (what the inner widget is actually
        # painted at) and the *target* that the slide animation is
        # heading toward. They're equal at rest; only differ during
        # an in-flight QPropertyAnimation on ``scrollOffset``.
        self._offset = 0
        self._target_offset = 0

    # --- Layout / inner ----------------------------------------------

    def inner_layout(self) -> QVBoxLayout:
        return self._inner_layout

    def set_row_spacing(self, spacing: int) -> None:
        self._inner_layout.setSpacing(max(0, int(spacing)))
        self._sync_inner_geometry()

    def inner_natural_height(self) -> int:
        # heightForWidth on the inner layout gives us the wrapped
        # height *at the current viewport width* -- which is what we
        # need to keep wrapping consistent with what the user sees.
        w = max(1, self.width())
        h = self._inner_layout.heightForWidth(w)
        if h <= 0:
            h = self._inner_layout.sizeHint().height()
        return max(0, h)

    def target_offset(self) -> int:
        """Return the stepped target offset.

        Behaviour:

        * Content fits (``inner_h < vh``) -- target = 0, rows
          top-align inside the viewport. No scroll.
        * Content reaches the viewport bottom for the first time, or
          continues growing until the bottom edge touches again --
          target jumps so that exactly ``vh / 3`` blank pixels are
          left at the bottom (i.e. content bottom lands at ``2 * vh / 3``).
        * Between two triggers the target stays constant: as new
          content arrives the bottom-of-content edge drifts back down
          toward the viewport bottom, but no animation fires until
          the bottom actually touches again.

        Shrinkage (history row evicted from the top) is handled
        out-of-band by :meth:`shrink_target` -- we never *decrease*
        the target here, except by resetting to 0 when all content
        suddenly fits in the viewport again.
        """
        vh = self.height()
        inner_h = self.inner_natural_height()

        if inner_h < vh:
            # Content strictly fits with room to spare -- top-align.
            # (Using ``<`` rather than ``<=`` so the moment the bottom
            # of inner content first touches the viewport bottom edge
            # we slide ONCE -- which is exactly the "just touched the
            # bottom, time to scroll up by 1/3" behaviour the user
            # asked for.)
            self._target_offset = 0
            return 0

        # bottom_in_vp == inner_h - target. Triggers when that hits vh
        # (i.e. content bottom reached the viewport bottom). Once
        # triggered, we want bottom_in_vp = 2 * vh / 3 so the user
        # sees a 1/3-vh blank below the latest line.
        bottom_target = (2 * vh) // 3
        if inner_h - self._target_offset >= vh:
            self._target_offset = inner_h - bottom_target
        # Safety clamp: never scroll past the end of content (would
        # leave the viewport entirely empty above the bottom edge).
        self._target_offset = max(0, min(self._target_offset, inner_h - 1))
        return self._target_offset

    def shrink_target(self, delta_px: int) -> None:
        """Account for content being removed from the *top* of the
        inner stack.

        After an oldest-row eviction the inner height shrinks by
        ``row_h + spacing``; we shrink the target by the same amount
        so the remaining visible rows stay pixel-locked. Without this
        the next ``target_offset()`` call would re-derive a smaller
        target from the smaller inner height and animate the rows
        downward, which the user would read as a glitch.
        """
        d = max(0, int(delta_px))
        self._target_offset = max(0, self._target_offset - d)

    def reset_target(self) -> None:
        """Force the target back to zero. Called on window resize so
        the new viewport size starts a fresh stepping cycle (the step
        is ``vh / 3``, so it scales with the window)."""
        self._target_offset = 0

    # --- Animated property ------------------------------------------

    def _get_scroll_offset(self) -> int:
        return self._offset

    def _set_scroll_offset(self, v: int) -> None:
        new_offset = max(0, int(v))
        if new_offset == self._offset:
            return
        self._offset = new_offset
        self._sync_inner_geometry()

    scrollOffset = Property(int, _get_scroll_offset, _set_scroll_offset)

    # --- Geometry ----------------------------------------------------

    def _sync_inner_geometry(self) -> None:
        nh = max(self.inner_natural_height(), self.height())
        self._inner.setGeometry(0, -self._offset, self.width(), nh)

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        # The step size depends on viewport height; resizing
        # invalidates the existing step counter. Reset and let the
        # next target_offset() call rebuild from scratch.
        self.reset_target()
        self._sync_inner_geometry()


class SubtitleWindow(QWidget):
    """Floating subtitle overlay. One per application."""

    settings_changed = Signal(SubtitleWindowConfig)

    def __init__(self, cfg: SubtitleWindowConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._translate_mode = True  # set by set_mode_label

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
        outer.setContentsMargins(18, 8, 18, 10)
        outer.setSpacing(2)
        # CRITICAL: don't let the layout system propagate its sizeHint
        # back up to us. Without this, a single long un-wrappable word
        # (a URL, a Japanese kanji compound, etc.) makes a child label
        # report a minimumSizeHint wider than our current width, Qt
        # honours it via the default SetDefaultConstraint, and the
        # overlay window mysteriously grows on its own. With
        # SetNoConstraint, our geometry is whatever the user dragged
        # to and the labels just clip / wrap whatever fits.
        outer.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        # Belt-and-suspenders: also clamp our own min/max so any code
        # path that bypasses the layout still can't auto-resize us.
        self.setMinimumSize(200, 60)

        # Scrolling viewport. Rows live inside ``_rows_viewport`` and
        # are addressed via ``_rows_layout`` (compatibility alias for
        # the inner layout). When a new sentence arrives we don't grow
        # any single row's height -- we animate ONE number, the
        # viewport's scrollOffset, so every visible row shifts upward
        # by the same amount at the same easing curve. The live
        # preview also lives in here (as the bottom-most entry while
        # Vosk is still streaming partials), so it scrolls and slides
        # exactly like history rows do.
        self._rows_viewport = _RowsViewport(
            row_spacing=cfg.row_spacing_px, parent=self
        )
        self._rows_layout = self._rows_viewport.inner_layout()
        outer.addWidget(self._rows_viewport, 1)

        # Bottom-right size grip so the user can resize without a frame.
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(16, 16)
        grip_row.addWidget(
            self._grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom
        )
        outer.addLayout(grip_row)

        self._set_click_through(cfg.click_through)
        self._apply_geometry()

        # Drag state.
        self._drag_origin: Optional[QPoint] = None

        # Entries indexed by item_id; maintained in insertion order so
        # we always know which one is "oldest" for eviction.
        self._entries: dict[str, _SubtitleEntry] = {}
        self._entry_order: list[str] = []
        self._active_id: Optional[str] = None

        # Single long-lived "slide" animation; reused across new-row
        # arrivals. Updating its endpoint mid-flight is cheaper and
        # smoother than tearing down + restarting on every event.
        self._slide_anim: Optional[QPropertyAnimation] = None
        self._slide_curve = QEasingCurve.Type.InOutCubic

    # ------------------------------------------------------------------ Geometry

    def _apply_geometry(self) -> None:
        c = self._cfg
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(c.width, c.height)
            return
        avail: QRect = screen.availableGeometry()
        x = c.x if c.x is not None else avail.x() + (avail.width() - c.width) // 2
        y = c.y if c.y is not None else avail.y() + avail.height() - c.height - 80
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

    # ------------------------------------------------------------------ Entry management

    def _max_entries(self) -> int:
        return max(1, min(_HARD_MAX_ENTRIES, self._cfg.max_visible_entries))

    def _get_or_create_entry(
        self,
        item_id: str,
        state: EntryState = "preview",
    ) -> _SubtitleEntry:
        existing = self._entries.get(item_id)
        if existing is not None:
            return existing

        # A new item_id means "new utterance" -> the previous active
        # row needs to step aside.
        #
        # Two cases:
        #
        # * Previous was "active" (a finalised sentence currently
        #   showing translation deltas): demote to "history" so it
        #   keeps its text and stays visible above the new row.
        # * Previous was "preview" (Vosk was still streaming partials
        #   when the cloud ASR jumped in with a polished delta):
        #   REMOVE it. Leaving a "history" row with the Vosk preview
        #   text right above the polished cloud version would look
        #   like two copies of the same sentence. The pipeline
        #   normally sends a LocalPreviewReset before the cloud delta
        #   precisely so this branch fires, but we also drop the row
        #   here as a safety net in case ordering ever differs.
        if self._active_id is not None and self._active_id != item_id:
            prev = self._entries.get(self._active_id)
            if prev is not None:
                if prev.state() == "preview":
                    self._entries.pop(self._active_id, None)
                    if self._active_id in self._entry_order:
                        self._entry_order.remove(self._active_id)
                    removed_h = max(0, prev.height())
                    spacing = (
                        self._rows_layout.spacing()
                        if len(self._entry_order) > 0
                        else 0
                    )
                    self._rows_layout.removeWidget(prev)
                    prev.setParent(None)
                    prev.deleteLater()
                    # Keep visible rows pixel-locked: shrink both the
                    # current offset and the target offset by exactly
                    # the height we just removed.
                    self._rows_viewport.shrink_target(removed_h + spacing)
                    cur = self._rows_viewport._get_scroll_offset()
                    self._rows_viewport._set_scroll_offset(
                        max(0, cur - removed_h - spacing)
                    )
                elif prev.state() != "history":
                    prev.set_state("history", self._cfg)

        entry = _SubtitleEntry(item_id, self._translate_mode, self._cfg, self, state=state)
        self._entries[item_id] = entry
        self._entry_order.append(item_id)
        self._active_id = item_id
        self._rows_layout.addWidget(entry)

        # DON'T prune to ``max_visible_entries`` here. If we did, the
        # oldest row would be freed *before* the slide ran -- the
        # inner-stack height would be unchanged, ``target_offset()``
        # would equal ``current``, and the slide animation would
        # collapse into a no-op. Instead we let the inner stack
        # temporarily grow by one (or more, when sentences fire in
        # quick succession), drive the slide on the bigger target,
        # and free the now-off-screen rows in ``_on_slide_finished``.
        #
        # Only the hard cap (``_HARD_MAX_ENTRIES``) is enforced here,
        # as a safety net against pathological "1000 sentences/sec"
        # input that could otherwise leak widgets.
        self._enforce_hard_cap()

        # Defer the slide to next event-loop tick: by then Qt has
        # given the new row a width and we can correctly compute its
        # wrapped natural height.
        QTimer.singleShot(0, lambda: self._pin_to_bottom(animate=True))
        return entry

    def _enforce_hard_cap(self) -> None:
        """Drop any rows beyond ``_HARD_MAX_ENTRIES`` immediately.

        Used during ``_get_or_create_entry`` to bound memory in the
        worst case while still keeping enough rows around to drive
        the slide animation visibly.
        """
        while len(self._entry_order) > _HARD_MAX_ENTRIES:
            self._drop_oldest_row()

    def _enforce_max_entries(self) -> None:
        """Trim down to the user-configured ``max_visible_entries``.

        Called from ``_on_slide_finished`` (the natural moment to
        garbage-collect rows that have just slid off the top of the
        viewport) and from settings dialogs that change the cap.
        """
        limit = self._max_entries()
        while len(self._entry_order) > limit:
            self._drop_oldest_row()

    def _drop_oldest_row(self) -> None:
        """Remove the top entry and shift the scroll offset down by
        exactly the same amount, so the rows that remain on screen
        stay pixel-locked in place (no visual jump on eviction).

        Crucially we also tell the viewport to shrink its *target*
        offset by the same delta -- otherwise the next
        ``target_offset()`` call would re-derive a fresh stepped
        target from the smaller inner height and the slide animation
        would jump backward.
        """
        oldest = self._entry_order.pop(0)
        widget = self._entries.pop(oldest, None)
        if widget is None:
            return
        removed_h = max(0, widget.height())
        spacing = self._rows_layout.spacing() if len(self._entry_order) > 0 else 0
        self._rows_layout.removeWidget(widget)
        widget.setParent(None)
        widget.deleteLater()
        delta = removed_h + spacing
        self._rows_viewport.shrink_target(delta)
        cur = self._rows_viewport._get_scroll_offset()
        self._rows_viewport._set_scroll_offset(max(0, cur - delta))

    # ------------------------------------------------------------------ Slide animation

    def _pin_to_bottom(self, animate: bool) -> None:
        """Move the viewport so the latest row's bottom is at the
        viewport's bottom edge.

        ``animate=True`` -> smooth ease-in-out cubic slide (used for
        brand-new sentence arrivals; everything visible shifts up
        together).

        ``animate=False`` -> snap immediately, but if a slide
        animation is *already* running we just retarget its end value
        so the moving curve doesn't jolt. This is what we use for
        per-token text growth (translation streaming): no extra
        animation is spawned, but the in-flight slide keeps heading
        to the correct (now-slightly-larger) destination.
        """
        target = self._rows_viewport.target_offset()
        current = self._rows_viewport._get_scroll_offset()
        if target == current and (
            self._slide_anim is None
            or self._slide_anim.state() != QAbstractAnimation.State.Running
        ):
            return

        duration = max(0, int(self._cfg.slide_animation_ms))

        running = (
            self._slide_anim is not None
            and self._slide_anim.state() == QAbstractAnimation.State.Running
        )

        if running:
            # Smoothly retarget the in-flight slide.
            self._slide_anim.setEndValue(target)
            return

        if not animate or duration == 0:
            self._rows_viewport._set_scroll_offset(target)
            # Slide skipped -> _on_slide_finished won't fire, but we
            # still need to garbage-collect off-screen rows so the
            # entry count doesn't drift above ``max_visible_entries``.
            self._enforce_max_entries()
            self._rows_viewport._set_scroll_offset(self._rows_viewport.target_offset())
            return

        anim = QPropertyAnimation(self._rows_viewport, b"scrollOffset", self)
        anim.setDuration(duration)
        anim.setEasingCurve(self._slide_curve)
        anim.setStartValue(current)
        anim.setEndValue(target)
        anim.finished.connect(self._on_slide_finished)
        self._slide_anim = anim
        anim.start()

    def _on_slide_finished(self) -> None:
        # The slide is what carried the new row into view and pushed
        # the oldest off the top edge. NOW is when we actually free
        # those off-screen rows -- doing it any earlier would have
        # squashed the slide into a no-op (see comment in
        # ``_get_or_create_entry``). ``_drop_oldest_row`` also subtracts
        # the removed row's height from the scroll offset, so the rows
        # that remain visible stay perfectly still as we trim.
        self._enforce_max_entries()
        # Snap to the latest target in case content grew while the
        # slide was in flight and we want pixel-perfect alignment.
        self._rows_viewport._set_scroll_offset(self._rows_viewport.target_offset())
        self._slide_anim = None

    # ------------------------------------------------------------------ Slots

    def on_transcript_delta(self, item_id: str, text: str) -> None:
        # The translator's polishing phase streams TranscriptDelta
        # events against an existing "active" entry (the same row that
        # Vosk just finalised). We just keep mutating the source label
        # -- the user sees the raw ASR text gradually morph into the
        # polished one with punctuation.
        entry = self._get_or_create_entry(item_id, state="active")
        entry.update_en(text)
        self._schedule_repin()

    def on_transcript_final(self, item_id: str, text: str) -> None:
        # Vosk locked in this sentence (or the translator finished
        # polishing it). Either way, promote the row to "active" so it
        # gets the regular ASR styling and -- crucially -- the
        # translation row becomes visible and ready to accept deltas.
        entry = self._get_or_create_entry(item_id, state="active")
        if entry.state() != "active":
            entry.set_state("active", self._cfg)
        entry.update_en(text)
        self._schedule_repin()

    def on_translation_delta(self, item_id: str, text_so_far: str) -> None:
        if not self._translate_mode:
            return
        entry = self._get_or_create_entry(item_id, state="active")
        entry.update_cn(text_so_far)
        self._schedule_repin()

    def on_translation_final(self, item_id: str, text: str) -> None:
        if not self._translate_mode:
            return
        entry = self._get_or_create_entry(item_id, state="active")
        entry.update_cn(text)
        self._schedule_repin()

    def _schedule_repin(self) -> None:
        """Defer a re-pin to the next event-loop tick (with animation).

        Text updates change the entry's wrapped height, which moves
        the bottom of the inner stack. With the *stepped* scroll
        model, most per-token updates don't change ``target_offset()``
        at all (the threshold isn't crossed) so ``_pin_to_bottom``
        is a no-op. When a token finally pushes the bottom across a
        ``vh / 3`` boundary, we DO want a smooth one-shot slide --
        hence ``animate=True``. (``_pin_to_bottom`` handles the
        in-flight retargeting itself.)
        """
        QTimer.singleShot(0, lambda: self._pin_to_bottom(animate=True))

    def on_preview_delta(self, item_id: str, text: str) -> None:
        """Slot for local Vosk word-level partials.

        ``text`` is the FULL accumulated preview (committed segments +
        current partial). We materialise the corresponding row as the
        bottom-most entry in preview state and just mutate its text.
        Once ``TranscriptFinal`` arrives with the same ``item_id``,
        the same widget is reused -- only its state flips to
        ``active`` -- so the user sees the live preview seamlessly
        lock in place, no row swap, no flicker.
        """
        if not text:
            return
        entry = self._get_or_create_entry(item_id, state="preview")
        entry.update_en(text)
        self._schedule_repin()

    def on_preview_reset(self, item_id: str = "") -> None:
        """Drop a stale preview row.

        Triggered by ``LocalPreviewReset`` (no current ASR backend
        actually emits this, but the signal is plumbed so future
        backends can cancel a partial without leaving an orphan row
        on screen). When called with an empty ``item_id`` we drop the
        currently active preview entry, if any.
        """
        target_id = item_id or self._active_id
        if not target_id:
            return
        entry = self._entries.pop(target_id, None)
        if entry is None:
            return
        if target_id in self._entry_order:
            self._entry_order.remove(target_id)
        if self._active_id == target_id:
            self._active_id = self._entry_order[-1] if self._entry_order else None
        self._rows_layout.removeWidget(entry)
        entry.setParent(None)
        entry.deleteLater()
        self._schedule_repin()

    def clear(self) -> None:
        # Immediate (non-animated) clear -- called from the right-click
        # "清空字幕" action and when restarting the pipeline. Stop any
        # in-flight slide so it doesn't try to scroll widgets we're
        # about to delete.
        if self._slide_anim is not None:
            self._slide_anim.stop()
            self._slide_anim = None

        for item_id in list(self._entry_order):
            widget = self._entries.pop(item_id, None)
            if widget is not None:
                self._rows_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
        self._entry_order.clear()
        self._active_id = None
        self._rows_viewport._set_scroll_offset(0)

    def set_mode_label(self, mode: str) -> None:
        """``asr_only`` -> hide the translation row of every entry."""
        self._translate_mode = mode == "translate"
        for entry in self._entries.values():
            entry.set_translate_mode(self._translate_mode)

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
            event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.hide()
            event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._persist_geometry()
        # Resizing the window changes both wrap widths (height of each
        # row) and the viewport height itself -- both feed into
        # target_offset(). Snap without animation so the user's drag
        # stays visually locked to the bottom row.
        QTimer.singleShot(0, lambda: self._pin_to_bottom(animate=False))

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802
        menu = QMenu(self)

        opacity_menu = menu.addMenu("背景透明度")
        for label, value in (("80%", 0.80), ("60%", 0.55), ("40%", 0.40), ("20%", 0.20), ("透明", 0.0)):
            act = QAction(label, self)
            act.triggered.connect(lambda _checked=False, v=value: self._set_opacity(v))
            opacity_menu.addAction(act)

        text_op_menu = menu.addMenu("文字透明度")
        for label, value in (("100%", 1.00), ("80%", 0.80), ("60%", 0.60), ("40%", 0.40)):
            act = QAction(label, self)
            act.triggered.connect(lambda _checked=False, v=value: self._set_text_opacity(v))
            text_op_menu.addAction(act)

        font_menu = menu.addMenu("译文字号")
        for pt in (14, 18, 22, 26, 32, 40):
            act = QAction(f"{pt}pt", self)
            act.triggered.connect(lambda _checked=False, p=pt: self._set_translation_font_size(p))
            font_menu.addAction(act)

        rows_menu = menu.addMenu("显示行数")
        for n in (2, 3, 4, 5, 6, 8):
            act = QAction(f"{n} 行", self)
            act.setCheckable(True)
            act.setChecked(self._cfg.max_visible_entries == n)
            act.triggered.connect(lambda _checked=False, k=n: self._set_max_visible(k))
            rows_menu.addAction(act)

        toggle_top = QAction("窗口置顶", self)
        toggle_top.setCheckable(True)
        toggle_top.setChecked(self._cfg.always_on_top)
        toggle_top.toggled.connect(self._set_always_on_top)
        menu.addAction(toggle_top)

        toggle_ct = QAction("鼠标穿透 (点击不影响)", self)
        toggle_ct.setCheckable(True)
        toggle_ct.setChecked(self._cfg.click_through)
        toggle_ct.toggled.connect(self._set_click_through_persisted)
        menu.addAction(toggle_ct)

        menu.addSeparator()
        act_clear = QAction("清空字幕", self)
        act_clear.triggered.connect(self.clear)
        menu.addAction(act_clear)

        act_hide = QAction("隐藏字幕", self)
        act_hide.triggered.connect(self.hide)
        menu.addAction(act_hide)

        menu.exec(event.globalPos())

    # ------------------------------------------------------------------ Helpers

    def _restyle_all_entries(self) -> None:
        for entry in self._entries.values():
            entry.apply_styling(self._cfg)

    def _set_opacity(self, value: float) -> None:
        self._cfg = self._cfg.model_copy(update={"background_opacity": value})
        self.update()
        self.settings_changed.emit(self._cfg)

    def _set_text_opacity(self, value: float) -> None:
        # Text alpha lives in every label's stylesheet via
        # ``_rgba_from_hex``, so we need to re-style every existing row
        # (active + history) and the preview line. The background
        # plate is untouched.
        self._cfg = self._cfg.model_copy(update={"text_opacity": value})
        self._restyle_all_entries()
        self.settings_changed.emit(self._cfg)

    def _set_translation_font_size(self, pt: int) -> None:
        self._cfg = self._cfg.model_copy(update={"translation_font_size_pt": pt})
        self._restyle_all_entries()
        self.settings_changed.emit(self._cfg)

    def _set_max_visible(self, n: int) -> None:
        self._cfg = self._cfg.model_copy(update={"max_visible_entries": n})
        self._enforce_max_entries()
        self.settings_changed.emit(self._cfg)

    def _set_always_on_top(self, on: bool) -> None:
        self._cfg = self._cfg.model_copy(update={"always_on_top": on})
        flags = self.windowFlags()
        if on:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()
        self.settings_changed.emit(self._cfg)

    def _set_click_through_persisted(self, on: bool) -> None:
        self._cfg = self._cfg.model_copy(update={"click_through": on})
        self._set_click_through(on)
        self.settings_changed.emit(self._cfg)

    def _persist_geometry(self) -> None:
        g = self.geometry()
        self._cfg = self._cfg.model_copy(
            update={"x": g.x(), "y": g.y(), "width": g.width(), "height": g.height()}
        )
        self.settings_changed.emit(self._cfg)

    def apply_config(self, cfg: SubtitleWindowConfig) -> None:
        """Called by the main window after settings dialog edits."""
        self._cfg = cfg
        self._set_click_through(cfg.click_through)
        self._rows_viewport.set_row_spacing(cfg.row_spacing_px)
        self._restyle_all_entries()
        self._enforce_max_entries()
        self._apply_geometry()
        self.update()
        QTimer.singleShot(0, lambda: self._pin_to_bottom(animate=False))
