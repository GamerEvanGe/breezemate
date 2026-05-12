"""Rich-based incremental subtitle renderer.

Displays a status line at the top, then a scrolling stack of recent
utterances. Each utterance has an ``item_id`` (assigned by the ASR
provider) which the sink uses to:

* update partial transcript text in place as ``TranscriptDelta`` events arrive,
* swap to bold once the matching ``TranscriptFinal`` arrives,
* stream the translation underneath as ``TranslationDelta`` / ``Final`` events flow.

Older utterances are dropped once the visible row count exceeds
``display.max_rows``.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.text import Text

from .config import DisplayConfig, Mode
from .events import (
    ConnectionStatus,
    PipelineEvent,
    TranscriptDelta,
    TranscriptFinal,
    TranslationDelta,
    TranslationFinal,
)


@dataclass
class UtteranceRow:
    item_id: str
    en_text: str = ""
    en_is_final: bool = False
    zh_text: str = ""
    zh_is_final: bool = False


_STATUS_MARKUP = {
    "connecting": "[bold yellow]● connecting[/]",
    "connected": "[bold green]● connected[/]",
    "reconnecting": "[bold yellow]○ reconnecting[/]",
    "error": "[bold red]● error[/]",
    "disconnected": "[dim]○ disconnected[/]",
}


class ConsoleSink:
    def __init__(
        self,
        mode: Mode,
        display_cfg: DisplayConfig,
        source_label: str,
        target_lang: str,
        console: Optional[Console] = None,
    ) -> None:
        self.mode = mode
        self.cfg = display_cfg
        self.source_label = source_label
        self.target_lang = target_lang
        self.console = console or Console(legacy_windows=False)
        self._rows: OrderedDict[str, UtteranceRow] = OrderedDict()
        self._status_state: str = "connecting"
        self._status_detail: str = ""

    def _get_or_create(self, item_id: str) -> UtteranceRow:
        row = self._rows.get(item_id)
        if row is None:
            row = UtteranceRow(item_id=item_id)
            self._rows[item_id] = row
            while len(self._rows) > self.cfg.max_rows:
                self._rows.popitem(last=False)
        else:
            # Move to end so it stays "fresh" in the window.
            self._rows.move_to_end(item_id)
        return row

    def handle(self, event: PipelineEvent) -> None:
        if isinstance(event, TranscriptDelta):
            row = self._get_or_create(event.item_id)
            row.en_text = event.text
            row.en_is_final = False
        elif isinstance(event, TranscriptFinal):
            row = self._get_or_create(event.item_id)
            row.en_text = event.text
            row.en_is_final = True
        elif isinstance(event, TranslationDelta):
            row = self._get_or_create(event.item_id)
            row.zh_text = event.text_so_far
            row.zh_is_final = False
        elif isinstance(event, TranslationFinal):
            row = self._get_or_create(event.item_id)
            row.zh_text = event.text
            row.zh_is_final = True
        elif isinstance(event, ConnectionStatus):
            self._status_state = event.state
            self._status_detail = event.detail

    def render(self) -> RenderableType:
        items: list[RenderableType] = [self._render_header(), Text("")]
        for row in self._rows.values():
            items.append(self._render_row(row))
        return Group(*items)

    def _render_header(self) -> Text:
        markup = _STATUS_MARKUP.get(self._status_state, self._status_state)
        suffix = f"  [dim]mode={self.mode}  source={self.source_label}"
        if self.mode == "translate":
            suffix += f"  →  {self.target_lang}"
        suffix += "[/]"
        if self._status_detail:
            suffix += f"  [dim red]({self._status_detail})[/dim red]"
        return Text.from_markup(markup + suffix)

    def _render_row(self, row: UtteranceRow) -> RenderableType:
        lines: list[RenderableType] = []
        if row.en_text:
            style = "bold white" if row.en_is_final else "italic grey62"
            lines.append(Text(row.en_text, style=style))
        if self.mode == "translate" and row.zh_text:
            style = "cyan" if row.zh_is_final else "italic grey62"
            lines.append(Text(row.zh_text, style=style))
        if not lines:
            lines.append(Text(""))
        # Trailing blank line between utterances for readability.
        lines.append(Text(""))
        return Group(*lines)

    async def run(self, queue: "asyncio.Queue[PipelineEvent]") -> None:
        """Consume events from the queue forever, updating the Live display."""
        with Live(
            self.render(),
            console=self.console,
            refresh_per_second=self.cfg.refresh_hz,
            transient=False,
        ) as live:
            try:
                while True:
                    event = await queue.get()
                    self.handle(event)
                    live.update(self.render())
            except asyncio.CancelledError:
                live.update(self.render())
                raise
