"""Audio device enumeration + interactive picker.

Supports two sources on the same machine, with the same dependency tree:

* ``loopback`` -- record whatever Windows speakers are currently playing
  (videos, video calls, music). Uses WASAPI loopback under the hood, no
  virtual cable required.
* ``mic`` -- record from a physical microphone, line-in, or USB audio
  interface, to translate sound coming from an external phone / TV / etc.

A first-run interactive picker writes the user's choice to
``%APPDATA%\\rt-translator\\device.json`` so subsequent launches are silent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import soundcard as sc
from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

from .paths import appdata_dir

__all__ = [
    "DeviceInfo",
    "SourceKind",
    "appdata_dir",
    "list_devices",
    "render_table",
    "interactive_select",
    "save_selection",
    "load_selection",
    "find_matching_device",
    "device_still_present",
]


SourceKind = Literal["loopback", "mic"]

DEVICE_FILE_NAME = "device.json"


@dataclass
class DeviceInfo:
    source: SourceKind
    name: str
    id: str
    is_default: bool


def list_devices() -> list[DeviceInfo]:
    """Enumerate loopback recording sources and microphones.

    On soundcard's API:
    * Loopback capture is exposed as a special "microphone" with
      ``isloopback=True``, obtained via
      ``sc.all_microphones(include_loopback=True)``. (``_Speaker`` itself
      has no ``recorder()`` method -- the speaker is for playback only.)
    * Regular physical microphones are the same call without loopback.
    """
    devices: list[DeviceInfo] = []

    try:
        default_speaker_name = sc.default_speaker().name
    except Exception:
        default_speaker_name = None
    try:
        default_mic_id = str(sc.default_microphone().id)
    except Exception:
        default_mic_id = None

    try:
        all_mics = sc.all_microphones(include_loopback=True)
    except Exception:
        all_mics = []

    loopback_seen = False
    for m in all_mics:
        is_loopback = bool(getattr(m, "isloopback", False))
        if is_loopback:
            # Loopback mic name on Windows mirrors the underlying speaker
            # name, so match by name to identify the default-speaker loopback.
            is_default = (
                default_speaker_name is not None
                and m.name == default_speaker_name
                and not loopback_seen
            )
            if is_default:
                loopback_seen = True
            devices.append(
                DeviceInfo(
                    source="loopback",
                    name=m.name,
                    id=str(m.id),
                    is_default=is_default,
                )
            )
        else:
            devices.append(
                DeviceInfo(
                    source="mic",
                    name=m.name,
                    id=str(m.id),
                    is_default=str(m.id) == str(default_mic_id),
                )
            )

    # If no loopback matched the default speaker by name, fall back to
    # marking the first one we found.
    if not any(d.source == "loopback" and d.is_default for d in devices):
        for d in devices:
            if d.source == "loopback":
                d.is_default = True
                break

    return devices


def render_table(devices: list[DeviceInfo]) -> Table:
    """Build a rich table for human consumption."""
    table = Table(
        title="Available audio devices",
        show_lines=False,
        expand=False,
        header_style="bold",
    )
    table.add_column("#", style="bold cyan", justify="right", no_wrap=True)
    table.add_column("Type", justify="center", no_wrap=True)
    table.add_column("Name", overflow="fold")
    table.add_column("Default", justify="center", no_wrap=True)
    for idx, dev in enumerate(devices, start=1):
        type_label = (
            "[bold blue]Loopback[/]" if dev.source == "loopback" else "[bold green]Microphone[/]"
        )
        table.add_row(str(idx), type_label, dev.name, "★" if dev.is_default else "")
    return table


def interactive_select(
    devices: list[DeviceInfo], console: Optional[Console] = None
) -> DeviceInfo:
    """Prompt the user to pick a device by index. Returns the chosen DeviceInfo."""
    console = console or Console(legacy_windows=False)
    if not devices:
        raise RuntimeError(
            "No audio devices were found on this system. "
            "Check that your audio drivers are working."
        )

    console.print(render_table(devices))
    console.print(
        "\n[dim]Loopback   = capture whatever Windows is playing (videos, calls).\n"
        "Microphone = capture a physical mic / line-in / external device.[/dim]\n"
    )

    default_index = next(
        (i for i, d in enumerate(devices, start=1) if d.source == "loopback" and d.is_default),
        1,
    )
    while True:
        choice = IntPrompt.ask(
            "Select a device by number",
            default=default_index,
            console=console,
        )
        if 1 <= choice <= len(devices):
            picked = devices[choice - 1]
            console.print(
                f"[green]Selected:[/] [{picked.source}] {picked.name}"
            )
            return picked
        console.print(f"[red]Invalid choice {choice}. Please pick 1..{len(devices)}.[/red]")


def save_selection(device: DeviceInfo, path: Optional[Path] = None) -> Path:
    target = path or (appdata_dir() / DEVICE_FILE_NAME)
    with target.open("w", encoding="utf-8") as f:
        json.dump(asdict(device), f, indent=2, ensure_ascii=False)
    return target


def load_selection(path: Optional[Path] = None) -> Optional[DeviceInfo]:
    target = path or (appdata_dir() / DEVICE_FILE_NAME)
    if not target.exists():
        return None
    try:
        with target.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return DeviceInfo(**data)
    except Exception:
        return None


def find_matching_device(
    source: SourceKind, name_hint: Optional[str]
) -> Optional[DeviceInfo]:
    """Look up a device by source + (optional) name substring or id.

    If ``name_hint`` is None, the system default for that source is returned.
    Returns None when nothing matches.
    """
    devices = [d for d in list_devices() if d.source == source]
    if not devices:
        return None
    if not name_hint:
        return next((d for d in devices if d.is_default), devices[0])
    needle = name_hint.lower()
    for d in devices:
        if needle in d.name.lower() or needle == d.id.lower():
            return d
    return None


def device_still_present(saved: DeviceInfo) -> bool:
    """Check whether a previously saved device is still plugged in."""
    return any(d.id == saved.id and d.source == saved.source for d in list_devices())
