"""Filesystem locations shared by CLI, GUI, secret store, and config.

Kept dependency-free on purpose (no soundcard / Qt imports here) so the
secrets / config layer can call ``appdata_dir`` without dragging audio
or GUI imports along for the ride.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


# Internal folder name. NOT renamed alongside the user-facing product
# rename ("BreezeMate" / "微伴") so existing users keep their downloaded
# Vosk models, secrets.json, device.json, etc. without manual migration.
_APPDATA_FOLDER = "rt-translator"


def appdata_dir() -> Path:
    """Return a per-user writable config dir for BreezeMate.

    * Windows:  ``%APPDATA%/rt-translator``
    * macOS:    ``~/Library/Application Support/rt-translator`` (falls
                back to ``~/.config`` if XDG override is set)
    * Linux:    ``$XDG_CONFIG_HOME/rt-translator`` or
                ``~/.config/rt-translator``

    Folder name stays ``rt-translator`` for backward-compat with users
    who installed earlier versions. Creates the directory on first call.
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = os.environ.get("XDG_CONFIG_HOME") or str(
            Path.home() / "Library" / "Application Support"
        )
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    target = Path(base) / _APPDATA_FOLDER
    target.mkdir(parents=True, exist_ok=True)
    return target


def asset_path(name: str) -> Optional[Path]:
    """Locate a bundled asset (icon, image, ...) on disk.

    Search order:

    1. ``sys._MEIPASS / assets / <name>`` -- the temp directory
       PyInstaller unpacks bundled data files into at runtime.
    2. ``<repo>/assets/<name>`` -- relative to this module, for source
       checkouts and ``pip install -e .``.
    3. ``<cwd>/assets/<name>`` -- last-ditch fallback.

    Returns ``None`` if no candidate exists; callers should fall back to
    a built-in Qt icon in that case.
    """
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "assets" / name)
    # paths.py lives at src/rt_translator/paths.py; the repo root is
    # two parents up.
    here = Path(__file__).resolve()
    candidates.append(here.parent.parent.parent / "assets" / name)
    candidates.append(Path.cwd() / "assets" / name)
    for p in candidates:
        if p.is_file():
            return p
    return None
