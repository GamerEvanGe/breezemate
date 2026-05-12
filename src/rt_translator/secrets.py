"""Persisted API-key store for the GUI.

Keys are written to ``%APPDATA%\\rt-translator\\secrets.json`` (Windows)
or ``~/.config/rt-translator/secrets.json`` elsewhere. The file is NOT
encrypted -- this is a single-user desktop app, so we rely on the OS's
per-user permissions for the AppData directory. If you need stronger
protection (shared machines, screen-recording risk, etc.), use the
``.env`` / environment-variable path instead.

Resolution order when an API key is requested:

    1. Explicit value passed in by the caller
    2. ``secrets.json`` entry (set via the GUI)
    3. Process environment variable (e.g. ``OPENAI_API_KEY`` from ``.env``)

This ordering means the GUI is authoritative when populated, but a CI /
headless install can still drop in keys via the environment.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import sys
from pathlib import Path
from typing import Optional

from .paths import appdata_dir

log = logging.getLogger(__name__)


SECRETS_FILE_NAME = "secrets.json"


class SecretStore:
    """Thin wrapper around a JSON dict of ``{env_var_name: api_key}``."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path: Path = path or (appdata_dir() / SECRETS_FILE_NAME)
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                raw = json.load(f) or {}
            if not isinstance(raw, dict):
                log.warning("secrets.json is not a JSON object, ignoring")
                return
            # Coerce all values to strings; anything else is corrupt.
            self._data = {str(k): str(v) for k, v in raw.items()}
        except Exception as e:
            log.warning("Failed to load %s: %s", self.path, e)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a tempfile then atomically replace, so a crash
            # mid-write does not leave an empty / corrupt secrets file.
            tmp = self.path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False, sort_keys=True)
            os.replace(tmp, self.path)
            # Best-effort restrict perms on POSIX. On Windows the user's
            # AppData dir is already user-private, so this is a no-op.
            if sys.platform != "win32":
                try:
                    self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
                except Exception:
                    pass
        except Exception as e:
            log.error("Failed to save %s: %s", self.path, e)

    # ------------------------------------------------------------------ API

    def get(self, env_name: str) -> str:
        """Return the stored key for ``env_name``, falling back to the
        process environment if the JSON store has no entry."""
        if env_name in self._data and self._data[env_name]:
            return self._data[env_name]
        return os.environ.get(env_name, "").strip()

    def set(self, env_name: str, value: str) -> None:
        """Persist a key. Empty string deletes the entry."""
        value = (value or "").strip()
        if value:
            self._data[env_name] = value
        else:
            self._data.pop(env_name, None)
        self._save()

    def has(self, env_name: str) -> bool:
        return bool(self.get(env_name))

    def all_keys(self) -> dict[str, str]:
        """Return a *copy* of the JSON-stored entries (env vars excluded)."""
        return dict(self._data)


# Module-level singleton: the GUI and the CLI both look here, and there's
# no scenario where you'd want two stores in the same process.
_singleton: Optional[SecretStore] = None


def get_secret_store() -> SecretStore:
    global _singleton
    if _singleton is None:
        _singleton = SecretStore()
    return _singleton
