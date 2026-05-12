"""``breezemate-gui`` entry point (alias: ``rt-translator-gui``).

Builds the QApplication, loads ``%APPDATA%\\rt-translator\\config.yaml``
(or defaults), prompts the user for an API key on first launch, then
hands off to ``MainWindow``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Headless / CLI imports first -- PySide6 itself comes after argparse so
# `--help` works without a display server.
from ..config import AppConfig, default_config_path, load_config, save_config
from ..secrets import get_secret_store


log = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> Path:
    """Mirror the CLI's logging setup but route everything to the log
    file by default (the GUI has no terminal in production use)."""
    from ..paths import appdata_dir

    log_path = appdata_dir() / "breezemate-gui.log"
    # Millisecond precision so the user can eyeball delta-arrival
    # intervals in the log file directly -- crucial for debugging
    # "deltas not streaming in real time" issues.
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    if verbose:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        stream.setLevel(logging.DEBUG)
        root.addHandler(stream)

    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.captureWarnings(True)

    root.info("=== BreezeMate GUI session start (log file: %s) ===", log_path)
    return log_path


def _load_or_default(path: Optional[Path]) -> tuple[AppConfig, Path]:
    """Load config from disk, falling back to defaults. Returns the
    config plus the path we'll write back to on Save."""
    target = path or default_config_path()
    if target.exists():
        try:
            return load_config(target), target
        except Exception:
            log.exception("Failed to load %s; using defaults", target)
    return AppConfig(), target


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="breezemate-gui",
        description="BreezeMate (Wei Ban) - floating real-time subtitle & translation overlay.",
    )
    p.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    # Enable high-DPI scaling before QApplication is constructed.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    # PySide6 import is deferred so `--help` and the logging setup above
    # don't carry the Qt import cost (~150ms cold).
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    # Apply rounded high-DPI policy on Qt < 6.x (no-op on 6.x but keeps
    # behaviour explicit).
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("BreezeMate")
    app.setApplicationDisplayName("BreezeMate · 微伴")
    app.setOrganizationName("BreezeMate")
    app.setQuitOnLastWindowClosed(False)  # tray icon keeps us alive

    # App-wide icon (taskbar, alt-tab thumbnail, child window default).
    # Falls back silently if the bundled PNG can't be found -- that's
    # fine for dev environments where the asset hasn't been generated.
    from ..paths import asset_path

    icon_path = asset_path("breezemate.png") or asset_path("breezemate.ico")
    if icon_path is not None:
        app_icon = QIcon(str(icon_path))
        app.setWindowIcon(app_icon)

    # Load config AFTER the app is up so we can show dialogs if loading
    # produced a recoverable error.
    cfg, cfg_path = _load_or_default(args.config)

    # First-run nudge: if no openai key is stored, pop the settings
    # dialog immediately. The user can still cancel to "just look".
    from .main_window import MainWindow
    from .settings_dialog import SettingsDialog

    window = MainWindow(cfg, config_path=cfg_path)

    # First-run nudge: pop the settings dialog if no Vosk model is
    # downloaded yet (speech recognition won't start without one).
    # API keys are optional -- only needed for the `translate` mode,
    # and even then Ollama / LM Studio work key-less.
    from ..providers.asr import vosk_model

    if not vosk_model.is_model_present(cfg.local_asr.model):
        QMessageBox.information(
            window,
            "首次使用 BreezeMate · 微伴",
            "尚未下载任何 Vosk 语音模型。\n\n"
            "打开下方设置 → 「语音识别」标签页，选择要识别的语言（默认英文），"
            "点击「下载模型」即可（约 40MB，一次性）。\n\n"
            "想用翻译功能，再到「翻译模型」标签页填一个免费的 key——"
            "推荐 智谱 BigModel 的 glm-4-flash，国内可直连且完全免费。",
        )
        dlg = SettingsDialog(cfg, window)
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            cfg = dlg.result_config()
            try:
                save_config(cfg, cfg_path)
            except Exception:
                log.exception("Failed to save initial config")
            window._cfg = cfg  # noqa: SLF001 (apply after dialog)
            window.subtitle_window.apply_config(cfg.subtitle_window)
            window._update_provider_label()  # noqa: SLF001

    window.show()

    app.aboutToQuit.connect(window.shutdown)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
