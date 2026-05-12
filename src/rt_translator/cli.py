"""Command-line entry point for BreezeMate · 微伴.

Subcommands::

    breezemate                     # alias for `breezemate run`
    breezemate run [flags]         # start the live subtitle pipeline
    breezemate devices             # list audio devices
    breezemate devices --select    # list + interactive pick + save
    breezemate gui                 # launch the Qt GUI

The legacy ``rt-translator`` command is preserved as an alias for
backward compatibility -- both entry points dispatch into ``main()``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console

from .config import AppConfig, AudioConfig, load_config
from .device_picker import (
    DeviceInfo,
    device_still_present,
    find_matching_device,
    interactive_select,
    list_devices,
    load_selection,
    render_table,
    save_selection,
)


log = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    # The log FILE always captures DEBUG so we can post-mortem WS event
    # streams even on a non-verbose run. The terminal only sees INFO+ by
    # default; -v upgrades it to DEBUG.
    stream_level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # rich.live owns stdout/stderr while running; route logs to a file so
    # the full traceback always survives even if the live display
    # overwrites the terminal mid-error.
    from .paths import appdata_dir

    log_path = appdata_dir() / "breezemate.log"
    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(stream_level)

    root = logging.getLogger()
    # Root must be at the lowest level so handlers can decide what to keep.
    root.setLevel(logging.DEBUG)
    # Replace any previous handlers (basicConfig calls only stick once).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Marker so log tails are easy to align with sessions.
    root.info("=== BreezeMate session start (log file: %s) ===", log_path)

    # Quiet down noisy third-party loggers.
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breezemate",
        description=(
            "BreezeMate (Wei Ban) - Realtime offline speech subtitling and "
            "translation. Recognition runs locally via Vosk (15+ languages); "
            "translation can use any OpenAI-compatible chat endpoint."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="cmd")

    # run (default)
    run_p = sub.add_parser("run", help="Start the realtime translator (default).")
    run_p.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    run_p.add_argument(
        "--mode",
        choices=("asr_only", "translate"),
        default=None,
        help="Subtitle mode. Overrides config.",
    )
    run_p.add_argument(
        "--source",
        choices=("loopback", "mic"),
        default=None,
        help="Audio source. Overrides config / saved selection.",
    )
    run_p.add_argument(
        "--device",
        default=None,
        help="Device name substring or id. Overrides config / saved selection.",
    )
    run_p.add_argument(
        "--asr-model",
        default=None,
        help=(
            "Vosk model id override (must already be downloaded). "
            "Examples: vosk-model-small-en-us-0.15, vosk-model-small-cn-0.22. "
            "Use the GUI to download new models."
        ),
    )
    run_p.add_argument(
        "--target-lang",
        default=None,
        help="Target language code for translation (default zh).",
    )

    # devices
    dev_p = sub.add_parser("devices", help="List available audio devices.")
    dev_p.add_argument(
        "--select",
        action="store_true",
        help="Interactively pick a device and save the choice for future runs.",
    )

    # gui
    gui_p = sub.add_parser("gui", help="Launch the graphical control panel (PySide6).")
    gui_p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (defaults to %%APPDATA%%/rt-translator/config.yaml).",
    )

    return parser


def _resolve_device(
    cfg: AppConfig,
    cli_source: Optional[str],
    cli_device: Optional[str],
    console: Console,
    force_interactive: bool = False,
) -> DeviceInfo:
    # Priority 1: CLI flags
    if cli_source:
        match = find_matching_device(cli_source, cli_device)  # type: ignore[arg-type]
        if not match:
            console.print(
                f"[red]No {cli_source} device matched '{cli_device or '(default)'}'.[/red]"
            )
            console.print(render_table(list_devices()))
            sys.exit(2)
        return match

    # Priority 2: config.yaml
    if cfg.audio.source:
        match = find_matching_device(cfg.audio.source, cfg.audio.device_name)
        if match:
            return match
        console.print(
            f"[yellow]Configured device not found "
            f"({cfg.audio.source}: {cfg.audio.device_name}). "
            f"Falling back to interactive selection.[/yellow]"
        )

    # Priority 3: saved selection
    if not force_interactive:
        saved = load_selection()
        if saved and device_still_present(saved):
            return saved

    # Priority 4: interactive
    console.print("[bold]First-run setup: choose an audio source[/bold]\n")
    devices = list_devices()
    picked = interactive_select(devices, console=console)
    path = save_selection(picked)
    console.print(f"[dim]Saved to {path}[/dim]\n")
    return picked


def _apply_run_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.mode:
        cfg.mode = args.mode  # type: ignore[assignment]
    if args.asr_model:
        # Vosk-only path: keep ``asr.model`` and ``local_asr.model``
        # in sync so the main window summary and the actual runtime
        # both pick up the override.
        cfg.asr.model = args.asr_model
        cfg.local_asr = cfg.local_asr.model_copy(update={"model": args.asr_model})
    if args.target_lang:
        cfg.translator.target_lang = args.target_lang
    # Audio source/device stay None on cfg if user wants saved selection to win.
    return cfg


async def _run_with_signals(coro):
    loop = asyncio.get_running_loop()
    main_task = asyncio.create_task(coro, name="rt-main")

    def _request_shutdown() -> None:
        if not main_task.done():
            main_task.cancel()

    handled_signals = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
            handled_signals.append(sig)
        except (NotImplementedError, RuntimeError):
            # Windows ProactorEventLoop does not implement add_signal_handler
            # for SIGTERM; SIGINT is delivered via KeyboardInterrupt below.
            pass

    try:
        await main_task
    except asyncio.CancelledError:
        pass
    finally:
        for sig in handled_signals:
            try:
                loop.remove_signal_handler(sig)
            except Exception:
                pass


def _cmd_devices(args: argparse.Namespace) -> int:
    console = Console(legacy_windows=False)
    devices = list_devices()
    if not devices:
        console.print("[red]No audio devices found.[/red]")
        return 1
    console.print(render_table(devices))
    console.print(
        "\n[dim]Loopback   = capture whatever Windows is playing.\n"
        "Microphone = capture a physical mic / line-in.[/dim]"
    )
    if args.select:
        picked = interactive_select(devices, console=console)
        path = save_selection(picked)
        console.print(f"[green]Saved selection to[/green] {path}")
    return 0


def _suppress_terminal_logging() -> list[logging.Handler]:
    """Detach the stderr StreamHandler(s) so rich.live's display isn't
    scrambled by log lines during the pipeline. File logging keeps running.

    Returns the handlers that were removed so we can restore them on exit.
    """
    root = logging.getLogger()
    removed: list[logging.Handler] = []
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler):
            root.removeHandler(handler)
            removed.append(handler)
    # Also route any future `warnings.warn(...)` through logging so they
    # land in the log file (handled by the surviving FileHandler) instead
    # of being printed directly to stderr.
    logging.captureWarnings(True)
    return removed


def _restore_terminal_logging(handlers: list[logging.Handler]) -> None:
    root = logging.getLogger()
    for h in handlers:
        root.addHandler(h)
    logging.captureWarnings(False)


def _cmd_run(args: argparse.Namespace) -> int:
    console = Console(legacy_windows=False)
    cfg = load_config(args.config) if args.config else AppConfig()
    cfg = _apply_run_overrides(cfg, args)

    device = _resolve_device(
        cfg, cli_source=args.source, cli_device=args.device, console=console
    )

    # Persist the audio source choice into the runtime config (so capture
    # can read it). We deliberately keep the *file* config untouched.
    cfg.audio = AudioConfig(
        source=device.source,
        device_name=device.name,
        chunk_ms=cfg.audio.chunk_ms,
    )

    from .pipeline import run_pipeline  # local import keeps `devices` fast

    # From here on, rich.live takes over the terminal. Mute stderr-bound
    # log handlers so they don't tear the live subtitle region apart;
    # logs continue to go to the file handler the user can `tail -f`.
    suppressed = _suppress_terminal_logging()
    try:
        asyncio.run(_run_with_signals(run_pipeline(cfg, device)))
    except KeyboardInterrupt:
        # Fallback for Windows where Ctrl+C raises KeyboardInterrupt instead.
        pass
    finally:
        _restore_terminal_logging(suppressed)
    return 0


_RUN_DEFAULTS = {
    "config": None,
    "mode": None,
    "source": None,
    "device": None,
    "asr_model": None,
    "target_lang": None,
}


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))

    cmd = args.cmd
    if cmd is None:
        # Bare `breezemate` (no subcommand) -> default to `run` with
        # defaults. argparse hasn't populated the run-subparser
        # attributes, so seed them ourselves before dispatching.
        for k, v in _RUN_DEFAULTS.items():
            if not hasattr(args, k):
                setattr(args, k, v)
        cmd = "run"

    if cmd == "devices":
        return _cmd_devices(args)
    if cmd == "run":
        return _cmd_run(args)
    if cmd == "gui":
        return _cmd_gui(args)
    parser.print_help()
    return 1


def _cmd_gui(args: argparse.Namespace) -> int:
    """Defer PySide6 import until the user actually asks for the GUI so
    the CLI keeps starting fast and stays usable on minimal installs."""
    try:
        from .gui.app import main as gui_main
    except ImportError as e:
        print(
            f"GUI dependencies missing: {e}\n"
            f"Install with: uv pip install 'breezemate[gui]'",
            file=sys.stderr,
        )
        return 2
    gui_argv: list[str] = []
    if args.config:
        gui_argv += ["--config", str(args.config)]
    if getattr(args, "verbose", False):
        gui_argv.append("-v")
    return gui_main(gui_argv)


if __name__ == "__main__":
    sys.exit(main())
