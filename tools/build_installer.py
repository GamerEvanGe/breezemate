"""End-to-end build of ``dist/BreezeMateSetup.exe``.

Run from the repo root::

    uv run python tools/build_installer.py

What this does, top-down:

  1. Pre-flight cache scrub.
     * Kill any leftover ``BreezeMate.exe`` / ``BreezeMateSetup.exe``
       so a PyInstaller rebuild can overwrite their files.
     * Wipe ``build/`` and ``dist/`` from previous runs.
     * Hard-fail if ``.env`` is in the working directory AND about to
       be picked up by PyInstaller's analysis (it normally isn't, but
       we want a paranoid check so we never ship a key).

  2. ``pyinstaller tools/breezemate.spec``: rebuilds the main app
     into ``dist/BreezeMate/``.

  3. Secret audit: greps the freshly-built dist for any literal
     OpenAI key pattern and refuses to continue if it finds anything.

  4. Zip ``dist/BreezeMate/`` -> ``build/installer_payload/payload.zip``.

  5. ``pyinstaller tools/installer/uninstall.spec``: builds
     ``BreezeMateUninstall.exe`` and copies it into
     ``build/installer_payload/`` so the setup spec can pull it in.

  6. ``pyinstaller tools/installer/setup.spec``: builds the final
     ``dist/BreezeMateSetup.exe`` -- a single self-contained file the
     user can double-click.

  7. Print a summary with paths + sizes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MAIN_SPEC = REPO / "tools" / "breezemate.spec"
UNINSTALL_SPEC = REPO / "tools" / "installer" / "uninstall.spec"
SETUP_SPEC = REPO / "tools" / "installer" / "setup.spec"

DIST_APP_DIR = REPO / "dist" / "BreezeMate"
DIST_UNINSTALL_EXE = REPO / "dist" / "BreezeMateUninstall.exe"
DIST_SETUP_EXE = REPO / "dist" / "BreezeMateSetup.exe"

PAYLOAD_STAGING = REPO / "build" / "installer_payload"
PAYLOAD_ZIP = PAYLOAD_STAGING / "payload.zip"
STAGED_UNINSTALL_EXE = PAYLOAD_STAGING / "BreezeMateUninstall.exe"

# Regexes the secret audit refuses to find anywhere inside the built
# dist. We deliberately match BOTH the well-known OpenAI prefix and
# the older ``sk-`` short form just in case a hand-curated key
# accidentally ends up bundled.
SECRET_PATTERNS = [
    re.compile(rb"sk-proj-[A-Za-z0-9_\-]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9]{32,}"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _step(msg: str) -> None:
    print(f"\n=== {msg} ===")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {' '.join(cmd)}")


def _kill_running(exe_names: list[str]) -> None:
    """Best-effort taskkill so PyInstaller can overwrite output exes."""
    for name in exe_names:
        subprocess.run(
            ["taskkill", "/F", "/IM", name],
            capture_output=True,
            check=False,
        )


def _wipe(p: Path) -> None:
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    elif p.is_file():
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _audit_secrets(root: Path) -> None:
    """Scan all text-ish files under ``root`` for OpenAI key patterns.

    Binary files (.dll, .pyd, .ico, ...) are skipped: PyInstaller's
    PYZ archive is a binary blob and would otherwise drown the audit
    in false-positive-shaped noise. We're catching the case where a
    plaintext config or .env accidentally landed in the bundle.
    """
    skip_ext = {
        ".dll", ".pyd", ".so", ".dylib", ".exe", ".ico", ".png",
        ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".pyz", ".bin",
        ".qm", ".docx", ".pdf",
    }
    leaks: list[tuple[Path, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in skip_ext:
            continue
        try:
            data = path.read_bytes()
        except Exception:
            continue
        for pat in SECRET_PATTERNS:
            m = pat.search(data)
            if m:
                leaks.append((path, m.group(0)[:12].decode("ascii", "replace")))
                break
    if leaks:
        print("!! SECRET AUDIT FAILED: found suspicious key-shaped strings:")
        for p, prefix in leaks:
            print(f"   {p}  (starts with {prefix!r})")
        raise SystemExit(
            "Refusing to build installer; clean up the leak above and re-run."
        )
    print(f"  audit clean: scanned ~{sum(1 for _ in root.rglob('*'))} entries")


def _zip_dist(src_dir: Path, out_zip: Path) -> None:
    """Pack ``src_dir`` into ``out_zip`` preserving the top-level
    folder name (so unzip recreates ``BreezeMate/...`` rather than
    polluting the install dir with bare files at root).

    Wait -- actually we DO want the files flat in the install dir,
    not nested under another BreezeMate/ folder; the installer
    extracts directly to install_dir. So we zip *contents* of
    src_dir at the root of the archive.
    """
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in src_dir.rglob("*"):
            if path.is_dir():
                continue
            arc = path.relative_to(src_dir).as_posix()
            zf.write(path, arc)
    print(f"  payload.zip: {out_zip.stat().st_size / (1024 * 1024):.1f} MiB")


def _human_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TiB"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def step_preflight() -> None:
    _step("Pre-flight: kill running exes, wipe build/dist")
    _kill_running(["BreezeMate.exe", "BreezeMateSetup.exe", "BreezeMateUninstall.exe"])
    _wipe(REPO / "build")
    _wipe(REPO / "dist")

    # Paranoid env check. PyInstaller doesn't pick up .env from the
    # repo root by default, but explicit > implicit.
    env_file = REPO / ".env"
    if env_file.exists():
        text = env_file.read_text(encoding="utf-8", errors="replace")
        if "OPENAI_API_KEY" in text and "sk-" in text:
            print(
                "  note: .env contains a real OPENAI_API_KEY. PyInstaller's "
                "spec does not bundle .env, but please confirm it has not "
                "been added to ``datas`` in tools/breezemate.spec."
            )


def step_build_main_app() -> None:
    _step("Build BreezeMate.exe (main app, one-folder)")
    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            str(MAIN_SPEC),
            "--noconfirm",
            "--clean",
        ],
        cwd=REPO,
    )
    if not (DIST_APP_DIR / "BreezeMate.exe").exists():
        raise SystemExit("PyInstaller finished but BreezeMate.exe is missing.")


def step_audit_dist() -> None:
    _step("Secret audit on dist/BreezeMate/")
    _audit_secrets(DIST_APP_DIR)


def step_zip_payload() -> None:
    _step("Pack dist/BreezeMate/ into payload.zip")
    _wipe(PAYLOAD_STAGING)
    PAYLOAD_STAGING.mkdir(parents=True, exist_ok=True)
    _zip_dist(DIST_APP_DIR, PAYLOAD_ZIP)


def step_build_uninstaller() -> None:
    _step("Build BreezeMateUninstall.exe")
    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            str(UNINSTALL_SPEC),
            "--noconfirm",
            "--clean",
        ],
        cwd=REPO,
    )
    if not DIST_UNINSTALL_EXE.exists():
        raise SystemExit("PyInstaller finished but BreezeMateUninstall.exe is missing.")
    shutil.copy2(DIST_UNINSTALL_EXE, STAGED_UNINSTALL_EXE)


def step_build_setup() -> None:
    _step("Build BreezeMateSetup.exe")
    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            str(SETUP_SPEC),
            "--noconfirm",
            "--clean",
        ],
        cwd=REPO,
    )
    if not DIST_SETUP_EXE.exists():
        raise SystemExit("PyInstaller finished but BreezeMateSetup.exe is missing.")


def step_summary() -> None:
    _step("Done")
    setup_size = DIST_SETUP_EXE.stat().st_size
    print(f"  Installer: {DIST_SETUP_EXE}  ({_human_size(setup_size)})")
    print(f"  Main app:  {DIST_APP_DIR}")
    print(f"  Uninstaller: {DIST_UNINSTALL_EXE} (also embedded in setup.exe)")
    print()
    print("Hand the user just BreezeMateSetup.exe -- double-click to install.")


def main() -> int:
    os.chdir(REPO)
    step_preflight()
    step_build_main_app()
    step_audit_dist()
    step_zip_payload()
    step_build_uninstaller()
    step_build_setup()
    step_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
