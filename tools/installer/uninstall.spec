# PyInstaller spec for ``BreezeMateUninstall.exe``.
#
# Compiled standalone (one-file) and shipped INSIDE the setup .exe so
# we end up with a self-contained uninstaller sitting next to the main
# app after install. Driven by ``tools/build_installer.py``.
from pathlib import Path

ROOT = Path(SPECPATH).parent.parent  # tools/installer -> tools -> repo root

a = Analysis(
    [str(ROOT / "tools" / "installer" / "uninstaller_main.py")],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6",
        "numpy",
        "vosk",
        "soundcard",
        "soxr",
        "openai",
        "websockets",
        "pydantic",
        "pydantic_settings",
        "pytest",
        "PIL",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="BreezeMateUninstall",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "breezemate.ico"),
    version=str(ROOT / "tools" / "installer" / "version_info_uninstall.txt"),
)
