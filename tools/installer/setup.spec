# PyInstaller spec for ``BreezeMateSetup.exe`` (the user-facing installer).
#
# Bundles three runtime data files via ``datas``:
#   - payload.zip       (the zipped one-folder BreezeMate dist)
#   - BreezeMateUninstall.exe (the uninstaller built from this same spec dir)
#   - breezemate.ico    (window/taskbar icon)
#
# Driven by ``tools/build_installer.py``; do not run directly unless
# build/installer_payload/{payload.zip, BreezeMateUninstall.exe} already
# exists.
from pathlib import Path

ROOT = Path(SPECPATH).parent.parent  # tools/installer -> tools -> repo root
PAYLOAD_DIR = ROOT / "build" / "installer_payload"

datas = [
    (str(PAYLOAD_DIR / "payload.zip"), "."),
    (str(PAYLOAD_DIR / "BreezeMateUninstall.exe"), "."),
    (str(ROOT / "assets" / "breezemate.ico"), "."),
]

a = Analysis(
    [str(ROOT / "tools" / "installer" / "installer_main.py")],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Setup wizards only need stdlib + tkinter; aggressively exclude
    # the heavy stuff so the installer .exe stays slim (~10 MB on top
    # of the payload).
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
    name="BreezeMateSetup",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,             # GUI wizard, no console flash
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "breezemate.ico"),
    version=str(ROOT / "tools" / "installer" / "version_info_setup.txt"),
)
