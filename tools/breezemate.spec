# PyInstaller spec for BreezeMate (BreezeMate.exe)
#
# Usage::
#
#     uv run pyinstaller tools/breezemate.spec --noconfirm
#
# Produces ``dist/BreezeMate/BreezeMate.exe`` (one-folder build). Users
# can just double-click that exe, or right-click -> "Send to" / "Pin to
# Start" / "Create shortcut" to put it somewhere convenient.
#
# One-folder is chosen over one-file because:
#   * cold start is ~10x faster (no per-launch unpack of ~150 MB to %TEMP%),
#   * the native DLLs that ``soundcard`` and ``vosk`` ship can be
#     loaded directly without first being extracted, which avoids a
#     handful of fragile loader-path edge cases on Windows,
#   * users still see exactly one .exe in the folder; everything else
#     can be ignored.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs


# Project root resolved from the spec file's location (PyInstaller
# evaluates this file with ``__file__`` set).
ROOT = Path(SPECPATH).parent  # noqa: F821 (SPECPATH is injected by PyInstaller)


# --- Data files bundled into the build ----------------------------------------
# Icon assets, plus everything vosk needs to find its model loader and
# soundcard needs for WASAPI bindings at runtime.
datas = [
    (str(ROOT / "assets" / "breezemate.png"), "assets"),
    (str(ROOT / "assets" / "breezemate.ico"), "assets"),
    (str(ROOT / "config.example.yaml"), "."),
]
datas += collect_data_files("vosk")           # libvosk.dll + tokenizer assets
datas += collect_data_files("soundcard")      # CFFI cdef + WASAPI definitions
datas += collect_data_files("soxr")

# Native DLLs / .pyd modules that PyInstaller's analysis sometimes misses.
binaries = []
binaries += collect_dynamic_libs("vosk")
binaries += collect_dynamic_libs("soxr")

# Hidden imports: dynamically imported submodules PyInstaller can't see
# through static analysis. The providers package is imported by config
# at runtime via string lookups; the asr / llm submodules are picked
# up by ``pipeline.py`` only when the selected profile is enabled.
hiddenimports = []
hiddenimports += collect_submodules("rt_translator.providers")
hiddenimports += collect_submodules("rt_translator.agents")
hiddenimports += collect_submodules("vosk")
hiddenimports += collect_submodules("soundcard")
hiddenimports += [
    "openai",
    "websockets",
    "pydantic",
    "pydantic_settings",
    "yaml",
    "dotenv",
    # PySide6 plugins/styles are usually picked up by the hook, but
    # listing the common ones is cheap insurance.
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    # M3 context-file loaders. Pulled in dynamically by
    # agents.context based on the uploaded file's extension, so
    # PyInstaller's static analysis can't see the import chain.
    "pypdf",
    "docx",
]


a = Analysis(
    [str(ROOT / "tools" / "breezemate_launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Big test/data dependencies that aren't needed at runtime.
        "pytest",
        "PIL.ImageQt",  # PIL is only used at build time for the ICO.
        "tkinter",
        "unittest",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # one-folder build
    name="BreezeMate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                  # GUI: no console window pops up
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "breezemate.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BreezeMate",
)
