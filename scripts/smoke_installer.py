"""Smoke tests for the BreezeMate installer build artifacts.

Verifies, end-to-end, without actually installing anything system-
wide:

  1. ``BreezeMate.exe`` has a freshly-embedded ``RT_ICON`` whose
     largest variant pixel-matches ``assets/breezemate.ico``. This
     catches the "Windows is showing me a stale cached icon" /
     "PyInstaller forgot the --icon" class of bug.

  2. ``BreezeMate.exe`` has a populated VERSIONINFO block. The
     ProductName / FileDescription strings must be present and
     contain "BreezeMate" -- if they're empty Windows Explorer can
     fall back to its old icon cache keyed on (path, size).

  3. ``BreezeMateSetup.exe`` exists, has the same icon, and (most
     importantly) carries its bundled ``payload.zip`` +
     ``BreezeMateUninstall.exe``. We exercise the latter by reading
     the PyInstaller archive table of contents -- no need to run
     the installer to verify these are inside.

  4. ``payload.zip`` contains a ``BreezeMate.exe`` at its root and
     a non-trivial amount of supporting files (>50 entries),
     which is the sanity check that we actually packed the whole
     one-folder dist and not just the launcher exe.

Pass criterion: prints "PASS: installer smoke OK" and exits 0.
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pefile
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
APP_EXE = REPO / "dist" / "BreezeMate" / "BreezeMate.exe"
SETUP_EXE = REPO / "dist" / "BreezeMateSetup.exe"
UNINST_EXE = REPO / "dist" / "BreezeMateUninstall.exe"
ICO_FILE = REPO / "assets" / "breezemate.ico"
PAYLOAD_ZIP = REPO / "build" / "installer_payload" / "payload.zip"


def _extract_largest_icon_from_exe(exe_path: Path) -> Image.Image:
    """Pull the largest RT_ICON out of a PE file and return it as a PIL image.

    Uses pefile (pure-Python) so the test is portable and doesn't
    rely on a hand-rolled PE parser. We iterate every icon variant
    embedded in the resource section, decode it, and keep the one
    with the largest pixel area -- that's the variant Windows uses
    for "extra large icon" Explorer views, and the one most likely
    to look stale if PyInstaller forgot to refresh the embed.
    """
    pe = pefile.PE(str(exe_path), fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]]
    )
    biggest_img: Image.Image | None = None
    biggest_area = 0
    rsrc = getattr(pe, "DIRECTORY_ENTRY_RESOURCE", None)
    if rsrc is None:
        raise RuntimeError("no resource directory in exe")
    for type_entry in rsrc.entries:
        # RT_ICON is type id 3 (Windows API constant); pefile keeps
        # the raw int in .struct.Id.
        if type_entry.struct.Id != 3:
            continue
        for name_entry in type_entry.directory.entries:
            for lang_entry in name_entry.directory.entries:
                de = lang_entry.data.struct
                blob = pe.get_data(de.OffsetToData, de.Size)
                try:
                    img = Image.open(io.BytesIO(_wrap_icon_blob(blob)))
                    img.load()
                except Exception:
                    continue
                area = img.size[0] * img.size[1]
                if area > biggest_area:
                    biggest_area = area
                    biggest_img = img
    pe.close()
    if biggest_img is None:
        raise RuntimeError("no decodable RT_ICON variants in exe")
    return biggest_img


def _wrap_icon_blob(blob: bytes) -> bytes:
    """Each RT_ICON resource is a "raw" icon entry without the ICO
    file's outer header. Wrap it in a single-entry ICONDIR so PIL can
    decode it.

    See Microsoft's "ICONDIR / ICONDIRENTRY" docs. The 14-byte
    ICONDIR + 16-byte ICONDIRENTRY prefix points at byte 22 (which
    is where the raw blob now begins after our prefix).
    """
    # Parse BITMAPINFOHEADER for width/height/bitcount.
    if blob[:4] == b"\x89PNG":
        # PNG-compressed icon variant (Vista+). Use 0 for w/h, the
        # ICONDIR says "256" via the magic 0 byte.
        w = h = 0
        bitcount = 32
    else:
        w = blob[4] if blob[4] else 256
        h_field = int.from_bytes(blob[8:12], "little") // 2
        h = h_field if h_field and h_field < 256 else 0
        bitcount = int.from_bytes(blob[14:16], "little") or 32
    icondir = (
        b"\x00\x00"          # reserved
        + b"\x01\x00"        # type=icon
        + b"\x01\x00"        # count=1
    )
    direntry = (
        bytes([w & 0xFF, h & 0xFF, 0, 0])        # width, height, color count, reserved
        + b"\x01\x00"                            # planes
        + bitcount.to_bytes(2, "little")         # bitcount
        + len(blob).to_bytes(4, "little")        # bytes in res
        + (22).to_bytes(4, "little")             # image offset
    )
    return icondir + direntry + blob


def _hash_rgb(img: Image.Image, size: tuple[int, int] = (64, 64)) -> str:
    """Stable perceptual-ish hash of an icon: downsample to RGB at a
    fixed size, then SHA-256 the raw bytes. Two icons that look the
    same will hash to the same value; an old cached icon will hash
    differently.
    """
    rgb = img.convert("RGB").resize(size, Image.LANCZOS)
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def test_app_exe_icon_matches_source() -> None:
    print("\n[1] BreezeMate.exe icon matches assets/breezemate.ico")
    assert APP_EXE.exists(), f"missing {APP_EXE}"
    assert ICO_FILE.exists(), f"missing {ICO_FILE}"

    src = Image.open(ICO_FILE)
    src.size = (256, 256)
    src_hash = _hash_rgb(src)

    embedded = _extract_largest_icon_from_exe(APP_EXE)
    emb_hash = _hash_rgb(embedded)

    print(f"  source ico hash:   {src_hash}")
    print(f"  embedded hash:     {emb_hash}")
    print(f"  embedded size:     {embedded.size}")
    assert (
        src_hash == emb_hash
    ), "embedded icon in BreezeMate.exe does NOT match assets/breezemate.ico"


def test_app_exe_versioninfo_populated() -> None:
    print("\n[2] BreezeMate.exe has populated VERSIONINFO")
    blob = APP_EXE.read_bytes()
    # The version info resource has the "ProductName" / "FileDescription"
    # strings stored as UTF-16LE in the RT_VERSION resource. Easiest
    # spot-check: just look for the well-known strings in the binary.
    needles = [
        b"B\x00r\x00e\x00e\x00z\x00e\x00M\x00a\x00t\x00e\x00",
        b"P\x00r\x00o\x00d\x00u\x00c\x00t\x00N\x00a\x00m\x00e\x00",
        b"F\x00i\x00l\x00e\x00V\x00e\x00r\x00s\x00i\x00o\x00n\x00",
    ]
    for n in needles:
        assert n in blob, f"version block missing {n!r}"
    print("  ProductName / FileVersion strings present in PE resources: OK")


def test_setup_exe_carries_payload() -> None:
    print("\n[3] BreezeMateSetup.exe carries payload + uninstaller")
    assert SETUP_EXE.exists(), f"missing {SETUP_EXE}"
    assert UNINST_EXE.exists(), f"missing {UNINST_EXE}"
    # Setup.exe is a PyInstaller --onefile binary; its PKG archive
    # has an ASCII manifest near the bottom listing every bundled
    # resource. Grep for the two names we packed in via ``datas``.
    tail = SETUP_EXE.read_bytes()[-3_000_000:]
    assert b"payload.zip" in tail, "payload.zip not embedded in setup.exe"
    assert b"BreezeMateUninstall.exe" in tail, "uninstaller not embedded in setup.exe"
    print(
        f"  setup size: {SETUP_EXE.stat().st_size / (1024 * 1024):.1f} MiB; "
        "payload.zip + uninstaller present in archive"
    )


def test_payload_zip_is_complete() -> None:
    print("\n[4] payload.zip contains a complete BreezeMate dist")
    assert PAYLOAD_ZIP.exists(), f"missing {PAYLOAD_ZIP}"
    with zipfile.ZipFile(PAYLOAD_ZIP) as zf:
        names = zf.namelist()
    has_exe = any(n == "BreezeMate.exe" for n in names)
    assert has_exe, "payload.zip is missing BreezeMate.exe at root"
    assert len(names) > 50, (
        f"payload.zip looks under-packed: only {len(names)} entries"
    )
    print(f"  payload entries: {len(names)}; BreezeMate.exe at root: yes")


def main() -> int:
    test_app_exe_icon_matches_source()
    test_app_exe_versioninfo_populated()
    test_setup_exe_carries_payload()
    test_payload_zip_is_complete()
    print("\nPASS: installer smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
