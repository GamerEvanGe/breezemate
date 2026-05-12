"""One-shot helper: strip the near-black bezel baked into the source PNG and
emit a clean RGBA PNG + multi-size .ico for Windows builds.

Run from the repo root:

    uv run python scripts/_regen_icon.py

Idempotent: rewrites assets/breezemate.{png,ico} in place.
"""
from __future__ import annotations

from PIL import Image

PNG = "assets/breezemate.png"
ICO = "assets/breezemate.ico"
THRESH = 25  # max R/G/B for a pixel to count as "bezel black"

src = Image.open(PNG).convert("RGBA")
w, h = src.size
px = src.load()
visited = bytearray(w * h)
stack = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
stripped = 0
while stack:
    x, y = stack.pop()
    if not (0 <= x < w and 0 <= y < h):
        continue
    idx = y * w + x
    if visited[idx]:
        continue
    r, g, b, _a = px[x, y]
    if r < THRESH and g < THRESH and b < THRESH:
        visited[idx] = 1
        px[x, y] = (0, 0, 0, 0)
        stripped += 1
        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

print(f"Stripped {stripped} bezel pixels out of {w * h} ({stripped * 100 / (w * h):.1f}%)")
src.save(PNG)
src.save(
    ICO,
    sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
)
print("Wrote", PNG, "and", ICO)
