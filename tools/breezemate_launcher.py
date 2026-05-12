"""PyInstaller entry point for BreezeMate.

PyInstaller resolves a module-attribute entry point (``rt_translator.gui.app:main``)
only when invoked through ``pip install`` console-script shims, not when freezing
a script. So we wrap the real entry point in a tiny script and point
PyInstaller at *this* file.

Keep it thin -- any startup logic belongs in ``rt_translator.gui.app``.
"""

from __future__ import annotations

import sys


def main() -> int:
    # Late import so PyInstaller's analysis sees the dependency tree
    # via the module reference rather than a string.
    from rt_translator.gui.app import main as gui_main

    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
