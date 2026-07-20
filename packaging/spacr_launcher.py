"""
Cross-platform launcher for the spacr GUI.

Every packaging script (Windows / macOS / Debian) wraps this single
entry point so the three installers behave identically at runtime.

Runs `spacr.gui.gui_app()`, which opens the main Tk window.
"""
from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    # Windows / macOS PyInstaller bundles need this for cellpose+torch
    # child processes to bootstrap cleanly.
    multiprocessing.freeze_support()
    try:
        from spacr.gui import gui_app
    except Exception as e:
        print(f"failed to import spacr: {e}", file=sys.stderr)
        return 2
    gui_app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
