"""
Cross-platform launcher for the spaCR GUI.

Every packaging script (Windows / macOS / Debian) wraps this single
entry point so the three installers behave identically at runtime.

Runs `spacr.qt.run()`, the same function the `spacr` console script
calls, so a frozen bundle and a pip install open the same window.
"""
from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    # Windows / macOS PyInstaller bundles need this for cellpose+torch
    # child processes to bootstrap cleanly.
    multiprocessing.freeze_support()
    try:
        from spacr.qt import run
    except Exception as e:
        # The usual cause is a bundle built without the Qt binding: say
        # that rather than let a frozen app die on a traceback nobody
        # sees, because a double-clicked bundle has no console to read.
        print(f"failed to import the spaCR GUI: {e}", file=sys.stderr)
        return 2
    return int(run() or 0)


if __name__ == "__main__":
    sys.exit(main())
