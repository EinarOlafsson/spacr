"""Safe-mode launcher: the least spaCR that can still change a setting.

When a saved preference is what makes spaCR die on launch there is otherwise
no way in -- the ordinary start reads that preference before it has drawn
anything. ``safespacr`` reads every preference as its default instead, forces
off the parts that are known to be able to take the process down with them,
and opens far enough for the user to change a value and save it.

WRITES ARE NOT SHADOWED. Reading defaults is what makes safe mode start;
writing to the real store is what makes it useful. A safe mode that saved to
a scratch file would leave the broken value in place and the next ordinary
start would die again on it.
"""
from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    """Launch spaCR with preferences read as defaults and extras off.

    :param argv: Optional spaCR command-line arguments. ``None`` reads
        ``sys.argv[1:]``.
    :returns: Application exit code from :func:`spacr.qt.run`.
    """
    from .preferences import enable_safe_mode
    enable_safe_mode()

    os.environ.pop("SPACR_TIMING", None)
    os.environ["SPACR_NO_GL"] = "1"

    print("spaCR safe mode: preferences are being READ as defaults; the "
          "backdrop, setting animations, verbose logging and preloading "
          "are off. Anything you save is written normally.",
          file=sys.stderr)

    argv = list(sys.argv[1:] if argv is None else argv)
    if "--no-setup" not in argv:
        argv = ["--no-setup", *argv]

    from . import run
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
