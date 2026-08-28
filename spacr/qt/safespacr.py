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
    # BEFORE THE FIRST PREFERENCE IS READ, and before Qt is imported: the
    # palette, the backdrop and the preloader all read preferences while the
    # application is being built, so a flag set after that reaches none of
    # them.
    from .preferences import enable_safe_mode
    enable_safe_mode()

    # Not a preference, so `enable_safe_mode` cannot reach it: the timing
    # instrumentation is chosen by the environment, and it patches the import
    # machinery for the life of the process.
    os.environ.pop("SPACR_TIMING", None)
    # A GL context is created before any Python of ours runs on the crashing
    # path, so refusing it has to happen in the environment too.
    os.environ["SPACR_NO_GL"] = "1"

    print("spaCR safe mode: preferences are being READ as defaults; the "
          "backdrop, setting animations, verbose logging and preloading "
          "are off. Anything you save is written normally.",
          file=sys.stderr)

    # AND NO FIRST-RUN SETUP. Reading preferences as defaults means "has
    # this profile been set up" reads as "no", so safe mode greeted a
    # long-standing user with the setup wizard -- in front of the settings
    # they opened it to repair. The flag is the same one `spacr-server`
    # uses.
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--no-setup" not in argv:
        argv = ["--no-setup", *argv]

    from . import run
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
