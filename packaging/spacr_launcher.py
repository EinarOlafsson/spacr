"""
Cross-platform launcher for the spaCR GUI.

Every packaging script (Windows / macOS / Debian) wraps this single
entry point so the three installers behave identically at runtime.

Runs `spacr.qt.run()`, the same function the `spacr` console script
calls, so a frozen bundle and a pip install open the same window.
"""
from __future__ import annotations

from importlib import import_module
import json
import multiprocessing
import os
from pathlib import Path
import sys


def _installer_identity(destination: str) -> int:
    """Write loaded-code identity and check GUI imports without starting Qt.

    The caller supplies a new path in private staging. A Windows onedir bundle
    must load spaCR from its own ``_internal`` import root. The receipt proves
    identity and imports, not successful GUI launch or bundle provenance.
    Existing destinations are never replaced; failed writes remove only the
    newly created file if it still has its original identity. A windowed app
    need not have standard output or standard error streams.

    :param destination: nonexistent JSON output path in caller-owned staging.
    :returns: zero after a complete receipt, or two on refusal or failure.
    """
    created = None
    try:
        import spacr
        import spacr.qt
        for name in ("PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"):
            import_module(name)

        if not callable(spacr.qt.run):
            return 2
        executable = Path(sys.executable).resolve()
        frozen = bool(getattr(sys, "frozen", False))
        module = Path(spacr.__file__).resolve()
        import_root = (Path(sys._MEIPASS).resolve() if frozen
                       else module.parent.parent)
        if frozen:
            if import_root != executable.parent / "_internal":
                return 2
            module.relative_to(import_root)
            Path(spacr.qt.__file__).resolve().relative_to(import_root)
        identity = {
            "schema": 1,
            "version": str(spacr.__version__),
            "module": str(module),
            "executable": str(executable),
            "frozen": frozen,
            "bundle_root": str(executable.parent),
            "import_root": str(import_root),
        }
        payload = json.dumps(identity, ensure_ascii=True, allow_nan=False) + "\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(destination, flags, 0o600)
        created = os.fstat(descriptor)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
    except Exception:
        if created is not None:
            try:
                current = os.lstat(destination)
                if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                    os.unlink(destination)
            except OSError:
                pass
        return 2
    return 0


def main() -> int:
    # Windows / macOS PyInstaller bundles need this for cellpose+torch
    # child processes to bootstrap cleanly.
    multiprocessing.freeze_support()
    if sys.argv[1:2] == ["--installer-identity"]:
        if len(sys.argv) != 3:
            return 2
        return _installer_identity(sys.argv[2])
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
