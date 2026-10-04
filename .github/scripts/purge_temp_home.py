"""Run 646's opt-in purge in a throwaway home on this host and check it.

The unit tests cover the purge on a sandboxed file system on Linux; this runs
the shipped ``install_cleanup.py purge`` as the uninstallers call it, with
``python -I``, against a temporary home on a real Windows or macOS runner.
A dry run must list spaCR's folders and delete nothing; ``--yes`` must delete
exactly those folders and keep shared caches and the user's own files.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "spacr" / "install_cleanup.py"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def _purge(home: Path, *args: str) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("XDG_", "SPACR_"))}
    env.update(HOME=str(home), USERPROFILE=str(home))
    return subprocess.run([sys.executable, "-I", str(SCRIPT), "purge", *args],
                          env=env, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=600)


def main() -> int:
    home = Path(tempfile.mkdtemp(prefix="spacr-purge-home-"))
    owned = [home / ".spacr" / "runs" / "r1" / "log.txt",
             home / ".cache" / "spacr" / "tiles.bin",
             home / "spacr-demos" / "plate1" / "a.tif"]
    if sys.platform == "darwin":
        owned.append(home / "Library" / "Caches" / "spacr" / "c.bin")
    kept = [home / ".cache" / "huggingface" / "model.bin",
            home / ".cache" / "torch" / "hub.bin",
            home / "Documents" / "experiment.csv"]
    for path in owned + kept:
        _touch(path)

    dry = _purge(home, "--dry-run")
    print(dry.stdout, dry.stderr)
    assert dry.returncode == 0, dry.returncode
    assert ".spacr" in dry.stdout, "the dry run did not list spaCR's home folder"
    assert all(path.exists() for path in owned + kept), "a dry run deleted files"

    refused = _purge(home)
    print(refused.stdout, refused.stderr)
    assert all(path.exists() for path in owned), "a purge without --yes or a terminal deleted files"

    done = _purge(home, "--yes")
    print(done.stdout, done.stderr)
    assert done.returncode == 0, done.returncode
    left = [str(path) for path in owned if path.exists()]
    assert not left, f"the purge left spaCR's files: {left}"
    lost = [str(path) for path in kept if not path.exists()]
    assert not lost, f"the purge deleted files it does not own: {lost}"
    print("purge in a temporary home: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
