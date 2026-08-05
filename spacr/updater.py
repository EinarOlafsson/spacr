"""
Auto-updater — compare local ``spacr`` to PyPI + the nightly branch.

Exposes a small API the Qt GUI's Help → "Check for updates" menu
entry can call. Nothing runs automatically; users always trigger a
check + confirm any upgrade.

The updater talks to two sources:

* **PyPI** — ``https://pypi.org/pypi/spacr/json`` for the latest
  released version.
* **GitHub** — the nightly branch's HEAD commit hash, so nightly
  users see how many commits they're behind.

Both fetches use ``urllib`` from the stdlib to avoid pulling in an
extra HTTP dependency. Timeouts are short (3 s) so a slow / offline
network doesn't block the UI. Errors are absorbed and surfaced as
"couldn't check" — never a crash.

Public API::

    from spacr.updater import check_for_updates, run_pip_upgrade

    info = check_for_updates()   # UpdateInfo
    if info.upgrade_available:
        run_pip_upgrade()
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("spacr.updater")


PYPI_URL = "https://pypi.org/pypi/spacr/json"
GITHUB_NIGHTLY_API = (
    "https://api.github.com/repos/EinarOlafsson/spacr/commits/nightly"
)


@dataclass
class UpdateInfo:
    """Result of a version check."""
    installed_version: str
    latest_release:    Optional[str]
    nightly_sha:       Optional[str]
    error:             Optional[str] = None

    @property
    def upgrade_available(self) -> bool:
        if not self.latest_release:
            return False
        return _lt(self.installed_version, self.latest_release)


def check_for_updates(timeout: float = 3.0) -> UpdateInfo:
    """Query PyPI + GitHub and return an :class:`UpdateInfo`.

    :param timeout: per-request timeout in seconds.
    """
    installed = _installed_version()
    latest = None
    nightly = None
    err = None
    try:
        import urllib.request
        req = urllib.request.Request(
            PYPI_URL, headers={"User-Agent": "spacr-updater"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
        latest = str(payload.get("info", {}).get("version") or "")
    except Exception as e:
        err = f"pypi: {e}"
        LOG.debug("pypi check failed: %s", e)
    try:
        import urllib.request
        req = urllib.request.Request(
            GITHUB_NIGHTLY_API,
            headers={"User-Agent": "spacr-updater",
                     "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
        nightly = str(payload.get("sha") or "")[:7]
    except Exception as e:
        if err is None: err = f"github: {e}"
        LOG.debug("nightly check failed: %s", e)
    return UpdateInfo(
        installed_version=installed,
        latest_release=latest or None,
        nightly_sha=nightly or None,
        error=err,
    )


def _installed_version() -> str:
    """Return the running ``spacr`` version, or ``"unknown"``."""
    try:
        from importlib.metadata import version
        return version("spacr")
    except Exception:
        try:
            from importlib.metadata import version
            return version("spacr-nightly")
        except Exception:
            return "unknown"


def _lt(a: str, b: str) -> bool:
    """Return True iff version ``a`` is strictly less than ``b``.

    Handles both 3-part and 4-part semver-ish strings, treating
    missing parts as 0 (so ``1.4.1 < 1.4.1.1`` and
    ``1.4.1.1 < 1.4.2``).
    """
    try:
        pa = tuple(int(x) for x in a.split(".") if x.isdigit())
        pb = tuple(int(x) for x in b.split(".") if x.isdigit())
    except Exception:
        return False
    # Pad to same length
    n = max(len(pa), len(pb))
    pa = pa + (0,) * (n - len(pa))
    pb = pb + (0,) * (n - len(pb))
    return pa < pb


def find_uv() -> Optional[str]:
    """The ``uv`` the desktop installers bootstrap, if this is such an install.

    The native installers build their environment with ``uv venv``, which does
    **not** seed ``pip``. On those installs ``python -m pip`` fails before it
    starts, so the updater has to use the same tool the installer did. ``uv``
    is bootstrapped one level above the venv::

        <install root>/bootstrap/uv
        <install root>/venv/            <- sys.prefix

    :returns: an executable path, or ``None`` when this is an ordinary
        pip-managed environment.
    """
    candidate = Path(sys.prefix).parent / "bootstrap" / "uv"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    found = shutil.which("uv")
    return found or None


def upgrade_command(pre_release: bool = False) -> list:
    """The command that upgrades this installation, whichever tool owns it."""
    uv = find_uv()
    if uv:
        args = [uv, "pip", "install", "--upgrade",
                "--python", sys.executable]
    else:
        args = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if pre_release:
        args.append("--pre")
    args.append("spacr")
    return args


def run_pip_upgrade(pre_release: bool = False):
    """Upgrade ``spacr`` in place, capturing what the packaging tool said.

    :param pre_release: pass ``--pre`` so pre-releases and ``.postN``
        versions are considered.
    :returns: ``(exit_code, output)``. The output is the combined stdout and
        stderr, and it is the whole point: these installers launch from a
        desktop entry with ``Terminal=false``, so anything written to the
        parent's streams goes nowhere and the GUI used to report a bare exit
        code with an invitation to "check the terminal" that could not be
        accepted.
    """
    args = upgrade_command(pre_release)
    LOG.info("running: %s", " ".join(args))
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError as exc:
        LOG.exception("Upgrade tool is missing")
        return 1, f"Could not run {args[0]}: {exc}"
    except subprocess.TimeoutExpired:
        LOG.error("Upgrade timed out after 30 minutes")
        return 1, "The upgrade timed out after 30 minutes."
    output = "".join(part for part in
                     (completed.stdout or "", completed.stderr or "") if part)
    if completed.returncode != 0:
        LOG.error("Upgrade failed (%s):\n%s", completed.returncode, output)
    return completed.returncode, output
