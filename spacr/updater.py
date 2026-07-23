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
import subprocess
import sys
from dataclasses import dataclass
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


def run_pip_upgrade(pre_release: bool = False) -> int:
    """Shell out to ``pip install --upgrade spacr``.

    :param pre_release: pass ``--pre`` so pip picks up pre-releases
        + post-releases (needed for 4-part ``.postN`` versions).
    :returns: process exit code.
    """
    args = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if pre_release:
        args.append("--pre")
    args.append("spacr")
    LOG.info("running: %s", " ".join(args))
    return subprocess.call(args)
