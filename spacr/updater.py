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
from typing import Optional, Tuple

LOG = logging.getLogger("spacr.updater")


PYPI_URL = "https://pypi.org/pypi/spacr/json"
GITHUB_NIGHTLY_API = (
    "https://api.github.com/repos/EinarOlafsson/spacr/commits/nightly"
)


@dataclass
class UpdateInfo:
    """Result of a version check.

    :param installed_version: locally installed spaCR version, or ``"unknown"``
        when neither distribution's metadata is readable.
    :param latest_release: latest spaCR version returned by PyPI, or ``None``
        when it is missing or unavailable.
    :param nightly_sha: first seven characters of the nightly branch head
        returned by GitHub, or ``None`` when unavailable.
    :param error: first PyPI or GitHub request failure, prefixed by service
        name, or ``None`` when neither request failed.
    """
    installed_version: str
    latest_release:    Optional[str]
    nightly_sha:       Optional[str]
    error:             Optional[str] = None

    @property
    def upgrade_available(self) -> bool:
        """Return whether PyPI advertises a version newer than this install."""
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
    bootstrap = Path(sys.prefix).parent / "bootstrap"
    # The Windows bootstrap writes uv.exe; POSIX installers write uv. Check
    # both names rather than relying on PATHEXT, because this directory is
    # deliberately private and is not added to PATH.
    for name in ("uv.exe", "uv") if os.name == "nt" else ("uv", "uv.exe"):
        candidate = bootstrap / name
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


def editable_install_location() -> Optional[str]:
    """Return the active spaCR checkout, or ``None`` for a regular install.

    :returns: the absolute path of the working tree, when this interpreter is
        running spaCR out of one.

    Detection first reads the PEP 610 ``direct_url.json`` editable-install
    record. If that metadata is unavailable, it checks whether the imported
    package resides outside ``site-packages`` and inside a directory containing
    ``.git`` or ``pyproject.toml``.
    """
    try:
        import importlib.metadata as md
        import json

        direct = md.distribution("spacr").read_text("direct_url.json")
        if direct:
            record = json.loads(direct)
            if record.get("dir_info", {}).get("editable"):
                url = str(record.get("url", ""))
                if url.startswith("file://"):
                    from urllib.parse import unquote, urlparse

                    return unquote(urlparse(url).path)
                if url:
                    return url
    except Exception:                                            # noqa: BLE001
        pass

    try:
        import spacr as _spacr

        here = os.path.abspath(os.path.dirname(
            os.path.dirname(os.path.abspath(_spacr.__file__))))
    except Exception:                                            # noqa: BLE001
        return None
    for entry in sys.path + [getattr(sys, "prefix", "")]:
        if not entry:
            continue
        marker = os.path.abspath(entry)
        if os.path.basename(marker) in ("site-packages", "dist-packages") \
                and here == marker:
            return None
    # Not under a site-packages: this is a checkout.
    return here if os.path.isdir(os.path.join(here, ".git")) or \
        os.path.isfile(os.path.join(here, "pyproject.toml")) else None


def run_pip_upgrade(pre_release: bool = False):
    """Upgrade ``spacr`` in place, capturing what the packaging tool said.

    :param pre_release: pass ``--pre`` so pre-releases and ``.postN``
        versions are considered.
    :returns: ``(exit_code, output)`` with combined stdout and stderr. Captured
        output remains available to desktop installations launched without a
        terminal.
    """
    # NEVER UPGRADE OVER A DEVELOPMENT CHECKOUT. `pip install --upgrade spacr`
    # uninstalls whatever is there and installs from the index -- including
    # when what is there is an EDITABLE install pointing at a working tree.
    # The developer's source stops being what runs, nothing says so, and every
    # change they make afterwards has no effect they can see. Reported
    # 2026-08-18: an update check ran mid-session and the console showed
    # "Uninstalling spacr-1.5.0.4 ... Successfully installed spacr-1.5.0.4",
    # which is that exact operation.
    #
    # An editable install is a statement that this checkout IS the package, so
    # the upgrade is refused rather than confirmed -- there is no version of
    # "yes" that leaves the checkout in charge, and `git pull` is the upgrade
    # for a checkout.
    editable = editable_install_location()
    if editable:
        return (0, (
            f"spaCR is installed in editable mode from {editable}, so there "
            f"is nothing to upgrade: that folder IS the package, and pip "
            f"would replace it with a release build. Update it with `git "
            f"pull` there instead.\n"))
    return run_install_command(upgrade_command(pre_release))


def run_install_command(args, timeout: float = 1800.0):
    """Run one packaging command, capturing everything it said.

    Install offers use the same capture behavior as :func:`run_pip_upgrade`,
    including when the application was launched without a terminal.

    :param args: the argv to run, from :func:`upgrade_command` or
        :func:`install_requirement_command`.
    :param timeout: seconds before the install is given up on.
    :returns: ``(exit_code, output)`` with stdout and stderr combined.
    """
    args = [str(part) for part in args]
    LOG.info("running: %s", " ".join(args))
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        LOG.exception("Packaging tool is missing")
        return 1, f"Could not run {args[0]}: {exc}"
    except subprocess.TimeoutExpired:
        LOG.error("Command timed out after %s seconds", timeout)
        return 1, (f"The command timed out after "
                   f"{int(timeout // 60)} minutes.")
    output = "".join(part for part in
                     (completed.stdout or "", completed.stderr or "") if part)
    if completed.returncode != 0:
        LOG.error("Command failed (%s):\n%s", completed.returncode, output)
    return completed.returncode, output


# ---------------------------------------------------------------------------
# Offering to install something spaCR needs but does not have
# ---------------------------------------------------------------------------
#
# Instruction 158. A greyed-out option has to be able to say what would ungrey
# it, and -- where that is honestly possible HERE -- to do it. The machinery
# lives in this module because this module already knows how to install into
# spaCR's own environment (:func:`find_uv`, :func:`upgrade_command`) and
# already handles the install where ``python -m pip`` was never seeded.
#
# It is deliberately Qt-free. The GUI half is
# :mod:`spacr.qt.widgets.availability_panel`, and the split is what lets the
# three answers below be tested without a screen.

#: Packages where an install that MOVES one is a change to spaCR's results and
#: not to its tooling. A user pressing Install on an optional accelerator has
#: asked for a faster lasso, not for a numpy major upgrade under an
#: image-analysis stack -- so a plan that touches any of these is refused by
#: default and needs a second, explicit confirmation naming what moves.
PROTECTED_PACKAGES = ("numpy", "torch", "pandas", "scikit-learn")


def canonical_package_name(name) -> str:
    """PEP 503 normalisation, so ``scikit_learn`` and ``Scikit-Learn`` match.

    :param name: any spelling of a distribution name.
    :returns: lower-case with runs of ``-``, ``_`` and ``.`` collapsed to a
        single ``-``.
    """
    import re
    return re.sub(r"[-_.]+", "-", str(name or "").strip()).lower()


def installed_version(name) -> Optional[str]:
    """The version of ``name`` installed here, or ``None`` if it is absent.

    :param name: distribution name to query from installed package metadata.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        # A bundler that ships only what it saw imported can leave this out.
        return None
    try:
        return str(version(str(name)))
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def pip_available() -> bool:
    """Is ``python -m pip`` usable in this interpreter?

    ``uv venv`` does not seed pip, which is the case :func:`find_uv` exists
    for. It matters here because pip is the only one of the two that can
    produce a machine-readable ``--report``.
    """
    import importlib.util
    try:
        return importlib.util.find_spec("pip") is not None
    except (ImportError, ValueError):
        return False


def install_requirement_command(requirement) -> list:
    """Return the command that installs ``requirement`` in this environment.

    The same tool choice :func:`upgrade_command` makes -- ``uv`` when this is
    a desktop install whose venv has no pip, ``python -m pip`` otherwise.

    :param requirement: a pip requirement string, e.g. ``'cuml-cu12'``.
    """
    uv = find_uv()
    if uv and not pip_available():
        return [uv, "pip", "install", "--python", sys.executable,
                str(requirement)]
    return [sys.executable, "-m", "pip", "install", str(requirement)]


def dry_run_command(requirement) -> list:
    """Return the command that previews installation of ``requirement``.

    pip's ``--report -`` writes a JSON document to stdout and installs
    nothing; ``uv pip install --dry-run`` prints ``+ name==version`` lines.
    Both are parsed by :func:`dry_run_install`, because the second is the only
    one available on the desktop installs whose venv has no pip.

    :param requirement: pip requirement string whose installation to preview.
    """
    if pip_available():
        return [sys.executable, "-m", "pip", "install", "--dry-run",
                "--report", "-", str(requirement)]
    uv = find_uv()
    if uv:
        return [uv, "pip", "install", "--dry-run", "--python", sys.executable,
                str(requirement)]
    return [sys.executable, "-m", "pip", "install", "--dry-run",
            "--report", "-", str(requirement)]


@dataclass(frozen=True)
class PackageChange:
    """One line of a dry-run report: what a package is now, and would be.

    :param name: distribution name as reported by the resolver; its spelling
        is retained.
    :param current: installed version, or the version reported as removed by
        uv, or ``None`` when the distribution is absent.
    :param proposed: version the resolver would install, or ``None`` when it
        would remove the distribution.
    """

    name: str
    current: Optional[str]
    proposed: Optional[str]

    @property
    def is_addition(self) -> bool:
        """Nothing is installed under this name today."""
        return self.current is None

    @property
    def is_move(self) -> bool:
        """A version already here would change."""
        return (self.current is not None
                and self.proposed is not None
                and self.current != self.proposed)

    @property
    def protected(self) -> bool:
        """Is this one of :data:`PROTECTED_PACKAGES`?"""
        return canonical_package_name(self.name) in {
            canonical_package_name(p) for p in PROTECTED_PACKAGES}

    def describe(self) -> str:
        """``'numpy 1.26.4 -> 2.2.6'`` or ``'cuml-cu12 26.8.0 (new)'``."""
        if self.is_addition:
            return f"{self.name} {self.proposed or '?'} (new)"
        if self.is_move:
            return f"{self.name} {self.current} -> {self.proposed}"
        return f"{self.name} {self.current} (unchanged)"


@dataclass(frozen=True)
class DryRun:
    """Represent a parsed ``pip install --dry-run`` result.

    ``ok`` is False when the resolver refused, when the tool could not be
    run, or when it returned no machine-readable plan.

    :param requirement: pip requirement string that was resolved.
    :param ok: whether the packaging command succeeded and returned a readable
        machine plan.
    :param changes: parsed resolver entries, including additions, version
        moves, and removals as :class:`PackageChange` records.
    :param error: resolver or launch failure detail when ``ok`` is false,
        otherwise ``None``.
    :param raw: concatenated resolver stdout and stderr retained for
        diagnostics.
    """

    requirement: str
    ok: bool
    changes: Tuple[PackageChange, ...] = ()
    error: Optional[str] = None
    raw: str = ""

    @property
    def additions(self) -> Tuple[PackageChange, ...]:
        """Packages that are not here at all today."""
        return tuple(c for c in self.changes if c.is_addition)

    @property
    def moves(self) -> Tuple[PackageChange, ...]:
        """Packages already installed whose version would change."""
        return tuple(c for c in self.changes if c.is_move)

    @property
    def protected_moves(self) -> Tuple[PackageChange, ...]:
        """The moves that land on :data:`PROTECTED_PACKAGES`."""
        return tuple(c for c in self.moves if c.protected)

    def summary(self) -> str:
        """Return the report shown before installation confirmation."""
        if not self.ok:
            return (f"Could not work out what installing {self.requirement} "
                    f"would change.\n{self.error or ''}".strip())
        moves = self.moves
        additions = self.additions
        lines = [f"Installing {self.requirement} would:"]
        if additions:
            # WITH THEIR VERSIONS. "adds cuml-cu12" and "adds cuml-cu12
            # 26.2.0" are different amounts of evidence, and the second is
            # what lets a reader check the wheel they are about to take.
            lines.append(f"  add {len(additions)} package(s): "
                         + ", ".join(f"{c.name} {c.proposed or '?'}"
                                     for c in additions[:12])
                         + (" ..." if len(additions) > 12 else ""))
        if moves:
            lines.append("  CHANGE the version of "
                         f"{len(moves)} package(s) already installed:")
            lines.extend(f"      {c.describe()}" for c in moves)
        if not additions and not moves:
            lines.append("  change nothing -- it is already satisfied.")
        return "\n".join(lines)


def dry_run_install(requirement, timeout: float = 600.0,
                    runner=None) -> DryRun:
    """Ask the packaging tool what installing ``requirement`` would change.

    This function does not install packages. It resolves and reports proposed
    additions and version changes so they can be reviewed before installation.

    :param requirement: a pip requirement string.
    :param timeout: seconds before the resolver is given up on.
    :param runner: injected for tests; defaults to :func:`subprocess.run`.
    :returns: a :class:`DryRun`.
    """
    args = dry_run_command(requirement)
    run = runner if runner is not None else subprocess.run
    LOG.info("dry run: %s", " ".join(args))
    try:
        completed = run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        return DryRun(str(requirement), False,
                      error=f"Could not run {args[0]}: {exc}")
    except subprocess.TimeoutExpired:
        return DryRun(str(requirement), False,
                      error=f"The resolver did not answer within "
                            f"{int(timeout)} seconds.")
    except Exception as exc:
        return DryRun(str(requirement), False, error=str(exc))

    out = str(getattr(completed, "stdout", "") or "")
    err = str(getattr(completed, "stderr", "") or "")
    raw = out + err
    if int(getattr(completed, "returncode", 1) or 0) != 0:
        return DryRun(str(requirement), False,
                      error=_resolver_error(raw), raw=raw)
    changes = _parse_pip_report(out)
    if changes is None:
        changes = _parse_uv_dry_run(raw)
    if changes is None:
        return DryRun(str(requirement), False,
                      error="The packaging tool produced no readable plan.",
                      raw=raw)
    return DryRun(str(requirement), True, tuple(changes), raw=raw)


def _resolver_error(text: str) -> str:
    """The last few lines of a failed resolve -- where pip puts the reason."""
    lines = [line for line in str(text).splitlines() if line.strip()]
    if not lines:
        return "The packaging tool failed and said nothing."
    return "\n".join(lines[-6:])


def _parse_pip_report(text: str):
    """``pip install --report -`` JSON to changes, or ``None`` if it is not.

    pip prints its own progress alongside the document on some versions, so
    the JSON is found rather than assumed to start at character zero.
    """
    body = str(text or "")
    # `json.loads` CANNOT BE USED HERE. Measured 2026-08-18 against pip 25.3:
    # `--report -` writes the document to stdout with pip's own progress
    # BEFORE it and a "Would install ..." line AFTER it, so a whole-string
    # parse fails on trailing data and the plan is lost -- the failure mode
    # being "the packaging tool produced no readable plan" on a resolve that
    # succeeded. `raw_decode` stops at the end of the first valid document.
    decoder = json.JSONDecoder()
    start = body.find("{")
    while start != -1:
        try:
            payload, _end = decoder.raw_decode(body, start)
        except Exception:
            start = body.find("{", start + 1)
            continue
        if not isinstance(payload, dict) or "install" not in payload:
            start = body.find("{", start + 1)
            continue
        changes = []
        for entry in payload.get("install") or []:
            meta = (entry or {}).get("metadata") or {}
            name = str(meta.get("name") or "").strip()
            if not name:
                continue
            proposed = str(meta.get("version") or "") or None
            changes.append(PackageChange(name, installed_version(name),
                                         proposed))
        return changes
    return None


def _parse_uv_dry_run(text: str):
    """``uv pip install --dry-run`` output to changes, or ``None``.

    uv prints ``+ name==version`` for what it would install and
    ``- name==version`` for what it would remove; a version move shows as
    both, which is why the two are merged by name rather than listed twice.
    """
    import re
    added, removed = {}, {}
    seen = False
    for line in str(text or "").splitlines():
        match = re.match(r"\s*([+-])\s+([A-Za-z0-9._-]+)==([^\s]+)\s*$", line)
        if not match:
            continue
        seen = True
        sign, name, version = match.groups()
        (added if sign == "+" else removed)[canonical_package_name(name)] = (
            name, version)
    if not seen:
        return None
    changes = []
    for key, (name, version) in added.items():
        current = removed.get(key, (None, None))[1] or installed_version(name)
        changes.append(PackageChange(name, current, version))
    for key, (name, version) in removed.items():
        if key not in added:
            changes.append(PackageChange(name, version, None))
    return changes


def install_decision(dry_run: DryRun) -> dict:
    """Whether a plan may proceed, and what a second confirmation must say.

    An install that would move NumPy, PyTorch, pandas, or scikit-learn is
    refused by default and needs a second confirmation naming what moves.

    :param dry_run: the result of :func:`dry_run_install`.
    :returns: ``{allowed, needs_second_confirmation, moves, headline,
        report}``. ``allowed`` is False when the dry run did not answer --
        an install whose consequences are unknown is not offered.
    """
    if not dry_run.ok:
        return {'allowed': False, 'needs_second_confirmation': False,
                'moves': (), 'headline': dry_run.summary(),
                'report': dry_run.summary()}
    protected = dry_run.protected_moves
    if protected:
        named = "; ".join(change.describe() for change in protected)
        return {
            'allowed': True,
            'needs_second_confirmation': True,
            'moves': protected,
            'headline': (
                "REFUSED BY DEFAULT. This install would move packages spaCR's "
                f"own results depend on: {named}. Every measurement made "
                "before and after would be made by different code. Confirm "
                "again only if that is what you meant."),
            'report': dry_run.summary(),
        }
    return {'allowed': True, 'needs_second_confirmation': False, 'moves': (),
            'headline': "", 'report': dry_run.summary()}


#: The four states an installation offer can report: already available,
#: installable here, installable elsewhere, or unavailable.
OFFER_ACTIONS = ("ready", "install", "elsewhere", "impossible")


@dataclass(frozen=True)
class InstallOffer:
    """What pressing **Install** on a greyed-out option should do.

    One shape, two callers -- the regression backend picker
    (:func:`spacr.regression_backends.backend_install_offer`) and the Image
    UMAP's GPU acceleration (:func:`spacr.gpu_reduce.install_offer`) -- so the
    panel that shows it does not have to know which asked.

    :param action: offer state, normally one of :data:`OFFER_ACTIONS`; only
        ``"install"`` with a nonempty requirement can produce a command.
    :param title: short capability heading shown by the availability interface.
    :param message: primary explanation shown with the offer.
    :param requirement: pip requirement used to build a local install command
        for an install action, or ``None`` when no local command is available.
    :param recipe: optional setup or external-environment instructions appended
        to the message.
    :param runs_anything: informational local-install marker set by
        :func:`offer_install`; :attr:`command`, not this flag, controls
        execution.
    """

    action: str
    title: str
    message: str
    requirement: Optional[str] = None
    recipe: str = ""
    runs_anything: bool = False

    @property
    def command(self) -> Optional[list]:
        """The install command, or ``None`` when nothing may be run."""
        if self.action != "install" or not self.requirement:
            return None
        return install_requirement_command(self.requirement)

    def as_text(self) -> str:
        """Message and recipe as one block, for a dialog or a log."""
        parts = [self.message.strip()]
        if self.recipe.strip():
            parts.append(self.recipe.strip())
        return "\n\n".join(part for part in parts if part)


def offer_ready(title: str, message: str) -> InstallOffer:
    """An offer for something that is already available.

    :param title: short heading shown for the available capability.
    :param message: explanation shown with the offer.
    """
    return InstallOffer("ready", title, message)


def offer_install(title: str, message: str, requirement: str,
                  recipe: str = "") -> InstallOffer:
    """An offer that may run pip here, after a dry run and a confirmation.

    :param title: short heading shown for the optional capability.
    :param message: explanation shown with the install offer.
    :param requirement: pip requirement string that can satisfy the feature.
    """
    return InstallOffer("install", title, message, str(requirement), recipe,
                        runs_anything=True)


def offer_elsewhere(title: str, message: str, recipe: str) -> InstallOffer:
    """An offer that names another environment and runs nothing.

    :param title: short heading shown for the optional capability.
    :param message: explanation of why installation must happen elsewhere.
    :param recipe: instructions for preparing the external environment.
    """
    return InstallOffer("elsewhere", title, message, None, recipe)


def offer_impossible(title: str, message: str, recipe: str = "") -> InstallOffer:
    """An offer that says installing cannot help, and why.

    :param title: short heading shown for the unavailable capability.
    :param message: explanation of why installation cannot satisfy it.
    """
    return InstallOffer("impossible", title, message, None, recipe)
