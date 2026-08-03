"""``spacr-doctor`` — diagnose a spaCR installation and say how to fix it.

One command, one line per check, and for every line that is not ``PASS`` a
concrete command the user can copy and run. A diagnostic that says
"GPU not available" without saying what to do about it is not a diagnostic,
so :class:`Result` makes ``fix`` a first-class field rather than an optional
afterthought, and :func:`format_report` always prints it.

The checks exist because these failures actually happened:

* **A stale editable install.** This repository is checked out more than once
  on the same machine, and an editable install points at exactly one of them.
  Editing checkout A while ``import spacr`` resolves to checkout B costs hours
  before anyone thinks to print ``spacr.__file__``. :func:`check_running_checkout`
  is the single most valuable function in this module.
* **A missing optional extra.** PySide6 lives in the ``qt`` extra, not in the
  core dependencies, so a plain ``pip install spacr`` used to install a
  ``spacr`` command that died on an unhandled ``ImportError`` six frames deep.
  :mod:`spacr.qt` already grew the friendly path for that; :func:`check_qt_extra`
  reuses its logic rather than writing a second copy that can drift.
* **A GPU that is present but unusable.** A CPU-only torch build on a machine
  with an NVIDIA card, or a driver older than the CUDA runtime torch was built
  against, both present as "cuda not available" and have entirely different
  fixes.
* **Cellpose version drift.** spaCR migrated to the Cellpose 4.x / SAM API.
  Cellpose 3 lingering in an environment breaks at the first ``CellposeModel``
  call, deep inside a run that has already spent an hour on masks.
* **A project database that is corrupt, locked, or from a newer spaCR.**
* **Settings whose combination cannot work**, which
  :func:`spacr.validate.validate_settings` already knows how to name.

Design rules, all of them load-bearing:

* Every check is an independent module-level function taking a
  :class:`Context` and returning a :class:`Result` (or a sequence of them),
  so each one is callable and assertable on its own.
* :func:`run_checks` wraps every call, so a check that raises becomes an
  ``ERROR`` row rather than taking down the report. A diagnostic tool that
  crashes while diagnosing is worse than no diagnostic tool.
* Nothing heavy is imported at module scope. ``spacr-doctor --help`` must not
  pay for torch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "PASS",
    "WARN",
    "FAIL",
    "ERROR",
    "SKIP",
    "Context",
    "Result",
    "CHECKS",
    "run_checks",
    "format_report",
    "summarize",
    "exit_code",
    "build_parser",
    "main",
]

# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
#: A check that raised. Distinct from FAIL: FAIL means "your installation is
#: broken", ERROR means "the doctor is broken". Both exit non-zero, because
#: either way the user has not been told their installation is healthy.
ERROR = "ERROR"
SKIP = "SKIP"

#: Verdicts that make ``spacr-doctor`` exit non-zero, so CI can gate on it.
FAILING = frozenset({FAIL, ERROR})

_CRASH_FIX = (
    "This is a bug in spacr-doctor itself, not necessarily in your install. "
    "Re-run with --json and open an issue at "
    "https://github.com/EinarOlafsson/spacr/issues with the output."
)


@dataclass(frozen=True)
class Result:
    """One check's verdict.

    :param check: short label shown in the left column of the report.
    :param status: one of :data:`PASS`, :data:`WARN`, :data:`FAIL`,
        :data:`ERROR`, :data:`SKIP`.
    :param message: what was found, in the user's terms.
    :param fix: a command or action the user can actually carry out. Required
        in spirit for every non-``PASS`` row; :func:`format_report` prints it
        verbatim, including newlines.
    :param details: supporting facts worth showing but not worth a verdict.
    """

    check: str
    status: str
    message: str
    fix: str = ""
    details: Tuple[str, ...] = ()

    @property
    def is_failure(self) -> bool:
        """True when this row should make the command exit non-zero."""
        return self.status in FAILING


@dataclass
class Context:
    """Everything the checks are allowed to know about the invocation.

    :param checkout: the directory the user believes they are editing.
        Defaults to the current working directory, which is the whole point:
        "am I running the code I am looking at" is a question about *here*.
    :param db: optional project database to inspect.
    :param settings: optional settings file (csv or json) to validate.
    :param app: app key the settings file is for (``mask``, ``measure``, ...).
    :param probe_gpu: allocate a tensor on the GPU to prove it really works.
        A driver/runtime mismatch is invisible until something is allocated.
    """

    checkout: Path = field(default_factory=Path.cwd)
    db: Optional[Path] = None
    settings: Optional[Path] = None
    app: str = ""
    probe_gpu: bool = True


#: Populated by :func:`_register` in definition order, which is display order.
CHECKS: List[Callable[[Context], Any]] = []


def _register(label: str) -> Callable[[Callable], Callable]:
    """Add a check to :data:`CHECKS` and give it the label the report shows."""

    def decorate(function: Callable) -> Callable:
        function.check_label = label  # type: ignore[attr-defined]
        CHECKS.append(function)
        return function

    return decorate


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------

def _canonical(name: str) -> str:
    """Normalise a distribution name the way PEP 503 does."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _distribution_version(name: str) -> Optional[str]:
    """Installed version of ``name``, or ``None`` when it is not installed."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _declared_requirement(name: str) -> Optional[str]:
    """Return the version specifier spaCR declares for ``name``.

    Read from the *installed* metadata rather than from setup.py, because the
    installed metadata is what the environment was actually resolved against —
    and because a user running ``spacr-doctor`` from a wheel has no setup.py.

    :returns: e.g. ``"<5.0,>=4.0"``, ``""`` when declared without bounds, or
        ``None`` when spaCR does not declare it at all.
    """
    try:
        from importlib.metadata import requires

        declared = requires("spacr") or ()
    except Exception:
        return None
    for raw in declared:
        text = raw.strip()
        if ";" in text:
            head, _, marker = text.partition(";")
            # Requirements guarded by `extra == "..."` belong to an extra, not
            # to the core dependency set the check is asking about.
            if "extra" in marker:
                continue
            text = head.strip()
        match = re.match(r"^([A-Za-z0-9._-]+)\s*(.*)$", text)
        if match and _canonical(match.group(1)) == _canonical(name):
            return match.group(2).strip()
    return None


# Version comparison, PEP 440 subset, stdlib only.
#
# ``packaging`` would be one import away and is installed in practically every
# environment — which is exactly the reasoning that put five undeclared
# module-scope imports into this project (requests via huggingface-hub, joblib
# via scikit-learn, ...), each one upstream decision away from an ImportError.
# A tool whose job is to explain broken environments cannot itself depend on
# an environment being unbroken, so the comparison is implemented here.
#
# The subset covers what spaCR's own metadata uses: `>= <= == != < > ~= ===`,
# comma-joined clauses, `.*` prefix matching, local segments (`2.9.1+cu128`)
# and pre/post/dev suffixes. Anything outside it returns ``None`` — "cannot
# tell" — rather than a guess.

#: Longest first: `==` must be tried before `=`-prefixed shorter operators,
#: and `===` before `==`.
_OPERATORS = ("===", "==", "!=", "<=", ">=", "~=", "<", ">")

#: Ordering within one release: dev < pre < final < post.
_PRE_RANKS = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2,
              "pre": 2, "preview": 2}

_VersionKey = Tuple[Tuple[int, ...], Tuple[int, int, int]]


def _parse_version(text: Any) -> Optional[_VersionKey]:
    """Turn a version string into a sortable key, or ``None`` if it is not one.

    :returns: ``(release numbers, (stage, stage rank, stage number))`` where
        stage is -1 dev, 0 pre-release, 1 final, 2 post-release.
    """
    cleaned = str(text).strip().lower().split("+", 1)[0]
    match = re.match(r"^v?(\d+(?:\.\d+)*)(.*)$", cleaned)
    if not match:
        return None
    release = tuple(int(part) for part in match.group(1).split("."))
    tail = match.group(2).strip()
    if not tail:
        return release, (1, 0, 0)
    suffix = re.fullmatch(
        r"[._-]?(a|b|c|rc|alpha|beta|pre|preview|post|dev)[._-]?(\d*)", tail
    )
    if suffix is None:
        return None
    label, number = suffix.group(1), int(suffix.group(2) or 0)
    if label == "dev":
        return release, (-1, 0, number)
    if label == "post":
        return release, (2, 0, number)
    return release, (0, _PRE_RANKS[label], number)


def _compare_versions(left: _VersionKey, right: _VersionKey) -> int:
    """Three-way compare, zero-padding the shorter release (``4.0`` == ``4.0.0``)."""
    left_release, left_stage = left
    right_release, right_stage = right
    width = max(len(left_release), len(right_release))
    padded_left = left_release + (0,) * (width - len(left_release))
    padded_right = right_release + (0,) * (width - len(right_release))
    if padded_left != padded_right:
        return -1 if padded_left < padded_right else 1
    if left_stage != right_stage:
        return -1 if left_stage < right_stage else 1
    return 0


def _clause_holds(
    operator: str, bound_text: str, version: _VersionKey, version_text: Any
) -> Optional[bool]:
    """Evaluate one comparison clause, or ``None`` when it cannot be parsed."""
    if operator == "===":
        return str(version_text).strip() == bound_text
    if bound_text.endswith(".*"):
        if operator not in ("==", "!="):
            return None
        prefix = _parse_version(bound_text[:-2])
        if prefix is None:
            return None
        head = prefix[0]
        matched = version[0][: len(head)] == head
        return matched if operator == "==" else not matched
    bound = _parse_version(bound_text)
    if bound is None:
        return None
    if operator == "~=":
        if len(bound[0]) < 2:
            return None
        upper = bound[0][:-1]
        return (
            _compare_versions(version, bound) >= 0
            and version[0][: len(upper)] == upper
        )
    order = _compare_versions(version, bound)
    return {
        "==": order == 0,
        "!=": order != 0,
        "<=": order <= 0,
        ">=": order >= 0,
        "<": order < 0,
        ">": order > 0,
    }[operator]


def _satisfies(specifier: str, version_text: str) -> Optional[bool]:
    """Does ``version_text`` satisfy ``specifier``?

    :returns: ``True``/``False``, or ``None`` when the question cannot be
        answered because either string is outside the supported subset. The
        callers turn ``None`` into a WARN rather than guessing, because a
        guessed version verdict is worse than an admitted unknown.
    """
    version = _parse_version(version_text)
    if version is None:
        return None
    for clause in (specifier or "").split(","):
        clause = clause.strip()
        if not clause:
            continue
        for operator in _OPERATORS:
            if clause.startswith(operator):
                break
        else:
            return None
        outcome = _clause_holds(
            operator, clause[len(operator):].strip(), version, version_text
        )
        if outcome is None:
            return None
        if not outcome:
            return False
    return True


def _package_root(module: Any) -> Optional[Path]:
    """Directory containing ``module``'s package, resolved through symlinks."""
    origin = getattr(module, "__file__", None)
    if not origin:
        return None
    return Path(origin).resolve().parent


def _checkout_root(start: Path) -> Optional[Path]:
    """Walk up from ``start`` looking for a spaCR source checkout.

    A checkout is a directory holding both a build description and the package
    directory — the shape you get from ``git clone``, and the shape
    ``pip install -e .`` expects to be pointed at.
    """
    try:
        here = Path(start).resolve()
    except OSError:
        return None
    for candidate in (here, *here.parents):
        try:
            has_package = (candidate / "spacr" / "__init__.py").is_file()
            has_build = (
                (candidate / "setup.py").is_file()
                or (candidate / "pyproject.toml").is_file()
            )
        except OSError:
            continue
        if has_package and has_build:
            return candidate
    return None


#: Distribution names that install a package directory literally called
#: ``spacr``. Both have shipped; installed together they overwrite each other.
SPACR_DISTRIBUTION_NAMES = ("spacr", "spacr-nightly")


def _spacr_distributions() -> List[Tuple[str, str, str]]:
    """Every installed metadata directory claiming to be spaCR.

    Deliberately enumerates rather than calling ``distribution("spacr")``,
    which returns the *first* match on ``sys.path`` and hides the rest. Inside
    a checkout that first match is often a leftover ``spacr.egg-info``, whose
    recorded version and dependency list can be arbitrarily old — and which
    has no ``direct_url.json``, so an editable install looks absent.

    :returns: ``(name, version, metadata directory)`` triples.
    """
    try:
        from importlib.metadata import distributions
    except Exception:
        return []
    found: List[Tuple[str, str, str]] = []
    for dist in distributions():
        try:
            name = dist.metadata["Name"] or ""
        except Exception:
            continue
        if _canonical(name) not in SPACR_DISTRIBUTION_NAMES:
            continue
        location = getattr(dist, "_path", None)
        if location is None:
            try:
                location = dist.locate_file("")
            except Exception:
                location = "unknown"
        found.append((name, dist.version or "unknown", str(location)))
    return found


def _editable_url_target(raw: Optional[str]) -> Optional[Path]:
    """Parse a PEP 610 ``direct_url.json`` body into an editable checkout path."""
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not payload.get("dir_info", {}).get("editable"):
        return None
    url = payload.get("url", "")
    if not url.startswith("file://"):
        return None
    from urllib.parse import unquote, urlparse

    try:
        return Path(unquote(urlparse(url).path)).resolve()
    except OSError:
        return None


def _editable_target() -> Optional[Path]:
    """The checkout an editable install of spaCR points at, if any.

    pip records this in ``direct_url.json`` (PEP 610). It is authoritative in
    a way that ``sys.path`` guesswork is not: it says which directory the
    install was *meant* to expose, which is exactly what a stale editable
    install gets wrong.
    """
    try:
        from importlib.metadata import distributions
    except Exception:
        return None
    for dist in distributions():
        try:
            name = dist.metadata["Name"] or ""
        except Exception:
            continue
        if _canonical(name) not in SPACR_DISTRIBUTION_NAMES:
            continue
        try:
            raw = dist.read_text("direct_url.json")
        except Exception:
            continue
        target = _editable_url_target(raw)
        if target is not None:
            return target
    return None


def _importable_spacr_dirs() -> List[Path]:
    """Every importable ``spacr`` package directory, winner first.

    More than one entry means two copies of spaCR can be imported and which
    one runs depends on the working directory — a bug that hides until the day
    it does not.

    The import machinery is consulted before ``sys.path`` is scanned, because
    an editable install ships a *finder* rather than a directory on
    ``sys.path``: scanning ``sys.path`` alone reports "no spacr installed" for
    the most common developer setup there is.
    """
    import importlib.util

    found: List[Path] = []

    def add(path: Path) -> None:
        try:
            resolved = Path(path).resolve()
        except OSError:
            return
        if resolved not in found:
            found.append(resolved)

    try:
        spec = importlib.util.find_spec("spacr")
    except Exception:
        spec = None
    if spec is not None:
        for location in spec.submodule_search_locations or ():
            add(Path(location))

    for entry in sys.path:
        base = Path(entry) if entry else Path.cwd()
        try:
            if (base / "spacr" / "__init__.py").is_file():
                add(base / "spacr")
        except OSError:
            continue
    return found


def _nvidia_driver() -> Optional[str]:
    """Driver version reported by ``nvidia-smi``, or ``None`` when absent.

    ``None`` means "no NVIDIA driver is answering", which is a different world
    from "the driver is there but torch cannot use it" — and the two need
    different remediations, which is why this is separate from torch.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        proc = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    return lines[0] if lines else None


# ---------------------------------------------------------------------------
# 1. the interpreter
# ---------------------------------------------------------------------------

#: Used only when the installed metadata cannot be read (a source tree that was
#: never installed). Kept in sync with pyproject.toml's ``requires-python``.
FALLBACK_REQUIRES_PYTHON = ">=3.9,<3.15,!=3.14.1"


@_register("python")
def check_python(ctx: Context) -> Result:
    """The running interpreter is one spaCR supports."""
    running = ".".join(str(part) for part in sys.version_info[:3])
    try:
        from importlib.metadata import metadata

        declared = metadata("spacr")["Requires-Python"] or FALLBACK_REQUIRES_PYTHON
    except Exception:
        declared = FALLBACK_REQUIRES_PYTHON

    verdict = _satisfies(declared, running)
    if verdict is None:
        return Result(
            "python",
            WARN,
            f"Python {running} could not be checked against spaCR's "
            f"requires-python ({declared}).",
            fix=(
                f"Compare by hand: spaCR supports {declared}. If this "
                "interpreter is outside that range:\n"
                "conda create -n spacr python=3.12 -y && conda activate spacr"
            ),
            details=(f"interpreter: {sys.executable}",),
        )
    if not verdict:
        return Result(
            "python",
            FAIL,
            f"Python {running} is outside spaCR's supported range ({declared}).",
            fix=(
                "conda create -n spacr python=3.12 -y && conda activate spacr && "
                'python -m pip install "spacr[qt]"'
            ),
            details=(f"interpreter: {sys.executable}",),
        )
    return Result(
        "python",
        PASS,
        f"Python {running} satisfies requires-python {declared}.",
        details=(f"interpreter: {sys.executable}",),
    )


# ---------------------------------------------------------------------------
# 2-5. which spacr am I actually running, and where did it come from
# ---------------------------------------------------------------------------

def _import_spacr() -> Any:
    """Import the ``spacr`` package. Split out so checks can be tested."""
    import spacr

    return spacr


@_register("spacr package")
def check_spacr_package(ctx: Context) -> Result:
    """``import spacr`` works, and reports where from."""
    try:
        spacr = _import_spacr()
    except Exception as exc:
        return Result(
            "spacr package",
            FAIL,
            f"`import spacr` failed: {type(exc).__name__}: {exc}",
            fix='python -m pip install "spacr[qt]"',
        )
    root = _package_root(spacr)
    version = getattr(spacr, "__version__", "unknown")
    if root is None:
        return Result(
            "spacr package",
            WARN,
            "spacr imported but has no __file__, so it cannot be located on "
            "disk (a namespace package left behind by a half-removed install?).",
            fix="python -m pip uninstall -y spacr && python -m pip install \"spacr[qt]\"",
        )
    if version == "unknown":
        return Result(
            "spacr package",
            WARN,
            f"spacr imports from {root} but no installed distribution provides "
            "it — this is a source tree on sys.path, not an install.",
            fix=f'python -m pip install -e "{root.parent}"',
        )
    return Result(
        "spacr package",
        PASS,
        f"spacr {version} imports from {root}.",
    )


@_register("running checkout")
def check_running_checkout(ctx: Context) -> Result:
    """The checkout you are standing in is the one that actually runs.

    The failure this exists for: two clones of spaCR on one machine, an
    editable install pointing at the first, and a developer editing the
    second. Every test passes, every edit does nothing, and nothing in the
    output says so.
    """
    try:
        spacr = _import_spacr()
    except Exception:
        return Result(
            "running checkout",
            SKIP,
            "spacr does not import, so there is no installation to compare "
            "this checkout against.",
            fix="Fix the `spacr package` row above first.",
        )
    package_dir = _package_root(spacr)
    if package_dir is None:
        return Result(
            "running checkout",
            SKIP,
            "spacr has no __file__, so its source directory is unknown.",
            fix="Fix the `spacr package` row above first.",
        )
    running_root = package_dir.parent
    editable = _editable_target()
    here = _checkout_root(ctx.checkout)

    details = [f"import spacr  -> {package_dir}"]
    if editable is not None:
        details.append(f"editable install points at -> {editable}")
    else:
        details.append("editable install: none (installed as a copy)")

    if here is None:
        if editable is not None and editable != running_root:
            return Result(
                "running checkout",
                FAIL,
                f"The editable install points at {editable}, but `import spacr` "
                f"resolves to {running_root}. The pointer is stale — that "
                "checkout has moved, been deleted, or been shadowed.",
                fix=f'python -m pip install -e "{editable}"',
                details=tuple(details),
            )
        return Result(
            "running checkout",
            PASS,
            f"Not inside a spaCR checkout; runs from {running_root}.",
            details=tuple(details),
        )

    details.insert(0, f"you are in -> {here}")

    if here != running_root:
        return Result(
            "running checkout",
            FAIL,
            f"You are working in {here}, but `import spacr` resolves to "
            f"{running_root}. Edits in this checkout do not change what runs.",
            fix=(
                f'python -m pip install -e "{here}"\n'
                'python -c "import spacr; print(spacr.__file__)"   # confirm'
            ),
            details=tuple(details),
        )

    if editable is None:
        return Result(
            "running checkout",
            WARN,
            f"`import spacr` lands in {here} only because this directory is on "
            "sys.path. Start a run from anywhere else and a different spaCR "
            "executes.",
            fix=f'python -m pip install -e "{here}"',
            details=tuple(details),
        )

    if editable != running_root:
        return Result(
            "running checkout",
            FAIL,
            f"The editable install points at {editable}, yet `import spacr` "
            f"resolves to {running_root} because you launched from inside it. "
            "Anything started from another directory uses the other checkout.",
            fix=f'python -m pip install -e "{here}"',
            details=tuple(details),
        )

    return Result(
        "running checkout",
        PASS,
        f"Editable install of {here} is what runs.",
        details=tuple(details),
    )


@_register("duplicate installs")
def check_duplicate_installs(ctx: Context) -> Result:
    """Exactly one importable ``spacr`` package directory exists."""
    directories = _importable_spacr_dirs()
    if not directories:
        return Result(
            "duplicate installs",
            SKIP,
            "No spacr package directory is reachable from sys.path.",
            fix='python -m pip install "spacr[qt]"',
        )
    if len(directories) == 1:
        return Result(
            "duplicate installs",
            PASS,
            f"One importable spacr package: {directories[0]}.",
        )
    listed = "\n".join(f"  {index + 1}. {path}" for index, path in enumerate(directories))
    return Result(
        "duplicate installs",
        FAIL,
        f"{len(directories)} importable spacr packages are on sys.path; "
        f"{directories[0]} wins today, and which one wins depends on the "
        "working directory.",
        fix=(
            "python -m pip uninstall -y spacr\n"
            "# repeat until pip says it is not installed, then reinstall once:\n"
            'python -m pip install "spacr[qt]"'
        ),
        details=tuple(listed.splitlines()),
    )


@_register("distributions")
def check_conflicting_distributions(ctx: Context) -> Result:
    """Exactly one metadata directory claims to be spaCR.

    Two shapes of the same problem. ``spacr`` and ``spacr-nightly`` installed
    together share one package directory and overwrite each other's files. A
    leftover ``spacr.egg-info`` inside a checkout shadows the real install
    whenever the checkout is on ``sys.path``, so version, dependency list and
    console-script table are all read from stale metadata.
    """
    present = _spacr_distributions()
    if not present:
        return Result(
            "distributions",
            WARN,
            "Neither `spacr` nor `spacr-nightly` is installed as a distribution.",
            fix='python -m pip install "spacr[qt]"',
        )
    if len(present) == 1:
        name, version, location = present[0]
        return Result(
            "distributions",
            PASS,
            f"{name} {version} is the only spaCR distribution.",
            details=(location,),
        )
    listed = tuple(
        f"{name} {version} — {location}" for name, version, location in present
    )
    distinct_names = {_canonical(name) for name, _, _ in present}
    if len(distinct_names) > 1:
        return Result(
            "distributions",
            FAIL,
            f"{len(present)} spaCR distributions are installed at once. They "
            "share the same `spacr` package directory and overwrite each other.",
            fix=(
                "python -m pip uninstall -y spacr spacr-nightly\n"
                'python -m pip install "spacr[qt]"'
            ),
            details=listed,
        )
    stale = [loc for _, _, loc in present if loc.endswith(".egg-info")]
    return Result(
        "distributions",
        WARN,
        f"{len(present)} metadata directories claim to be spaCR; the first on "
        "sys.path wins, so version and dependency metadata depend on the "
        "working directory.",
        fix=(
            "\n".join(f'rm -rf "{path}"' for path in stale)
            or 'python -m pip install --force-reinstall "spacr[qt]"'
        ),
        details=listed,
    )


@_register("console scripts")
def check_console_scripts(ctx: Context) -> Result:
    """Every installed ``spacr-*`` command points at a module that exists.

    ``sim=spacr.app_sim:gui_sim`` outlived the file it named, so the installed
    ``sim`` command died with ImportError. A partially upgraded install
    reproduces that for any command.
    """
    import importlib.util
    from importlib.metadata import distribution

    try:
        entry_points = list(distribution("spacr").entry_points)
    except Exception as exc:
        return Result(
            "console scripts",
            SKIP,
            f"No installed spaCR distribution to read entry points from "
            f"({type(exc).__name__}).",
            fix='python -m pip install "spacr[qt]"',
        )
    scripts = [ep for ep in entry_points if ep.group == "console_scripts"]
    if not scripts:
        return Result(
            "console scripts",
            WARN,
            "The installed spaCR distribution declares no console scripts.",
            fix='python -m pip install --force-reinstall "spacr[qt]"',
        )
    broken: List[str] = []
    for entry in scripts:
        module = entry.value.split(":", 1)[0]
        try:
            spec = importlib.util.find_spec(module)
        except Exception:
            spec = None
        if spec is None:
            broken.append(f"{entry.name} -> {entry.value}")
    if broken:
        return Result(
            "console scripts",
            FAIL,
            f"{len(broken)} of {len(scripts)} installed commands point at "
            "modules that do not exist; running them raises ImportError.",
            fix='python -m pip install --force-reinstall "spacr[qt]"',
            details=tuple(broken),
        )
    return Result(
        "console scripts",
        PASS,
        f"All {len(scripts)} installed spaCR commands resolve.",
    )


@_register("PATH")
def check_command_on_path(ctx: Context) -> Result:
    """The ``spacr`` you type belongs to the Python you are running.

    A second environment earlier on ``PATH`` is the other half of "which spacr
    am I actually running": the import can be right while the command is not.
    """
    located = shutil.which("spacr-doctor") or shutil.which("spacr")
    if located is None:
        return Result(
            "PATH",
            WARN,
            "No `spacr` command is on PATH; only `python -m spacr.doctor` and "
            "the other module entry points will work.",
            fix=f'export PATH="{Path(sys.executable).parent}:$PATH"',
        )
    script = Path(located).resolve()
    prefix = Path(sys.prefix).resolve()
    if prefix not in script.parents:
        return Result(
            "PATH",
            FAIL,
            f"The `spacr` command on PATH is {script}, which does not belong "
            f"to the environment you are running ({prefix}). Typing `spacr` "
            "starts a different installation than the one this report describes.",
            fix=f'export PATH="{Path(sys.executable).parent}:$PATH"',
        )
    return Result("PATH", PASS, f"`{script.name}` on PATH comes from {prefix}.")


# ---------------------------------------------------------------------------
# 6-7. optional extras
# ---------------------------------------------------------------------------

def _import_qt_app() -> Any:
    """Import the Qt GUI entry point exactly the way ``spacr`` does."""
    from .qt.app import launch

    return launch


@_register("qt extra")
def check_qt_extra(ctx: Context) -> Result:
    """The GUI's optional extra is installed.

    Reuses :mod:`spacr.qt`'s own diagnosis — ``_missing_qt_extra`` and
    ``_QT_MISSING_MESSAGE`` — rather than restating which distributions are in
    the extra, so the two cannot drift apart.
    """
    from .qt import _QT_MISSING_MESSAGE, _missing_qt_extra

    try:
        _import_qt_app()
    except ImportError as exc:
        module = _missing_qt_extra(exc)
        if module is None:
            return Result(
                "qt extra",
                FAIL,
                f"The Qt GUI failed to import for a reason unrelated to the "
                f"optional extra: {exc}",
                fix=_CRASH_FIX,
            )
        return Result(
            "qt extra",
            FAIL,
            f"The Qt GUI is unavailable: {module} is not installed.",
            fix=_QT_MISSING_MESSAGE.format(module=module),
        )
    except Exception as exc:
        return Result(
            "qt extra",
            FAIL,
            f"The Qt GUI raised while importing: {type(exc).__name__}: {exc}",
            fix=_CRASH_FIX,
        )
    version = _distribution_version("PySide6") or "unknown version"
    return Result("qt extra", PASS, f"Qt GUI available (PySide6 {version}).")


@_register("display")
def check_display(ctx: Context) -> Result:
    """A GUI can actually open a window here."""
    if not sys.platform.startswith("linux"):
        return Result(
            "display",
            PASS,
            f"{sys.platform} manages its own display; no DISPLAY needed.",
        )
    platform_override = os.environ.get("QT_QPA_PLATFORM", "")
    if platform_override in {"offscreen", "minimal", "vnc"}:
        return Result(
            "display",
            PASS,
            f"QT_QPA_PLATFORM={platform_override}: Qt is deliberately headless.",
        )
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return Result("display", PASS, "A display server is reachable.")
    return Result(
        "display",
        WARN,
        "No DISPLAY or WAYLAND_DISPLAY is set, so the GUI cannot open a "
        "window. The pipelines still run headless.",
        fix=(
            "spacr-run --list          # run pipelines without a GUI\n"
            "ssh -X user@host          # or forward a display\n"
            "export QT_QPA_PLATFORM=offscreen   # or render off-screen"
        ),
    )


#: Modules spaCR imports during a run, and the distribution that provides each
#: when the two names differ. Only core dependencies belong here — an optional
#: extra missing is not a broken install.
CORE_MODULES: Tuple[Tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("skimage", "scikit-image"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("cv2", "opencv-python"),
    ("PIL", "pillow"),
    ("tifffile", "tifffile"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("cellpose", "cellpose"),
)


@_register("core imports")
def check_core_dependencies(ctx: Context) -> Result:
    """Every core dependency imports."""
    import importlib

    missing: List[Tuple[str, str]] = []
    for module, dist in CORE_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append((module, f"{dist} ({type(exc).__name__}: {exc})"))
    if not missing:
        return Result(
            "core imports",
            PASS,
            f"All {len(CORE_MODULES)} core dependencies import.",
        )
    dists = " ".join(sorted({dist.split(" ")[0] for _, dist in missing}))
    return Result(
        "core imports",
        FAIL,
        f"{len(missing)} core dependencies cannot be imported; a run would "
        "fail partway through.",
        fix=f"python -m pip install {dists}",
        details=tuple(f"{module}: {detail}" for module, detail in missing),
    )


#: Extras whose absence is normal but whose *partial* presence is not.
OPTIONAL_EXTRAS: Dict[str, Tuple[str, ...]] = {
    "qt": ("PySide6", "qtawesome"),
    "umap": ("umap-learn",),
    "boosting": ("catboost", "lightgbm"),
    "czi": ("pylibCZIrw", "czifile"),
    "nd2": ("nd2reader",),
    "lif": ("readlif",),
    "zernike": ("mahotas",),
    "btrack": ("btrack",),
    "trackastra": ("trackastra",),
    "ultrack": ("ultrack",),
    "attribution": ("torchcam",),
}


@_register("optional extras")
def check_optional_extras(ctx: Context) -> Result:
    """No optional extra is half-installed.

    A missing extra is fine and expected. An extra with some of its
    distributions present and others not is a resolve that went wrong, and it
    fails at the moment the feature is used rather than at install time.
    """
    installed: List[str] = []
    absent: List[str] = []
    partial: List[str] = []
    for extra, dists in sorted(OPTIONAL_EXTRAS.items()):
        found = [name for name in dists if _distribution_version(name) is not None]
        if len(found) == len(dists):
            installed.append(extra)
        elif not found:
            absent.append(extra)
        else:
            missing = sorted(set(dists) - set(found))
            partial.append(f"{extra}: missing {', '.join(missing)}")
    summary = (
        f"installed: {', '.join(installed) or 'none'}; "
        f"not installed: {', '.join(absent) or 'none'}"
    )
    if partial:
        return Result(
            "optional extras",
            WARN,
            f"{len(partial)} optional extras are half-installed and will fail "
            "when their feature is used.",
            fix="\n".join(
                f'python -m pip install "spacr[{entry.split(":", 1)[0]}]"'
                for entry in partial
            ),
            details=tuple(partial) + (summary,),
        )
    return Result("optional extras", PASS, summary)


# ---------------------------------------------------------------------------
# 8-10. compute
# ---------------------------------------------------------------------------

def _import_torch() -> Any:
    """Import torch. Split out so the GPU checks can be tested without one."""
    import torch

    return torch


@_register("torch")
def check_torch(ctx: Context) -> Result:
    """torch imports, and says whether it was built with CUDA at all."""
    try:
        torch = _import_torch()
    except Exception as exc:
        return Result(
            "torch",
            FAIL,
            f"torch does not import: {type(exc).__name__}: {exc}",
            fix="python -m pip install torch torchvision",
        )
    built = getattr(getattr(torch, "version", None), "cuda", None)
    suffix = f"built against CUDA {built}" if built else "CPU-only build"
    return Result("torch", PASS, f"torch {torch.__version__} ({suffix}).")


@_register("gpu")
def check_gpu(ctx: Context) -> Result:
    """CUDA is not merely reported as present but is actually usable."""
    try:
        torch = _import_torch()
    except Exception:
        return Result(
            "gpu",
            SKIP,
            "torch does not import, so CUDA cannot be checked.",
            fix="Fix the `torch` row above first.",
        )
    driver = _nvidia_driver()
    built = getattr(getattr(torch, "version", None), "cuda", None)

    if not built:
        if driver:
            return Result(
                "gpu",
                FAIL,
                f"An NVIDIA driver ({driver}) is present, but this torch is a "
                "CPU-only build and will never use the card.",
                fix=(
                    "python -m pip install --force-reinstall torch torchvision "
                    "--index-url https://download.pytorch.org/whl/cu124"
                ),
            )
        return Result(
            "gpu",
            WARN,
            "No NVIDIA driver and a CPU-only torch: spaCR will run, but "
            "segmentation and training will be very slow.",
            fix=(
                "Run on a CUDA machine, or accept CPU speed. To rule out a "
                "driver problem: nvidia-smi"
            ),
        )

    if not torch.cuda.is_available():
        reason = ""
        try:
            torch.cuda.init()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        if driver is None:
            return Result(
                "gpu",
                FAIL,
                f"torch was built against CUDA {built} but no NVIDIA driver is "
                "answering, so no GPU can be used.",
                fix=(
                    "nvidia-smi    # if this fails, install or reload the driver:\n"
                    "sudo apt install nvidia-driver-550 && sudo reboot"
                ),
                details=(reason,) if reason else (),
            )
        return Result(
            "gpu",
            FAIL,
            f"Driver {driver} is loaded and torch was built against CUDA "
            f"{built}, but torch.cuda.is_available() is False — a driver / "
            "runtime mismatch.",
            fix=(
                "Install the torch build that matches your driver, e.g.:\n"
                "python -m pip install --force-reinstall torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu121"
            ),
            details=(reason,) if reason else (),
        )

    count = torch.cuda.device_count()
    try:
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(count))
    except Exception as exc:
        names = f"unnamed ({type(exc).__name__})"

    if ctx.probe_gpu:
        try:
            tensor = torch.zeros(8, 8, device="cuda")
            float((tensor @ tensor).sum().item())
            torch.cuda.synchronize()
        except Exception as exc:
            return Result(
                "gpu",
                FAIL,
                f"CUDA reports {count} device(s) but the first allocation "
                f"failed: {type(exc).__name__}: {exc}",
                fix=(
                    "Usually a driver/runtime mismatch or an out-of-memory "
                    "card. Check `nvidia-smi` for other processes, then:\n"
                    "python -m pip install --force-reinstall torch torchvision "
                    "--index-url https://download.pytorch.org/whl/cu124"
                ),
            )
        return Result(
            "gpu",
            PASS,
            f"{count} CUDA device(s) usable: {names} (driver {driver}, "
            f"torch CUDA {built}).",
        )
    return Result(
        "gpu",
        PASS,
        f"{count} CUDA device(s) reported: {names} (driver {driver}, torch "
        f"CUDA {built}); allocation probe skipped.",
    )


#: The Cellpose major version spaCR's code is written against. spaCR calls
#: ``CellposeModel(pretrained_model=...)`` and the ``cpsam`` weights; the
#: Cellpose 3 wrapper class ``models.Cellpose`` is called nowhere.
EXPECTED_CELLPOSE_MAJOR = 4
FALLBACK_CELLPOSE_SPECIFIER = ">=4.0,<5.0"


@_register("cellpose")
def check_cellpose(ctx: Context) -> Result:
    """The installed Cellpose is the 4.x / SAM API spaCR calls."""
    try:
        import cellpose
    except Exception as exc:
        return Result(
            "cellpose",
            FAIL,
            f"cellpose does not import: {type(exc).__name__}: {exc}",
            fix=f'python -m pip install "cellpose{FALLBACK_CELLPOSE_SPECIFIER}"',
        )
    installed = (
        getattr(cellpose, "version", None)
        or getattr(cellpose, "__version__", None)
        or _distribution_version("cellpose")
    )
    if not installed:
        return Result(
            "cellpose",
            WARN,
            "cellpose imports but reports no version, so drift cannot be checked.",
            fix=f'python -m pip install --force-reinstall "cellpose{FALLBACK_CELLPOSE_SPECIFIER}"',
        )
    specifier = _declared_requirement("cellpose") or FALLBACK_CELLPOSE_SPECIFIER
    verdict = _satisfies(specifier, str(installed))
    fix = f'python -m pip install "cellpose{FALLBACK_CELLPOSE_SPECIFIER}"'
    if verdict is False:
        return Result(
            "cellpose",
            FAIL,
            f"cellpose {installed} does not satisfy spaCR's requirement "
            f"({specifier}). spaCR calls the Cellpose 4 API — "
            "CellposeModel(pretrained_model='cpsam') — which a 3.x install "
            "does not provide, so masking dies at the first model call.",
            fix=fix,
        )

    from cellpose import models

    problems: List[str] = []
    if not hasattr(models, "CellposeModel"):
        problems.append("cellpose.models.CellposeModel is missing")
    if hasattr(models, "Cellpose"):
        problems.append(
            "cellpose.models.Cellpose exists — that wrapper is the 3.x API"
        )
    names = tuple(getattr(models, "MODEL_NAMES", ()) or ())
    if "cpsam" not in names:
        problems.append(f"'cpsam' is not in cellpose.models.MODEL_NAMES {names}")

    if problems:
        return Result(
            "cellpose",
            FAIL if verdict is not None else WARN,
            f"cellpose {installed} does not expose the API spaCR calls.",
            fix=fix,
            details=tuple(problems),
        )
    if verdict is None:
        return Result(
            "cellpose",
            WARN,
            f"cellpose {installed} exposes the Cellpose 4 API, but its version "
            f"string could not be compared with {specifier}.",
            fix=fix,
        )
    return Result(
        "cellpose",
        PASS,
        f"cellpose {installed} satisfies {specifier} and exposes the "
        "Cellpose 4 / SAM API.",
    )


@_register("declared pins")
def check_declared_pins(ctx: Context) -> Result:
    """A checkout's environment.yaml does not contradict its setup.py.

    Only meaningful inside a source checkout, and it is there that it matters:
    ``conda env create -f environment.yaml`` is how a new user builds an
    environment, so a pin in that file that setup.py forbids produces an
    install that is broken before anyone runs anything.
    """
    root = _checkout_root(ctx.checkout)
    if root is None:
        return Result(
            "declared pins",
            SKIP,
            "Not inside a spaCR checkout; nothing to cross-check.",
        )
    setup_py = root / "setup.py"
    env_yaml = root / "environment.yaml"
    if not setup_py.is_file() or not env_yaml.is_file():
        return Result(
            "declared pins",
            SKIP,
            f"{root} has no setup.py/environment.yaml pair to cross-check.",
        )
    declared = _parse_setup_dependencies(setup_py)
    if not declared:
        return Result(
            "declared pins",
            SKIP,
            "Could not read the dependency list out of setup.py.",
        )
    pinned = _parse_environment_pins(env_yaml)
    conflicts: List[str] = []
    for name, pinned_version in sorted(pinned.items()):
        specifier = declared.get(name)
        if not specifier:
            continue
        if _satisfies(specifier, pinned_version) is False:
            conflicts.append(
                f"environment.yaml pins {name}=={pinned_version}, setup.py "
                f"requires {name}{specifier}"
            )
    if conflicts:
        return Result(
            "declared pins",
            WARN,
            f"{len(conflicts)} pins in environment.yaml contradict setup.py; "
            "an environment built from that file cannot run this spaCR.",
            fix=(
                f'python -m pip install -e "{root}[qt]"   # install from '
                "setup.py instead of environment.yaml"
            ),
            details=tuple(conflicts),
        )
    return Result(
        "declared pins",
        PASS,
        f"environment.yaml agrees with setup.py on {len(pinned)} pinned packages.",
    )


def _parse_setup_dependencies(setup_py: Path) -> Dict[str, str]:
    """Map distribution name to version specifier from setup.py's list.

    ``install_requires=dependencies`` is a name reference, so the list itself
    has to be found by its assignment rather than in the ``setup()`` call.
    """
    import ast

    try:
        tree = ast.parse(setup_py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return {}
    found: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "dependencies" not in targets:
            continue
        try:
            values = ast.literal_eval(node.value)
        except ValueError:
            continue
        for raw in values:
            if not isinstance(raw, str):
                continue
            match = re.match(r"^([A-Za-z0-9._-]+)\s*(.*)$", raw.strip())
            if match:
                found[_canonical(match.group(1))] = match.group(2).strip()
    return found


def _parse_environment_pins(env_yaml: Path) -> Dict[str, str]:
    """Map distribution name to exact version from an environment.yaml.

    Deliberately a regex over the ``- name==version`` / ``- name=version``
    lines rather than a YAML parse: PyYAML is not a spaCR dependency, and a
    doctor that needs an extra install to run is a doctor nobody runs.
    """
    try:
        text = env_yaml.read_text(encoding="utf-8")
    except OSError:
        return {}
    pins: Dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(
            r"^\s*-\s*([A-Za-z0-9._-]+)\s*={1,2}\s*([0-9][^\s#]*)\s*$", line
        )
        if match:
            pins[_canonical(match.group(1))] = match.group(2)
    return pins


# ---------------------------------------------------------------------------
# 11-12. project data
# ---------------------------------------------------------------------------

@_register("database")
def check_database(ctx: Context) -> Union[Result, List[Result]]:
    """A project database is readable, intact, unlocked, and the right schema."""
    if ctx.db is None:
        return Result(
            "database",
            SKIP,
            "No database given.",
            fix="spacr-doctor --db /path/to/plate/measurements/measurements.db",
        )
    path = Path(ctx.db)
    if not path.is_file():
        return Result(
            "database",
            FAIL,
            f"{path} is not a file.",
            fix=(
                "spaCR writes <src>/measurements/measurements.db. Point --db "
                "at that file, or run the measure step to create it."
            ),
        )

    import sqlite3

    from .database_concurrency import inspect_database

    results: List[Result] = []
    try:
        health = inspect_database(path, quick_check=True)
    except sqlite3.DatabaseError as exc:
        return Result(
            "database",
            FAIL,
            f"{path} cannot be opened as SQLite: {exc}",
            fix=(
                "The file is truncated or is not a database. Restore it from "
                "backup, or re-run the measure step. To confirm:\n"
                f'sqlite3 "{path}" ".schema"'
            ),
        )

    if health.quick_check not in (None, "ok"):
        results.append(
            Result(
                "database",
                FAIL,
                f"SQLite reports corruption in {path}: {health.quick_check}",
                fix=(
                    f'sqlite3 "{path}" ".recover" > recovered.sql && '
                    f'sqlite3 recovered.db < recovered.sql'
                ),
            )
        )
    else:
        results.append(
            Result(
                "database",
                PASS,
                f"{path} passes SQLite quick_check "
                f"(journal={health.journal_mode}, "
                f"busy_timeout={health.busy_timeout_ms} ms).",
            )
        )

    for warning in health.warnings:
        if health.quick_check not in (None, "ok") and "quick_check" in warning:
            continue
        results.append(
            Result(
                "database",
                WARN,
                warning,
                fix=(
                    "WAL journaling is unreliable on network filesystems. "
                    "Copy the project to local disk, or set "
                    "journal_mode=DELETE before running."
                ),
            )
        )

    results.extend(_database_schema_rows(path))
    results.extend(_database_lock_rows(path))
    return results


def _database_schema_rows(path: Path) -> List[Result]:
    """Compare a database's on-disk schema version with this spaCR's."""
    from .database_schema import CURRENT_SCHEMA_VERSION, database_schema_version

    try:
        found = database_schema_version(path)
    except Exception as exc:
        return [
            Result(
                "database schema",
                WARN,
                f"Could not read the schema version of {path}: "
                f"{type(exc).__name__}: {exc}",
                fix="Confirm the file is a spaCR measurements database.",
            )
        ]
    if found > CURRENT_SCHEMA_VERSION:
        return [
            Result(
                "database schema",
                FAIL,
                f"{path} uses spaCR database schema {found}, but this "
                f"installation supports up to {CURRENT_SCHEMA_VERSION}. It was "
                "written by a newer spaCR.",
                fix=(
                    'python -m pip install --upgrade "spacr[qt]"   # never '
                    "downgrade the database file"
                ),
            )
        ]
    rows: List[Result] = []
    if found < CURRENT_SCHEMA_VERSION:
        rows.append(
            Result(
                "database schema",
                WARN,
                f"{path} is at schema {found}; this spaCR expects "
                f"{CURRENT_SCHEMA_VERSION}.",
                fix=(
                    "python -c \"from spacr.database_schema import "
                    f"ensure_database_schema; ensure_database_schema(r'{path}')\""
                ),
            )
        )
    else:
        rows.append(
            Result(
                "database schema",
                PASS,
                f"Schema version {found} matches this spaCR.",
            )
        )
    rows.extend(_database_table_rows(path))
    return rows


def _database_table_rows(path: Path) -> List[Result]:
    """Warn when a database has none of the tables spaCR writes."""
    import sqlite3

    from .schema import OWNED_TABLES

    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0) as conn:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
    except sqlite3.Error as exc:
        return [
            Result(
                "database schema",
                WARN,
                f"Could not list the tables in {path}: {exc}",
                fix="Close any process holding the database and re-run.",
            )
        ]
    known = sorted(names & set(OWNED_TABLES))
    if not known:
        return [
            Result(
                "database schema",
                WARN,
                f"{path} is a valid SQLite file but contains none of spaCR's "
                f"tables ({len(names)} tables found). This is probably not a "
                "measurements database.",
                fix=(
                    "Point --db at <src>/measurements/measurements.db, or run "
                    "the measure step to create it."
                ),
            )
        ]
    return [
        Result(
            "database schema",
            PASS,
            f"{len(known)} spaCR tables present: {', '.join(known)}.",
        )
    ]


def _database_lock_rows(path: Path) -> List[Result]:
    """Detect a database another process is holding a write lock on."""
    from .database_concurrency import DatabaseBusy, connect, is_busy_error, transaction

    if not os.access(path, os.W_OK):
        return [
            Result(
                "database locking",
                WARN,
                f"{path} is not writable by this user, so a run that writes to "
                "it would fail.",
                fix=f'chmod u+w "{path}"',
            )
        ]
    try:
        conn = connect(path, timeout=1.0)
    except Exception as exc:
        return [
            Result(
                "database locking",
                WARN,
                f"Could not open {path} for writing: {type(exc).__name__}: {exc}",
                fix="Close any spaCR GUI or run still holding the database.",
            )
        ]
    try:
        with transaction(conn, attempts=1):
            pass
    except DatabaseBusy as exc:
        return [
            Result(
                "database locking",
                FAIL,
                f"{path} is locked by another process: {exc}",
                fix=(
                    "Close the spaCR GUI or the run still writing to it. To "
                    f'find the holder:\nfuser -v "{path}"'
                ),
            )
        ]
    except Exception as exc:
        if is_busy_error(exc):
            return [
                Result(
                    "database locking",
                    FAIL,
                    f"{path} is locked by another process: {exc}",
                    fix=f'fuser -v "{path}"',
                )
            ]
        return [
            Result(
                "database locking",
                WARN,
                f"Lock probe on {path} failed: {type(exc).__name__}: {exc}",
                fix="Close any process holding the database and re-run.",
            )
        ]
    finally:
        conn.close()
    return [Result("database locking", PASS, f"{path} is writable and unlocked.")]


@_register("settings")
def check_settings(ctx: Context) -> Union[Result, List[Result]]:
    """A settings file names a combination that can actually run.

    Delegates to :func:`spacr.validate.validate_settings`, which already knows
    every combination this project has seen fail — ``normalize=True`` with
    ``measure``, a ``crop_mode`` naming an object with no mask dimension, a
    mask run with all four object channels unset.
    """
    if ctx.settings is None:
        return Result(
            "settings",
            SKIP,
            "No settings file given.",
            fix="spacr-doctor --settings settings.csv --app measure",
        )
    path = Path(ctx.settings)
    if not path.is_file():
        return Result(
            "settings",
            FAIL,
            f"{path} is not a file.",
            fix="Point --settings at a settings .csv or .json file.",
        )
    if not ctx.app:
        return Result(
            "settings",
            WARN,
            f"{path} cannot be validated without knowing which app it is for.",
            fix="spacr-doctor --settings "
            f'"{path}" --app measure    # or mask, classify, ...',
        )

    from .cli import load_settings_file
    from .validate import ERROR as SETTING_ERROR
    from .validate import validate_settings

    try:
        settings = load_settings_file(str(path))
    except Exception as exc:
        return Result(
            "settings",
            FAIL,
            f"{path} could not be read: {type(exc).__name__}: {exc}",
            fix="Export a fresh settings file from the GUI, or check the "
            "file is valid csv/json.",
        )
    problems = validate_settings(settings, ctx.app)
    if not problems:
        return Result(
            "settings",
            PASS,
            f"{path} is a valid {ctx.app} configuration ({len(settings)} keys).",
        )
    rows: List[Result] = []
    for problem in problems:
        label = f"[{problem.setting}] " if problem.setting else ""
        rows.append(
            Result(
                "settings",
                FAIL if problem.severity == SETTING_ERROR else WARN,
                f"{label}{problem.message}",
                fix=problem.fix,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# running and reporting
# ---------------------------------------------------------------------------

def run_checks(
    ctx: Context, checks: Optional[Sequence[Callable[[Context], Any]]] = None
) -> List[Result]:
    """Run every check and return its rows.

    A check that raises becomes an ``ERROR`` row and the run continues: the
    whole value of this command is the rows it does produce, and losing all of
    them because one probe hit an unexpected filesystem would be absurd.
    ``KeyboardInterrupt`` is the one exception — the user asking to stop is not
    a diagnostic finding.
    """
    selected = CHECKS if checks is None else checks
    results: List[Result] = []
    for function in selected:
        label = getattr(
            function, "check_label", getattr(function, "__name__", "check")
        )
        try:
            outcome = function(ctx)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - a doctor must not die
            results.append(
                Result(
                    label,
                    ERROR,
                    f"the check itself failed: {type(exc).__name__}: {exc}",
                    fix=_CRASH_FIX,
                )
            )
            continue
        if outcome is None:
            continue
        if isinstance(outcome, Result):
            results.append(outcome)
        else:
            results.extend(outcome)
    return results


def summarize(results: Iterable[Result]) -> Dict[str, int]:
    """Count rows by verdict, always returning every key."""
    counts = {status: 0 for status in (PASS, WARN, FAIL, ERROR, SKIP)}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def exit_code(results: Iterable[Result], strict: bool = False) -> int:
    """``0`` when the installation is healthy, ``1`` otherwise.

    :param strict: also fail on ``WARN``, for CI that wants a clean bill.
    """
    rows = list(results)
    if any(row.is_failure for row in rows):
        return 1
    if strict and any(row.status == WARN for row in rows):
        return 1
    return 0


def format_report(results: Sequence[Result]) -> str:
    """Render the rows as the text the command prints."""
    width = max((len(row.check) for row in results), default=0)
    lines: List[str] = []
    for row in results:
        lines.append(f"{row.status:<5} {row.check.ljust(width)}  {row.message}")
        for detail in row.details:
            lines.append(f"      {' ' * width}  {detail}")
        if row.status != PASS and row.fix:
            for index, fix_line in enumerate(row.fix.splitlines()):
                prefix = "fix: " if index == 0 else "     "
                lines.append(f"      {' ' * width}  {prefix}{fix_line}")
    counts = summarize(results)
    lines.append("")
    lines.append(
        f"{counts[PASS]} passed, {counts[WARN]} warnings, {counts[FAIL]} failed, "
        f"{counts[ERROR]} errored, {counts[SKIP]} skipped"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Return the ``spacr-doctor`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="spacr-doctor",
        description=(
            "Diagnose a spaCR installation: which spacr is actually running, "
            "which optional extras are missing, whether the GPU is usable, "
            "whether Cellpose matches what the code calls, and whether a "
            "project database or settings file is sound."
        ),
        epilog="Exits non-zero if any check fails, so CI can gate on it.",
    )
    parser.add_argument(
        "--checkout",
        metavar="DIR",
        default=None,
        help="the checkout you believe you are editing (default: current directory)",
    )
    parser.add_argument(
        "--db", metavar="PATH", default=None, help="project database to inspect"
    )
    parser.add_argument(
        "--settings", metavar="PATH", default=None, help="settings .csv/.json to validate"
    )
    parser.add_argument(
        "--app", metavar="KEY", default="", help="app key the settings file is for"
    )
    parser.add_argument(
        "--no-gpu-probe",
        action="store_true",
        help="do not allocate a tensor on the GPU (report what torch says instead)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run every check, print the report, and return a shell exit code."""
    args = build_parser().parse_args(argv)
    ctx = Context(
        checkout=Path(args.checkout) if args.checkout else Path.cwd(),
        db=Path(args.db) if args.db else None,
        settings=Path(args.settings) if args.settings else None,
        app=args.app,
        probe_gpu=not args.no_gpu_probe,
    )
    results = run_checks(ctx)
    status = exit_code(results, strict=args.strict)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": status == 0,
                    "summary": summarize(results),
                    "results": [asdict(row) for row in results],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(format_report(results))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
