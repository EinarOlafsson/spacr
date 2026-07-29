"""The packaging metadata must be self-consistent, and honest.

Three things went wrong at once in spaCR's packaging, and each one was
invisible from inside a working install:

1. ``setup.py`` ended with a module-level loop that shelled out to
   ``subprocess.run(['pip', 'install', dep])``. It ran on every build and on
   every ``pip install .``: it breaks PEP 517 isolated builds (there is no
   pip inside the isolated env), breaks every offline install, swallowed all
   failures behind ``except CalledProcessError: pass``, invoked the bare name
   ``pip`` (absent from PATH in many venv layouts), and installed an entire
   second, unused Qt binding — pyqtgraph, pyqt6, pyqt6.sip, qtpy, superqt,
   a hand-copy of cellpose's ``gui`` extra, duplicated within itself. 75
   files under ``spacr/`` import PySide6; zero import any of those five.

2. There was no ``python_requires`` anywhere, and ``pyproject.toml`` was
   three lines of build-system. On Python 3.13 ``pip install spacr``
   therefore resolved, downloaded several hundred MB, and died inside a
   *source build of numpy 1.26.4* with a compiler error — which reads to a
   user as "spaCR is broken", not "spaCR does not support 3.13 yet".

3. The classifiers claimed ``Operating System :: OS Independent``, which is
   not true: Windows-on-ARM cannot install spaCR at all, because torch has
   never published a ``win_arm64`` wheel for any version (and neither has
   ``numpy<2.0``, ``opencv-python-headless``, ``mahotas`` or ``pylibCZIrw``).

These tests pin all three shut, plus the invariants that keep the
``pyproject.toml`` / ``setup.py`` split working. They deliberately do **not**
import :mod:`spacr` — they read ``setup.py`` and ``pyproject.toml`` as data,
so they can run in a CI job with no scientific stack installed, which is
exactly the job that would otherwise hide a metadata bug behind a resolver
failure.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_PY = REPO_ROOT / "setup.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_TXT = REPO_ROOT / "requirements.txt"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


# ---------------------------------------------------------------------------
# Readers. No dependency on spacr, and no hard dependency on a TOML parser:
# tomllib is stdlib only from 3.11; spaCR also supports Python 3.9/3.10.
# ---------------------------------------------------------------------------

def _toml_loads(text: str):
    """Return parsed TOML, or ``None`` when no parser is available."""
    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # declared in `dev` below Python 3.11
        except ModuleNotFoundError:
            return None
    return tomllib.loads(text)


def _pyproject_text() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def _requires_python() -> str:
    """``[project].requires-python``, parser or no parser.

    The regex fallback is deliberately narrow — one scalar string on one
    line — and the assertion below makes a silent miss impossible.
    """
    data = _toml_loads(_pyproject_text())
    if data is not None:
        return data["project"]["requires-python"]
    m = re.search(r'^\s*requires-python\s*=\s*"([^"]+)"',
                  _pyproject_text(), re.MULTILINE)
    assert m, "could not find requires-python in pyproject.toml"
    return m.group(1)


def _classifiers() -> list[str]:
    data = _toml_loads(_pyproject_text())
    if data is not None:
        return list(data["project"]["classifiers"])
    block = re.search(r"^classifiers\s*=\s*\[(.*?)^\]",
                      _pyproject_text(), re.MULTILINE | re.DOTALL)
    assert block, "could not find classifiers in pyproject.toml"
    found = re.findall(r'"([^"]+)"', block.group(1))
    assert found, "classifiers block parsed as empty"
    return found


def _dynamic_fields() -> list[str]:
    data = _toml_loads(_pyproject_text())
    if data is not None:
        return list(data["project"].get("dynamic", []))
    block = re.search(r"^dynamic\s*=\s*\[(.*?)^\]",
                      _pyproject_text(), re.MULTILINE | re.DOTALL)
    assert block, "could not find dynamic in pyproject.toml"
    return re.findall(r'"([^"]+)"', block.group(1))


def _setup_tree() -> ast.Module:
    return ast.parse(SETUP_PY.read_text(encoding="utf-8"))


def _setup_kwarg_node(name: str):
    """The AST node for one ``setup(...)`` keyword, or ``None``."""
    for node in ast.walk(_setup_tree()):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            for kw in node.keywords:
                if kw.arg == name:
                    return kw.value
    return None


def _literal_kwarg(name: str):
    """Pull one keyword out of the ``setup(...)`` call, without executing it.

    ``install_requires=dependencies`` passes a *name*, not a literal, so a
    bare ``literal_eval`` is not enough; module-level list/dict assignments
    are resolved by hand.
    """
    node = _setup_kwarg_node(name)
    if node is None:
        return None
    if isinstance(node, ast.Name):
        for stmt in _setup_tree().body:
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == node.id for t in stmt.targets
            ):
                return ast.literal_eval(stmt.value)
        pytest.fail(f"setup({name}=...) refers to {node.id!r}, which is not a "
                    f"module-level literal")
    return ast.literal_eval(node)


def _core_dependencies() -> list[str]:
    """The module-level ``dependencies = [...]`` list in setup.py."""
    for node in _setup_tree().body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "dependencies" in targets:
                return list(ast.literal_eval(node.value))
    pytest.fail("could not find the `dependencies` list in setup.py")


def _extras() -> dict[str, list[str]]:
    extras = _literal_kwarg("extras_require")
    assert extras, "setup(extras_require=...) not found"
    return {k: list(v) for k, v in extras.items()}


def _name_of(requirement: str) -> str:
    """PEP 503 normalised project name of a requirement string."""
    from packaging.requirements import Requirement
    return re.sub(r"[-_.]+", "-", Requirement(requirement).name).lower()


# ---------------------------------------------------------------------------
# 1. requires-python exists, and says what we can defend
# ---------------------------------------------------------------------------

def test_pyproject_declares_requires_python():
    """Without this, pip cannot decline early and the user meets a compiler."""
    spec = _requires_python()
    assert spec.strip(), "requires-python is empty"


def test_requires_python_admits_39_through_314_except_3141():
    """The supported range is evidence-bounded, in both directions.

    Floor 3.9: this is a supported interpreter in real use. Its resolver
    selects torch 2.8 and the last compatible PySide6, numba, llvmlite,
    pingouin and IPython lines; a blocking CI cell exercises that branch.

    Ceiling <3.15: every admitted minor has a blocking CI cell. Native
    dependencies without CPython 3.14 wheels are optional and lazily loaded.
    Python 3.14.1 is excluded because torchvision excludes that exact patch
    release in its own package metadata.

    This test used to be called
    ``test_requires_python_admits_310_through_312_and_nothing_else`` and it
    asserted the opposite of what it asserts now: that 3.13 was **not**
    admitted. That was correct while ``numpy>=1.26.4,<2.0`` admitted only
    1.26.4 (cp39-cp312 wheels) and while mahotas — which has never published a
    cp313 wheel at any version — was a core dependency. Both facts changed on
    2026-07-27: numpy is ``>=1.26.4,<3.0``, mahotas moved to the ``zernike``
    extra and torchcam (which declares ``numpy<2.0.0``) moved to
    ``attribution``. 3.13 therefore moved from the ``unsupported`` list to the
    ``supported`` one, and ``.github/workflows/compat-matrix.yml`` grew a real
    3.13 install cell in place of the job that asserted 3.13 was refused.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    spec = SpecifierSet(_requires_python())
    supported = ["3.9", "3.10", "3.11", "3.12", "3.13", "3.14"]
    unsupported = ["3.7", "3.8", "3.14.1", "3.15"]

    for v in supported:
        version = Version(v if v.count(".") == 2 else v + ".0")
        assert spec.contains(version), \
            f"requires-python {spec} excludes {v}, which CI runs and the classifiers claim"
    for v in unsupported:
        version = Version(v if v.count(".") == 2 else v + ".0")
        assert not spec.contains(version), \
            (f"requires-python {spec} admits {v}. If that is intentional, add the "
             f"classifier and the CI cell in the same commit — an untested claim "
             f"is how the 3.13 numpy compiler error reached users.")


def test_python_classifiers_match_requires_python_exactly():
    """A classifier is a claim of support. It may not outrun requires-python."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    spec = SpecifierSet(_requires_python())
    claimed = sorted(
        c.rsplit("::", 1)[1].strip()
        for c in _classifiers()
        if c.startswith("Programming Language :: Python :: ")
        and re.fullmatch(r"\s*3\.\d+\s*", c.rsplit("::", 1)[1])
    )
    admitted = sorted(
        v for v in (
            "3.7", "3.8", "3.9", "3.10", "3.11", "3.12", "3.13", "3.14",
        )
        if spec.contains(Version(v + ".0"))
    )
    assert claimed == admitted, (
        f"classifiers claim {claimed} but requires-python {spec} admits {admitted}"
    )


def test_no_os_independent_classifier():
    """It was never true, and Windows-on-ARM makes it demonstrably false."""
    assert "Operating System :: OS Independent" not in _classifiers(), (
        "`OS Independent` is back. torch has never published a win_arm64 wheel "
        "for any version, so Windows-on-ARM cannot install spaCR at all. Name "
        "the three operating systems CI exercises instead."
    )


def test_operating_system_classifiers_are_named_explicitly():
    cls = _classifiers()
    for expected in (
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ):
        assert expected in cls, f"missing OS classifier: {expected}"


# ---------------------------------------------------------------------------
# 2. Every claimed Python version has a CI cell
# ---------------------------------------------------------------------------

def test_every_claimed_python_version_has_a_ci_cell():
    """The rule that stops this file drifting back into fiction.

    Matching is textual (the version string appears somewhere in a workflow)
    rather than a real YAML parse, because pyyaml is not a spaCR dependency
    and a false *pass* here is cheap while a false *fail* would be noise. It
    still catches the case that matters: a classifier added with no cell.
    """
    workflow_text = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(WORKFLOWS.glob("*.yml"))
    )
    assert workflow_text, f"no workflows found under {WORKFLOWS}"

    claimed = [
        c.rsplit("::", 1)[1].strip()
        for c in _classifiers()
        if c.startswith("Programming Language :: Python :: ")
        and re.fullmatch(r"\s*3\.\d+\s*", c.rsplit("::", 1)[1])
    ]
    missing = [v for v in claimed if f'"{v}"' not in workflow_text]
    assert not missing, (
        f"classifiers claim Python {missing} but no CI workflow mentions "
        f"{missing}. Add the cell in the same commit as the claim."
    )


# ---------------------------------------------------------------------------
# 3. setup.py does nothing at import
# ---------------------------------------------------------------------------

_IMPORT_PROBE = textwrap.dedent(
    r"""
    import builtins, json, os, runpy, socket, subprocess, sys, urllib.request

    calls = []

    def _forbid(label):
        def _hook(*a, **k):
            calls.append({"call": label, "args": [repr(x)[:200] for x in a]})
            raise AssertionError(label + " called at setup.py import time")
        return _hook

    # Anything that would reach the network or the shell.
    subprocess.run = _forbid("subprocess.run")
    subprocess.call = _forbid("subprocess.call")
    subprocess.check_call = _forbid("subprocess.check_call")
    subprocess.check_output = _forbid("subprocess.check_output")
    subprocess.Popen = _forbid("subprocess.Popen")
    os.system = _forbid("os.system")
    os.execv = _forbid("os.execv")
    urllib.request.urlopen = _forbid("urllib.request.urlopen")
    socket.socket.connect = _forbid("socket.connect")
    socket.create_connection = _forbid("socket.create_connection")

    # setup() itself must not run a build; capture its keywords instead.
    import setuptools
    captured = {}

    def _fake_setup(**kwargs):
        captured.update(kwargs)

    setuptools.setup = _fake_setup

    err = None
    try:
        runpy.run_path(sys.argv[1], run_name="__main__")
    except BaseException as exc:            # noqa: BLE001 - reported, not raised
        err = "%s: %s" % (type(exc).__name__, exc)

    print("<<<PROBE>>>" + json.dumps({
        "calls": calls,
        "error": err,
        "keywords": sorted(captured),
        "extras": sorted(captured.get("extras_require", {})),
    }))
    """
)


def _run_import_probe():
    # The probe lives outside the repo so it can never be collected by
    # pytest, never be picked up by find_packages(), and never survive a
    # crashed run as an untracked file in someone's working copy.
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "setup_import_probe.py"
        probe.write_text(_IMPORT_PROBE, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(probe), str(SETUP_PY)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
        )
    marker = "<<<PROBE>>>"
    assert marker in proc.stdout, (
        "probe produced no result.\nstdout:\n" + proc.stdout
        + "\nstderr:\n" + proc.stderr
    )
    return json.loads(proc.stdout.split(marker, 1)[1].splitlines()[0])


def test_setup_py_performs_no_subprocess_or_network_work_at_import():
    """Executing setup.py must describe the package and nothing else.

    This is the regression that broke PEP 517 isolated builds and every
    offline install: a ``for dep in deps: subprocess.run(['pip', 'install',
    dep])`` loop at module scope, after the ``setup()`` call.
    """
    result = _run_import_probe()
    assert result["calls"] == [], (
        "setup.py performed shell/network work at import: "
        f"{result['calls']}"
    )
    assert result["error"] is None, (
        f"executing setup.py raised: {result['error']}"
    )
    assert "install_requires" in result["keywords"]
    assert "entry_points" in result["keywords"]


def test_setup_py_does_not_import_subprocess():
    """Cheap static twin of the test above — fails on the *intent*, not just
    the effect, so a rewritten shell-out with a different call name is still
    caught."""
    tree = _setup_tree()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("subprocess", "urllib", "socket", "requests", "pip"):
        assert banned not in imported, (
            f"setup.py imports {banned!r}; a setup script must only describe "
            f"the package"
        )


def test_setup_py_contains_no_pip_install_shellout():
    """Comment lines are stripped first: the file legitimately *documents*
    `pip install spacr[czi]` in prose, and describing an install command is
    not the same as running one."""
    src = SETUP_PY.read_text(encoding="utf-8")
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    for pattern in ("'pip'", '"pip"', "pip install", "python -m pip"):
        assert pattern not in code, (
            f"setup.py invokes an installer again ({pattern!r}). That is what "
            f"broke PEP 517 isolated builds and every offline install."
        )


# ---------------------------------------------------------------------------
# 4. The unused Qt binding stays gone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("banned", ["pyqt6", "pyqtgraph", "qtpy", "superqt"])
def test_no_second_qt_binding_is_declared(banned):
    """spaCR uses PySide6 in 75 files and none of these in any file.

    The old subprocess loop installed all four (plus pyqt6.sip), which is
    ~100 MB of second Qt binding, an ABI hazard next to PySide6, and a
    guaranteed failure on any platform where PyQt6 has no wheel.

    Extended on 2026-07-27 to cover ``requirements.txt`` as well. Until then
    this test guarded ``setup.py`` only — and the whole banned block was
    sitting in ``requirements.txt`` the entire time, untouched, because
    nothing read it. Anyone who ran ``pip install -r requirements.txt``
    (a normal thing to do, and what several CI templates do by default) got
    pyqtgraph, pyqt6, pyqt6.sip, qtpy and superqt installed next to PySide6.
    A guard that checks one of the two files that can declare a dependency is
    a guard that has a hole in it.
    """
    specs = list(_core_dependencies())
    for extra in _extras().values():
        specs.extend(extra)
    offenders = [s for s in specs if _name_of(s).startswith(banned)]
    assert not offenders, f"{banned} re-declared: {offenders}"

    for path in (SETUP_PY, REQUIREMENTS_TXT):
        if not path.exists():
            continue
        code = "\n".join(
            ln for ln in path.read_text(encoding="utf-8").splitlines()
            if not ln.lstrip().startswith("#")
        )
        assert banned not in code.lower(), (
            f"{banned} re-appeared in {path.name}"
        )


def test_requirements_txt_does_not_contradict_setup_py():
    """``requirements.txt`` must not hand-copy the dependency list.

    It used to, and the copy had gone stale in the worst possible way:
    ``cellpose>=3.0.6,<4.0`` where setup.py declares ``cellpose>=4.0,<5.0``.
    Those two cannot both be satisfied, so ``pip install -r requirements.txt``
    built an environment in which spaCR's own cellpose code cannot run, and
    installing from both files was an unsatisfiable resolve. Fourteen other
    bounds were stale in the same direction, and eight named packages that are
    now removed or moved to extras.

    The fix is structural rather than clerical: requirements.txt delegates to
    setup.py with ``-e .[qt,dev]``, so there is exactly one dependency list in
    the repository. This test keeps it that way — any line that pins a version
    is a hand-written dependency creeping back in.
    """
    if not REQUIREMENTS_TXT.exists():
        pytest.skip("no requirements.txt")

    lines = [
        ln.strip() for ln in REQUIREMENTS_TXT.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert lines, "requirements.txt declares nothing at all"

    pinned = [ln for ln in lines if re.search(r"[<>=!~]=?\s*\d", ln)]
    assert not pinned, (
        "requirements.txt pins versions again: "
        f"{pinned}\nsetup.py's `dependencies` list is the single source of "
        "truth. A second copy drifts — last time it drifted to "
        "`cellpose<4.0` against setup.py's `cellpose>=4.0`, which is an "
        "unsatisfiable pair. Delegate with `-e .[qt,dev]` instead."
    )
    assert any(ln.startswith("-e") or ln == "." for ln in lines), (
        "requirements.txt no longer delegates to setup.py. It should contain "
        "`-e .[qt,dev]` so the two files cannot disagree."
    )


def test_attribution_extra_is_not_in_all():
    """``spacr[all]`` must stay installable on every supported Python.

    torchcam declares ``numpy<2.0.0`` at every release satisfying spaCR's pin,
    and no numpy satisfying that publishes a cp313 wheel. Measured in a
    throwaway CPython 3.13.14 env on 2026-07-27, ``pip install torchcam`` is a
    hard ``ResolutionImpossible``, not a slow build. Aggregating it into
    ``all`` would therefore break ``pip install spacr[all]`` on Python 3.13
    outright — reintroducing, through the extra most likely to be typed by
    someone who just wants everything, the failure that moving torchcam out of
    the core dependencies removed.
    """
    extras = _extras()
    assert "attribution" in extras, "the `attribution` extra disappeared"
    assert any(_name_of(s) == "torchcam" for s in extras["attribution"]), \
        "the `attribution` extra no longer provides torchcam"
    in_all = [s for s in extras.get("all", []) if _name_of(s) == "torchcam"]
    assert not in_all, (
        f"torchcam is in `all` ({in_all}). On Python 3.13 — which "
        "requires-python and the classifiers both claim — that makes "
        "`pip install spacr[all]` backtrack into a SOURCE BUILD of numpy "
        "1.26.4 (measured: pip reports `Would install ... numpy-1.26.4 "
        "torchcam-0.4.1`), which is the failure moving torchcam out of the "
        "core dependencies removed. Keep it in `attribution` only."
    )


def test_the_python_313_limits_of_the_extras_are_written_down():
    """Two extras cannot be installed on Python 3.13, and both are upstream.

    This test does not assert the limitation — it asserts that the limitation
    is *documented next to the pin*, because the failure mode it guards is
    somebody re-deriving it from a confusing pip error six months from now.
    Both were measured in a throwaway CPython 3.13.14 env on 2026-07-27:

    * ``ultrack`` — every release from 0.1.0 through 0.7.2 declares
      ``requires-python >=3.9,<3.13``. pip refuses cleanly. This is what makes
      ``spacr[all]`` uninstallable on 3.13, *not* torchcam.
    * ``torchcam`` — declares ``numpy<2.0.0``, which has no cp313 wheel, so a
      plain install backtracks into a numpy 1.26.4 source build.

    Neither is a spaCR bug and neither blocks ``pip install spacr``, which is
    the promise that matters and which resolves entirely to wheels on 3.13.
    """
    src = SETUP_PY.read_text(encoding="utf-8")
    for needle, why in (
        ("requires-python >=3.9,<3.13",
         "ultrack's own Python ceiling, which is what stops spacr[all] on 3.13"),
        ("numpy<2.0.0",
         "torchcam's spurious numpy pin, the reason it is an extra at all"),
    ):
        assert needle in src, (
            f"setup.py no longer documents {needle!r} ({why}). If the upstream "
            f"limitation is gone, say so and widen the extra; do not just "
            f"delete the note — the next person will hit the same pip error "
            f"with no explanation."
        )


@pytest.mark.parametrize(
    "banned, why",
    [
        ("segment-anything",
         "PyPI's segment-anything has one release (1.0, 2023-04-06) with empty "
         "author/homepage/summary; Meta never published SAM to PyPI. spacr "
         "imports it in zero files and cellpose depends on segment_anything "
         "itself."),
        ("aicspylibczi",
         "zero imports and zero raw-string references under spacr/; ships a "
         "manylinux x86_64 wheel only and its sdist needs CMake + libCZI "
         "headers, so it was the one dependency forcing a C++ source build on "
         "ARM Linux for a package spacr never imports."),
    ],
)
def test_removed_dependencies_stay_removed(banned, why):
    specs = list(_core_dependencies())
    for extra in _extras().values():
        specs.extend(extra)
    offenders = [s for s in specs if _name_of(s) == banned]
    assert not offenders, f"{banned} is back ({offenders}). It was removed because: {why}"


# ---------------------------------------------------------------------------
# 5. Dependencies and extras are well-formed and mutually consistent
# ---------------------------------------------------------------------------

def test_every_core_dependency_is_a_valid_requirement():
    from packaging.requirements import InvalidRequirement, Requirement
    for spec in _core_dependencies():
        try:
            Requirement(spec)
        except InvalidRequirement as exc:
            pytest.fail(f"invalid requirement {spec!r}: {exc}")


def test_no_duplicate_core_dependencies():
    """The deleted block declared pyqtgraph, pyqt6, pyqt6.sip, qtpy and
    superqt twice each. Duplication is how that went unnoticed."""
    seen: dict[str, str] = {}
    dupes = []
    for spec in _core_dependencies():
        n = _name_of(spec)
        if n in seen:
            dupes.append((seen[n], spec))
        seen[n] = spec
    assert not dupes, f"duplicate core dependencies: {dupes}"


def test_native_features_not_qualified_on_python_314_are_optional():
    """Core imports must not require native features outside the 3.14 profile."""
    core = {_name_of(spec) for spec in _core_dependencies()}
    extras = _extras()
    for package, extra in (("pylibczirw", "czi"), ("btrack", "btrack")):
        assert package not in core, (
            f"{package} is a core dependency again; that blocks Python 3.14"
        )
        assert package in {_name_of(spec) for spec in extras[extra]}, (
            f"{extra!r} no longer provides its optional {package} dependency"
        )


def test_every_extra_is_non_empty_and_well_formed():
    from packaging.requirements import InvalidRequirement, Requirement
    extras = _extras()
    assert extras, "no extras declared"
    for name, specs in extras.items():
        assert re.fullmatch(r"[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?", name), \
            f"extra name {name!r} is not a valid PEP 685 extra"
        assert specs, f"extra {name!r} is empty; an empty extra is a lie in the metadata"
        for spec in specs:
            try:
                Requirement(spec)
            except InvalidRequirement as exc:
                pytest.fail(f"extra {name!r}: invalid requirement {spec!r}: {exc}")


def test_extras_do_not_contradict_the_core_pins():
    """A package may appear in both core and an extra — several do, on
    purpose, because spacr/io.py and spacr/measure.py still import the format
    readers at module scope — but the two copies must be the *same* pin.
    Two different pins for one package is a resolver conflict waiting for the
    day the core copy is removed.
    """
    from packaging.requirements import Requirement

    def _shape(spec):
        """Compare what pip compares, not the whitespace: 'a>=1, <2' and
        'a>=1,<2' are the same pin and must not be reported as a conflict."""
        r = Requirement(spec)
        return (set(str(s) for s in r.specifier), r.marker is None or str(r.marker),
                sorted(r.extras))

    core = {_name_of(s): s for s in _core_dependencies()}
    problems = []
    for extra, specs in _extras().items():
        for spec in specs:
            n = _name_of(spec)
            if n in core and _shape(core[n]) != _shape(spec):
                problems.append((extra, spec, core[n]))
    assert not problems, (
        "extra and core declare different pins for the same package: " + str(problems)
    )


def test_all_extra_is_exactly_the_union_of_what_it_aggregates():
    """``spacr[all]`` is spelled out rather than recursive, so it can drift.
    This is the thing that stops it.

    ``boosting`` joined the aggregation on 2026-07-27, when catboost and
    lightgbm were declared for the first time (both are imported inside the
    ``elif`` that selects them, in spacr/ml.py and spacr/hyperparam.py, and
    neither was declared anywhere).

    ``attribution`` is deliberately NOT aggregated, and that is the
    interesting entry. torchcam declares ``numpy<2.0.0``; no numpy satisfying
    that has a cp313 wheel; so putting it in ``all`` would make
    ``pip install spacr[all]`` a hard ResolutionImpossible on Python 3.13 —
    reintroducing, through the extra most likely to be typed by someone who
    just wants everything, exactly the failure that moving torchcam out of the
    core dependencies removed. See ``test_attribution_extra_is_not_in_all``.
    """
    extras = _extras()
    assert "all" in extras, "the `all` extra disappeared"
    aggregated = ("qt", "tutorial", "trackastra", "ultrack", "boosting",
                  "czi", "nd2", "lif", "zernike", "btrack")
    expected = set()
    for name in aggregated:
        assert name in extras, f"`all` claims to aggregate {name!r}, which is gone"
        expected.update(extras[name])
    assert set(extras["all"]) == expected, (
        "spacr[all] drifted from the union of "
        f"{aggregated}.\n  only in all: {sorted(set(extras['all']) - expected)}"
        f"\n  missing from all: {sorted(expected - set(extras['all']))}"
    )


def test_format_reader_extras_exist():
    """The Wave-3 boundary, declared in advance.

    ``mahotas`` is the load-bearing one: 1.4.18 publishes cp310-cp312 with
    manylinux **x86_64 only**, has never published a cp313 or cp314 wheel at
    any version, and last published aarch64 in 1.4.13. It is what caps the
    Python ceiling next to numpy, and what forces a C++ toolchain on ARM
    Linux. ``pylibCZIrw`` blocks only 3.14 (5.1.1 already ships cp313 plus
    aarch64 and macOS arm64); czifile, readlif and nd2reader are pure Python
    and block nothing.

    pylibCZIrw is now optional and loaded only by the high-performance CZI
    conversion path, which keeps core installation viable on Python 3.14.
    The remaining reader extras preserve explicit feature installs.
    """
    extras = _extras()
    for name in ("czi", "nd2", "lif", "zernike"):
        assert name in extras, f"format extra {name!r} is missing"


# ---------------------------------------------------------------------------
# 6. The pyproject.toml / setup.py split keeps working
# ---------------------------------------------------------------------------

def test_dynamic_covers_every_field_setup_py_supplies():
    """If a field is supplied by setup.py but not listed in ``dynamic``,
    setuptools drops it from the built metadata — silently. Losing
    ``entry-points`` this way would ship a package with no ``spacr`` command
    and a green build."""
    dynamic = set(_dynamic_fields())
    required = {
        "version", "description", "readme",
        "dependencies", "optional-dependencies",
        "entry-points", "scripts",
    }
    missing = required - dynamic
    assert not missing, (
        f"pyproject.toml [project].dynamic is missing {sorted(missing)}, but "
        f"setup.py still supplies them"
    )


def test_pyproject_does_not_statically_declare_dynamic_fields():
    """A field cannot be both static and dynamic; setuptools errors out."""
    data = _toml_loads(_pyproject_text())
    if data is None:
        pytest.skip(
            "no TOML parser available (install tomli on Python 3.9/3.10)")
    project = data["project"]
    for field in _dynamic_fields():
        assert field not in project, (
            f"{field!r} is listed in `dynamic` and also declared statically"
        )


def test_build_backend_declares_a_setuptools_floor():
    """setuptools < 61 ignores ``[project]`` entirely — it would drop
    requires-python from the built metadata rather than fail loudly."""
    data = _toml_loads(_pyproject_text())
    if data is None:
        text = _pyproject_text()
        assert re.search(r'setuptools\s*>=\s*\d+', text), \
            "build-system.requires does not pin a setuptools floor"
        return
    reqs = data["build-system"]["requires"]
    from packaging.requirements import Requirement
    floors = [
        s for r in reqs
        for s in Requirement(r).specifier
        if Requirement(r).name == "setuptools" and s.operator in (">=", "==", ">")
    ]
    assert floors, f"build-system.requires={reqs} pins no setuptools floor"


def test_setup_py_still_exposes_what_the_packaging_scripts_parse():
    """Guards the *next* refactor, not this one.

    packaging/build_macos.sh, packaging/build_windows.ps1 and
    packaging/build_debian.sh all regex ``VERSION = "..."`` straight out of
    setup.py, and build_debian.sh then runs ``python3 setup.py`` under stdeb.
    Four tests ast-parse or regex other parts of it. Finishing the move to
    pyproject.toml means updating those first — not discovering it from a
    broken installer build.
    """
    src = SETUP_PY.read_text(encoding="utf-8")
    assert re.search(r"^VERSION\s*=\s*['\"][^'\"]+['\"]", src, re.MULTILINE), \
        "packaging/build_{macos,windows,debian} regex this line out of setup.py"
    assert _literal_kwarg("entry_points"), \
        "tests/test_gui_dispatch_call_styles.py ast-parses entry_points here"
    assert _literal_kwarg("extras_require"), \
        "tests/test_timelapse_ultrack.py regexes extras_require['ultrack'] here"
    assert _literal_kwarg("install_requires") is not None


def test_console_scripts_and_extras_agree_about_qt():
    """`spacr` is the default console script and it launches the Qt GUI, but
    PySide6 lives in the `qt` extra, not the core deps — so a plain
    `pip install spacr` installs a `spacr` command that raises ImportError.

    This test does not assert the fix (moving PySide6 into the core deps
    would force ~150 MB of Qt onto headless cluster users who only ever run
    `spacr-run`, and the better fix is a friendly error inside
    ``spacr/qt/__init__.py``). It asserts the two halves stay *visible* to
    each other, so the trade-off is made on purpose rather than by accident.
    """
    entry_points = _literal_kwarg("entry_points") or {}
    scripts = entry_points.get("console_scripts", [])
    qt_scripts = [s for s in scripts if "spacr.qt" in s]
    assert qt_scripts, "no Qt console script found; did the entry points move?"

    core = {_name_of(s) for s in _core_dependencies()}
    extras = _extras()
    if "pyside6" not in core:
        assert "pyside6" in {_name_of(s) for s in extras.get("qt", [])}, (
            "PySide6 is in neither the core dependencies nor the `qt` extra, "
            f"yet these console scripts launch the Qt GUI: {qt_scripts}"
        )
