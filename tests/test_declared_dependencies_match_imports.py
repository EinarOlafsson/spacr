"""Every third-party import spaCR makes must be a dependency spaCR declares.

This is the census that produced the 2026-07-27 dependency change, kept as a
test so it cannot rot back. It runs it in both directions:

  * **imported but not declared.** Five module-scope imports were undeclared
    and worked only because something else dragged them in — ``requests`` via
    huggingface-hub, ``joblib`` via scikit-learn, ``natsort`` via cellpose,
    ``patsy`` via statsmodels, ``sympy`` via torch. Each was one upstream
    dependency change away from an ``ImportError`` at ``import spacr.utils``,
    and huggingface-hub in particular is *actively* moving off requests (hub
    1.x replaces it with httpx).

  * **declared but not imported.** 26 distributions had zero imports. Eight of
    those are legitimately indirect and are documented as such in setup.py
    (bottleneck, numexpr, openpyxl, fastremap, tqdm, protobuf, lxml, tables);
    the other 18 were removed.

The method matters as much as the result, so it is written out rather than
hand-waved:

  1. ``ast.parse`` every ``.py`` under ``spacr/``, then ``ast.walk`` each tree.
     Walking rather than iterating ``tree.body`` is the point — it counts
     imports inside functions, methods, ``try``/``except`` bodies, ``if
     TYPE_CHECKING`` blocks and class bodies, all of which spaCR uses.
  2. Resolve each top-level module name to the distribution that provides it
     through the explicit table below, rather than by guessing from the name.
     Guessing gets ``cv2``, ``PIL``, ``Bio``, ``sklearn``, ``skimage``,
     ``GPUtil`` and ``umap`` wrong, which is six of the most important entries.

**The census has one blind spot, and it is load-bearing.** A module imported
through a *string literal* is invisible to it. spaCR has two such cases, and
both are listed in :data:`STRING_LITERAL_ONLY` so that a distribution showing
zero imports can be told apart from one that is genuinely unused:

  * ``umap = _LazyModule('umap.umap_', block_roots=_TF_BACKED_ROOTS)`` at
    ``spacr/utils.py:197``. ``umap-learn`` therefore shows zero imports and is
    nonetheless a real core dependency used at three call sites. There is a
    test below that pins that specific fact in place, so the next person to
    run a census and see "umap-learn: unused" finds the answer instead of
    deleting it.
  * ``importlib.import_module(OMERO_GATEWAY_MODULE)`` in ``spacr/omero.py``,
    where ``OMERO_GATEWAY_MODULE = "omero.gateway"``. That file is *itself*
    called ``omero.py``, so a bare ``import omero.gateway`` inside it reads
    like a self-import to everyone who meets it (it is not — absolute imports
    are the default — but "it is not what it looks like" is a poor thing to
    rely on). Holding the name in one constant next to the guard that checks
    what came back is worth the blind spot. Pinned from the other side by
    ``tests/test_omero.py::
    test_omero_py_is_reached_through_a_string_literal_and_must_not_be_removed``.

Both are in :data:`IMPORT_TO_DIST` even though nothing imports them by
statement, so that if either ever becomes a literal the name resolves to the
right distribution (``umap-learn``, ``omero-py``) rather than to one that does
not exist on PyPI.

Nothing here imports :mod:`spacr` or any scientific package — it is AST and
text only, so it runs in the metadata CI job that deliberately has no stack
installed.
"""
from __future__ import annotations

import ast
import os
import pkgutil
import re
import sys
import sysconfig
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG = REPO_ROOT / "spacr"
SETUP_PY = REPO_ROOT / "setup.py"


# ---------------------------------------------------------------------------
# What counts as the standard library.
#
# `sys.stdlib_module_names` arrived in CPython 3.10. spaCR claims 3.9, and on
# 3.9 every test in this file died with
# `AttributeError: module 'sys' has no attribute 'stdlib_module_names'` --
# collected, run, and erroring, on both the 3.9 matrix cell and the
# minimum-dependencies job. The census that proves every import is declared
# was therefore not running on the one interpreter where an undeclared import
# is most likely to bite, and the AttributeError read like an infrastructure
# problem rather than a missing-dependency one.
#
# The 3.9 fallback ENUMERATES the interpreter's own stdlib rather than
# carrying a hand-written list that would rot: `sysconfig`'s stdlib path plus
# its `lib-dynload` (where the C extension modules live), scanned with
# `pkgutil.iter_modules`, plus the modules compiled into the binary. Third
# party code lives in `site-packages`, a subdirectory with no `__init__.py`,
# so it is not reported by that scan.
# ---------------------------------------------------------------------------

#: Stdlib modules a scan cannot see because they ship only on one OS. The
#: fallback below enumerates THIS interpreter, so on Linux it misses the
#: Windows-only names and vice versa; without them a `import winreg` in a
#: platform branch would be reported as an undeclared third-party import on
#: the 3.9 cell, which runs on Linux. Measured, not assumed: diffing the scan
#: against `sys.stdlib_module_names` on 3.10/Linux, the Windows five below are
#: the ENTIRE shortfall. The POSIX five are their mirror, listed so a Windows
#: run of this file is not the one that discovers the asymmetry.
_PLATFORM_ONLY_STDLIB = frozenset({
    "msilib", "msvcrt", "nt", "winreg", "winsound",    # Windows
    "fcntl", "grp", "posix", "pwd", "termios",         # POSIX
})


def _stdlib_module_names() -> frozenset[str]:
    names = getattr(sys, "stdlib_module_names", None)
    if names is not None:                       # CPython 3.10+
        return frozenset(names)

    stdlib = sysconfig.get_paths().get("stdlib")
    search = [p for p in (stdlib, os.path.join(stdlib or "", "lib-dynload"))
              if p and os.path.isdir(p)]
    found = {module.name for module in pkgutil.iter_modules(search)}
    found |= set(sys.builtin_module_names)
    found |= _PLATFORM_ONLY_STDLIB
    # `pkgutil` reports what is importable, which on 3.9 includes a handful
    # of private bootstrap names; harmless here, since this set is only ever
    # used to EXCLUDE names from the third-party census.
    return frozenset(found)


STDLIB_MODULE_NAMES = _stdlib_module_names()


# ---------------------------------------------------------------------------
# import name -> distribution name.
#
# Only entries where the two differ, plus every entry that a reader would
# otherwise have to look up. `foo -> foo` is left implicit.
# ---------------------------------------------------------------------------
IMPORT_TO_DIST = {
    "Bio": "biopython",
    "GPUtil": "gputil",
    "PIL": "pillow",
    "PySide6": "PySide6",
    "cv2": "opencv-python-headless",
    "cuml": "cuml-cu12",
    "cupy": "cupy-cuda12x",
    # mpl_toolkits ships INSIDE matplotlib -- there is no `mpl-toolkits` on
    # PyPI to depend on, so without this the check asks for a distribution
    # that cannot be installed.
    "mpl_toolkits": "matplotlib",
    "huggingface_hub": "huggingface-hub",
    # No import statement anywhere -- see STRING_LITERAL_ONLY. Here so that the
    # day one is written, `omero` resolves to `omero-py` (which the `omero`
    # extra declares) rather than to a distribution of that name, which is a
    # different, unrelated package.
    "omero": "omero-py",
    "pynvml": "nvidia-ml-py",
    "scikit_posthocs": "scikit-posthocs",
    "shiboken6": "PySide6",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
}

# ---------------------------------------------------------------------------
# Directories excluded from the census, with the reason.
#
# `**/_generators/` holds the developer-only scripts that draw spaCR's icons
# and home-screen art. They live under spacr/resources/ so the assets and the
# code that produced them stay together, but they are *run directly*
# (`python _draw.py`), never imported as part of the package: they import
# their siblings by bare filename (`import common`, `from _draw import ...`),
# which is only valid when the script's own directory is sys.path[0]. Counting
# them would demand a `common` distribution on PyPI, and would also require
# PySide6 in the core dependencies for art nobody regenerates at runtime.
# ---------------------------------------------------------------------------
EXCLUDED_DIRS = ("_generators",)

# Subpackages whose module-scope imports are satisfied by an extra ON PURPOSE.
#
# `spacr/qt/` is the PySide6 GUI. `import spacr` does not import it, the
# `spacr-run` headless CLI never touches it, and PySide6 is ~150 MB that
# cluster users have no use for — so PySide6 lives in the `qt` extra and the
# GUI subpackage is unimportable without it by design. That trade-off is
# asserted separately, and deliberately, by
# tests/test_packaging_metadata.py::test_console_scripts_and_extras_agree_about_qt.
EXTRA_GATED_SUBPACKAGES = {"spacr/qt/": "qt"}

#: Distributions spaCR reaches only through a string literal, so no import
#: statement exists for the table below to be checked against. See
#: `test_umap_is_reached_through_a_string_literal_and_must_not_be_removed`,
#: and `tests/test_omero.py` for the matching pin on the other one. The module
#: docstring says why each is written that way; the list is short on purpose,
#: because every entry is a hole in this file's guarantee.
STRING_LITERAL_ONLY = {"umap", "omero"}


def _is_censused(rel_path: str) -> bool:
    return not any(f"/{d}/" in f"/{rel_path}" for d in EXCLUDED_DIRS)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _setup_tree() -> ast.Module:
    return ast.parse(SETUP_PY.read_text(encoding="utf-8"))


def _core_dependencies() -> list[str]:
    for node in _setup_tree().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "dependencies" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    pytest.fail("could not find the `dependencies` list in setup.py")


def _extras() -> dict[str, list[str]]:
    for node in ast.walk(_setup_tree()):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            for kw in node.keywords:
                if kw.arg == "extras_require":
                    return {k: list(v) for k, v in ast.literal_eval(kw.value).items()}
    pytest.fail("setup(extras_require=...) not found")


def _declared_names() -> set[str]:
    """Every distribution declared anywhere in setup.py, normalised."""
    from packaging.requirements import Requirement

    specs = list(_core_dependencies())
    for extra in _extras().values():
        specs.extend(extra)
    return {_norm(Requirement(s).name) for s in specs}


def _imports() -> dict[str, set[str]]:
    """top-level module name -> set of repo-relative files importing it.

    ``ast.walk`` rather than ``tree.body``: an import inside a function or a
    ``try`` block is still an import, and spaCR defers many of them on purpose.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(PKG.rglob("*.py")):
        rel = str(path.relative_to(REPO_ROOT))
        if not _is_censused(rel):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"),
                         filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.setdefault(alias.name.split(".")[0], set()).add(rel)
            elif isinstance(node, ast.ImportFrom):
                if node.level:          # relative import: spaCR's own package
                    continue
                if node.module:
                    found.setdefault(node.module.split(".")[0], set()).add(rel)
    return found


def _third_party_imports() -> dict[str, set[str]]:
    return {
        mod: files for mod, files in _imports().items()
        if mod not in STDLIB_MODULE_NAMES and mod != "spacr"
    }


# ---------------------------------------------------------------------------
# 1. Nothing is imported that is not declared
# ---------------------------------------------------------------------------

def test_every_third_party_import_is_a_declared_dependency():
    """The direction that produces ImportError in the field.

    An undeclared import does not fail on the developer's machine — that is
    exactly what makes it dangerous. It fails on a fresh install, months
    later, when the package that happened to bring it along drops it.
    """
    declared = _declared_names()
    undeclared = {}
    for mod, files in sorted(_third_party_imports().items()):
        dist = _norm(IMPORT_TO_DIST.get(mod, mod))
        if dist not in declared:
            undeclared[mod] = (dist, sorted(files)[:4])

    assert not undeclared, (
        "imported under spacr/ but declared nowhere in setup.py:\n"
        + "\n".join(f"  import {mod!r} -> would need {dist!r}, e.g. {files}"
                    for mod, (dist, files) in undeclared.items())
        + "\n\nAdd it to `dependencies` (if the import is module scope) or to "
          "an extra (if it is function-local and already guarded with an "
          "actionable ImportError). If the import name differs from the "
          "distribution name, add the mapping to IMPORT_TO_DIST in this file."
    )


def test_the_import_to_dist_table_has_no_dead_entries():
    """A mapping for something nothing imports is stale documentation.

    ``STRING_LITERAL_ONLY`` is exempt: ``umap`` is kept in the table precisely
    *because* there is no import statement, so that if one is ever written the
    name resolves to ``umap-learn`` rather than to a non-existent ``umap``
    distribution.
    """
    imported = set(_third_party_imports())
    dead = sorted(set(IMPORT_TO_DIST) - imported - STRING_LITERAL_ONLY)
    assert not dead, (
        f"IMPORT_TO_DIST maps {dead}, which nothing under spacr/ imports any "
        f"more. Remove the entries, and check whether the distribution should "
        f"leave setup.py too."
    )


# ---------------------------------------------------------------------------
# 2. The module-scope subset, which is the strict one
# ---------------------------------------------------------------------------

def _module_scope_imports() -> dict[str, set[str]]:
    """Only imports at module top level — the ones that make `import spacr.x`
    fail rather than degrading a single feature."""
    found: dict[str, set[str]] = {}
    for path in sorted(PKG.rglob("*.py")):
        rel = str(path.relative_to(REPO_ROOT))
        if not _is_censused(rel):
            continue
        if any(rel.startswith(p) for p in EXTRA_GATED_SUBPACKAGES):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"),
                         filename=str(path))
        for node in tree.body:                       # top level only
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.setdefault(alias.name.split(".")[0], set()).add(rel)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                found.setdefault(node.module.split(".")[0], set()).add(rel)
    return found


def test_every_module_scope_import_is_a_CORE_dependency_not_an_extra():
    """An extra cannot satisfy a module-scope import.

    If ``spacr/foo.py`` imports ``bar`` at module scope and ``bar`` lives in an
    extra, then ``import spacr.foo`` is an ImportError for everyone who did not
    type the extra — which is a broken install presented as an optional
    feature.

    Mahotas used to be the sole temporary offender after it moved to the
    ``zernike`` extra. Its import now lives inside the feature boundary, so no
    exemption remains: any extra-only module-scope import is an immediate
    packaging error.
    """
    KNOWN_PENDING = {}

    from packaging.requirements import Requirement

    core = {_norm(Requirement(s).name) for s in _core_dependencies()}
    extras = _extras()
    extra_only: dict[str, str] = {}
    for name, specs in extras.items():
        for spec in specs:
            dist = _norm(Requirement(spec).name)
            if dist not in core:
                extra_only.setdefault(dist, name)

    offenders = {}
    for mod, files in sorted(_module_scope_imports().items()):
        if mod in STDLIB_MODULE_NAMES or mod == "spacr":
            continue
        dist = _norm(IMPORT_TO_DIST.get(mod, mod))
        if dist in extra_only and dist not in KNOWN_PENDING:
            offenders[dist] = (extra_only[dist], sorted(files)[:4])

    assert not offenders, (
        "imported at MODULE SCOPE but only declared in an extra:\n"
        + "\n".join(f"  {dist!r} (extra {extra!r}) imported by {files}"
                    for dist, (extra, files) in offenders.items())
        + "\n\nEither move it back into `dependencies`, or make the import "
          "lazy/guarded with an actionable `pip install spacr[<extra>]` "
          "message the way spacr/timelapse.py does for trackastra and ultrack."
    )

    # The exemption may not outlive its reason.
    still_pending = {
        d for d in KNOWN_PENDING
        if d in extra_only
        and any(_norm(IMPORT_TO_DIST.get(m, m)) == d
                for m in _module_scope_imports())
    }
    resolved = sorted(set(KNOWN_PENDING) - still_pending)
    assert not resolved, (
        f"{resolved} no longer needs its exemption in this test — the "
        f"module-scope import is gone or the package is back in the core "
        f"dependencies. Delete the KNOWN_PENDING entry."
    )


# ---------------------------------------------------------------------------
# 3. The blind spot, pinned so nobody re-derives it the hard way
# ---------------------------------------------------------------------------

def test_umap_is_reached_through_a_string_literal_and_must_not_be_removed():
    """``umap-learn`` has zero import statements and is a real dependency.

    This is the one case where "the census says unused" is wrong, and it is
    wrong for a reason worth preserving: importing umap eagerly costs roughly
    6.5 s and 1.4 GB *per worker process* (every field-measuring worker of a
    spawn/forkserver pool re-imports the chain from a cold interpreter), and
    ``umap/__init__.py`` reaches TensorFlow through ``parametric_umap``, which
    spaCR refuses outright. So spaCR loads it by name, on first attribute
    access, with the TF-backed roots blocked.

    If this test fails because the ``_LazyModule`` line moved, do not delete
    the dependency — find the new call site and update the test.
    """
    utils = (PKG / "utils.py").read_text(encoding="utf-8")
    assert re.search(r"_LazyModule\(\s*['\"]umap\.umap_['\"]", utils), (
        "the `_LazyModule('umap.umap_', ...)` indirection at spacr/utils.py "
        "is gone. umap-learn has NO plain import statement anywhere in spaCR, "
        "so if this is now a real `import umap`, remove this test; if umap is "
        "genuinely unused, remove the dependency. Do not leave it ambiguous — "
        "the next dependency census will read a zero and act on it."
    )
    assert _norm("umap-learn") in {
        _norm(re.split(r"[<>=!~ ,\[;]", d.strip())[0]) for d in _core_dependencies()
    }, "umap-learn left the core dependencies while spacr/utils.py still loads it"


# ---------------------------------------------------------------------------
# 4. The kept-but-unimported eight, with their reasons
# ---------------------------------------------------------------------------

#: Declared, never imported, and correct to keep. The value is the reason,
#: which must also appear as a comment next to the pin in setup.py.
INDIRECT_BUT_REQUIRED = {
    "bottleneck": "pandas' nan-aware reduction accelerator; picked up by presence",
    "numexpr": "pandas' pd.eval / DataFrame.query backend; picked up by presence",
    "openpyxl": "the engine pandas needs for pd.read_excel (plot.py, foreign.py)",
    "fastremap": "imported by cellpose itself, and by fill_voids underneath it",
    "tqdm": "spacr/cli.py sets TQDM_DISABLE for its dependencies' progress bars",
    "protobuf": "transitive via shap and onnxruntime; declared to lift a cap",
    "lxml": "the parser pandas prefers for the pd.read_html at sim.py:655",
    "tables": "backs pd.HDFStore at sequencing.py:77 (annotated_reads.h5)",
}


def test_the_indirect_dependencies_are_still_declared_and_still_unimported():
    """These eight are the exception to "declared implies imported".

    They are kept deliberately, and the test asserts both halves so the
    exception cannot quietly become something else: they must still be
    declared, and they must still have no import — because the day one of them
    grows a real import statement, it stops being an indirect dependency and
    the comment in setup.py explaining why it has none becomes wrong.
    """
    declared = _declared_names()
    imported = set(_third_party_imports())

    missing = sorted(d for d in INDIRECT_BUT_REQUIRED if _norm(d) not in declared)
    assert not missing, (
        "these are needed indirectly and are no longer declared: "
        + ", ".join(f"{d} ({INDIRECT_BUT_REQUIRED[d]})" for d in missing)
    )

    now_imported = sorted(
        d for d in INDIRECT_BUT_REQUIRED
        if any(_norm(IMPORT_TO_DIST.get(m, m)) == _norm(d) for m in imported)
    )
    assert not now_imported, (
        f"{now_imported} now has a real import statement under spacr/. That is "
        f"fine, but it is no longer an *indirect* dependency — update the "
        f"comment beside its pin in setup.py and remove it from "
        f"INDIRECT_BUT_REQUIRED here."
    )


# ---------------------------------------------------------------------------
# The census must be able to RUN on the oldest interpreter spaCR claims.
# ---------------------------------------------------------------------------

def test_the_stdlib_set_is_built_without_a_python_3_10_api(monkeypatch):
    """`sys.stdlib_module_names` does not exist on 3.9, and spaCR claims 3.9.

    Every test in this module used to raise
    `AttributeError: module 'sys' has no attribute 'stdlib_module_names'`
    on the 3.9 matrix cell and in the minimum-dependencies job -- so the one
    check that proves no import is undeclared was dead on the one interpreter
    where a resolver is most likely to hand a user a different package set.

    The fallback is exercised here on EVERY interpreter by hiding the 3.10
    attribute, because a branch that only runs on the oldest cell is a branch
    that is only tested when it is already too late.
    """
    monkeypatch.delattr(sys, "stdlib_module_names", raising=False)
    assert not hasattr(sys, "stdlib_module_names")

    names = _stdlib_module_names()

    # It found a real stdlib, not an empty set that would make every stdlib
    # import look like an undeclared dependency.
    assert len(names) > 100
    for expected in ("os", "sys", "json", "ast", "sqlite3", "pathlib",
                     "importlib", "multiprocessing", "tkinter", "winreg"):
        assert expected in names, f"{expected} missing from the 3.9 stdlib set"

    # And it did not sweep site-packages in, which would make an undeclared
    # third-party import invisible -- the exact failure this file exists to
    # catch. Only names spaCR actually imports are worth asserting.
    for third_party in ("numpy", "pandas", "cv2", "PIL", "torch", "cellpose",
                        "skimage", "sklearn", "shap", "statsmodels", "pytest"):
        assert third_party not in names, (
            f"{third_party} leaked into the stdlib set; every undeclared "
            f"import would then be silently excused")


def test_the_census_still_classifies_correctly_through_the_fallback(monkeypatch):
    """The fallback is not merely non-empty -- it gives the same verdict.

    Comparing the two paths directly is what proves the 3.9 branch is a
    substitute rather than a second opinion.
    """
    real = getattr(sys, "stdlib_module_names", None)
    if real is None:                            # pragma: no cover - on 3.9
        pytest.skip("no 3.10 API to compare the fallback against")

    monkeypatch.delattr(sys, "stdlib_module_names", raising=False)
    fallback = _stdlib_module_names()

    imported = set(_imports())
    disagreements = sorted(
        mod for mod in imported
        if (mod in real) != (mod in fallback)
    )
    assert not disagreements, (
        "the 3.9 stdlib fallback disagrees with sys.stdlib_module_names "
        f"about modules spaCR actually imports: {disagreements}")
