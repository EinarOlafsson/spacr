"""Every version bound in ``setup.py`` must be tied to the code that needs it.

``tests/test_packaging_metadata.py`` already checks that the dependency list is
*well-formed*. This file checks that it is *true*: each bound below is asserted
together with the exact call site that justifies it, so the pin and the source
cannot drift apart in either direction.

That coupling is the point, and it is not theoretical. The 2026-07-26 audit
found three ceilings that were too **loose** — a decorative ``<1.0`` on a 0.x
dependency, which protects nothing, because a 0.x project breaks at the MINOR:

  * ``pingouin<1.0`` admitted 0.6.0, which renamed every dashed result column
    (``p-val`` -> ``p_val``). ``spacr/plot.py`` indexes the dashed names
    directly, so a fresh install would have raised ``KeyError`` inside every
    paired statistical test.
  * ``scikit-image<1.0`` admitted 0.27, which removes
    ``skimage.morphology.square`` — imported at ``spacr/utils.py`` module
    scope, so its removal is an ImportError, not a warning.
  * ``statsmodels<1.0`` admitted 0.15, which removes the lowercase
    ``links.logit`` alias that ``spacr/ml.py`` imports at module scope and
    *calls* as a default argument.

...and two floors that were too **low** to be honest: ``scikit-learn>=1.4.1``
admitted a release where ``TSNE(max_iter=)`` is a ``TypeError``, and
``torchvision>=0.1`` named a 2017 release for code that needs the multi-weight
API.

Each test therefore reads BOTH files and fails with a message naming the other
half. If someone fixes ``spacr/plot.py`` to use ``p_val``, the pingouin test
tells them the pin may now be widened; if someone widens the pin without
fixing the source, it tells them what will break. Neither half can move alone.

Like ``test_packaging_metadata.py``, nothing here imports :mod:`spacr` — these
are text/AST reads, so they run in a CI cell with no scientific stack.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_PY = REPO_ROOT / "setup.py"
PKG = REPO_ROOT / "spacr"


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _core_dependencies() -> list[str]:
    """The module-level ``dependencies = [...]`` list in setup.py."""
    tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "dependencies" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    pytest.fail("could not find the `dependencies` list in setup.py")


def _norm(n: str) -> str:
    """PEP 503 normalised project name."""
    return re.sub(r"[-_.]+", "-", n).lower()


def _spec_for(name: str) -> SpecifierSet:
    """The declared SpecifierSet for one distribution, by normalised name."""
    from packaging.requirements import Requirement

    for dep in _core_dependencies():
        req = Requirement(dep)
        if _norm(req.name) == _norm(name):
            return req.specifier
    pytest.fail(f"{name!r} is not in setup.py's core dependencies")


def _src(relpath: str) -> str:
    return (PKG / relpath).read_text(encoding="utf-8")


def _admits(name: str, version: str) -> bool:
    """Would the declared pin accept this version?"""
    return _spec_for(name).contains(Version(version), prereleases=True)


# ---------------------------------------------------------------------------
# 1. Ceilings that must stay CLOSED, because spaCR still calls the old API
# ---------------------------------------------------------------------------

def test_pingouin_cap_excludes_the_column_rename():
    """pingouin 0.6.0 renamed ``p-val`` -> ``p_val`` and ``W-val`` -> ``W_val``.

    ``<1.0`` looked like a ceiling and was not one. Verified against pingouin
    0.6.1: ``pg.ttest(...).iloc[0][['T', 'p-val']]`` raises
    ``KeyError: "['p-val'] not in index"``.
    """
    plot = _src("plot.py")
    uses_dashed = "'p-val'" in plot or '"p-val"' in plot

    if uses_dashed:
        assert not _admits("pingouin", "0.6.0"), (
            "spacr/plot.py still indexes pingouin's dashed result columns "
            "('p-val' / 'W-val'), but the pingouin pin admits 0.6.0, which "
            "renamed them to 'p_val' / 'W_val'. Every paired test in "
            "plot.py would raise KeyError. Keep the cap below 0.6, or "
            "switch plot.py to the underscored names first."
        )
    else:
        pytest.fail(
            "spacr/plot.py no longer uses pingouin's dashed column names. "
            "That is the fix this cap was waiting for — widen "
            "`pingouin>=0.5.5,<0.6` in setup.py and delete this branch."
        )


def test_scikit_image_cap_excludes_the_square_removal():
    """``skimage.morphology.square`` is removed in 0.26 -> 0.27.

    The import at ``spacr/utils.py`` is module scope, so removal is an
    ImportError for every consumer of ``spacr.utils``, not a warning at the
    two call sites.
    """
    utils = _src("utils.py")
    imports_square = re.search(
        r"^from skimage\.morphology import .*\bsquare\b", utils, re.MULTILINE
    )

    if imports_square:
        assert not _admits("scikit-image", "0.27.0"), (
            "spacr/utils.py imports `square` from skimage.morphology at module "
            "scope, but the scikit-image pin admits 0.27, where it is removed "
            "(deprecated 0.25, `removed_version='0.27'`). Use "
            "`skimage.morphology.footprint_rectangle` before widening the cap."
        )
    else:
        pytest.fail(
            "spacr/utils.py no longer imports skimage.morphology.square — "
            "widen the scikit-image cap past 0.27 and delete this branch."
        )


def test_statsmodels_cap_excludes_the_lowercase_link_alias_removal():
    """statsmodels: "The logit link alias will be removed after 0.15.0."

    ``spacr/ml.py`` imports the alias at module scope *and* calls it as a
    default argument, so its removal breaks ``import spacr.ml`` outright.
    """
    ml = _src("ml.py")
    imports_alias = re.search(
        r"^from statsmodels\.genmod\.families\.links import .*\blogit\b",
        ml, re.MULTILINE,
    )

    if imports_alias:
        assert not _admits("statsmodels", "0.15.0"), (
            "spacr/ml.py imports the lowercase `logit` link alias at module "
            "scope, but the statsmodels pin admits 0.15, after which the alias "
            "is removed. Switch to the CamelCase `Logit` (ml.py already uses "
            "it elsewhere) before widening the cap."
        )
    else:
        pytest.fail(
            "spacr/ml.py no longer imports the lowercase `logit` alias — "
            "widen the statsmodels cap and delete this branch."
        )


# ---------------------------------------------------------------------------
# 2. Floors that must stay RAISED, because the old floor was already broken
# ---------------------------------------------------------------------------

def test_scikit_learn_floor_covers_the_tsne_max_iter_rename():
    """``TSNE(max_iter=)`` did not exist before scikit-learn 1.5.0.

    Before 1.5 the parameter was ``n_iter``, so ``>=1.4.1`` admitted a
    resolve where ``spacr/utils.py`` raises
    ``TypeError: TSNE.__init__() got an unexpected keyword argument``.
    """
    utils = _src("utils.py")
    if not re.search(r"TSNE\((?:[^)]|\n)*?max_iter\s*=", utils):
        pytest.skip("spacr/utils.py no longer passes max_iter= to TSNE")

    assert not _admits("scikit-learn", "1.4.2"), (
        "spacr/utils.py calls TSNE(max_iter=...), which needs scikit-learn "
        ">=1.5.0 (before that the parameter was n_iter). The declared floor "
        "still admits 1.4.x, where that call is a TypeError."
    )


def test_torchvision_floor_covers_the_multi_weight_api():
    """``ResNet*_Weights`` landed in torchvision 0.13.0.

    ``spacr/utils.py`` imports the enums at module scope, unguarded, so a
    floor below 0.13 describes a version spaCR cannot import at all. The
    declared floor is 0.15 because torchvision pins torch exactly and 0.15.x
    is the line built against ``torch>=2.0``, which is declared alongside it.
    """
    utils = _src("utils.py")
    if "ResNet50_Weights" not in utils:
        pytest.skip("spacr/utils.py no longer imports the weights enums")

    assert not _admits("torchvision", "0.12.0"), (
        "spacr/utils.py imports ResNet*_Weights from torchvision.models.resnet "
        "at module scope. Those enums arrived in torchvision 0.13.0, so any "
        "floor admitting 0.12 or below (`>=0.1` was the old value) is false."
    )
    assert not _admits("torchvision", "0.14.1"), (
        "torchvision pins torch exactly (0.14.1 -> torch==1.13.1). With "
        "`torch>=2.0` declared, the torchvision floor must be >=0.15, which "
        "is the release built against torch 2.0."
    )


# ---------------------------------------------------------------------------
# 3. numpy: the pin and the code that forces it, asserted together
# ---------------------------------------------------------------------------

#: The three modules that called ``np.trapz``, with the pattern that detects
#: an UNGUARDED use — one that would raise ``AttributeError`` on numpy 2.
#:
#: The distinction matters, and getting it wrong is what this test used to do.
#: Its old patterns were ``np\.trapz\(`` for utils.py and attribution.py and
#: the bare string ``from numpy import trapz`` for timelapse.py. The first two
#: are still right. The third was not: after the fix, spacr/timelapse.py:25-28
#: reads
#:
#:     try:
#:         from numpy import trapezoid as trapz
#:     except ImportError:                     # numpy < 2.0
#:         from numpy import trapz
#:
#: — and the ``except ImportError`` branch, which is the numpy 1.x fallback and
#: is exactly what makes the module work on BOTH numpy lines, matched the old
#: pattern. So the test went on reporting timelapse.py as an unfixed blocker
#: and holding ``numpy<2.0`` shut against a file that had already been fixed.
#: It was pinning the shape of the bug rather than the bug.
#:
#: The patterns below therefore look for a *bare* ``np.trapz(`` call and for a
#: ``from numpy import trapz`` that is NOT the fallback arm of a
#: ``trapezoid``-preferring try/except.
_NUMPY2_BLOCKERS = (
    ("utils.py", r"(?<![\w.])np\.trapz\("),
    ("attribution.py", r"(?<![\w.])np\.trapz\("),
    ("timelapse.py", r"(?<![\w.])np\.trapz\("),
)

#: A module is exempt from the ``from numpy import trapz`` check when it
#: prefers ``trapezoid`` first — that is the documented, working pattern.
_TRAPEZOID_PREFERRED = r"from numpy import trapezoid"


def test_no_module_calls_the_np_trapz_removed_in_numpy_2():
    """``np.trapz`` was removed in numpy 2.0; ``np.trapezoid`` replaces it.

    This is the assertion the old
    ``test_numpy_cap_stays_while_the_np_trapz_call_sites_remain`` was reaching
    for, stated directly. The old test said "IF a call site remains THEN the
    numpy cap must stay below 2.0", which made it a cap-guard that could never
    do anything once the cap moved, and — because its timelapse.py pattern
    matched that module's numpy-1 *fallback* — it never let the cap move at
    all.

    ``spacr/timelapse.py`` was the acute case. Before the fix it guarded the
    import with ``from scipy.integrate import trapz``, which was **already
    dead**: SciPy removed ``integrate.trapz`` in 1.14 and ``scipy<2.0``
    resolves 1.18. Under numpy 2 it was the one import failure in the whole
    package. It now prefers ``numpy.trapezoid`` and falls back to
    ``numpy.trapz`` only on numpy 1.x.
    """
    offenders = [f for f, p in _NUMPY2_BLOCKERS if re.search(p, _src(f))]
    assert not offenders, (
        f"np.trapz (removed in numpy 2.0) is called unguarded in: {offenders}. "
        f"spacr/utils.py and spacr/attribution.py resolve it once at module "
        f"scope with `_trapezoid = getattr(np, 'trapezoid', None) or np.trapz`; "
        f"use that, or np.trapezoid directly. The numpy pin admits 2.x, so "
        f"this is an AttributeError at runtime, not a theoretical risk."
    )

    tl = _src("timelapse.py")
    if re.search(r"^\s*from numpy import trapz\s*$", tl, re.MULTILINE):
        assert re.search(_TRAPEZOID_PREFERRED, tl), (
            "spacr/timelapse.py imports `trapz` from numpy without preferring "
            "`trapezoid` first. On numpy 2 that is an ImportError at module "
            "scope, which makes `import spacr.timelapse` fail outright."
        )


def test_numpy_admits_2x_now_that_all_three_blockers_are_closed():
    """The widening itself, asserted rather than assumed.

    Three things blocked ``numpy>=1.26.4,<2.0`` from moving, and this test
    fails if any of them silently comes back:

    1. ``np.trapz`` at three call sites — covered by the test above.
    2. ``torchcam``, every release of which declares ``numpy<2.0.0``. It is
       now in the ``attribution`` extra.
    3. ``tests/test_diameter_estimator.py`` calling the removed
       ``ndarray.ptp()`` method — now ``np.ptp(field)``.

    The old ``test_torchcam_is_the_other_numpy2_blocker_and_is_still_in_core``
    asserted (2) from the other side: it required the numpy cap to stay below
    2.0 *while* torchcam was in core, and ``pytest.skip``-ed the moment
    torchcam left — becoming a test that could never fail again. Replaced with
    the invariant that actually needs holding: while numpy admits 2.x,
    torchcam must NOT be a core dependency.
    """
    assert _admits("numpy", "2.0.0"), (
        "the numpy cap no longer admits 2.x. If that is a deliberate revert, "
        "say why here — it re-caps Python at 3.12, because numpy 1.26.4 is "
        "the only release satisfying `<2.0` and its wheels stop at cp312."
    )

    core = {re.split(r"[<>=!~ ,\[;]", d.strip())[0].lower()
            for d in _core_dependencies()}
    assert "torchcam" not in core, (
        "torchcam is back in the core dependencies while numpy admits 2.x. "
        "Every release satisfying `>=0.4.0,<1.0` declares `numpy<2.0.0`, so "
        "pip resolves numpy back to 1.26.4 — and on Python 3.13, where 1.26.4 "
        "has no wheel, that is a numpy source build. Measured 2026-07-27 in a "
        "3.13.14 env: `pip install torchcam` is ResolutionImpossible. It "
        "belongs in the `attribution` extra."
    )
    assert "mahotas" not in core, (
        "mahotas is back in the core dependencies. It has never published a "
        "cp313 wheel at any version, so in the core list it forces a C++ "
        "source build on every Python 3.13 install — including on machines "
        "with no toolchain, where it simply fails. It belongs in the "
        "`zernike` extra."
    )

    ptp_test = (REPO_ROOT / "tests" / "test_diameter_estimator.py")
    if ptp_test.exists():
        # Comments are stripped before matching, and that is not a detail.
        # The line that fixed this bug carries the trailing comment
        # `# ndarray.ptp() removed in numpy 2.0`, so a naive search finds
        # "...ay.ptp()" inside the *explanation of the fix* and reports the
        # fixed file as broken. Matching prose about a bug instead of the bug
        # is precisely what the old timelapse.py pattern did a few tests up;
        # doing it again here would be poor form.
        code = "\n".join(
            re.sub(r"#.*$", "", ln)
            for ln in ptp_test.read_text(encoding="utf-8").splitlines()
        )
        assert not re.search(r"\w\.ptp\(\s*\)", code), (
            "tests/test_diameter_estimator.py calls the `ndarray.ptp()` "
            "method, removed in numpy 2.0. Use the `np.ptp(x)` function form."
        )


# ---------------------------------------------------------------------------
# 4. Ceilings that must EXIST, because their absence changes other pins
# ---------------------------------------------------------------------------

def test_the_unimported_dependencies_stay_removed():
    """The 18 distributions with zero imports do not come back.

    This replaces ``test_monai_is_capped_so_it_cannot_raise_the_torch_floor``,
    which asserted that ``monai`` carried an upper bound — monai 1.6.0
    requires ``torch>=2.8.0`` and 1.5.1 requires >=2.4.1, so an uncapped
    ``monai>=1.3.0`` silently overrode the ``torch>=2.0`` declared a few lines
    above it. That test is obsolete for the best possible reason: capping a
    package spaCR imports in zero files was treating the symptom. monai is
    gone, so there is no cap to check.

    The census that removed these was run over all 159 files under ``spacr/``
    with ``ast.walk`` (so imports inside functions and ``try``/``except``
    bodies count, not just module scope), then re-checked with a raw
    ``grep -rIn -w`` across the whole tree including non-Python files. Each
    name below had zero import statements, zero dynamic references and zero
    string references that were not prose.

    The census is fallible in one specific direction, and the guard against
    re-adding something must not become a guard against noticing that:
    ``umap-learn`` has zero import statements too, and is nonetheless a real,
    load-bearing core dependency, because spaCR reaches it through
    ``umap = _LazyModule('umap.umap_', ...)`` at spacr/utils.py:197 — a string
    literal no import census can see. Before deleting anything from this list,
    grep for the string form as well.
    """
    removed = {
        "transformers", "monai", "segmentation-models-pytorch",
        "torch-geometric", "pywavelets", "rapidfuzz", "wandb", "gdown",
        "pytz", "ipykernel", "ttkthemes", "ttf-opensans", "brokenaxes",
        "gpustat", "customtkinter", "openai", "keyring", "importlib-metadata",
    }
    core = {_norm(re.split(r"[<>=!~ ,\[;]", d.strip())[0])
            for d in _core_dependencies()}
    back = sorted(removed & core)
    assert not back, (
        f"{back} returned to setup.py's core dependencies. Each was removed "
        f"on 2026-07-27 for having zero imports, zero string references and "
        f"zero dynamic references anywhere under spacr/. If one is genuinely "
        f"needed now, add the import in the same commit — and if it is needed "
        f"through a string literal the way umap-learn is, say so in a comment "
        f"next to the pin, because the next census will not be able to tell."
    )


def test_every_directly_imported_third_party_module_is_declared():
    """The other half: an import with no declaration is a latent break.

    Five of spaCR's module-scope imports were undeclared until 2026-07-27 and
    worked only because something else dragged them in — ``requests`` via
    huggingface-hub (which drops requests for httpx at 1.0), ``joblib`` via
    scikit-learn, ``natsort`` via cellpose, ``patsy`` via statsmodels and
    ``sympy`` via torch. Every one of those was a dependency-of-a-dependency
    away from an ImportError at ``import spacr.utils``.
    """
    declared = {_norm(re.split(r"[<>=!~ ,\[;]", d.strip())[0])
                for d in _core_dependencies()}
    must_be_declared = {
        # distribution name -> the file that imports it at module scope
        "requests": "utils.py",
        "joblib": "utils.py",
        "natsort": "submodules.py",
        "patsy": "ml.py",
        "sympy": "gui_elements.py",
    }
    missing = []
    for dist, where in sorted(must_be_declared.items()):
        if re.search(rf"^\s*(import|from)\s+{dist}\b", _src(where), re.MULTILINE):
            if _norm(dist) not in declared:
                missing.append((dist, where))
    assert not missing, (
        f"imported at module scope but not declared: {missing}. These arrive "
        f"transitively today and vanish the day the package that brings them "
        f"changes its own dependencies."
    )


# ---------------------------------------------------------------------------
# 5. No bound may name a version that cannot exist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dep", _core_dependencies())
def test_no_dependency_declares_an_empty_range(dep):
    """A floor above its own ceiling installs nothing, and pip says so only
    at install time — never at build time, and never in CI's metadata job."""
    from packaging.requirements import Requirement

    spec = Requirement(dep).specifier
    floors = [Version(s.version) for s in spec if s.operator in (">=", ">", "==")]
    caps = [Version(s.version) for s in spec if s.operator in ("<", "<=")]
    if floors and caps:
        assert max(floors) < max(caps), (
            f"{dep!r} declares a floor at or above its ceiling — no release "
            f"can satisfy it"
        )
