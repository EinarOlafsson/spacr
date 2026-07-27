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


def _spec_for(name: str) -> SpecifierSet:
    """The declared SpecifierSet for one distribution, by normalised name."""
    from packaging.requirements import Requirement

    def norm(n: str) -> str:
        return re.sub(r"[-_.]+", "-", n).lower()

    for dep in _core_dependencies():
        req = Requirement(dep)
        if norm(req.name) == norm(name):
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

#: The three things that must be fixed before `numpy<2.0` can move. Proven on
#: 2026-07-26: with these done, the full dependency set installs and every
#: spaCR module imports under numpy 2.4.4 on CPython 3.12 and 3.13.
_NUMPY2_BLOCKERS = (
    ("utils.py", r"np\.trapz\("),
    ("attribution.py", r"np\.trapz\("),
    ("timelapse.py", r"from numpy import trapz"),
)


def test_numpy_cap_stays_while_the_np_trapz_call_sites_remain():
    """``np.trapz`` was removed in numpy 2.0; ``np.trapezoid`` replaces it.

    ``spacr/timelapse.py`` is the acute case: it guards the import with a
    ``from scipy.integrate import trapz`` fallback that is **already dead**,
    because SciPy removed ``integrate.trapz`` in 1.14 and ``scipy<2.0`` here
    resolves 1.18. Under numpy 2 that module is the one import failure in the
    whole package.
    """
    remaining = [(f, p) for f, p in _NUMPY2_BLOCKERS
                 if re.search(p, _src(f))]

    if remaining:
        assert not _admits("numpy", "2.0.0"), (
            "numpy's cap admits 2.x, but np.trapz (removed in numpy 2.0) is "
            f"still called at: {[f for f, _ in remaining]}. Replace it with "
            "np.trapezoid first — spacr/timelapse.py additionally needs its "
            "dead scipy.integrate.trapz fallback removed."
        )
    else:
        pytest.fail(
            "No np.trapz call sites remain. That was blocker 1 of 3 for "
            "numpy 2. Blockers 2 and 3: torchcam declares `numpy<2.0.0` at "
            "every release satisfying its pin, so it must leave the core "
            "dependencies; and tests/test_diameter_estimator.py calls the "
            "removed `ndarray.ptp()` method. With all three done, "
            "`numpy>=1.26.4,<3.0` and `requires-python = '>=3.10,<3.14'` "
            "were both verified working."
        )


def test_torchcam_is_the_other_numpy2_blocker_and_is_still_in_core():
    """torchcam 0.4.0 and 0.4.1 both declare ``numpy<2.0.0``.

    The pin is spurious — torchcam touches numpy only in an overlay helper,
    and GradCAM was verified running correctly under numpy 2.4.4 — but pip
    cannot be argued with. This test exists so that whoever widens numpy is
    reminded that torchcam has to move to an extra in the same commit.
    """
    core = {re.split(r"[<>=!~ ,\[]", d.strip())[0].lower()
            for d in _core_dependencies()}
    if "torchcam" not in core:
        pytest.skip("torchcam has left the core dependencies")

    assert not _admits("numpy", "2.0.0"), (
        "numpy's cap admits 2.x while torchcam is still a core dependency. "
        "Every torchcam release satisfying `>=0.4.0,<1.0` declares "
        "`numpy<2.0.0`, so pip will resolve numpy back to 1.26.4 — and on "
        "Python 3.13, where 1.26.4 has no wheel, drop the user into the numpy "
        "source build the cap exists to prevent."
    )


# ---------------------------------------------------------------------------
# 4. Ceilings that must EXIST, because their absence changes other pins
# ---------------------------------------------------------------------------

def test_monai_is_capped_so_it_cannot_raise_the_torch_floor():
    """monai 1.6.0 requires ``torch>=2.8.0``; monai 1.5.1 requires >=2.4.1.

    Uncapped, ``monai>=1.3.0`` silently overrode the ``torch>=2.0`` declared
    a few lines above it and dragged torchvision along — a multi-GB resolver
    swing for a package spaCR imports in zero files.
    """
    spec = _spec_for("monai")
    has_cap = any(s.operator in ("<", "<=", "==") for s in spec)
    assert has_cap, (
        "monai has no upper bound. It raised its own torch floor to 2.4.1 in "
        "1.5.1 and to 2.8.0 in 1.6.0, so an uncapped monai silently overrides "
        "spaCR's declared `torch>=2.0`."
    )
    assert not _admits("monai", "1.6.0"), (
        "the monai cap admits 1.6.0, which requires torch>=2.8.0 and so "
        "contradicts the `torch>=2.0` declared in the same list."
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
