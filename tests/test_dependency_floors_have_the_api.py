"""A declared floor must be a version that has the API spaCR calls.

`tests/test_declared_dependencies_match_imports.py` proves every import is
DECLARED. Nothing proved the declared VERSION works, and that gap is the more
expensive one: pip resolves a too-low floor happily, the install succeeds, and
the failure arrives later and somewhere else.

It had already bitten twice. `statsmodels>=0.13.0` was justified in setup.py
by ``othermod.betareg.BetaModel`` -- the first API spaCR needed, not the last
-- and `shap>=0.45.0` had no justification at all. The "Minimum dependencies"
CI job, which installs exactly the declared floors, failed 16 tests that no
other job failed:

    statsmodels 0.13.5   links.Identity                 AttributeError
                         BetaResults.get_influence      AttributeError
                         GLM perfect separation         raises, never returns
    shap 0.45.0          summary_plot(rng=...)          TypeError

Each was verified in an isolated venv on the rest of the minimum profile
(numpy 1.26.4, pandas 2.2.1, scipy 1.12.0, patsy 0.5.6): every row below
fails on the release under `min_version` and passes on `min_version` itself.

WHAT EACH ROW ASSERTS, and why all three parts are needed:

  * `uses` -- spaCR's source really does reach for the API. A floor
    justified by an API nobody calls is folklore; when the last caller goes,
    this fails and tells you to drop the row rather than keeping a floor
    higher than the code earns.
  * the declared floor in setup.py admits nothing below `min_version`.
    This is the assertion that was red before the floors were raised.
  * the INSTALLED release still provides the API. That is the mirror image
    -- upstream removing something spaCR depends on -- and it is the same
    trick `test_cellpose_api_contract.py` uses: record the contract, check
    it against what is actually importable.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Callable, Optional

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "spacr"


# ---------------------------------------------------------------------------
# setup.py's declared floors, read from the AST rather than by regex so a
# requirement inside a comment or an entry-point string cannot be mistaken
# for one.
# ---------------------------------------------------------------------------

def declared_requirements() -> dict[str, list[Requirement]]:
    """Every dependency string in setup.py, as {lowercased name: [Requirement]}."""
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    found: dict[str, list[Requirement]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        try:
            req = Requirement(node.value)
        except InvalidRequirement:
            continue
        if not req.specifier:
            continue
        found.setdefault(req.name.lower(), []).append(req)
    return found


def _uses_substring(relative_path: str, needle: str) -> Callable[[], bool]:
    def check() -> bool:
        return needle in (PKG / relative_path).read_text(encoding="utf-8")
    return check


def _summary_plot_is_called_with_rng() -> bool:
    """spacr/sim.py calls ``shap.summary_plot(..., rng=...)``.

    Checked through the AST, not a substring: `rng=` appears all over the
    file for unrelated numpy generators, and only the keyword ON THIS CALL
    is what the shap floor has to cover.
    """
    tree = ast.parse((PKG / "sim.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "summary_plot"):
            continue
        if any(kw.arg == "rng" for kw in node.keywords):
            return True
    return False


def _installed_has_links_identity() -> bool:
    from statsmodels.genmod.families import links
    return hasattr(links, "Identity")


def _installed_has_beta_get_influence() -> bool:
    from statsmodels.othermod.betareg import BetaResults
    return hasattr(BetaResults, "get_influence")


def _installed_warns_on_perfect_separation() -> bool:
    from statsmodels.tools import sm_exceptions
    return hasattr(sm_exceptions, "PerfectSeparationWarning")


def _installed_summary_plot_takes_rng() -> bool:
    import shap
    return "rng" in inspect.signature(shap.summary_plot).parameters


@dataclass(frozen=True)
class FloorFact:
    package: str
    api: str
    min_version: str
    reason: str
    uses: Optional[Callable[[], bool]]
    installed_has: Callable[[], bool]
    used_at: str
    label: str


FLOOR_FACTS: tuple[FloorFact, ...] = (
    FloorFact(
        package="statsmodels",
        api="statsmodels.genmod.families.links.Identity",
        min_version="0.14.0",
        reason=(
            "0.13 spells the Power-derived links in lowercase "
            "(links.identity); the CamelCase classes arrive in 0.14.0, so "
            "on 0.13 this is AttributeError rather than a deprecation. "
            "links.Logit and links.Log, also used by spacr/ml.py, DO "
            "predate it -- which is why the gap went unnoticed."
        ),
        uses=_uses_substring("ml.py", "links.Identity("),
        installed_has=_installed_has_links_identity,
        used_at="spacr/ml.py",
        label="links.Identity",
    ),
    FloorFact(
        package="statsmodels",
        api="statsmodels.othermod.betareg.BetaResults.get_influence",
        min_version="0.14.0",
        reason=(
            "spacr/regression_qc.py asks a fitted model for its own leverage "
            "and falls back to the design matrix when it cannot. On 0.13 the "
            "fallback always fires for a beta fit, so the residual is "
            "standardised by the wrong hat diagonal -- silently."
        ),
        uses=_uses_substring("regression_qc.py", "get_influence"),
        installed_has=_installed_has_beta_get_influence,
        used_at="spacr/regression_qc.py",
        label="BetaResults.get_influence",
    ),
    FloorFact(
        package="statsmodels",
        api="statsmodels.tools.sm_exceptions.PerfectSeparationWarning",
        min_version="0.14.0",
        reason=(
            "Behaviour, not an import: GLM._fit_irls RAISES "
            "PerfectSeparationError on 0.13 and only WARNS from 0.14.0. "
            "spaCR's binomial backends need the fit to come back. The "
            "warning class is the symbol that marks the change."
        ),
        # No source anchor: spaCR never names the class. What it depends on
        # is the fit returning, and the class existing is the marker.
        uses=None,
        installed_has=_installed_warns_on_perfect_separation,
        used_at="spacr/ml.py (binomial GLM backends)",
        label="PerfectSeparationWarning",
    ),
    FloorFact(
        package="shap",
        api="shap.summary_plot(rng=...)",
        min_version="0.47.0",
        reason=(
            "summary_plot IS shap.plots._beeswarm.summary_legacy, which "
            "grew `rng` in 0.47.0. Neither 0.45.0 nor 0.46.0 takes it, and "
            "neither has a **kwargs to swallow it, so the call is a "
            "TypeError on both."
        ),
        uses=_summary_plot_is_called_with_rng,
        installed_has=_installed_summary_plot_takes_rng,
        used_at="spacr/sim.py",
        label="summary_plot-rng",
    ),
)


def _ids(fact: FloorFact) -> str:
    return f"{fact.package}-{fact.label}"


def _assert_requirement_starts_at_api_floor(
        requirement: Requirement, fact: FloorFact) -> None:
    """Reject any explicit lower bound below the first compatible release.

    Checking one sample version below the API floor is not sufficient.  For
    example, ``statsmodels>=0.13.5`` excludes 0.13.0 but still admits 0.13.5,
    where every 0.14 API recorded above is absent.  Read the declared
    ``>=`` clauses themselves so an in-between floor cannot slip through.
    """
    lower_bounds = [
        Version(clause.version)
        for clause in requirement.specifier
        if clause.operator == ">="
    ]
    assert lower_bounds, (
        f"setup.py declares {requirement} without an explicit >= floor. "
        f"{fact.api} needs >={fact.min_version}: {fact.reason}"
    )

    declared_floor = max(lower_bounds)
    api_floor = Version(fact.min_version)
    assert declared_floor >= api_floor, (
        f"setup.py declares {requirement}, whose effective >= floor "
        f"{declared_floor} is below {fact.min_version}. {fact.api} needs "
        f">={fact.min_version}: {fact.reason}"
    )
    assert api_floor in requirement.specifier, (
        f"setup.py declares {requirement}, which excludes {fact.min_version} "
        f"-- the release that introduced {fact.api}. The floor and the "
        "recorded API now disagree in the other direction."
    )


@pytest.mark.parametrize("fact", FLOOR_FACTS, ids=_ids)
def test_spacr_still_uses_the_api_the_floor_is_declared_for(fact: FloorFact):
    """A floor is only earned while something still calls the API.

    Without this the file rots into a list of numbers nobody can justify,
    which is the exact failure mode that put statsmodels at 0.13.0.
    """
    if fact.uses is None:
        pytest.skip(f"{fact.api} is a behaviour marker, not a named call")
    assert fact.uses(), (
        f"{fact.used_at} no longer uses {fact.api}. Either the row is stale "
        f"and should be deleted, or the call moved and the anchor needs "
        f"updating -- do not leave the floor unexplained."
    )


@pytest.mark.parametrize("fact", FLOOR_FACTS, ids=_ids)
def test_the_declared_floor_is_not_below_the_release_that_has_the_api(
        fact: FloorFact):
    """setup.py must not admit a release that lacks the API spaCR calls."""
    reqs = declared_requirements().get(fact.package.lower())
    assert reqs, f"setup.py no longer declares {fact.package}"

    for req in reqs:
        _assert_requirement_starts_at_api_floor(req, fact)


@pytest.mark.parametrize(
    ("package", "bad_requirement"),
    (
        ("statsmodels", "statsmodels>=0.13.5,<0.15"),
        ("shap", "shap>=0.46.1,<1.0"),
    ),
)
def test_floor_guard_rejects_an_in_between_lower_bound(
        package: str, bad_requirement: str):
    """A floor between the old pin and the real API release is still wrong.

    The former one-sample probe checked 0.13.0 and 0.46.0 respectively, so
    both mutations here passed despite admitting releases without the API.
    """
    fact = next(fact for fact in FLOOR_FACTS if fact.package == package)
    with pytest.raises(AssertionError, match="effective >= floor.*is below"):
        _assert_requirement_starts_at_api_floor(
            Requirement(bad_requirement), fact)


@pytest.mark.parametrize("fact", FLOOR_FACTS, ids=_ids)
def test_the_installed_release_still_provides_the_api(fact: FloorFact):
    """Upstream removing the API is the mirror image of the floor being low.

    The declared range has a ceiling; this is what notices when a release
    inside it stops providing what spaCR calls.
    """
    try:
        installed = Version(distribution_version(fact.package))
    except PackageNotFoundError:
        pytest.fail(
            f"{fact.package} is not installed; cannot verify {fact.api}")

    api_floor = Version(fact.min_version)
    assert installed >= api_floor, (
        f"installed {fact.package}=={installed} is below the declared API "
        f"floor {fact.package}>={fact.min_version}. This environment is "
        "outside spaCR's supported dependency range; it is not evidence "
        f"that {fact.api} was removed upstream. Synchronise the environment "
        "with setup.py before using this API-removal probe."
    )

    assert fact.installed_has(), (
        f"{fact.api} is gone from the installed {fact.package}. Either "
        f"spaCR's call site must change or the ceiling in setup.py must "
        f"come down -- do not delete this row."
    )


def test_the_minimum_constraints_file_pins_these_exact_floors():
    """The min-deps CI profile is where a wrong floor actually shows.

    Coupled here as well as in test_minimum_constraints_match_setup.py so
    that raising a floor for an API reason cannot leave the one job that
    exercises floors testing a version spaCR no longer supports.
    """
    constraints = ROOT / ".github" / "constraints" / "minimum-py39.txt"
    pins: dict[str, Version] = {}
    for line in constraints.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        pins[req.name.lower()] = Version(str(req.specifier).lstrip("="))

    for fact in FLOOR_FACTS:
        pinned = pins.get(fact.package.lower())
        assert pinned is not None, (
            f"{constraints} no longer pins {fact.package}")
        assert pinned >= Version(fact.min_version), (
            f"{constraints} pins {fact.package}=={pinned}, below the "
            f"{fact.min_version} that {fact.api} needs: {fact.reason}"
        )
