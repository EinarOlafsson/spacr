"""The two ends of the dependency policy stay declared, tested and blocking.

"Always use the optimal dependency versions" has two readings that imply
opposite CI jobs -- the oldest set that passes, or the newest set that passes
-- and a dependency list edited under both readings at once drifts in both
directions at once. spaCR resolves it one way: what a user GETS is the newest
version in every declared range, and a lower bound is a promise to whoever is
pinned by something else rather than a recommendation.

That decision only means something while both halves are executed. This file
asserts the shape rather than the prose:

* the newest end (``fast``) installs with no constraints file, so it measures
  what a fresh ``pip install spacr`` resolves to today;
* the floor end (``minimum-dependencies``) installs
  ``.github/constraints/minimum-py39.txt``, so every declared lower bound is a
  version CI has actually run;
* both are in the release gate, so neither end can quietly stop deciding;
* no constraint pin sits above the floor setup.py declares unless the
  constraints file names the package and says why.

The last one is the sharp edge. A pin above its floor leaves the floor the one
bound in ``setup.py`` that nothing installs and nothing tests, which is exactly
the failure the cellpose floor was: ``>=4.0`` resolved to a release whose
``CellposeModel`` signature spaCR had never been developed against, and the
only job that could have caught it was the one that installs the floors.

Nothing here imports :mod:`spacr` -- these are text, YAML and AST reads, so the
file runs in a CI cell with no scientific stack installed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.py"
CONSTRAINTS = ROOT / ".github" / "constraints" / "minimum-py39.txt"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

#: The heading that carries the decision. Named here so deleting the block is a
#: test failure rather than a silent loss of the only place it is written down.
POLICY_HEADING = 'WHAT "THE OPTIMAL DEPENDENCY VERSIONS" MEANS'

#: The job that resolves newest, and the job that installs the declared floors.
NEWEST_JOB = "fast"
FLOOR_JOB = "minimum-dependencies"

#: Extras the floor job installs alongside the core requirements.
FLOOR_EXTRAS = ("qt", "dev", "zernike", "attribution", "btrack", "czi")


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps_script(job: dict) -> str:
    return "\n".join(
        step["run"]
        for step in job.get("steps", [])
        if isinstance(step.get("run"), str)
    )


def _job_script(job_name: str) -> str:
    """Return every ``run:`` script a job executes, joined into one string.

    A job that delegates to a reusable workflow has no ``steps`` of its own, so
    the called workflow's scripts are what it actually runs -- and are where a
    constraints file would have to hide.
    """
    job = _workflow()["jobs"][job_name]
    called = job.get("uses")
    if isinstance(called, str) and called.startswith("./"):
        reusable = yaml.safe_load(
            (ROOT / called[len("./"):]).read_text(encoding="utf-8")
        )
        return "\n".join(
            _steps_script(inner) for inner in reusable["jobs"].values()
        )
    return _steps_script(job)


def _declared() -> dict[str, Requirement]:
    """Map canonical distribution name -> requirement, core plus floor extras."""
    tree = ast.parse(SETUP.read_text(encoding="utf-8"))
    core: list[str] | None = None
    extras: dict[str, list[str]] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "dependencies"
            for target in node.targets
        ):
            core = ast.literal_eval(node.value)
        call = node.value if isinstance(node, (ast.Expr, ast.Assign)) else None
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "setup":
            for keyword in call.keywords:
                if keyword.arg == "extras_require":
                    extras = ast.literal_eval(keyword.value)
    assert core is not None, "setup.py no longer assigns `dependencies`"
    assert extras is not None, "setup.py no longer passes `extras_require`"

    selected = list(core)
    for extra in FLOOR_EXTRAS:
        selected.extend(extras[extra])
    declared: dict[str, Requirement] = {}
    for value in selected:
        requirement = Requirement(value)
        declared.setdefault(canonicalize_name(requirement.name), requirement)
    return declared


def _constraint_comments() -> str:
    """Return only the comment lines of the constraints file."""
    return "\n".join(
        line
        for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    )


def _pins() -> dict[str, Version]:
    pins: dict[str, Version] = {}
    for raw in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        exact = [
            specifier.version
            for specifier in requirement.specifier
            if specifier.operator in {"==", "==="}
        ]
        assert len(exact) == 1, f"{line!r} must pin exactly one version"
        pins[canonicalize_name(requirement.name)] = Version(exact[0])
    return pins


def _declared_floor(requirement: Requirement) -> Version | None:
    lows = [
        Version(specifier.version)
        for specifier in requirement.specifier
        if specifier.operator in {">=", "==", "===", "~="}
    ]
    return min(lows) if lows else None


def test_the_decision_is_written_down_where_the_bounds_are():
    """The one place the reading is settled must live beside the version list.

    A policy recorded only in a task file is a policy the next person editing
    ``setup.py`` never sees, which is how the list came to be edited under both
    readings at once.
    """
    text = SETUP.read_text(encoding="utf-8")
    assert POLICY_HEADING in text, (
        "setup.py no longer states what 'optimal dependency versions' means"
    )
    block = text[text.index(POLICY_HEADING):]
    block = block[: block.index("dependencies = [")]
    assert "NEWEST" in block, "the policy must say which end a user gets"
    assert "Fast / Full suite control" in block, "the newest end is unnamed"
    assert "Minimum dependencies" in block, "the floor end is unnamed"
    assert "minimum-py39.txt" in block, "the file that executes the floor is unnamed"


def test_the_newest_end_installs_with_no_constraints_file():
    """The job that decides the recommendation has to resolve freely.

    Constraining it would make CI test a set no user ever installs, and the
    upper end of every declared range is precisely what pip hands a user.
    """
    script = _job_script(NEWEST_JOB)
    assert "pip install -e" in script, f"{NEWEST_JOB} no longer installs spaCR"
    assert "constraints/" not in script, (
        f"{NEWEST_JOB} installs a constraints file; it exists to resolve newest"
    )


def test_the_floor_end_installs_the_declared_floors():
    """A lower bound nothing installs is a promise nothing keeps."""
    script = _job_script(FLOOR_JOB)
    assert "-c .github/constraints/minimum-py39.txt" in script, (
        f"{FLOOR_JOB} no longer installs the minimum profile"
    )


def test_only_the_floor_end_constrains_its_resolution():
    """Exactly one job may pin; every other job answers the newest question."""
    workflow = _workflow()
    constrained = sorted(
        name
        for name in workflow["jobs"]
        if "constraints/" in _job_script(name)
    )
    assert constrained == [FLOOR_JOB], (
        f"jobs installing a constraints file: {constrained}; only {FLOOR_JOB!r} "
        "may, because every other job exists to test the newest resolution"
    )


def test_both_ends_block_the_release_gate():
    """Either end silently dropping out is how a policy stops being enforced."""
    needs = _workflow()["jobs"]["release-gate"]["needs"]
    assert NEWEST_JOB in needs, "the newest-resolution job stopped deciding"
    assert FLOOR_JOB in needs, "the floor job stopped deciding"


def test_every_constraint_pin_is_the_floor_or_explains_itself():
    """A pin above its floor leaves that floor the one bound nothing tests.

    The gap is legitimate only when the floors cannot be installed together --
    torch/torchvision/sympy is the case that arises -- and then the constraints
    file has to name the package, because otherwise the version a pinned user's
    resolver lands on is exactly the version CI skipped.
    """
    declared = _declared()
    comments = _constraint_comments()
    unexplained = []
    for name, pin in _pins().items():
        requirement = declared.get(name)
        assert requirement is not None, (
            f"{name!r} is constrained but not declared by setup.py"
        )
        floor = _declared_floor(requirement)
        assert floor is not None, (
            f"{name!r} is pinned in the minimum profile but declares no floor"
        )
        if pin == floor:
            continue
        if canonicalize_name(requirement.name) in {
            canonicalize_name(word.strip(".,;:()"))
            for word in comments.split()
        }:
            continue
        unexplained.append(f"{requirement.name}: floor {floor}, pinned {pin}")
    assert not unexplained, (
        "constraint pins above their declared floor with no reason recorded in "
        f"{CONSTRAINTS.name}: {unexplained}. Either lower the pin to the floor, "
        "raise the floor in setup.py to the version CI installs, or write down "
        "the coupling that forces the gap."
    )


def test_no_pin_sits_below_the_floor_it_is_meant_to_verify():
    """The mirror-image drift: a profile installing what setup.py refuses.

    This is the shape the cellpose floor failed in -- the constraints file
    pinned 4.0.1 while setup.py required 4.0.7, so the only job that installs
    the floors was installing a release the package declares unsupported and
    failing tests every other job passed.
    """
    declared = _declared()
    below = [
        f"{name}: pinned {pin}, declared {declared[name].specifier}"
        for name, pin in _pins().items()
        if pin not in declared[name].specifier
    ]
    assert not below, f"minimum profile installs versions setup.py refuses: {below}"


@pytest.mark.parametrize("job", [NEWEST_JOB, FLOOR_JOB])
def test_both_ends_run_the_same_selection(job):
    """Two ends of one range are only comparable while they run one suite.

    If the floor job ran a narrower marker expression than the newest job, a
    failure that appears only on the floors could be a real incompatibility or
    could be a test the newest job never ran, and the two are indistinguishable
    from the log.
    """
    script = _job_script(job)
    assert "tools/run_pytest_batches.py" in script, (
        f"{job} no longer runs the batched suite"
    )
    assert (
        "not integration and not slow and not heavy and not qt and not gpu "
        "and not network and not nas and not gui"
    ) in script, f"{job} runs a different selection from its opposite end"
