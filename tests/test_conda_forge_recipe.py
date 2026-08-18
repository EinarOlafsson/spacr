"""Contracts for the one-time conda-forge recipe and update-bot setup."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "conda-forge" / "recipe" / "recipe.yaml"
BOT_CONFIG = ROOT / "conda-forge" / "conda-forge.yml"
SETUP = ROOT / "setup.py"

CONDA_NAMES = {
    "torch": "pytorch",
    "opencv-python-headless": "opencv",
    "matplotlib": "matplotlib-base",
    "matplotlib-venn": "matplotlib-venn",
    "nvidia-ml-py": "pynvml",
    "tables": "pytables",
    "huggingface-hub": "huggingface_hub",
}

# The conda package is an application distribution, not only setup.py's core
# library wheel. This is the one deliberate conda-only runtime addition:
# Cellpose's recipe currently omits its SAM import.
#
# pyside6 and qtawesome used to be here for the reason the name says -- spaCR's
# Qt extra cannot be selected through conda package extras. They are core
# dependencies of the wheel as of 2026-08-17 ("lets stop hiding the qt behind
# a qt"), so they arrive through `expected` now and naming them here as well
# would only hide it if they were ever removed from setup.py.
CONDA_APPLICATION_DEPENDENCIES = {
    "segment-anything",
}


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _core_dependency_names() -> set[str]:
    tree = ast.parse(SETUP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "dependencies"
            for target in node.targets
        ):
            requirements = ast.literal_eval(node.value)
            # A REQUIREMENT WITH AN ENVIRONMENT MARKER IS NOT EXPECTED IN
            # THE RECIPE. `win10toast` is declared
            # `platform_system == "Windows"`, and this recipe is
            # `noarch: python` -- one package for every platform, with no way
            # to express the marker. Listing it would make a Windows-only
            # toast library a hard runtime dependency on Linux and macOS,
            # where it does not build.
            return {
                _normalise(Requirement(requirement).name)
                for requirement in requirements
                if not Requirement(requirement).marker
            }
    raise AssertionError("setup.py has no literal dependencies assignment")


def test_conda_recipe_covers_every_core_dependency():
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    run = recipe["requirements"]["run"]
    conda_names = {
        _normalise(str(requirement).split()[0])
        for requirement in run
        if not str(requirement).startswith("python ")
    }
    expected = {
        _normalise(CONDA_NAMES.get(name, name))
        for name in _core_dependency_names()
    }
    assert conda_names == expected | CONDA_APPLICATION_DEPENDENCIES


def test_conda_recipe_is_noarch_and_uses_a_verified_tag_archive():
    text = RECIPE.read_text(encoding="utf-8")
    recipe = yaml.safe_load(text)
    assert recipe["build"]["noarch"] == "python"
    assert "--no-deps" in recipe["build"]["script"]
    assert "archive/refs/tags/v${{ version }}.tar.gz" in recipe["source"]["url"]
    assert re.fullmatch(r"[0-9a-f]{64}", recipe["source"]["sha256"])
    assert recipe["extra"]["recipe-maintainers"] == ["EinarOlafsson"]


def test_conda_recipe_exercises_heavy_and_desktop_imports_without_pip_metadata():
    """The SAM conda package imports but has no dist-info for ``pip check``."""
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    python_imports = {
        name
        for test in recipe["tests"]
        for name in test.get("python", {}).get("imports", [])
    }
    scripts = [
        command
        for test in recipe["tests"]
        for command in test.get("script", [])
    ]
    assert {"spacr.measure", "spacr.qt", "cellpose", "segment_anything"} <= (
        python_imports
    )
    assert "spacr-run --list" in scripts
    assert "pip check" not in RECIPE.read_text(encoding="utf-8")


def test_conda_recipe_preserves_the_license_of_its_tagged_source():
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    assert recipe["context"]["version"] in {"1.4.9.8", "1.4.9.9"}
    assert recipe["about"]["license"] == "MIT"
    assert recipe["about"]["license_file"] == "LICENSE"


def test_conda_forge_bot_tracks_pypi_and_automerge_is_limited_to_versions():
    config = yaml.safe_load(BOT_CONFIG.read_text(encoding="utf-8"))
    assert config["bot"]["automerge"] == "version"
    assert config["bot"]["check_solvable"] is True
    assert config["bot"]["version_updates"]["sources"] == ["pypi"]
    assert config["conda_forge_output_validation"] is True


# ---------------------------------------------------------------------------
# Versions, not just names.
#
# `test_conda_recipe_covers_every_core_dependency` above checks that the run
# list NAMES the same packages as setup.py, and that is all it checked. The
# recipe therefore drifted on versions without anything noticing: when
# instruction 54 raised the cellpose floor to 4.0.7 on 2026-08-08 -- because
# 4.0.1 has a different `CellposeModel.__init__` and spaCR has never been
# developed against it -- the recipe kept saying `>=4.0`. A conda user would
# have resolved to exactly the release the PyPI package refuses.
#
# The rule is one-sided on purpose. A recipe floor ABOVE setup.py's is a
# packaging decision (conda-forge may simply not have the older build); a
# floor BELOW it is spaCR promising support it has withdrawn.
# ---------------------------------------------------------------------------

#: Where conda-forge versions a package differently from PyPI, and why.
#: `opencv-python-headless 4.9.0.80` is a wrapper whose fourth component is
#: the wrapper build, not the OpenCV release; conda-forge ships the library
#: itself as `opencv 4.9.0`, so the two spellings name the same floor.
FLOOR_TRANSLATIONS = {
    "opencv": {"4.9.0.80": "4.9.0"},
}


def _core_dependency_specifiers() -> dict[str, str]:
    """{conda name: setup.py specifier} for every core dependency."""
    tree = ast.parse(SETUP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "dependencies"
            for target in node.targets
        ):
            out = {}
            for requirement in ast.literal_eval(node.value):
                req = Requirement(requirement)
                name = _normalise(req.name)
                out[_normalise(CONDA_NAMES.get(name, name))] = req.specifier
            return out
    raise AssertionError("setup.py has no literal dependencies assignment")


def _lower_bound(specifier) -> str | None:
    for clause in specifier:
        if clause.operator in (">=", "=="):
            return clause.version
    return None


def _assert_no_conda_recipe_floor_is_below_declared(run) -> None:
    from packaging.version import Version

    declared = _core_dependency_specifiers()

    too_low = []
    for entry in run:
        parts = str(entry).split(None, 1)
        name = _normalise(parts[0])
        if name == "python":
            continue
        specifier = declared.get(name)
        if specifier is None:
            continue
        wanted = _lower_bound(specifier)
        if wanted is None:
            continue
        wanted = FLOOR_TRANSLATIONS.get(name, {}).get(wanted, wanted)
        if len(parts) == 1:
            too_low.append(
                f"{name}: recipe has no lower bound, setup.py requires "
                f">={wanted}")
            continue
        got = None
        for clause in parts[1].split(","):
            clause = clause.strip()
            if clause.startswith(">="):
                got = clause[2:].strip()
                break
        if got is None:
            too_low.append(
                f"{name}: recipe has no lower bound, setup.py requires "
                f">={wanted}")
            continue
        if Version(got) < Version(wanted):
            too_low.append(
                f"{name}: recipe >={got}, setup.py requires >={wanted}")

    assert not too_low, (
        "conda-forge recipe floors below setup.py's — a conda user would "
        "resolve to a release the PyPI package refuses:\n" + "\n".join(too_low)
    )


def test_no_conda_recipe_floor_is_below_the_one_setup_py_declares():
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    _assert_no_conda_recipe_floor_is_below_declared(
        recipe["requirements"]["run"])


def test_conda_floor_guard_rejects_a_bare_requirement():
    """Removing a conda lower bound must not be mistaken for no work.

    The original loop skipped every one-token entry before consulting
    setup.py, so changing ``shap >=0.47.0,<1.0`` to bare ``shap`` passed.
    """
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    mutated = [
        "shap" if _normalise(str(entry).split()[0]) == "shap" else entry
        for entry in recipe["requirements"]["run"]
    ]
    assert "shap" in mutated
    with pytest.raises(AssertionError, match=r"shap: recipe has no lower bound"):
        _assert_no_conda_recipe_floor_is_below_declared(mutated)
