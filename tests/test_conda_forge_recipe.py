"""Contracts for the one-time conda-forge recipe and update-bot setup."""

from __future__ import annotations

import ast
import re
from pathlib import Path

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
            return {
                _normalise(Requirement(requirement).name)
                for requirement in requirements
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
    assert conda_names == expected


def test_conda_recipe_is_noarch_and_uses_a_verified_tag_archive():
    text = RECIPE.read_text(encoding="utf-8")
    recipe = yaml.safe_load(text)
    assert recipe["build"]["noarch"] == "python"
    assert "--no-deps" in recipe["build"]["script"]
    assert "archive/refs/tags/v${{ version }}.tar.gz" in recipe["source"]["url"]
    assert re.fullmatch(r"[0-9a-f]{64}", recipe["source"]["sha256"])
    assert recipe["extra"]["recipe-maintainers"] == ["EinarOlafsson"]


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
