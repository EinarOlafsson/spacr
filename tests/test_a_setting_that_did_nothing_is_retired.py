"""Item 364: `infection_pca_n_clusters` is withdrawn.

It was a live setting a user could set, and its own tooltip said it changed
nothing: the pca/umap/tsne QC always ran KMeans with exactly two clusters.
Nothing in spaCR ever read it. The maintainer retired it on 2026-09-20.

What these hold is the pair that matters: it is gone from everywhere a
setting is offered, and a settings file saved before today still loads and
is told what happened to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RETIRED = "infection_pca_n_clusters"
ROOT = Path(__file__).resolve().parents[1]


def test_it_is_no_longer_a_setting_anyone_can_set():
    from spacr import settings

    assert RETIRED not in settings.expected_types
    assert RETIRED not in settings.descriptions
    assert RETIRED not in settings.motility_advanced_settings


def test_an_old_settings_file_still_loads_and_is_told_what_happened():
    """The standing rule is to migrate rather than to break."""
    from spacr import validate

    assert RETIRED in validate.RETIRED_SETTINGS
    assert validate.RETIRED_SETTINGS[RETIRED] == "", (
        "it has no replacement: nothing else does what it claimed to do")
    problems = validate._check_retired_keys({RETIRED: 4})
    assert problems, "a retired key must be reported, not silently ignored"
    assert any(problem.setting == RETIRED for problem in problems)


def test_it_is_gone_from_the_settings_panel_and_the_catalogs():
    from spacr.qt.screens import settings_model

    text = (ROOT / "spacr" / "qt" / "screens" / "settings_model.py").read_text()
    assert RETIRED not in text
    for catalog in sorted((ROOT / "spacr" / "qt" / "i18n_catalogs").glob("*.py")):
        assert RETIRED not in catalog.read_text(encoding="utf-8"), catalog.name


def test_the_generated_indexes_no_longer_point_at_it():
    """Both are generated, and a stale row here sends Help at a setting that
    does not exist."""
    assert RETIRED not in (
        ROOT / "spacr" / "qt" / "help_api_index.py").read_text(encoding="utf-8")
    assert RETIRED not in (
        ROOT / "docs" / "setting_consumers.json").read_text(encoding="utf-8")


def test_the_settings_it_sat_beside_are_untouched():
    """Its neighbours in the Infection Clustering group are real settings and
    this retirement must not have taken any of them with it."""
    from spacr import settings

    for neighbour in ("infection_pca_random_state", "infection_pca_method",
                      "infection_pca_pathogen_weight",
                      "infection_pca_max_cells"):
        assert neighbour in settings.expected_types, neighbour
