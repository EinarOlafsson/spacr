"""The package has one splitter, and a well never lands on both sides.

Instruction 94's promise is a property of the whole package rather than of
the five call sites it started from: a split written later that reaches
``train_test_split`` directly reports an optimistic score with nothing on
screen to say so.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

import spacr.annotation_umap_qc as qc

SPACR = pathlib.Path(qc.__file__).resolve().parent
SHARED = "classifier_evaluation.py"


def _direct_callers() -> list[str]:
    """Modules calling ``train_test_split`` outside the shared splitter."""
    guilty = []
    for path in sorted(SPACR.rglob("*.py")):
        if path.name == SHARED:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue
            if "train_test_split(" in stripped:
                guilty.append(f"{path.relative_to(SPACR)}: {stripped}")
    return guilty


def test_no_module_splits_outside_the_shared_splitter():
    assert _direct_callers() == []


@pytest.fixture()
def recording(monkeypatch):
    """Capture the row ids each side of the split is embedded with."""
    import spacr.hyperparam as hyperparam

    seen: list[set[int]] = []

    def embed(values, params, seed):
        seen.append({int(v) for v in np.asarray(values)[:, 0]})
        return np.asarray(values, dtype=float)

    def scores(values, embedding, marks, neighbours):
        return {"silhouette": 0.5}

    monkeypatch.setattr(hyperparam, "_default_umap_embed", embed)
    monkeypatch.setattr(hyperparam, "_umap_scores", scores)
    return seen


def _controls(n_wells: int = 4, per_well: int = 3):
    """Two classes over ``n_wells`` wells each, one row id per cell."""
    features, marks, wells = [], [], []
    row = 0
    for mark in (qc.POSITIVE, qc.NEGATIVE):
        for well in range(n_wells):
            for _ in range(per_well):
                features.append([row, float(well), float(row)])
                marks.append(mark)
                wells.append(f"{mark}-{well}")
                row += 1
    return np.asarray(features, dtype=float), marks, wells


def test_a_control_well_never_appears_on_both_sides(recording):
    features, marks, wells = _controls()
    result = qc.fit_on_controls(features, marks,
                                recipes=[{"n_neighbors": 5, "min_dist": 0.1}],
                                groups=wells)
    assert "error" not in result, result
    assert result["split_level"] == "well"
    fitted, held = recording[0], recording[1]
    lookup = {i: w for i, w in enumerate(wells)}
    assert not ({lookup[i] for i in fitted} & {lookup[i] for i in held})


def test_without_groups_the_result_says_it_split_per_object(recording):
    features, marks, _ = _controls()
    result = qc.fit_on_controls(features, marks,
                                recipes=[{"n_neighbors": 5, "min_dist": 0.1}])
    assert result["split_level"] == "cell"


def test_one_well_per_class_is_refused_rather_than_randomised(recording):
    features, marks, wells = _controls(n_wells=1, per_well=6)
    result = qc.fit_on_controls(features, marks,
                                recipes=[{"n_neighbors": 5, "min_dist": 0.1}],
                                groups=wells)
    assert "error" in result
    assert "well" in result["error"]


def test_a_group_is_required_for_every_control_cell(recording):
    features, marks, wells = _controls()
    result = qc.fit_on_controls(features, marks,
                                recipes=[{"n_neighbors": 5, "min_dist": 0.1}],
                                groups=wells[:-1])
    assert result["error"] == "one group is needed per control cell"
