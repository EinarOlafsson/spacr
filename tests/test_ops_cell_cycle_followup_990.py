"""Full-population model inputs, conditional S2, S5 power and refreshed model cards."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def normalization_case(monkeypatch):
    """An enriched label subset whose robust z differs markedly from the full wells."""
    from spacr import ops_cell_cycle as cc

    full = pd.DataFrame({"plate": np.repeat(["P1", "P2"], 80), "well": "A1",
                         "site": np.tile(np.repeat([1, 2], 40), 2),
                         "label": np.tile(np.arange(80), 2) + 1,
                         "nucleus_dapi_mean_intensity": np.r_[np.arange(80), 1000 + np.arange(80)]})
    selected = np.r_[np.arange(60, 80), np.arange(140, 160)]
    labelled = full.iloc[selected].copy()
    labelled["stage"] = np.tile(["interphase", "metaphase"], 20)
    features = ["nucleus_dapi_mean_intensity"]
    captures = []

    class Model:
        """A learner exposing the matrix supplied to prediction without costly model fitting."""
        classes_ = np.arange(len(cc.CLASSES))

        def predict_proba(self, values):
            """Record query features and emit a fixed valid probability distribution."""
            captures.append(np.array(values, copy=True))
            out = np.zeros((len(values), len(cc.CLASSES)))
            out[:, 0] = 1
            return out

    def fit(tabular, seed, x, y, weight=None):
        """Record the real training path matrix and return its lightweight learner."""
        captures.append(np.array(x, copy=True))
        return Model()

    def out_of_fold(tabular, seed, x, y, groups, weight=None, scored=None, **kwargs):
        """Record the S3 model inputs while retaining genuine class encodings."""
        captures.append(np.array(x, copy=True))
        return np.array(y, copy=True)

    def shuffled(*args, **kwargs):
        """Return a nonempty finite control without fitting redundant models."""
        return np.array([0.1, 0.2]), {}

    def bootstrap(*args, **kwargs):
        """Leave interval computation to the existing model tests."""
        return {}

    monkeypatch.setattr(cc, "_fit", fit)
    monkeypatch.setattr(cc, "_out_of_fold", out_of_fold)
    monkeypatch.setattr(cc, "_shuffled_f1", shuffled)
    monkeypatch.setattr(cc, "_bootstrap_f1", bootstrap)
    return SimpleNamespace(cc=cc, full=full, labelled=labelled, selected=selected,
                           features=features, captures=captures, model=Model())


def test_enriched_training_and_prediction_share_the_full_well_matrix(normalization_case):
    """All four labelled consumers match full-well prediction, including repeated well names."""
    case = normalization_case
    cc, full, labelled, features = case.cc, case.full, case.labelled, case.features
    expected = cc._matrix(full, features)[case.selected]
    wrong = cc._matrix(labelled, features)
    assert not np.allclose(expected, wrong)
    cc.predict_stages({"model": case.model, "features": features}, full)
    np.testing.assert_array_equal(case.captures[-1][case.selected], expected)
    bundle = cc.train_cell_cycle_model(labelled, labelled.stage, dna="dapi",
                                       repeats=1, permutations=2, reference_cells=full)
    np.testing.assert_array_equal(bundle["training"][features], expected)
    tuned = cc.fine_tune({"card": bundle["card"], "features": features}, labelled,
                         labelled.stage, reference_cells=full)
    np.testing.assert_array_equal(tuned["training"][features], expected)
    rows = cc._labelled_rows(labelled, labelled.stage, features, "screen", reference_cells=full)
    np.testing.assert_array_equal(rows[features], expected)
    cc._labelled_scores({"features": features}, labelled, {"tabular": "hgb", "seed": 0},
                         reference_cells=full, permutations=2)
    np.testing.assert_array_equal(case.captures[-1], expected)


def test_reference_omitting_a_plate_is_refused(normalization_case):
    """A repeated well name cannot silently normalize against a different plate."""
    case = normalization_case
    with pytest.raises(ValueError, match="plate"):
        case.cc._matrix(case.labelled, case.features,
                        reference_cells=case.full[case.full.plate == "P1"])


def _pair_case():
    """Two fields include eligible pairs, an isolated daughter and nearby ineligible cells."""
    cells = pd.DataFrame({"plate": "P1", "well": "A1", "site": np.repeat([1, 2], 6),
                          "nucleus_centroid_y": np.tile([0, 1, 100, 101, 300, 301], 2),
                          "nucleus_centroid_x": 0.0, "nucleus_area": np.pi})
    two_n = np.tile([True, True, True, True, False, False], 2)
    called = np.tile([True, True, False, True, True, False], 2)
    log_dna = np.tile([1.0, 1.1, 1.0, 1.5, 2.0, 2.0], 2)
    return cells, two_n, called, log_dna


def _brute_pair_share(cells, called, two_n, log_dna):
    """Evaluate the specified directed geometry and DNA gates independently by all pairs."""
    points = cells[["nucleus_centroid_y", "nucleus_centroid_x"]].to_numpy(float)
    selected = np.flatnonzero(called & two_n)
    hits = 0
    for row in selected:
        peers = selected[(cells.site.to_numpy()[selected] == cells.site.iloc[row])
                         & (selected != row)]
        radius = 5 * np.sqrt(cells.nucleus_area.iloc[row] / np.pi)
        hits += bool(np.any((np.linalg.norm(points[peers] - points[row], axis=1) <= radius)
                            & (np.abs(log_dna[peers] - log_dna[row]) <= 0.3)))
    return hits / len(selected)


def test_cached_s2_pairs_match_independent_geometry_for_each_grouped_permutation():
    """Bincount reductions preserve the original statistic for every tested label draw."""
    from spacr import ops_cell_cycle as cc

    cells, two_n, called, log_dna = _pair_case()
    groups = [rows[two_n[rows]] for rows in cc._field_rows(cells)]
    neighbours = cc._pair_neighbours(cells, two_n, log_dna, groups)
    rng = np.random.default_rng(0)
    for _ in range(15):
        shuffled = called.copy()
        for rows in groups:
            shuffled[rows] = called[rng.permutation(rows)]
            assert shuffled[rows].sum() == called[rows].sum()
        assert cc._pair_share(cells, shuffled, two_n, log_dna,
                              neighbours=neighbours) == _brute_pair_share(
                                  cells, shuffled, two_n, log_dna)


def test_s2_builds_one_graph_and_permutes_only_eligible_rows(monkeypatch):
    """The real self-check reuses one graph and its null leaves all 4N labels unchanged."""
    from spacr import ops_cell_cycle as cc

    cells, two_n, called, _log_dna = _pair_case()
    stages = pd.DataFrame({"model_stage": np.where(called, "ana_telo_cyto", "interphase")})
    dna = pd.DataFrame({"dna_content_centred": np.where(two_n, 2.0, 4.0)})
    fits = {"A1": {"fit": SimpleNamespace(g1=2.0, g2=4.0)}}
    original_graph, original_share = cc._pair_neighbours, cc._pair_share
    graphs, permutations = [], []

    def graph(*args, **kwargs):
        """Count graph construction without changing the spatial calculation."""
        graphs.append(True)
        return original_graph(*args, **kwargs)

    def share(cells, labels, two_n, dna, **kwargs):
        """Observe actual label draws before using the real shared reducer."""
        permutations.append(labels.copy())
        return original_share(cells, labels, two_n, dna, **kwargs)

    def quality(*args):
        """Keep this regression focused on S2 rather than unrelated DNA model fitting."""
        return 1.0, 2.0

    monkeypatch.setattr(cc, "_pair_neighbours", graph)
    monkeypatch.setattr(cc, "_pair_share", share)
    monkeypatch.setattr(cc, "_fit_quality", quality)
    result = cc.self_checks(cells, stages, dna, fits, card={}, permutations=11, arrest_genes=None)
    assert len(graphs) == 1 and len(permutations) == 12
    for labels in permutations:
        np.testing.assert_array_equal(labels[~two_n], called[~two_n])
    assert len(result.attrs["samples"]["S2"][1]) == 11


def test_paper_s5_inputs_give_the_reported_required_counts():
    """The specified one-sample approximation rounds the measured example up to 2867."""
    from spacr import ops_cell_cycle as cc

    measured = cc._s5_power(0.0010413, enrichment=2.7597)
    assert measured["required_cells"] == 2867
    assert measured["alternative"] == pytest.approx(0.00287367561)
    assert cc._s5_power(0.0010413)["required_cells"] == 2307
    assert cc._s5_power(0.0)["required_cells"] == float("inf")


def test_112_arrest_cells_are_underpowered_and_do_not_collapse_stages(monkeypatch):
    """A small arrest set is reported as UNDERPOWERED even when its observed enrichment is large."""
    from spacr import ops_cell_cycle as cc

    count = 112 + 10000
    cells = pd.DataFrame({"well": "A1", "site": 1,
                          "gene": ["KIF11"] * 112 + ["nontargeting"] * 10000})
    stages = pd.DataFrame({"model_stage": ["metaphase"] * 30 + ["interphase"] * 82
                          + ["metaphase"] * 10 + ["interphase"] * 9990})
    dna = pd.DataFrame({"dna_content_centred": np.full(count, 4.0)})

    def quality(*args):
        """Hold the unrelated DNA fit quality fixed."""
        return 1.0, 2.0

    monkeypatch.setattr(cc, "_fit_quality", quality)
    checks = cc.self_checks(cells, stages, dna,
                            {"A1": {"fit": SimpleNamespace(g1=2.0, g2=4.0)}},
                            card={}, permutations=3)
    row = checks.set_index("check").loc["S5"]
    assert row.status == "UNDERPOWERED" and cc._verdict(row.passed) is None
    assert row.judged_cells == 112 and row.required_cells > 2300
    assert row.baseline == 0.001 and row.enrichment == 3.0
    assert row.alpha == 0.05 and row.power == 0.8
    assert "metaphase" not in cc.failed_stages(checks)


def test_checkpoint_card_replaces_old_revision_even_after_a_different_dump(tmp_path, capsys):
    """Canonical prediction identity survives byte changes and every card save is published."""
    from sklearn.dummy import DummyClassifier
    from spacr import ops_cell_cycle as cc

    training = pd.DataFrame({"feature": np.arange(10, dtype=float),
                             "stage": ["interphase", "metaphase"] * 5})
    model = DummyClassifier(strategy="prior").fit(training[["feature"]].to_numpy(), [0, 3] * 5)
    bundle = {"model": model, "features": ["feature"], "training": training,
              "card": {"input": "DNA", "format": cc.CHECKPOINT_FORMAT,
                       "classes": list(cc.CLASSES), "reference": {"S1": 0.8}}}
    path = Path(cc.save_cell_cycle_checkpoint(bundle, str(tmp_path)))
    first_model = (path / "model.joblib").read_bytes()
    loaded = cc.load_cell_cycle_checkpoint(str(path))
    loaded["card"].pop("checkpoint_id")
    loaded["model"].serialization_revision = 2
    loaded["card"]["reference"] = {"S1": 0.9}
    again = Path(cc.save_cell_cycle_checkpoint(loaded, str(tmp_path)))
    assert again == path
    assert first_model != (path / "model.joblib").read_bytes()
    current = cc.load_cell_cycle_checkpoint(str(path))
    assert current["card"]["reference"] == {"S1": 0.9}
    loaded["card"]["reference"] = {"S1": 0.95}
    assert Path(cc.save_cell_cycle_checkpoint(loaded, str(tmp_path))) == path
    assert cc.load_cell_cycle_checkpoint(str(path))["card"]["reference"] == {"S1": 0.95}
    assert len(list(tmp_path.glob("cell_cycle_tabular_*"))) == 1
    assert not list(tmp_path.glob(".cell_cycle_*"))
    assert "saved with current card" in capsys.readouterr().out


def test_ph_measure_adds_exact_nuclear_geometry_and_preserves_every_old_column(monkeypatch):
    """The production worker adds S2 mask geometry while existing table bytes remain equal."""
    from spacr import ops_engine, ops_sample
    from tests.test_measure_ops_features_934 import _ops_field
    from skimage.measure import regionprops_table

    nuclei, cells, image, names = _ops_field()

    def field(*args, **kwargs):
        """Supply the unchanged synthetic image through the existing field reader."""
        return image

    def masks(*args, **kwargs):
        """Supply the exact nuclear and cell masks through the existing mask reader."""
        return {"nuclei": nuclei, "cells": cells}

    monkeypatch.setattr(ops_engine, "_ph_read_field", field)
    monkeypatch.setattr(ops_engine, "_ph_read_masks", masks)
    task = {"site": 7, "files": {}, "channels": names, "mask_path": "unused", "texture": False}
    old = ops_sample.measure_phenotype_field(nuclei, cells, image, names, texture=False)
    old.insert(0, "site", 7)
    actual, _unreadable, _shape = ops_engine._ph_measure_table(task)
    pd.testing.assert_frame_equal(actual[old.columns], old, check_exact=True)
    assert set(actual) - set(old) <= {"nucleus_centroid_y", "nucleus_centroid_x", "nucleus_area"}
    props = pd.DataFrame(regionprops_table(nuclei, properties=("label", "centroid", "area")))
    props = props.set_index("label").loc[actual.label]
    np.testing.assert_array_equal(actual.nucleus_centroid_y, props["centroid-0"])
    np.testing.assert_array_equal(actual.nucleus_centroid_x, props["centroid-1"])
    np.testing.assert_array_equal(actual.nucleus_area, props.area)


def _current_followup_api(entries, docs=None):
    """Bind exactly five revised public bodies to their unchanged historical sources.

    :param entries: original item-990 arrival bodies after unrelated approved adapters.
    :param docs: optional canonical source inventory from the caller.
    :returns: the same keys with five exact reviewed source revisions.
    """
    import ast
    import hashlib
    import json

    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "tests/data/release_contracts"
                           / "990fix_full_well_api_sources_2026-11-22.json").read_text())
    rows = contract["symbols"]
    assert set(rows) == {"spacr.ops_cell_cycle." + name for name in (
        "train_cell_cycle_model", "fine_tune", "save_cell_cycle_checkpoint",
        "self_checks", "run_cell_cycle_phase")}
    tree = ast.parse((root / "spacr/ops_cell_cycle.py").read_text())
    owners = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    constants = {n.targets[0].id: ast.dump(n.value, include_attributes=False)
                 for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name)}
    result = dict(entries)
    for symbol, row in rows.items():
        assert entries[symbol] == row["before"]
        assert hashlib.sha256(entries[symbol].encode()).hexdigest() == row["before_sha256"]
        node = owners[symbol.rsplit(".", 1)[1]]
        body = ast.get_docstring(node)
        assert body == row["source"]
        assert hashlib.sha256(body.encode()).hexdigest() == row["source_sha256"]
        expected_args = ast.parse("def reviewed(" + row["signature"] + "): pass").body[0].args
        assert ast.dump(node.args, include_attributes=False) == ast.dump(expected_args, include_attributes=False)
        assert ast.dump(node.returns, include_attributes=False) == row["return_annotation_ast"]
        assert {key: constants[key] for key in row["default_bindings"]} == row["default_bindings"]
        if docs is not None:
            assert docs[symbol] == body
        result[symbol] = body
    return result


def test_followup_api_preserves_all_historical_keys():
    """The source-only adapter revises five exact bodies and preserves all twenty symbols."""
    import json

    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "tests/data/release_contracts"
                      / "990_ops_cell_cycle_api_arrivals_2026-11-21.json").read_text())
    revised = _current_followup_api(old)
    assert len(revised) == len(old) == 20
    assert {key for key in old if revised[key] != old[key]} == {
        "spacr.ops_cell_cycle." + name for name in ("train_cell_cycle_model", "fine_tune",
            "save_cell_cycle_checkpoint", "self_checks", "run_cell_cycle_phase")}
    wrong = dict(old)
    wrong["spacr.ops_cell_cycle.self_checks"] += " changed"
    with pytest.raises(AssertionError):
        _current_followup_api(wrong)


@pytest.mark.parametrize("original,replacement", [
    ("Checks S1-S6 of one plate.", "Changed check body."),
    ("arrest_enrichment: float = 3.0", "arrest_enrichment: float = 4.0"),
    ("PERMUTATIONS = 500", "PERMUTATIONS = 500.0"),
    ("\n                           reference_cells: Optional[pd.DataFrame] = None",
     "\n                           reference_cells: Optional[pd.DataFrame] = False"),
    ("repeats: int = 5", "repeats: int = 6"),
])
def test_followup_api_rejects_unreviewed_body_signature_and_default_changes(monkeypatch, original, replacement):
    """The approved revision cannot hide a changed body, optional parameter or default type."""
    import json

    root = Path(__file__).resolve().parents[1]
    entries = json.loads((root / "tests/data/release_contracts"
                          / "990_ops_cell_cycle_api_arrivals_2026-11-21.json").read_text())
    source = root / "spacr/ops_cell_cycle.py"
    read = Path.read_text

    def changed(path, *args, **kwargs):
        """Mutate only the apparent production source while retaining immutable fixtures."""
        text = read(path, *args, **kwargs)
        if path == source:
            assert original in text
            return text.replace(original, replacement, 1)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(AssertionError):
        _current_followup_api(entries)


def test_queue_inherited_and_followup_adapters_compose_in_fixed_order():
    """Each reviewed adapter accepts its predecessor and revises only its own symbols."""
    import json
    from tests.test_ops_cell_cycle_queue_990 import (
        _current_queue_api, _current_inherited_module_api)

    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "tests/data/release_contracts"
                      / "990_ops_cell_cycle_api_arrivals_2026-11-21.json").read_text())
    queue = _current_queue_api(old)
    inherited = _current_inherited_module_api(queue)
    followup = _current_followup_api(inherited)
    assert set(old) == set(queue) == set(inherited) == set(followup)
    assert {key for key in old if old[key] != queue[key]} == {
        "spacr.ops_cell_cycle.write_label_queue"}
    assert {key for key in queue if queue[key] != inherited[key]} <= {
        "spacr.ops_cell_cycle"}
    assert {key for key in inherited if inherited[key] != followup[key]} == {
        "spacr.ops_cell_cycle." + name for name in ("train_cell_cycle_model", "fine_tune",
            "save_cell_cycle_checkpoint", "self_checks", "run_cell_cycle_phase")}
