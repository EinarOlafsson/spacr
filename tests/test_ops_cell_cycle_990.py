"""The OPS cell-cycle option (item 990, alpha): stage, probability and flag per cell.

Synthetic nuclei carry planted 2N and 4N DNA content, condensed 4N metaphases, 2N
ana_telo_cyto daughter pairs and G2 prophase cells. Item 995's field-centred fit
(``spacr.measure._cell_table_phases_measure_fc``) is not on this base yet, so the tests
stand in for it with the same EM fit per well and no field shift.
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import ops_cell_cycle as cc
from spacr import ops_engine

PLATE = "P1"


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """Few permutations, one BLAS thread and item 995's call stood in for."""
    from spacr import measure

    monkeypatch.setattr(cc, "PERMUTATIONS", 20)
    monkeypatch.setattr(measure, "_cell_table_phases_measure_fc", _fake_995, raising=False)


def _fake_995(cells, *, dna="_dna_integrated", group="well", field="site", mitotic="mitotic",
              min_cells=50):
    """Item 995's OPS entry point without the field shift: one EM fit per group."""
    from spacr.measure import _fit_dna_content, _gate_dna_content

    for column in (dna, group, field, mitotic):
        if column not in cells.columns:
            raise ValueError(column)
    frame = pd.DataFrame(index=cells.index)
    frame["field_offset_log2"] = 0.0
    frame["dna_content_centred"] = cells[dna].to_numpy(float)
    for name in ("phase", "confidence", "posterior_G1", "posterior_S", "posterior_G2"):
        frame[name] = np.nan if name != "phase" else None
    fits = {}
    for name, index in cells.groupby(group).groups.items():
        content = cells.loc[index, dna].to_numpy(float)
        calm = ~cells.loc[index, mitotic].to_numpy(bool)
        fit = _fit_dna_content(pd.Series(content[calm]))
        phases, confidence, posterior = _gate_dna_content(content, fit)
        phases = np.where(calm, phases, "M")
        frame.loc[index, "phase"] = phases
        frame.loc[index, "confidence"] = confidence
        for k, ph in enumerate(("G1", "S", "G2")):
            frame.loc[index, f"posterior_{ph}"] = posterior[:, k]
        fits[name] = {"fit": fit, "fields_shifted": 0, "pooled": False}
    return frame, fits


SHARES = {"G1": 0.5, "S": 0.2, "G2": 0.18, "g2_prophase": 0.03, "prometaphase": 0.02,
          "metaphase": 0.03, "ana_telo_cyto": 0.04}


def _cells(seed=0, wells=("A1", "A2"), sites=6, per_site=220, shuffle_stage=False):
    """Synthetic cells with their true stage; daughters sit next to each other."""
    rng = np.random.default_rng(seed)
    rows = []
    names, shares = list(SHARES), np.array(list(SHARES.values()))
    for well in wells:
        for site in range(sites):
            label = 0
            for _ in range(per_site):
                stage = names[rng.choice(len(names), p=shares / shares.sum())]
                content = {"G1": 2.0, "S": rng.uniform(2.2, 3.8), "G2": 4.0, "g2_prophase": 4.0,
                           "prometaphase": 4.0, "metaphase": 4.0, "ana_telo_cyto": 2.0}[stage]
                content *= np.exp(rng.normal(0, 0.05))
                condensed = stage in ("prometaphase", "metaphase", "ana_telo_cyto")
                area = rng.normal({"prometaphase": 140, "metaphase": 80, "ana_telo_cyto": 90,
                                   "g2_prophase": 230}.get(stage, 200), 8)
                y, x = rng.uniform(0, 2000, 2)
                spots = rng.normal({"g2_prophase": 8, "prometaphase": 5, "metaphase": 2,
                                    "ana_telo_cyto": 2}.get(stage, 1), 0.5)
                for twin in range(2 if stage == "ana_telo_cyto" else 1):
                    label += 1
                    rows.append({"well": well, "site": site, "label": label,
                                 "stage": "interphase" if stage in ("G1", "S", "G2") else stage,
                                 "true_phase": stage, "nucleus_area": area,
                                 "nucleus_eccentricity": rng.normal(
                                     {"prometaphase": 0.65, "metaphase": 0.9}.get(
                                         stage, 0.8 if condensed else 0.5), 0.04),
                                 "nucleus_centroid_y": y + 12 * twin, "nucleus_centroid_x": x,
                                 "nucleus_dapi_integrated_intensity": 1000 * content,
                                 "nucleus_dapi_mean_intensity": 1000 * content / area,
                                 "nucleus_dapi_std_intensity": spots * 10 + rng.normal(0, 1)})
    frame = pd.DataFrame(rows)
    if shuffle_stage:
        frame["stage"] = rng.permutation(frame["stage"].to_numpy())
    return frame


def _base(tmp_path, seed=0):
    """A base checkpoint trained on planted labels."""
    cells = _cells(seed=seed, wells=("B1",), sites=8)
    bundle = cc.train_cell_cycle_model(cells, cells["stage"], input_mode="DNA", dna="dapi",
                                       seed=0, repeats=1, permutations=5)
    return cc.save_cell_cycle_checkpoint(bundle, str(tmp_path / "models"))


def _run(tmp_path, base, *, labels=None, folder="run", seed=3):
    """The ph_cell_cycle phase on a two-well plate."""
    destination = tmp_path / folder
    destination.mkdir(exist_ok=True)
    cells = _cells(seed=seed)
    for well, rows in cells.groupby("well"):
        rows.drop(columns=["well", "stage", "true_phase"]).to_parquet(
            ops_engine._ph_cells_path(str(destination), well), index=False)
    settings = {"ops_cell_cycle": "DNA", "ops_cell_cycle_base_model": base,
                "nucleus_channel": "dapi", "ops_cell_cycle_labels": labels or ""}
    db = str(destination / "measurements.db")
    out = cc.run_cell_cycle_phase(db, PLATE, ["A1", "A2"], settings, str(destination))
    from spacr.ops_store import read_table

    return out, cells, read_table(db, "ops_cell_cycle"), read_table(db, "ops_cell_cycle_checks")


def test_the_collapse_rule_follows_the_parent_map():
    assert cc.collapse_stage("metaphase", ["metaphase"]) == "mitotic"
    assert cc.collapse_stage("metaphase", ["metaphase", "mitotic"]) == "G2/M"
    assert cc.collapse_stage("g2_prophase", ["g2_prophase"]) == "G2"
    assert cc.collapse_stage("S", ["G1", "S", "G2"]) == "interphase"
    assert cc.collapse_stage("G2", ["G2", "interphase"]) == "interphase"
    assert cc.collapse_stage("artefact", ["artefact"]) == "artefact"
    assert cc.screen_collapses({"P1": ["metaphase"], "P2": ["ana_telo_cyto"]}) == [
        "ana_telo_cyto", "metaphase"]


def test_failed_checks_name_the_stages_to_collapse():
    def checks(**passed):
        return pd.DataFrame([{"check": k, "passed": v, "value": 0.95 if k == "S6" else 1.0}
                             for k, v in passed.items()])

    assert cc.failed_stages(checks(S1=True, S2=True, S4=True, S6=True)) == []
    assert cc.failed_stages(checks(S1=False, S2=False, S4=True)) == [
        "ana_telo_cyto", "metaphase", "prometaphase"]
    assert "mitotic" in cc.failed_stages(checks(S1=False, S2=True, S4=False))
    assert cc.failed_stages(checks(S6=False)) == ["S/G2"]
    low = pd.DataFrame([{"check": "S6", "passed": False, "value": 0.5}])
    assert cc.failed_stages(low) == ["G1", "G2", "S"]
    assert cc.failed_stages(checks(S1=None, S2=None, S3=None)) == []
    assert cc.failed_stages(checks(S3=False)) == [
        "ana_telo_cyto", "g2_prophase", "metaphase", "prometaphase"]


def test_planted_stages_pass_their_self_checks_and_are_called(tmp_path):
    out, cells, table, checks = _run(tmp_path, _base(tmp_path))
    assert out["collapsed"] == [] and set(table["cell_cycle_flag"]) == {"auto"}
    passed = dict(zip(checks["check"], checks["passed"]))
    assert passed["S1"] == 1 and passed["S2"] == 1 and passed["S6"] == 1
    called = cells[["well", "site", "label", "true_phase"]].merge(
        table[["well", "site", "label", "cell_cycle_stage"]], on=["well", "site", "label"])
    for stage in ("metaphase", "ana_telo_cyto", "G1", "G2"):
        rows = called[called["true_phase"] == stage]
        assert (rows["cell_cycle_stage"] == stage).mean() > 0.8, stage
    assert {"p_metaphase", "dna_log2_fc", "dna_phase_posterior_G1", "model_id"} <= set(table)
    assert (tmp_path / "run" / f"ops_cell_cycle_report_{PLATE}.html").is_file()


def test_shuffled_labels_fail_the_shuffled_check_and_collapse_to_parents(tmp_path):
    cells = _cells(seed=3, shuffle_stage=True)
    labels = cells.sample(300, random_state=0)[["well", "site", "label", "stage"]]
    labels.insert(0, "plate", PLATE)
    labels.to_csv(tmp_path / "labels.csv", index=False)
    out, _cells_, table, checks = _run(tmp_path, _base(tmp_path),
                                       labels=str(tmp_path / "labels.csv"))
    assert dict(zip(checks["check"], checks["passed"]))["S3"] == 0
    assert {"metaphase", "prometaphase", "ana_telo_cyto"} <= set(out["collapsed"])
    assert not table["cell_cycle_stage"].isin(["metaphase", "ana_telo_cyto"]).any()
    assert (table.loc[table["cell_cycle_stage"] == "mitotic", "cell_cycle_flag"]
            == "unresolved").all()


def test_true_labels_verify_their_stages(tmp_path):
    cells = _cells(seed=3)
    labels = cells.groupby("stage", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 60), random_state=0))[["well", "site", "label", "stage"]]
    labels.insert(0, "plate", PLATE)
    labels.to_csv(tmp_path / "labels.csv", index=False)
    _out, _c, table, checks = _run(tmp_path, _base(tmp_path), labels=str(tmp_path / "labels.csv"))
    assert dict(zip(checks["check"], checks["passed"]))["S3"] == 1
    assert (table.loc[table["cell_cycle_stage"] == "metaphase", "cell_cycle_flag"]
            == "verified").all()


def test_a_plate_collapse_reaches_every_plate_of_the_screen(tmp_path):
    base = _base(tmp_path)
    _run(tmp_path, base)
    db = str(tmp_path / "run" / "measurements.db")
    from spacr.ops_store import read_table, write_table

    checks = read_table(db, "ops_cell_cycle_checks")
    write_table(db, "ops_cell_cycle_checks",
                pd.concat([checks, checks.assign(plate="P0", collapsed="metaphase")]),
                if_exists="replace")
    out, _c, table, _k = _run(tmp_path, base)
    assert out["collapsed"] == ["metaphase"] and out["plate_collapsed"] == []
    assert not (table["cell_cycle_stage"] == "metaphase").any()


def test_a_rerun_of_the_same_settings_is_identical(tmp_path):
    base = _base(tmp_path)
    assert _base(tmp_path) == base
    first = _run(tmp_path, base, folder="a")
    second = _run(tmp_path, base, folder="b")
    pd.testing.assert_frame_equal(first[2], second[2])
    pd.testing.assert_frame_equal(first[3], second[3])


def test_a_checkpoint_with_a_wrong_hash_or_format_is_refused(tmp_path):
    base = _base(tmp_path)
    assert cc.load_cell_cycle_checkpoint(base)["card"]["format"] == cc.CHECKPOINT_FORMAT
    with open(os.path.join(base, "model.joblib"), "ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        cc.load_cell_cycle_checkpoint(base)
    other = _base(tmp_path / "second")
    card = json.load(open(os.path.join(other, "card.json")))
    card["format"] = "something-else"
    json.dump(card, open(os.path.join(other, "card.json"), "w"))
    with pytest.raises(ValueError):
        cc.load_cell_cycle_checkpoint(other)


def test_cnn_falls_back_to_the_tabular_learner_and_says_so(tmp_path, monkeypatch):
    base = _base(tmp_path)
    destination = tmp_path / "run"
    destination.mkdir()
    for well, rows in _cells(seed=3).groupby("well"):
        rows.drop(columns=["well", "stage", "true_phase"]).to_parquet(
            ops_engine._ph_cells_path(str(destination), well), index=False)
    out = cc.run_cell_cycle_phase(str(destination / "measurements.db"), PLATE, ["A1", "A2"],
                                  {"ops_cell_cycle": "DNA", "ops_cell_cycle_base_model": base,
                                   "ops_cell_cycle_model_family": "cnn",
                                   "nucleus_channel": "dapi"}, str(destination))
    assert any("tabular" in note and "cnn" in note for note in out["notes"])


def test_the_knockout_check_needs_the_arrest_set_enriched(tmp_path):
    cells = _cells(seed=5, wells=("A1",), sites=4, per_site=400)
    stage = np.full(len(cells), "interphase", object)
    rng = np.random.default_rng(0)
    genes = np.array([f"G{i}" for i in rng.integers(0, 40, len(cells))], object)
    genes[:300] = "KIF11"
    genes[300:600] = "nontargeting"
    stages = pd.DataFrame({"model_stage": stage}, index=cells.index)
    dna = pd.DataFrame({"dna_content_centred": cells["nucleus_dapi_integrated_intensity"]})
    fit = _fake_995(cells.assign(_dna_integrated=cells["nucleus_dapi_integrated_intensity"],
                                 mitotic=False))[1]
    results = {}
    for fold in (1, 3):
        stage = np.full(len(cells), "interphase", object)
        stage[rng.random(len(cells)) < 0.05] = "metaphase"
        stage[:300] = np.where(rng.random(300) < 0.05 * fold, "metaphase", "interphase")
        stages["model_stage"] = stage
        checks = cc.self_checks(cells.assign(gene=genes), stages, dna, fit, card={},
                                permutations=100, seed=1)
        results[fold] = checks.set_index("check").loc["S5", "passed"]
    assert results == {1: False, 3: True}


def test_none_leaves_the_run_unchanged(tmp_path):
    """ops_cell_cycle None writes the same files as a run without the key."""
    from tests.test_ops_phenotype_path import test_merge_and_aggregate_run_without_a_sequencing_folder as run

    outputs = {}
    for tag, extra in (("without", {}), ("none", {"ops_cell_cycle": None})):
        folder = tmp_path / tag
        folder.mkdir()
        original = ops_engine.run_ops

        def patched(settings, **kwargs):
            """run_ops with the extra settings."""
            return original({**settings, **extra}, **kwargs)

        ops_engine.run_ops = patched
        try:
            run(folder)
        finally:
            ops_engine.run_ops = original
        outputs[tag] = {p.name: (gzip.decompress(p.read_bytes()) if p.suffix == ".gz"
                                 else p.read_bytes())
                        for p in sorted(folder.iterdir())
                        if p.suffix in (".csv", ".parquet") and "report" not in p.name}
        assert not list(folder.glob("ops_ph_cell_cycle_*.json"))
    assert outputs["without"] == outputs["none"] and outputs["none"]


def test_the_settings_are_alpha_with_short_tooltips():
    from spacr.ops_settings import OPS_CATEGORIES, OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import ALPHA_FEATURES

    keys = ALPHA_FEATURES[990]["settings"]
    assert set(keys) == {k for k in OPS_DEFAULTS if k.startswith("ops_cell_cycle")}
    assert set(keys) == set(OPS_CATEGORIES["Cell-cycle stages"])
    assert OPS_DEFAULTS["ops_cell_cycle"] is None and OPS_DEFAULTS["ops_cell_cycle_tabular"] == "hgb"
    for key in keys:
        assert len(OPS_TOOLTIPS[key]) <= 600 and "Alpha" in OPS_TOOLTIPS[key]
    assert "ops_mitotic_source 'cell_cycle'" in OPS_TOOLTIPS["ops_cell_cycle"]
