"""Item 969: the OPS settings panel's layout, labels, model choices and power curve.

The OPS panel hides the "Ops" every label started with, nests its settings in
subcategories of at most twenty, keeps plate layout and controls under OPS
input, offers the model zoo for its three Cellpose models, names the nuclei
model for what it segments (old files still load), and leaves the e-test
power curve to the Power module's OPS power curve section.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr import ops_settings  # noqa: E402
from spacr.qt.screens import settings_model as sm  # noqa: E402

POWER_KEYS = {"ops_power_cells", "ops_power_seeds", "ops_power_retrieval",
              "ops_power_gene_list", "ops_power_reference_rows"}
MODEL_KEYS = ("ops_sbs_nucleus_model", "ops_ph_nucleus_model", "ops_ph_cell_model")


def _rendered():
    """The OPS panel's categories, as the panel lays them out."""
    keys = set(sm.resolve_default_settings("ops"))
    return {title: [k for k in rows if k in keys]
            for title, rows in sm.categories_for_app("ops", sm.get_categories()).items()
            if any(k in keys for k in rows)}


def test_the_ops_prefix_is_hidden_on_the_ops_panel_only():
    assert sm.app_setting_label("ops", "ops_read_threshold") == "Read threshold"
    assert sm.app_setting_label("ops", "ops_gpu") == "GPU"
    assert sm.app_setting_label("ops", "ops_ph_zernike") == "Ph Zernike moments"
    assert sm.app_setting_label("ops", "dst_root") == "Dst root"
    assert sm.app_setting_label("ops", "ops_sbs_nucleus_model") == \
        "SBS nucleus segmentation model"
    assert sm.app_setting_label("", "ops_read_threshold") == "Ops read threshold"
    assert sm.app_setting_label("spatial_masks", "ops_gpu") == "Ops GPU"
    widgets = sm.SettingsWidgets.__new__(sm.SettingsWidgets)
    widgets.app_key = "ops"
    assert widgets._label_for("ops_min_reads") == "Min reads"
    assert sm._translated_setting_name("ops_min_reads", "en", "ops") == "Min reads"


def test_every_ops_category_holds_at_most_twenty_settings():
    rendered = _rendered()
    too_big = {title: len(keys) for title, keys in rendered.items() if len(keys) > 20}
    assert not too_big, too_big
    parents = sm._category_parents("ops")
    for title in ops_settings.OPS_CATEGORIES:
        if title not in ("OPS input", "OPS power curve"):
            assert parents.get(title, "").startswith("OPS "), title
    assert {parents[t] for t in ops_settings.OPS_CATEGORIES if t in parents} == {
        "OPS input", "OPS alignment", "OPS decoding", "OPS phenotype", "OPS merge",
        "OPS aggregate", "OPS cluster", "OPS evaluation"}


def test_the_subcategories_are_new_names_or_the_shared_plate_layout():
    """A subcategory sharing a title with another module's category would share its help."""
    from spacr import settings as shared

    others = {title for title, keys in shared.categories.items()
              if not set(keys) & set(ops_settings.OPS_DEFAULTS)}
    clash = set(ops_settings.OPS_CATEGORIES) & others
    assert not clash, clash


def test_plate_layout_and_controls_sit_in_ops_input():
    rendered = _rendered()
    assert rendered["Plate Layout & Controls"] == [
        "plate", "ops_design", "ops_control_prefix", "ops_genotype_table"]
    assert sm._category_parents("ops")["Plate Layout & Controls"] == "OPS input"
    assert sm._category_parents("ops")["OPS performance"] == "OPS input"
    assert "Plate Layout & Controls" not in sm._category_parents("mask")
    flat = [sm.SettingsSection(title, [(k, None) for k in keys])
            for title, keys in rendered.items()]
    nested = sm._nest_sections(flat, "ops")
    top = [section.title for section in nested]
    assert top[0] == "OPS input" and "OPS performance" not in top
    first = nested[0]
    assert [child.title for child in first.children][:2] == [
        "Plate Layout & Controls", "Cell Painting Gallery input"]


def test_the_three_ops_models_offer_the_model_zoo():
    from spacr import model_zoo
    from spacr.qt.screens.app_screen import AppScreen

    for key in MODEL_KEYS:
        assert AppScreen._takes_a_cellpose_checkpoint(key)
        assert AppScreen._model_kinds_for(key) == ("cellpose",)
    kinds = model_zoo._model_input_kinds("ops", dict.fromkeys(MODEL_KEYS, "cpsam"))
    assert kinds == dict.fromkeys(MODEL_KEYS, ("cellpose",))
    assert ops_settings.OPS_DEFAULTS["ops_sbs_nucleus_model"] == "cpsam"
    assert "cellpose_model" not in ops_settings.OPS_DEFAULTS


def test_an_old_cellpose_model_setting_loads_into_the_nuclei_model(tmp_path, monkeypatch):
    from spacr import ops_engine
    from spacr.cli import load_settings_file
    from spacr.settings import _fold_renamed_settings, surviving_setting_name

    assert surviving_setting_name("cellpose_model") == ("ops_sbs_nucleus_model",)
    record = {"requested_value": "custom"}
    folded = _fold_renamed_settings({
        "cellpose_model": "custom",
        "model_metadata": {"schema_version": 1, "settings": {"cellpose_model": record}}})
    assert folded == {"ops_sbs_nucleus_model": "custom", "model_metadata": {
        "schema_version": 1, "settings": {"ops_sbs_nucleus_model": record}}}
    csv = tmp_path / "old.csv"
    csv.write_text("Key,Value\ncellpose_model,nuclei_v2\nops_layout,cpg_cropped\n")
    seen = {}
    monkeypatch.setattr(ops_engine, "_run_cpg",
                        lambda settings, phases: seen.update(settings) or {})
    ops_engine.run_ops(load_settings_file(str(csv)))
    assert seen["ops_sbs_nucleus_model"] == "nuclei_v2" and "cellpose_model" not in seen


def test_the_power_curve_left_the_ops_panel_for_the_power_module(qtbot, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.power import OPS_POWER_FIELDS, PowerScreen, ops_power_settings
    from spacr.settings import ALPHA_FEATURES

    shown = {k for keys in _rendered().values() for k in keys}
    assert not POWER_KEYS & shown
    assert POWER_KEYS <= set(sm.resolve_default_settings("ops"))
    assert "PowerOpsCurve" in ALPHA_FEATURES[917]["widgets"]
    assert POWER_KEYS <= set(OPS_POWER_FIELDS)
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    hidden = PowerScreen(threaded=False)
    qtbot.addWidget(hidden)
    assert hidden._ops_power_box.isHidden()
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    screen = PowerScreen(threaded=False)
    qtbot.addWidget(screen)
    assert not screen._ops_power_box.isHidden()
    values = {key: (f.isChecked() if hasattr(f, "isChecked") else f.text())
              for key, f in screen._ops_power_fields.items()}
    assert ops_power_settings(values) == ops_settings.ops_defaults()
    assert ops_power_settings({**values, "ops_power_cells": "10, all"})[
        "ops_power_cells"] == [10, None]


def _tree(root: Path) -> dict:
    """Every file under ``root`` by relative path, with its SHA-256.

    A ``.gz`` file is hashed decompressed: its header carries the time it was
    written, so two runs of the same phase never match byte for byte there.
    """
    import gzip

    return {str(p.relative_to(root)): hashlib.sha256(
                gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
            ).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_power_section_writes_what_the_ops_etest_run_writes(tmp_path):
    from spacr.cli import load_settings_file
    from spacr.ops_engine import run_ops
    from spacr.qt.screens.power import (OPS_POWER_FIELDS, _ops_power_text,
                                        ops_power_settings, run_ops_power_curve)
    from tests.test_ops_etest_power_917 import _cells_folder

    folder = _cells_folder(tmp_path)
    dst = tmp_path / "run"
    csv = tmp_path / "ops.csv"
    pd.DataFrame({"Key": ["dst_root", "ops_etest_cells", "ops_etest_reference_cells",
                          "ops_etest_null", "ops_etest_holdout", "ops_power_cells",
                          "ops_power_seeds", "ops_etest_metric"],
                  "Value": [str(dst), str(folder), "500", "99", "50", "10,all", "[0, 1]",
                            "euclidean"]}).to_csv(csv, index=False)
    run_ops(ops_settings.ops_defaults(dict(load_settings_file(str(csv)))), phases=["etest"])
    before = _tree(dst)
    assert any(name.endswith("power_hits.csv") for name in before)
    for path in dst.rglob("*"):
        if path.is_file():
            path.unlink()
    loaded = ops_settings.ops_defaults(dict(load_settings_file(str(csv))))
    values = {key: (bool(loaded[key]) if key == "ops_power_retrieval"
                    else _ops_power_text(loaded[key])) for key in OPS_POWER_FIELDS}
    payload = {"settings": ops_power_settings(values, str(csv))}
    run_ops_power_curve(payload)
    assert _tree(dst) == before
    core = ["genes", "hits_q05", "hit_pct", "genes_p05_pct", "null_cells"]
    assert np.isfinite(pd.read_csv(dst / "results" / "ops_etest" / "power_hits.csv")[core]
                       .to_numpy(float)).all()
