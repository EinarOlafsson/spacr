"""Item 288: Plaque Assay's growth estimate, model loading and figure notes.

* With ``plaque_estimate_growth`` on, every per-image and per-plaque row of
  ``plaques_analysis.db`` carries the growth estimate for its image, and
  where it came from.
* The plaque model loads on the CPU when the accelerator cannot be asked;
  a Cellpose 3 checkpoint is explained, and any other load error is left
  as it was.
* A training base named by a zoo key with no local file is fetched into
  ``~/.spacr/models``.
* Figure mode's summary names the duplicates, the near-duplicates and the
  label/legend conflicts it found.
"""
from __future__ import annotations

import os
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

import spacr.submodules as SUB
from tests.test_cov_submodules_recruitment_plaques import (  # noqa: F401
    _write_label_tif, plaque_env)


def test_the_growth_estimate_reaches_every_row(tmp_path, plaque_env):
    src = tmp_path / "plaques"
    masks = src / "masks"
    masks.mkdir(parents=True)
    _write_label_tif(masks / "a.tif")
    SUB.analyze_plaques({"src": str(src), "masks": False,
                         "plaque_estimate_growth": True})
    with sqlite3.connect(masks / "plaques_analysis.db") as con:
        per_image = pd.read_sql("SELECT * FROM per_image", con)
        per_plaque = pd.read_sql("SELECT * FROM per_plaque", con)
    for table in (per_image, per_plaque):
        assert "estimated_formation_hours" in table.columns
        assert "growth_estimate_provenance" in table.columns
        assert table["estimation_source"].notna().all()
    assert len(per_plaque) == 2


def test_the_plaque_model_loads_on_the_cpu_when_the_accelerator_is_silent(
        monkeypatch):
    import spacr.accelerator as accelerator

    def silent():
        raise RuntimeError("no accelerator answer")

    built = []
    monkeypatch.setattr(accelerator, "cellpose_kwargs", silent)
    monkeypatch.setattr(SUB.cp_models, "CellposeModel",
                        lambda **kw: built.append(kw) or "model")
    assert SUB._plaque_cellpose_model("/m/plaque.CP_model") == "model"
    assert built == [{"pretrained_model": "/m/plaque.CP_model",
                      "device": None, "gpu": False}]


class _Explained(Exception):
    pass


@pytest.mark.parametrize("explains", [True, False])
def test_a_load_error_is_explained_only_when_there_is_an_explanation(
        monkeypatch, explains):
    failure = ValueError("state dict mismatch")

    def refuse(**kw):
        raise failure

    monkeypatch.setattr(SUB.cp_models, "CellposeModel", refuse)
    monkeypatch.setattr(SUB, "explain_cellpose3", lambda exc, model:
                        _Explained("a Cellpose 3 model") if explains else exc)
    expected = _Explained if explains else ValueError
    with pytest.raises(expected) as excinfo:
        SUB._plaque_cellpose_model("/m/old.pth")
    if explains:
        assert excinfo.value.__cause__ is failure
    else:
        assert excinfo.value is failure


def test_a_zoo_base_without_a_local_file_is_fetched_into_the_model_folder(
        tmp_path, monkeypatch):
    import spacr.model_zoo as model_zoo

    monkeypatch.setenv("HOME", str(tmp_path))
    entry = SimpleNamespace(key="cpsam_plaque_r5", path="",
                            uri="https://x/plaque.CP_model")
    fetched = []
    monkeypatch.setattr(model_zoo, "catalogue", lambda remote=True: [entry])
    monkeypatch.setattr(model_zoo, "fetch", lambda e, dest: fetched.append(
        (e, dest)) or os.path.join(dest, "plaque.CP_model"))
    path = SUB._resolve_training_base("cpsam_plaque_r5")
    dest = str(tmp_path / ".spacr" / "models")
    assert fetched == [(entry, dest)]
    assert path == os.path.join(dest, "plaque.CP_model")
    assert os.path.isdir(dest)


def test_figure_mode_names_what_it_did_not_measure_cleanly(tmp_path,
                                                           monkeypatch,
                                                           capsys):
    import spacr.plaque_papers as plaque_papers

    monkeypatch.setattr(plaque_papers, "figure_folders",
                        lambda src: [str(tmp_path)])
    monkeypatch.setattr(plaque_papers, "measure_figure_folder",
                        lambda *a, **k: {
                            "figures": 3, "regions": 4, "plaques": 20,
                            "database": "figures.db", "duplicates": 1,
                            "possible_duplicates": 2, "conflicts": 1,
                            "with_ruler": 4})
    summary = SUB._analyze_plaque_figures({"src": str(tmp_path)},
                                          "/m/plaque.CP_model")
    out = capsys.readouterr().out
    assert summary["duplicates"] == 1
    assert "1 figure(s) were already measured under another name" in out
    assert "2 figure(s) look like one already measured" in out
    assert "1 plaque image(s): the label and the legend disagree" in out
    assert "have no scale bar" not in out


def test_figure_mode_with_no_plaque_images_says_nothing_about_scale(
        tmp_path, monkeypatch, capsys):
    import spacr.plaque_papers as plaque_papers

    monkeypatch.setattr(plaque_papers, "figure_folders",
                        lambda src: [str(tmp_path)])
    monkeypatch.setattr(plaque_papers, "measure_figure_folder",
                        lambda *a, **k: {"figures": 1, "regions": 0,
                                         "plaques": 0, "database": "f.db"})
    SUB._analyze_plaque_figures({"src": str(tmp_path)}, "/m/plaque.CP_model")
    out = capsys.readouterr().out
    assert "Figure mode: 1 figure(s), 0 plaque image(s)" in out
    assert "scale bar" not in out and "measured under" not in out
