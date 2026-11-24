"""Development stage bundles load and convert to release checkpoints (spec G1)."""
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from spacr import ops_cell_cycle as cc

OWN = ["interphase", "g2_prophase", "prometaphase", "metaphase", "ana_telo_cyto", "artefact"]
FEATURES = ["cell_dapi_mean_intensity", "cell_dapi_std_intensity", "nucleus_area"]


def _bundle(tmp_path, normalisation="robust z per well against ALL cells of the well"):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 600)
    x = rng.standard_normal((600, 3)) + y[:, None]
    model = HistGradientBoostingClassifier(max_iter=20).fit(x, y)
    path = tmp_path / "stage.joblib"
    joblib.dump({"format": "spacr-cellcycle-stage-model-1 (development)", "classes": OWN,
                 "model": model, "features": FEATURES, "normalisation": normalisation}, path)
    return str(path), model


def _cells(n=300):
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.standard_normal((n, 3)) * 5 + 20, columns=FEATURES)
    frame["well"] = np.where(np.arange(n) % 2, "A1", "A2")
    return frame


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cc.load_cell_cycle_checkpoint(path)


def test_dev_bundle_loads_with_class_map_and_zero_absent_class(tmp_path):
    path, model = _bundle(tmp_path)
    loaded = _load(path)
    stages = cc.predict_stages(loaded, _cells())
    assert list(stages.columns[:-1]) == [f"p_{c}" for c in cc.CLASSES]
    assert (stages["p_telophase_daughters"] == 0).all()
    assert np.allclose(stages.filter(like="p_").sum(axis=1), 1)
    card = loaded["card"]["imported_from"]
    assert card["absent_classes"] == ["telophase_daughters"]
    assert card["class_map"]["artefact"] == "artefact"
    own = model.predict_proba(cc._matrix(_cells(), FEATURES))
    assert np.allclose(stages["p_artefact"], own[:, 5])
    assert np.allclose(stages["p_ana_telo_cyto"], own[:, 4])


def test_convert_round_trip_keeps_probabilities(tmp_path):
    path, _ = _bundle(tmp_path)
    direct = _load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folder = cc.import_stage_bundle(path, str(tmp_path / "ckpt"))
    again = _load(folder)
    cells = _cells()
    assert np.allclose(cc.predict_stages(direct, cells).filter(like="p_"),
                       cc.predict_stages(again, cells).filter(like="p_"))
    assert again["card"]["imported_from"]["sha256"] == direct["sha256"]
    assert again["card"]["dna"] == "dapi"


def test_other_normalisation_is_refused(tmp_path):
    path, _ = _bundle(tmp_path, "min-max per plate")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="robust z"):
            cc.stage_bundle(path)


def test_apply_normalisation_equals_the_well_robust_z():
    from spacr.ops_mitotic import _well_robust_z

    cells = _cells()
    expected = np.nan_to_num(_well_robust_z(cells, FEATURES, "well"), nan=0.0)
    assert np.allclose(cc._matrix(cells, FEATURES), expected, atol=1e-5)
