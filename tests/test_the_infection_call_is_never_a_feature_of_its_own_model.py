"""The XGBoost infection QC must not learn from the answer.

The classifier is trained to predict which cells are infected, on features
taken from the tracked object's own morphology and pathogen-channel
intensity. The infection call itself is aggregated separately and merged back
onto the per-cell table, so it sits in that table beside the features -- and
the feature selector reads its candidates off that same table.

Which matters because the call is not always spelled ``infected``. A
user-created tracking table can name it anything with "infect" in it, and a
name in an object namespace -- ``cell_infection_index`` -- is a name the
shared feature schema recognises as a measurement. Selected as a feature it
would be a perfect predictor of itself: gain concentrated on one column, an
apparently flawless model, and a relabelling that only ever repeats the
labels it was given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

import spacr.timelapse  # noqa: E402,F401


PATHOGEN_CHAN = 1
CALL_COLUMN = "cell_infection_index"


@pytest.fixture(autouse=True)
def _close_figures():
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


def _all_df(n_per_class=14, wells=("A01", "A02"), n_frames=2, seed=5):
    """A tracking table whose infection call is a ``cell_*`` measurement."""
    rng = np.random.default_rng(seed)
    rows = []
    cid = 0
    for well in wells:
        for infected in (True, False):
            for _ in range(n_per_class):
                cid += 1
                intensity = float(rng.normal(1000.0 if infected else 300.0, 90.0))
                area = float(rng.uniform(200.0, 900.0)) + (300.0 if infected else 0.0)
                solidity = float(rng.uniform(0.70, 0.99))
                for frame in range(n_frames):
                    rows.append({
                        "plateID": "plate1",
                        "wellID": well,
                        "fieldID": "1",
                        "cellID": cid,
                        "frame": frame,
                        CALL_COLUMN: 1.0 if infected else 0.0,
                        f"cell_p95_intensity_ch{PATHOGEN_CHAN}": intensity,
                        f"cell_mean_intensity_ch{PATHOGEN_CHAN}":
                            intensity * 0.6 + float(rng.normal(0, 2.0)),
                        "cell_mean_intensity_ch0": float(rng.uniform(100.0, 200.0)),
                        "cell_area": area + float(rng.normal(0, 5.0)),
                        "cell_perimeter": 0.4 * area + float(rng.normal(0, 3.0)),
                        "cell_solidity": solidity + float(rng.normal(0, 0.005)),
                    })
    return pd.DataFrame(rows)


def _settings(**over):
    base = {
        "tracked_object": "cell",
        "infection_xgb_n_estimators": 12,
        "infection_xgb_max_depth": 2,
        "infection_xgb_n_jobs": 1,
        "infection_intensity_mode": "relabel",
    }
    base.update(over)
    return base


def _run(all_df, settings, motility_dir):
    from spacr.timelapse import _infection_qc_xgboost
    return _infection_qc_xgboost(
        all_df=all_df,
        settings=settings,
        infection_col="infected",
        pathogen_chan=PATHOGEN_CHAN,
        motility_dir=str(motility_dir),
    )


def test_a_call_column_named_for_the_object_is_recovered(tmp_path, capsys):
    """The named column is missing, so the one that mentions infection is used."""
    settings = _settings()

    out, col = _run(_all_df(), settings, tmp_path)

    printed = capsys.readouterr().out
    assert f"using {CALL_COLUMN!r} instead" in printed
    assert col == "adjusted_infected"
    assert "adjusted_infected" in out.columns


def test_the_recovered_call_is_kept_out_of_the_feature_set(tmp_path):
    """The answer may not be one of the columns the answer is predicted from."""
    settings = _settings()

    _run(_all_df(), settings, tmp_path)

    payload = settings["infection_xgb_importance"]
    assert payload is not None, "the model never trained"
    assert payload["feature_names"], "no features were used at all"
    assert CALL_COLUMN not in payload["feature_names"]


def test_the_features_that_are_kept_are_the_objects_own_measurements(tmp_path):
    """The control: excluding the call must not exclude everything else."""
    settings = _settings()

    _run(_all_df(), settings, tmp_path)

    names = settings["infection_xgb_importance"]["feature_names"]
    assert all(name.startswith("cell_") for name in names)
    assert any("intensity_ch1" in name for name in names)
    # A non-pathogen channel is not evidence of infection by that pathogen.
    assert not any("intensity_ch0" in name for name in names)


def test_the_model_still_separates_the_two_classes(tmp_path):
    """A guard that cost the model its signal would be the worse bug.

    The ambiguous band is off here: with the call excluded the classifier has
    only morphology and one intensity to go on, and on a table this small it
    puts most cells in the middle -- which is the band doing its job, not the
    separation failing.
    """
    settings = _settings(infection_xgb_drop_ambiguous=False)

    out, _col = _run(_all_df(), settings, tmp_path)

    assert "infection_prob" in out.columns
    infected = out.loc[out[CALL_COLUMN] == 1.0, "infection_prob"].mean()
    uninfected = out.loc[out[CALL_COLUMN] == 0.0, "infection_prob"].mean()
    assert infected > uninfected
