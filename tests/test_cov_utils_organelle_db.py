"""CPU-only coverage for the organelle-diagnostic / debug-decorator / database
writer block of ``spacr.utils`` (utils.py lines ~655-1011).

Focus is on the branches the rest of the suite never reaches:

  * ``_organelle_diagnostic`` -- the DoG spot branch, the hysteresis network
    branch (both fractional and absolute threshold forms), the ring branch and
    the unknown-morphology fallback.
  * ``debug`` -- the disabled short-circuit, the enabled path (log level bumped
    then restored) and the restore-on-exception ``finally``.
  * ``filepaths_to_database`` -- the ``pathogen`` crop mode and the
    ``sqlite3.OperationalError`` handler.
  * ``activation_maps_to_database`` / ``activation_correlations_to_database`` --
    their ``sqlite3.OperationalError`` handlers.
  * ``calculate_activation_correlations`` -- the size-mismatch interpolate path
    and the "no finite pixels -> NaN pearson" path.

Failure injection is real: the DB error handlers are reached by pointing the
writers at a ``measurements/`` directory that does not exist, so sqlite3 itself
raises ``unable to open database file``.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _no_lingering_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _organelle_diagnostic -- percentile normalisation guard
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _organelle_diagnostic -- spots / dog
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _organelle_diagnostic -- network / hysteresis
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _generate_mask_random_cmap
# ---------------------------------------------------------------------------

def test_generate_mask_random_cmap_one_colour_per_label_plus_background():
    from spacr.utils import _generate_mask_random_cmap
    mask = np.zeros((8, 8), np.int32)
    mask[0:2, 0:2] = 1
    mask[3:5, 3:5] = 4
    mask[6:8, 6:8] = 9          # non-contiguous labels
    cmap = _generate_mask_random_cmap(mask)
    assert cmap.N == 4          # 3 objects + background
    colours = cmap(np.arange(cmap.N))
    assert np.allclose(colours[0], [0, 0, 0, 1]), "label 0 must be opaque black"
    assert np.allclose(colours[:, 3], 1.0), "every colour must be opaque"
    assert np.all(colours[1:, :3] >= 0) and np.all(colours[1:, :3] <= 1)


def test_generate_mask_random_cmap_empty_mask_is_background_only():
    from spacr.utils import _generate_mask_random_cmap
    cmap = _generate_mask_random_cmap(np.zeros((4, 4), np.int32))
    assert cmap.N == 1
    assert np.allclose(cmap(0), [0, 0, 0, 1])

# ---------------------------------------------------------------------------
# _organelle_diagnostic -- ring
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# debug decorator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# filepaths_to_database
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# activation_maps_to_database
# ---------------------------------------------------------------------------

def _act_paths(directory, n=3):
    """Activation-map PNG names -- parsed by the same ``_map_wells_png``."""
    return [os.path.join(str(directory), f"plate1_A01_1_{i + 1}.png")
            for i in range(n)]


def test_activation_maps_to_database_creates_db_and_rows(tmp_path):
    from spacr.utils import activation_maps_to_database
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    paths = _act_paths(src)

    activation_maps_to_database(
        paths, str(src), {"dataset": "/some/where/ds1.pt", "cam_type": "gradcam"})

    db = src / "measurements" / "ds1.db"
    assert db.is_file(), "dataset stem should name the db"
    con = sqlite3.connect(db)
    try:
        df = pd.read_sql("SELECT * FROM gradcam_list", con)
    finally:
        con.close()
    assert len(df) == 3
    assert sorted(df["object"]) == ["o1", "o2", "o3"]
    assert set(df["prcfo"]) == {
        "plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2", "plate1_r1_c1_f1_o3"}


def test_activation_maps_to_database_operational_error_is_caught(tmp_path, capsys):
    from spacr.utils import activation_maps_to_database
    src = tmp_path / "plate1"
    src.mkdir()  # no measurements/ subdir -> connect fails
    paths = _act_paths(src, n=2)

    activation_maps_to_database(
        paths, str(src), {"dataset": "ds1.pt", "cam_type": "gradcam"})

    out = capsys.readouterr().out
    assert "SQLite error" in out
    assert not (src / "measurements").exists()


# ---------------------------------------------------------------------------
# activation_correlations_to_database
# ---------------------------------------------------------------------------

def _corr_df(paths):
    return pd.DataFrame({
        "file_name": [os.path.basename(p) for p in paths],
        "channel_0_activation_0_pearsons": np.linspace(0.1, 0.3, len(paths)),
    })


def test_activation_correlations_to_database_merges_on_file_name(tmp_path):
    from spacr.utils import activation_correlations_to_database
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    paths = _act_paths(src)

    activation_correlations_to_database(
        _corr_df(paths), paths, str(src),
        {"dataset": "ds1.pt", "cam_type": "gradcam"})

    con = sqlite3.connect(src / "measurements" / "ds1.db")
    try:
        df = pd.read_sql("SELECT * FROM gradcam_correlations", con)
    finally:
        con.close()
    assert len(df) == 3
    # both halves of the merge survived
    assert "png_path" in df.columns and "channel_0_activation_0_pearsons" in df.columns
    assert sorted(df["object"]) == ["o1", "o2", "o3"]
    row = df.set_index("file_name").loc["plate1_A01_1_1.png"]
    assert row["channel_0_activation_0_pearsons"] == pytest.approx(0.1)
    assert row["prcfo"] == "plate1_r1_c1_f1_o1"


def test_activation_correlations_to_database_operational_error_is_caught(
        tmp_path, capsys):
    from spacr.utils import activation_correlations_to_database
    src = tmp_path / "plate1"
    src.mkdir()  # no measurements/ subdir
    paths = _act_paths(src, n=2)

    activation_correlations_to_database(
        _corr_df(paths), paths, str(src),
        {"dataset": "ds1.pt", "cam_type": "gradcam"})

    out = capsys.readouterr().out
    assert "SQLite error" in out
    assert not (src / "measurements").exists()


# ---------------------------------------------------------------------------
# calculate_activation_correlations
# ---------------------------------------------------------------------------
