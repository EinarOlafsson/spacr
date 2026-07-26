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

import logging
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

def _blob_img(size=48):
    """Two bright Gaussian spots on a black background, float32."""
    img = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[:size, :size]
    for cy, cx in ((14, 14), (32, 30)):
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 18.0) * 900
    return img


def _smoothed(img, sigma=1):
    from skimage.filters import gaussian
    return gaussian(img, sigma=sigma)


def _crop_png_names(directory, n=3, crop_mode="cell"):
    """Filenames shaped the way ``_generate_names`` emits cropped-object PNGs.

    ``_generate_names`` builds ``<plate>_<well>_<field>_<cell_id>.png`` for
    cell/cytoplasm crops and ``<plate>_<well>_<field>_<cell_id>_<child_id>.png``
    for nucleus/pathogen crops; ``_map_wells_png`` then reads ``parts[2]`` as
    the field and ``parts[-1]`` as the object id, so both forms must be bare
    integers.

    The files never need to exist on disk -- the DB writers only ever call
    ``os.path.basename`` on the paths.
    """
    if crop_mode in ("nucleus", "pathogen"):
        names = [f"plate1_A01_1_9_{i + 1}.png" for i in range(n)]
    else:
        names = [f"plate1_A01_1_{i + 1}.png" for i in range(n)]
    return [os.path.join(str(directory), nm) for nm in names]


# ---------------------------------------------------------------------------
# _organelle_diagnostic -- percentile normalisation guard
# ---------------------------------------------------------------------------

def test_organelle_diagnostic_constant_image_skips_normalisation():
    """A flat image has pmax == pmin, so the rescale is skipped entirely."""
    from spacr.utils import _organelle_diagnostic
    img = np.full((16, 16), 7.5, np.float32)
    diag, title = _organelle_diagnostic(img, "unknown-morphology", "otsu", {})
    assert title == "Normalised image"
    # No rescale happened: the raw value survives rather than becoming 0 or 1.
    assert np.allclose(diag, 7.5)
    assert diag.dtype == np.float64


def test_organelle_diagnostic_unknown_morphology_returns_normalised():
    """Fallback branch: percentile-normalised copy clipped to [0, 1]."""
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    diag, title = _organelle_diagnostic(img, "blobby", "otsu", {})
    assert title == "Normalised image"
    assert diag.shape == img.shape
    assert diag.min() == pytest.approx(0.0)
    assert diag.max() == pytest.approx(1.0)
    # exactly the documented 1st/99th-percentile rescale, clipped
    p1, p99 = np.percentile(img.astype(np.float64), (1, 99))
    expected = np.clip((img.astype(np.float64) - p1) / (p99 - p1), 0, 1)
    assert np.allclose(diag, expected)
    assert diag[14, 14] == pytest.approx(1.0)   # blob centre saturates
    assert diag[0, 0] < 1e-6                    # empty corner ~ background


# ---------------------------------------------------------------------------
# _organelle_diagnostic -- spots / dog
# ---------------------------------------------------------------------------

def test_organelle_diagnostic_spots_dog_paints_detected_blobs():
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    settings = {
        "organelle_dog_sigma_low": 1.0,
        "organelle_dog_sigma_high": 3.0,
        "organelle_log_threshold": 0.01,
    }
    diag, title = _organelle_diagnostic(img, "spots", "dog", settings)

    assert diag.shape == img.shape
    assert diag.dtype == np.float64
    assert title.startswith("DoG detections (") and title.endswith("blobs)")
    n_blobs = int(title.split("(")[1].split(" ")[0])
    assert n_blobs == 2, f"expected both synthetic spots to be found, got {title}"

    # Each blob centre was painted to exactly 1.0 ...
    assert diag[14, 14] == 1.0
    assert diag[32, 30] == 1.0
    # ... and the painting really changed pixels that were dark beforehand.
    assert diag[0, 0] < 1.0
    assert np.count_nonzero(diag == 1.0) > 2
    assert diag.max() == 1.0


def test_organelle_diagnostic_spots_dog_no_detections_leaves_image_untouched():
    """An empty image yields zero DoG blobs, so the paint loop never runs."""
    from spacr.utils import _organelle_diagnostic
    img = np.zeros((32, 32), np.float32)
    diag, title = _organelle_diagnostic(
        img, "spots", "dog",
        {"organelle_dog_sigma_low": 1.0, "organelle_dog_sigma_high": 3.0,
         "organelle_log_threshold": 0.5})
    assert title == "DoG detections (0 blobs)"
    assert np.all(diag == 0.0)


def test_organelle_diagnostic_spots_dog_uses_settings_sigmas(monkeypatch):
    """The sigma / threshold settings are actually forwarded to blob_dog."""
    import spacr.utils as U
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    captured = {}
    real = U.blob_dog

    def spy(image, **kwargs):
        captured.update(kwargs)
        return real(image, **kwargs)

    monkeypatch.setattr(U, "blob_dog", spy)
    _organelle_diagnostic(img, "spots", "dog", {
        "organelle_dog_sigma_low": 2.0,
        "organelle_dog_sigma_high": 6.0,
        "organelle_log_threshold": 0.02,
    })
    assert captured["min_sigma"] == 2.0
    assert captured["max_sigma"] == 6.0
    assert captured["threshold"] == 0.02


def test_organelle_diagnostic_spots_log_paints_detected_blobs():
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    diag, title = _organelle_diagnostic(img, "spots", "log", {
        "organelle_log_min_sigma": 1,
        "organelle_log_max_sigma": 5,
        "organelle_log_num_sigma": 5,
        "organelle_log_threshold": 0.05,
    })
    assert title.startswith("LoG detections (")
    assert int(title.split("(")[1].split(" ")[0]) == 2
    assert diag.shape == img.shape
    assert diag[14, 14] == 1.0 and diag[32, 30] == 1.0
    assert diag.max() == 1.0
    assert diag[0, 0] < 1.0


def test_organelle_diagnostic_spots_otsu_uses_white_tophat():
    from skimage.morphology import disk, white_tophat
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    diag, title = _organelle_diagnostic(
        img, "spots", "otsu", {"organelle_tophat_radius": 8})
    assert title == "Top-hat filtered (r=8)"
    assert np.allclose(diag, white_tophat(img, disk(8)))
    # white top-hat is non-negative and never exceeds the original
    assert np.all(diag >= 0)
    assert np.all(diag <= img + 1e-3)


# ---------------------------------------------------------------------------
# _organelle_diagnostic -- network / hysteresis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filter_name", ["frangi", "sato", "meijering"])
def test_organelle_diagnostic_network_ridge_filters(filter_name):
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    sigmas = [1, 2]
    diag, title = _organelle_diagnostic(img, "network", "ridge", {
        "organelle_ridge_sigmas": sigmas,
        "organelle_ridge_filter": filter_name,
    })
    assert title == f"{filter_name} ridge (sigmas={sigmas})"
    assert diag.shape == img.shape
    assert np.all(np.isfinite(diag))
    assert diag.max() > 0, "ridge filter produced an all-zero response"


def test_organelle_diagnostic_network_unknown_method_smooths():
    from skimage.filters import gaussian
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    diag, title = _organelle_diagnostic(img, "network", "otsu", {})
    assert title == "Gaussian smoothed (σ=1)"
    assert np.allclose(diag, gaussian(img, sigma=1))
    # smoothing lowers the peak but preserves total signal
    assert diag.max() < img.max()


@pytest.mark.parametrize("radius,sigma", [(6, 3.0), (1, 1.0), (2, 1.0)])
def test_organelle_diagnostic_irregular_sigma_floor(radius, sigma):
    """sigma is max(radius/2, 1) -- small radii are floored at 1."""
    from skimage.filters import gaussian
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    diag, title = _organelle_diagnostic(
        img, "irregular", "otsu", {"organelle_morph_radius": radius})
    assert title == f"Gaussian smoothed (σ={sigma:.1f})"
    assert np.allclose(diag, gaussian(img, sigma=sigma))


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

def test_organelle_diagnostic_network_hysteresis_fractional_thresholds():
    """low/high < 1.0 are interpreted as percentiles of the smoothed image."""
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    diag, title = _organelle_diagnostic(
        img, "network", "hysteresis",
        {"organelle_hysteresis_low": 0.9, "organelle_hysteresis_high": 0.99})

    assert title == "Hysteresis (low=0.9, high=0.99)"
    assert diag.dtype == np.float64
    assert set(np.unique(diag)).issubset({0.0, 1.0})

    smooth = _smoothed(img)
    low_abs = np.percentile(smooth, 90)
    high_abs = np.percentile(smooth, 99)
    selected = diag > 0
    assert selected.sum() > 0
    # every selected pixel clears the low (percentile) threshold ...
    assert np.all(smooth[selected] >= low_abs)
    # ... every seed pixel above the high threshold is selected ...
    assert np.all(diag[smooth >= high_abs] == 1.0)
    # ... and the result is strictly smaller than "everything".
    assert selected.sum() < diag.size


def test_organelle_diagnostic_network_hysteresis_absolute_thresholds():
    """low/high >= 1.0 are used as raw intensities, not percentiles."""
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    diag, title = _organelle_diagnostic(
        img, "network", "hysteresis",
        {"organelle_hysteresis_low": 50.0, "organelle_hysteresis_high": 300.0})

    assert title == "Hysteresis (low=50.0, high=300.0)"
    assert set(np.unique(diag)).issubset({0.0, 1.0})

    smooth = _smoothed(img)
    selected = diag > 0
    assert selected.sum() > 0
    assert np.all(smooth[selected] >= 50.0)
    assert np.all(diag[smooth >= 300.0] == 1.0)
    # An absolute threshold of 50 on a mostly-black image selects far less than
    # the 90th-percentile form would.
    assert selected.sum() < diag.size


def test_organelle_diagnostic_network_hysteresis_high_threshold_empties_mask():
    """A high threshold above every pixel leaves no seeds -> all-zero mask."""
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    diag, _ = _organelle_diagnostic(
        img, "network", "hysteresis",
        {"organelle_hysteresis_low": 10.0, "organelle_hysteresis_high": 1e9})
    assert diag.sum() == 0.0


# ---------------------------------------------------------------------------
# _organelle_diagnostic -- ring
# ---------------------------------------------------------------------------

def test_organelle_diagnostic_ring_default_sigmas():
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    diag, title = _organelle_diagnostic(img, "ring", "otsu", {})
    assert title == "DoG ring enhancement (σ=1.0/3.0)"
    assert diag.shape == img.shape
    # np.abs() was applied
    assert np.all(diag >= 0)
    assert diag.max() > 0
    # response is concentrated on the blobs, not the empty corner
    assert diag[14, 14] > diag[0, 0]


def test_organelle_diagnostic_ring_custom_sigmas_change_response():
    from spacr.utils import _organelle_diagnostic
    img = _blob_img()
    settings = {"organelle_ring_sigma_inner": 0.5,
                "organelle_ring_sigma_outer": 6.0}
    diag, title = _organelle_diagnostic(img, "ring", "otsu", settings)
    assert title == "DoG ring enhancement (σ=0.5/6.0)"
    default, _ = _organelle_diagnostic(img, "ring", "otsu", {})
    assert not np.allclose(diag, default), "custom sigmas were ignored"
    assert np.all(diag >= 0)


# ---------------------------------------------------------------------------
# debug decorator
# ---------------------------------------------------------------------------

class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _attach(logger):
    """Attach a record collector and stop propagation to the root logger."""
    handler = _Collector()
    logger.addHandler(handler)
    logger.propagate = False
    return handler


def _detach(logger, handler, level=logging.NOTSET):
    logger.removeHandler(handler)
    logger.propagate = True
    logger.setLevel(level)


def test_debug_disabled_is_a_pure_passthrough():
    """enabled=False must not touch the logger at all."""
    from spacr.utils import debug

    calls = []

    @debug(enabled=False, logger_name="spacr.test.debug.disabled")
    def add(a, b=1):
        calls.append((a, b))
        return a + b

    log = logging.getLogger("spacr.test.debug.disabled")
    log.setLevel(logging.WARNING)
    handler = _attach(log)
    try:
        assert add(2, b=3) == 5
        # the logger level was never touched
        assert log.level == logging.WARNING
    finally:
        _detach(log, handler)

    assert calls == [(2, 3)]
    assert handler.records == [], "disabled decorator must not log"


def test_debug_enabled_logs_entry_exit_and_restores_level():
    from spacr.utils import debug

    name = "spacr.test.debug.enabled"
    log = logging.getLogger(name)
    log.setLevel(logging.ERROR)
    handler = _attach(log)

    @debug(enabled=True, logger_name=name)
    def multiply(a, b):
        # level is DEBUG *during* the call
        assert log.level == logging.DEBUG
        return a * b

    try:
        assert multiply(6, 7) == 42
        # ... and restored to the caller's level afterwards
        assert log.level == logging.ERROR
    finally:
        _detach(log, handler)

    msgs = [r.getMessage() for r in handler.records]
    assert ">>> Entering multiply" in msgs
    assert "<<< Exiting multiply" in msgs
    assert msgs.index(">>> Entering multiply") < msgs.index("<<< Exiting multiply")
    assert all(r.levelno == logging.DEBUG for r in handler.records)


def test_debug_restores_level_when_wrapped_function_raises():
    from spacr.utils import debug

    name = "spacr.test.debug.raises"
    log = logging.getLogger(name)
    log.setLevel(logging.CRITICAL)
    handler = _attach(log)

    @debug(enabled=True, logger_name=name)
    def boom():
        raise ValueError("kaboom")

    try:
        with pytest.raises(ValueError, match="kaboom"):
            boom()
        # finally: restored the pre-call level
        assert log.level == logging.CRITICAL
        msgs = [r.getMessage() for r in handler.records]
        assert ">>> Entering boom" in msgs
        assert "<<< Exiting boom" not in msgs
    finally:
        _detach(log, handler)


def test_debug_defaults_logger_to_wrapped_functions_module():
    """logger_name=None falls back to func.__module__."""
    from spacr.utils import debug

    @debug()
    def identity(x):
        return x

    assert identity.__name__ == "identity"      # functools.wraps applied
    log = logging.getLogger(identity.__module__)
    assert log.name == __name__
    old = log.level
    handler = _attach(log)
    try:
        assert identity("hello") == "hello"
        assert log.level == old
    finally:
        _detach(log, handler, level=old)
    msgs = [r.getMessage() for r in handler.records]
    assert ">>> Entering identity" in msgs
    assert "<<< Exiting identity" in msgs


# ---------------------------------------------------------------------------
# filepaths_to_database
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("crop_mode,id_col", [
    ("cell", "cell_id"),
    ("nucleus", "nucleus_id"),
    ("pathogen", "pathogen_id"),
    ("cytoplasm", "cytoplasm_id"),
])
def test_filepaths_to_database_object_id_column_per_crop_mode(
        tmp_path, crop_mode, id_col):
    """Each crop mode gets its own <object>_id column filled with 'o<N>'."""
    from spacr.utils import filepaths_to_database
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    paths = _crop_png_names(src, n=3, crop_mode=crop_mode)

    filepaths_to_database(paths, {"timelapse": False}, str(src), crop_mode)

    con = sqlite3.connect(src / "measurements" / "measurements.db")
    try:
        df = pd.read_sql("SELECT * FROM png_list", con)
    finally:
        con.close()

    assert len(df) == 3
    assert id_col in df.columns
    assert sorted(df[id_col].tolist()) == ["o1", "o2", "o3"]
    assert set(df["plateID"]) == {"plate1"}
    assert set(df["rowID"]) == {"r1"}
    assert set(df["columnID"]) == {"c1"}
    assert set(df["fieldID"]) == {"f1"}
    assert sorted(df["prcfo"]) == [
        "plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2", "plate1_r1_c1_f1_o3"]
    # only the crop mode's own id column is present
    others = {"cell_id", "nucleus_id", "pathogen_id", "cytoplasm_id"} - {id_col}
    assert not (others & set(df.columns))
    assert "timeID" not in df.columns and "time_id" not in df.columns


def test_filepaths_to_database_timelapse_adds_time_id_column(tmp_path):
    """timelapse=True inserts an extra timeID column parsed from parts[3].

    Spelled ``timeID``, matching every object table. It used to be written as
    ``time_id`` while _merge_and_save_to_database wrote ``timeID``, so one
    database carried two names for one concept and no join between png_list
    and cell on time could match.
    """
    from spacr.utils import filepaths_to_database
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    # plate _ well _ field _ time _ object
    paths = [os.path.join(str(src), f"plate1_A01_1_{t}_5.png") for t in (1, 2, 3)]

    filepaths_to_database(paths, {"timelapse": True}, str(src), "cell")

    con = sqlite3.connect(src / "measurements" / "measurements.db")
    try:
        df = pd.read_sql("SELECT * FROM png_list", con)
    finally:
        con.close()

    assert len(df) == 3
    assert "time_id" not in df.columns
    assert sorted(df["timeID"]) == ["t1", "t2", "t3"]
    assert set(df["fieldID"]) == {"f1"}
    assert set(df["cell_id"]) == {"o5"}
    # prcfo carries the time id between field and object
    assert sorted(df["prcfo"]) == [
        "plate1_r1_c1_f1_t1_o5", "plate1_r1_c1_f1_t2_o5", "plate1_r1_c1_f1_t3_o5"]


def test_filepaths_to_database_operational_error_is_caught(tmp_path, capsys):
    """No measurements/ dir -> sqlite3 cannot open the DB; error is swallowed."""
    from spacr.utils import filepaths_to_database
    src = tmp_path / "plate1"
    src.mkdir()
    # deliberately do NOT create src/measurements
    paths = _crop_png_names(src, n=2)

    filepaths_to_database(paths, {"timelapse": False}, str(src), "cell")

    out = capsys.readouterr().out
    assert "SQLite error" in out
    assert not (src / "measurements").exists()
    # sanity: sqlite really does refuse this path
    with pytest.raises(sqlite3.OperationalError):
        sqlite3.connect(f"{src}/measurements/measurements.db", timeout=5)


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

def test_calculate_activation_correlations_interpolates_mismatched_maps():
    """Smaller activation maps are bilinearly resized up to the input size.

    If the resize did not happen, pearsonr would receive a 64-element and a
    256-element vector and raise, so finite pearson values prove the branch ran.
    """
    torch = pytest.importorskip("torch")
    from spacr.utils import calculate_activation_correlations
    rng = np.random.default_rng(7)
    inputs = torch.tensor(rng.random((2, 3, 16, 16)), dtype=torch.float32)
    maps = torch.tensor(rng.random((2, 1, 8, 8)), dtype=torch.float32)

    out = calculate_activation_correlations(inputs, maps, ["a.png", "b.png"])

    assert list(out["file_name"]) == ["a.png", "b.png"]
    # 3 input channels x 1 activation channel x (1 pearson + 3 thresholds x 2)
    assert len(out.columns) == 1 + 3 * 1 * (1 + 2 * 3)
    pearson_cols = [c for c in out.columns if c.endswith("_pearsons")]
    assert len(pearson_cols) == 3
    vals = out[pearson_cols].to_numpy()
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= -1.0) and np.all(vals <= 1.0)
    # the caller's tensor was not mutated in place
    assert maps.shape == (2, 1, 8, 8)


def test_calculate_activation_correlations_all_nan_input_gives_nan_pearson():
    """No finite pixels -> the NaN fallbacks fire instead of pearsonr."""
    torch = pytest.importorskip("torch")
    from spacr.utils import calculate_activation_correlations
    inputs = torch.full((1, 2, 8, 8), float("nan"))
    maps = torch.ones((1, 2, 8, 8))

    out = calculate_activation_correlations(inputs, maps, ["nan.png"])

    assert len(out) == 1
    assert out.loc[0, "file_name"] == "nan.png"
    stat_cols = [c for c in out.columns if c != "file_name"]
    assert stat_cols, "no statistic columns produced"
    assert out[stat_cols].isna().all().all(), \
        "empty channels must yield NaN pearson and NaN Manders"


def test_calculate_activation_correlations_3d_maps_get_a_channel_axis():
    """(B, H, W) activation maps are unsqueezed to a single channel."""
    torch = pytest.importorskip("torch")
    from spacr.utils import calculate_activation_correlations
    rng = np.random.default_rng(3)
    inputs = torch.tensor(rng.random((2, 2, 10, 10)), dtype=torch.float32)
    maps = torch.tensor(rng.random((2, 10, 10)), dtype=torch.float32)

    out = calculate_activation_correlations(inputs, maps, ["a.png", "b.png"])

    # 2 input channels x exactly 1 activation channel
    assert [c for c in out.columns if c.endswith("_pearsons")] == [
        "channel_0_activation_0_pearsons", "channel_1_activation_0_pearsons"]
    assert len(out.columns) == 1 + 2 * 1 * (1 + 2 * 3)
    assert np.all(np.isfinite(out.drop(columns=["file_name"]).to_numpy()))


def test_calculate_activation_correlations_disjoint_thresholds_give_nan_manders():
    """Anti-correlated channels -> the two threshold masks never overlap.

    ``input`` is [1, 2] and ``activation`` is [2, 1]; at the 75th percentile the
    thresholds are 1.75 for both, so ``mask`` is all-False and the Manders
    coefficients fall back to NaN while pearson is still computable (-1).
    """
    torch = pytest.importorskip("torch")
    from spacr.utils import calculate_activation_correlations
    inputs = torch.tensor([[[[1.0, 2.0]]]])
    maps = torch.tensor([[[[2.0, 1.0]]]])

    out = calculate_activation_correlations(
        inputs, maps, ["disjoint.png"], manders_thresholds=[75])

    assert out.loc[0, "channel_0_activation_0_pearsons"] == pytest.approx(-1.0)
    assert np.isnan(out.loc[0, "channel_0_activation_0_75_M1"])
    assert np.isnan(out.loc[0, "channel_0_activation_0_75_M2"])


def test_calculate_activation_correlations_partial_nan_uses_joint_finite_mask():
    """A single NaN pixel in one channel must not blow up the whole batch.

    The finite-pixel filter has to be applied jointly so the two vectors stay
    index-aligned; correlating pixel i of the input against pixel i+1 of the
    activation map would be wrong even if the lengths happened to match.
    """
    torch = pytest.importorskip("torch")
    from scipy.stats import pearsonr
    from spacr.utils import calculate_activation_correlations

    inputs = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).clone()
    inputs[0, 0, 0, 0] = float("nan")
    rng = np.random.default_rng(11)
    maps = torch.tensor(rng.random((1, 1, 4, 4)), dtype=torch.float32)

    out = calculate_activation_correlations(inputs, maps, ["partial.png"])

    valid_in = inputs.flatten().numpy()[1:]
    valid_act = maps.flatten().numpy()[1:]
    expected, _ = pearsonr(valid_in, valid_act)
    assert out.loc[0, "channel_0_activation_0_pearsons"] == pytest.approx(
        expected, abs=1e-6)
    assert np.isfinite(out.loc[0, "channel_0_activation_0_50_M1"])


def test_calculate_activation_correlations_infinite_activation_is_dropped():
    """Non-finite activation pixels are filtered before the correlation."""
    torch = pytest.importorskip("torch")
    from spacr.utils import calculate_activation_correlations
    maps = torch.full((1, 1, 6, 6), float("inf"))
    inputs = torch.arange(36, dtype=torch.float32).reshape(1, 1, 6, 6)

    out = calculate_activation_correlations(inputs, maps, ["inf.png"])

    assert np.isnan(out.loc[0, "channel_0_activation_0_pearsons"])
    assert np.isnan(out.loc[0, "channel_0_activation_0_15_M1"])
    assert np.isnan(out.loc[0, "channel_0_activation_0_75_M2"])
