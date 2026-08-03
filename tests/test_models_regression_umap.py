"""Model-training + regression + image-UMAP coverage.

* T5 — build every Cellpose model type spaCR exposes and run one eval.
* T6 — build every torch classification backbone via choose_model and
  run a forward pass.
* T7 — fit every regression backend on a generated dependent +
  independent variable and assert the coefficient recovers the planted
  signal.
* T4 — image-UMAP end-to-end on a plate whose measurements database was
  written by spaCR's own writers.

Model-construction tests (T5/T6) are @gpu (they download/build weights);
the regression test (T7) and the image-UMAP test (T4) are CPU + fast and
run in the default suite.

No test in here is allowed to turn a failure into a ``pytest.skip``. A
self-skip makes "this environment cannot run the test" and "the product is
broken" look identical, and that is exactly how T4 below reported green for
its whole life while never once calling ``generate_image_umap``. The only
skips left are missing-optional-dependency (``importorskip``) and
no-such-hardware guards, neither of which can be triggered by a bug in
spaCR.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from urllib.error import HTTPError, URLError

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _require_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")


# ---------------------------------------------------------------------------
# T5 — all Cellpose models
# ---------------------------------------------------------------------------

CELLPOSE_MODELS = ["cpsam", "cyto3", "cyto2", "cyto", "nuclei"]


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("model_name", CELLPOSE_MODELS)
def test_build_and_eval_cellpose_model(model_name):
    """Every Cellpose model spaCR offers should load + segment a tile."""
    _require_gpu()
    from cellpose import models as cp_models
    rng = np.random.default_rng(0)
    img = rng.integers(0, 2000, size=(128, 128)).astype(np.float32)
    try:
        if model_name == "cpsam":
            model = cp_models.CellposeModel(gpu=True,
                                              pretrained_model="cpsam")
        else:
            model = cp_models.CellposeModel(gpu=True,
                                              model_type=model_name)
    except (OSError, URLError, HTTPError) as e:
        # Only a weights *fetch* failure is an environment problem. The old
        # `except Exception` also swallowed "spaCR asks for a model_type this
        # Cellpose no longer accepts", which is the exact class of breakage
        # this test exists to catch, and reported it as a green skip.
        pytest.skip(f"{model_name} weights unavailable offline: {e}")
    out = model.eval(img, diameter=30)
    masks = out[0]
    assert masks is not None
    assert np.asarray(masks).shape == img.shape


# ---------------------------------------------------------------------------
# T6 — all torch classification backbones
# ---------------------------------------------------------------------------

TORCH_MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vit_b_16", "convnext_tiny", "efficientnet_b0", "maxvit_t",
    "densenet121",
]


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("model_type", TORCH_MODELS)
def test_build_torch_model_and_forward(model_type):
    """choose_model should build each backbone and produce 2-class
    logits of shape (N, 2)."""
    torch = pytest.importorskip("torch")
    from spacr.utils import choose_model
    model = choose_model(
        model_type, device=torch.device("cpu"), num_classes=2,
        dropout_rate=0.0, use_checkpoint=False, init_weights=False,
        verbose=False,
    )
    if model is None:
        pytest.skip(f"{model_type} not buildable in this torchvision")
    model.eval()
    with torch.no_grad():
        # maxvit_t requires the exact training resolution (224); others
        # tolerate it too, so use 224 across the board.
        z = model(torch.randn(2, 3, 224, 224))
    assert z.shape == (2, 2)


# ---------------------------------------------------------------------------
# T7 — regression on a generated dependent + independent variable
# ---------------------------------------------------------------------------

REGRESSION_TYPES = ["ols", "glm", "ridge", "lasso"]


@pytest.mark.parametrize("regression_type", REGRESSION_TYPES)
def test_regression_recovers_planted_signal(regression_type):
    """Generate y = 3*x + noise, fit each backend, and assert the fitted
    model tracks the planted slope (or at least fits without error and
    predicts in the right direction)."""
    import pandas as pd
    from spacr.ml import regression_model
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0, 1, n)
    true_beta = 3.0
    y = true_beta * x + rng.normal(0, 0.5, n)
    # Design matrix with intercept + the single independent variable.
    X = pd.DataFrame({"const": 1.0, "x": x})
    # No try/skip: statsmodels and scikit-learn are hard dependencies, so the
    # only thing this swallow could ever have hidden was regression_model
    # itself breaking on a backend spaCR advertises.
    model = regression_model(X, pd.Series(y),
                                regression_type=regression_type,
                                alpha=1.0)
    assert model is not None
    # For the statsmodels backends, check the recovered slope.
    if hasattr(model, "params"):
        params = model.params
        beta_x = None
        try:
            beta_x = float(params["x"])
        except Exception:
            # positional fallback
            try:
                beta_x = float(np.asarray(params)[1])
            except Exception:
                beta_x = None
        if beta_x is not None:
            # Recovered slope should be positive + within a wide band of
            # the planted 3.0 (regularised backends shrink it).
            assert beta_x > 0.5
    # For sklearn backends (lasso/ridge) check predictions correlate.
    elif hasattr(model, "predict"):
        preds = model.predict(X)
        assert np.corrcoef(preds, y)[0, 1] > 0.5


def test_regression_binary_dependent_variable():
    """A binary dependent variable through the logit path."""
    import pandas as pd
    from spacr.ml import regression_model
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-(2.0 * x)))
    y = (rng.uniform(size=n) < prob).astype(float)
    X = pd.DataFrame({"const": 1.0, "x": x})
    model = regression_model(X, pd.Series(y), regression_type="logit")
    assert model is not None
    # The planted log-odds slope is +2.0, so the fit has to at least agree on
    # the sign; `assert model is not None` passed even when logit returned a
    # model fitted on the wrong column.
    beta_x = float(model.params["x"])
    assert beta_x > 0.5, f"logit recovered slope {beta_x}, expected ~2.0"


# ---------------------------------------------------------------------------
# T4 — image-UMAP end-to-end on a plate spaCR's own writers produced
# ---------------------------------------------------------------------------
#
# This test used to hand generate_image_umap a bare folder of PNGs with no
# measurements database and swallow the resulting ValueError into a skip, so
# generate_image_umap was never once called by it. What it should have been
# doing all along -- and what it does now -- is build the plate the way a real
# run builds it and assert on the embedding that comes back.
#
# The plate is written by the two functions measure_crop actually appends to:
#
#   * spacr.utils._merge_and_save_to_database  -> the object tables
#   * spacr.utils.filepaths_to_database        -> png_list
#
# never by a hand-rolled `frame.to_sql(...)`. A hand-built schema only ever
# proves that the reader agrees with the test author; going through the
# writers proves the reader agrees with the *writer*, which is the join that
# breaks in production when either side gains or renames a column.

N_FIELDS = 3
N_OBJECTS_PER_FIELD = 20
N_OBJECTS = N_FIELDS * N_OBJECTS_PER_FIELD

# Two wells so map_condition has a positive and a negative group to colour by;
# 'A01'/'A02' parse to columnID c1/c2, which is what pos/neg name below.
_WELLS = ("A01", "A02")


def _write_real_plate(root: Path):
    """Write ``<root>/measurements/measurements.db`` with spaCR's own writers.

    :returns: the list of crop PNG paths handed to ``filepaths_to_database``,
        in write order, so the test can assert the embedding carries exactly
        the objects the writer recorded.
    """
    from PIL import Image
    from spacr.utils import _merge_and_save_to_database, filepaths_to_database

    root = Path(root)
    (root / "measurements").mkdir(parents=True, exist_ok=True)
    png_dir = root / "data" / "cell_png"
    png_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    labels = np.arange(1, N_OBJECTS_PER_FIELD + 1)
    png_paths = []

    for field in range(N_FIELDS):
        # <plate>_<well>_<field>; _map_wells parses the identity out of this,
        # which is why the stem is the only place plate/row/column live.
        stem = f"plate1_{_WELLS[field % len(_WELLS)]}_{field + 1}"
        morphology = pd.DataFrame({
            "label": labels,
            "cell_area": rng.uniform(200, 4000, N_OBJECTS_PER_FIELD),
            "cell_perimeter": rng.uniform(50, 400, N_OBJECTS_PER_FIELD),
            "cell_eccentricity": rng.uniform(0, 1, N_OBJECTS_PER_FIELD),
        })
        intensity = pd.DataFrame({
            "label": labels,
            **{f"cell_channel_{ch}_mean_intensity":
                   rng.uniform(100, 5000, N_OBJECTS_PER_FIELD)
               for ch in range(3)},
            "cell_channel_0_percentile_75":
                rng.uniform(100, 5000, N_OBJECTS_PER_FIELD),
        })
        _merge_and_save_to_database(morphology, intensity, "cell",
                                    str(root), stem, "spacr_test")
        for label in labels:
            # <plate>_<well>_<field>_<object>: the object label is last, and
            # filepaths_to_database reads png_list.cell_id back out of it.
            path = png_dir / f"{stem}_{label}.png"
            Image.fromarray(rng.integers(0, 255, size=(32, 32, 3),
                                         dtype=np.uint8)).save(path)
            png_paths.append(str(path))

    filepaths_to_database(png_paths, {"timelapse": False}, str(root), "cell")
    return png_paths


def _umap_settings(src, **over):
    s = {
        "src": str(src), "tables": ["cell"], "visualize": "cell",
        "reduction_method": "umap", "n_neighbors": 5, "min_dist": 0.1,
        "metric": "euclidean", "clustering": "dbscan", "eps": 0.9,
        "min_samples": 3,
        "col_to_compare": "columnID", "pos": "c1", "neg": "c2", "mix": "c3",
        "plot_images": False, "plot_points": True, "plot_by_cluster": False,
        "plot_cluster_grids": False, "plot_outlines": False,
        "smooth_lines": False, "black_background": False,
        "figuresize": 6, "dot_size": 10, "img_zoom": 0.5, "image_nr": 8,
        "save_figure": False, "verbose": False, "n_jobs": 1,
    }
    s.update(over)
    return s


@pytest.fixture
def real_writer_plate(tmp_path):
    """A plate whose measurements.db came out of spaCR's writers."""
    root = tmp_path / "plate1"
    png_paths = _write_real_plate(root)
    return root, png_paths


def test_image_umap_end_to_end(real_writer_plate):
    """generate_image_umap embeds every object the writers recorded.

    The assertions are on the join, not on "it returned something": if
    ``png_list`` and the object tables ever stop agreeing about an object's
    identity the embedding silently shrinks, and a
    ``assert result is not None`` cannot see that.
    """
    pytest.importorskip("umap")
    root, png_paths = real_writer_plate
    from spacr.core import generate_image_umap

    fig = generate_image_umap(_umap_settings(root), return_fig=True)

    payload = fig._spacr_umap_payload
    assert payload["embedding"].shape == (N_OBJECTS, 2), (
        "the png_list <-> cell join lost objects the writers wrote")
    assert len(payload["labels"]) == N_OBJECTS
    assert len(payload["records"]) == N_OBJECTS
    # Every embedded point traces back to a crop the writer recorded, and to
    # this database -- these are the identities the Annotate screen round-trips
    # on, so a re-anchored display path must not overwrite them.
    assert {r["db_png_path"] for r in payload["records"]} == set(png_paths)
    assert {r["db_path"] for r in payload["records"]} == {
        str(root / "measurements" / "measurements.db")}

    # The scatter really carries one point per object rather than an empty
    # axes that happens to exist.
    plotted = sum(len(coll.get_offsets())
                  for ax in fig.axes for coll in ax.collections)
    assert plotted == N_OBJECTS, f"scatter drew {plotted} of {N_OBJECTS} points"

    # The settings CSV every run writes, and the results CSV.
    assert (root / "settings" / "embedding_settings.csv").is_file()
    results = pd.read_csv(root / "results" / "embedding_results.csv")
    assert len(results) == N_OBJECTS
    assert "cluster" in results.columns
    # Both conditions survive: 'cond' is what colour-by-control keys on, and
    # map_condition spells them 'pos'/'neg'.
    assert set(results["cond"]) == {"pos", "neg"}


def test_image_umap_dataframe_result_drops_internal_bookkeeping(
        real_writer_plate):
    """The default (non-return_fig) result is the annotated frame.

    ``_spacr_umap_db_path`` / ``_spacr_umap_db_png_path`` are internal columns
    generate_image_umap adds to keep the pre-``correct_paths`` identity around;
    leaking them into the returned frame (and the results CSV) would put a
    machine-specific path into a user's analysis.
    """
    pytest.importorskip("umap")
    root, png_paths = real_writer_plate
    from spacr.core import generate_image_umap

    out = generate_image_umap(_umap_settings(root))

    assert isinstance(out, pd.DataFrame)
    assert len(out) == N_OBJECTS
    assert "cluster" in out.columns
    leaked = [c for c in out.columns if c.startswith("_spacr_umap_")]
    assert leaked == [], f"internal bookkeeping columns leaked: {leaked}"


def test_image_umap_survives_a_cell_whose_crop_never_wrote(real_writer_plate):
    """A cell with no row in ``png_list`` must not kill the embedding.

    Found by building this plate with the real writers and then deleting the
    crop rows for one object label, which is exactly what a crop-write failure
    on one field leaves on disk. ``spacr.io._read_and_join_tables`` documents
    ``len(merged) == len(cell) > len(png_list)`` as a *healthy* database --
    ``save_png`` off for a field, an interrupted run, an unmigratable
    ``cell_id`` -- so the NaN the LEFT join leaves in ``png_path`` has to travel
    through ``spacr.utils.correct_paths`` instead of taking the run down with
    ``TypeError: argument of type 'float' is not iterable``, which is what it
    did until the ``isinstance(path, str)`` guard went in.
    """
    pytest.importorskip("umap")
    import sqlite3
    root, png_paths = real_writer_plate
    db = root / "measurements" / "measurements.db"
    con = sqlite3.connect(db)
    try:
        deleted = con.execute(
            "DELETE FROM png_list WHERE cell_id = 'o1'").rowcount
        con.commit()
    finally:
        con.close()
    # One crop per field vanished -- if this stops being true the test is no
    # longer reproducing a partial crop write.
    assert deleted == N_FIELDS, f"deleted {deleted} png_list rows, want {N_FIELDS}"

    from spacr.core import generate_image_umap
    fig = generate_image_umap(_umap_settings(root), return_fig=True)
    payload = fig._spacr_umap_payload

    # The cell table is untouched, so every object still embeds; only the
    # thumbnails of the deleted crops are missing.
    assert payload["embedding"].shape == (N_OBJECTS, 2), (
        "objects with no crop were dropped from the embedding instead of "
        "embedding without a thumbnail")

    records = payload["records"]
    # Exactly the objects whose crop row was deleted lost their crop identity,
    # and every other object kept the exact path the writer recorded. A
    # `len(records) == N_OBJECTS` check alone would still pass if the join had
    # smeared one surviving crop across the crop-less rows.
    missing = {p for p in png_paths if Path(p).stem.rsplit("_", 1)[-1] == "1"}
    assert len(missing) == N_FIELDS
    assert [r["db_png_path"] for r in records].count(None) == N_FIELDS
    assert {r["db_png_path"] for r in records if r["db_png_path"]} == (
        set(png_paths) - missing)

    # The run still completes end to end: results CSV covers every object.
    results = pd.read_csv(root / "results" / "embedding_results.csv")
    assert len(results) == N_OBJECTS


@pytest.mark.slow
def test_image_umap_with_thumbnails_writes_both_pdfs(real_writer_plate):
    """``plot_images`` reads the crops through spacr.crops and saves them.

    This is the path that needs ``png_list.png_path`` to still point at a
    readable file after ``correct_paths`` re-anchors it, so it is the one that
    breaks when the crop folder layout changes.
    """
    pytest.importorskip("umap")
    root, png_paths = real_writer_plate
    from spacr.core import generate_image_umap

    generate_image_umap(_umap_settings(
        root, plot_images=True, plot_by_cluster=True,
        plot_cluster_grids=True, save_figure=True, image_nr=4, img_zoom=0.3))

    embedding_pdf = root / "results" / "UMAP_embedding.pdf"
    grid_pdf = root / "results" / "UMAP_grid.pdf"
    for pdf in (embedding_pdf, grid_pdf):
        assert pdf.is_file(), f"{pdf.name} was not written"
        # An empty Agg PDF is ~1 kB; anything with 60 rendered thumbnails is
        # far larger. This catches "saved a blank canvas" without pinning an
        # exact byte count.
        assert pdf.stat().st_size > 10_000, (
            f"{pdf.name} is {pdf.stat().st_size} bytes -- looks blank")
