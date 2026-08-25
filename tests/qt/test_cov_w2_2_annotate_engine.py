"""The annotator's engine where the inputs are not the clean case.

A crop that is one channel deep, a colour that cannot reach the contrast
target, a filter expression with a dangling operator, a metadata column the
project does not have, and a writer thread whose database has gone. Each of
those is a real thing a user's project does to this module, and each one has
a rule that only shows up in a branch.

The databases here are real sqlite files with the schema a spaCR project
actually has -- `png_list` anchored onto `cell` through `prcfo` -- because
the join in `fetch_filtered_paths` is the part that has broken before, and a
stand-in frame would not have a join in it.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call

from spacr.qt import annotate_engine as ae
from spacr.qt.annotate_engine import (METADATA_COLUMNS, SaveWorker,
                                      annotation_batch, class_counts,
                                      clear_column, count_rows,
                                      fetch_filtered_paths, fetch_page,
                                      find_last_annotated_offset, forget_outline_masks,
                                      gate_paths, label_to_hex, load_crop_image,
                                      metadata_values, normalize_pil,
                                      outline_image, parse_image_type,
                                      paths_by_measurements, paths_by_metadata)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project whose png_list has real metadata and a measurement table."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir(parents=True)

    rng = np.random.default_rng(0)
    rows, cells = [], []
    for i in range(8):
        path = src / "data" / f"cell_{i:02d}.png"
        Image.fromarray(
            rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)).save(path)
        well = "c1" if i < 4 else "c2"
        prcfo = f"plate1_r1_{well}_f1_o{i}"
        rows.append((str(path), "plate1", f"r1{well}", "r1", well, "f1",
                     i, None, prcfo, i))
        cells.append((prcfo, "plate1", "r1", well, "f1", i,
                      float(100 + i * 100), float(10 + i)))

    db = src / "measurements" / "measurements.db"
    con = sqlite3.connect(db)
    try:
        con.execute(
            'CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, plateID TEXT,'
            ' wellID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT,'
            ' label INTEGER, annotate INTEGER, prcfo TEXT, cell_id INTEGER)')
        con.executemany(
            'INSERT INTO "png_list" VALUES (?,?,?,?,?,?,?,?,?,?)', rows)
        con.execute(
            'CREATE TABLE "cell" (prcfo TEXT PRIMARY KEY, plateID TEXT,'
            ' rowID TEXT, columnID TEXT, fieldID TEXT, object_label INTEGER,'
            ' cell_area REAL, nucleus_area REAL)')
        con.executemany('INSERT INTO "cell" VALUES (?,?,?,?,?,?,?,?)', cells)
        con.commit()
    finally:
        con.close()
    return src


def _db(project: Path) -> str:
    return str(project / "measurements" / "measurements.db")


# ---------------------------------------------------------------------------
# class colours
# ---------------------------------------------------------------------------

def test_a_hue_that_cannot_reach_the_target_stops_at_a_usable_darkness():
    """Below 0.2 every hue is the same near-black and class identity is lost.

    A colour that cannot pass the contrast target by then is made as dark as
    it is useful to make it, and no darker.
    """
    assert ae._darken_until_readable(0.16, 1.0, 1.0) >= 0.2

    # a hue that reads at full value is not darkened at all
    bright = ae._darken_until_readable(0.66, 1.0, 1.0)
    assert bright == 1.0

    # and a colour that is already below the floor is clamped to it
    assert ae._darken_until_readable(0.16, 1.0, 0.15) == 0.2


def test_the_class_colours_differ_between_the_themes():
    """Contrast depends on what is behind the tile, so the palette does too."""
    for value in (1, 2, 4, 7):
        assert label_to_hex(value, dark=True) != label_to_hex(value,
                                                              dark=False)


def test_a_light_tile_colour_actually_clears_the_readability_floor():
    """Issue #6 was five of the first six classes below 3.0 on a light tile."""
    for value in range(1, 12):
        hex_colour = label_to_hex(value, dark=False)
        rgb = tuple(int(hex_colour[i:i + 2], 16) / 255.0
                    for i in (1, 3, 5))
        contrast = ((ae._LIGHT_TILE_LUMINANCE + 0.05)
                    / (ae._relative_luminance(rgb) + 0.05))
        assert contrast >= 2.9, (value, hex_colour, contrast)


def test_nothing_annotated_gets_no_border():
    """None, 0 and a non-integer all mean "not labelled"."""
    assert label_to_hex(None) is None
    assert label_to_hex(0) is None
    assert label_to_hex("two") is None


# ---------------------------------------------------------------------------
# reading a crop
# ---------------------------------------------------------------------------

def test_an_unknown_stored_channel_order_is_refused(tmp_path):
    """Guessing would silently permute the planes of every crop."""
    path = tmp_path / "crop.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)

    with pytest.raises(ValueError) as raised:
        load_crop_image(str(path), stored_channel_order="brg")
    assert "legacy_bgr" in str(raised.value)


def test_a_single_channel_image_is_stretched_as_one_plane():
    """A 16-bit single-channel crop opened as RGB is clipped to solid white.

    The grey path exists so it is normalised as the one plane it is.
    """
    array = np.linspace(0, 255, 64, dtype=np.uint8).reshape(8, 8)
    array[0, 0] = 0
    out = np.asarray(normalize_pil(Image.fromarray(array),
                                   normalize_channels=["r"]))
    assert out.ndim == 2
    assert out.min() == 0
    assert out.max() == 255


def test_a_channel_letter_the_image_does_not_have_is_skipped():
    """A crop with three planes cannot normalise a fourth."""
    array = np.zeros((8, 8, 3), np.uint8)
    array[..., 0] = np.linspace(10, 200, 64).reshape(8, 8)
    before = array.copy()

    out = np.asarray(normalize_pil(Image.fromarray(array),
                                   normalize_channels=["x", "r"]))
    assert not np.array_equal(out[..., 0], before[..., 0])
    assert np.array_equal(out[..., 1], before[..., 1])


def test_normalising_nothing_leaves_the_image_alone():
    """An empty channel list is "do not stretch", not "stretch everything"."""
    array = np.full((8, 8, 3), 40, np.uint8)
    out = np.asarray(normalize_pil(Image.fromarray(array)))
    assert np.array_equal(out, array)


# ---------------------------------------------------------------------------
# outlines
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def outline_caches_cleared():
    """The mask and edge caches are module globals."""
    forget_outline_masks()
    yield
    forget_outline_masks()


def _blob_image(size=32):
    """One bright square on a dark field, in every channel."""
    array = np.zeros((size, size, 3), np.uint8)
    array[8:24, 8:24, :] = 220
    return Image.fromarray(array)


def test_asking_for_no_outline_returns_the_image_unchanged():
    """Three ways to mean "no outline", each of them a no-op."""
    base = _blob_image()
    assert outline_image(base, base, outline_channels=None) is base
    assert outline_image(base, base, outline_channels=["r"],
                         edge_transparency=0) is base
    assert outline_image(base, base, outline_channels=["x", "q"]) is base


def test_an_image_that_is_not_three_channel_gets_no_outline():
    """The channel map is r/g/b; a grey crop has nowhere to put an edge."""
    grey = Image.fromarray(np.zeros((16, 16), np.uint8))
    assert outline_image(grey, grey, outline_channels=["r"]) is grey


def test_an_outline_is_drawn_and_the_channel_is_otherwise_blanked():
    """The default view is the outline alone, not the outline over the data."""
    base = _blob_image()
    out = np.asarray(outline_image(base, base, outline_channels=["r"],
                                   edge_transparency=100.0))
    assert out[:, :, 0].max() == 255
    assert out[16, 16, 0] == 0, "the channel's interior was not blanked"
    assert np.array_equal(out[:, :, 1], np.asarray(base)[:, :, 1])


def test_the_edge_image_view_keeps_the_data_under_the_outline():
    """`edge_image` is the other view: the outline drawn over the channel."""
    base = _blob_image()
    out = np.asarray(outline_image(base, base, outline_channels=["r"],
                                   edge_image=True, edge_transparency=100.0))
    assert out[16, 16, 0] == 220, "the interior was blanked in edge_image mode"
    assert out[:, :, 0].max() == 255


def test_an_object_size_that_is_not_a_pair_is_treated_as_no_filter():
    """The field is free text in a dialog, so it arrives as anything."""
    base = _blob_image()
    out = np.asarray(outline_image(base, base, outline_channels=["r"],
                                   object_size=None))
    assert out[:, :, 0].max() == 255


def test_an_object_size_window_drops_the_objects_outside_it():
    """One threshold is a gate; the window is what selects a population."""
    base = _blob_image()
    kept = np.asarray(outline_image(base, base, outline_channels=["r"],
                                    object_size=(1, 100000)))
    assert kept[:, :, 0].max() == 255

    dropped = np.asarray(outline_image(base, base, outline_channels=["r"],
                                       object_size=(100000, 200000)))
    assert dropped[:, :, 0].max() == 0, "an out-of-window object was outlined"


def test_a_flat_channel_does_not_break_the_threshold():
    """Otsu has nothing to separate on a constant image and raises."""
    flat = Image.fromarray(np.full((16, 16, 3), 30, np.uint8))
    out = outline_image(flat, flat, outline_channels=["r"])
    assert np.asarray(out).shape == (16, 16, 3)


def test_a_mask_is_computed_once_and_reused_for_a_display_only_change(
        monkeypatch):
    """Normalisation, opacity and thickness must not re-run the threshold."""
    from skimage import filters

    calls = []
    real = filters.threshold_otsu

    def counted(image, *args, **kwargs):
        calls.append(1)
        return real(image, *args, **kwargs)

    monkeypatch.setattr(filters, "threshold_otsu", counted)

    base = _blob_image()
    outline_image(base, base, outline_channels=["r"], edge_transparency=100.0)
    first = len(calls)
    outline_image(base, base, outline_channels=["r"], edge_transparency=40.0)
    assert len(calls) == first, "a transparency change recomputed the mask"


def test_the_caches_are_bounded(monkeypatch):
    """A montage tab is a few hundred crops; the caches cannot grow with it."""
    monkeypatch.setattr(ae, "_MASK_CACHE_SIZE", 2)
    for i in range(5):
        array = np.zeros((32, 32, 3), np.uint8)
        # a differently sized blob each time, so the MASK and the EDGE both
        # differ -- two caches, and both of them have to stay bounded
        array[4:8 + i * 3, 4:8 + i * 3, :] = 220
        image = Image.fromarray(array)
        outline_image(image, image, outline_channels=["r"])

    assert len(ae._MASK_CACHE) <= 2
    assert len(ae._EDGE_CACHE) <= 2


def test_a_threshold_that_cannot_be_computed_falls_back_to_the_median(
        monkeypatch):
    """Otsu has nothing to separate on some crops, and an outline is still
    better than an exception out of a paint."""
    from skimage import filters

    def explode(_image):
        raise ValueError("threshold_otsu is not defined for this image")

    monkeypatch.setattr(filters, "threshold_otsu", explode)

    base = _blob_image()
    out = np.asarray(outline_image(base, base, outline_channels=["r"],
                                   edge_transparency=100.0))
    assert out.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# the Cellpose outline route
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_cellpose(monkeypatch):
    """`CellposeModel` swapped for a double on the real module, so no
    weights and no GPU.

    The real model downloads `cpsam` and takes the card; what is under test
    here is the lazy construction, the caching, and the fallback -- none of
    which is about segmentation quality.

    The double replaces the class ON the installed `cellpose.models` rather
    than standing a synthetic package up in `sys.modules`. A synthetic
    package answers any import, so it would keep passing if
    `_get_cellpose_outline_model` reached for a module layout the library no
    longer has -- and it also hides `cellpose.transforms`, which is the real
    validator `check_cellpose_eval_call` runs the arguments through.
    """
    from cellpose import models as cp_models

    built = []

    class _Model:
        """A `CellposeModel` stand-in declaring the installed signatures.

        Written out in full, with no `**kwargs`: a double that accepts every
        keyword cannot fail when spaCR passes one this cellpose removed, and
        `annotate_engine._cellpose_foreground` is a real call site.
        """

        def __init__(self, gpu=False, pretrained_model="cpsam",
                     model_type=None, diam_mean=None, device=None, nchan=None,
                     use_bfloat16=True):
            built.append({"gpu": gpu, "pretrained_model": pretrained_model})

        def eval(self, image, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            # This call site hands cellpose one 2-D plane and lets it
            # auto-detect, so an axis is not required -- but whatever arrives
            # must be one convert_image accepts.
            check_cellpose_eval_call(image, channel_axis, z_axis=z_axis,
                                     do_3D=do_3D,
                                     stitch_threshold=stitch_threshold,
                                     require_channel_axis=False)
            mask = np.zeros(np.shape(image), dtype=np.int32)
            mask[4:12, 4:12] = 1
            return [mask], None, None

    monkeypatch.setattr(cp_models, "CellposeModel", _Model)
    monkeypatch.setattr(ae, "_cellpose_outline_model", None)
    yield built
    ae._cellpose_outline_model = None


def test_the_outline_model_is_built_once_and_kept(stub_cellpose):
    """Rebuilding it per crop would cost more than the segmentation."""
    first = ae._get_cellpose_outline_model()
    second = ae._get_cellpose_outline_model()
    assert first is second
    assert len(stub_cellpose) == 1
    assert stub_cellpose[0]["pretrained_model"] == "cpsam"
    assert stub_cellpose[0]["gpu"] is False, "a test asked for the card"


def test_a_cellpose_outline_is_drawn_from_its_mask(stub_cellpose,
                                                   monkeypatch):
    """A list of masks is what CellposeModel.eval returns."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    base = _blob_image(16)
    out = np.asarray(outline_image(base, base, outline_channels=["r"],
                                   outline_method="cellpose",
                                   edge_transparency=100.0))
    assert out[:, :, 0].max() == 255


def test_a_cellpose_that_is_not_installed_falls_back_to_otsu(monkeypatch):
    """The outline is the point; which threshold drew it is not."""
    monkeypatch.setattr(ae, "_cellpose_outline_model", None)
    monkeypatch.setitem(sys.modules, "cellpose", None)

    base = _blob_image()
    out = np.asarray(outline_image(base, base, outline_channels=["r"],
                                   outline_method="cellpose",
                                   edge_transparency=100.0))
    assert out[:, :, 0].max() == 255


# ---------------------------------------------------------------------------
# the image-type filter language
# ---------------------------------------------------------------------------

def test_an_empty_filter_filters_nothing():
    """No expression, and an expression of only whitespace, are the same."""
    assert parse_image_type(None) == ("", [])
    assert parse_image_type("") == ("", [])
    assert parse_image_type("   ") == ("", [])


def test_a_bare_bang_is_the_word_not(project):
    """`! pathogen` with a space is what someone types half the time."""
    sql, params = parse_image_type("! pathogen")
    assert sql == "(NOT png_path LIKE ?)"
    assert params == ["%pathogen%"]

    attached, attached_params = parse_image_type("!pathogen")
    assert (attached, attached_params) == (sql, params)


def test_and_binds_tighter_than_or():
    """As everywhere else, so the expression means what it reads as."""
    sql, params = parse_image_type("cell AND nucleus OR pathogen")
    assert sql == "((png_path LIKE ? AND png_path LIKE ?) OR png_path LIKE ?)"
    assert params == ["%cell%", "%nucleus%", "%pathogen%"]


def test_every_term_is_a_bound_parameter():
    """A path fragment containing a quote is a fragment, not an injection."""
    sql, params = parse_image_type("o'brien")
    assert "o'brien" not in sql
    assert params == ["%o'brien%"]


def test_an_expression_that_cannot_be_read_names_what_was_wrong():
    """A dangling operator, an empty NOT, an unclosed bracket."""
    for expression, fragment in (
            ("cell AND", "ends after an operator"),
            ("NOT", "ends after an operator"),
            ("(cell", "unclosed"),
            ("AND cell", "where a path fragment was expected"),
            ("cell nucleus", "could not read the image filter after"),
    ):
        with pytest.raises(ValueError) as raised:
            parse_image_type(expression)
        assert fragment in str(raised.value), expression


def test_brackets_group_the_way_they_look():
    """`cell AND (nucleus OR pathogen)` is not the same as without them."""
    sql, params = parse_image_type("cell AND (nucleus OR pathogen)")
    assert sql == \
        "(png_path LIKE ? AND ((png_path LIKE ? OR png_path LIKE ?)))"
    assert params == ["%cell%", "%nucleus%", "%pathogen%"]


# ---------------------------------------------------------------------------
# reading the database that is not there
# ---------------------------------------------------------------------------

def test_a_missing_database_is_empty_rather_than_an_error(tmp_path):
    """A project folder without measurements is an ordinary state."""
    missing = str(tmp_path / "nothing" / "measurements.db")
    assert count_rows(missing) == 0
    assert fetch_page(missing, "annotate", 0, 10) == []
    assert class_counts(missing, "annotate") == []
    assert find_last_annotated_offset(missing, "annotate", 10) is None
    assert metadata_values(missing, "plateID") == []
    assert paths_by_metadata(missing, "plateID", ["plate1"]) == []
    assert fetch_filtered_paths(missing, "annotate", ["cell_area"], [1],
                                ["higher"]) == []
    clear_column(missing, "annotate")          # must not raise


def test_a_column_outside_the_metadata_set_is_refused(project):
    """It would otherwise interpolate an arbitrary name into SQL."""
    for helper in (metadata_values, ):
        with pytest.raises(ValueError) as raised:
            helper(_db(project), "cell_area")
        assert "cell_area" in str(raised.value)

    with pytest.raises(ValueError):
        paths_by_metadata(_db(project), "png_path", ["x"])


def test_a_metadata_column_the_project_lacks_is_empty(project):
    """A non-timelapse project has no timeID, and that is not an error."""
    assert "timeID" in METADATA_COLUMNS
    assert metadata_values(_db(project), "timeID") == []
    assert paths_by_metadata(_db(project), "timeID", ["t1"]) == []


def test_selecting_no_metadata_values_selects_nothing(project):
    """An empty picker selection is an empty population."""
    assert paths_by_metadata(_db(project), "columnID", []) == []


def test_no_measurement_rules_select_nothing(project):
    """Nothing to threshold on is an empty population, not everything."""
    assert paths_by_measurements(_db(project), "annotate", []) == []


def test_a_rule_missing_a_field_is_refused(project):
    """A silently dropped rule is a plausible-looking wrong population."""
    with pytest.raises(ValueError) as raised:
        paths_by_measurements(_db(project), "annotate",
                              [{"threshold": 1.0}])
    assert "needs a column and a threshold" in str(raised.value)

    with pytest.raises(ValueError) as raised:
        paths_by_measurements(_db(project), "annotate",
                              [{"column": "cell_area", "threshold": 1.0,
                                "direction": "sideways"}])
    assert "higher" in str(raised.value)


def test_several_measurements_are_anded(project):
    """One threshold is a gate, not a population."""
    both = paths_by_measurements(
        _db(project), "annotate",
        [{"column": "cell_area", "threshold": 350, "direction": "higher"},
         {"column": "nucleus_area", "threshold": 16, "direction": "lower"}])
    wide = paths_by_measurements(
        _db(project), "annotate",
        [{"column": "cell_area", "threshold": 350, "direction": "higher"}])
    assert both
    assert set(both) < set(wide)


def test_a_direction_that_is_neither_leaves_the_frame_alone():
    """`_apply_threshold` filters on the two directions it documents."""
    import pandas as pd

    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert len(ae._apply_threshold(frame, "a", 2.0, "sideways")) == 3
    assert len(ae._apply_threshold(frame, "missing", 2.0, "higher")) == 3
    assert len(ae._apply_threshold(frame, "a", None, "higher")) == 3
    assert len(ae._apply_threshold(frame, "a", 2.0, "higher")) == 1
    assert len(ae._apply_threshold(frame, "a", 2.0, "lower")) == 1


def test_one_threshold_is_broadcast_over_several_columns(project):
    """The documented shorthand, and it must keep working."""
    rows = fetch_filtered_paths(_db(project), "annotate",
                                ["cell_area", "nucleus_area"], [5],
                                ["higher"])
    assert rows
    assert all(len(row) == 2 for row in rows)

    as_string = fetch_filtered_paths(_db(project), "annotate",
                                     ["cell_area", "nucleus_area"], [5],
                                     "higher")
    assert as_string == rows


def test_three_columns_and_two_thresholds_is_refused(project):
    """Both fields are free text, so the two lists disagreeing is a typo away.

    There is no defensible pairing to guess, and the consequence is not a
    crash but a wrong population that gets hand-labelled.
    """
    with pytest.raises(ValueError) as raised:
        fetch_filtered_paths(_db(project), "annotate",
                             ["cell_area", "nucleus_area", "object_label"],
                             [5, 6], ["higher", "higher", "higher"])
    message = str(raised.value)
    assert "3 measurement column(s)" in message
    assert "2 threshold(s)" in message


def test_the_filter_narrows_by_path_as_well_as_by_measurement(project):
    """The image-type filter composes with the thresholds."""
    all_rows = fetch_filtered_paths(_db(project), "annotate", ["cell_area"],
                                    [0], ["higher"])
    assert all_rows

    narrowed = fetch_filtered_paths(_db(project), "annotate", ["cell_area"],
                                    [0], ["higher"], image_type="cell_00")
    assert 0 < len(narrowed) < len(all_rows)


# ---------------------------------------------------------------------------
# the gate hand-off
# ---------------------------------------------------------------------------

def test_no_gates_select_nothing(project):
    """An empty chain is an empty population, not the whole plate."""
    assert gate_paths(_db(project), []) == []


def test_a_gate_chain_selects_the_same_population_it_plots(project):
    """The gate maths is not reproduced here; `GateClause` evaluates it.

    A population gated on screen and one annotated from it are the same
    population by construction, which is only true if this route really goes
    through the same clause.
    """
    from spacr.qt.widgets.gate_spec import ThresholdGate

    gate = ThresholdGate(name="big cells", column="cell_area", low=350.0)
    paths = gate_paths(_db(project), [gate])

    assert paths
    assert all(path.endswith(".png") for path in paths)
    everything = [row[0] for row in fetch_page(_db(project), "annotate", 0,
                                               100)]
    assert set(paths) < set(everything)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def test_a_batch_is_a_path_to_value_mapping():
    """Every auto-annotation source ends at the same call."""
    assert annotation_batch(["a", Path("b")], 2) == {"a": 2, "b": 2}
    assert annotation_batch(["a"], None) == {"a": None}
    assert annotation_batch([], 1) == {}


def _worker(project):
    worker = SaveWorker(_db(project), "annotate")
    worker.start()
    return worker


def test_a_worker_is_started_once(project):
    """A second start would put two writers on one sqlite file."""
    worker = _worker(project)
    try:
        thread = worker._thread
        worker.start()
        assert worker._thread is thread
        assert worker.is_alive is True
    finally:
        worker.stop()


def test_an_empty_batch_is_not_queued(project):
    """Nothing to write is not a pending write."""
    worker = SaveWorker(_db(project), "annotate")
    worker.submit({})
    assert worker.pending_batches == 0


def test_annotations_reach_the_database_and_can_be_cleared(project):
    """The round trip: submit, drain, read the counts back, then reset."""
    worker = _worker(project)
    try:
        rows = fetch_page(_db(project), "annotate", 0, 100)
        paths = [row[0] for row in rows]
        worker.submit(annotation_batch(paths[:3], 1))
        worker.submit(annotation_batch(paths[3:5], 2))
    finally:
        worker.stop()

    assert class_counts(_db(project), "annotate") == [(1, 3), (2, 2)]
    assert worker.last_save_ts is not None
    assert worker.pending_batches == 0
    assert worker.last_error is None

    offset = find_last_annotated_offset(_db(project), "annotate", 2)
    assert offset == 4

    clear_column(_db(project), "annotate")
    assert class_counts(_db(project), "annotate") == []


def test_a_clear_is_written_as_null_not_as_zero(project):
    """None clears, exactly as it does for a keystroke."""
    rows = [row[0] for row in fetch_page(_db(project), "annotate", 0, 100)]
    worker = _worker(project)
    try:
        worker.submit(annotation_batch(rows[:2], 1))
    finally:
        worker.stop()
    assert class_counts(_db(project), "annotate") == [(1, 2)]

    worker = _worker(project)
    try:
        worker.submit(annotation_batch(rows[:2], None))
    finally:
        worker.stop()
    assert class_counts(_db(project), "annotate") == []


def test_a_burst_of_batches_is_coalesced_into_one_commit(project):
    """A held key sends one batch per press; sqlite wants one transaction."""
    rows = [row[0] for row in fetch_page(_db(project), "annotate", 0, 100)]
    worker = SaveWorker(_db(project), "annotate")
    for index, path in enumerate(rows):
        worker.submit({path: (index % 2) + 1})
    assert worker.pending_batches == len(rows)

    worker.start()
    worker.stop()

    assert worker.pending_batches == 0
    counts = dict(class_counts(_db(project), "annotate"))
    assert sum(counts.values()) == len(rows)


def test_a_writer_that_cannot_open_the_database_says_so(tmp_path):
    """The message names the failure and tells the user not to close yet."""
    unreachable = tmp_path / "no_such_dir" / "measurements.db"
    worker = SaveWorker(str(unreachable), "annotate")
    worker.start()
    worker.stop()

    assert worker.last_error is not None
    assert "could not start" in worker.last_error


def test_edits_made_after_a_failure_are_retained_not_discarded(project):
    """They are not called saved, and they are never thrown away."""
    worker = SaveWorker(_db(project), "annotate")
    with worker._lock:
        worker._last_error = "the database is read-only"

    worker.submit({"a.png": 1})
    worker.submit({"b.png": 2})

    assert worker.pending_batches == 1
    assert worker._failed_batch == {"a.png": 1, "b.png": 2}


def test_a_failed_write_is_reported_and_the_batch_is_kept(project,
                                                          monkeypatch):
    """A rolled-back transaction leaves the edits on the worker, not lost."""
    import spacr.qt.annotate_engine as engine

    def explode(_conn):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine, "transaction", explode)

    worker = _worker(project)
    try:
        worker.submit({"a.png": 1})
    finally:
        worker.stop()

    assert worker.last_error is not None
    assert "database is locked" in worker.last_error
    assert "resolve the database problem" in worker.last_error
    assert worker._failed_batch == {"a.png": 1}
    assert worker.busy is False


def test_stopping_twice_is_harmless(project):
    """A screen closing can call it, and so can application shutdown."""
    worker = _worker(project)
    worker.stop()
    worker.stop()
    assert worker.is_alive is False


def test_a_thread_that_cannot_be_joined_does_not_block_the_close(project,
                                                                 monkeypatch):
    """The screen must close even if the join itself raises."""
    worker = _worker(project)
    try:
        real_thread = worker._thread

        class _Refusing:
            def join(self, *_args):
                raise RuntimeError("cannot join the current thread")

            def is_alive(self):
                return False

        worker._thread = _Refusing()

        worker.stop()

        # STOPPED, not merely survived. The worker has to report itself
        # dead even though the join threw, or the screen closes believing
        # a thread is still running that it can never wait for.
        assert not worker.is_alive
    finally:
        worker._thread = real_thread
        worker.stop()


def test_a_cursor_that_will_not_close_does_not_take_the_thread_down(
        project, monkeypatch):
    """sqlite raising on close is the last thing before the thread exits."""
    import spacr.qt.annotate_engine as engine

    real_connect = engine.connect_database

    class _Cursor:
        def __init__(self, cur):
            self._cur = cur

        def __getattr__(self, name):
            return getattr(self._cur, name)

        def close(self):
            raise sqlite3.ProgrammingError("cursor already closed")

    class _Connection:
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def cursor(self):
            return _Cursor(self._conn.cursor())

        def close(self):
            self._conn.close()

    def wrapped(*args, **kwargs):
        return _Connection(real_connect(*args, **kwargs))

    monkeypatch.setattr(engine, "connect_database", wrapped)

    worker = _worker(project)
    worker.submit({"a.png": 1})
    worker.stop()                              # must not raise
    assert worker.is_alive is False
    assert worker.last_error is None, worker.last_error


def test_an_idle_writer_waits_without_spinning_on_a_closed_queue(project):
    """The thread polls with a timeout, so an idle screen costs nothing."""
    import time

    worker = _worker(project)
    try:
        time.sleep(0.3)
        assert worker.is_alive is True
        assert worker.pending_batches == 0
    finally:
        worker.stop()
    assert worker.is_alive is False


def test_a_burst_that_arrives_while_the_writer_runs_is_coalesced(project):
    """The coalescing loop drains what is queued and commits it once."""
    import time

    rows = [row[0] for row in fetch_page(_db(project), "annotate", 0, 100)]
    worker = _worker(project)
    try:
        for index, path in enumerate(rows):
            worker.submit({path: (index % 3) + 1})
        time.sleep(0.5)
        assert worker.pending_batches == 0
    finally:
        worker.stop()

    counts = dict(class_counts(_db(project), "annotate"))
    assert sum(counts.values()) == len(rows)


# ---------------------------------------------------------------------------
# reading a database that is there
# ---------------------------------------------------------------------------

def test_the_row_count_follows_the_image_filter(project):
    """The count under the grid has to mean the same thing the grid shows."""
    assert count_rows(_db(project)) == 8
    assert count_rows(_db(project), "cell_00") == 1
    assert count_rows(_db(project), "cell_00 OR cell_01") == 2
    assert count_rows(_db(project), "NOT cell_00") == 7


def test_a_page_is_read_in_insertion_order(project):
    """"Page 2" has to mean the same rows every time it is asked for."""
    first = fetch_page(_db(project), "annotate", 0, 3)
    second = fetch_page(_db(project), "annotate", 3, 3)
    assert len(first) == 3 and len(second) == 3
    assert not set(p for p, _ in first) & set(p for p, _ in second)
    assert fetch_page(_db(project), "annotate", 0, 3) == first


def test_the_metadata_picker_reads_the_project_s_own_values(project):
    """Plates are named by whoever ran them."""
    assert metadata_values(_db(project), "columnID") == ["c1", "c2"]
    assert metadata_values(_db(project), "plateID") == ["plate1"]

    selected = paths_by_metadata(_db(project), "columnID", ["c2"])
    assert len(selected) == 4
    assert all(path.endswith(".png") for path in selected)
    assert paths_by_metadata(_db(project), "columnID", ["c1", "c2"]) != \
        selected


def test_nothing_annotated_has_no_last_page(project):
    """A fresh column has no offset to resume at."""
    assert find_last_annotated_offset(_db(project), "annotate", 4) is None
