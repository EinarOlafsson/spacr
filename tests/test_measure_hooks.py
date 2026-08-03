"""Contract tests for the measurement extension points.

Two hooks are wired into :func:`spacr.measure._measure_crop_core`:

* a **preprocessing** hook that replaces the intensity channels just before
  any feature is computed (the consumer is illumination / flat-field
  correction), and
* a **region filter** that decides, per object, whether it is measured at all
  (the consumer is a user-drawn ROI).

The tests below drive both of them through the real ``_measure_crop_core`` on a
hand-built merged stack -- no GPU, no Cellpose -- and check the four things
that actually matter: the default path is untouched, a preprocessing hook moves
the numbers by the amount it says it does, a region filter removes exactly the
objects it names and no others, and a hook that misbehaves stops the field
instead of quietly producing wrong rows.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import measure_hooks as mh
from spacr.settings import get_measure_crop_settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _pristine_registries(monkeypatch):
    """Every test starts and ends with both registries empty.

    The registries are module-level process state; a hook leaking out of one
    test would silently change every measurement made by the next one.
    """
    monkeypatch.delenv(mh.HOOKS_ENV_VAR, raising=False)
    mh.clear_measurement_hooks()
    yield
    mh.clear_measurement_hooks()


def _flat_masks():
    """Four identical, well-separated cells with a nucleus and a pathogen each.

    Deliberately noiseless: every object has the same area and the same
    intensity, so any difference a test sees is the hook's doing and not the
    fixture's.
    """
    height = width = 128
    cell = np.zeros((height, width), dtype=np.uint16)
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    yy, xx = np.mgrid[:height, :width]
    for label, (cy, cx) in enumerate([(30, 30), (30, 90), (90, 30), (90, 90)],
                                     start=1):
        cell[(yy - cy) ** 2 + (xx - cx) ** 2 <= 18 ** 2] = label
        nucleus[(yy - cy) ** 2 + (xx - cx) ** 2 <= 7 ** 2] = label
        pathogen[(yy - (cy + 10)) ** 2 + (xx - cx) ** 2 <= 3 ** 2] = label
    return cell, nucleus, pathogen


#: Background and in-cell signal of each intensity channel, so a test can
#: predict the exact mean a hook should produce.
_BACKGROUND = (100, 110)
_CELL_SIGNAL = 1000


def _project(tmp_path, name="plate1_A01_F001.npy"):
    """Write one merged field and return ``(project_dir, merged_dir, file)``."""
    cell, nucleus, pathogen = _flat_masks()
    channels = []
    for background in _BACKGROUND:
        plane = np.full(cell.shape, background, dtype=np.uint16)
        plane[cell > 0] += _CELL_SIGNAL
        channels.append(plane)
    data = np.stack(channels + [cell, nucleus, pathogen], axis=-1)
    merged = tmp_path / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    # _merge_and_save_to_database opens <parent>/measurements/measurements.db
    # without creating the folder; the real pipeline makes it upstream.
    (tmp_path / "measurements").mkdir(parents=True, exist_ok=True)
    np.save(merged / name, data.astype(np.uint16))
    return tmp_path, str(merged), name


def _settings(merged, **over):
    """measure_crop settings for the fixture above.

    Homogeneity / radial distribution / correlation are off: they are covered
    elsewhere, they dominate the runtime, and none of them changes what these
    tests assert.
    """
    settings = get_measure_crop_settings(settings={})
    settings.update({
        "src": merged,
        "channels": [0, 1],
        "cell_mask_dim": 2, "nucleus_mask_dim": 3, "pathogen_mask_dim": 4,
        "save_measurements": True, "save_png": False, "save_arrays": False,
        "plot": False, "verbose": False, "timelapse": False,
        "crop_mode": ["cell"], "normalize": [1, 99], "normalize_by": "png",
        "experiment": "exp", "n_jobs": 1, "test_mode": False,
        "cytoplasm": True,
        "homogeneity": False, "radial_dist": False,
        "calculate_correlation": False,
    })
    settings.update(over)
    return settings


def _run(tmp_path, **over):
    """Measure one field into a fresh project and return ``(result, tables)``."""
    from spacr.measure import _measure_crop_core

    project, merged, name = _project(tmp_path)
    result = _measure_crop_core(0, [], name, _settings(merged, **over))
    return result, _tables(project)


def _tables(project):
    """Read every table of the project's measurements.db into DataFrames.

    ``path_name`` is dropped: it holds the absolute source folder, which
    differs between two tmp dirs for reasons that have nothing to do with the
    measurement.
    """
    db = os.path.join(str(project), "measurements", "measurements.db")
    if not os.path.isfile(db):
        return {}
    frames = {}
    con = sqlite3.connect(db)
    try:
        names = [row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        for name in names:
            frame = pd.read_sql(f'SELECT * FROM "{name}"', con)
            frames[name] = frame.drop(columns=["path_name"], errors="ignore")
    finally:
        con.close()
    return frames


# ---------------------------------------------------------------------------
# The no-hook path is untouched
# ---------------------------------------------------------------------------

def test_registries_are_empty_and_the_entry_points_are_pass_through():
    """Nothing is registered by default, and both hooks return their input."""
    assert mh.preprocessing_hooks() == ()
    assert mh.region_filter_hooks() == ()
    assert mh.describe_hooks() == "no measurement hooks registered"

    array = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    context = mh.PreprocessingContext(
        file_name="f", channels=[0, 1, 2, 3], settings={})
    # `is`, not ==: the default path must not copy, cast or round-trip the
    # intensity array at all.
    assert mh.apply_preprocessing_hooks(array, context) is array

    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[1:4, 1:4] = 1
    kept, dropped = mh.apply_region_filter_hooks(
        mask, object_type="cell", file_name="f", settings={})
    assert kept is mask
    assert dropped == ()


def test_measurements_are_unchanged_when_no_hook_is_registered(tmp_path):
    """Two no-hook runs of the same field agree column for column.

    This is the byte-identity guard: it is what fails if the hook wiring ever
    starts copying, casting or reordering something on the default path.
    """
    (_, baseline) = _run(tmp_path / "a")
    (_, again) = _run(tmp_path / "b")

    assert set(baseline) == {"cell", "nucleus", "pathogen", "cytoplasm"}
    for name, frame in baseline.items():
        pd.testing.assert_frame_equal(frame, again[name])


def test_registering_then_clearing_leaves_the_default_result(tmp_path):
    """A hook that is registered and then removed changes nothing at all."""
    (_, baseline) = _run(tmp_path / "a")

    name = mh.register_preprocessing_hook(
        lambda array, context: array * 3, name="temporary")
    assert mh.unregister_preprocessing_hook(name) is True
    assert mh.unregister_preprocessing_hook(name) is False

    (_, after) = _run(tmp_path / "b")
    for table, frame in baseline.items():
        pd.testing.assert_frame_equal(frame, after[table])


# ---------------------------------------------------------------------------
# Preprocessing hook: the numbers move, and only the intensity numbers
# ---------------------------------------------------------------------------

def test_preprocessing_hook_scales_the_measured_intensities(tmp_path):
    """A x2 gain doubles every intensity column and touches no morphology."""
    (_, baseline) = _run(tmp_path / "a")

    def double(array, context):
        # Same dtype back out -- the rounding is the hook's decision, which is
        # exactly why apply_preprocessing_hooks refuses to make it.
        return np.clip(array.astype(np.float64) * 2.0,
                       0, np.iinfo(array.dtype).max).astype(array.dtype)

    mh.register_preprocessing_hook(double, name="x2")
    (_, scaled) = _run(tmp_path / "b")

    for channel, background in enumerate(_BACKGROUND):
        column = f"cell_channel_{channel}_mean_intensity"
        expected = float(background + _CELL_SIGNAL)
        assert baseline["cell"][column].tolist() == [expected] * 4
        assert scaled["cell"][column].tolist() == [expected * 2] * 4
        # The direction is asserted per object as well, not only in aggregate.
        assert (scaled["cell"][column].to_numpy()
                > baseline["cell"][column].to_numpy()).all()

    # Morphology is computed from the label masks, which the hook never saw.
    pd.testing.assert_series_equal(baseline["cell"]["cell_area"],
                                   scaled["cell"]["cell_area"])
    assert baseline["cell"]["object_label"].tolist() == \
        scaled["cell"]["object_label"].tolist()

    # The child objects are measured from the same corrected array.
    for channel, background in enumerate(_BACKGROUND):
        column = f"nucleus_channel_{channel}_mean_intensity"
        assert scaled["nucleus"][column].tolist() == \
            [float(background + _CELL_SIGNAL) * 2] * 4


def test_preprocessing_hook_sees_the_selected_channels_and_the_field(tmp_path):
    """The context names the field and the channel indices, in array order."""
    seen = {}

    def record(array, context):
        seen["shape"] = array.shape
        seen["dtype"] = array.dtype
        seen["file_name"] = context.file_name
        seen["channels"] = context.channels
        seen["volumetric"] = context.volumetric
        seen["src"] = context.settings["src"]
        return array

    mh.register_preprocessing_hook(record, name="record")
    _run(tmp_path, channels=[1, 0])

    assert seen["file_name"] == "plate1_A01_F001"
    assert seen["channels"] == (1, 0)
    assert seen["shape"] == (128, 128, 2)
    assert seen["dtype"] == np.uint16
    assert seen["volumetric"] is False
    assert seen["src"].endswith("merged")


def test_preprocessing_hook_cannot_mutate_the_run_settings(tmp_path):
    """The settings handed to a hook are a read-only view."""
    captured = {}

    def meddle(array, context):
        captured["settings"] = context.settings
        return array

    mh.register_preprocessing_hook(meddle, name="meddle")
    _run(tmp_path)

    with pytest.raises(TypeError):
        captured["settings"]["save_png"] = True


# ---------------------------------------------------------------------------
# Region filter: exactly the excluded objects go missing
# ---------------------------------------------------------------------------

def test_region_filter_removes_exactly_the_excluded_objects(tmp_path):
    """Dropping cells 2 and 4 leaves 1 and 3, measured identically."""
    (_, baseline) = _run(tmp_path / "a")
    assert baseline["cell"]["object_label"].tolist() == [1, 2, 3, 4]

    dropped_labels = {2, 4}

    def only_cells_1_and_3(context):
        if context.object_type != "cell":
            # Everything else is left alone; _exclude_objects propagates the
            # cell cull to the nuclei and pathogens inside them.
            return np.ones(len(context.labels), dtype=bool)
        return np.array([int(label) not in dropped_labels
                         for label in context.labels])

    mh.register_region_filter_hook(only_cells_1_and_3, name="roi")
    (_, filtered) = _run(tmp_path / "b")

    assert filtered["cell"]["object_label"].tolist() == [1, 3]
    # The cascade: a nucleus/pathogen whose host cell was excluded goes too.
    assert filtered["nucleus"]["object_label"].tolist() == [1, 3]
    assert filtered["pathogen"]["object_label"].tolist() == [1, 3]

    # The surviving objects are measured exactly as they were: the filter
    # suppressed objects, it did not perturb the measurement of the rest.
    kept = baseline["cell"][baseline["cell"]["object_label"].isin([1, 3])]
    pd.testing.assert_frame_equal(
        kept.reset_index(drop=True),
        filtered["cell"].reset_index(drop=True))


def test_verbose_reports_what_the_region_filter_dropped(tmp_path, capsys):
    """verbose=True names the field, the count and the object type."""
    mh.register_region_filter_hook(
        lambda context: (context.labels < 3
                         if context.object_type == "cell"
                         else np.ones(len(context.labels), dtype=bool)),
        name="roi")
    _run(tmp_path, verbose=True)

    printed = capsys.readouterr().out
    assert "plate1_A01_F001: region filter dropped 2 of 4 cell object(s)." \
        in printed


def test_region_filter_dropping_everything_writes_no_rows(tmp_path):
    """An ROI that contains nothing produces an empty field, not wrong rows."""
    mh.register_region_filter_hook(
        lambda context: np.zeros(len(context.labels), dtype=bool),
        name="empty-roi")
    (result, tables) = _run(tmp_path)

    # Not the int-0 failure sentinel: this field succeeded, it just held
    # nothing the user asked for.
    assert isinstance(result[2], np.ndarray)
    for frame in tables.values():
        assert frame.empty


def test_region_filter_receives_labels_and_centroids(tmp_path):
    """The context carries the label ids and their centroids, in step."""
    seen = {}

    def record(context):
        if context.object_type == "cell":
            seen["labels"] = context.labels.tolist()
            seen["centroids"] = np.round(context.centroids).astype(int).tolist()
            seen["ndim"] = context.ndim
        seen.setdefault("types", []).append(context.object_type)
        return np.ones(len(context.labels), dtype=bool)

    mh.register_region_filter_hook(record, name="record")
    _run(tmp_path)

    assert seen["labels"] == [1, 2, 3, 4]
    assert seen["ndim"] == 2
    # Centroids are in array index order (row, col), matching _flat_masks.
    assert seen["centroids"] == [[30, 30], [30, 90], [90, 30], [90, 90]]
    # Every object type that has objects is offered, so a filter can act on
    # whichever it wants. 'organelle' is absent because this field has none:
    # a mask with no labels short-circuits before any hook is called.
    assert seen["types"] == ["cell", "nucleus", "pathogen", "cytoplasm"]
    assert set(seen["types"]) < set(mh.OBJECT_TYPES)


def test_region_filter_is_applied_before_features_are_computed(tmp_path,
                                                               monkeypatch):
    """Excluded objects never reach the feature extractors.

    The filter has to suppress objects cheaply -- by removing them from the
    masks -- rather than by subsetting a DataFrame after morphology,
    intensity, texture, radial distribution and Zernike have all been computed
    for objects the user excluded. Spying on the two extractors is the direct
    test of that.
    """
    import spacr.measure as measure

    real_morphology = measure._morphological_measurements
    real_intensity = measure._intensity_measurements
    seen = {}

    def morphology_spy(cell_mask, *args, **kwargs):
        seen["morphology"] = sorted(
            int(v) for v in np.unique(cell_mask) if v != 0)
        return real_morphology(cell_mask, *args, **kwargs)

    def intensity_spy(cell_mask, *args, **kwargs):
        seen["intensity"] = sorted(
            int(v) for v in np.unique(cell_mask) if v != 0)
        return real_intensity(cell_mask, *args, **kwargs)

    monkeypatch.setattr(measure, "_morphological_measurements", morphology_spy)
    monkeypatch.setattr(measure, "_intensity_measurements", intensity_spy)

    mh.register_region_filter_hook(
        lambda context: (np.ones(len(context.labels), dtype=bool)
                         if context.object_type != "cell"
                         else context.labels == 1),
        name="one-cell")
    _run(tmp_path)

    assert seen["morphology"] == [1]
    assert seen["intensity"] == [1]


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_preprocessing_hooks_chain_in_priority_then_registration_order():
    """Lower priority first; ties keep the order they were registered in."""
    order = []

    def make(tag, factor):
        def hook(array, context):
            order.append(tag)
            return array * factor
        hook.__qualname__ = f"hook_{tag}"
        return hook

    mh.register_preprocessing_hook(make("late", 3), priority=10)
    mh.register_preprocessing_hook(make("early_a", 2), priority=-1)
    mh.register_preprocessing_hook(make("early_b", 5), priority=-1)

    array = np.ones((2, 2, 1), dtype=np.uint16)
    context = mh.PreprocessingContext(file_name="f", channels=[0], settings={})
    result = mh.apply_preprocessing_hooks(array, context)

    assert order == ["early_a", "early_b", "late"]
    # 1 * 2 * 5 * 3, applied in that order.
    assert result.tolist() == [[[30], [30]], [[30], [30]]]
    assert [entry.priority for entry in mh.preprocessing_hooks()] == [-1, -1, 10]


def test_region_filters_intersect_and_do_not_depend_on_order():
    """An object survives only if every filter kept it, whatever the order."""
    mask = np.zeros((4, 12), dtype=np.uint16)
    for label in (1, 2, 3):
        mask[1:3, (label - 1) * 4 + 1:(label - 1) * 4 + 3] = label

    def drop_three(context):
        return context.labels != 3

    def drop_two(context):
        return context.labels != 2

    mh.register_region_filter_hook(drop_three, name="a", priority=5)
    mh.register_region_filter_hook(drop_two, name="b", priority=-5)
    kept, dropped = mh.apply_region_filter_hooks(
        mask, object_type="cell", file_name="f", settings={})
    assert dropped == (2, 3)
    assert sorted(int(v) for v in np.unique(kept)) == [0, 1]

    # Same two filters, opposite priorities: identical outcome.
    mh.clear_measurement_hooks()
    mh.register_region_filter_hook(drop_three, name="a", priority=-5)
    mh.register_region_filter_hook(drop_two, name="b", priority=5)
    kept2, dropped2 = mh.apply_region_filter_hooks(
        mask, object_type="cell", file_name="f", settings={})
    assert dropped2 == dropped
    np.testing.assert_array_equal(kept, kept2)


def test_reregistering_the_same_hook_replaces_it():
    """Re-installing an extension cannot double-apply its correction."""
    def gain(array, context):
        return array * 2

    first = mh.register_preprocessing_hook(gain)
    second = mh.register_preprocessing_hook(gain)
    assert first == second
    assert len(mh.preprocessing_hooks()) == 1

    array = np.ones((1, 1, 1), dtype=np.uint16)
    context = mh.PreprocessingContext(file_name="f", channels=[0], settings={})
    assert mh.apply_preprocessing_hooks(array, context).tolist() == [[[2]]]


def test_two_distinct_hooks_sharing_a_qualname_both_register():
    """Different callables are never silently collapsed into one."""
    def make():
        def hook(array, context):
            return array
        return hook

    first = mh.register_preprocessing_hook(make())
    second = mh.register_preprocessing_hook(make())
    assert first != second
    assert second.endswith("#2")
    assert len(mh.preprocessing_hooks()) == 2

    third = mh.register_preprocessing_hook(make())
    assert third.endswith("#3")


def test_an_explicit_name_replaces_whatever_holds_it():
    """The name is the identity, so a GUI can re-install under a fixed key."""
    mh.register_preprocessing_hook(lambda a, c: a * 2, name="roi-gain")
    mh.register_preprocessing_hook(lambda a, c: a * 3, name="roi-gain")
    assert len(mh.preprocessing_hooks()) == 1

    array = np.ones((1, 1, 1), dtype=np.uint16)
    context = mh.PreprocessingContext(file_name="f", channels=[0], settings={})
    assert mh.apply_preprocessing_hooks(array, context).tolist() == [[[3]]]


def test_describe_hooks_lists_both_kinds():
    mh.register_preprocessing_hook(lambda a, c: a, name="gain", priority=2)
    mh.register_region_filter_hook(lambda c: np.ones(len(c.labels), bool),
                                   name="roi")
    described = mh.describe_hooks()
    assert "preprocessing: gain (priority=2, source=api)" in described
    assert "region filter: roi (priority=0, source=api)" in described


# ---------------------------------------------------------------------------
# A misbehaving hook surfaces
# ---------------------------------------------------------------------------

def test_a_raising_preprocessing_hook_raises_and_keeps_the_cause():
    def broken(array, context):
        raise ZeroDivisionError("no flat field for this plate")

    mh.register_preprocessing_hook(broken, name="broken-gain")
    context = mh.PreprocessingContext(
        file_name="plate1_A01_F001", channels=[0], settings={})

    with pytest.raises(mh.MeasurementHookError) as caught:
        mh.apply_preprocessing_hooks(np.ones((1, 1, 1), np.uint16), context)

    message = str(caught.value)
    assert "broken-gain" in message
    assert "no flat field for this plate" in message
    assert "plate1_A01_F001" in message
    assert "unregister_preprocessing_hook" in message
    assert isinstance(caught.value.__cause__, ZeroDivisionError)


def test_a_raising_region_filter_raises_and_keeps_the_cause():
    def broken(context):
        raise RuntimeError("the polygon has no vertices")

    mh.register_region_filter_hook(broken, name="broken-roi")
    mask = np.zeros((4, 4), dtype=np.uint16)
    mask[1:3, 1:3] = 1

    with pytest.raises(mh.MeasurementHookError) as caught:
        mh.apply_region_filter_hooks(mask, object_type="cell",
                                     file_name="f", settings={})
    assert "broken-roi" in str(caught.value)
    assert "unregister_region_filter_hook" in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize("hook,fragment", [
    (lambda array, context: None, "returned None"),
    (lambda array, context: array[..., :1], "may transform values, not geometry"),
    (lambda array, context: array.astype(np.float32), "cast the result yourself"),
])
def test_a_preprocessing_hook_that_returns_junk_is_refused(hook, fragment):
    """Shape and dtype are checked, never silently coerced."""
    mh.register_preprocessing_hook(hook, name="junk")
    context = mh.PreprocessingContext(file_name="f", channels=[0, 1],
                                      settings={})
    with pytest.raises(mh.MeasurementHookError) as caught:
        mh.apply_preprocessing_hooks(np.ones((2, 2, 2), np.uint16), context)
    assert fragment in str(caught.value)
    assert "junk" in str(caught.value)


@pytest.mark.parametrize("hook,fragment", [
    (lambda context: None, "returned None"),
    (lambda context: np.ones(99, dtype=bool), "aligned with context.labels"),
    (lambda context: np.array([7, 9]), "not a label list or a score"),
])
def test_a_region_filter_that_returns_junk_is_refused(hook, fragment):
    mh.register_region_filter_hook(hook, name="junk")
    mask = np.zeros((4, 12), dtype=np.uint16)
    mask[1:3, 1:3] = 1
    mask[1:3, 5:7] = 2
    with pytest.raises(mh.MeasurementHookError) as caught:
        mh.apply_region_filter_hooks(mask, object_type="cell",
                                     file_name="f", settings={})
    assert fragment in str(caught.value)
    assert "junk" in str(caught.value)


def test_a_region_filter_may_return_zeros_and_ones():
    """0/1 integers are an unambiguous keep-mask and are accepted."""
    mh.register_region_filter_hook(lambda context: np.array([1, 0]),
                                   name="ints")
    mask = np.zeros((4, 12), dtype=np.uint16)
    mask[1:3, 1:3] = 1
    mask[1:3, 5:7] = 2
    kept, dropped = mh.apply_region_filter_hooks(
        mask, object_type="cell", file_name="f", settings={})
    assert dropped == (2,)
    assert sorted(int(v) for v in np.unique(kept)) == [0, 1]


def test_a_broken_hook_fails_the_field_instead_of_measuring_it(tmp_path,
                                                              capsys):
    """Through _measure_crop_core: the field is reported failed, not written.

    ``cells`` is the int 0 rather than an ndarray -- the cross-process failure
    sentinel ``measure_crop``'s job_callback files on the RunLedger -- and no
    measurements.db exists, so nothing downstream can mistake this for a
    completed field.
    """
    def broken(array, context):
        raise ValueError("flat-field model missing for plate1")

    mh.register_preprocessing_hook(broken, name="broken-gain")
    (result, tables) = _run(tmp_path)

    index, _average, cells, _figs = result
    assert index == 0
    assert isinstance(cells, int) and cells == 0
    assert tables == {}

    printed = capsys.readouterr()
    combined = printed.out + printed.err
    assert "broken-gain" in combined
    assert "flat-field model missing for plate1" in combined
    assert "MeasurementHookError" in combined


def test_a_broken_region_filter_fails_the_field(tmp_path, capsys):
    mh.register_region_filter_hook(
        lambda context: (_ for _ in ()).throw(KeyError("no such shape layer")),
        name="broken-roi")
    (result, tables) = _run(tmp_path)

    assert result[2] == 0
    assert tables == {}
    combined = "".join(capsys.readouterr())
    assert "broken-roi" in combined


def test_an_env_installer_that_registers_junk_reports_its_own_error(monkeypatch):
    """A MeasurementHookError from inside an installer is not re-wrapped."""
    def install():
        mh.register_preprocessing_hook("not a function")

    monkeypatch.setitem(sys.modules, "spacr_fake_hooks_5",
                        _installer_module("spacr_fake_hooks_5", install))
    monkeypatch.setenv(mh.HOOKS_ENV_VAR, "spacr_fake_hooks_5:install")
    with pytest.raises(mh.MeasurementHookError) as caught:
        mh.preprocessing_hooks()
    assert "must be callable" in str(caught.value)
    assert "installer" not in str(caught.value)


def test_unregister_region_filter_hook_round_trip():
    name = mh.register_region_filter_hook(
        lambda context: np.ones(len(context.labels), bool), name="roi")
    assert mh.region_filter_hooks()
    assert mh.unregister_region_filter_hook(name) is True
    assert mh.region_filter_hooks() == ()
    assert mh.unregister_region_filter_hook(name) is False


def test_a_non_dict_mapping_is_passed_through_unwrapped():
    """A settings object that is already immutable is not double-wrapped."""
    from types import MappingProxyType

    frozen = MappingProxyType({"src": "/x"})
    context = mh.PreprocessingContext(file_name="f", channels=[0],
                                      settings=frozen)
    assert context.settings is frozen


def test_registration_refuses_a_non_callable_and_a_bad_priority():
    with pytest.raises(mh.MeasurementHookError) as caught:
        mh.register_preprocessing_hook("not a function")
    assert "must be callable" in str(caught.value)

    with pytest.raises(mh.MeasurementHookError) as caught:
        mh.register_region_filter_hook(lambda context: None, priority="soon")
    assert "must be an int" in str(caught.value)

    assert mh.preprocessing_hooks() == ()
    assert mh.region_filter_hooks() == ()


# ---------------------------------------------------------------------------
# Reaching worker processes
# ---------------------------------------------------------------------------

def _installer_module(name, install):
    module = types.ModuleType(name)
    module.install = install
    return module


def test_env_var_installs_hooks_in_this_process(monkeypatch):
    """SPACR_MEASURE_HOOKS is the route that survives a spawned worker."""
    def install():
        mh.register_preprocessing_hook(lambda a, c: a * 2, name="env-gain")

    monkeypatch.setitem(sys.modules, "spacr_fake_hooks",
                        _installer_module("spacr_fake_hooks", install))
    monkeypatch.setenv(mh.HOOKS_ENV_VAR, " spacr_fake_hooks:install , ")

    hooks = mh.preprocessing_hooks()
    assert [entry.name for entry in hooks] == ["env-gain"]
    # Tagged 'env', which is what stops the spawn warning firing for it.
    assert hooks[0].source == "env"


@pytest.mark.parametrize("value,fragment", [
    ("no_colon_here", 'not of the form "module:attribute"'),
    ("spacr_missing_module_xyz:install", "could not be resolved"),
])
def test_a_bad_env_entry_is_refused_loudly(monkeypatch, value, fragment):
    monkeypatch.setenv(mh.HOOKS_ENV_VAR, value)
    with pytest.raises(mh.MeasurementHookError) as caught:
        mh.preprocessing_hooks()
    assert fragment in str(caught.value)


def test_an_env_entry_that_is_not_callable_is_refused(monkeypatch):
    module = types.ModuleType("spacr_fake_hooks_2")
    module.install = 42
    monkeypatch.setitem(sys.modules, "spacr_fake_hooks_2", module)
    monkeypatch.setenv(mh.HOOKS_ENV_VAR, "spacr_fake_hooks_2:install")
    with pytest.raises(mh.MeasurementHookError) as caught:
        mh.region_filter_hooks()
    assert "not callable" in str(caught.value)


def test_an_env_installer_that_raises_is_refused(monkeypatch):
    def install():
        raise OSError("flat-field cache unreadable")

    monkeypatch.setitem(sys.modules, "spacr_fake_hooks_3",
                        _installer_module("spacr_fake_hooks_3", install))
    monkeypatch.setenv(mh.HOOKS_ENV_VAR, "spacr_fake_hooks_3:install")
    with pytest.raises(mh.MeasurementHookError) as caught:
        mh.preprocessing_hooks()
    assert "flat-field cache unreadable" in str(caught.value)


def test_the_env_var_is_read_once_per_process(monkeypatch):
    """A failing entry does not re-raise on all 384 wells."""
    monkeypatch.setenv(mh.HOOKS_ENV_VAR, "no_colon_here")
    with pytest.raises(mh.MeasurementHookError):
        mh.preprocessing_hooks()
    # Second call: already attempted, so it is a plain empty registry.
    assert mh.preprocessing_hooks() == ()


def test_spawn_warns_about_hooks_the_workers_will_never_see(capsys):
    """A silent no-op in every worker is the one failure mode worth shouting about."""
    assert mh.warn_if_hooks_will_not_reach_workers("spawn") is False

    mh.register_preprocessing_hook(lambda a, c: a, name="in-process-gain")
    # fork inherits the parent's registries, so there is nothing to say.
    assert mh.warn_if_hooks_will_not_reach_workers("fork") is False

    assert mh.warn_if_hooks_will_not_reach_workers("spawn") is True
    printed = capsys.readouterr().out
    assert "in-process-gain" in printed
    assert mh.HOOKS_ENV_VAR in printed


def test_env_installed_hooks_do_not_trigger_the_spawn_warning(monkeypatch):
    def install():
        mh.register_region_filter_hook(
            lambda context: np.ones(len(context.labels), bool), name="env-roi")

    monkeypatch.setitem(sys.modules, "spacr_fake_hooks_4",
                        _installer_module("spacr_fake_hooks_4", install))
    monkeypatch.setenv(mh.HOOKS_ENV_VAR, "spacr_fake_hooks_4:install")
    assert mh.region_filter_hooks()[0].source == "env"
    assert mh.warn_if_hooks_will_not_reach_workers("spawn") is False


def test_measure_crop_checks_the_start_method_before_it_starts_the_pool(
        tmp_path, monkeypatch):
    """The warning is wired into measure_crop, not just available to call."""
    import spacr.measure as measure

    calls = []
    monkeypatch.setattr(measure, "warn_if_hooks_will_not_reach_workers",
                        lambda start_method: calls.append(start_method) or False)

    project, merged, _name = _project(tmp_path)
    measure.measure_crop(_settings(merged, n_jobs=1))

    assert calls, "measure_crop did not consult the hook/start-method check"
    assert calls[0] in {"fork", "spawn", "forkserver"}
    assert os.path.isfile(
        os.path.join(str(project), "measurements", "measurements.db"))


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------

def test_region_context_labels_centroids_and_read_only_mask():
    mask = np.zeros((10, 20), dtype=np.uint16)
    mask[2:5, 2:5] = 4
    mask[6:9, 12:15] = 1
    context = mh.RegionContext(object_type="cell", file_name="f", mask=mask,
                               settings={}, spacing=None)

    assert context.labels.tolist() == [1, 4]
    np.testing.assert_allclose(context.centroids, [[7.0, 13.0], [3.0, 3.0]])
    # Cached: the second access is the same object.
    assert context.centroids is context.centroids
    assert context.ndim == 2
    with pytest.raises(ValueError):
        context.mask[0, 0] = 9
    # The caller's array is untouched by the read-only view.
    mask[0, 0] = 7
    assert mask[0, 0] == 7


def test_region_context_on_an_empty_mask():
    context = mh.RegionContext(object_type="cell", file_name="f",
                               mask=np.zeros((4, 4), np.uint16), settings={})
    assert context.labels.size == 0
    assert context.centroids.shape == (0, 2)

    mh.register_region_filter_hook(
        lambda c: np.zeros(len(c.labels), bool), name="roi")
    kept, dropped = mh.apply_region_filter_hooks(
        np.zeros((4, 4), np.uint16), object_type="cell", file_name="f",
        settings={})
    assert dropped == ()
    assert kept.shape == (4, 4)


def test_region_context_handles_a_3d_mask():
    mask = np.zeros((4, 8, 8), dtype=np.uint16)
    mask[1:3, 2:4, 2:4] = 1
    context = mh.RegionContext(object_type="cell", file_name="f", mask=mask,
                               settings={}, spacing=(2.0, 0.5, 0.5))
    assert context.ndim == 3
    assert context.centroids.shape == (1, 3)
    np.testing.assert_allclose(context.centroids[0], [1.5, 2.5, 2.5])
    assert context.spacing == (2.0, 0.5, 0.5)


def test_preprocessing_context_reports_spacing_and_volumetric():
    context = mh.PreprocessingContext(
        file_name="f", channels=np.array([2, 0]), settings={"src": "/x"},
        volumetric=True, spacing=(2.0, 0.5, 0.5))
    assert context.channels == (2, 0)
    assert context.volumetric is True
    assert context.spacing == (2.0, 0.5, 0.5)
    assert context.settings["src"] == "/x"
