"""Metadata choices and explicit display-only normalization mathematics."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from spacr.qt.widgets import qc_field_browser as browser


def _field(plate="p", well="A01", row="A", column=1, number=0):
    return SimpleNamespace(plate=plate, well=well, row=row, column=column,
                           identity=(plate, well, number), files=(("0", "image.tif"),))


def _reader(arrays, calls):
    def read(field, channels, *, edge):
        calls.append((field.identity, tuple(channels), edge))
        return tuple((channel, str(channel), arrays[field.identity][channel]) for channel in channels)
    return read


def test_per_image_normalization_keeps_raw_values_and_handles_constant_nonfinite():
    raw = np.array([[10., 20., np.nan], [30., np.inf, -np.inf]])
    constant = np.full((2, 3), 9, dtype=np.uint16)
    old = raw.copy()
    plan = browser._overview_normalization_plan((), (0, 1), {"percentiles": (0, 100)}, None)
    result = browser._overview_normalized_planes(((0, "a", raw), (1, "b", constant)), plan)
    np.testing.assert_array_equal(result[0][2], [[0., .5, 0.], [1., 0., 0.]])
    np.testing.assert_array_equal(result[1][2], np.zeros((2, 3)))
    np.testing.assert_array_equal(raw, old)
    assert constant.dtype == np.uint16 and np.all(constant == 9)
    assert all(not np.shares_memory(item[2], raw) for item in result)


@pytest.mark.parametrize("mode,reference", [
    ("reference_well", ("p", "A01")),
    ("reference_row", ("p", "A")),
    ("reference_column", ("p", 1)),
    ("reference_condition", ("p", "control")),
])
def test_reference_scope_uses_only_explicit_plate_qualified_or_condition_fields(mode, reference):
    selected = _field()
    other_plate = _field("other")
    other_well = _field(well="B02", row="B", column=2)
    fields = (selected, other_plate, other_well)
    arrays = {field.identity: {0: np.array([0., 10.]) if field is selected else np.array([100., 200.])}
              for field in fields}
    calls = []
    options = {"mode": mode, "reference": reference, "percentiles": (0, 100),
               "conditions": {("p", "A01"): "control", ("other", "A01"): "control"}}
    plan = browser._overview_normalization_plan(fields, (0,), options, _reader(arrays, calls))
    assert calls == [(selected.identity, (0,), 256)]
    assert plan["bounds"] == {0: (0., 10.)}
    assert plan["sample_pixels"] == {0: 2}
    assert plan["field_identities"] == (selected.identity,)


def test_shared_bounds_pool_original_finite_pixels_without_mutation():
    fields = (_field(), _field(number=1))
    arrays = {fields[0].identity: {0: np.array([0., 2., np.nan])},
              fields[1].identity: {0: np.array([4., 6., np.inf])}}
    before = {key: value[0].copy() for key, value in arrays.items()}
    plan = browser._overview_normalization_plan(fields, (0,),
        {"mode": "shared", "percentiles": (25, 75)}, _reader(arrays, []))
    assert plan["bounds"] == {0: (1.5, 4.5)}
    assert plan["sample_pixels"] == {0: 4}
    for key in arrays:
        np.testing.assert_array_equal(arrays[key][0], before[key])


@pytest.mark.parametrize("mean_match", [False, True])
def test_reference_channel_bounds_and_mean_matching_precede_same_affine_transform(mean_match):
    field = _field()
    arrays = {field.identity: {0: np.array([10., 20., 30.]), 1: np.array([20., 40., 60.])}}
    plan = browser._overview_normalization_plan((field,), (0, 1),
        {"mode": "shared", "percentiles": (0, 100), "reference_channel": 1,
         "mean_match": mean_match}, _reader(arrays, []))
    assert plan["bounds"] == {0: (20., 60.), 1: (20., 60.)}
    assert plan["gains"] == {0: 2. if mean_match else 1., 1: 1.}
    result = browser._overview_normalized_planes(
        tuple((channel, str(channel), raw) for channel, raw in arrays[field.identity].items()), plan)
    np.testing.assert_array_equal(result[0][2], [0., .5, 1.] if mean_match else [0., 0., .25])
    np.testing.assert_array_equal(result[1][2], [0., .5, 1.])


def test_mean_matching_includes_zero_samples_and_is_not_a_pixel_ratio():
    field = _field()
    arrays = {field.identity: {0: np.array([0., 0., 6.]), 1: np.array([0., 6., 6.])}}
    plan = browser._overview_normalization_plan((field,), (0, 1),
        {"mode": "shared", "percentiles": (0, 100), "reference_channel": 1,
         "mean_match": True}, _reader(arrays, []))
    assert plan["gains"] == {0: 2., 1: 1.}
    assert plan["sample_pixels"] == {0: 3, 1: 3}
    result = browser._overview_normalized_planes(
        tuple((channel, str(channel), raw) for channel, raw in arrays[field.identity].items()), plan)
    np.testing.assert_array_equal(result[0][2], [0., 0., 1.])
    assert np.isfinite(result[0][2]).all()


@pytest.mark.parametrize("values", [[0., 0.], [-1., 0.], [np.nan, np.inf]])
def test_undefined_reference_means_are_refused(values):
    field = _field()
    arrays = {field.identity: {0: np.array(values), 1: np.array([1., 2.])}}
    with pytest.raises(ValueError, match="finite"):
        browser._overview_normalization_plan((field,), (0, 1),
            {"mode": "shared", "reference_channel": 1, "mean_match": True}, _reader(arrays, []))


def test_missing_reference_is_refused_before_any_image_read():
    with pytest.raises(ValueError, match="reference has no fields"):
        browser._overview_normalization_plan((_field(),), (0,),
            {"mode": "reference_well", "reference": ("missing", "A01")}, None)


def test_normalization_samples_are_bounded_original_pixels_and_cancellable():
    fields = tuple(_field(number=number) for number in range(129))
    raw = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    arrays = {field.identity: {0: raw} for field in fields}
    calls = []
    plan = browser._overview_normalization_plan(fields, (0,),
        {"mode": "shared", "percentiles": (0, 100)}, _reader(arrays, calls))
    assert len(calls) == plan["sample_fields"] == 128
    assert plan["sample_pixels"] == {0: 128 * 2048}
    assert calls[0][0] == fields[0].identity and calls[-1][0] == fields[-1].identity
    assert plan["bounds"] == {0: (0., 65535.)}
    assert all(call[2] == 256 for call in calls)
    np.testing.assert_array_equal(raw, np.arange(65536, dtype=np.uint16).reshape(256, 256))
    cancelled = threading.Event()
    cancelled.set()
    from concurrent.futures import CancelledError
    with pytest.raises(CancelledError):
        browser._overview_normalization_plan(fields, (0,), {"mode": "shared"}, None, cancelled)


@pytest.mark.parametrize("key,name,expected", [
    ("cellvoyager", "plate1_E01_T0001F001L01A01Z01C02.tif",
     {"plateID": "plate1", "wellID": "E01", "fieldID": "001", "timeID": "0001",
      "sliceID": "01", "chanID": "02"}),
    ("canonical", "p_A01_F2_C4.tif", {"plateID": "p", "wellID": "A01", "fieldID": "2", "chanID": "4"}),
])
def test_metadata_preview_reuses_generate_masks_presets(key, name, expected):
    from spacr.qt.regex_detect import BUILTIN_REGEXES
    rows, missed = browser._overview_metadata_preview((name, "not-an-image.txt"), BUILTIN_REGEXES[key])
    assert missed == ("not-an-image.txt",)
    assert rows[0][0] == name
    assert {key: rows[0][1][key] for key in expected} == expected
    assert rows[1][1] == {}


def test_custom_regex_preview_uses_actual_filename_captures():
    pattern = r"(?P<plateID>P\d+)-(?P<wellID>[A-Z]\d+)-s(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif$"
    rows, missed = browser._overview_metadata_preview(("P7-B03-s2-c9.tif",), pattern)
    assert not missed
    assert rows == (("P7-B03-s2-c9.tif", {"plateID": "P7", "wellID": "B03", "fieldID": "2", "chanID": "9"}),)


def test_metadata_preview_matches_reader_case_insensitive_filename_policy():
    from spacr.qt.regex_detect import BUILTIN_REGEXES
    rows, missed = browser._overview_metadata_preview(("p_a01_f2_c4.TIF",), BUILTIN_REGEXES["canonical"])
    assert not missed
    assert rows[0][1] == {"plateID": "p", "wellID": "a01", "fieldID": "2", "chanID": "4"}


def test_metadata_wizard_reads_real_preset_channels_and_explicit_scope(qtbot, tmp_path):
    import tifffile
    from PIL import Image

    for channel in (1, 2, 3, 4):
        tifffile.imwrite(tmp_path / f"plate1_E01_T0001F001L01A01Z01C{channel:02}.tif",
                         np.full((12, 16), channel, dtype=np.uint16))
    derived = tmp_path / "data" / "cell_png"
    derived.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(derived / "derived.png")
    wizard = browser._OverviewMetadataWizard(str(tmp_path), {"channels": [0, 1, 2, 3, 4, 7]})
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    config = wizard.configuration()
    assert wizard.preset.currentData() == "cellvoyager"
    assert config["source_scope"] == "folder"
    assert config["channels"] == (1, 2, 3, 4)
    assert config["geometry"] == (6, 8)
    assert config["time_index"] == config["z_index"] == 0
    assert wizard._info["channels"] == (1, 2, 3, 4)
    assert {entry["axes"] for entry in wizard._info["files"]} == {"YX"}
    assert {entry["dtype"] for entry in wizard._info["files"]} == {"uint16"}
    assert wizard.preview.rowCount() == 4
    from PySide6.QtWidgets import QLabel
    from spacr.qt.widgets.sortable_table import SortableTableItem
    assert not wizard.channels.toolTip()
    assert any(label.toolTip() for label in wizard.findChildren(QLabel))
    assert all(isinstance(wizard.preview.item(row, column), SortableTableItem)
               for row in range(wizard.preview.rowCount()) for column in range(7))
    assert all(isinstance(wizard.conditions.item(row, column), SortableTableItem)
               for row in range(wizard.conditions.rowCount()) for column in range(3))
    assert len(wizard._fields) == 1
    assert {field.time_id for field in wizard._fields} == {"0001"}
    assert {field.z_id for field in wizard._fields} == {"01"}
    assert "derived.png" not in {wizard.preview.item(row, 0).text()
                                  for row in range(wizard.preview.rowCount())}
    wizard.channels.setText("7")
    with pytest.raises(ValueError, match="channel IDs"):
        wizard.configuration()
    wizard.channels.setText("1, 4")
    wizard.geometry.setCurrentIndex(5)
    wizard.representatives.setValue(2)
    config = wizard.configuration()
    assert config["channels"] == (1, 4)
    assert config["geometry"] == (8, 12)
    assert config["representatives"] == 2
    wizard.conditions.item(0, 2).setText("vehicle")
    wizard.mode.setCurrentIndex(wizard.mode.findData("reference_condition"))
    config = wizard.configuration()
    assert config["normalization"]["conditions"] == {("plate1", "E01"): "vehicle"}
    assert config["normalization"]["reference"] == ("plate1", "vehicle")
    wizard.reject()
    assert all(np.all(tifffile.imread(path) == int(path.stem[-2:]))
               for path in tmp_path.glob("*.tif"))


def test_metadata_wizard_custom_edit_invalidates_preview_until_rescanned(qtbot, tmp_path):
    import tifffile
    from PySide6.QtCore import Qt

    name = "P7-B03-s2-c9.tif"
    tifffile.imwrite(tmp_path / name, np.arange(48, dtype=np.uint16).reshape(6, 8))
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    pattern = r"(?P<plateID>P\d+)-(?P<wellID>[A-Z]\d+)-s(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif$"
    wizard.regex.setFocus()
    qtbot.keyClick(wizard.regex, Qt.Key_A, Qt.ControlModifier)
    qtbot.keyClicks(wizard.regex, pattern)
    assert wizard.preset.currentData() == "custom"
    assert not wizard.next.isEnabled()
    with pytest.raises(ValueError, match="Preview"):
        wizard.configuration()
    wizard._scan()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    assert wizard.configuration()["channels"] == (9,)
    assert wizard.configuration()["custom_regex"] == pattern
    assert wizard.preview.item(0, 2).text() == "B03"
    assert wizard.preview.item(0, 3).text() == "2"
    assert wizard.preview.item(0, 6).text() == "9"
    wizard.reject()


def test_metadata_wizard_immediate_cancel_never_starts_work_or_accumulates_forms(qtbot, monkeypatch):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid

    owner = QWidget()
    qtbot.addWidget(owner)
    existing_browser = object()
    owner._field_overview = existing_browser
    submissions = []
    monkeypatch.setattr(browser.JobRunner, "submit", lambda *args, **kwargs: submissions.append(args))
    previous = None
    for _ in range(3):
        wizard = browser._open_field_overview(owner, {"source": "/not-read-after-cancellation"})
        assert wizard is owner._field_overview_wizard
        wizard.reject()
        assert wizard._cancelled.is_set()
        assert not wizard._startup_timer.isActive()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QCoreApplication.processEvents()
        assert submissions == []
        assert owner._field_overview is existing_browser
        if previous is not None:
            assert not isValid(previous)
        assert len(owner.findChildren(browser._OverviewMetadataWizard)) == 1
        previous = wizard


def test_overview_known_constraints_use_configuration_error_without_reclassifying_faults():
    from spacr.errors import ConfigurationError

    with pytest.raises(browser._OverviewValidationError) as refusal:
        browser._overview_normalization_plan((_field(),), (0,),
            {"mode": "reference_well", "reference": ("missing", "A01")}, None)
    assert isinstance(refusal.value, ConfigurationError)
    assert isinstance(refusal.value, ValueError)
    with pytest.raises(browser._OverviewValidationError):
        browser._overview_metadata_preview(("name.tif",), "(")
    with pytest.raises(KeyError):
        browser._overview_normalized_planes(((0, "a", np.ones((2, 2))),), {})


@pytest.mark.parametrize("mode,reference", [
    ("reference_well", ("p", "")),
    ("reference_row", ("p", "")),
    ("reference_column", ("p", 0)),
    ("reference_condition", ("p", "control")),
])
def test_unknown_physical_identities_cannot_form_normalization_reference_groups(mode, reference):
    field = _field(well="", row="", column=0)
    assert browser._overview_reference_key(field, mode, {("p", ""): "control"}) is None
    with pytest.raises(browser._OverviewValidationError, match="reference has no fields"):
        browser._overview_normalization_plan((field,), (0,),
            {"mode": mode, "reference": reference, "conditions": {("p", ""): "control"}}, None)
    assert (field.plate, field.well, field.row, field.column) == ("p", "", "", 0)


def test_metadata_wizard_geometry_fits_real_row_l_and_preserves_explicit_rejection(qtbot, tmp_path):
    import tifffile
    from PySide6.QtCore import Qt

    tifffile.imwrite(tmp_path / "plate1_L02_T0001F001L01A01Z01C01.tif",
                     np.ones((6, 8), dtype=np.uint16))
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    assert browser._overview_geometry_extent(wizard._fields) == (12, 2)
    assert wizard.configuration()["geometry"] == (16, 24)
    wizard.geometry.setFocus()
    qtbot.keyClick(wizard.geometry, Qt.Key_Home)
    for _ in range(5):
        qtbot.keyClick(wizard.geometry, Qt.Key_Down)
    assert wizard.geometry.currentData() == (8, 12)
    assert wizard._geometry_explicit
    with pytest.raises(browser._OverviewValidationError, match="12 rows, 2 columns"):
        wizard.configuration()
    wizard._scan()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    assert wizard.geometry.currentData() == (8, 12)
    with pytest.raises(browser._OverviewValidationError, match="indexed wells"):
        wizard.configuration()
    assert wizard._fields[0].well == "L02"
    wizard.reject()


def test_overview_source_entry_reuses_only_the_current_normalized_first_source(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid

    owner = QWidget()
    qtbot.addWidget(owner)
    existing_browser = object()
    owner._field_overview = existing_browser
    submissions = []
    monkeypatch.setattr(browser.JobRunner, "submit", lambda *args, **kwargs: submissions.append(args))
    first_source, second_source = tmp_path / "first", tmp_path / "second"
    first = browser._open_field_overview(owner, {"src": [str(first_source), str(second_source)]})
    same = browser._open_field_overview(owner, {"src": str(first_source) + "/./"})
    assert same is first
    nested = browser._open_field_overview(owner,
        {"src": [[str(first_source), str(second_source)], ["unused"]]})
    assert nested is first
    assert nested.source.text() == str(first_source)
    assert same.source.text() == str(first_source)
    replacement = browser._open_field_overview(owner, {"src": [str(second_source), str(first_source)]})
    assert replacement is not first
    assert replacement.source.text() == str(second_source)
    assert first._cancelled.is_set()
    replacement.reject()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QCoreApplication.processEvents()
    assert not isValid(first)
    assert owner._field_overview is existing_browser
    assert submissions == []
    assert not first_source.exists() and not second_source.exists()


def _well_mean_options(**changes):
    return {"mode": "reference_average", "reference_kind": "reference_column",
            "reference": ("p", 1), "percentiles": (0, 100),
            "selection": {"representatives": 2, "seed": 3, "generation": 7}, **changes}


def test_well_mean_equal_field_then_equal_well_statistics_are_lazy_and_raw_immutable():
    import json

    fields = (_field(), _field(number=1), _field(well="B01", row="B"),
              _field(well="C02", row="C", column=2))
    arrays = {fields[0].identity: {0: np.array([0., 4.])},
              fields[1].identity: {0: np.full(100, 6.)},
              fields[2].identity: {0: np.full(3, 10.)},
              fields[3].identity: {0: np.array([0., 4.])}}
    before = {key: value[0].copy() for key, value in arrays.items()}
    calls = []
    plan, context = browser._overview_prepare_normalization(fields, (0,), _well_mean_options(),
                                                            _reader(arrays, calls))
    assert plan["reference_means"] == {0: 7.}
    assert len(calls) == 3
    assert {row["bounds_sample_pixels"][0] for row in plan["reference_wells"]} == {3}
    resolved = context.plan_for(fields[3])
    assert resolved["gains"] == {0: 3.5}
    assert len(calls) == 4
    assert context.plan_for(fields[3]) == resolved
    assert len(calls) == 4
    shown = browser._overview_normalized_planes(((0, "raw", arrays[fields[3].identity][0]),),
                                                resolved, fields[3])[0][2]
    low, high = resolved["bounds"][0]
    np.testing.assert_allclose(shown, np.clip((np.array([0., 4.]) * 3.5 - low) / (high - low), 0, 1))
    assert context.provenance()["used_wells"][0]["field_identities"] == (fields[3].identity,)
    json.dumps(context.provenance(), allow_nan=False)
    with pytest.raises(RuntimeError, match="resolved per-well"):
        browser._overview_normalized_planes(((0, "raw", arrays[fields[3].identity][0]),), plan, fields[3])
    for key in arrays:
        np.testing.assert_array_equal(arrays[key][0], before[key])


def test_well_mean_condition_reference_is_plate_qualified():
    fields = (_field(), _field("other"))
    arrays = {fields[0].identity: {0: np.array([2., 4.])},
              fields[1].identity: {0: np.array([100., 200.])}}
    calls = []
    plan, context = browser._overview_prepare_normalization(fields, (0,),
        _well_mean_options(reference_kind="reference_condition", reference=("p", "control"),
                           conditions={("p", "A01"): "control", ("other", "A01"): "control"}),
        _reader(arrays, calls))
    assert plan["reference_means"] == {0: 3.}
    assert calls == [(fields[0].identity, (0,), 256)]
    with pytest.raises(ValueError, match="same-plate"):
        context.plan_for(fields[1])


@pytest.mark.parametrize("values", [[0., 0.], [-1., 0.], [np.nan, np.inf]])
def test_well_mean_requires_positive_finite_means(values):
    field = _field()
    with pytest.raises(ValueError, match="finite"):
        browser._overview_prepare_normalization((field,), (0,), _well_mean_options(),
            _reader({field.identity: {0: np.array(values)}}, []))


@pytest.mark.parametrize("extra", [{"reference_channel": 0}, {"mean_match": True}])
def test_well_mean_refuses_combining_distinct_channel_scaling(extra):
    with pytest.raises(ValueError, match="cannot be combined"):
        browser._overview_prepare_normalization((_field(),), (0,), _well_mean_options(**extra), None)


def test_well_mean_lazy_reads_stop_on_owned_cancellation_before_next_field():
    from concurrent.futures import CancelledError

    fields = (_field(), _field(well="B02", row="B", column=2),
              _field(well="B02", row="B", column=2, number=1))
    cancelled = threading.Event()
    calls = []

    def read(field, channels, *, edge):
        calls.append(field.identity)
        if field.well == "B02":
            cancelled.set()
        return ((0, "raw", np.array([1., 2.])),)

    _, context = browser._overview_prepare_normalization(fields, (0,), _well_mean_options(), read, cancelled)
    with pytest.raises(CancelledError):
        context.plan_for(fields[1])
    assert calls == [fields[0].identity, fields[1].identity]


def test_well_mean_lazy_scalar_cache_is_bounded_and_selection_specific():
    fields = (_field(),) + tuple(_field(well=f"B{number:03}", row="B", column=number + 2)
                                for number in range(129))
    arrays = {field.identity: {0: np.array([1., 3.])} for field in fields}
    calls = []
    _, context = browser._overview_prepare_normalization(fields, (0,), _well_mean_options(),
                                                         _reader(arrays, calls))
    for field in fields[1:]:
        context.plan_for(field)
    assert len(context._cache) == 128
    assert len(calls) == 130
    context.plan_for(fields[1])
    assert len(calls) == 131
    updated = _field(number=5)
    arrays[updated.identity] = {0: np.array([4., 8.])}
    plan, other = browser._overview_prepare_normalization((updated,), (0,),
        _well_mean_options(selection={"representatives": 1, "generation": 8}), _reader(arrays, calls))
    assert plan["reference_means"] == {0: 6.}
    assert other.provenance()["reference_wells"][0]["field_identities"] == (updated.identity,)
    assert context.provenance()["reference_means"] == {0: 2.}


def test_well_mean_wizard_transports_plate_qualified_condition_and_refuses_channel_combination(qtbot, tmp_path):
    import tifffile

    for plate in ("p", "other"):
        tifffile.imwrite(tmp_path / f"{plate}_A01_F1_C0.tif", np.array([[1, 2]], dtype=np.uint16))
    wizard = browser._OverviewMetadataWizard(str(tmp_path), {"channels": [0]})
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    for row in range(wizard.conditions.rowCount()):
        wizard.conditions.item(row, 2).setText("control")
    wizard.mode.setCurrentIndex(wizard.mode.findData("reference_average"))
    assert wizard.reference_kind.isEnabled()
    wizard.reference_kind.setCurrentIndex(wizard.reference_kind.findData("reference_condition"))
    assert {wizard.reference.itemData(index) for index in range(wizard.reference.count())} == {
        ("p", "control"), ("other", "control")}
    selected = next(index for index in range(wizard.reference.count())
                    if wizard.reference.itemData(index) == ("p", "control"))
    wizard.reference.setCurrentIndex(selected)
    options = wizard.configuration()["normalization"]
    assert options["mode"] == "reference_average"
    assert options["reference_kind"] == "reference_condition"
    assert options["reference"] == ("p", "control")
    wizard.gain.setCurrentIndex(wizard.gain.findData(0))
    with pytest.raises(ValueError, match="cannot be combined"):
        wizard.configuration()
    wizard.reject()


def test_metadata_wizard_restores_explicit_geometry_and_reference_tuple(qtbot, tmp_path):
    import tifffile

    for plate in ("p", "other"):
        tifffile.imwrite(tmp_path / f"{plate}_A01_F1_C0.tif", np.array([[1, 2]], dtype=np.uint16))
    wizard = browser._OverviewMetadataWizard(str(tmp_path), {"channels": [0], "geometry": (16, 24)})
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    assert wizard.geometry.currentData() == (16, 24)
    wizard.mode.setCurrentIndex(wizard.mode.findData("reference_well"))
    assert wizard.reference.count() == 2
    wizard.reference.setCurrentIndex(1)
    selected = wizard.reference.currentData()
    wizard._reference_choices()
    assert wizard.reference.currentIndex() == 1
    assert wizard.reference.currentData() == selected
    config = wizard.configuration()
    assert config["geometry"] == (16, 24)
    assert config["normalization"]["reference"] == selected
    wizard.reject()


def test_concurrent_well_requests_share_reader_and_cancel_only_the_waiter(monkeypatch):
    import json
    from concurrent.futures import CancelledError, ThreadPoolExecutor, TimeoutError

    reference = _field()
    target = _field(well="B02", row="B", column=2)
    raw = np.array([0., 4.])
    owner_cancel = threading.Event()
    waiter_cancel = threading.Event()
    reading, release, waiting, timed_out = (threading.Event() for _ in range(4))
    calls = []

    def read(field, channels, *, edge):
        calls.append(field.identity)
        if field is target:
            reading.set()
            if not release.wait(10):
                raise RuntimeError("The controlled reader was not released.")
        return ((0, "raw", raw),)

    _, context = browser._overview_prepare_normalization((reference, target), (0,),
        _well_mean_options(), read, owner_cancel)
    executor = ThreadPoolExecutor(max_workers=3)
    try:
        owner = executor.submit(context.plan_for, target)
        assert reading.wait(5)
        pending = context._pending[(target.plate, target.well)]
        real_result = pending.result

        def observe_result(timeout=None):
            waiting.set()
            try:
                return real_result(timeout=timeout)
            except TimeoutError:
                timed_out.set()
                raise

        monkeypatch.setattr(pending, "result", observe_result)
        cancelled_waiter = executor.submit(context.plan_for, target, waiter_cancel)
        assert timed_out.wait(5)
        waiter_cancel.set()
        with pytest.raises(CancelledError):
            cancelled_waiter.result(timeout=5)
        assert not owner.done()
        waiting.clear()
        successful_waiter = executor.submit(context.plan_for, target)
        assert waiting.wait(5)
        assert calls == [reference.identity, target.identity]
        release.set()
        first = owner.result(timeout=5)
        second = successful_waiter.result(timeout=5)
        assert first == second
        assert first["gains"] == {0: 1.}
        assert context.plan_for(target) == first
        assert calls == [reference.identity, target.identity]
        provenance = context.provenance()
        assert len(provenance["used_wells"]) == 1
        assert provenance["used_wells"][0]["field_identities"] == (target.identity,)
        assert provenance["used_wells"][0]["gains"] == {0: 1.}
        json.dumps(provenance, allow_nan=False)
        owner_cancel.set()
        with pytest.raises(CancelledError):
            context.plan_for(target)
        assert context.provenance() == provenance
        assert calls == [reference.identity, target.identity]
        np.testing.assert_array_equal(raw, [0., 4.])
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_metadata_wizard_accepts_caller_percentiles_without_enabling_normalization(qtbot, tmp_path):
    import tifffile

    tifffile.imwrite(tmp_path / "p_A01_F1_C0.tif", np.array([[1, 2]], dtype=np.uint16))
    wizard = browser._OverviewMetadataWizard(str(tmp_path), {
        "channels": [0], "normalization_percentiles": (3., 97.), "normalize": False})
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    assert (wizard.low.value(), wizard.high.value()) == (3., 97.)
    accepted = []
    wizard.accepted.connect(lambda: accepted.append(wizard.configuration()))
    wizard.next.click()
    assert len(accepted) == 1
    assert accepted[0]["normalization"]["percentiles"] == (3., 97.)
    assert accepted[0]["normalize"] is False
    dialog = browser._FieldOverview(settings=accepted[0])
    qtbot.addWidget(dialog)
    assert not dialog.normalise.isChecked()
    assert (dialog.low.value(), dialog.high.value()) == (3., 97.)
    dialog.close()


def test_automatic_wizard_pattern_preserves_nested_folder_channel_groups(qtbot, tmp_path):
    import tifffile

    folder = tmp_path / "A01" / "field1"
    folder.mkdir(parents=True)
    paths = tuple(folder / f"ch{channel}.tif" for channel in (1, 2))
    for channel, path in enumerate(paths, 1):
        tifffile.imwrite(path, np.full((4, 5), channel, dtype=np.uint16))
    before = {path: path.read_bytes() for path in paths}
    indexed = browser._overview_index(str(tmp_path), source_scope="recursive")
    assert len(indexed) == 1
    assert indexed[0].well == "A01" and indexed[0].field_id == "1"
    wizard = browser._OverviewMetadataWizard(str(tmp_path), {"channels": (1, 2)})
    wizard.scope.setCurrentIndex(wizard.scope.findData("recursive"))
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    assert wizard._fields == indexed
    assert wizard._fields[0].files == (("1", str(paths[0])), ("2", str(paths[1])))
    assert wizard._info["channels"] == (1, 2)
    assert wizard.regex.text() == ""
    accepted = []
    wizard.accepted.connect(lambda: accepted.append(wizard.configuration()))
    wizard.next.click()
    assert len(accepted) == 1
    assert accepted[0]["channels"] == (1, 2)
    assert accepted[0]["custom_regex"] == ""
    assert accepted[0]["source_scope"] == "recursive"
    assert all(path.read_bytes() == before[path] for path in paths)


@pytest.fixture
def real_normalization_source(tmp_path):
    import tifffile

    paths = []
    for channel, offset in ((1, 0), (2, 10)):
        path = tmp_path / f"p_A01_F1_C{channel}.tif"
        tifffile.imwrite(path, np.array([[0, 10], [20, 30]], dtype=np.uint16) + offset)
        paths.append(path)
    originals = {path: path.read_bytes() for path in paths}
    fields = browser._overview_index(str(tmp_path), source_scope="folder")
    assert len(fields) == 1 and fields[0].well == "A01"
    yield tmp_path, fields
    assert all(path.read_bytes() == data for path, data in originals.items())


@pytest.mark.parametrize("channels,options,message", [
    ((1,), {"mode": "retired-display-mode"}, "supported normalization"),
    ((1,), {"percentiles": (97., 3.)}, "lower percentile"),
    ((1,), {"percentiles": (3.,)}, "lower percentile"),
    ((1,), {"percentiles": (float("nan"), 97.)}, "lower percentile"),
    ((), {}, "distinct available channel"),
    ((1, 1), {}, "distinct available channel"),
    ((1,), {"reference_channel": 2}, "reference channel must be selected"),
    ((1,), {"mean_match": True}, "Choose a reference channel"),
])
def test_saved_display_choices_refuse_invalid_configuration_without_changing_source(
        real_normalization_source, channels, options, message):
    _, fields = real_normalization_source
    with pytest.raises(browser._OverviewValidationError, match=message):
        browser._overview_prepare_normalization(fields, channels, options, browser._overview_raw_planes)


def test_reference_channel_image_bounds_and_serialized_well_reference_use_real_raw_planes(real_normalization_source):
    _, fields = real_normalization_source
    field = fields[0]
    source = tuple(browser._overview_raw_planes(field, (1, 2), edge=256))
    plan = browser._overview_normalization_plan(fields, (1, 2), {
        "mode": "image", "percentiles": (0., 100.), "reference_channel": 2}, browser._overview_raw_planes)
    shown = browser._overview_normalized_planes(source, plan, field)
    np.testing.assert_allclose(shown[0][2], [[0., 0.], [1 / 3, 2 / 3]])
    np.testing.assert_allclose(shown[1][2], [[0., 1 / 3], [2 / 3, 1.]])
    serialized_reference = browser._overview_normalization_plan(fields, (1,), {
        "mode": "reference_well", "reference": [field.plate, field.well],
        "percentiles": (0., 100.)}, browser._overview_raw_planes)
    assert serialized_reference["reference"] == (field.plate, field.well)
    assert serialized_reference["field_identities"] == (field.identity,)


def test_channel_changes_refuse_stale_reference_and_shared_bounds(real_normalization_source):
    _, fields = real_normalization_source
    field = fields[0]
    reference_plan = browser._overview_normalization_plan(fields, (1, 2), {
        "mode": "image", "reference_channel": 2}, browser._overview_raw_planes)
    first_channel = tuple(browser._overview_raw_planes(field, (1,), edge=256))
    with pytest.raises(browser._OverviewValidationError, match="reference channel must be selected"):
        browser._overview_normalized_planes(first_channel, reference_plan, field)
    shared_plan = browser._overview_normalization_plan(fields, (1,), {"mode": "shared"},
                                                       browser._overview_raw_planes)
    second_channel = tuple(browser._overview_raw_planes(field, (2,), edge=256))
    with pytest.raises(browser._OverviewValidationError, match="no learned normalization bounds"):
        browser._overview_normalized_planes(second_channel, shared_plan, field)


@pytest.mark.parametrize("mode", ["image", "shared"])
def test_nonfinite_tiff_channel_is_refused_in_display_and_reference_learning(tmp_path, mode):
    import tifffile

    path = tmp_path / "p_A01_F1_C1.tif"
    tifffile.imwrite(path, np.array([[np.nan, np.inf]], dtype=np.float32))
    original = path.read_bytes()
    fields = browser._overview_index(str(tmp_path), source_scope="folder")
    with pytest.raises(browser._OverviewValidationError, match="no finite pixels"):
        plan = browser._overview_normalization_plan(fields, (1,), {"mode": mode}, browser._overview_raw_planes)
        browser._overview_normalized_planes(tuple(browser._overview_raw_planes(fields[0], (1,), edge=256)),
                                            plan, fields[0])
    assert path.read_bytes() == original


@pytest.mark.parametrize("options,message", [
    ({"reference_kind": "reference_unknown"}, "supported normalization"),
    ({"reference": "p/A01"}, "reference has no fields"),
    ({"reference": ("absent", 1)}, "reference has no fields"),
    ({"reference": ("p", 9)}, "reference has no fields"),
])
def test_well_mean_saved_reference_must_resolve_to_actual_metadata(real_normalization_source, options, message):
    _, fields = real_normalization_source
    with pytest.raises(browser._OverviewValidationError, match=message):
        browser._overview_prepare_normalization(fields, (1,), _well_mean_options(**options),
                                                browser._overview_raw_planes)


@pytest.mark.parametrize("axis", ["time", "z"])
def test_well_mean_refuses_combining_distinct_real_filename_axes(tmp_path, axis):
    import tifffile

    pattern = (r"(?P<plateID>p)_(?P<wellID>A01)_F(?P<fieldID>1)_"
               r"T(?P<timeID>\d+)_Z(?P<sliceID>\d+)_C(?P<chanID>1)\.tif")
    for index in (1, 2):
        time, z = (index, 1) if axis == "time" else (1, index)
        tifffile.imwrite(tmp_path / f"p_A01_F1_T{time}_Z{z}_C1.tif", np.ones((2, 2), dtype=np.uint16))
    fields = browser._overview_index(str(tmp_path), pattern, source_scope="folder")
    assert len(fields) == 2
    with pytest.raises(browser._OverviewValidationError, match="Choose one filename time point and Z plane"):
        browser._overview_prepare_normalization(fields, (1,), _well_mean_options(), browser._overview_raw_planes)


def test_unresolved_real_source_can_be_viewed_but_not_used_as_physical_well_reference(tmp_path):
    import tifffile

    path = tmp_path / "unidentified.tif"
    tifffile.imwrite(path, np.array([[1, 2]], dtype=np.uint16))
    fields = browser._overview_index(str(tmp_path), source_scope="folder")
    assert len(fields) == 1 and fields[0].well == ""
    info = browser._overview_channel_info(fields[0])
    assert info["channels"] == (0,)
    image = browser._overview_pixels(fields[0], (0,), (0., 100.))
    assert image.size
    with pytest.raises(browser._OverviewValidationError, match="resolved physical well identities"):
        browser._overview_prepare_normalization(fields, (0,),
            _well_mean_options(reference=(fields[0].plate, 1)), browser._overview_raw_planes)


def test_empty_real_selection_cannot_learn_shared_bounds(tmp_path):
    fields = browser._overview_index(str(tmp_path), source_scope="folder")
    assert fields == ()
    with pytest.raises(browser._OverviewValidationError, match="No fields are available"):
        browser._overview_prepare_normalization(fields, (0,), {"mode": "shared"}, browser._overview_raw_planes)


def test_wizard_refuses_invalid_display_edits_then_accepts_repaired_choices(qtbot, real_normalization_source):
    source, _ = real_normalization_source
    wizard = browser._OverviewMetadataWizard(str(source), {"channels": (1, 2)})
    qtbot.addWidget(wizard)
    wizard.show()
    qtbot.waitUntil(wizard.next.isEnabled, timeout=10000)
    accepted = []
    wizard.accepted.connect(lambda: accepted.append(wizard.configuration()))
    wizard.channels.setText("red")
    wizard.next.click()
    assert "channel IDs" in wizard.status.text() and not accepted
    wizard.channels.setText("1, 1")
    wizard.next.click()
    assert "channel IDs" in wizard.status.text() and not accepted
    wizard.channels.setText("1")
    wizard.low.setValue(97.)
    wizard.high.setValue(3.)
    wizard.next.click()
    assert "lower percentile" in wizard.status.text() and not accepted
    wizard.low.setValue(3.)
    wizard.high.setValue(97.)
    wizard.gain.setCurrentIndex(wizard.gain.findData(2))
    wizard.next.click()
    assert "reference channel must be selected" in wizard.status.text() and not accepted
    wizard.gain.setCurrentIndex(0)
    wizard.mean_match.setChecked(True)
    wizard.next.click()
    assert "Choose a reference channel" in wizard.status.text() and not accepted
    wizard.mean_match.setChecked(False)
    wizard.mode.setCurrentIndex(wizard.mode.findData("reference_condition"))
    wizard.next.click()
    assert "reference has no fields" in wizard.status.text() and not accepted
    wizard.conditions.item(0, 2).setText("control")
    wizard.channels.setText("1, 2")
    wizard.next.click()
    assert len(accepted) == 1
    assert accepted[0]["channels"] == (1, 2)
    assert accepted[0]["normalization"]["percentiles"] == (3., 97.)
    assert accepted[0]["normalization"]["reference"] == ("p", "control")


def test_wizard_discards_stale_metadata_and_refuses_empty_missing_and_invalid_sources(
        qtbot, monkeypatch, real_normalization_source):
    import re

    source, _ = real_normalization_source
    empty = source / "empty"
    empty.mkdir()
    entered, release = threading.Event(), threading.Event()
    actual_probe = browser._overview_wizard_probe

    def gated_probe(path, *args, **kwargs):
        result = actual_probe(path, *args, **kwargs)
        if path == str(source):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("The real metadata probe was not released.")
        return result

    monkeypatch.setattr(browser, "_overview_wizard_probe", gated_probe)
    wizard = browser._OverviewMetadataWizard(str(source), {"channels": (1, 2)})
    qtbot.addWidget(wizard)
    wizard.show()
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        wizard.source.selectAll()
        qtbot.keyClicks(wizard.source, str(empty))
        assert not wizard.next.isEnabled()
        release.set()
        qtbot.waitUntil(lambda: not wizard._runner.is_busy(), timeout=10000)
        assert wizard._fields == () and wizard._info is None
        with pytest.raises(browser._OverviewValidationError, match="Preview the current source"):
            wizard.configuration()
        wizard._scan()
        qtbot.waitUntil(lambda: not wizard._runner.is_busy(), timeout=10000)
        assert "No readable image fields" in wizard.status.text()
        assert not wizard.next.isEnabled()
        wizard.source.setText(str(empty / "missing"))
        wizard._scan()
        qtbot.waitUntil(lambda: not wizard._runner.is_busy(), timeout=10000)
        assert "existing image folder" in wizard.status.text()
        assert not wizard.next.isEnabled()
        wizard.source.setText(str(source))
        wizard.regex.setText("(")
        try:
            re.compile("(")
        except re.error as error:
            diagnostic = str(error)
        wizard._scan()
        qtbot.waitUntil(lambda: not wizard._runner.is_busy(), timeout=10000)
        error_type = browser._OverviewValidationError
        expected = f"{error_type.__module__}.{error_type.__qualname__}: {diagnostic}"
        assert wizard.status.text() == expected
        assert not wizard.next.isEnabled()
    finally:
        release.set()
        wizard.reject()


def test_finite_reference_does_not_hide_nonfinite_target_field(tmp_path):
    import tifffile

    tifffile.imwrite(tmp_path / "p_A01_F1_C1.tif", np.array([[1., 2.]], dtype=np.float32))
    target_path = tmp_path / "p_B02_F1_C1.tif"
    tifffile.imwrite(target_path, np.array([[np.nan, np.inf]], dtype=np.float32))
    original = target_path.read_bytes()
    fields = browser._overview_index(str(tmp_path), source_scope="folder")
    plan = browser._overview_normalization_plan(fields, (1,), {
        "mode": "reference_well", "reference": ("p", "A01"),
        "percentiles": (0., 100.)}, browser._overview_raw_planes)
    assert plan["bounds"] == {1: (1., 2.)}
    target = next(field for field in fields if field.well == "B02")
    with pytest.raises(browser._OverviewValidationError, match="no finite pixels"):
        browser._overview_normalized_planes(tuple(browser._overview_raw_planes(target, (1,), edge=256)),
                                            plan, target)
    assert target_path.read_bytes() == original


@pytest.mark.parametrize("cancel_export", [False, True])
def test_actual_export_button_forks_well_mean_generation_and_preserves_raw_data(
        qtbot, tmp_path, monkeypatch, cancel_export):
    import json
    import tifffile
    from PySide6.QtWidgets import QFileDialog, QPushButton
    from spacr import tabular

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    base = np.array([[0, 2], [2, 4]], dtype=np.uint16)
    originals = {}
    for well, field_id, offset in (("A01", 1, 0), ("A01", 2, 4), ("B01", 1, 8), ("C02", 1, 0)):
        for channel in (1, 2):
            path = source / f"p1_{well}_F{field_id}_C{channel}.tif"
            tifffile.imwrite(path, (base + offset) * channel)
            originals[path] = path.read_bytes()
    conditions = {("p1", "A01"): "reference A", ("p1", "B01"): "reference B",
                  ("p1", "C02"): "treated"}
    dialog = browser._FieldOverview(settings={
        "source": str(source), "source_scope": "folder", "channels": (1, 2),
        "geometry": (4, 6), "representatives": 2, "conditions": conditions,
        "normalization": {"mode": "reference_average", "reference_kind": "reference_column",
                          "reference": ("p1", 1), "percentiles": (0., 100.)}})
    qtbot.addWidget(dialog)
    entered, release = threading.Event(), threading.Event()
    try:
        dialog.show()
        dialog._scan()
        qtbot.waitUntil(lambda: len(dialog._fields) == 4 and dialog._normalization_ready
                        and not dialog._index_jobs.is_busy(), timeout=10000)
        assert dialog._normalization_plan["reference_means"] == {1: 7., 2: 14.}
        dialog.export_scope.setCurrentIndex(dialog.export_scope.findData("selection"))
        dialog.wells.clearSelection()
        dialog.wells.item(*dialog._well_positions["C02"]).setSelected(True)
        dialog.measurements.setChecked(True)
        dialog.pdf_dpi.setCurrentIndex(dialog.pdf_dpi.findData(150))
        old_context, old_cancelled = dialog._normalization_context, dialog._cancelled
        old_generation = dialog._display_generation
        if cancel_export:
            actual_read = browser._overview_raw_planes

            def paused_full_source(field, channels, **options):
                yield from actual_read(field, channels, **options)
                if field.well == "C02" and options.get("edge", 256) is None:
                    entered.set()
                    if not release.wait(10):
                        raise RuntimeError("The real full-source reader was not released.")

            monkeypatch.setattr(browser, "_overview_raw_planes", paused_full_source)
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: str(output))
        export = next(button for button in dialog.findChildren(QPushButton)
                      if button.text() == browser.tr("Export PDFs"))
        qtbot.mouseClick(dialog.options_button, browser.Qt.LeftButton)
        dialog._controls_scroll.ensureWidgetVisible(export)
        qtbot.mouseClick(export, browser.Qt.LeftButton)
        fresh = dialog._normalization_context
        assert old_cancelled.is_set() and old_context._cancelled is old_cancelled
        assert fresh is not old_context and fresh._cancelled is dialog._cancelled
        assert not dialog._cancelled.is_set() and dialog._display_generation > old_generation
        assert fresh._pending is not old_context._pending and fresh._lock is not old_context._lock
        assert fresh._cache is not old_context._cache and fresh._used is not old_context._used
        if cancel_export:
            qtbot.waitUntil(entered.is_set, timeout=10000)
            cancel = next(button for button in dialog.findChildren(QPushButton)
                          if button.text() == browser.tr("Cancel loading or export"))
            dialog._controls_scroll.ensureWidgetVisible(cancel)
            qtbot.mouseClick(cancel, browser.Qt.LeftButton)
            release.set()
        qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy()
                        and (output / "overview_selection.json").exists(), timeout=10000)
        metadata = json.loads((output / "overview_selection.json").read_text())
        assert metadata["normalization"]["reference_means"] == {"1": 7., "2": 14.}
        assert [(row["well"], row["means"]["1"])
                for row in metadata["normalization"]["reference_wells"]] == [("A01", 4.), ("B01", 10.)]
        assert metadata["normalization"]["selection"]["generation"] == dialog._display_generation
        if cancel_export:
            assert metadata["cancelled"] is True and metadata["fields"] == []
            assert not list(output.glob("*.pdf"))
            assert not (output / "overview_measurements.csv").exists()
            assert not list(output.glob(".*pending*"))
        else:
            from pypdf import PdfReader

            assert metadata["cancelled"] is False and metadata["written_field_count"] == 1
            page, = PdfReader(next(output.glob("*.pdf"))).pages
            embedded, = page.images
            expected = np.zeros((2, 2, 3), dtype=np.uint8)
            expected[..., 0] = expected[..., 1] = np.uint8(np.clip(base * 3.5 / 12., 0., 1.) * 255)
            np.testing.assert_array_equal(np.asarray(embedded.image.convert("RGB"))[::-1], expected)
            table = tabular.read_table(output / "overview_measurements.csv", canonicalise=False)
            assert len(table) == 1
            row = table.iloc[0]
            assert (row["plate"], row["well"], row["condition"]) == ("p1", "C02", "treated")
            assert row["channel_1_intensity_mean"] == 2. and row["channel_2_intensity_mean"] == 4.
            target = next(item for item in metadata["normalization"]["used_wells"] if item["well"] == "C02")
            assert target["gains"] == {"1": 3.5, "2": 3.5}
        assert all(path.read_bytes() == data for path, data in originals.items())
    finally:
        release.set()
        dialog._cancel()
        runners = (dialog._index_jobs, dialog._image_jobs, dialog._export_jobs,
                   dialog._thumbnail_jobs, dialog._montage_jobs, dialog._field_thumbnail_jobs)
        qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners), timeout=10000)
        for runner in runners:
            runner.shutdown(timeout_ms=3000)
        dialog.close()
