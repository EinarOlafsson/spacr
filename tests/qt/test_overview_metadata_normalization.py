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
    ("reference_condition", "control"),
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
               "conditions": {("p", "A01"): "control", ("other", "A01"): "other"}}
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
    assert config["normalization"]["reference"] == "vehicle"
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
    ("reference_condition", "control"),
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
