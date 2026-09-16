"""418: raw own-channel means filter cached labels, including every slot."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox

from spacr.qt.widgets import live_preview as LP


@pytest.fixture
def panel(qtbot):
    widget = LP.LivePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


def _labels():
    mask = np.zeros((20, 20), np.int32)
    for label, (y, x) in enumerate(((2, 2), (2, 10), (10, 2), (10, 10)), 1):
        mask[y:y + 2, x:x + 2] = label
    return mask


def _kept(mask, labels):
    """Explicit surviving label IDs, independent of the production filter."""
    expected = np.zeros_like(mask)
    for new, old in enumerate(labels, 1):
        expected[mask == old] = new
    return expected


def _choose(panel, caption):
    index = panel._object_box.findData(caption)
    assert index >= 0, f"missing object choice: {caption}"
    panel._object_box.setCurrentIndex(index)


@pytest.mark.parametrize("role,caption,channel", [
    ("cell", "cell", 3), ("nucleus", "nucleus", 2),
    ("pathogen", "pathogen", 4), ("organelle", "organelle", 1),
    ("organelleb", "organelle 2", 5), ("organellec", "organelle 3", 0),
])
def test_live_bounds_keep_equal_means_and_restore_raw_labels_without_a_model_run(
        panel, qtbot, monkeypatch, role, caption, channel):
    raw = _labels()
    image = np.full((*raw.shape, 6), 999.0, np.float64)
    plane = image[..., channel]
    # Object 1 has a mean exactly at the floor, but a peak above the cap:
    # a maximum-pixel implementation would wrongly remove it.
    plane[raw == 1] = [0.25, 0.25, 20.25, 20.25]
    plane[raw == 2] = 16.75
    plane[raw == 3] = 9.25
    plane[raw == 4] = 25.0
    panel.apply_settings({
        "number_of_organelles": 3,
        "cell_channel": 3, "nucleus_channel": 2, "pathogen_channel": 4,
        "organelle_channel": 1, "organelleb_channel": 5,
        "organellec_channel": 0,
        # The fixture has four-pixel labels, below Mask's organelle area
        # default of 10. Keep an explicit, inclusive positive area bound.
        f"{role}_min_area": 4,
    })
    _choose(panel, caption)
    panel._image = image
    calls = []

    def segment(request):
        calls.append(request)
        assert request.object_types == (role,)
        assert request.channels[role] == channel
        return {role: raw.copy()}, {}

    monkeypatch.setattr(LP, "_segment_multi", segment)
    monkeypatch.setattr(panel, "_model_for_this_pass", lambda: ("cpsam", ""))
    ready = []
    panel.preview_ready.connect(ready.append)
    panel.run_preview()
    qtbot.waitUntil(lambda: bool(ready), timeout=5000)
    qtbot.waitUntil(lambda: panel._worker is None
                   or not panel._worker.isRunning(), timeout=5000)
    assert len(calls) == 1
    np.testing.assert_array_equal(panel._masks[role], raw)

    group = panel._compartment_widgets[
        "organelle" if role.startswith("organelle") else role]
    group["min_intensity"].setValue(10.25)
    group["max_intensity"].setValue(16.75)
    expected = _kept(raw, (1, 2))
    np.testing.assert_array_equal(panel._masks[role], expected)
    np.testing.assert_array_equal(ready[-1][role], expected)
    np.testing.assert_array_equal(panel._raw_masks[role], raw)
    np.testing.assert_array_equal(panel._image, image)
    propagated = panel.settings_for_propagation()
    assert propagated[f"{role}_min_intensity"] == 10.25
    assert propagated[f"{role}_max_intensity"] == 16.75
    assert len(calls) == 1

    # Area and intensity compose: an area floor above four removes even
    # intensity-qualified objects, then relaxing it restores the same labels.
    group["min_area"].setValue(5)
    np.testing.assert_array_equal(panel._masks[role], np.zeros_like(raw))
    group["min_area"].setValue(4)
    np.testing.assert_array_equal(panel._masks[role], expected)
    np.testing.assert_array_equal(panel._raw_masks[role], raw)

    # Display normalisation must not change the raw values being judged.
    panel._normalise_check.setChecked(False)
    panel._recompute_masks()
    np.testing.assert_array_equal(panel._masks[role], expected)
    group["min_intensity"].setValue(0)
    np.testing.assert_array_equal(panel._masks[role], _kept(raw, (1, 2, 3)))
    group["max_intensity"].setValue(0)
    np.testing.assert_array_equal(panel._masks[role], raw)
    np.testing.assert_array_equal(panel._raw_masks[role], raw)
    assert len(calls) == 1, "changing filters must not re-run segmentation"


def test_filter_rows_and_seeded_float_bounds_propagate_for_each_compartment(panel):
    suffixes = [row[0] for row in LP.COMPARTMENT_FIELDS]
    assert suffixes == ["min_area", "max_area", "min_intensity", "max_intensity",
                        "perimeter_fraction", "remove_border_objects"]
    settings = {"number_of_organelles": 1}
    for role in LP.COMPARTMENTS:
        settings[f"{role}_min_intensity"] = 0.125
        settings[f"{role}_max_intensity"] = 64000.875
    panel.apply_settings(settings)
    propagated = panel.settings_for_propagation()
    for role in LP.COMPARTMENTS:
        group = panel._compartment_widgets[role]
        for suffix in ("min_intensity", "max_intensity"):
            assert isinstance(group[suffix], QDoubleSpinBox)
            assert group[suffix].decimals() == 6
            assert propagated[f"{role}_{suffix}"] == settings[f"{role}_{suffix}"]


@pytest.mark.parametrize("suffix", ["min_intensity", "max_intensity"])
def test_small_intensity_limits_are_visible_and_propagate_exactly(panel, suffix):
    key = f"cell_{suffix}"
    panel.apply_settings({key: 0.0004})
    widget = panel._compartment_widgets["cell"][suffix]
    assert widget.value() == 0.0004
    assert "0004" in widget.text()
    assert panel.settings_for_propagation()[key] == 0.0004
    widget.setValue(0.000125)
    assert panel.settings_for_propagation()[key] == 0.000125


@pytest.mark.parametrize("suffix", ["min_intensity", "max_intensity"])
def test_edit_then_zero_does_not_restore_a_hidden_seeded_limit(panel, suffix):
    raw = _labels()
    key = f"cell_{suffix}"
    # Below the six-decimal display resolution: preserve a loaded value
    # until the user edits it, then never resurrect it on a return to 0.
    seeded = 0.0000004
    value = 0.0 if suffix == "min_intensity" else 1.0
    panel._image = np.full(raw.shape, value, np.float64)
    panel._raw_masks = {"cell": raw.copy()}
    panel.apply_settings({key: seeded})
    widget = panel._compartment_widgets["cell"][suffix]
    assert widget.value() == 0
    assert panel.settings_for_propagation()[key] == seeded
    np.testing.assert_array_equal(panel._masks["cell"], np.zeros_like(raw))

    widget.setValue(2)
    assert panel.settings_for_propagation()[key] == 2
    assert id(widget) not in panel._clamped_on_seeding
    widget.setValue(0)
    assert panel.settings_for_propagation()[key] == 0
    np.testing.assert_array_equal(panel._masks["cell"], raw)
    np.testing.assert_array_equal(panel._raw_masks["cell"], raw)
    assert panel._worker is None


@pytest.mark.parametrize("suffix", ["min_intensity", "max_intensity"])
@pytest.mark.parametrize("seeded", [0.0000004, -1, np.nan],
                         ids=["rounded-active-bound", "negative", "nan"])
def test_typing_zero_directly_discards_only_an_edited_hidden_bound(
        panel, qtbot, suffix, seeded):
    raw = _labels()
    key = f"cell_{suffix}"
    value = 0.0 if suffix == "min_intensity" else 1.0
    panel._image = np.full(raw.shape, value, np.float64)
    panel._raw_masks = {"cell": raw.copy()}
    panel.apply_settings({key: seeded})
    widget = panel._compartment_widgets["cell"][suffix]
    assert widget.value() == 0
    assert id(widget) in panel._clamped_on_seeding
    panel.show()
    widget.show()
    widget.setFocus()

    # Merely committing an untouched control must preserve the loaded value.
    # Otherwise opening/focusing a rounded field silently changes the run.
    qtbot.keyClick(widget, Qt.Key_Return)
    preserved = panel.settings_for_propagation()[key]
    if np.isnan(seeded):
        assert np.isnan(preserved)
    else:
        assert preserved == seeded
    assert id(widget) in panel._clamped_on_seeding
    if np.isfinite(seeded) and seeded > 0:
        np.testing.assert_array_equal(panel._masks["cell"], np.zeros_like(raw))
    else:
        assert panel._masks == {}
        assert "Preview failed:" in panel._status.text()

    # Actual keyboard replacement, not a signal fake or a detour through 2.
    # The numeric value stays 0, so valueChanged alone cannot see this edit.
    widget.selectAll()
    qtbot.keyClicks(widget.lineEdit(), "0")
    qtbot.keyClick(widget, Qt.Key_Return)

    assert widget.value() == 0
    assert panel.settings_for_propagation()[key] == 0
    assert id(widget) not in panel._clamped_on_seeding
    np.testing.assert_array_equal(panel._masks["cell"], raw)
    np.testing.assert_array_equal(panel._raw_masks["cell"], raw)
    assert panel._worker is None


def test_perimeter_merge_precedes_mean_filter_and_uses_the_untouched_cache(panel):
    raw = np.zeros((16, 16), np.int32)
    raw[4:10, 3:6] = 1
    raw[4:10, 6:9] = 2
    image = np.zeros(raw.shape, np.float64)
    image[raw == 1] = 2
    image[raw == 2] = 18
    panel._image = image
    panel._on_worker_done({"cell": raw.copy()}, None)
    group = panel._compartment_widgets["cell"]
    group["min_intensity"].setValue(8)
    group["max_intensity"].setValue(12)
    np.testing.assert_array_equal(panel._masks["cell"], np.zeros_like(raw))

    group["perimeter_fraction"].setValue(0.1)
    # The adjacent equal-area objects have a merged mean of 10. Filtering
    # before merging, or omitting perimeter merging, would reject both.
    np.testing.assert_array_equal(panel._masks["cell"], (raw > 0).astype(raw.dtype))
    np.testing.assert_array_equal(panel._raw_masks["cell"], raw)
    group["min_intensity"].setValue(0)
    group["max_intensity"].setValue(0)
    group["perimeter_fraction"].setValue(0)
    np.testing.assert_array_equal(panel._masks["cell"], raw)
    assert panel._worker is None


def test_switching_organelle_slots_keeps_their_own_bounds_and_raw_channels(
        panel, monkeypatch):
    raw = _labels()
    image = np.zeros((*raw.shape, 3), np.float64)
    for channel, means in enumerate(((60, 50, 40, 30), (10, 20, 30, 40),
                                     (400, 300, 200, 100))):
        for label, mean in enumerate(means, 1):
            image[..., channel][raw == label] = mean
    panel.apply_settings({
        "number_of_organelles": 3,
        "organelle_channel": 1, "organelleb_channel": 2,
        "organellec_channel": 0,
        "organelle_min_area": 4, "organelleb_min_area": 4,
        "organellec_min_area": 4,
        "organelle_min_intensity": 15, "organelle_max_intensity": 35,
        "organelleb_min_intensity": 150, "organelleb_max_intensity": 250,
        "organellec_min_intensity": 45, "organellec_max_intensity": 65,
    })
    panel._image = image
    roles = ("organelle", "organelleb", "organellec")
    panel._on_worker_done({role: raw.copy() for role in roles}, None)
    monkeypatch.setattr(panel, "run_preview",
                        lambda: pytest.fail("cached filters started segmentation"))
    expected = {"organelle": (2, 3), "organelleb": (3,), "organellec": (1, 2)}
    for role, caption, bounds in (
            ("organelleb", "organelle 2", (150, 250)),
            ("organellec", "organelle 3", (45, 65)),
            ("organelle", "organelle", (15, 35))):
        _choose(panel, caption)
        group = panel._compartment_widgets["organelle"]
        assert (group["min_intensity"].value(),
                group["max_intensity"].value()) == bounds
        for cached_role, kept in expected.items():
            np.testing.assert_array_equal(panel._masks[cached_role], _kept(raw, kept))
            np.testing.assert_array_equal(panel._raw_masks[cached_role], raw)

    group["min_intensity"].setValue(25)
    _choose(panel, "organelle 2")
    group["max_intensity"].setValue(350)
    _choose(panel, "organelle")
    assert group["min_intensity"].value() == 25
    np.testing.assert_array_equal(panel._masks["organelle"], _kept(raw, (3,)))
    np.testing.assert_array_equal(panel._masks["organelleb"], _kept(raw, (2, 3)))

    for caption in ("organelle", "organelle 2", "organelle 3"):
        _choose(panel, caption)
        group["min_intensity"].setValue(0)
        group["max_intensity"].setValue(0)
    for role in roles:
        np.testing.assert_array_equal(panel._masks[role], raw)
        np.testing.assert_array_equal(panel._raw_masks[role], raw)
    assert panel._worker is None


def test_legacy_organelle_border_flag_can_be_switched_off_and_stays_off(panel):
    raw = np.zeros((20, 20), np.int32)
    raw[0:2, 0:2] = 1
    raw[8:10, 8:10] = 2
    panel.apply_settings({
        "number_of_organelles": 2,
        "organelle_channel": 0, "organelleb_channel": 1,
        "organelle_min_area": 4, "organelleb_min_area": 4,
        "organelle_min_intensity": 5, "organelleb_min_intensity": 5,
        "organelleb_remove_border": True,
        "organelleb_remove_border_objects": False,
    })
    panel._image = np.full((*raw.shape, 2), 10.0)
    panel._on_worker_done({"organelle": raw.copy(), "organelleb": raw.copy()}, None)
    # The second slot has not owned the shared widgets yet. Its legacy
    # setting must still apply to its cached labels, without affecting slot 1.
    np.testing.assert_array_equal(panel._masks["organelleb"], _kept(raw, (2,)))
    np.testing.assert_array_equal(panel._masks["organelle"], raw)

    _choose(panel, "organelle 2")
    checkbox = panel._compartment_widgets["organelle"]["remove_border_objects"]
    assert checkbox.isChecked()
    propagated = panel.settings_for_propagation()
    assert propagated["organelleb_remove_border"] is True
    assert propagated["organelleb_remove_border_objects"] is True
    checkbox.setChecked(False)
    propagated = panel.settings_for_propagation()
    assert propagated["organelleb_remove_border"] is False
    assert propagated["organelleb_remove_border_objects"] is False
    np.testing.assert_array_equal(panel._masks["organelleb"], raw)

    _choose(panel, "organelle")
    assert panel._settings["organelleb_remove_border"] is False
    assert panel._settings["organelleb_remove_border_objects"] is False
    np.testing.assert_array_equal(panel._masks["organelleb"], raw)
    _choose(panel, "organelle 2")
    assert not checkbox.isChecked()
    np.testing.assert_array_equal(panel._masks["organelleb"], raw)
    np.testing.assert_array_equal(panel._raw_masks["organelleb"], raw)
    assert panel._worker is None


@pytest.mark.parametrize("bad_input,detail", [
    ("missing", "Load an image"),
    ("shape", "same shape"),
    ("nonfinite", "finite object mean"),
])
def test_invalid_active_intensity_input_fails_visibly_and_keeps_the_raw_cache(
        panel, bad_input, detail):
    raw = _labels()
    image = np.full(raw.shape, 10.0)
    panel.apply_settings({"cell_min_intensity": 5})
    panel._image = image
    panel._on_worker_done({"cell": raw.copy()}, None)
    np.testing.assert_array_equal(panel._masks["cell"], raw)
    ready = []
    panel.preview_ready.connect(ready.append)
    if bad_input == "missing":
        panel._image = None
    elif bad_input == "shape":
        panel._image = image[:-1]
    else:
        panel._image = image.copy()
        panel._image[raw == 1] = np.nan
    panel._recompute_masks()
    assert panel._masks == {}
    assert ready == [None]
    assert "Preview failed:" in panel._status.text()
    assert detail in panel._status.text()
    np.testing.assert_array_equal(panel._raw_masks["cell"], raw)

    panel._image = image
    panel._recompute_masks()
    np.testing.assert_array_equal(panel._masks["cell"], raw)
    assert ready[-1] is not None


@pytest.mark.parametrize("bound", [np.nan, np.inf, -1, "not-a-number"])
def test_invalid_seeded_intensity_bounds_are_not_silently_disabled(panel, bound):
    raw = _labels()
    panel._image = np.full(raw.shape, 10.0)
    panel._raw_masks = {"cell": raw.copy()}
    panel.apply_settings({"cell_min_intensity": bound})
    assert panel._masks == {}
    assert "Preview failed:" in panel._status.text()
    np.testing.assert_array_equal(panel._raw_masks["cell"], raw)
