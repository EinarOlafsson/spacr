"""The crop preview accepts a folder, and opens on `src` by itself.

The preview exists to answer "will this crop size cut the cell in half" before
a run costs an hour. It answered nothing until the user had found and chosen
one of fifty-two merged arrays by hand, through a dialog that would only accept
a `.npy` -- while the thing the user actually has, and has already typed, is
the run folder.

Reported 2026-09-01: "when the user puts in a path for src that path should be
auto checked for a merged folder and crops loaded from a random image with
default settings. in the choose merged array the user should be able to past a
folder path".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spacr.qt.widgets.measure_preview import (load_merged_array,
                                              resolve_merged_source)


def _an_array(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros((4, 4, 7), dtype=np.uint16))
    return path


def test_a_file_resolves_to_itself(tmp_path):
    array = _an_array(tmp_path / "plate1_E01_1_1.npy")
    assert resolve_merged_source(array) == array


def test_a_run_folder_resolves_into_its_merged_subfolder(tmp_path):
    """What `src` holds. A Measure run is pointed at the plate directory and
    finds `merged/` itself; the preview now does the same."""
    wanted = _an_array(tmp_path / "merged" / "plate1_E01_1_1.npy")
    (tmp_path / "measurements").mkdir()
    (tmp_path / "qc").mkdir()

    assert resolve_merged_source(tmp_path) == wanted


def test_a_merged_folder_resolves_directly(tmp_path):
    wanted = _an_array(tmp_path / "plate1_E01_1_1.npy")
    assert resolve_merged_source(tmp_path) == wanted


def test_the_merged_subfolder_wins_over_a_loose_array(tmp_path):
    """A run folder holds `merged/` beside `measurements/` and `qc/`; an array
    loose in the run folder is not the one meant."""
    _an_array(tmp_path / "stray.npy")
    wanted = _an_array(tmp_path / "merged" / "real.npy")
    assert resolve_merged_source(tmp_path) == wanted


def test_the_field_is_random_not_always_the_first(tmp_path):
    """Sorted-first is always the same well and the same field. A preview that
    always opens on E01 field 1 reports one corner of one condition -- and if
    that field is clean, a crop size that halves cells everywhere else looks
    fine."""
    for well in ("E01", "E02", "L01", "L02"):
        for field in (1, 9, 10, 11):
            _an_array(tmp_path / "merged" / f"plate1_{well}_{field}_1.npy")

    seen = {resolve_merged_source(tmp_path).name for _ in range(80)}

    assert len(seen) > 1, "the same field came back every time"


def test_a_folder_with_no_arrays_resolves_to_nothing(tmp_path):
    (tmp_path / "merged").mkdir()
    assert resolve_merged_source(tmp_path) is None


def test_a_path_that_does_not_exist_resolves_to_nothing(tmp_path):
    assert resolve_merged_source(tmp_path / "nope") is None


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_nothing_resolves_to_nothing(blank):
    assert resolve_merged_source(blank) is None


def test_the_loader_accepts_a_folder(tmp_path):
    """End to end: the worker resolves and reads in one call."""
    _an_array(tmp_path / "merged" / "plate1_E01_1_1.npy")

    payload = load_merged_array(str(tmp_path))

    assert payload["error"] == ""
    assert payload["data"] is not None
    assert payload["path"].endswith("plate1_E01_1_1.npy"), (
        "the payload must name the file it actually read, not the folder")


def test_an_empty_folder_says_what_is_wrong(tmp_path):
    """"No merged .npy found" is an answer; a bare load failure is not."""
    (tmp_path / "merged").mkdir()
    payload = load_merged_array(str(tmp_path))
    assert "No merged .npy found" in payload["error"]
    assert payload["data"] is None


# ---------------------------------------------------------------------------
# Opening on `src` by itself
# ---------------------------------------------------------------------------

@pytest.fixture
def panel(qapp):
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel

    return MeasurePreviewPanel(threaded=False)


def test_a_src_setting_loads_a_field(panel, tmp_path, monkeypatch):
    _an_array(tmp_path / "merged" / "plate1_E01_1_1.npy")
    asked = []
    monkeypatch.setattr(panel, "load_array_async",
                        lambda p, **k: asked.append(p) or True)

    panel.apply_settings({"src": str(tmp_path)})

    assert asked == [str(tmp_path)]


def test_it_does_not_reload_on_every_settings_change(panel, tmp_path,
                                                     monkeypatch):
    """Re-randomising the field each time an unrelated setting is touched
    would make the preview unusable to tune against."""
    asked = []
    monkeypatch.setattr(panel, "load_array_async",
                        lambda p, **k: asked.append(p) or True)

    panel.apply_settings({"src": str(tmp_path)})
    panel.apply_settings({"src": str(tmp_path), "cell_min_size": 42})

    assert len(asked) == 1


def test_a_new_src_does_load_again(panel, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(panel, "load_array_async",
                        lambda p, **k: asked.append(p) or True)

    panel.apply_settings({"src": str(tmp_path / "a")})
    panel.apply_settings({"src": str(tmp_path / "b")})

    assert len(asked) == 2


def test_an_array_the_user_chose_is_not_taken_away(panel, tmp_path,
                                                   monkeypatch):
    """Auto-loading over a deliberate choice is worse than not auto-loading."""
    panel._data = np.zeros((4, 4, 7), dtype=np.uint16)
    asked = []
    monkeypatch.setattr(panel, "load_array_async",
                        lambda p, **k: asked.append(p) or True)

    panel.apply_settings({"src": str(tmp_path)})

    assert asked == []


def test_no_src_asks_for_nothing(panel, monkeypatch):
    monkeypatch.setattr(panel, "load_array_async",
                        lambda p, **k: pytest.fail("loaded with no src"))
    panel.apply_settings({"cell_min_size": 42})


def test_a_pasted_folder_is_loaded(panel, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(panel, "load_array_async",
                        lambda p, **k: asked.append(p) or True)

    panel._paste_box.setText(str(tmp_path))
    panel._load_the_pasted_path()

    assert asked == [str(tmp_path)]


def test_an_empty_paste_box_does_nothing(panel, monkeypatch):
    monkeypatch.setattr(panel, "load_array_async",
                        lambda p, **k: pytest.fail("loaded an empty path"))
    panel._paste_box.setText("   ")
    panel._load_the_pasted_path()


# ---------------------------------------------------------------------------
# One organelle slot per slot the run declares
# ---------------------------------------------------------------------------
#
# The crop settings offered a fixed four, so a one-organelle run had three
# mask-slice and three minimum-area fields for objects it does not have -- and
# each of them propagates into the settings the run reads.


def _visible_organelle_rows(dialog):
    return sorted({role for role, form, widget in dialog._organelle_rows
                   if form.isRowVisible(form.getWidgetPosition(widget)[0])})


@pytest.mark.parametrize("count, expected", [
    (1, ["organelle"]),
    (2, ["organelle", "organelleb"]),
    (3, ["organelle", "organelleb", "organellec"]),
])
def test_the_declared_count_decides_how_many_slots_are_shown(
        panel, count, expected):
    from spacr.qt.widgets.measure_preview import CropSettingsDialog

    panel.apply_settings({"number_of_organelles": count})

    assert _visible_organelle_rows(CropSettingsDialog(panel)) == expected


def test_an_unset_count_shows_every_slot(panel):
    """Better a field too many than one hidden that a run is using."""
    from spacr.qt.widgets.measure_preview import CropSettingsDialog
    from spacr.object_roles import ORGANELLE_ROLES

    assert (_visible_organelle_rows(CropSettingsDialog(panel))
            == sorted(ORGANELLE_ROLES))


def test_lowering_then_raising_the_count_keeps_the_values(panel):
    """Hidden, not removed -- the same promise `_set_organelle_defaults`
    makes for the settings themselves."""
    from spacr.qt.widgets.measure_preview import CropSettingsDialog

    panel.apply_settings({"number_of_organelles": 2})
    panel._mask_dims["organelleb"].setValue(5)

    panel.apply_settings({"number_of_organelles": 1})
    panel.apply_settings({"number_of_organelles": 2})

    assert panel._mask_dims["organelleb"].value() == 5
    assert "organelleb" in _visible_organelle_rows(CropSettingsDialog(panel))


def test_an_open_dialog_is_re_gated_when_the_count_changes(panel):
    """The count can change while the dialog is open; it must follow."""
    from spacr.qt.widgets.measure_preview import CropSettingsDialog

    panel.apply_settings({"number_of_organelles": 3})
    dialog = CropSettingsDialog(panel)
    panel._crop_settings_dialog = dialog
    assert "organellec" in _visible_organelle_rows(dialog)

    panel.set_organelle_count(1)

    assert _visible_organelle_rows(dialog) == ["organelle"]


def test_a_junk_count_is_ignored_rather_than_hiding_everything(panel):
    from spacr.qt.widgets.measure_preview import CropSettingsDialog

    panel.apply_settings({"number_of_organelles": 2})
    panel.set_organelle_count("not a number")

    assert _visible_organelle_rows(CropSettingsDialog(panel)) == [
        "organelle", "organelleb"]


# ---------------------------------------------------------------------------
# Propagate is a button, as it is in the Mask live settings dialog
# ---------------------------------------------------------------------------

def test_propagate_is_a_checkable_button_not_a_slider(panel):
    """Two dialogs that sit side by side used two different controls for the
    same action, and a slider reads as a SETTING of the crop preview rather
    than as something that reaches out of it."""
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.widgets.toggle import Toggle

    assert isinstance(panel._propagate_btn, QPushButton)
    assert not isinstance(panel._propagate_btn, Toggle)
    assert panel._propagate_btn.isCheckable()


def test_it_is_styled_as_the_live_dialog_styles_its_toggle(panel):
    """`ToggleButton` is the object name the theme paints blue, which is what
    makes it read as the same control as the Mask one."""
    assert panel._propagate_btn.objectName() == "ToggleButton"


def test_toggling_it_still_drives_propagation(panel):
    """The widget changed; the behaviour must not have."""
    sent = []
    panel.set_propagate_callback(lambda values: sent.append(values))

    panel._propagate_btn.setChecked(True)

    assert sent, "turning it on pushed nothing into the main settings"
