"""The Measure crop preview's quiet paths: the ones that do nothing on purpose.

Everything here is a branch the panel takes in order *not* to act -- a cap
change with no sentence to show, a cancelled file dialog, a settings file whose
``normalize`` key is empty, a click on a thumbnail index that no longer exists,
a control that offers no signal to connect to. Each of them is a place where
the panel has to keep the state the user can see exactly as it was; the ones
that got this wrong shipped a status line describing a field of view nobody had
loaded, and a Crop-settings pass that stopped wiring the rest of its controls
at the first widget that did not answer to ``valueChanged``.

Every test drives the *acting* half of the same branch as well, so an assertion
about something not happening cannot pass on code that never ran.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QLabel, QSizePolicy, QSpacerItem, QWidget

from spacr.qt.widgets import measure_preview as MP

_PLACEHOLDER = "No array loaded — drop a merged .npy here, or choose one"


def _merged(directory, name="plate1_A01_f1.npy"):
    """One 8-plane merged array: three image planes plus the four masks."""
    data = np.zeros((32, 32, 8), np.float32)
    data[..., :3] = 20
    cell = np.zeros((32, 32), np.int32)
    cell[2:14, 2:14] = 1
    cell[18:30, 18:30] = 2
    nucleus = np.zeros_like(cell)
    nucleus[4:8, 4:8] = 1
    pathogen = np.zeros_like(cell)
    pathogen[20:24, 20:24] = 1
    data[..., 4] = cell
    data[..., 5] = nucleus
    data[..., 6] = pathogen
    path = directory / name
    np.save(path, data)
    return str(path)


def _merged_folder(tmp_path, count=3):
    """A folder holding ``count`` merged arrays, one per field of view."""
    return [_merged(tmp_path, f"plate1_A01_f{i + 1}.npy") for i in range(count)]


class _MuteSpin(QWidget):
    """A spinbox-shaped control that publishes no ``valueChanged``."""

    def value(self):
        return 0


class _MuteToggle(QWidget):
    """A toggle-shaped control that publishes no ``toggled``."""

    def isChecked(self):
        return False


@pytest.fixture(autouse=True)
def _application(qapp):
    """Every Qt object built here needs the shared application to exist."""
    return qapp


@pytest.fixture
def panel(qtbot):
    """An unthreaded panel, so every load and crop lands before the call ends."""
    widget = MP.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Parsing the measured-channel list
# ---------------------------------------------------------------------------


def test_a_typo_in_the_channel_list_costs_only_that_entry():
    """A half-typed channel field must not empty the measured-channel list.

    ``channels`` is a free-text field: while the user is still typing, "0,,2"
    and "0, x, 2" both exist for a keystroke or two, and the parser is called
    on every one of them because the field re-propagates as it changes. If a
    non-numeric fragment stopped the scan, the run would be handed a shorter
    channel list than the field shows, and the measurement would silently skip
    a stain the user believes they asked for.
    """
    assert MP._parse_channels("2, x, 3;4") == [2, 3, 4]
    assert MP._parse_channels("0,,2") == [0, 2]
    # The digits-only path, so the skip above is a real detour and not the
    # only thing this parser can do.
    assert MP._parse_channels("0;1;2") == [0, 1, 2]


# ---------------------------------------------------------------------------
# Propagation switch
# ---------------------------------------------------------------------------


def test_switching_propagation_off_copies_nothing_into_the_run(panel):
    """Turning the switch off must not push one last settings copy.

    The Propagate toggle is the user's answer to "should this dialog overwrite
    my Measure form?". A copy fired while it is being switched OFF would
    overwrite the form the user just decided to protect, and the values it
    wrote would be the preview's, not theirs.
    """
    seen = []
    panel.set_propagate_callback(seen.append)

    panel._on_propagate_toggled(True)
    assert len(seen) == 1
    assert seen[0]["experiment"] == "exp"

    panel._on_propagate_toggled(False)
    assert len(seen) == 1, "switching propagation off still copied settings"


# ---------------------------------------------------------------------------
# Choosing an array
# ---------------------------------------------------------------------------


def _stub_file_dialog(monkeypatch, chosen):
    class _Dialog:
        @staticmethod
        def getOpenFileName(*_args, **_kwargs):
            return chosen, "NumPy arrays (*.npy)"

    monkeypatch.setattr(MP, "QFileDialog", _Dialog)


def test_a_cancelled_choose_dialog_leaves_the_loaded_array_alone(
        panel, monkeypatch, tmp_path):
    """Pressing Cancel in the file chooser must not disturb the panel.

    ``getOpenFileName`` returns an empty string when the user cancels. Handing
    that to the loader would clear the array on screen and put a "Loading …"
    line up for a file nobody picked -- the user pressing Cancel is saying
    "keep what I have".
    """
    _stub_file_dialog(monkeypatch, "")
    panel._pick_file()
    assert panel._data is None
    assert panel._path_label.text() == _PLACEHOLDER

    _stub_file_dialog(monkeypatch, _merged(tmp_path))
    panel._pick_file()
    assert panel._data is not None and panel._data.shape == (32, 32, 8)
    assert "plate1_A01_f1.npy" in panel._path_label.text()
    assert "shape (32, 32, 8)" in panel._path_label.text()


# ---------------------------------------------------------------------------
# FOV / channel selectors
# ---------------------------------------------------------------------------


def test_the_selectors_refresh_before_any_array_is_loaded(panel, tmp_path):
    """Refreshing the dropdowns with no array must not scan a folder.

    ``_refresh_source_selectors`` is called from the cap spinbox as well as
    from a load, so it runs while ``_data_path`` is still None. Enumerating
    then would mean ``Path(None).parent`` -- and a 384-well plate's merged
    folder is thousands of files, which is exactly why the panel only lists
    one once it knows which folder it is looking at.
    """
    panel._refresh_source_selectors()
    assert panel._fov_box.count() == 0
    assert [panel._channel_box.itemText(i)
            for i in range(panel._channel_box.count())] == ["All channels"]

    # The same call once a path IS known fills both dropdowns from the folder.
    paths = _merged_folder(tmp_path)
    assert panel.load_array(paths[0]) is True
    assert panel._fov_box.count() == len(paths)
    assert panel._channel_box.count() == 9   # 8 planes plus "All channels"


def test_a_new_cap_with_no_sentence_to_show_keeps_the_status_line(
        panel, monkeypatch, tmp_path):
    """A resample must only overwrite the status line when it has something to say.

    The cap spinbox reports the new sample on the status line. When the sampler
    produces no sentence -- nothing enumerated to describe -- writing it anyway
    would blank whatever the line last said, which is usually the crop count or
    the reason the last pass could not run.
    """
    monkeypatch.setattr(MP, "apply_sample_to_combo",
                        lambda *_args, **_kwargs: "")
    panel._status.setText("2 object(s) · 1 category")
    panel._on_max_sets_changed(7)
    assert panel.sample_note() == ""
    assert panel._status.text() == "2 object(s) · 1 category"

    # With a real sample behind it the same call does report, capitalised.
    monkeypatch.undo()
    panel.load_array(_merged_folder(tmp_path)[0])
    panel._on_max_sets_changed(2)
    assert panel.sample_note().startswith("showing ")
    assert panel._status.text() == (
        panel.sample_note()[:1].upper() + panel.sample_note()[1:])
    assert panel._status.text()[0].isupper()


# ---------------------------------------------------------------------------
# Seeding from a settings dict
# ---------------------------------------------------------------------------


def test_an_empty_normalize_key_leaves_the_percentiles_alone(panel):
    """A settings file whose ``normalize`` is None must not unset normalisation.

    ``normalize`` is a tri-state in the Measure settings: a [lo, hi] pair, a
    bool, or absent/None meaning "not specified". Reading None as False would
    turn percentile scaling off for every settings file that simply never
    mentioned it, and the crops would come back dark for a run that normalises.
    """
    panel._normalise.setChecked(True)
    panel._lo_pct.setValue(2.0)
    panel._hi_pct.setValue(98.0)

    panel.apply_settings({"normalize": None, "experiment": "seeded"})
    assert panel._normalise.isChecked() is True
    assert (panel._lo_pct.value(), panel._hi_pct.value()) == (2.0, 98.0)
    assert panel._experiment.text() == "seeded"   # the rest was still applied

    # A stated bool is honoured, so the None case really is a separate path.
    panel.apply_settings({"normalize": False})
    assert panel._normalise.isChecked() is False
    panel.apply_settings({"normalize": [5.0, 95.0]})
    assert panel._normalise.isChecked() is True
    assert (panel._lo_pct.value(), panel._hi_pct.value()) == (5.0, 95.0)


# ---------------------------------------------------------------------------
# Grid rendering
# ---------------------------------------------------------------------------


def test_clearing_the_grid_empties_it_even_past_a_spacer(panel):
    """A non-widget item in the crop grid must not strand the crops behind it.

    ``_clear_grid`` runs before every redraw. A layout item that is not a
    widget -- a stretch or spacer -- has no ``widget()`` to delete, and a clear
    that could not step past one would leave the previous pass's thumbnails on
    screen underneath the new ones, so the grid would show crops from two
    different arrays at once.
    """
    label = QLabel("old crop", panel)
    panel._grid.addWidget(label, 0, 0)
    panel._grid.addItem(QSpacerItem(8, 8, QSizePolicy.Expanding), 0, 1)
    assert panel._grid.count() == 2

    panel._clear_grid()
    assert panel._grid.count() == 0
    assert panel._grid.indexOf(label) == -1


def test_a_click_on_a_thumbnail_that_is_gone_is_ignored(panel, tmp_path):
    """A stale thumbnail index must not select or describe a phantom crop.

    The thumbnails carry the index they were built with, and a re-crop can
    shorten the list while an old thumbnail is still on screen. Reading
    ``self._crops[index]`` for one of those indices would raise inside a signal
    handler. It must also stay out of ``current_params``: downstream callers
    cannot use an index beyond the advertised crop count.
    """
    panel.load_array(_merged(tmp_path))
    assert len(panel._crops) == 2

    panel._on_thumb_clicked(0)
    entry = panel._crops[0]
    described = panel._status.text()
    assert described == (
        f"label {entry['label']} · {entry['area']} px² · "
        f"{entry['category']} · 1 selected")

    panel._on_thumb_clicked(99)
    assert 99 not in panel._selected
    assert panel.current_params()["selected"] == [0]
    assert panel._status.text() == described


# ---------------------------------------------------------------------------
# Wiring the Crop-settings controls
# ---------------------------------------------------------------------------


def test_a_control_with_no_signal_does_not_stop_the_wiring_pass(panel):
    """One un-connectable control must not cost every control after it.

    ``_connect_controls`` walks the Crop-settings widgets looking for the first
    signal each one publishes. A widget answering to none of them -- a plain
    label-like control, or a stub standing in for one -- has to be stepped
    over: if the walk stopped or raised there, every setting later in the list
    would be dead, silently, and the dialog would look wired while half of its
    knobs neither re-cropped nor propagated.
    """
    seen = []
    panel.set_propagate_callback(seen.append)
    # One in the re-crop list and one that is only propagated, so both walks
    # meet a control with nothing to connect to.
    panel._max_area = _MuteSpin(panel)
    panel._plot = _MuteToggle(panel)

    panel._normalise.setChecked(False)
    panel._lo_pct.setEnabled(True)
    panel._connect_controls()
    # The gate pass at the very end of the method ran, so the walk completed.
    assert panel._lo_pct.isEnabled() is False

    panel._propagate_btn.setChecked(True)
    before = len(seen)
    panel._max_crops.setValue(7)
    assert len(seen) > before, "a spinbox after the mute control was not wired"
    assert seen[-1]["plot"] is False          # the mute control still read
    assert seen[-1]["png_size"] == [224, 224]
