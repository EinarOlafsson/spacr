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

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QLabel, QSizePolicy, QSpacerItem

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
    # WHAT WAS COPIED IS THE PANEL'S OWN ANSWER, checked against the panel
    # rather than against a literal. The experiment field ships Measure's own
    # default ("experiment"); "exp" is only the fallback for a field the user
    # has emptied, and asserting the fallback here tested nothing about
    # propagation.
    assert seen[0] == panel.settings_for_propagation()

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


def test_a_new_cap_with_no_sentence_to_show_keeps_the_status_line(
        panel, tmp_path):
    """A resample must only overwrite the status line when it has something to say.

    The cap spinbox reports a real sample on the status line. A fresh sampler's
    technically truthful ``showing all 0 image sets`` is not a useful update:
    it must not erase the crop count or the reason the last load failed.
    """
    panel._status.setText("2 object(s) · 1 category")
    panel._on_max_sets_changed(7)
    assert panel.sample_note() == "showing all 0 image sets"
    assert panel._status.text() == "2 object(s) · 1 category"

    # With a real sample behind it the same call does report, capitalised.
    panel.load_array(_merged_folder(tmp_path)[0])
    panel._on_max_sets_changed(2)
    assert panel.sample_note().startswith("showing ")
    assert panel._status.text() == (
        panel.sample_note()[:1].upper() + panel.sample_note()[1:])
    assert panel._status.text()[0].isupper()


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
        f"{Path(entry['source_path']).name} · "
        f"label {entry['label']} · {entry['area']} px² · "
        f"{entry['category']} · 1 selected")

    panel._on_thumb_clicked(99)
    assert 99 not in panel._selected
    assert panel.current_params()["selected"] == [0]
    assert panel._status.text() == described
