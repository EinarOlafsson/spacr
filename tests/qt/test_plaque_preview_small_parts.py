"""Plaque preview's small parts: colours, stored style, pickers, the object view.

Each is reached by the full preview only on a path an ordinary run does not
take: an unreadable colour, a stored style that is not JSON, a zoo that
cannot be listed, a file dialog the user cancels.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtGui import QColor

from spacr.qt.widgets import plaque_preview as pp


def test_overlay_colour_reads_every_form_and_refuses_the_rest():
    assert pp.overlay_colour(" Random ") == pp.RANDOM_COLOUR
    assert pp.overlay_colour("#ff8000") == (255, 128, 0)
    assert pp.overlay_colour(QColor(1, 2, 3)) == (1, 2, 3)
    assert pp.overlay_colour("not a colour", (9, 9, 9)) == (9, 9, 9)
    assert pp.overlay_colour(QColor(), (8, 8, 8)) == (8, 8, 8)
    assert pp.overlay_colour([300, -4, 7]) == (255, 0, 7)
    assert pp.overlay_colour(["a", "b", "c"], (7, 7, 7)) == (7, 7, 7)
    assert pp.overlay_colour(None, (6, 6, 6)) == (6, 6, 6)


class _Store:
    def __init__(self, value):
        self._value = value

    def value(self, key, default=None):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


@pytest.mark.parametrize("stored", ["", "{not json", OSError("locked")])
def test_a_stored_style_that_cannot_be_read_is_the_default(
        monkeypatch, stored):
    monkeypatch.setattr(pp, "_preferences", lambda: _Store(stored))
    assert pp.load_overlay_style() == pp.OverlayStyle()


def test_a_stored_style_is_read_back(monkeypatch):
    style = pp.OverlayStyle()
    import json
    monkeypatch.setattr(pp, "_preferences",
                        lambda: _Store(json.dumps(style.as_dict())))
    assert pp.load_overlay_style() == style


def test_a_zoo_that_cannot_be_listed_leaves_the_defaults(monkeypatch):
    from spacr import model_zoo

    def _broken():
        raise OSError("catalogue unreadable")

    monkeypatch.setattr(model_zoo, "catalogue", _broken)
    assert pp._catalogue() == []
    assert pp.plaque_model_choices() == [pp.DEFAULT_PLAQUE_MODEL, "bundled"]
    assert pp.detector_choices() == [pp.DEFAULT_DETECTOR]


def test_the_object_view_draws_only_a_two_dimensional_label_image():
    assert pp.render_objects(None) is None
    assert pp.render_objects(np.zeros((2, 3, 3), np.int32)) is None
    labels = np.zeros((5, 5), np.uint16)
    labels[1:3, 1:3] = 4
    picture = pp.render_objects(labels)
    assert picture.shape == (5, 5, 3)
    assert picture[1, 1].any() and not picture[4, 4].any()


def test_the_paper_dialog_keeps_what_the_pickers_return(qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    dialog = pp.PaperDialog(folder="/data")
    qtbot.addWidget(dialog)

    answers = iter([("", ""), ("/papers/smith.pdf", "PDF files (*.pdf)")])
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: next(answers)))
    dialog._pick_pdf()
    assert dialog.reference.text() == ""
    dialog._pick_pdf()
    assert dialog.reference.text() == "/papers/smith.pdf"

    folders = iter(["", "/elsewhere"])
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: next(folders)))
    dialog._pick_folder()
    assert dialog.folder.text() == "/data"
    dialog._pick_folder()
    assert dialog.folder.text() == "/elsewhere"


def test_the_colour_swatch_keeps_only_a_real_colour(qtbot, monkeypatch):
    from spacr.qt.widgets import colour_picker

    choice = pp._ColourChoice((10, 20, 30))
    qtbot.addWidget(choice)
    seen = []
    choice.changed.connect(seen.append)

    monkeypatch.setattr(colour_picker, "pick_colour",
                        lambda *a, **k: QColor())
    choice._choose()
    assert choice.value() == (10, 20, 30)
    assert seen == []

    monkeypatch.setattr(colour_picker, "pick_colour",
                        lambda *a, **k: QColor(200, 100, 50))
    choice._choose()
    assert choice.value() == (200, 100, 50)
    assert seen[-1] == (200, 100, 50)


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setitem(pp._SESSION, "style", pp.OverlayStyle())
    monkeypatch.setattr(pp, "missing_papers_packages", lambda *a, **k: [])
    widget = pp.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_random_colour_toggles_back_to_the_last_fixed_one(panel):
    from dataclasses import replace

    panel.set_overlay_style(replace(panel.overlay_style(),
                                    display=pp.OVERLAY_FILL))
    fixed = panel._fixed_colours["fill"]
    assert fixed != pp.RANDOM_COLOUR

    panel._set_random(True)
    assert panel.overlay_style().fill_colour == pp.RANDOM_COLOUR
    panel._set_random(False)
    assert panel.overlay_style().fill_colour == fixed

    panel.set_overlay_style(replace(panel.overlay_style(),
                                    display=pp.OVERLAY_OUTLINES))
    panel._set_random(True)
    assert panel.overlay_style().outline_colour == pp.RANDOM_COLOUR
    assert panel.overlay_style().fill_colour == fixed


def test_legends_and_the_figure_folder_follow_the_source(panel, tmp_path):
    from spacr.plaque_papers import LEGENDS_FILE

    panel._src = ""
    assert panel._legend_for("fig1") == ""
    assert panel._folder() is None

    (tmp_path / LEGENDS_FILE).write_text(
        "file,legend\nfig1.png,Plaques at 7 days\n")
    panel._src = str(tmp_path)
    assert panel._legend_for("fig1") == "Plaques at 7 days"
    assert panel._legend_for("fig2") == ""
    assert panel._folder() == tmp_path

    panel._src = str(tmp_path / "fig1.png")
    assert panel._legend_for("fig1") == "Plaques at 7 days"
    assert panel._folder() == tmp_path
