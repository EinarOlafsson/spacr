"""The figure font size reaches all of the text, and outlives the redraw.

GitHub issue #108's SECOND bug -- the one the title does not name. The
reporter wrote:

    "the font size in the figure previews does not work as expected. Font
     size is by default to large to be visible. Adjusting font size using
     the 'Figure settings...' button from 10 to 2, does not reduce the font
     size, in fact increases it, and when returning to the 'Figure
     settings...' button menu the font size has been returned to 10."

Three symptoms, and each had its own cause. All three were still live on
2026-08-18 and each is reproduced here before it is asserted fixed:

1.  "BY DEFAULT TOO LARGE".  ``_FigureSettingsDialog`` seeded its spin box
    from ``get_figure_text_size() or 10`` -- a RESOLVED 10 standing in for a
    stored 0, which means "leave every figure the sizes it was drawn with" --
    and ``_apply_and_accept`` wrote the shown number back. So pressing OK
    once, even to change a colour, froze 10 into every figure that user would
    ever draw. That is instruction 152 section A's bug exactly, in the size
    key rather than the colour keys: NEVER PERSIST A RESOLVED DEFAULT.

2.  "10 -> 2 INCREASES IT".  ``_style_figure_colors`` -- the pass EVERY
    render goes through -- built its own list of text objects, and that list
    missed an annotation, the suptitle and a legend's title. On a volcano the
    annotations are the gene labels and are the LARGEST text on the plot, so
    shrinking "all text" shrank everything except the biggest thing on the
    figure, which then dominated it.

3.  "RETURNED TO 10".  The dialog's own control set the sizes on the artists
    and nowhere else, so the next full render re-applied the global
    preference over the top; and the control reads its opening value off the
    figure, so it then showed the preference back.

IT REACHES THE FILE, NOT ONLY THE PREVIEW. ``render_figure_to_png`` calls
``set_fontsize`` on the Figure's own artists, so whatever it decides is what
the sibling vector page written in the same call gets, and what a later
"Save figure as..." writes. Measured on the rendered PNG, ink composited onto
white so a transparent page cannot be counted as text:

    at the figure's own sizes   36,920 dark px in 1144x972
    after asking for 2pt        17,536 dark px in 1058x906
    the kept file, save_figure_as 19,470 dark px in 1058x906

so this is a preference that never reached all of the text, not display
scaling.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from PySide6.QtWidgets import QFormLayout

from spacr.qt.widgets import figure_queue as fq
from spacr.qt.widgets import figure_settings as fs


# --------------------------------------------------------------------------- #
#  fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def prefs(monkeypatch, tmp_path):
    """The real preference module, writing to a throwaway ini file."""
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences as preferences_module

    store = tmp_path / "prefs.ini"
    monkeypatch.setattr(
        preferences_module, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    return preferences_module


@pytest.fixture
def volcano():
    """The shape that produced the report: labelled hits, suptitle, legend.

    The 22pt annotations are not decoration -- they are the artists the old
    list missed, and they are what "in fact increases it" was about.
    """
    figure, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [1, 0], label="x")
    ax.set_xlabel("coefficient")
    ax.set_ylabel("-log10(p)")
    ax.set_title("volcano")
    figure.suptitle("a run")
    ax.annotate("EAF1", (0.5, 0.5), fontsize=22)
    ax.annotate("TSG101", (0.3, 0.7), fontsize=22)
    ax.legend(title="condition")
    yield figure
    plt.close(figure)


def _sizes(figure):
    return {round(float(t.get_fontsize()), 1)
            for t in fq.figure_text_items(figure)
            if str(t.get_text()).strip()}


def _all_text_box(dialog):
    """The dialog's "All text size" spin box, found the way a user finds it."""
    form = dialog.tabs.widget(0).widget().layout()
    for row in range(form.rowCount()):
        label = form.itemAt(row, QFormLayout.LabelRole)
        if (label is not None and label.widget() is not None
                and label.widget().text() == "All text size"):
            return form.itemAt(row, QFormLayout.FieldRole).widget()
    raise AssertionError("the Figure tab has no 'All text size' control")


# --------------------------------------------------------------------------- #
#  symptom 2 -- the render pass missed the biggest text on the plot
# --------------------------------------------------------------------------- #

def test_the_render_pass_reaches_the_annotation(prefs, volcano, tmp_path):
    """THE ONE THAT READ AS "IT GOT BIGGER". Before the fix the gene labels
    stayed at 22 while everything around them went to 2."""
    prefs.set_figure_text_size(2)
    fq.render_figure_to_png(volcano, str(tmp_path / "f.png"))
    labels = [t for t in fq.figure_text_items(volcano)
              if t.get_text() in ("EAF1", "TSG101")]
    assert labels, "the fixture lost its annotations"
    assert [t.get_fontsize() for t in labels] == [2.0, 2.0]


def test_the_render_pass_reaches_the_suptitle(prefs, volcano, tmp_path):
    prefs.set_figure_text_size(2)
    fq.render_figure_to_png(volcano, str(tmp_path / "f.png"))
    assert volcano._suptitle.get_fontsize() == 2.0


def test_the_render_pass_reaches_the_legend_title(prefs, volcano, tmp_path):
    prefs.set_figure_text_size(2)
    fq.render_figure_to_png(volcano, str(tmp_path / "f.png"))
    legend = volcano.axes[0].get_legend()
    assert legend.get_title().get_fontsize() == 2.0


def test_nothing_on_the_figure_is_left_behind(prefs, volcano, tmp_path):
    """The whole point: ONE size afterwards, not one plus three stragglers."""
    prefs.set_figure_text_size(7)
    fq.render_figure_to_png(volcano, str(tmp_path / "f.png"))
    assert _sizes(volcano) == {7.0}


def test_the_render_pass_and_the_dialog_share_one_list(volcano):
    """They drifted once and that drift IS the bug, so it is asserted rather
    than trusted: `_every_text` is `figure_text_items`, not a copy of it."""
    assert ([id(t) for t in fs._every_text(volcano)]
            == [id(t) for t in fq.figure_text_items(volcano)])


def test_zero_leaves_the_figure_the_sizes_it_was_drawn_with(prefs, volcano,
                                                            tmp_path):
    """0 is "automatic". A pass that resized on 0 would be the "by default
    too large" half arriving from the other direction."""
    before = _sizes(volcano)
    assert prefs.get_figure_text_size() == 0
    fq.render_figure_to_png(volcano, str(tmp_path / "f.png"))
    assert _sizes(volcano) == before


def test_the_theme_reaches_the_suptitle_and_the_annotation(volcano):
    """The same list carried the COLOUR, so a dark theme left a black
    suptitle on a black page and a black gene label on top of it."""
    fq._style_figure_colors(volcano, "#000000", "#ffffff", 0, "#ffffff")
    assert volcano._suptitle.get_color() == "#ffffff"
    labels = [t for t in volcano.axes[0].texts]
    assert {t.get_color() for t in labels} == {"#ffffff"}


# --------------------------------------------------------------------------- #
#  symptom 3 -- the user's size did not survive the next render
# --------------------------------------------------------------------------- #

def test_the_size_the_user_set_survives_a_full_render(prefs, volcano,
                                                      tmp_path, qtbot):
    prefs.set_figure_text_size(10)
    dialog = fs.FigureSettingsDialog(volcano)
    qtbot.addWidget(dialog)
    _all_text_box(dialog).setValue(2)
    fq.render_figure_to_png(volcano, str(tmp_path / "f.png"))
    assert _sizes(volcano) == {2.0}


def test_reopening_the_dialog_shows_what_the_user_set(prefs, volcano,
                                                      tmp_path, qtbot):
    """The reporter's own sentence, driven: set 2, redraw, open it again."""
    prefs.set_figure_text_size(10)
    first = fs.FigureSettingsDialog(volcano)
    qtbot.addWidget(first)
    _all_text_box(first).setValue(2)
    fq.render_figure_to_png(volcano, str(tmp_path / "f.png"))
    second = fs.FigureSettingsDialog(volcano)
    qtbot.addWidget(second)
    assert _all_text_box(second).value() == 2


def test_the_control_records_the_size_on_the_figure(volcano, qtbot):
    dialog = fs.FigureSettingsDialog(volcano)
    qtbot.addWidget(dialog)
    assert fq.figure_text_size_override(volcano) == 0
    _all_text_box(dialog).setValue(14)
    assert fq.figure_text_size_override(volcano) == 14


def test_opening_the_dialog_does_not_record_anything(volcano, qtbot):
    """Seeding is not choosing. The spin box is filled from the figure, and
    a control that wrote its seed back is symptom 1 in another costume."""
    qtbot.addWidget(fs.FigureSettingsDialog(volcano))
    assert fq.figure_text_size_override(volcano) == 0


def test_cancel_puts_the_per_figure_size_back(volcano, qtbot):
    """`reject` restores the figure by copying axes out of a pickle, which
    an attribute on the Figure would have outlived."""
    fq.set_figure_text_size_override(volcano, 9)
    dialog = fs.FigureSettingsDialog(volcano)
    qtbot.addWidget(dialog)
    _all_text_box(dialog).setValue(3)
    dialog.reject()
    assert fq.figure_text_size_override(volcano) == 9


def test_a_per_figure_size_beats_the_global_default(prefs, volcano, tmp_path):
    prefs.set_figure_text_size(18)
    fq.set_figure_text_size_override(volcano, 5)
    fq.render_figure_to_png(volcano, str(tmp_path / "f.png"))
    assert _sizes(volcano) == {5.0}


def test_an_override_survives_the_spill_that_evicts_a_figure(volcano):
    """A figure past the live window is pickled to the temp directory and
    read back. A per-figure size kept in a side table would not come with
    it, and the user's restyle would vanish when the figure left RAM."""
    import pickle

    fq.set_figure_text_size_override(volcano, 6)
    revived = pickle.loads(pickle.dumps(volcano))
    try:
        assert fq.figure_text_size_override(revived) == 6
    finally:
        plt.close(revived)


def test_clearing_the_override_returns_to_the_global(prefs, volcano, tmp_path):
    fq.set_figure_text_size_override(volcano, 5)
    fq.set_figure_text_size_override(volcano, 0)
    prefs.set_figure_text_size(18)
    fq.render_figure_to_png(volcano, str(tmp_path / "f.png"))
    assert _sizes(volcano) == {18.0}


# --------------------------------------------------------------------------- #
#  symptom 1 -- the resolved seed written back
# --------------------------------------------------------------------------- #

def test_ok_without_touching_the_box_leaves_the_store_automatic(prefs,
                                                                volcano,
                                                                qtbot):
    """Pressing OK to change a colour must not freeze 10 into every figure
    this user will ever draw."""
    assert prefs.get_figure_text_size() == 0
    dialog = fq._FigureSettingsDialog(volcano)
    qtbot.addWidget(dialog)
    assert dialog._size.value() == 10, "the box still SHOWS a resolved 10"
    dialog._apply_and_accept()
    assert prefs.get_figure_text_size() == 0


def test_ok_without_touching_the_box_keeps_a_size_already_chosen(prefs,
                                                                 volcano,
                                                                 qtbot):
    """The other direction: not writing must not un-write either."""
    prefs.set_figure_text_size(16)
    dialog = fq._FigureSettingsDialog(volcano)
    qtbot.addWidget(dialog)
    assert dialog._size.value() == 16
    dialog._apply_and_accept()
    assert prefs.get_figure_text_size() == 16


def test_moving_the_box_writes_the_preference(prefs, volcano, qtbot):
    dialog = fq._FigureSettingsDialog(volcano)
    qtbot.addWidget(dialog)
    dialog._size.setValue(24)
    dialog._apply_and_accept()
    assert prefs.get_figure_text_size() == 24


def test_the_global_size_clears_a_per_figure_one(prefs, volcano, qtbot,
                                                 tmp_path):
    """A stale override would make the figure in front of the user the one
    figure that ignored the size they just set for every figure."""
    fq.set_figure_text_size_override(volcano, 5)
    dialog = fq._FigureSettingsDialog(volcano)
    qtbot.addWidget(dialog)
    dialog._size.setValue(24)
    dialog._apply_and_accept()
    assert fq.figure_text_size_override(volcano) == 0
    fq.render_figure_to_png(volcano, str(tmp_path / "f.png"))
    assert _sizes(volcano) == {24.0}


def test_the_untouched_dialog_leaves_a_per_figure_size_alone(prefs, volcano,
                                                            qtbot):
    fq.set_figure_text_size_override(volcano, 5)
    dialog = fq._FigureSettingsDialog(volcano)
    qtbot.addWidget(dialog)
    dialog._apply_and_accept()
    assert fq.figure_text_size_override(volcano) == 5
