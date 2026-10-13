"""187 D2: comprehensive figure settings that change the figure live.

"in save styled there should be comprehensive figure setting that change the
figure live."

TEXT IS THE FIRST CANDIDATE because the report that prompted this was a
figure whose text was wrong for the page. On the pyqtgraph side that was a
device-scale bug, fixed separately. A matplotlib figure has the opposite
problem: resized for a journal column the axes shrink and the labels do not,
so a figure drawn at 10 inches and saved at 3.4 is all text.

SCALED ON THE ARTISTS, NOT THROUGH rcParams, because rcParams only reach an
artist when it is CREATED and the figure being previewed already exists --
the same trap `setup_plot` fell into.
"""
from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr.qt.widgets.save_figure_dialog import style_for_file


@pytest.fixture
def figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("coefficient")
    ax.set_ylabel("-log10(p)")
    ax.set_title("a title")
    yield fig
    plt.close(fig)


class TestTheTextComesDownWithThePage:

    def test_every_text_artist_is_scaled(self, figure):
        axes = figure.axes[0]
        before = (axes.xaxis.label.get_fontsize(),
                  axes.yaxis.label.get_fontsize(),
                  axes.title.get_fontsize())

        style_for_file(figure, font_scale=0.5)

        after = (axes.xaxis.label.get_fontsize(),
                 axes.yaxis.label.get_fontsize(),
                 axes.title.get_fontsize())
        assert all(b / 2 == pytest.approx(a) for b, a in zip(before, after))

    def test_the_ticks_come_too(self, figure):
        axes = figure.axes[0]
        figure.canvas.draw()
        before = [label.get_fontsize() for label in axes.get_xticklabels()]

        style_for_file(figure, font_scale=2.0)

        after = [label.get_fontsize() for label in axes.get_xticklabels()]
        assert before and all(b * 2 == pytest.approx(a)
                              for b, a in zip(before, after))

    def test_a_scale_of_one_changes_nothing(self, figure):
        axes = figure.axes[0]
        before = axes.title.get_fontsize()

        style_for_file(figure, font_scale=1.0)

        assert axes.title.get_fontsize() == before

    def test_no_scale_given_changes_nothing(self, figure):
        axes = figure.axes[0]
        before = axes.title.get_fontsize()

        style_for_file(figure)

        assert axes.title.get_fontsize() == before

    def test_it_composes_with_the_page_size(self, figure):
        """The two together are the point: a smaller page with smaller type."""
        style_for_file(figure, width=3.4, height=2.6, font_scale=0.5)

        assert tuple(figure.get_size_inches()) == pytest.approx((3.4, 2.6))
        assert figure.axes[0].title.get_fontsize() < 12


class TestTheControlIsLive:

    @pytest.fixture
    def dialog(self, qtbot, figure):
        pytest.importorskip("PySide6")
        from spacr.qt.widgets.save_figure_dialog import SaveFigureDialog

        widget = SaveFigureDialog(figure)
        qtbot.addWidget(widget)
        return widget

    def test_the_control_is_on_the_dialog(self, dialog):
        assert dialog.font_scale.value() == pytest.approx(1.0)

    def test_changing_it_redraws_the_preview(self, dialog):
        """Every control here redraws; that is what makes it a preview
        rather than a form."""
        dialog.font_scale.setValue(0.5)

        preview = dialog._preview
        assert preview is not None
        assert preview.axes[0].title.get_fontsize() < \
            dialog._source.axes[0].title.get_fontsize()

    def test_the_source_figure_is_never_touched(self, dialog, figure):
        """The preview is a COPY -- a dialog that shrank the user's on-screen
        figure while they looked at options would be worse than no preview."""
        before = figure.axes[0].title.get_fontsize()

        dialog.font_scale.setValue(2.0)

        assert figure.axes[0].title.get_fontsize() == before

    def test_what_is_written_is_what_was_previewed(self, dialog, tmp_path):
        dialog.font_scale.setValue(0.5)
        previewed = dialog._preview.axes[0].title.get_fontsize()

        dialog.save(str(tmp_path / "figure.png"))

        assert (tmp_path / "figure.png").exists()
        assert dialog._preview.axes[0].title.get_fontsize() == previewed


class TestUntickingTheGridTurnsItOff:
    """Found by a warning while testing something else, and it was real.

    `axes.grid(False, linewidth=..., alpha=...)` does not turn the grid off.
    matplotlib says so out loud -- "First parameter to grid() is false, but
    line properties are supplied. The grid will be enabled." -- and then
    enables it. So the one control that says "draw a grid" drew one whichever
    way it was set.
    """

    def _grid_is_on(self, figure):
        axes = figure.axes[0]
        return bool(axes.xaxis._major_tick_kw.get("gridOn"))

    def test_off_is_off(self, figure):
        style_for_file(figure, grid=False)

        assert not self._grid_is_on(figure)

    def test_on_is_on(self, figure):
        style_for_file(figure, grid=True)

        assert self._grid_is_on(figure)

    def test_it_can_be_turned_off_again(self, figure):
        """The path that matters: a user ticks it, looks, and unticks it."""
        style_for_file(figure, grid=True)
        style_for_file(figure, grid=False)

        assert not self._grid_is_on(figure)
