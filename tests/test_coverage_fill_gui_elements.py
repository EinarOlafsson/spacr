"""Coverage-fill for spacr.gui_elements matplotlib figure helpers."""
from __future__ import annotations

import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import gui_elements as GE


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _fig_with_line():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4], label="series")
    ax.set_title("t"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend()
    return fig


# ---------------------------------------------------------------------------
# modify_figure_properties
# ---------------------------------------------------------------------------

def test_modify_figure_properties_full():
    fig = _fig_with_line()
    GE.modify_figure_properties(
        fig, scale_x=1.2, scale_y=0.8, line_width=2, font_size=14,
        x_lim=(0, 3), y_lim=(0, 5), grid=True, title="new",
        x_label_rotation=45, remove_axes=False,
        bg_color="white", text_color="black", line_color="red")
    ax = fig.get_axes()[0]
    assert ax.get_xlim() == (0.0, 3.0)


def test_modify_figure_properties_remove_axes():
    fig = _fig_with_line()
    GE.modify_figure_properties(fig, remove_axes=True)
    ax = fig.get_axes()[0]
    assert ax.xaxis.get_visible() is False


def test_modify_figure_properties_none():
    # None figure → prints error, no raise
    GE.modify_figure_properties(None)


# ---------------------------------------------------------------------------
# save_figure_as_format (filedialog mocked)
# ---------------------------------------------------------------------------

def test_save_figure_as_format(tmp_path, monkeypatch):
    out = tmp_path / "fig.png"
    monkeypatch.setattr(GE.filedialog, "asksaveasfilename",
                        lambda **kw: str(out))
    GE.save_figure_as_format(_fig_with_line(), "png")
    assert out.exists()


def test_save_figure_as_format_cancelled(monkeypatch):
    # user cancels the dialog → returns "" → no save, no raise
    monkeypatch.setattr(GE.filedialog, "asksaveasfilename",
                        lambda **kw: "")
    GE.save_figure_as_format(_fig_with_line(), "pdf")
