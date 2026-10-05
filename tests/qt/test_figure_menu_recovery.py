"""Figure menu actions keep working when data or optional services disappear."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from matplotlib.figure import Figure  # noqa: E402

from spacr.figures import bundle, style  # noqa: E402
from spacr.qt.widgets import figure_settings as fs  # noqa: E402

pytestmark = pytest.mark.qt


def test_a_failed_redraw_notification_does_not_break_a_figure_action(
        monkeypatch):
    figure = Figure()
    called = []
    monkeypatch.setattr(figure.canvas, "draw_idle",
                        lambda: called.append("draw"))
    fs._redraw_after(figure, None)
    assert called == ["draw"]

    def refusing_draw():
        raise RuntimeError("canvas already closed")

    monkeypatch.setattr(figure.canvas, "draw_idle", refusing_draw)
    fs._redraw_after(figure, None)

    legacy = []
    fs._redraw_after(figure, lambda: legacy.append("changed"))
    assert legacy == ["changed"]
    fs._redraw_after(figure, lambda: refusing_draw())
    fs._redraw_after(figure, lambda **_kwargs: refusing_draw())


def test_retyping_without_data_or_with_an_unsupported_kind_keeps_the_recipe(
        monkeypatch):
    figure = Figure()
    assert not fs._retype(figure, "violin")

    frame = pd.DataFrame({"group": ["a", "b"], "value": [1.0, 2.0]})
    bundle._register_figure_data(figure, frame, x="group", y="value",
                                 kind="bar")
    original = dict(figure._spacr_spec)
    original_draw = bundle._draw
    monkeypatch.setattr(bundle, "_draw",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            ValueError("the data cannot be drawn")))
    assert not fs._retype(figure, "violin")
    assert figure._spacr_spec == original
    monkeypatch.setattr(bundle, "_draw", original_draw)

    monkeypatch.setattr(style, "_apply_user_style",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("style unavailable")))
    changed = []
    assert fs._retype(figure, "bar", lambda **kwargs: changed.append(kwargs))
    assert figure._spacr_spec["kind"] == "bar"
    assert changed == [{"preview": False}]


def test_statistics_ignore_unusable_probabilities_but_keep_valid_pairs():
    table = pd.DataFrame([
        {"test_stage": "pairwise", "test_name": "invalid",
         "groups": "a vs b", "correction": "", "p_adjusted": float("nan"),
         "p_value": float("nan")},
        {"test_stage": "pairwise", "test_name": "Welch",
         "groups": "a vs b", "correction": "", "p_adjusted": float("nan"),
         "p_value": 0.001},
        {"test_stage": "pairwise", "test_name": "Welch",
         "groups": "all", "correction": "", "p_adjusted": 0.001,
         "p_value": 0.001},
    ])
    note, brackets = fs._annotations_from(table)
    assert note == "Welch: p = 0.001"
    assert brackets == [{"pair": ["a", "b"], "label": "**"}]


def test_a_replacement_recipe_redraws_the_canvas_without_swapping_it(
        monkeypatch):
    figure = Figure()
    figure._spacr_data = object()
    redrawn = []
    canvas = SimpleNamespace(figure=figure,
                             draw_idle=lambda: redrawn.append("draw"))
    monkeypatch.setattr(fs, "_retype",
                        lambda fig, kind, on_change: redrawn.append(kind))
    replacement = SimpleNamespace(
        _spacr_replot={"graph_type": "jitter_box"})

    fs._canvas_changed(canvas, replacement)

    assert figure._spacr_replot == replacement._spacr_replot
    assert figure._spacr_data is None
    assert redrawn == ["box_strip", "draw"]

    canvas.draw_idle = lambda: (_ for _ in ()).throw(
        RuntimeError("canvas already deleted"))
    fs._canvas_changed(canvas)


def test_local_figure_writer_survives_missing_preferences_and_failed_write(
        tmp_path, monkeypatch):
    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    monkeypatch.setitem(sys.modules, "spacr.plot", None)
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", None)
    destination = tmp_path / "fallback.svg"

    assert fs.save_figure_as(None, figure, str(destination)) == str(destination)
    assert destination.exists() and destination.stat().st_size > 0

    blocked = tmp_path / "blocker"
    blocked.write_text("a file")
    assert fs.save_figure_as(None, figure, str(blocked / "bad.svg")) == ""
