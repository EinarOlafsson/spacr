"""One real figure per figure-producing area carries its data to the menu.

Each producer below is called as spaCR calls it. The figure it makes must
arrive with its tidy data and plot spec (or, for a picture, its arrays), so
that the right-click menu offers exactly what applies: a plot gets Edit
figure, Change graph type, Statistics and the zip; a picture gets Edit
figure and a zip of the image with its metadata, and nothing else. For
every plot the test changes the graph type, writes the zip and reads the
statistics CSV back.
"""
from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("seaborn")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from spacr.figures import bundle  # noqa: E402

PLOT_ENTRIES = {"Edit figure…", "Change graph type", "Statistics…",
                "Save figure (zip)…"}


@pytest.fixture
def made(monkeypatch):
    """Every figure registered while the test runs, newest last."""
    seen = []
    original = bundle._register_figure_data

    def spy(figure, *args, **kwargs):
        original(figure, *args, **kwargs)
        seen.append(bundle._as_figure(figure))

    monkeypatch.setattr(bundle, "_register_figure_data", spy)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield seen
    plt.close("all")


def _rng():
    return np.random.default_rng(0)


def _regression(tmp_path):
    from spacr.regression_diagnostics import plot_residual_diagnostics

    rng = _rng()
    fitted = rng.normal(size=60)
    plot_residual_diagnostics(fitted + rng.normal(0, 0.3, 60), fitted)


def _plate_map(tmp_path):
    from spacr.figures.plates import build_plates

    rng = _rng()
    rows = [{"prc": f"plate1_r{r}_c{c}", "pred": float(rng.normal())}
            for r in range(1, 5) for c in range(1, 7) for _ in range(3)]
    build_plates(pd.DataFrame(rows), "pred")


def _umap(tmp_path):
    from spacr.utils import plot_embedding

    rng = _rng()
    embedding = np.vstack([rng.normal(0, 1, (30, 2)),
                           rng.normal(5, 1, (30, 2))])
    labels = np.repeat([0, 1], 30)
    plot_embedding(embedding, None, labels, 0, 0.1,
                   [(0.12, 0.47, 0.71), (1.0, 0.5, 0.05)], False, False, True, False,
                   False, False, 4, 5, False, False)


def _training_curves(tmp_path):
    from spacr.deep_spacr import _plot_training_curves

    train = [{"epoch": e, "loss": 1.0 / e, "accuracy": 0.5 + 0.04 * e}
             for e in range(1, 9)]
    val = [{"epoch": e, "loss": 1.2 / e, "accuracy": 0.45 + 0.04 * e}
           for e in range(1, 9)]
    _plot_training_curves(train, val)


def _confusion(tmp_path):
    from spacr.classifier_evaluation import _write_confusion_figure

    names = ["a", "b", "c"]
    frame = pd.DataFrame([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1],
                          [0.1, 0.2, 0.7]], index=names, columns=names)
    _write_confusion_figure(frame, tmp_path / "confusion.png")


def _measure_wound(tmp_path):
    from spacr.measure import _wound_curve_figure

    rows, wells = [], []
    for condition, slope in (("ctrl", 0.05), ("drug", 0.02)):
        for t in range(6):
            rows.append({"condition": condition, "time": float(t),
                         "relative_open_area": max(0.0, 1 - slope * t),
                         "relative_open_area_sd": 0.02, "n_wells": 2,
                         "time_unit": "h"})
            for well in (1, 2):
                wells.append({"condition": condition, "plateID": "p1",
                              "rowID": well, "columnID": 1, "time": float(t),
                              "relative_open_area": max(0.0, 1 - slope * t)})
    _wound_curve_figure(pd.DataFrame(rows), pd.DataFrame(wells))


def _plaque(tmp_path):
    from spacr.plaque import _colony_size_figure

    rng = _rng()
    _colony_size_figure([{"diameter_px": float(v)}
                         for v in rng.gamma(4, 5, 80)])


def _motility(tmp_path):
    from spacr.timelapse import _make_motility_plots

    rng = _rng()
    tracks = [{"x_px": np.cumsum(rng.normal(size=10)),
               "y_px": np.cumsum(rng.normal(size=10)),
               "infected": bool(i % 2)} for i in range(6)]
    table = pd.DataFrame({"track_id": range(6),
                          "infected": [bool(i % 2) for i in range(6)],
                          "velocity": rng.random(6)})
    _make_motility_plots(table, {("p1", "A1"): tracks}, pd.DataFrame(),
                         str(tmp_path), 1.0, 1.0, "px/frame", {})


def _survival(tmp_path):
    from spacr.measure import _time_to_event_figure

    rows = []
    for condition, rate in (("ctrl", 0.1), ("drug", 0.25)):
        survival = 1.0
        for t in range(8):
            survival *= 1 - rate
            rows.append({"condition": condition, "time": float(t),
                         "survival": survival, "ci_lower": survival * 0.9,
                         "ci_upper": min(1.0, survival * 1.1),
                         "censored": int(t == 7), "time_unit": "h"})
    summary = pd.DataFrame({"level": "condition", "group": ["ctrl", "drug"],
                            "n": [20, 20], "events": [10, 15]})
    _time_to_event_figure(pd.DataFrame(rows), summary, pd.DataFrame(),
                          "time to lysis")


def _dose_response(tmp_path):
    from spacr.measure import _viability_dose_figure

    dose = np.logspace(-2, 2, 9)

    def fit(group, ec50):
        response = 1 / (1 + (dose / ec50) ** 1.2)
        result = SimpleNamespace(
            dose=dose, response=response, ec50=ec50,
            curve=lambda: (dose, response))
        return SimpleNamespace(group=group, result=result)

    _viability_dose_figure({"viability": SimpleNamespace(
        fits=[fit("A", 0.5), fit("B", 5.0)])})


def _control_chart(tmp_path):
    from spacr.qt.screens.control_chart import ControlChartCanvas
    from spacr.qt.widgets.control_chart import ControlChartSpec, control_chart

    rng = _rng()
    rows = [{"plateID": f"P{i:02d}", "run_order": i,
             "signal": 100 + float(rng.normal())}
            for i in range(20) for _ in range(3)]
    canvas = ControlChartCanvas()
    canvas.set_result(control_chart(pd.DataFrame(rows), ControlChartSpec(
        value="signal", plate="plateID", order="run_order")))
    return canvas


PLOTS = {
    "regression": _regression,
    "plate map": _plate_map,
    "umap": _umap,
    "training curves": _training_curves,
    "confusion matrix": _confusion,
    "measure": _measure_wound,
    "plaque": _plaque,
    "motility": _motility,
    "survival": _survival,
    "dose-response": _dose_response,
    "control chart": _control_chart,
}


def _visible(menu):
    """The visible entries of ``menu`` by text."""
    return {a.text(): a for a in menu.actions()
            if a.isVisible() and a.text()}


@pytest.mark.parametrize("producer", list(PLOTS.values()), ids=list(PLOTS))
def test_each_plot_carries_its_data_to_the_menu_and_the_zip(
        qapp, made, tmp_path, producer):
    """Data and spec arrive; menu, type change, statistics and zip work."""
    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    keep = producer(tmp_path)
    assert made, "the producer registered no figure"
    figure = made[-1]
    assert not bundle._is_image_figure(figure)
    frame, spec = bundle._figure_record(figure)
    assert isinstance(frame, pd.DataFrame) and len(frame)

    entries = _visible(build_figure_context_menu(None, figure))
    assert PLOT_ENTRIES <= set(entries)
    kinds = [a for a in entries["Change graph type"].menu().actions()
             if a.isEnabled()]
    assert kinds, "no graph type fits the registered data"

    out = bundle._save_zip(figure, str(tmp_path / "figure"),
                           formats=["png"], name="figure")
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        assert {"figure.png", "data.csv", "statistics.csv", "spec.json",
                "recreate_figure.py"} <= names
        archive.extractall(tmp_path / "unzipped")
    assert len(pd.read_csv(tmp_path / "unzipped" / "data.csv")) == len(frame)
    table = pd.read_csv(tmp_path / "unzipped" / "statistics.csv")
    assert {"test_stage", "test_name", "p_value", "chosen_by"} <= set(
        table.columns)

    current = str(spec.get("kind") or "")
    target = next((a for a in kinds if not a.isChecked()), kinds[0])
    target.trigger()
    assert figure._spacr_spec["kind"] != current or len(kinds) == 1
    assert figure.axes, "the redraw left the figure empty"
    del keep


def test_a_figure_queue_figure_keeps_its_data(qapp, made, tmp_path):
    """A figure handed to the queue offers the same plot entries there."""
    from spacr.qt.widgets.figure_queue import FigureQueue
    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    _plaque(tmp_path)
    queue = FigureQueue()
    try:
        index = queue.add_figure(made[-1])
        figure = queue.figure_for(index)
        entries = _visible(build_figure_context_menu(queue, figure))
        assert PLOT_ENTRIES <= set(entries)
        assert bundle._figure_record(figure)[0] is not None
    finally:
        queue.deleteLater()


def test_a_picture_gets_edit_and_an_image_zip_only(qapp, made, tmp_path):
    """Masks: Edit figure and an image zip; no type change, no statistics."""
    from spacr.plot import visualize_masks
    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    masks = [np.zeros((16, 16), dtype=np.uint16) for _ in range(3)]
    for number, mask in enumerate(masks, start=1):
        mask[2:8, 2:8] = number
    visualize_masks(*masks, title="three masks")
    figure = made[-1]
    assert bundle._is_image_figure(figure)

    entries = _visible(build_figure_context_menu(None, figure))
    assert {"Edit figure…", "Save figure (zip)…"} <= set(entries)
    for absent in ("Change graph type", "Statistics…", "Graph type",
                   "Axis scale", "Grid", "Save"):
        assert absent not in entries

    out = bundle._save_zip(figure, str(tmp_path / "masks.zip"),
                           formats=["png"], name="masks")
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        metadata = json.loads(archive.read("metadata.json"))
    assert "masks.png" in names and "metadata.json" in names
    assert not {"data.csv", "statistics.csv"} & names
    assert len(metadata["arrays"]) == 3
    assert metadata["arrays"][0]["shape"] == [16, 16]
    assert {entry["file"] for entry in metadata["arrays"]} <= names


def test_registration_never_breaks_the_figure():
    """A data builder that raises leaves the figure drawn and unregistered."""
    figure = plt.figure()
    figure.add_subplot(111).plot([1, 2])

    def broken():
        raise KeyError("missing column")

    bundle._register_figure_data(figure, broken, x="a", y="b")
    assert bundle._figure_record(figure) == (None, {})
    plt.close(figure)
