"""Saving a figure and showing it are the same event.

Instruction 139 C, reported 2026-08-18: "all graphs should be sable and
observable in the software, currently several graphs are saved but I cannot
see them in the software".

THE CAUSE. A figure reached the GUI by one route -- `spacr/qt/bridge.py`
replaces `matplotlib.pyplot.show` and emits everything in
`plt.get_fignums()`. So a figure was visible if and only if it was IN pyplot's
registry AND somebody called `show`.

`spacr.regression_qc` fails both halves, which is why its ~19-panel report was
invisible: it builds bare `matplotlib.figure.Figure` objects -- the correct
thing for a library to do -- and writes them with savefig. Every panel on
disk, none in the application.
"""
from __future__ import annotations

import os

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from spacr import figure_sink  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sink_left_behind():
    """A sink is global; a test that leaks one breaks the next."""
    figure_sink.clear_sink()
    yield
    figure_sink.clear_sink()


def _figure():
    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    return fig


def test_a_published_figure_is_written_and_announced(tmp_path):
    seen = []
    figure_sink.set_sink(lambda fig, path: seen.append(path))

    written = figure_sink.publish(_figure(), str(tmp_path / "panel.pdf"))

    assert written and os.path.isfile(written)
    assert seen == [written]


def test_a_figure_pyplot_never_saw_is_still_announced(tmp_path):
    """The whole point. `plt.get_fignums()` is empty and it still arrives."""
    import matplotlib.pyplot as plt

    plt.close("all")
    seen = []
    figure_sink.set_sink(lambda fig, path: seen.append(path))

    figure_sink.publish(_figure(), str(tmp_path / "panel.pdf"))

    assert plt.get_fignums() == [], "the fixture figure must not be in pyplot"
    assert len(seen) == 1


def test_headless_still_writes_the_file(tmp_path):
    """`spacr-run` and a notebook install no sink. The run's output must not
    depend on a GUI being attached."""
    assert figure_sink.sink() is None
    written = figure_sink.publish(_figure(), str(tmp_path / "panel.pdf"))
    assert written and os.path.isfile(written)


def test_a_sink_that_raises_does_not_lose_the_file(tmp_path):
    """A GUI that has gone away must not take the run's output with it."""
    def angry(fig, path):
        raise RuntimeError("the window is gone")

    figure_sink.set_sink(angry)
    written = figure_sink.publish(_figure(), str(tmp_path / "panel.pdf"))
    assert written and os.path.isfile(written)


def test_the_save_honours_the_figure_format_preference(tmp_path, monkeypatch):
    """Through `spacr.plot.save_figure`, not a literal extension -- a
    complaint this project has already had twice."""
    from spacr import plot

    calls = []
    real = plot.save_figure
    monkeypatch.setattr(plot, "save_figure",
                        lambda *a, **k: calls.append((a, k)) or real(*a, **k))

    figure_sink.publish(_figure(), str(tmp_path / "panel.pdf"))
    assert calls, "publish did not go through save_figure"


def test_publish_without_a_path_announces_and_writes_nothing(tmp_path):
    seen = []
    figure_sink.set_sink(lambda fig, path: seen.append(path))
    assert figure_sink.publish(_figure()) is None
    assert seen == [None]


def test_close_happens_after_the_sink_has_had_the_figure(tmp_path):
    """A cleared figure has nothing left to render."""
    axes_seen = []
    figure_sink.set_sink(lambda fig, path: axes_seen.append(len(fig.axes)))

    figure_sink.publish(_figure(), str(tmp_path / "p.pdf"), close=True)

    assert axes_seen == [1], "the sink got an already-cleared figure"


def test_set_sink_hands_back_the_previous_one():
    first, second = (lambda *a: None), (lambda *a: None)
    assert figure_sink.set_sink(first) is None
    assert figure_sink.set_sink(second) is first
    assert figure_sink.sink() is second


def test_the_qc_report_goes_through_the_sink(tmp_path):
    """The suite the report was about. `_save` is its one funnel."""
    import inspect

    from spacr import regression_qc

    source = inspect.getsource(regression_qc._save)
    assert "publish(fig, path" in source
    # THE CALL, not the word. The docstring explains what it replaced, so a
    # bare substring match finds its own prose and passes on a function that
    # still writes directly.
    assert "fig.savefig(" not in source, (
        "the QC panels still write directly, so they are still invisible")


def test_the_bridge_installs_and_clears_the_sink():
    """A sink left installed after a run holds the worker alive and emits
    into a dead signal on the next one."""
    import inspect

    from spacr.qt import bridge

    source = inspect.getsource(bridge)
    assert "set_sink(_publish_figure)" in source
    assert "clear_sink()" in source


def test_a_figure_that_cannot_be_cleared_is_not_an_error(tmp_path):
    """`close=True` is best-effort. A panel that parked something odd on its
    figure must not take down the run that was only trying to save it."""
    class Awkward(Figure):
        def clf(self, *args, **kwargs):
            raise RuntimeError("this figure refuses to be cleared")

    fig = Awkward()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    written = figure_sink.publish(fig, str(tmp_path / "p.pdf"), close=True)
    assert written and os.path.isfile(written)


def test_a_figure_that_was_never_drawn_is_not_published(tmp_path):
    """``fig=None`` writes nothing and announces nothing.

    `ml_analysis` returns ``feature_importance_fig = None`` for every model
    without ``feature_importances_`` (logistic regression,
    HistGradientBoostingClassifier), and `generate_ml_scores` handed it
    straight to ``savefig`` -- an AttributeError that killed the run AFTER the
    model had been fitted and every object scored.
    """
    seen = []
    figure_sink.set_sink(lambda fig, path: seen.append(path))
    assert figure_sink.publish(None, str(tmp_path / "never_drawn.pdf")) is None
    assert seen == [], "a figure that does not exist was announced"
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# The acceptance for instruction 139 C: the gallery and the run folder hold
# the SAME figures. Counted, on a real fit, through the real report driver.
# --------------------------------------------------------------------------

def _small_ols():
    """A tiny but genuine OLS fit the QC suite can describe."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    sm = pytest.importorskip("statsmodels.api")

    rng = np.random.default_rng(11)
    n = 90
    frame = pd.DataFrame({
        "gene_a": rng.normal(0, 1, n),
        "gene_b": rng.normal(0, 1, n),
        "gene_c": rng.normal(0, 1, n),
    })
    y = pd.Series(0.7 * frame["gene_a"] - 0.4 * frame["gene_b"]
                  + rng.normal(0, 0.5, n), name="fraction")
    design = sm.add_constant(frame)
    model = sm.OLS(y, design).fit()
    metadata = pd.DataFrame({
        "plateID": [f"plate{i % 3 + 1}" for i in range(n)],
        "rowID": [f"r{i % 8 + 1}" for i in range(n)],
        "columnID": [f"c{i % 12 + 1}" for i in range(n)],
        "prc": [f"p{i}" for i in range(n)],
        "cell_count": rng.integers(30, 400, n),
    }, index=design.index)
    return model, design, y, metadata


_IMAGE_SUFFIXES = {".png", ".pdf"}


def _image_files(folder):
    return sorted(name for name in os.listdir(folder)
                  if os.path.splitext(name)[1].lower() in _IMAGE_SUFFIXES)


def _opens(path):
    """Does this file actually open as the picture its extension claims?

    A PNG written to a ``.pdf`` name is the second half of the reported bug --
    it is on disk, it is in the manifest, and no viewer will show it.
    """
    with open(path, "rb") as handle:
        head = handle.read(8)
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".pdf":
        return head.startswith(b"%PDF")
    if suffix == ".png":
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    return False


@pytest.mark.parametrize("preference", ["png", "pdf"])
def test_the_qc_suite_gives_the_gallery_one_tile_per_file(tmp_path, monkeypatch,
                                                          preference):
    """"the number of figures in the gallery equals the number of image files
    in the run folder, and every one of them opens"."""
    regression_qc = pytest.importorskip("spacr.regression_qc")
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences",
                        lambda: (preference, 150))
    announced = []
    figure_sink.set_sink(lambda fig, path: announced.append(path))

    model, design, y, metadata = _small_ols()
    manifest = regression_qc.regression_qc_report(
        model, design, y, dst=str(tmp_path), metadata=metadata,
        regression_type="ols")

    folder = manifest["directory"]
    on_disk = _image_files(folder)
    assert on_disk, "the QC report wrote no figures at all"
    assert len(announced) == len(on_disk), (
        f"{len(announced)} figure(s) reached the gallery for {len(on_disk)} "
        f"file(s) on disk: {on_disk}")
    for name in on_disk:
        path = os.path.join(folder, name)
        assert _opens(path), f"{name} does not open as a {preference}"
        assert name.lower().endswith(preference), (
            f"{name} ignores the '{preference}' figure-format preference")

    # And the manifest names files that are there. A path recorded for a file
    # that was written under another extension is "saved but I cannot see it"
    # wearing a different hat.
    for path in manifest["written"]:
        assert os.path.isfile(path), f"the manifest names a missing file: {path}"
    assert os.path.isfile(manifest["combined"])
    assert set(announced) == {
        *manifest["written"], manifest["combined"]}


def test_the_diagnostic_panels_are_published_not_only_saved(tmp_path, monkeypatch):
    """`regression_diagnostics` wrote with a bare ``fig.savefig`` and never
    showed anything, so its design/residual/inference panels were on disk and
    none of them was in the application."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    diagnostics = pytest.importorskip("spacr.regression_diagnostics")
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("png", 150))
    announced = []
    figure_sink.set_sink(lambda fig, path: announced.append(path))

    rng = np.random.default_rng(3)
    fractions = pd.DataFrame(rng.random((60, 5)),
                             columns=[f"g{i}" for i in range(5)])
    # The caller asks for '.pdf'; the preference says PNG and must win, name
    # and content together.
    written, _report = diagnostics.plot_design_diagnostics(
        fractions, save_path=str(tmp_path / "design.pdf"))

    assert announced == [written]
    assert written.endswith(".png"), "the format preference did not reach it"
    assert _opens(written)


def test_the_diagnostic_suite_writes_one_file_per_panel_that_opens(tmp_path,
                                                                   monkeypatch):
    """One panel, one file, one tile.

    It used to default to ``formats=("pdf", "png")`` -- every panel computed
    and drawn twice, two files and (once published) two tiles for one picture,
    and the figure-format preference ignored by both.
    """
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    diagnostics = pytest.importorskip("spacr.regression_diagnostics")
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("png", 150))
    announced = []
    figure_sink.set_sink(lambda fig, path: announced.append(path))

    rng = np.random.default_rng(4)
    fractions = pd.DataFrame(rng.random((80, 4)),
                             columns=[f"g{i}" for i in range(4)])
    block = pd.Series([f"plate{i % 3 + 1}" for i in range(80)])
    p_values = np.concatenate([rng.uniform(0, 1, 40), rng.uniform(0, 1e-4, 10)])

    written = diagnostics.write_diagnostic_suite(
        tmp_path, fractions=fractions, block=block, p_values=p_values)

    assert not [key for key in written if key.endswith("_error")], written
    on_disk = _image_files(tmp_path)
    assert len(announced) == len(on_disk) == 2, (announced, on_disk)
    for name in on_disk:
        assert name.endswith(".png"), name
        assert _opens(os.path.join(tmp_path, name))
    # The manifest is keyed by what was WRITTEN, so every key names a file.
    for key, value in written.items():
        if key == "diagnostic_summary":
            continue
        assert os.path.isfile(value), f"{key} -> {value}"


def test_the_regression_sheet_reaches_the_gallery(tmp_path, monkeypatch):
    """``regression_figure.pdf`` is THE publication figure of a run, and it
    was written and closed in the next breath -- the one figure of the run
    nobody could look at."""
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    ml = pytest.importorskip("spacr.ml")
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("png", 150))
    announced = []
    figure_sink.set_sink(lambda fig, path: announced.append(path))

    rng = np.random.default_rng(5)
    n = 40
    coef_df = pd.DataFrame({
        "feature": [f"gene_{i}" for i in range(n)],
        "coefficient": rng.normal(0, 0.3, n),
        "p_value": rng.uniform(1e-6, 1, n),
        "adjusted_p_value": rng.uniform(1e-4, 1, n),
        "n": rng.integers(3, 40, n),
    })

    path = ml._write_regression_sheet(coef_df, str(tmp_path))

    assert path and os.path.isfile(path)
    assert announced == [path], (
        "the run's publication figure was saved and never announced")


def test_a_headless_run_still_writes_its_regression_figures(tmp_path, monkeypatch):
    """`spacr-run regression` and a notebook have no GUI at all.

    Instruction 139 A moves the generated figures to pyqtgraph, which is a
    SCREEN library and needs a QApplication. "A run that silently stops
    writing figures when there is no display is the worst outcome here", so
    the guarantee is pinned here BEFORE the renderer changes: no
    QApplication, no sink, and the QC suite still puts its twenty files on
    disk.
    """
    regression_qc = pytest.importorskip("spacr.regression_qc")
    from spacr import plot

    try:
        from PySide6.QtWidgets import QApplication
        assert QApplication.instance() is None, (
            "this test is only meaningful without a QApplication")
    except ImportError:                                # pragma: no cover
        pass

    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("png", 150))
    assert figure_sink.sink() is None, "the fixture left a sink installed"

    model, design, y, metadata = _small_ols()
    manifest = regression_qc.regression_qc_report(
        model, design, y, dst=str(tmp_path), metadata=metadata,
        regression_type="ols")

    on_disk = _image_files(manifest["directory"])
    assert len(on_disk) == len(manifest["written"]) + 1, on_disk
    assert len(on_disk) >= 20, (
        f"a headless run wrote only {len(on_disk)} figure(s)")
    for name in on_disk:
        assert _opens(os.path.join(manifest["directory"], name))


def test_an_explicit_format_wins_and_none_lets_the_preference_through(tmp_path,
                                                                      monkeypatch):
    """`fmt` used to default to 'pdf' and be baked into the file NAME.

    Two failures in one line. A user who had chosen PNG got PDFs, because a
    hard default is not a preference; and a caller who explicitly asked for
    PNG got its request thrown away by `save_figure`, which saw no `fmt` and
    used the preference -- while the manifest still recorded the `.png` name
    the caller had asked for, naming a file that was not there.
    """
    regression_qc = pytest.importorskip("spacr.regression_qc")
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("pdf", 150))
    model, design, y, metadata = _small_ols()

    forced = regression_qc.regression_qc_report(
        model, design, y, dst=str(tmp_path / "forced"), metadata=metadata,
        regression_type="ols", panels=("residuals_vs_fitted",),
        combined=False, fmt="png", verbose=False)
    assert forced["written"], forced["skipped"]
    for path in forced["written"]:
        assert path.endswith(".png"), "an explicit fmt did not win"
        assert os.path.isfile(path)
        assert _opens(path)

    preferred = regression_qc.regression_qc_report(
        model, design, y, dst=str(tmp_path / "preferred"), metadata=metadata,
        regression_type="ols", panels=("residuals_vs_fitted",),
        combined=False, verbose=False)
    for path in preferred["written"]:
        assert path.endswith(".pdf"), "the format preference did not decide"
        assert os.path.isfile(path)
        assert _opens(path)


def test_the_pipeline_qc_report_follows_the_format_preference(tmp_path,
                                                              monkeypatch):
    """`ml._write_regression_qc` is the pipeline's route into the QC suite.

    It used to resolve the preference itself and pass it as `fmt=`, which made
    a PREFERENCE indistinguishable from a caller FORCING a format. The report
    reads the preference now, so there is one place that decides; this asserts
    the pipeline path still lands on the user's choice.
    """
    pd = pytest.importorskip("pandas")
    ml = pytest.importorskip("spacr.ml")
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences", lambda: ("png", 150))
    model, design, y, metadata = _small_ols()
    frame = pd.DataFrame(metadata)

    manifest = ml._write_regression_qc(model, design, y, frame, str(tmp_path),
                                       regression_type="ols")

    assert manifest and manifest["written"], manifest
    for path in manifest["written"]:
        assert path.endswith(".png"), (
            "the pipeline's QC panels ignored the figure-format preference")
        assert os.path.isfile(path)
