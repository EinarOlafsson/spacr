"""The generated regression figures are drawn by the screen's renderer.

Instruction 139 A, requested 2026-08-18: "in the regression modual all of the
generated graphs should be generated with pyqtgraph, not matplotlib".

`spacr.figures.fast_render` did the seven plots with an interactive twin;
`spacr.figures.scene` does the other thirty-four by translating the finished
matplotlib artists of a panel into a pyqtgraph scene and exporting THAT. The
statistics are computed once, by the code that already computes them, so the
file and the panel cannot come to different numbers.

WHAT THESE TESTS ARE FOR, and it is not "did it run". Every failure recorded
below was found by looking at a rendered PNG rather than by a green test:

  * every bar in the suite was dropped, because a Patch's `get_transform` is
    not the data transform and the identity check excluded all of them. Twenty
    panels with correct axes and no data on them;
  * a correlation matrix came out on its anti-diagonal, because the array was
    flipped as well as transposed;
  * the axis label sat on top of the tick labels, because an unshown widget
    has never been laid out and an AxisItem measures itself while it paints;
  * markers came out half size, because matplotlib's sizes are POINTS and a
    pyqtgraph scene is PIXELS;
  * the variance panel's y label read `$\\sqrt{|\\mathrm{...}|}$`, because
    pyqtgraph has no mathtext.

So the assertions here are about PIXELS and about GEOMETRY wherever the defect
would be invisible to an assertion about calls.
"""
from __future__ import annotations

import os

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
np = pytest.importorskip("numpy")
from matplotlib.figure import Figure  # noqa: E402

from spacr import figure_sink  # noqa: E402
from spacr.figures import scene  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sink_left_behind():
    figure_sink.clear_sink()
    yield
    figure_sink.clear_sink()


@pytest.fixture(autouse=True)
def _no_renderer_asked_for(monkeypatch):
    """Every test states its own renderer; none inherits the shell's."""
    monkeypatch.delenv("SPACR_FIGURE_RENDERER", raising=False)


def _qt_or_skip():
    ok, why = scene.pyqtgraph_ready()
    if not ok:
        pytest.skip(f"pyqtgraph is not available here: {why}")


def _figure(dpi=140):
    fig = Figure(figsize=(4.0, 3.0), dpi=dpi)
    ax = fig.subplots()
    ax.plot([0, 1, 2], [0, 1, 0.5], color="#2E77BC")
    ax.scatter([0, 1, 2], [0.2, 0.8, 0.4], s=36, color="#B4B4B4")
    ax.axhline(0.5, color="#7F7F7F", linestyle="--")
    ax.set_title("a title")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def _colours(path):
    Image = pytest.importorskip("PIL.Image")
    with Image.open(path) as image:
        return set(image.convert("RGB").getdata())


# --------------------------------------------------------------------------- #
#  Choosing a renderer
# --------------------------------------------------------------------------- #

def test_the_environment_can_force_either_renderer(monkeypatch):
    monkeypatch.setenv("SPACR_FIGURE_RENDERER", "matplotlib")
    assert scene.scene_renderer()[0] == "matplotlib"
    assert scene.scene_renderer()[1], "matplotlib always says why"

    monkeypatch.setenv("SPACR_FIGURE_RENDERER", "MATPLOTLIB")
    assert scene.scene_renderer()[0] == "matplotlib", "case is not a setting"


def test_a_misspelt_renderer_is_auto_and_not_an_error(monkeypatch):
    """A run must not lose its figures over a typo in an environment
    variable."""
    monkeypatch.setenv("SPACR_FIGURE_RENDERER", "pyqtgrpah")
    assert scene.requested_renderer() == "auto"


def test_an_explicit_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("SPACR_FIGURE_RENDERER", "pyqtgraph")
    assert scene.scene_renderer("matplotlib")[0] == "matplotlib"


def test_without_qt_the_figure_is_still_written(tmp_path, monkeypatch):
    """THE HEADLESS GUARANTEE. A machine with no Qt writes the page it always
    wrote, says why, and loses nothing."""
    monkeypatch.setattr(scene, "pyqtgraph_ready",
                        lambda: (False, "no Qt on this machine"))

    written, renderer, why = scene.write_figure(
        _figure(), str(tmp_path / "panel"), fmt="png")

    assert renderer == "matplotlib"
    assert "no Qt on this machine" in why
    assert written and os.path.isfile(written)


def test_a_renderer_the_gallery_cannot_show_is_not_used(tmp_path):
    """139 C's rule as a PRECONDITION, and it is the thing that stops 139 A
    reintroducing 139 C's bug.

    There are two routes into the gallery -- a Figure through ``publish`` and a
    finished FILE through ``publish_file`` -- and ``spacr/qt/bridge.py``
    installs only the first. Measured 2026-08-18. So in the running
    application a pyqtgraph export is announced to nobody, and converting a
    twenty-panel suite to it would take all twenty out of the gallery. That
    state chooses matplotlib until the bridge listens for files, and then
    changes back with no edit here.
    """
    figure_sink.set_sink(lambda fig, path: None)
    renderer, why = scene.scene_renderer()
    assert renderer == "matplotlib"
    assert "file sink" in why, why

    figure_sink.set_file_sink(lambda path, title: None)
    _qt_or_skip()
    assert scene.scene_renderer()[0] == "pyqtgraph", (
        "with both routes listening there is nothing left to lose")


def test_forcing_the_renderer_overrules_the_gallery_guard():
    """A person overruling the rule is not the rule failing."""
    _qt_or_skip()
    figure_sink.set_sink(lambda fig, path: None)
    assert scene.scene_renderer("pyqtgraph")[0] == "pyqtgraph"


# --------------------------------------------------------------------------- #
#  The translation
# --------------------------------------------------------------------------- #

def test_a_plain_panel_translates_completely_and_is_drawn():
    _qt_or_skip()
    widget, report = scene.build_scene(_figure())
    try:
        assert report.complete, report.reason()
        assert report.axes == 1
        # line + scatter + reference line, at least.
        assert report.items >= 3, report.items
    finally:
        widget.deleteLater()


def test_the_bars_of_a_histogram_survive_the_translation(tmp_path):
    """THE BUG THAT WOULD HAVE SHIPPED TWENTY EMPTY PANELS.

    `Patch.get_transform` is `get_patch_transform() + the artist's`, so it is
    never `ax.transData` itself and an identity test against it excluded every
    bar in the suite. The axes, the ranges, the reference line and the
    annotation all still drew, so the figure looked finished and had no data
    on it. Asserted on the PIXELS, because that is the only place it showed.
    """
    _qt_or_skip()
    fig = Figure(figsize=(4.0, 3.0), dpi=100)
    ax = fig.subplots()
    ax.hist(np.repeat([0.1, 0.5, 0.9], 20), bins=10, color="#C4441C")
    ax.set_xlim(0, 1)

    written, report = scene.render_figure(fig, str(tmp_path / "hist"),
                                          fmt="png", announce=False)

    assert written, report.reason()
    assert (196, 68, 28) in _colours(written), "the bars are not on the page"


def test_a_bar_is_held_inside_the_axes():
    """A ViewBox does not clip its children, so a bar that starts outside the
    range drew straight across the page. The geometry is clamped instead of
    the item being clipped, because an annotation deliberately placed below
    the axes must still be drawn."""
    _qt_or_skip()
    fig = Figure(figsize=(4.0, 3.0), dpi=100)
    ax = fig.subplots()
    ax.barh([0, 1], [3.0, 4.0], color="#B4B4B4")
    ax.set_xlim(1.0, 5.0)

    widget, report = scene.build_scene(fig)
    try:
        assert report.complete, report.reason()
        bars = [item for item in widget.ci.items
                if type(item).__name__ == "PlotItem"]
        assert bars
        for plot in bars:
            for item in plot.items:
                if type(item).__name__ == "BarGraphItem":
                    assert min(item.opts["x0"]) >= 1.0 - 1e-9, item.opts["x0"]
    finally:
        widget.deleteLater()


def test_an_image_keeps_the_way_up_it_was_drawn(tmp_path):
    """A correlation matrix came out on its anti-diagonal: the array was
    flipped as well as transposed, and `imshow` states the way up in its
    extent already."""
    _qt_or_skip()
    array = np.zeros((3, 3))
    array[0, 0] = 1.0                     # top-left, and it must stay there
    fig = Figure(figsize=(3.0, 3.0), dpi=100)
    ax = fig.subplots()
    ax.imshow(array, cmap="gray", vmin=0, vmax=1)
    ax.set_axis_off()

    written, report = scene.render_figure(fig, str(tmp_path / "image"),
                                          fmt="png", announce=False)
    assert written, report.reason()

    Image = pytest.importorskip("PIL.Image")
    with Image.open(written) as image:
        pixels = image.convert("L")
        width, height = pixels.size
        top_left = pixels.getpixel((width // 6, height // 6))
        bottom_left = pixels.getpixel((width // 6, height * 5 // 6))
    assert top_left > bottom_left + 100, (top_left, bottom_left)


def test_a_marker_is_sized_in_pixels_not_in_points():
    """matplotlib sizes are POINTS and a pyqtgraph scene is PIXELS, so a
    140 dpi figure's marks are 1.94x the number matplotlib carries. Passing
    the number through drew a panel of dust."""
    _qt_or_skip()
    fig = Figure(figsize=(4.0, 3.0), dpi=144)
    ax = fig.subplots()
    ax.scatter([0, 1], [0, 1], s=36.0)     # 6 pt across
    widget, report = scene.build_scene(fig)
    try:
        assert report.complete, report.reason()
        plot = [item for item in widget.ci.items
                if type(item).__name__ == "PlotItem"][0]
        points = [item for item in plot.items
                  if type(item).__name__ == "ScatterPlotItem"][0]
        # `opts['size']` is the DEFAULT; a per-point array lands in `data`.
        drawn = float(points.data["size"][0])
        # matplotlib's s = 36 is an AREA in points squared, so 6 pt across,
        # and 6 pt at 144 dpi is 12 px.
        assert abs(drawn - 12.0) < 0.5, drawn
    finally:
        widget.deleteLater()


def test_a_categorical_axis_keeps_its_own_labels():
    """The screen-level panels put plate names on the axis. A renderer that
    re-derives ticks from the range writes 0, 1, 2 where the panel wrote
    plate1, plate2, plate3 -- a different figure, silently."""
    _qt_or_skip()
    fig = Figure(figsize=(4.0, 3.0), dpi=100)
    ax = fig.subplots()
    ax.plot([0, 1, 2], [1, 2, 3])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["plate1", "plate2", "plate3"])

    widget, report = scene.build_scene(fig)
    try:
        plot = [item for item in widget.ci.items
                if type(item).__name__ == "PlotItem"][0]
        ticks = plot.getAxis("bottom")._tickLevels
        drawn = {label for _, label in (ticks[0] if ticks else [])}
        assert {"plate1", "plate2", "plate3"} <= drawn, ticks
    finally:
        widget.deleteLater()


# --------------------------------------------------------------------------- #
#  Mathtext
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("residual", "residual"),
    (r"$\sqrt{|\mathrm{standardised\ residual}|}$",
     "√(|standardised residual|)"),
    (r"$x^2$", "x²"),
    (r"$-\log_{10}(p)$", "-log₁₀(p)"),
    (r"$\mathdefault{10^{-1}}$", "10⁻¹"),
])
def test_the_mathtext_this_translator_knows(raw, expected):
    body, understood = scene._plain_text(raw)
    assert understood, body
    assert body == expected


@pytest.mark.parametrize("raw", [r"$\frac{a}{b}$", r"$\lambda_{GC}$"])
def test_mathtext_it_does_not_know_is_refused_rather_than_guessed(raw):
    """A label that comes out as raw TeX is a figure with a bug printed on its
    axis. Half a formula silently rewritten is worse."""
    assert scene._plain_text(raw)[1] is False


def test_a_figure_with_unknown_mathtext_falls_back_and_names_it(tmp_path):
    _qt_or_skip()
    fig = Figure(figsize=(3.0, 2.0), dpi=100)
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_ylabel(r"$\frac{a}{b}$")

    written, renderer, why = scene.write_figure(fig, str(tmp_path / "panel"),
                                                fmt="png",
                                                renderer="pyqtgraph")
    assert renderer == "matplotlib"
    assert "mathtext" in why
    assert written and os.path.isfile(written), "the picture is never lost"


# --------------------------------------------------------------------------- #
#  A saved figure is for paper (instruction 150 C, the generated half)
# --------------------------------------------------------------------------- #

def test_the_shared_print_rule_decides_the_scene_chrome(monkeypatch):
    """The pyqtgraph half asks `spacr.figure_style.export_colour`, which is
    the whole point of that function existing: two renderers deciding
    separately what 'print' means is the defect 150 C names."""
    _qt_or_skip()
    from spacr import figure_style

    asked = []
    real = figure_style.export_colour

    def spy(current, kind, look=None):
        asked.append(kind)
        return real(current, kind, look)

    monkeypatch.setattr(figure_style, "export_colour", spy)
    widget, report = scene.build_scene(_figure(), mode="print")
    try:
        assert report.complete, report.reason()
    finally:
        widget.deleteLater()
    assert "chrome" in asked, asked
    assert "data" not in asked, (
        "the data must never be handed to the flip rule -- a white data point "
        "turned black is, on a volcano, the colour of 'not a hit'")


def test_a_data_colour_survives_the_print_ground(tmp_path):
    """The chrome flips and the data does not. The mark is drawn in the colour
    the panel chose, on a page it can be read on."""
    _qt_or_skip()
    fig = Figure(figsize=(3.0, 2.0), dpi=100)
    ax = fig.subplots()
    ax.scatter([0.5], [0.5], s=900, color="#C4441C")
    ax.set_facecolor("#1E1E1E")

    written, report = scene.render_figure(fig, str(tmp_path / "dot"),
                                          fmt="png", mode="print",
                                          announce=False)
    assert written, report.reason()
    colours = _colours(written)
    assert (196, 68, 28) in colours, "the data colour did not survive"
    assert (255, 255, 255) in colours, "the page is not light"


def test_a_colour_map_is_not_reported_as_an_illegible_data_colour(tmp_path,
                                                                  capsys):
    """A diverging map's midpoint is pale BY DESIGN -- RdBu_r at r = 0 is near
    white -- so sampling the ramp fired the warning on every correlation panel
    ever written. A warning that fires on everything is a warning nobody
    reads."""
    _qt_or_skip()
    fig = Figure(figsize=(3.0, 3.0), dpi=100)
    ax = fig.subplots()
    ax.imshow(np.linspace(-1, 1, 9).reshape(3, 3), cmap="RdBu_r",
              vmin=-1, vmax=1)

    written, report = scene.render_figure(fig, str(tmp_path / "map"),
                                          fmt="png", announce=False)
    assert written, report.reason()
    assert "almost no contrast" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  The QC suite, end to end
# --------------------------------------------------------------------------- #

def _small_ols():
    pd = pytest.importorskip("pandas")
    sm = pytest.importorskip("statsmodels.api")

    rng = np.random.default_rng(11)
    n = 90
    frame = pd.DataFrame({"gene_a": rng.normal(0, 1, n),
                          "gene_b": rng.normal(0, 1, n),
                          "gene_c": rng.normal(0, 1, n)})
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


@pytest.mark.parametrize("preference", ["png", "pdf"])
def test_the_whole_qc_suite_is_rendered_saved_and_announced(tmp_path,
                                                            monkeypatch,
                                                            preference):
    """The acceptance is a COUNT: every file on disk is a tile in the gallery
    and every tile is a file. A renderer change that quietly drops one is 139
    C's bug reintroduced by 139 A."""
    _qt_or_skip()
    regression_qc = pytest.importorskip("spacr.regression_qc")
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences",
                        lambda: (preference, 150))
    announced = []
    figure_sink.set_file_sink(lambda path, title: announced.append(path))
    figure_sink.set_sink(lambda fig, path: announced.append(path))

    model, design, y, metadata = _small_ols()
    manifest = regression_qc.regression_qc_report(
        model, design, y, dst=str(tmp_path), metadata=metadata,
        regression_type="ols", renderer="pyqtgraph", verbose=False)

    assert manifest["renderer"] == "pyqtgraph"
    assert manifest["renderer_fallbacks"] == [], manifest["renderer_fallbacks"]
    assert manifest["renderer_counts"] == {"pyqtgraph": 20}, \
        manifest["renderer_counts"]

    on_disk = sorted(name for name in os.listdir(manifest["directory"])
                     if os.path.splitext(name)[1].lower() in (".png", ".pdf"))
    assert len(on_disk) == 20, on_disk
    assert len(announced) == len(on_disk), (len(announced), on_disk)
    for name in on_disk:
        assert name.endswith("." + preference), name
        head = open(os.path.join(manifest["directory"], name), "rb").read(4)
        assert head == (b"\x89PNG" if preference == "png" else b"%PDF"), name


def test_the_renderer_is_decided_once_for_the_whole_suite(tmp_path,
                                                          monkeypatch):
    """A suite half in one library and half in the other is worse than a suite
    entirely in the old one -- an earlier attempt elsewhere in this project
    drew one run's first figure in matplotlib and its other six in pyqtgraph
    because the first figure itself changed the answer."""
    regression_qc = pytest.importorskip("spacr.regression_qc")
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences",
                        lambda: ("png", 100))
    model, design, y, metadata = _small_ols()
    manifest = regression_qc.regression_qc_report(
        model, design, y, dst=str(tmp_path), metadata=metadata,
        regression_type="ols", renderer="matplotlib", verbose=False,
        panels=("residuals_vs_fitted", "vif"), combined=False)

    assert manifest["renderer"] == "matplotlib"
    assert set(manifest["renderer_counts"]) == {"matplotlib"}
    assert manifest["renderer_fallbacks"] == [], (
        "a suite that ASKED for matplotlib has not fallen back to it")


# --------------------------------------------------------------------------- #
#  Every diagnostic carries its own verdict (instruction 115)
# --------------------------------------------------------------------------- #

def test_a_rank_deficient_design_fails_rather_than_warns():
    """A coefficient from a rank-deficient fit is one of infinitely many
    solutions, and statsmodels answers with a pseudo-inverse rather than
    refusing -- so the numbers look like any other fit."""
    rd = pytest.importorskip("spacr.regression_diagnostics")
    pd = pytest.importorskip("pandas")

    fractions = pd.DataFrame(np.eye(4), columns=list("abcd"))
    verdict = rd.score_design(rd.design_report(fractions))
    assert verdict.level == "fail"
    assert "identify" in verdict.headline


def test_a_healthy_design_passes():
    rd = pytest.importorskip("spacr.regression_diagnostics")
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(1)
    fractions = pd.DataFrame(rng.random((120, 6)), columns=list("abcdef"))
    verdict = rd.score_design(rd.design_report(fractions))
    assert verdict.level == "pass", verdict.detail


def test_one_dominating_observation_fails_the_residual_sheet():
    rd = pytest.importorskip("spacr.regression_diagnostics")

    rng = np.random.default_rng(2)
    n = 40
    x = np.linspace(0, 1, n)
    design = np.column_stack([np.ones(n), x])
    observed = 2.0 * x + rng.normal(0, 0.01, n)
    observed[-1] += 25.0                       # one well, a long way out
    fitted = design @ np.linalg.lstsq(design, observed, rcond=None)[0]

    verdict = rd.score_residuals(rd.residual_report(observed, fitted,
                                                    design=design))
    assert verdict.level == "fail", verdict.detail
    assert "Cook" in verdict.detail


def test_a_calibrated_null_passes_and_an_inflated_one_fails():
    """lambda is a MEDIAN, so a handful of real hits barely move it -- scoring
    the spike itself would flag every successful screen."""
    rd = pytest.importorskip("spacr.regression_diagnostics")

    calibrated = {"tests": 500, "pi0": 0.98, "estimated_non_null": 10.0,
                  "genomic_inflation": 1.01}
    inflated = dict(calibrated, genomic_inflation=3.4)
    assert rd.score_inference(calibrated).level == "pass"
    assert rd.score_inference(inflated).level == "fail"
    assert rd.score_inference({"tests": 0}).level == "unknown"


def test_the_inference_sheet_records_the_inflation_it_draws(tmp_path):
    """lambda was computed to colour one annotation and thrown away, so the
    number that says whether the whole family of p-values can be believed was
    readable by eye and by nothing else."""
    rd = pytest.importorskip("spacr.regression_diagnostics")

    rng = np.random.default_rng(4)
    _path, report = rd.plot_inference_diagnostics(rng.random(400))
    assert "genomic_inflation" in report
    assert np.isfinite(report["genomic_inflation"])
    assert report["verdict_level"] in ("pass", "check", "fail")


def test_the_suite_verdict_is_its_worst_sheet(tmp_path, monkeypatch):
    """Nineteen clean panels and one unidentifiable design is not 'mostly
    fine'; it is a run whose coefficients are one of infinitely many answers."""
    rd = pytest.importorskip("spacr.regression_diagnostics")
    pd = pytest.importorskip("pandas")
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences",
                        lambda: ("png", 100))
    fractions = pd.DataFrame(np.eye(5), columns=list("abcde"))
    written = rd.write_diagnostic_suite(str(tmp_path), fractions=fractions)

    # THE MAPPING STAYS A MAP OF FILES. A verdict string in it is a path that
    # is not there, and a caller checking its own output would trip on it --
    # which one did, the moment this was first written that way.
    for key, value in written.items():
        assert os.path.isfile(value), f"{key} -> {value}"

    summary = pd.read_csv(written["diagnostic_summary"])
    suite = summary[summary["section"] == "suite"].set_index("metric")["value"]
    assert suite["verdict_level"] == "fail", summary
    assert "design_diagnostics" in suite["verdict"]


def test_a_diagnostic_sheet_reaches_the_gallery_whatever_drew_it(tmp_path,
                                                                 monkeypatch):
    """139 C's rule survives the move: saved and visible are one event, and a
    file pyqtgraph wrote is announced through `publish_file` because there is
    no matplotlib figure left to render."""
    rd = pytest.importorskip("spacr.regression_diagnostics")
    pd = pytest.importorskip("pandas")
    from spacr import plot

    monkeypatch.setattr(plot, "figure_output_preferences",
                        lambda: ("png", 100))
    announced = []
    figure_sink.set_file_sink(lambda path, title: announced.append(path))
    figure_sink.set_sink(lambda fig, path: announced.append(path))

    rng = np.random.default_rng(7)
    fractions = pd.DataFrame(rng.random((60, 5)), columns=list("abcde"))
    written = rd.write_diagnostic_suite(str(tmp_path), fractions=fractions)

    sheet = written.get("design_diagnostics_png")
    assert sheet and os.path.isfile(sheet)
    assert sheet in announced, announced
