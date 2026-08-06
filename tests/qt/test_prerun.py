"""The two things a user sees before pressing Run, on the real screens.

Both are advisory, and the single most important assertion in this file is
the boring one: :func:`test_the_banner_never_disables_run`. A segmentation
verdict informs a decision; a plate that failed QC is still a plate its owner
may have every reason to measure, and a quality report that stops people is a
quality report they switch off. So the banner is checked against the *real*
``AppScreen``'s real Run button — before it refreshes, after it refreshes on
the worst verdict this codebase can produce, and after the scoring button has
been pressed.

The rest:

* the banner is reached the way a user reaches it — through the registered
  screen factory ``MainWindow._build_screen`` consults — and installing it
  does not cost the screen the chaining strip it already had, in either
  registration order;
* what it says names the plate, the rows and the cause, because "3 plates
  failed" changes nothing;
* it reads the card the mask run wrote and does not score a mask to draw
  itself (asserted by making scoring raise);
* the diameter panel measures discs of a known size and reports how many
  objects it measured, and nothing it proposes reaches a settings field until
  the user presses Use.

Registration is done and undone by a fixture: ``APP_FACTORIES`` is a
process-global dict a dozen other test modules build screens through, and a
factory leaked out of this file would be a failure somewhere else entirely.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel  # noqa: E402

from spacr import seg_qc  # noqa: E402
from spacr.qt import prerun  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# fixtures: real masks, real images, real screens
# ---------------------------------------------------------------------------

def _disc(labels, cy, cx, radius, value):
    h, w = labels.shape
    y0, y1 = max(0, int(cy - radius) - 1), min(h, int(cy + radius) + 2)
    x0, x1 = max(0, int(cx - radius) - 1), min(w, int(cx + radius) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sub = labels[y0:y1, x0:x1]
    sub[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius] = value
    return labels


def _field_of_n(n, shape=(512, 512), radius=6, seed=0):
    """Exactly ``n`` separated discs, none on the border."""
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, np.int32)
    margin, step = radius + 4, 2 * radius + 8
    slots = [(cy, cx)
             for cy in range(margin, shape[0] - margin, step)
             for cx in range(margin, shape[1] - margin, step)]
    for value, index in enumerate(sorted(rng.permutation(len(slots))[:n]), 1):
        _disc(labels, *slots[index], radius, value)
    return labels


def _write_plate(root, fields, object_type="cell"):
    folder = Path(root) / "norm_channel_stack" / f"{object_type}_mask_stack"
    folder.mkdir(parents=True, exist_ok=True)
    for name, mask in fields.items():
        np.save(folder / f"{name}.npy", mask.astype(np.uint16))
    return str(root)


def _plate(plate, counts):
    """``{row: n}`` → one field per (row, column 01/02)."""
    return {
        f"{plate}_{row}{col:02d}_1": _field_of_n(n, seed=i * 11 + col)
        for i, (row, n) in enumerate(counts.items())
        for col in (1, 2)
    }


@pytest.fixture
def stepped_project(tmp_path):
    """plate2: rows E-H hold 4x the objects of rows A-D. Every field is clean."""
    root = tmp_path / "plate2"
    _write_plate(root, _plate("plate2", {
        "A": 10, "B": 10, "C": 10, "D": 10,
        "E": 40, "F": 40, "G": 40, "H": 40,
    }))
    seg_qc.score_digest(str(root))
    return str(root)


@pytest.fixture
def clean_project(tmp_path):
    """plate1: nothing wrong with it, scored."""
    root = tmp_path / "plate1"
    _write_plate(root, _plate("plate1", dict.fromkeys("ABCDEFGH", 20)))
    seg_qc.score_digest(str(root))
    return str(root)


@pytest.fixture
def unscored_project(tmp_path):
    """Masks on disk that nothing has ever looked at."""
    root = tmp_path / "plate3"
    _write_plate(root, _plate("plate3", dict.fromkeys("ABCD", 20)))
    return str(root)


@pytest.fixture
def registered():
    """Register the factories for one test, then hand the keys back."""
    from spacr.qt import chaining
    chaining.register()
    prerun.register()
    try:
        yield
    finally:
        prerun.unregister()


def _screen(qtbot, app_key="measure"):
    """Build a module screen the way ``MainWindow._build_screen`` does."""
    from spacr.qt.app import APP_FACTORIES, _call_screen_factory
    factory = APP_FACTORIES.get(app_key)
    assert factory is not None, f"nothing is registered for {app_key!r}"
    screen = _call_screen_factory(factory, app_key, None)
    qtbot.addWidget(screen)
    return screen


def _set(screen, key, value):
    assert screen._settings_model.set_value_for_key(key, value), key


def _texts(widget):
    """Every label's text under ``widget``, joined — what the user can read."""
    return "\n".join(
        label.text() for label in widget.findChildren(QLabel) if label.text())


@pytest.fixture
def banner(qtbot, registered):
    screen = _screen(qtbot, "measure")
    found = prerun.qc_banner(screen)
    assert found is not None
    return found


# ---------------------------------------------------------------------------
# 1. it informs, it does not block
# ---------------------------------------------------------------------------

def test_the_banner_never_disables_run(qtbot, banner, stepped_project):
    """The point of the whole feature, asserted against the real Run button.

    Checked at every stage, because "it does not disable Run" is only worth
    anything if it survives the moment the verdict turns bad.
    """
    screen = banner._screen
    assert screen._btn_run.isEnabled()

    _set(screen, "src", stepped_project)
    banner.refresh()

    assert banner.digest.verdict == "fail"
    assert screen._btn_run.isEnabled(), "a failing verdict disabled Run"
    assert prerun.BLOCKS_RUN is False
    assert banner.digest.blocks_run is False

    banner._on_score_clicked()
    with qtbot.waitSignal(banner.refreshed, timeout=60000):
        pass
    assert screen._btn_run.isEnabled(), "scoring the masks disabled Run"


def test_nothing_in_this_module_can_reach_the_run_button():
    """Not merely "does not disable it" — cannot touch it.

    The behavioural test above proves Run survives one bad verdict. This one
    cannot rot: it reads the module's own source and fails if a reference to
    the Run button (or to the actions row's enablement) ever appears in it,
    which is the only way "advisory" stays true after the next edit.
    """
    source = Path(prerun.__file__).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    )
    for forbidden in ("_btn_run", "setEnabled(False)  # run", "_on_run"):
        assert forbidden not in code, (
            f"{forbidden!r} appears in spacr/qt/prerun.py: the banner is "
            f"advisory and must not be able to gate a run")


def test_the_banner_is_reached_through_the_registered_factory(qtbot, registered):
    """A user reaches it by opening Measure; that path is what is tested."""
    screen = _screen(qtbot, "measure")
    found = prerun.qc_banner(screen)
    assert isinstance(found, prerun.SegQCBanner)
    assert found.objectName() == prerun.QC_OBJECT_NAME

    # Immediately above the Run row: the last thing the eye crosses on its
    # way to the button. A panel the user has to go and open is a panel
    # nobody opens.
    layout = screen._runtime_wrap.layout()
    assert layout.indexOf(found) == layout.indexOf(screen._actions_row) - 1


@pytest.mark.parametrize("order", ["chaining_first", "prerun_first"])
def test_installing_the_banner_never_costs_a_screen_its_chaining_strip(qtbot, order):
    """Both modules register a factory for `measure`; neither may lose."""
    from spacr.qt import chaining
    from spacr.qt.app import APP_FACTORIES

    prerun.unregister()
    chaining.unregister()
    try:
        if order == "chaining_first":
            chaining.register()
            prerun.register()
        else:
            prerun.register()
            chaining.register()
        assert APP_FACTORIES.get("measure") is prerun._prerun_screen
        screen = _screen(qtbot, "measure")
        assert prerun.qc_banner(screen) is not None
        assert chaining.chaining_bar(screen) is not None
    finally:
        prerun.unregister()
        chaining.unregister()
        chaining.register()


def test_the_launch_list_imports_this_module_so_a_user_can_reach_it():
    """A finished feature nobody can open is not a finished feature.

    ``register()`` is only called for modules named in this tuple, and that
    call is what puts the banner on the Measure screen at all.
    """
    from spacr.qt import SELF_REGISTERING_MODULES
    assert "spacr.qt.prerun" in SELF_REGISTERING_MODULES
    # After chaining, so the ordinary launch order composes onto chaining's
    # screen. Both orders are covered above; this pins the ordinary one.
    assert (SELF_REGISTERING_MODULES.index("spacr.qt.prerun")
            > SELF_REGISTERING_MODULES.index("spacr.qt.chaining"))


def test_unregistering_hands_the_screen_back_to_whoever_had_it(qtbot):
    from spacr.qt import chaining
    from spacr.qt.app import APP_FACTORIES

    chaining.register()
    before = APP_FACTORIES.get("measure")
    prerun.register()
    assert prerun.unregister() == 2
    assert APP_FACTORIES.get("measure") is before


# ---------------------------------------------------------------------------
# 2. what it says
# ---------------------------------------------------------------------------

def test_a_failing_plate_is_named_with_its_rows_and_its_likely_cause(
        banner, stepped_project):
    """"3 plates failed QC" is useless. This is the sentence that is not."""
    _set(banner._screen, "src", stepped_project)
    banner.refresh()

    text = _texts(banner)
    assert "failed" in text
    assert "plate2" in text
    assert "rows E-H" in text
    assert "rows A-D" in text
    assert "4.0x" in text
    assert "illumination" in text
    # Every field passed on its own; saying so is what stops the user
    # hunting through per-field cards for a field that is not there.
    assert "no single field was flagged" in text
    assert "never stops Measure" in text


def test_a_clean_project_produces_a_clean_banner(banner, clean_project):
    _set(banner._screen, "src", clean_project)
    banner.refresh()

    assert banner.digest.verdict == "ok"
    text = _texts(banner)
    assert "passed" in text
    assert "none flagged" in text
    assert "illumination" not in text, "a clean plate is offered no fixes"
    assert banner._findings_box.isHidden() or not banner.digest.findings


def test_masks_nobody_has_scored_are_reported_as_not_run(banner, unscored_project):
    """Missing is not clean, and the banner has to be clear about which it is."""
    _set(banner._screen, "src", unscored_project)
    banner.refresh()

    assert banner.digest.verdict == "missing"
    text = _texts(banner)
    assert "not run" in text
    assert "not the same as clean" in text
    assert banner._btn_score.text() == "Score the masks now"


def test_a_card_older_than_its_masks_is_shown_as_out_of_date(banner, clean_project):
    card = Path(clean_project) / "qc" / "segmentation_qc_cell.csv"
    when = card.stat().st_mtime - 600
    import os
    os.utime(card, (when, when))

    _set(banner._screen, "src", clean_project)
    banner.refresh()

    assert banner.digest.stale is True
    text = _texts(banner)
    assert "out of date" in text
    assert "describes the previous masks" in text
    assert banner._btn_score.text() == "Score the masks now"


def test_no_source_hides_the_banner(banner):
    """A screen pointed at nothing has nothing to say, and says nothing.

    ``src`` is cleared explicitly rather than assumed empty: a module screen
    opens on the last project the user worked in, so a fresh Measure screen
    normally arrives with a folder already in it — which is exactly when the
    banner is most useful.
    """
    _set(banner._screen, "src", "")
    banner.refresh()
    assert banner.isHidden()
    assert banner.digest is None


def test_show_all_findings_expands_and_collapses(banner, stepped_project):
    _set(banner._screen, "src", stepped_project)
    banner.refresh()
    collapsed = _texts(banner)
    assert not banner._btn_more.isHidden() or len(banner.digest.findings) <= 2

    banner._on_toggle_findings()
    expanded = _texts(banner)
    assert len(expanded) > len(collapsed)
    assert "rarely biology" in expanded, "the detail is what expanding is for"
    banner._on_toggle_findings()
    assert _texts(banner) == collapsed


def test_copy_report_puts_the_whole_verdict_on_the_clipboard(
        qapp, banner, stepped_project):
    _set(banner._screen, "src", stepped_project)
    banner.refresh()
    banner._on_copy_clicked()
    assert qapp.clipboard().text() == seg_qc.format_digest(banner.digest)


# ---------------------------------------------------------------------------
# 3. what it costs
# ---------------------------------------------------------------------------

def test_drawing_the_banner_never_scores_a_mask(banner, clean_project, monkeypatch):
    """Opening Measure must cost a CSV read, not a plate of masks."""
    def _explode(*_a, **_k):
        raise AssertionError("the banner scored a mask to draw itself")

    monkeypatch.setattr(seg_qc, "score_masks", _explode)
    monkeypatch.setattr(seg_qc, "score_field", _explode)

    _set(banner._screen, "src", clean_project)
    banner.refresh()
    banner.refresh()
    assert banner.digest.verdict == "ok"


def test_a_second_refresh_reuses_the_parsed_card(banner, clean_project):
    """Returning to the screen ten times must not re-parse ten times."""
    reads = []
    real = seg_qc.read_scorecard

    def _counted(path):
        reads.append(path)
        return real(path)

    banner._reader = None
    _set(banner._screen, "src", clean_project)
    banner.refresh()
    import spacr.seg_qc as module
    original, module.read_scorecard = module.read_scorecard, _counted
    try:
        banner.refresh()
        banner.refresh()
    finally:
        module.read_scorecard = original
    assert reads == [], "the card was re-parsed although nothing changed"


def test_the_scoring_result_is_handled_on_the_gui_thread(
        qtbot, qapp, banner, unscored_project):
    """``PipelineWorker.finished`` is emitted *in the worker thread*.

    PySide6 invokes a plain closure connected to it directly, on that same
    thread — and this handler builds and deletes QLabels. So the completion
    slot is a bound method of a GUI-thread QObject, which Qt queues instead;
    if that ever regresses to a lambda, widgets would be built off the GUI
    thread and this test is the only thing that would notice.
    """
    from PySide6.QtCore import QThread

    seen = {}
    original = banner._on_scored

    def _spy(box):
        seen["thread"] = QThread.currentThread()
        return original(box)

    banner._on_scored = _spy
    _set(banner._screen, "src", unscored_project)
    with qtbot.waitSignal(banner.refreshed, timeout=60000):
        banner._on_score_clicked()

    assert seen["thread"] is qapp.thread()


def test_scoring_from_the_banner_writes_the_card_and_updates_the_verdict(
        qtbot, banner, unscored_project):
    """The one expensive path: only on request, on a worker thread, and it
    leaves a card behind so the next visit is cheap again."""
    _set(banner._screen, "src", unscored_project)
    banner.refresh()
    assert banner.digest.verdict == "missing"

    with qtbot.waitSignal(banner.refreshed, timeout=60000):
        banner._on_score_clicked()

    assert banner.digest.verdict in ("ok", "warn", "fail")
    assert (Path(unscored_project) / "qc" / "segmentation_qc_cell.csv").is_file()
    assert banner._btn_score.isEnabled()
    assert seg_qc.read_digest(unscored_project).verdict == banner.digest.verdict


# ---------------------------------------------------------------------------
# 4. the diameter estimator
# ---------------------------------------------------------------------------

def _disc_plane(shape=(384, 384), radius=12, amp=3000.0, background=200.0,
                noise=25.0, seed=0):
    """A field of discs of exactly known radius on a noisy background."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, background, np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    step, margin = 4 * radius, radius + 2
    for cy in range(margin, shape[0] - margin, step):
        for cx in range(margin, shape[1] - margin, step):
            img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] += amp
    img += rng.normal(0.0, noise, shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


@pytest.fixture
def image_source(tmp_path):
    """5 fields of discs of radius 12, i.e. a true diameter of 24 px."""
    stack = tmp_path / "stack"
    stack.mkdir()
    for i in range(5):
        arr = np.stack([_disc_plane(seed=i)], axis=-1)
        np.save(stack / f"plate1_A{i + 1:02d}_1_t0.npy", arr)
    return str(tmp_path)


@pytest.fixture
def panel(qtbot, registered):
    screen = _screen(qtbot, "mask")
    found = prerun.diameter_panel(screen)
    assert found is not None
    return found


def test_the_panel_is_on_the_mask_screen_above_the_run_row(panel):
    screen = panel._screen
    layout = screen._runtime_wrap.layout()
    assert layout.indexOf(panel) >= 0
    assert layout.indexOf(panel) < layout.indexOf(screen._actions_row)
    assert panel.objectName() == prerun.DIAMETER_OBJECT_NAME
    # Why the number matters, on screen, next to the button that measures it.
    assert "30/diameter" in _texts(panel)


def test_discs_of_known_size_are_measured_and_the_evidence_is_shown(
        qtbot, panel, image_source):
    """Plant objects of a known size; the suggestion has to find them, and it
    has to say how many objects it found them in."""
    _set(panel._screen, "src", image_source)
    _set(panel._screen, "cell_channel", 0)

    with qtbot.waitSignal(panel.estimated, timeout=120000) as caught:
        panel._on_measure_clicked()
    assert caught.args[0] == ["cell"]

    est = panel.estimates["cell"]
    assert est.usable, est.note
    assert est.diameter == pytest.approx(24.0, rel=0.20), est.note
    assert est.n_objects >= 10
    assert est.n_fields == 5

    text = _texts(panel)
    assert f"measured on {est.n_objects} object(s)" in text
    assert f"across {est.n_fields} field(s)" in text
    assert "10th-90th percentile" in text
    assert est.confidence in text
    assert est.method in text


def test_use_writes_the_measurement_into_the_settings_field(
        qtbot, panel, image_source):
    _set(panel._screen, "src", image_source)
    _set(panel._screen, "cell_channel", 0)
    with qtbot.waitSignal(panel.estimated, timeout=120000):
        panel._on_measure_clicked()

    before = panel._screen._settings_model.collect().get("cell_diameter")
    assert panel.apply("cell") is True
    after = panel._screen._settings_model.collect().get("cell_diameter")

    assert after != before
    assert isinstance(after, int), (
        "cell_diameter is declared int; a float leaks through collect() as a "
        f"string, and this came back {after!r}")
    assert after == pytest.approx(panel.estimates["cell"].diameter, abs=1)


def test_nothing_is_written_until_the_user_presses_use(qtbot, panel, image_source):
    model = panel._screen._settings_model
    before = model.collect().get("cell_diameter")
    _set(panel._screen, "src", image_source)
    _set(panel._screen, "cell_channel", 0)
    with qtbot.waitSignal(panel.estimated, timeout=120000):
        panel._on_measure_clicked()
    assert model.collect().get("cell_diameter") == before


def test_an_unusable_estimate_is_never_written(qtbot, registered):
    """A NaN diameter exists precisely so a fabricated number cannot leak."""
    from spacr.diameter import DiameterEstimate

    screen = _screen(qtbot, "mask")
    panel = prerun.diameter_panel(screen)
    panel._estimates = {"cell": DiameterEstimate(
        object_type="cell", diameter=float("nan"), low=float("nan"),
        high=float("nan"), n_objects=0, n_fields=0, method="none",
        confidence="low", note="no usable signal in channel 0")}
    panel._draw_rows()

    assert panel.apply("cell") is False
    assert screen._settings_model.collect().get("cell_diameter") in (None, "")
    text = _texts(panel)
    assert "cell: no estimate" in text
    assert "no usable signal" in text


def test_a_measurement_with_no_source_says_so_instead_of_running(panel):
    _set(panel._screen, "src", "")
    panel._on_measure_clicked()
    assert not panel.busy
    assert "Point src at a plate folder" in _texts(panel)


def test_a_measurement_with_no_channel_names_the_settings_to_fill(
        panel, image_source):
    _set(panel._screen, "src", image_source)
    for key in ("cell_channel", "nucleus_channel", "pathogen_channel"):
        panel._screen._settings_model.set_value_for_key(key, None)
    panel._on_measure_clicked()
    assert not panel.busy
    text = _texts(panel)
    assert "cell_channel" in text
    assert "0-based" in text


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------

def test_the_findings_box_paints_nothing_of_its_own(banner, qt_theme_applied):
    """The container behind the findings text must not paint a black box.

    What this defends: `_findings_box` is a plain QWidget used only to hold a
    layout, and a plain QWidget inherits the blanket
    `QWidget { background-color: bg }` rule. `bg` is the WINDOW colour --
    #000000 on the dark theme -- so the verdict text sat on a solid black
    rectangle inside a panel whose own background follows the user's page
    opacity. That is INVARIANTS 3, and the visible symptom was "the text is
    on top of a black box".

    Asserted on the property and the rule rather than on pixels, because
    QWidget.render() cannot reproduce paint-ordering bugs (INVARIANTS 7) and
    reported a clean page four times for a screen that was black on the
    user's display.
    """
    from spacr.qt import theme

    box = banner._findings_box
    assert box.property(theme.TRANSPARENT_PROPERTY) is True, (
        "the findings container is not tagged transparent, so it paints the "
        "window colour over the panel behind it")
    # The tag is only worth anything if the stylesheet acts on it.
    sheet = theme.stylesheet()
    assert f'[{theme.TRANSPARENT_PROPERTY}="true"]' in sheet
    # ...and a plain QWidget really does lack the tag, so the assertion above
    # is distinguishing something.
    from PySide6.QtWidgets import QWidget
    assert QWidget().property(theme.TRANSPARENT_PROPERTY) is None


def test_the_diameter_panel_rows_box_paints_nothing_either(panel,
                                                           qt_theme_applied):
    """Same container, same rule, the other panel. Fixing one and leaving the
    other is how the black box came back the last four times."""
    from spacr.qt import theme

    assert panel._rows_box.property(theme.TRANSPARENT_PROPERTY) is True


def test_the_panel_stylesheet_is_in_the_sheet_at_launch(qt_theme_applied):
    """The QC panel's own QSS block must be registered at IMPORT time.

    It used to be registered only from `prerun.register()`. That happens to
    run before the sheet is applied today, so this was not the cause of the
    black box -- but it made the block's presence depend on a call order
    nothing states, and INVARIANTS 1 is the record of what that costs. The
    module now registers at import and is listed in
    `theme.WIDGET_QSS_MODULES`, so the block is there however the app starts.
    """
    from spacr.qt import theme

    assert "spacr.qt.prerun" in theme.WIDGET_QSS_MODULES

    # Not asserted against a bare `theme.stylesheet()`: `prerun.teardown()`
    # calls `unregister_widget_qss`, and the module-level registration cannot
    # undo that -- import-time code runs once per process, so a test that has
    # torn the module down leaves every later test without the block. That is
    # an isolation leak of the INVARIANTS 5 family and it is why this asserts
    # the CYCLE instead: register() must put the block back.
    prerun.register()
    assert "MeasureQCBanner" in theme.stylesheet()
