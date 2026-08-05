"""Tests for the Model Compare screen.

Offscreen, CPU-only and offline: the segmentation callable is injected, so no
Cellpose model is constructed and nothing is downloaded. That the screen *has*
that seam is itself one of the properties under test — it is what the Model Zoo
will reuse, and it is why this file can exist at all.

The suite pins the four things the panel lives or dies by:

* it **shows what reached the model** — an argument Cellpose 4 accepts and then
  ignores has to appear in the parameter table as a no-op, or a comparison that
  changed nothing reads as a comparison that found nothing;
* it **runs off the GUI thread**, and the synchronous path used by these tests
  behaves identically to the threaded one;
* it **reports inline** — a missing folder, an unparseable settings line and a
  worker traceback all land in the status label, never in a modal dialog (an
  autouse fixture makes a QMessageBox an immediate test failure);
* it **colours the two previews by correspondence**, not by label id, because
  label 3 in A has nothing to do with label 3 in B.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.screens.model_compare import (
    FIELD_RANGE,
    ModelCompareScreen,
    compose_overlay,
    parse_extra,
    to_display_gray,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_run_journal(monkeypatch, tmp_path):
    """Keep this screen's manifests out of the user's real run history.

    ``_run_job`` goes through :func:`spacr.qt.bridge.make_thread`, which
    journals by default, so every load and every comparison writes a
    reproducibility record into ``~/.spacr/runs``. Left alone the suite
    buries the records of actual analyses under its own debris (measured:
    1100 of 1173 folders in the developer's history). Same isolation as
    ``test_threading_cancellation_audit``.
    """
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    ``MakeMasksScreen._load_current`` once hung the whole headless suite on a
    QMessageBox; this fixture makes that failure mode impossible to reintroduce
    here without a red test.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


@pytest.fixture(autouse=True)
def _never_load_a_real_model(monkeypatch):
    """Belt and braces: constructing a Cellpose model in this file is a failure.

    Every test injects its own segmentation callable. If one ever forgets, this
    turns a 2 GB download into an assertion instead of a hung CI job.
    """
    import spacr.model_compare as mc

    def _boom(*_a, **_k):
        raise AssertionError(
            "the screen tried to load a real Cellpose model — inject a "
            "segment_fn instead")

    monkeypatch.setattr(mc, "segment_with_cellpose", _boom)
    yield


def a_field(size: int = 60) -> np.ndarray:
    """A deterministic field with two bright blobs on a dim background."""
    image = np.full((size, size), 100.0, dtype=np.float32)
    image[5:20, 5:20] = 900.0
    image[35:50, 35:50] = 700.0
    return image


def mask_two_objects(size: int = 60) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.int32)
    mask[5:20, 5:20] = 1
    mask[35:50, 35:50] = 2
    return mask


def mask_split_second(size: int = 60) -> np.ndarray:
    """The same field, but the second object has shattered into two pieces."""
    mask = np.zeros((size, size), dtype=np.int32)
    mask[5:20, 5:20] = 1
    mask[35:44, 35:50] = 2
    mask[44:50, 35:50] = 3
    return mask


class FakeSegmenter:
    """Stands in for Cellpose and records what it was asked to run."""

    def __init__(self, by_name):
        self.by_name = by_name
        self.calls = []

    def __call__(self, images, config):
        self.calls.append((config.name, len(images), config.eval_kwargs()))
        template = self.by_name[config.name]
        return [template.copy() for _ in images]


@pytest.fixture
def fields(tmp_path):
    """A folder of three ``.npy`` fields, the shape spaCR leaves on disk."""
    folder = tmp_path / "plate1"
    folder.mkdir()
    for i in range(3):
        np.save(folder / f"field_{i}.npy", a_field())
    return str(folder)


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — the comparison runs inline so assertions are exact."""
    widget = ModelCompareScreen(threaded=False)
    widget.set_segment_fn(FakeSegmenter({"A": mask_two_objects(),
                                         "B": mask_split_second()}))
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# construction and registration
# ---------------------------------------------------------------------------

def test_the_screen_builds_offscreen(screen):
    assert screen.source_folder() == ""
    assert screen.report() is None
    assert screen.metric_rows() == [] and screen.parameter_rows() == []
    assert "Neither model is treated as ground truth" in screen.status_text()
    assert screen.last_error == ""
    assert FIELD_RANGE[0] >= 1


def test_the_default_is_three_fields(screen):
    assert screen._fields_box.value() == 3


def test_the_two_panels_start_on_different_settings(screen):
    """A screen that opens with both sides identical invites the one comparison
    that cannot show anything."""
    config_a, config_b = screen.model_configs()
    assert config_a.name == "A" and config_b.name == "B"
    assert config_a.diameter != config_b.diameter


def test_the_screen_is_registered_under_segmentation_models(
        qtbot, qt_theme_applied):
    """Back under Segmentation models, and separately marked alpha.

    #16i staged that whole section by maturity — Model Compare and Model
    Zoo into alpha, the three Cellpose apps into beta — and emptied the
    section doing it. Maturity is a colour now, so the section is intact
    and the stage is asserted on its own axis."""
    from spacr.qt.app import APPS, _icon_for_app
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES

    entry = next((a for a in APPS if a[0] == "model_compare"), None)
    assert entry is not None, "model_compare missing from APPS"
    key, name, description, section = entry
    assert name == "Model Compare"
    from spacr.qt.app import SECTION_MODELS, app_stage
    assert section == SECTION_MODELS
    # `spacr.qt.maturity` reassessed every alpha module against the
    # evidence in the repository and this one no longer qualifies; the
    # reason is recorded beside the decision. Applied here because the
    # promotions land in `register_self_registering_modules`, which every
    # launch calls but a bare test process may not have. `apply` alone,
    # not the whole registration pass: it touches only APP_STAGE, so it
    # cannot re-register a module a test has deliberately removed.
    from spacr.qt import maturity
    maturity.apply()
    assert app_stage(key) == "stable"
    assert description
    assert APP_TITLES[key] == "Model Compare"
    assert APP_INTROS[key]
    assert not _icon_for_app(key).isNull()


# ---------------------------------------------------------------------------
# loading fields
# ---------------------------------------------------------------------------

def test_loading_a_folder_lists_its_fields(screen, fields, qtbot):
    with qtbot.waitSignal(screen.fields_loaded, timeout=1000) as caught:
        assert screen.set_source(fields) is True
    assert caught.args == [fields, 3]
    assert screen.field_names() == ["field_0", "field_1", "field_2"]
    assert screen.source_folder() == fields
    assert "Loaded 3 field(s)" in screen.status_text()
    assert screen.last_error == ""
    assert screen._btn_compare.isEnabled()


def test_a_bad_folder_is_reported_inline_and_never_in_a_dialog(screen, tmp_path):
    """The autouse fixture would fire on a QMessageBox; this must stay text."""
    assert screen.set_source(str(tmp_path / "does-not-exist")) is False
    assert "no such folder" in screen.status_text()
    assert screen.last_error == screen.status_text()
    assert screen.field_names() == []
    assert not screen._btn_compare.isEnabled()


def test_a_folder_with_no_images_is_reported_inline(screen, tmp_path):
    empty = tmp_path / "csvs"
    empty.mkdir()
    (empty / "settings.csv").write_text("a,b\n1,2\n")
    assert screen.set_source(str(empty)) is False
    assert "no readable field" in screen.status_text()
    assert screen.report() is None


def test_the_load_button_opens_whatever_was_typed(screen, fields, qtbot):
    screen._path_edit.setText(fields)
    qtbot.mouseClick(screen._btn_load, __import__("PySide6").QtCore.Qt.LeftButton)
    assert screen.field_names() == ["field_0", "field_1", "field_2"]


def test_the_folder_picker_loads_what_it_returns(screen, fields, monkeypatch):
    """The picker is a non-modal file dialog; a cancelled one must change
    nothing. The autouse fixture makes the unpatched call an error, so this is
    the only place the button's own code path runs."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *_a, **_k: ""))
    screen._pick_folder()
    assert screen.field_names() == []

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *_a, **_k: fields))
    screen._pick_folder()
    assert screen.field_names() == ["field_0", "field_1", "field_2"]


def test_changing_the_field_count_reloads_the_folder(screen, fields):
    assert screen.set_source(fields)
    screen._fields_box.setValue(2)
    assert screen.field_names() == ["field_0", "field_1"]


# ---------------------------------------------------------------------------
# comparing
# ---------------------------------------------------------------------------

def test_comparing_fills_the_metric_table_from_a_mocked_run(screen, fields):
    assert screen.set_source(fields)
    assert screen.compare() is True

    segmenter = screen._segment_fn
    assert [name for name, _, _ in segmenter.calls] == ["A", "B"]
    assert [n for _, n, _ in segmenter.calls] == [3, 3]

    rows = screen.metric_rows()
    assert len(rows) == 3
    header = ["field", "A objects", "B objects", "Δ", "ARI"]
    assert rows[0][0] == "field_0"
    assert rows[0][1] == "2" and rows[0][2] == "3"
    assert rows[0][3] == "+1"
    assert len(rows[0]) == screen._row_table.columnCount() == 13
    assert header  # the column order the assertions above assume

    report = screen.report()
    assert report is not None and report.n_fields == 3
    assert report.total_splits == 3 and report.total_new_objects_b == 0
    assert "B found 3 more" in screen.summary_text()
    assert "mean ARI" in screen.status_text()
    assert screen.last_error == ""


def test_a_split_reaches_the_table_as_a_split_not_as_a_new_object(screen, fields):
    """The screen must not undo the module's attribution: three extra objects
    that are fragments are not three discoveries."""
    assert screen.set_source(fields)
    assert screen.compare()
    row = screen.metric_rows()[0]
    splits, merges, only_b, only_a = row[7], row[8], row[9], row[10]
    assert (splits, merges, only_b, only_a) == ("1", "0", "0", "0")


def test_the_parameter_table_shows_what_reached_the_model(screen, fields):
    assert screen.set_source(fields)
    screen._panel_a.diameter_box.setValue(30.0)
    screen._panel_b.diameter_box.setValue(60.0)
    assert screen.compare()

    rows = {row[0]: row for row in screen.parameter_rows()}
    assert rows["diameter"][1] == "30" and rows["diameter"][2] == "60"
    assert rows["diameter"][3] == "yes — varied by this run"
    assert rows["model"][3] == "yes"
    assert rows["flow_threshold"][3] == "yes"


def test_an_ignored_argument_is_visible_rather_than_mysterious(screen, fields):
    """diam_mean looks like the size knob and does nothing. The screen has to
    say so, or a run that changed nothing reads as two equivalent models."""
    assert screen.set_source(fields)
    screen._panel_b.extra_edit.setText("diam_mean=17")
    assert screen.compare()

    rows = {row[0]: row for row in screen.parameter_rows()}
    assert "diam_mean" in rows
    assert rows["diam_mean"][1] == "-" and rows["diam_mean"][2] == "17"
    assert rows["diam_mean"][3].startswith("no —")
    assert "not used in v4.0.1+" in rows["diam_mean"][3]

    # …and it never reached the model.
    _, _, kwargs_b = screen._segment_fn.calls[1]
    assert "diam_mean" not in kwargs_b


def test_a_legacy_model_name_is_shown_as_remapped(screen, fields):
    """cyto2 versus cyto3 with everything else equal is one model against
    itself, and the banner has to say so before anybody reads a number."""
    assert screen.set_source(fields)
    screen._panel_a.model_edit.setText("cyto2")
    screen._panel_b.model_edit.setText("cyto3")
    screen._panel_b.diameter_box.setValue(screen._panel_a.diameter_box.value())
    assert screen.compare()

    warning = screen.warning_text()
    assert "predates Cellpose-SAM" in warning
    assert "same model with the same settings" in warning
    rows = {row[0]: row for row in screen.parameter_rows()}
    assert rows["model"][1] == "cpsam" and rows["model"][2] == "cpsam"
    assert rows["model"][3] == "yes"
    # what was asked for is on screen next to what will load, so "no
    # difference" cannot be mistaken for "these two models are equivalent"
    assert rows["model (requested)"][1] == "cyto2"
    assert rows["model (requested)"][2] == "cyto3"
    assert rows["model (requested)"][3].startswith("no —")


def test_comparing_without_a_folder_is_reported_inline(screen):
    assert screen.compare() is False
    assert "Load a folder of fields first" in screen.status_text()
    assert screen.report() is None


def test_an_unparseable_extra_line_is_reported_inline(screen, fields):
    assert screen.set_source(fields)
    screen._panel_a.extra_edit.setText("diam_mean 17")
    assert screen.compare() is False
    assert "not a key=value pair" in screen.status_text()
    assert screen.last_error == screen.status_text()
    assert screen.report() is None


def test_a_failing_segmentation_lands_in_the_status_label(screen, fields):
    assert screen.set_source(fields)

    def explode(images, config):
        raise RuntimeError("CUDA out of memory")

    screen.set_segment_fn(explode)
    assert screen.compare() is False
    assert "Comparison failed: CUDA out of memory" in screen.status_text()
    assert screen.report() is None
    assert screen.metric_rows() == []


def test_a_second_comparison_replaces_the_first(screen, fields):
    assert screen.set_source(fields)
    assert screen.compare()
    first = screen.report()
    screen.set_segment_fn(FakeSegmenter({"A": mask_two_objects(),
                                         "B": mask_two_objects()}))
    assert screen.compare()
    assert screen.report() is not first
    assert screen.report().identical_masks
    assert len(screen.metric_rows()) == 3


# ---------------------------------------------------------------------------
# the side-by-side preview
# ---------------------------------------------------------------------------

def test_the_two_masks_are_drawn_side_by_side(screen, fields):
    assert screen.set_source(fields)
    assert screen.compare()

    size_a, size_b = screen.preview_sizes()
    assert size_a[0] > 0 and size_b[0] > 0
    caption_a, caption_b = screen.preview_captions()
    assert "field_0" in caption_a and "field_0" in caption_b
    assert "2 object(s)" in caption_a
    assert "3 object(s)" in caption_b
    assert "with a partner" in caption_a


def test_selecting_another_field_redraws_both_panels(screen, fields):
    assert screen.set_source(fields)
    assert screen.compare()
    assert screen.select_field(2) is True
    assert all("field_2" in caption for caption in screen.preview_captions())
    assert screen.select_field(99) is False


def test_a_run_that_kept_no_masks_says_so_instead_of_crashing(screen, fields):
    assert screen.set_source(fields)
    assert screen.compare()
    screen.report().masks_a = []
    assert screen.select_field(0) is False
    assert "nothing to draw" in screen.preview_captions()[0]


def test_a_mask_that_cannot_be_drawn_leaves_a_message_not_a_traceback(screen,
                                                                     fields):
    assert screen.set_source(fields)
    assert screen.compare()
    screen.report().masks_a[0] = np.zeros((4, 4, 4), dtype=np.int32)
    assert screen.select_field(0) is False
    assert screen.preview_sizes()[0] == (0, 0)
    assert screen.preview_sizes()[1][0] > 0        # B still drew


# ---------------------------------------------------------------------------
# the drawing helpers, which are plain numpy
# ---------------------------------------------------------------------------

def test_the_overlay_colours_by_correspondence_not_by_label_id():
    """Label 3 in A has nothing to do with label 3 in B, so a shared palette
    would invite exactly the wrong comparison."""
    mask = mask_two_objects()
    composed = compose_overlay(a_field(), mask, matched=[1])

    assert composed.shape == (60, 60, 3) and composed.dtype == np.uint8
    matched_pixel = composed[10, 10].astype(int)
    unmatched_pixel = composed[40, 40].astype(int)
    background_pixel = composed[30, 5].astype(int)
    assert list(matched_pixel) != list(unmatched_pixel)
    # teal for a partner, amber for none: blue beats red on one, red on the other
    assert matched_pixel[2] > matched_pixel[0]
    assert unmatched_pixel[0] > unmatched_pixel[2]
    assert background_pixel[0] == background_pixel[1] == background_pixel[2]


def test_the_overlay_survives_a_missing_or_mismatched_image():
    mask = mask_two_objects()
    assert compose_overlay(None, mask) is not None
    assert compose_overlay(np.zeros((5, 5)), mask) is not None
    assert compose_overlay(a_field(), np.zeros((0, 0))) is None
    assert compose_overlay(a_field(), np.zeros((3, 3, 3))) is None
    assert compose_overlay(a_field(), np.full((60, 60), -1)) is None


def test_display_gray_stretches_contrast_and_never_raises():
    image = np.full((20, 20), 5.0)
    image[0, 0] = 1000.0
    out = to_display_gray(image, (20, 20))
    assert out.dtype == np.uint8 and out.max() == 255
    assert to_display_gray(None, (4, 4)).shape == (4, 4)
    assert to_display_gray(np.zeros((3, 3)), (4, 4)).max() == 0
    noisy = np.full((8, 8), np.nan)
    assert to_display_gray(noisy, (8, 8)).shape == (8, 8)
    # a flat field is legitimately black; a multi-channel one is max-projected,
    # so a signal in any one channel survives
    assert to_display_gray(np.full((8, 8), 7.0), (8, 8)).max() == 0
    stack = np.zeros((8, 8, 2))
    stack[0:4, :, 1] = 50
    assert to_display_gray(stack, (8, 8)).max() == 255


def test_extra_settings_parse_into_typed_values():
    assert parse_extra("") == {}
    assert parse_extra("  ") == {}
    parsed = parse_extra("diam_mean=17, augment=true; niter=None, name=cpsam")
    assert parsed == {"diam_mean": 17, "augment": True, "niter": None,
                      "name": "cpsam"}
    assert parse_extra("tile_overlap=0.25")["tile_overlap"] == 0.25
    with pytest.raises(ValueError, match="not a key=value pair"):
        parse_extra("diam_mean 17")


# ---------------------------------------------------------------------------
# threading
#
# KNOWN INTERMITTENT FAILURE — read this before "fixing" a red run here.
# ---------------------------------------------------------------------------
#
# Symptom: one of the ``_wait_for_jobs_to_retire`` calls below times out with
# ``pytestqt.exceptions.TimeoutError``, and nothing about the screen is wrong
# — re-running is green. Measured 2 failures in 40 runs of this file's
# threaded subset, and 10-ish in 60 on a loaded machine. It is the same shape
# a reviewer hit in ``tests/qt/test_db_browser.py`` (same ``active_jobs() ==
# 0`` wait, same 10 s budget), so it is not specific to this screen.
#
# Two causes were measured, and they are separate:
#
# 1. **The first ``PipelineWorker`` in a process imports matplotlib on the
#    worker thread.** ``spacr/qt/bridge.py`` line ~747 does ``import
#    matplotlib`` / ``import matplotlib.pyplot`` inside ``run()``, before the
#    job body. ``faulthandler.dump_traceback_later`` caught the worker
#    sitting in that import; with a cold page cache the job body did not
#    start for **10.4-11.3 s** (three measurements). The waits here allow
#    10 000 ms, so the first threaded test in a fresh process is a coin flip
#    against a one-off import. Under pytest ``matplotlib.pyplot`` is *not*
#    already in ``sys.modules`` when the first threaded test runs — checked.
#    ``_matplotlib_is_warm`` below takes that one-off out of the measured
#    window; it is not a sleep and not a retry, it just stops these tests
#    from timing an import they are not about. The product behaviour it
#    reflects — the first Run of a session pays a ~10 s import on the worker
#    thread — is real and belongs to ``spacr/qt/bridge.py``.
#
# 2. **``_retire_job`` is wired through a bare closure**, and at least once it
#    provably did not run. ``ModelCompareScreen._run_job`` does
#    ``thread.finished.connect(lambda t=thread: self._retire_job(t))``.
#    An instrumented failure captured this end state:
#
#        active_jobs=1  busy=False  pending=0
#        thread wrapper: RuntimeError "Internal C++ object already deleted"
#        registry=0
#        after 50 further manual processEvents/DeferredDelete pumps: still 1
#
#    So ``thread.finished`` *was* delivered — ``thread.deleteLater`` ran (the
#    wrapper is gone) and ``RunHandle.retire`` ran (the registry is empty) —
#    but the screen's closure did not, and no amount of further pumping
#    recovered it. Both slots that survived are bound methods of long-lived
#    QObjects. ``make_thread`` already writes the rule down in its own
#    comment: "The slot is a bound method of a GUI-thread QObject, not a
#    closure". Nine screens break it (``model_compare``, ``agreement``,
#    ``foreign``, ``train_compare``, ``report``, ``batch``, ``model_zoo``,
#    ``convert``, ``plate_view``). Fixing that is a change to
#    ``spacr/qt/screens/*``, not to this file.
#
# What NOT to do: raise the timeout, add a retry, or add a sleep. The 10 s
# budget is what makes cause 1 visible, and a retry would hide cause 2
# entirely.


@pytest.fixture(scope="module", autouse=True)
def _matplotlib_is_warm():
    """Import matplotlib before any worker thread has to.

    See cause 1 in the block above: ``PipelineWorker.run`` imports it
    lazily, on the worker, inside the window these tests are timing.
    Doing it here means the tests measure the screen instead of a cold
    ``import matplotlib.pyplot`` — which was measured at 10.4-11.3 s
    against a 10 000 ms budget.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot  # noqa: F401


def _job_state(widget) -> str:
    """Everything worth knowing when a job refuses to retire."""
    from spacr.qt.bridge import registry

    lines = [f"active_jobs()={widget.active_jobs()} "
             f"is_busy()={widget.is_busy()} "
             f"pending={len(widget._pending)} "
             f"registry={len(registry().active())}"]
    for thread, _worker in list(widget._jobs):
        try:
            state = (f"isRunning={thread.isRunning()} "
                     f"isFinished={thread.isFinished()}")
        except RuntimeError as exc:
            state = f"wrapper already reaped ({exc})"
        lines.append("  thread: " + state)
    return "\n".join(lines)


def _wait_for_jobs_to_retire(qtbot, widget, timeout: int = 10000) -> None:
    """Wait for ``active_jobs()`` to reach 0, reporting *why* it did not.

    Same condition and same budget as the bare ``qtbot.waitUntil`` this
    replaces — the only difference is that a failure prints the state
    described in the block above instead of a bare ``TimeoutError``, so
    the next person does not have to re-derive it.
    """
    try:
        qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=timeout)
    except Exception as exc:
        raise AssertionError(
            f"{type(exc).__name__}: a comparison thread never retired.\n"
            f"{_job_state(widget)}\n"
            "See the KNOWN INTERMITTENT FAILURE block above this test "
            "section before treating this as a new defect.") from None


def _load_threaded(qtbot, widget, fields):
    """Wait for the production asynchronous field loader."""
    with qtbot.waitSignal(widget.job_finished, timeout=10000) as caught:
        assert widget.set_source(fields)
        assert widget.is_busy()
    assert caught.args == [True]
    _wait_for_jobs_to_retire(qtbot, widget)


def test_the_threaded_path_produces_the_same_report(qtbot, qt_theme_applied,
                                                    fields):
    """Segmentation is minutes, so it cannot run on the GUI thread. The
    synchronous path the rest of this file uses has to be the same code."""
    widget = ModelCompareScreen(threaded=True)
    widget.set_segment_fn(FakeSegmenter({"A": mask_two_objects(),
                                         "B": mask_split_second()}))
    qtbot.addWidget(widget)
    _load_threaded(qtbot, widget, fields)

    with qtbot.waitSignal(widget.job_finished, timeout=10000) as caught:
        assert widget.compare() is True
        assert widget.is_busy()
        assert not widget._btn_compare.isEnabled()

    assert caught.args == [True]
    assert not widget.is_busy()
    assert widget._btn_compare.isEnabled()
    assert len(widget.metric_rows()) == 3
    assert widget.report().total_splits == 3
    _wait_for_jobs_to_retire(qtbot, widget)
    widget.close()


def test_a_worker_traceback_becomes_one_inline_line(qtbot, qt_theme_applied,
                                                    fields):
    widget = ModelCompareScreen(threaded=True)

    def explode(images, config):
        raise RuntimeError("the checkpoint is not a Cellpose model")

    widget.set_segment_fn(explode)
    qtbot.addWidget(widget)
    _load_threaded(qtbot, widget, fields)

    with qtbot.waitSignal(widget.job_finished, timeout=10000) as caught:
        widget.compare()
    assert caught.args == [False]
    assert "Comparison failed" in widget.status_text()
    assert "not a Cellpose model" in widget.status_text()
    assert widget.status_text().count("\n") == 0
    assert widget.report() is None
    _wait_for_jobs_to_retire(qtbot, widget)
    widget.close()


def test_closing_mid_run_waits_for_the_worker_instead_of_taking_the_process_down(
        qtbot, qt_theme_applied, fields):
    """A QThread collected while still running kills the interpreter, so the
    close path has to join it — this is the test that a user quitting the
    window during a five-minute segmentation does not crash spaCR."""
    import time

    widget = ModelCompareScreen(threaded=True)

    def slow(images, config):
        time.sleep(0.2)
        return [mask_two_objects() for _ in images]

    widget.set_segment_fn(slow)
    qtbot.addWidget(widget)
    _load_threaded(qtbot, widget, fields)
    widget.compare()
    assert widget.is_busy()

    widget.close()                       # while the worker is still going
    _wait_for_jobs_to_retire(qtbot, widget)


def test_closing_survives_a_thread_whose_c_plus_plus_side_is_already_gone(
        qtbot, qt_theme_applied):
    """Qt deletes the QThread when its event loop exits, so by the time the
    widget closes the Python wrapper can be pointing at nothing. Touching it
    raises RuntimeError, and that must not stop the window from closing."""
    class DeadThread:
        def __init__(self):
            self.raised = 0

        def isRunning(self):
            self.raised += 1
            raise RuntimeError("Internal C++ object already deleted.")

    class LiveThread:
        """A wrapper whose C++ side is still there, and is not running."""

        def __init__(self):
            self.asked = 0

        def isRunning(self):
            self.asked += 1
            return False

    widget = ModelCompareScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.show()
    assert widget.isVisible()

    dead, live = DeadThread(), LiveThread()
    widget._jobs.extend([(dead, None), (live, None)])

    assert widget.close() is True         # accepted, not vetoed
    assert not widget.isVisible()
    # The RuntimeError path was really taken, and the corpse did not stop
    # the sweep. Without these the test stays green for a `closeEvent`
    # that never looked at `_jobs` at all — which is the whole subject.
    assert dead.raised == 1
    assert live.asked == 1


def test_a_second_comparison_is_refused_while_one_is_running(qtbot,
                                                             qt_theme_applied,
                                                             fields):
    widget = ModelCompareScreen(threaded=True)
    widget.set_segment_fn(FakeSegmenter({"A": mask_two_objects(),
                                         "B": mask_split_second()}))
    qtbot.addWidget(widget)
    _load_threaded(qtbot, widget, fields)

    with qtbot.waitSignal(widget.job_finished, timeout=10000):
        widget.compare()
        assert widget.compare() is False
        assert "already running" in widget.status_text()
    _wait_for_jobs_to_retire(qtbot, widget)
    widget.close()
