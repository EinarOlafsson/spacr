"""The Model Zoo's quiet branches: the ones that do nothing, and must.

Round-4 coverage for :mod:`spacr.qt.screens.model_zoo`. Everything here is a
path where the screen deliberately declines to act, paired in the same test
with the input that makes it act -- an absence nobody drove to presence is not
evidence of anything:

* changing the field count with **no folder loaded** relabels the Test button
  and loads nothing; with a folder loaded the same edit reloads the fields;
* a benchmark result with **no rows** fills no table and selects no field,
  while one with rows selects the first and draws it;
* a comparison screen built while the zoo has **no injected segmenter and no
  field folder** inherits neither -- it falls back to
  ``model_compare.segment_with_cellpose`` and starts with no fields -- while
  one built after both are set inherits both;
* the job sweep run while a worker is **still in flight** keeps that job, and
  retires it only once its QThread has actually stopped.
"""
from __future__ import annotations

import hashlib
import os
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr import model_zoo as zoo
from spacr.qt.screens.model_zoo import ModelZooScreen


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_run_journal(monkeypatch, tmp_path):
    """Keep this screen's manifests out of the user's real run history."""
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """A modal dialog on any path here would hang a headless run."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


def write_checkpoint(path, payload: bytes = b"weights") -> str:
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"PK\x03\x04" + payload)
    return hashlib.sha256(b"PK\x03\x04" + payload).hexdigest()


def a_field(size: int = 40, seed: int = 0) -> np.ndarray:
    image = np.full((size, size), 100.0, dtype=np.float32)
    image[4:14, 4:14] = 800.0 + seed
    image[24:34, 24:34] = 600.0 + seed
    return image


def masks_two_objects(size: int = 40) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.int32)
    mask[4:14, 4:14] = 1
    mask[24:34, 24:34] = 2
    return mask


@pytest.fixture
def screen(qtbot):
    """A synchronous screen — jobs run inline so assertions are exact."""
    widget = ModelZooScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def local_models(tmp_path):
    """Two Cellpose checkpoints that really are on disk."""
    folder = tmp_path / "screen1" / "models" / "cellpose_model"
    a = folder / "with_provenance.CP_model"
    b = folder / "no_provenance.CP_model"
    write_checkpoint(a)
    write_checkpoint(b, b"other")
    return tmp_path / "screen1", a, b


@pytest.fixture
def fields(tmp_path):
    """A folder of three ``.npy`` fields — the shape spaCR leaves on disk."""
    folder = tmp_path / "plate1" / "1"
    folder.mkdir(parents=True)
    for i in range(3):
        np.save(folder / f"A01_f{i:02d}.npy", a_field(seed=i))
    return folder


def _result(entry, rows, masks=(), images=()):
    return zoo.BenchmarkResult(entry=entry, fieldset="fs", fieldset_label="3",
                               rows=list(rows), masks=list(masks),
                               images=list(images))


# ---------------------------------------------------------------------------
# the field count with, and without, a folder behind it
# ---------------------------------------------------------------------------

def test_the_field_count_relabels_the_button_but_reloads_only_a_real_folder(
        screen, fields):
    """"Test on N fields" is a promise the screen can keep before any folder.

    The count is editable from the first frame, so ``_reload_fields`` runs with
    nothing loaded. It must still relabel the button — and must not invent a
    load out of an empty folder string.
    """
    assert screen.fields_folder() == ""
    screen._fields_box.setValue(2)
    assert screen._btn_test.text() == "Test on 2 fields"
    assert screen.fields_folder() == ""
    assert screen.field_names() == []

    # Same edit, once a folder is behind it: now it reloads.
    assert screen.set_fields_source(str(fields)) is True
    assert len(screen.field_names()) == 2
    screen._fields_box.setValue(3)
    assert screen._btn_test.text() == "Test on 3 fields"
    assert screen.fields_folder() == str(fields)
    assert len(screen.field_names()) == 3


# ---------------------------------------------------------------------------
# a benchmark that scored nothing
# ---------------------------------------------------------------------------

def test_a_benchmark_with_no_rows_selects_no_field_and_draws_nothing(screen):
    """An empty result must not select row 0 of a table that has no row 0.

    ``_apply_benchmark`` is called directly because ``run_benchmark`` refuses
    before it ever starts a job with no fields loaded — the completion handler
    is the only place an empty result can arrive, and it is what is under test.
    """
    entry = zoo.ModelEntry(key="k", name="m.CP_model")

    screen._apply_benchmark(_result(entry, []))
    assert screen.benchmark_rows() == []
    assert screen._bench_table.currentRow() == -1
    assert screen.preview_size() == (0, 0)
    assert "0 cell(s) over 0 field(s)" in screen.status_text()
    assert "Field set fs" in screen.summary_text()

    # The same handler, one row and its mask: now it selects and draws.
    rows = [zoo.FieldBenchmark(field="A01_f00", n_objects=2, severity="ok")]
    screen._apply_benchmark(_result(entry, rows, masks=[masks_two_objects()],
                                    images=[a_field()]))
    assert [r[0] for r in screen.benchmark_rows()] == ["A01_f00"]
    assert screen._bench_table.currentRow() == 0
    assert screen.preview_size() != (0, 0)


# ---------------------------------------------------------------------------
# what a handed-off comparison screen does and does not inherit
# ---------------------------------------------------------------------------

def test_the_comparison_screen_inherits_only_the_seams_the_zoo_actually_has(
        screen, local_models, fields, qtbot, monkeypatch):
    """No injected segmenter and no folder means neither is pushed across.

    Pushing ``None`` as a segment_fn would be harmless; pushing ``""`` as a
    source folder would not — it would start a field load of the current
    directory. Both are proved by behaviour: the bare screen falls back to
    ``model_compare.segment_with_cellpose`` and starts with no fields.
    """
    from spacr import model_compare as mc

    fallback_calls = []

    def _fallback(images, config):
        fallback_calls.append(config.name)
        return [masks_two_objects() for _ in images]

    monkeypatch.setattr(mc, "segment_with_cellpose", _fallback)

    root, _a, _b = local_models
    assert screen.scan(str(root), include_catalogue=False) is True
    screen.select(0, 1)
    assert len(screen.selected_entries()) == 2
    assert screen.fields_folder() == ""

    bare = screen.build_comparison_screen(threaded=False)
    qtbot.addWidget(bare)
    assert bare is not None
    assert bare.source_folder() == ""
    assert bare.field_names() == []
    assert bare.set_source(str(fields)) is True
    assert bare.compare() is True
    assert len(fallback_calls) == 2, "the zoo pushed a segmenter it never had"

    # Now give the zoo both seams: the next screen must inherit both.
    injected = []

    def _injected(images, config):
        injected.append(config.name)
        return [masks_two_objects() for _ in images]

    screen.set_segment_fn(_injected)
    assert screen.set_fields_source(str(fields)) is True
    wired = screen.build_comparison_screen(threaded=False)
    qtbot.addWidget(wired)
    assert wired.source_folder() == str(fields)
    assert wired.field_names() == screen.field_names()

    before = len(fallback_calls)
    assert wired.compare() is True
    assert len(injected) == 2
    assert len(fallback_calls) == before, "the injected segmenter was ignored"


# ---------------------------------------------------------------------------
# the job sweep
# ---------------------------------------------------------------------------

def test_the_job_sweep_keeps_a_thread_that_has_not_stopped_yet(
        qtbot, tmp_path, monkeypatch):
    """Sweeping mid-flight must not drop a live job's only strong reference.

    ``_jobs`` is what keeps the QThread and its worker alive, and a QThread
    garbage-collected while still running takes the process down. The sweep is
    normally reached from ``thread.finished`` — i.e. only ever with a stopped
    thread — so it is called directly here, while a worker is provably still
    inside its job.
    """
    from spacr import model_compare as mc

    screen = ModelZooScreen(threaded=True)
    qtbot.addWidget(screen)

    began = threading.Event()
    resume = threading.Event()

    def slow_load(folder, n_fields):
        began.set()
        resume.wait(20)
        return ["A01_f00"], [np.zeros((8, 8), dtype=np.uint16)]

    monkeypatch.setattr(mc, "load_fields", slow_load)

    with qtbot.waitSignal(screen.job_finished, timeout=30000):
        assert screen.set_fields_source(str(tmp_path)) is True
        assert began.wait(20), "the field-loading job never started"
        screen._retire_finished_jobs()
        assert screen.active_jobs() == 1, "a running job was retired"
        resume.set()

    # …and the same sweep, from thread.finished, does retire it once stopped.
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=30000)
    assert screen.field_names() == ["A01_f00"]
    screen.close()
