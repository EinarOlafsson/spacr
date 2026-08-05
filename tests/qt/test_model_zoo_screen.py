"""Tests for the Model Zoo screen.

Offscreen, CPU-only and offline. Nothing here downloads anything: an autouse
fixture replaces ``socket.socket`` with a landmine, the byte source is injected
through :meth:`ModelZooScreen.set_opener`, and the segmentation callable is
injected through :meth:`ModelZooScreen.set_segment_fn`. That those two seams
exist is itself part of what is being tested.

The suite pins the properties the panel lives or dies by:

* it **lists provenance as a column**, spelling ``unknown`` where a model does
  not record what it was trained on — a blank cell reads as "no constraints";
* the download **runs off the GUI thread** and its completion handler runs
  back **on** it (the ``_job_settled`` relay: a plain closure on
  ``worker.finished`` is invoked directly on the worker thread, and this
  screen's handler fills a QPlainTextEdit and two QTableWidgets);
* a **cancel leaves nothing** at the destination and no orphaned QThread — a
  QThread garbage-collected while running takes the process down;
* every error — a bad folder, a checksum mismatch, a corrupt checkpoint,
  nothing selected — lands **inline**, never in a modal dialog (an autouse
  fixture makes a QMessageBox an immediate failure);
* two selected models **hand off** to the Model Compare screen rather than
  growing a second, weaker comparison here.
"""
from __future__ import annotations

import hashlib
import os
import threading

import numpy as np
import pytest

from spacr import model_zoo as zoo
from spacr.qt.screens.model_zoo import (
    DEFAULT_DOWNLOAD_DIR,
    FIELD_RANGE,
    ModelZooScreen,
    compose_labels,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

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
def _no_network(monkeypatch):
    """A real download in this file is a test failure, not a slow test."""
    import socket

    def _boom(*_a, **_k):
        raise AssertionError(
            "the screen attempted a network connection — inject an opener")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom, raising=False)
    yield


@pytest.fixture(autouse=True)
def _never_load_a_real_model(monkeypatch):
    """Belt and braces: constructing a Cellpose model here is a failure."""
    import spacr.model_compare as mc

    def _boom(*_a, **_k):
        raise AssertionError(
            "the screen tried to load a real Cellpose model — inject a "
            "segment_fn instead")

    monkeypatch.setattr(mc, "segment_with_cellpose", _boom)
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


def fake_segment(images, config):
    return [masks_two_objects() for _ in images]


@pytest.fixture
def screen(qtbot):
    """A synchronous screen — jobs run inline so assertions are exact."""
    w = ModelZooScreen(threaded=False)
    qtbot.addWidget(w)
    return w


@pytest.fixture
def local_models(tmp_path):
    """Two Cellpose checkpoints, one with provenance and one without."""
    folder = tmp_path / "screen1" / "models" / "cellpose_model"
    a = folder / "with_provenance.CP_model"
    b = folder / "no_provenance.CP_model"
    write_checkpoint(a)
    write_checkpoint(b, b"other")
    with open(str(folder / "with_provenance.CP_model_settings.csv"), "w") as fh:
        fh.write("Key,Value\nimg_src,/data/hela60x/train\nn_epochs,25\n"
                 "diameter,30\n")
    return tmp_path / "screen1", a, b


@pytest.fixture
def fields(tmp_path):
    """A folder of three ``.npy`` fields — the shape spaCR leaves on disk."""
    folder = tmp_path / "plate1" / "1"
    folder.mkdir(parents=True)
    for i in range(3):
        np.save(folder / f"A01_f{i:02d}.npy", a_field(seed=i))
    return folder


@pytest.fixture
def remote_entry(tmp_path):
    """A catalogue entry whose bytes are served from a local file."""
    source = tmp_path / "server" / "hela_60x.CP_model"
    digest = write_checkpoint(source, b"h" * 4096)
    entry = zoo.ModelEntry(
        key="hela_60x", name="hela_60x.CP_model", source="remote",
        uri=f"file://{source}", sha256=digest,
        size_bytes=source.stat().st_size,
        trained_on="HeLa, 60x, confluent monolayer",
        trained_by="A. Researcher")
    return entry, source, digest


# ---------------------------------------------------------------------------
# construction and listing
# ---------------------------------------------------------------------------

def test_the_screen_builds_offscreen_with_nothing_selected(screen):
    assert screen.entries() == []
    assert screen.selected_entries() == []
    assert screen.last_error == ""
    assert "Scan a folder" in screen.status_text()
    assert screen.detail_text() == ""
    assert FIELD_RANGE == (1, 25)
    assert DEFAULT_DOWNLOAD_DIR.endswith(os.path.join(".spacr", "models"))


def test_scanning_a_folder_lists_its_checkpoints(screen, local_models):
    root, a, b = local_models
    assert screen.scan(str(root), include_catalogue=False) is True
    names = [row[0] for row in screen.rows()]
    assert sorted(names) == ["no_provenance.CP_model",
                             "with_provenance.CP_model"]
    assert screen.last_error == ""
    assert "2 model(s)" in screen.status_text()


def test_the_listing_spells_out_unknown_provenance_and_never_leaves_it_blank(
        screen, local_models):
    """A blank provenance cell reads as 'no constraints'."""
    root, _a, _b = local_models
    screen.scan(str(root), include_catalogue=False)
    by_name = {row[0]: row for row in screen.rows()}
    trained_on = 6

    assert by_name["no_provenance.CP_model"][trained_on] == "unknown"
    assert by_name["no_provenance.CP_model"][trained_on] != ""
    assert "/data/hela60x/train" in by_name["with_provenance.CP_model"][trained_on]
    assert "do not record what they were trained on" in screen.status_text()


def test_selecting_a_model_shows_its_provenance_card(screen, local_models):
    root, _a, _b = local_models
    screen.scan(str(root), include_catalogue=False)
    row = [r[0] for r in screen.rows()].index("no_provenance.CP_model")
    screen.select(row)
    card = screen.detail_text()
    assert "no_provenance.CP_model" in card
    assert "trained on unknown" in card
    assert "does not say what it was trained on" in card


def test_a_folder_that_is_not_there_reports_inline(screen, tmp_path):
    """No exception, no dialog (the autouse fixture would fire), just text."""
    assert screen.scan(str(tmp_path / "nope")) is False
    assert "No such folder" in screen.status_text()
    assert screen.last_error
    # …and the screen is still usable.
    assert screen.scan(str(tmp_path), include_catalogue=False) is True


def test_a_field_folder_that_is_not_there_reports_inline(screen, tmp_path):
    assert screen.set_fields_source(str(tmp_path / "gone")) is False
    assert "no such folder" in screen.status_text().lower()
    assert screen.last_error
    assert screen.field_names() == []


def test_a_field_folder_with_no_images_reports_inline(screen, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("nothing here")
    assert screen.set_fields_source(str(empty)) is False
    assert "no readable field" in screen.status_text()
    assert screen.last_error


# ---------------------------------------------------------------------------
# downloading
# ---------------------------------------------------------------------------

def test_a_download_installs_verified_and_replaces_the_catalogue_row(
        screen, remote_entry, tmp_path):
    entry, source, digest = remote_entry
    dest = tmp_path / "dest"
    screen.set_entries([entry])
    screen.select(0)
    screen._dest_edit.setText(str(dest))

    assert screen.download_selected() is True
    installed, = screen.entries()
    assert installed.source == "local"
    assert installed.sha256 == digest
    assert installed.verified is True
    assert (dest / "hela_60x.CP_model").read_bytes() == source.read_bytes()
    assert [row[5] for row in screen.rows()] == ["verified"]
    # Provenance survived the download; that is the point of the catalogue.
    assert "HeLa, 60x, confluent monolayer" in screen.rows()[0][6]


def test_a_checksum_mismatch_is_reported_inline(screen, remote_entry, tmp_path):
    from dataclasses import replace

    entry, _source, _digest = remote_entry
    dest = tmp_path / "dest"
    dest.mkdir()
    screen.set_entries([replace(entry, sha256="f" * 64)])
    screen.select(0)
    screen._dest_edit.setText(str(dest))

    assert screen.download_selected() is False
    assert "does not match its published checksum" in screen.status_text()
    assert screen.last_error
    assert list(dest.iterdir()) == [], "a mismatched download was installed"
    assert screen.download_progress() == 0


def test_an_entry_with_no_published_checksum_is_refused_until_accepted(
        screen, tmp_path):
    source = tmp_path / "server" / "nohash.CP_model"
    write_checkpoint(source)
    entry = zoo.ModelEntry(key="nohash", name="nohash.CP_model",
                           source="remote", uri=f"file://{source}")
    dest = tmp_path / "dest"
    screen.set_entries([entry])
    screen.select(0)
    screen._dest_edit.setText(str(dest))

    assert screen.download_selected() is False
    assert "no sha256 was published" in screen.status_text()
    assert not dest.exists() or list(dest.iterdir()) == []

    screen._allow_unverified.setChecked(True)
    assert screen.download_selected() is True
    installed, = screen.entries()
    assert installed.verified is False
    assert "proves nothing about where they came from" in screen.status_text()


def test_downloading_twice_lands_as_a_new_version(screen, remote_entry,
                                                  tmp_path):
    entry, _source, _digest = remote_entry
    dest = tmp_path / "dest"
    screen._dest_edit.setText(str(dest))
    for _ in range(2):
        screen.set_entries([entry])
        screen.select(0)
        assert screen.download_selected() is True
    assert sorted(p.name for p in dest.iterdir()) == [
        "hela_60x.CP_model", "hela_60x_v2.CP_model"]


def test_download_needs_exactly_one_selected_model(screen, remote_entry):
    entry, _source, _digest = remote_entry
    assert screen.download_selected() is False
    assert "Select a model to download" in screen.status_text()

    screen.set_entries([entry, entry])
    screen.select(0, 1)
    assert screen.download_selected() is False
    assert "exactly one" in screen.status_text()


def test_a_model_already_here_is_not_offered_for_download(screen, local_models):
    root, _a, _b = local_models
    screen.scan(str(root), include_catalogue=False)
    screen.select(0)
    assert screen._btn_download.isEnabled() is False
    assert screen.download_selected() is False
    assert "already here" in screen.status_text()


def test_cancelling_when_nothing_is_running_says_so(screen):
    assert screen.cancel_download() is False
    assert "Nothing is downloading" in screen.status_text()


# ---------------------------------------------------------------------------
# threading
# ---------------------------------------------------------------------------

def test_field_loading_runs_off_the_gui_thread(
        qtbot, qt_theme_applied, monkeypatch, tmp_path):
    """NAS scans and image decoding must not block Qt."""
    from spacr import model_compare as mc

    screen = ModelZooScreen(threaded=True)
    qtbot.addWidget(screen)
    gui_thread = threading.current_thread()
    observed = {}

    def fake_load(folder, n_fields):
        observed["load"] = threading.current_thread()
        return ["A01_f00"], [np.zeros((8, 8), dtype=np.uint16)]

    monkeypatch.setattr(mc, "load_fields", fake_load)
    with qtbot.waitSignal(screen.job_finished, timeout=10000) as caught:
        assert screen.set_fields_source(str(tmp_path))
        assert screen.is_busy()

    assert caught.args == [True]
    assert observed["load"] is not gui_thread
    assert screen.field_names() == ["A01_f00"]
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)
    screen.close()


def test_the_download_runs_off_the_gui_thread_and_settles_back_on_it(
        qtbot, remote_entry, tmp_path):
    """The ``_job_settled`` relay, tested by thread identity.

    PySide6 delivers a plain closure connected to ``worker.finished`` as a
    direct call *on the worker thread*. The completion handler here fills a
    QPlainTextEdit and two QTableWidgets, which is undefined behaviour off the
    GUI thread — so it must be reached through a bound method of the widget.
    """
    entry, _source, _digest = remote_entry
    dest = tmp_path / "dest"
    screen = ModelZooScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.set_entries([entry])
    screen.select(0)
    screen._dest_edit.setText(str(dest))

    gui_thread = threading.current_thread()
    where = {}

    def opener(uri):
        where["download"] = threading.current_thread()
        return zoo.open_uri(uri)

    screen.set_opener(opener)
    screen.models_listed.connect(
        lambda *_: where.setdefault("settled", threading.current_thread()))

    with qtbot.waitSignal(screen.download_finished, timeout=20000) as blocker:
        assert screen.download_selected() is True
        assert screen.is_busy() is True

    assert blocker.args[0] is True
    assert where["download"] is not gui_thread, "the download blocked the GUI"
    assert where["settled"] is gui_thread, "the handler ran off the GUI thread"
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)
    screen.close()


def test_a_cancelled_download_leaves_nothing_and_no_orphan_thread(
        qtbot, remote_entry, tmp_path):
    """Cancel mid-stream: no file, no ``.part``, no live QThread.

    A QThread garbage-collected while still running takes the process down, so
    "cancelled cleanly" means the thread wound down as well as the file being
    gone.
    """
    entry, _source, _digest = remote_entry
    dest = tmp_path / "dest"
    dest.mkdir()
    screen = ModelZooScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.set_entries([entry])
    screen.select(0)
    screen._dest_edit.setText(str(dest))

    began = threading.Event()
    resume = threading.Event()

    def slow_opener(uri):
        def chunks():
            yield b"PK\x03\x04" + b"x" * 4096
            began.set()
            resume.wait(20)
            for _ in range(200):
                yield b"x" * 4096
        return chunks(), 4096 * 201

    screen.set_opener(slow_opener)

    with qtbot.waitSignal(screen.download_finished, timeout=25000) as blocker:
        assert screen.download_selected() is True
        qtbot.waitUntil(began.is_set, timeout=20000)
        assert screen.cancel_download() is True
        resume.set()

    assert blocker.args == [False, ""]
    assert "cancelled" in screen.status_text().lower()
    assert list(dest.iterdir()) == [], "a cancelled download left a file behind"
    assert screen.entries()[0].source == "remote", "nothing was registered"
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)
    screen.close()


def test_progress_reaches_the_bar(qtbot, remote_entry, tmp_path):
    entry, _source, _digest = remote_entry
    screen = ModelZooScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.set_entries([entry])
    screen.select(0)
    screen._dest_edit.setText(str(tmp_path / "dest"))
    assert screen.download_selected() is True
    assert screen.download_progress() == 100


# ---------------------------------------------------------------------------
# benchmarking
# ---------------------------------------------------------------------------

def test_testing_on_three_fields_fills_the_table_and_draws_the_masks(
        screen, local_models, fields):
    root, _a, _b = local_models
    screen.scan(str(root), include_catalogue=False)
    screen.select(0)
    screen.set_segment_fn(fake_segment)
    assert screen.set_fields_source(str(fields)) is True
    assert screen.field_names() == ["A01_f00", "A01_f01", "A01_f02"]

    assert screen.run_benchmark() is True
    rows = screen.benchmark_rows()
    assert len(rows) == 3
    assert [r[1] for r in rows] == ["2", "2", "2"]
    assert all(r[2] in ("ok", "warn", "fail") for r in rows)
    assert screen.preview_size() != (0, 0)

    result = screen.result()
    assert result is not None
    assert result.fieldset == zoo.fieldset_id(screen.field_names(),
                                             [np.load(fields / f"{n}.npy")
                                              for n in screen.field_names()])


def test_the_benchmark_summary_names_the_field_set_it_is_only_valid_for(
        screen, local_models, fields):
    """A score on your three fields says nothing about anybody else's."""
    root, _a, _b = local_models
    screen.scan(str(root), include_catalogue=False)
    screen.select(0)
    screen.set_segment_fn(fake_segment)
    screen.set_fields_source(str(fields))
    screen.run_benchmark()

    summary = screen.summary_text()
    assert "Field set" in summary
    assert screen.result().fieldset in summary
    assert str(fields) in summary
    assert "not an accuracy" in summary


def test_the_benchmark_needs_a_model_and_some_fields(screen, local_models,
                                                    fields):
    root, _a, _b = local_models
    assert screen.run_benchmark() is False
    assert "Select a model to test" in screen.status_text()

    screen.scan(str(root), include_catalogue=False)
    screen.select(0)
    assert screen.run_benchmark() is False
    assert "folder of fields" in screen.status_text()


def test_a_remote_model_cannot_be_benchmarked_before_it_is_downloaded(
        screen, remote_entry, fields):
    entry, _source, _digest = remote_entry
    screen.set_entries([entry])
    screen.select(0)
    screen.set_fields_source(str(fields))
    assert screen.run_benchmark() is False
    assert "download it first" in screen.status_text()


def test_a_corrupt_checkpoint_reports_the_filename_inline(screen, tmp_path,
                                                          fields):
    bad = tmp_path / "models" / "cellpose_model" / "broken.CP_model"
    os.makedirs(bad.parent, exist_ok=True)
    bad.write_text("<html>404 Not Found</html>")
    screen.set_entries([zoo.entry_from_file(bad)])
    screen.select(0)
    screen.set_segment_fn(fake_segment)
    screen.set_fields_source(str(fields))

    assert screen.run_benchmark() is False
    assert "broken.CP_model" in screen.status_text()
    assert "not a PyTorch checkpoint" in screen.status_text()
    assert screen.last_error


def test_changing_the_field_count_reloads_and_relabels(screen, fields):
    screen.set_fields_source(str(fields))
    assert len(screen.field_names()) == 3
    screen._fields_box.setValue(2)
    assert len(screen.field_names()) == 2
    assert screen._btn_test.text() == "Test on 2 fields"


# ---------------------------------------------------------------------------
# hand-off to Model Compare
# ---------------------------------------------------------------------------

def test_two_selected_models_hand_off_to_the_comparison(screen, local_models,
                                                        fields, qtbot):
    root, _a, _b = local_models
    screen.scan(str(root), include_catalogue=False)
    screen.set_fields_source(str(fields))
    screen.select(0, 1)
    assert len(screen.selected_entries()) == 2
    assert screen._btn_compare.isEnabled() is True

    with qtbot.waitSignal(screen.compare_requested, timeout=2000) as blocker:
        assert screen.compare_selected() is True
    request, = blocker.args
    names = {request["name_a"], request["name_b"]}
    assert names == {"no_provenance.CP_model", "with_provenance.CP_model"}
    assert os.path.isfile(request["model_a"])
    assert os.path.isfile(request["model_b"])
    assert request["folder"] == str(fields)
    assert request["n_fields"] == 3
    assert "neither is treated as ground truth" in screen.status_text()


def test_the_handed_off_screen_is_a_configured_model_compare_screen(
        screen, local_models, fields, qtbot):
    from spacr.qt.screens.model_compare import ModelCompareScreen

    root, a, b = local_models
    screen.scan(str(root), include_catalogue=False)
    screen.set_segment_fn(fake_segment)
    screen.set_fields_source(str(fields))
    screen.select(0, 1)

    compare = screen.build_comparison_screen(threaded=False)
    qtbot.addWidget(compare)
    assert isinstance(compare, ModelCompareScreen)
    config_a, config_b = compare.model_configs()
    assert {config_a.model, config_b.model} == {str(a.resolve()),
                                               str(b.resolve())}
    assert compare.field_names() == screen.field_names()
    # And it runs, on the injected segmenter — no Cellpose anywhere.
    assert compare.compare() is True
    assert compare.report() is not None


def test_comparing_needs_exactly_two_selected(screen, local_models):
    root, _a, _b = local_models
    screen.scan(str(root), include_catalogue=False)
    assert screen.compare_selected() is False
    assert "exactly two" in screen.status_text()
    screen.select(0)
    assert screen.compare_selected() is False
    assert screen.build_comparison_screen() is None


def test_comparing_refuses_a_model_that_is_not_downloaded(screen, remote_entry,
                                                          local_models):
    root, _a, _b = local_models
    entry, _source, _digest = remote_entry
    screen.scan(str(root), include_catalogue=False)
    screen.set_entries(screen.entries() + [entry])
    screen.select(0, 2)
    assert screen.compare_selected() is False
    assert "Download hela_60x.CP_model before comparing" in screen.status_text()


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def test_compose_labels_draws_objects_over_the_field():
    out = compose_labels(a_field(), masks_two_objects())
    assert out.shape == (40, 40, 3)
    assert out.dtype == np.uint8
    # Background is grey (r == g == b); an object is tinted, so it is not.
    assert out[0, 0, 0] == out[0, 0, 1] == out[0, 0, 2]
    assert not (out[8, 8, 0] == out[8, 8, 1] == out[8, 8, 2])


def test_compose_labels_refuses_a_mask_that_is_not_a_label_image():
    assert compose_labels(None, np.zeros((2, 2, 3))) is None
    assert compose_labels(None, np.array([])) is None
    assert compose_labels(None, -np.ones((4, 4))) is None


def test_selecting_a_field_with_no_result_is_a_no_op(screen):
    assert screen.select_field(0) is False
