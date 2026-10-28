"""Explicit Easy-mode image/mask copying preserves sources and field identity."""
from pathlib import Path
import threading

import numpy as np
import pytest
import tifffile

from spacr.qt.screens import make_masks as mm


def _pair(root, name="field.ome.tif", label=70001):
    images = root / "loose_images"
    masks = root / "loose_masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    image = images / name
    mask = masks / name
    tifffile.imwrite(image, np.arange(48, dtype=np.uint16).reshape(6, 8))
    labels = np.zeros((6, 8), np.uint32)
    labels[1:4, 2:5] = label
    tifffile.imwrite(mask, labels)
    return {str(image): str(mask)}, labels


def test_copy_keeps_original_bytes_and_exact_mask_identity(tmp_path):
    pairs, labels = _pair(tmp_path)
    originals = {path: Path(path).read_bytes() for path in (*pairs, *pairs.values())}
    result = mm._copy_paired_layout(pairs, tmp_path / "copy")
    assert result["files"] == ["field.ome.tif"]
    image, mask = mm.engine.load_image_and_mask(
        result["images"], result["files"][0], masks_dir=result["masks"])
    assert image.shape == labels.shape
    np.testing.assert_array_equal(mask, labels)
    assert mask.dtype == np.uint32
    assert all(Path(path).read_bytes() == raw for path, raw in originals.items())
    assert not list(tmp_path.glob(".spacr-paired-*"))


def test_existing_destination_and_ambiguous_stems_are_refused(tmp_path):
    pairs, _ = _pair(tmp_path / "one", name="field.tif")
    other, _ = _pair(tmp_path / "two", name="field.png")
    with pytest.raises(ValueError, match="ambiguous"):
        mm._copy_paired_layout({**pairs, **other}, tmp_path / "ambiguous")
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        mm._copy_paired_layout(pairs, destination)
    assert sentinel.read_text() == "original"
    assert not (tmp_path / "ambiguous").exists()


def test_multiple_masks_claiming_an_image_are_refused(tmp_path):
    pairs, _ = _pair(tmp_path, name="first.tif")
    second, _ = _pair(tmp_path / "second", name="second.tif")
    pairs.update(second)
    wrong = Path(next(iter(second.values()))).with_name("first_mask.tif")
    Path(next(iter(second.values()))).rename(wrong)
    pairs[next(iter(second))] = str(wrong)
    with pytest.raises(ValueError, match="exactly one"):
        mm._copy_paired_layout(pairs, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_failure_and_cancellation_remove_only_new_output(tmp_path, monkeypatch, failure):
    pairs, _ = _pair(tmp_path)
    destination = tmp_path / "copy"
    with pytest.raises(InterruptedError):
        mm._copy_paired_layout(pairs, destination, cancelled=lambda: True)
    assert not destination.exists()
    from spacr import tiff_io
    if failure == "write":
        monkeypatch.setattr(tiff_io, "write_tiff", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    else:
        original_rename = mm.os.rename
        def fail_second(source, target):
            if Path(target).name == "masks":
                raise OSError("disk full")
            return original_rename(source, target)
        monkeypatch.setattr(mm.os, "rename", fail_second)
    with pytest.raises(OSError, match="disk full"):
        mm._copy_paired_layout(pairs, destination)
    assert not destination.exists()
    assert all(Path(path).is_file() for path in (*pairs, *pairs.values()))
    assert not list(tmp_path.glob(".spacr-paired-*"))


def test_bad_mask_dimensions_leave_no_partial_output(tmp_path):
    pairs, _ = _pair(tmp_path)
    tifffile.imwrite(next(iter(pairs.values())), np.zeros((2, 3), np.uint16))
    with pytest.raises(ValueError, match="shape"):
        mm._copy_paired_layout(pairs, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()
    assert not list(tmp_path.glob(".spacr-paired-*"))


@pytest.fixture
def screen(qtbot, qt_theme_applied, monkeypatch):
    monkeypatch.setattr(mm, "is_headless", lambda: True)
    value = mm.MakeMasksScreen()
    qtbot.addWidget(value)
    yield value
    value.close()


def test_confirmation_decline_and_destination_cancel_do_nothing(screen, tmp_path, monkeypatch):
    pairs, _ = _pair(tmp_path)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    monkeypatch.setattr(screen, "_confirm", lambda *a: False)
    monkeypatch.setattr(mm.QFileDialog, "getExistingDirectory", lambda *a, **k: pytest.fail("No chooser after decline"))
    assert not screen._open_paired_layout(pairs)
    monkeypatch.setattr(screen, "_confirm", lambda *a: True)
    monkeypatch.setattr(mm.QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    assert not screen._open_paired_layout(pairs)
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before
    assert getattr(screen, "_paired_layout_job", None) is None
    assert not screen._folder


def test_copy_runs_off_gui_and_opens_selected_copy_not_sibling(screen, tmp_path, monkeypatch, qtbot):
    pairs, labels = _pair(tmp_path)
    sibling = Path(next(iter(pairs))).with_name("unrelated.tif")
    tifffile.imwrite(sibling, np.ones((4, 4), np.uint16))
    destination_parent = tmp_path / "chosen"
    destination_parent.mkdir()
    monkeypatch.setattr(screen, "_confirm", lambda *a: True)
    monkeypatch.setattr(mm.QFileDialog, "getExistingDirectory", lambda *a, **k: str(destination_parent))
    main_thread = threading.get_ident()
    called = []
    original = mm._copy_paired_layout
    def observed(*args, **kwargs):
        called.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(mm, "_copy_paired_layout", observed)
    assert screen._open_paired_layout(pairs)
    qtbot.waitUntil(lambda: getattr(screen, "_paired_layout_job", None) is None, timeout=20000)
    assert called and called[0] != main_thread
    assert screen._folder == str(destination_parent / "make_masks" / "images")
    assert screen._image_files == ["field.ome.tif"]
    assert screen._masks_dir == str(destination_parent / "make_masks" / "masks")
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                   and screen._load_worker is None and screen._canvas.mask is not None)
    np.testing.assert_array_equal(screen._canvas.mask, labels)
    assert sibling.is_file()


def test_late_copy_does_not_replace_a_newer_field(screen, tmp_path, monkeypatch):
    pairs, _ = _pair(tmp_path)
    result = mm._copy_paired_layout(pairs, tmp_path / "copy")
    worker = mm._FolderJobWorker(None, screen)
    worker.result = result
    worker.open_token = screen._load_token - 1
    screen._folder_job = screen._paired_layout_job = worker
    monkeypatch.setattr(screen, "_open_folder", lambda *a, **k: pytest.fail("Stale copy opened"))
    screen._on_paired_layout_finished()
    assert screen._paired_layout_job is None


def test_close_interrupts_a_live_copy_without_opening_result(screen, tmp_path, monkeypatch, qtbot):
    pairs, _ = _pair(tmp_path)
    started = threading.Event()
    monkeypatch.setattr(screen, "_confirm", lambda *a: True)
    monkeypatch.setattr(mm.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    def copy_until_closed(pairs, destination, cancelled):
        started.set()
        while not cancelled():
            threading.Event().wait(0.01)
        raise InterruptedError("cancelled")
    monkeypatch.setattr(mm, "_copy_paired_layout", copy_until_closed)
    monkeypatch.setattr(screen, "_open_folder", lambda *a, **k: pytest.fail("Closed screen reopened"))
    assert screen._open_paired_layout(pairs)
    qtbot.waitUntil(started.is_set, timeout=10000)
    worker = screen._paired_layout_job
    screen.close()
    assert not worker.isRunning()
    qtbot.waitUntil(lambda: screen._paired_layout_job is None, timeout=10000)
    assert not (tmp_path / "make_masks").exists()


def test_empty_pairing_never_prompts_or_changes_the_current_field(screen, monkeypatch):
    token, folder = screen._load_token, screen._folder
    monkeypatch.setattr(screen, '_confirm', lambda *a: pytest.fail('Empty request prompted'))
    assert not screen._open_paired_layout({})
    assert screen._load_token == token and screen._folder == folder
    assert getattr(screen, '_paired_layout_job', None) is None


def test_busy_owned_copy_refuses_a_second_request_and_keeps_originals(
        screen, tmp_path, monkeypatch, qtbot):
    pairs, _ = _pair(tmp_path)
    originals = {path: Path(path).read_bytes() for path in (*pairs, *pairs.values())}
    entered, release = threading.Event(), threading.Event()
    warnings = []
    worker = mm._FolderJobWorker(None, screen)
    worker.job = lambda: (entered.set(), release.wait(5))
    screen._folder_job = worker
    monkeypatch.setattr(screen, '_warn', lambda *a: warnings.append(a))
    monkeypatch.setattr(screen, '_confirm', lambda *a: pytest.fail('Busy copy prompted'))
    worker.start()
    try:
        qtbot.waitUntil(entered.is_set)
        assert not screen._open_paired_layout(pairs)
        assert screen._folder_job is worker and worker.isRunning()
        assert len(warnings) == 1 and 'running' in warnings[0][1]
        assert all(Path(path).read_bytes() == data for path, data in originals.items())
    finally:
        release.set()
        assert worker.wait(5000)
        screen._finish_folder_job(worker, lambda done: None)


def test_copy_publication_error_retires_worker_without_replacing_field_or_sources(
        screen, tmp_path, monkeypatch, qtbot):
    pairs, _ = _pair(tmp_path)
    originals = {path: Path(path).read_bytes() for path in (*pairs, *pairs.values())}
    token, folder = screen._load_token, screen._folder
    warnings = []
    monkeypatch.setattr(screen, '_confirm', lambda *a: True)
    monkeypatch.setattr(mm.QFileDialog, 'getExistingDirectory', lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(screen, '_warn', lambda *a: warnings.append(a))
    original = mm.os.rename
    def refuse_publication(source, destination):
        if Path(destination).name == 'masks':
            raise OSError('publication denied')
        return original(source, destination)
    monkeypatch.setattr(mm.os, 'rename', refuse_publication)
    assert screen._open_paired_layout(pairs)
    qtbot.waitUntil(lambda: screen._paired_layout_job is None, timeout=20000)
    assert screen._folder_job is None and screen._load_token == token and screen._folder == folder
    assert warnings and 'publication denied' in warnings[0][1]
    assert not (tmp_path / 'make_masks').exists() and not list(tmp_path.glob('.spacr-paired-*'))
    assert all(Path(path).read_bytes() == data for path, data in originals.items())


def test_pair_with_an_unsupported_image_dimension_refuses_without_output_or_source_changes(tmp_path):
    pairs, _ = _pair(tmp_path)
    image = Path(next(iter(pairs)))
    tifffile.imwrite(image, np.zeros((2, 6, 8, 5), dtype=np.uint16), photometric='minisblack')
    before = {path: Path(path).read_bytes() for path in (*pairs, *pairs.values())}
    with pytest.raises(ValueError, match='dimensions are incompatible'):
        mm._copy_paired_layout(pairs, tmp_path / 'copy')
    assert not (tmp_path / 'copy').exists() and not list(tmp_path.glob('.spacr-paired-*'))
    assert all(Path(path).read_bytes() == data for path, data in before.items())


def test_duplicate_completion_after_a_real_copy_cannot_reopen_or_change_the_current_field(
        screen, tmp_path, monkeypatch, qtbot):
    pairs, labels = _pair(tmp_path)
    originals = {path: Path(path).read_bytes() for path in (*pairs, *pairs.values())}
    monkeypatch.setattr(screen, '_confirm', lambda *a: True)
    monkeypatch.setattr(mm.QFileDialog, 'getExistingDirectory', lambda *a, **k: str(tmp_path))
    assert screen._open_paired_layout(pairs)
    qtbot.waitUntil(lambda: screen._paired_layout_job is None, timeout=20000)
    token, folder = screen._load_token, screen._folder
    assert folder == str(tmp_path / 'make_masks' / 'images')
    monkeypatch.setattr(screen, '_open_folder', lambda *a, **k: pytest.fail('Completed copy reopened'))
    screen._on_paired_layout_finished()
    assert screen._load_token == token and screen._folder == folder and screen._folder_job is None
    np.testing.assert_array_equal(screen._canvas.mask, labels)
    assert all(Path(path).read_bytes() == data for path, data in originals.items())
