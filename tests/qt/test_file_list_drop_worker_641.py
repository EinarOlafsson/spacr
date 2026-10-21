"""Settings-list folder drops keep filesystem work away from the GUI thread."""
import os
import threading
from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPointF, QUrl, Qt
from PySide6.QtGui import QDropEvent

from spacr.qt import path_probe
from spacr.qt.widgets import file_list as files


@pytest.fixture
def listing(qtbot):
    widget = files.FilePathListWidget(kind='table')
    qtbot.addWidget(widget)
    widget.show()
    yield widget
    scanner = getattr(widget, '_dnd_scanner', None)
    if scanner is not None:
        scanner.shutdown()


def _drop(widget, paths):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    event = QDropEvent(QPointF(2, 2), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    widget.dropEvent(event)
    return event.isAccepted()


def _folder(tmp_path, name='selected'):
    folder = tmp_path / name
    folder.mkdir()
    for name in ('b.csv', 'ignore.png', 'a.csv'):
        (folder / name).write_text('value\n1\n', encoding="utf-8")
    (folder / 'not-a-file.csv').mkdir()
    path_probe.prime(str(folder), True, want_dir=True)
    return folder


def _gate(monkeypatch, folder):
    original = os.listdir
    main = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()
    threads = []
    def listdir(path):
        if os.fspath(path) == str(folder):
            threads.append(threading.get_ident())
            assert threading.get_ident() != main, 'Folder enumeration ran on the GUI thread'
            entered.set()
            assert release.wait(5), 'Test did not release the filesystem gate'
        return original(path)
    monkeypatch.setattr(files.os, 'listdir', listdir)
    return entered, release, threads


def test_drop_returns_while_enumeration_is_blocked_then_delivers_once(listing, tmp_path, qtbot, monkeypatch):
    folder = _folder(tmp_path)
    entered, release, threads = _gate(monkeypatch, folder)
    values, contents = [], []
    main = threading.get_ident()
    listing.value_changed.connect(lambda: values.append(threading.get_ident()))
    listing.contents_changed.connect(lambda: contents.append(threading.get_ident()))
    try:
        assert _drop(listing, [folder])
        qtbot.waitUntil(entered.is_set)
        assert listing.paths() == []
        assert not values and not contents
        release.set()
        qtbot.waitUntil(lambda: listing.paths() == [str(folder / 'a.csv'), str(folder / 'b.csv')])
        assert values == contents == [main]
        assert threads and all(thread != main for thread in threads)
    finally:
        release.set()


@pytest.mark.parametrize('replace_contents', [False, True])
def test_clear_or_replacement_discards_inflight_drop(listing, tmp_path, qtbot, monkeypatch, replace_contents):
    folder = _folder(tmp_path)
    entered, release, _ = _gate(monkeypatch, folder)
    replacement = tmp_path / 'chosen.csv'
    replacement.write_text('value\n2\n', encoding="utf-8")
    try:
        assert _drop(listing, [folder])
        qtbot.waitUntil(entered.is_set)
        if replace_contents:
            listing.set_value([str(replacement)])
        else:
            listing.clear()
        release.set()
        qtbot.waitUntil(lambda: not listing._dnd_scanner.is_busy())
        assert listing.paths() == ([str(replacement)] if replace_contents else [])
    finally:
        release.set()


def test_concurrent_drop_results_keep_input_order_and_deduplicate(listing, tmp_path, qtbot, monkeypatch):
    first = _folder(tmp_path, 'first')
    second = _folder(tmp_path, 'second')
    entered, release, _ = _gate(monkeypatch, first)
    original = files.os.listdir
    second_entered = threading.Event()
    def listing_directory(path):
        if os.fspath(path) == str(second):
            second_entered.set()
        return original(path)
    monkeypatch.setattr(files.os, 'listdir', listing_directory)
    try:
        assert _drop(listing, [first])
        qtbot.waitUntil(entered.is_set)
        assert _drop(listing, [second, first / 'a.csv'])
        qtbot.waitUntil(second_entered.is_set)
        assert listing.paths() == []
        release.set()
        expected = [str(folder / name) for folder in (first, second) for name in ('a.csv', 'b.csv')]
        qtbot.waitUntil(lambda: listing.paths() == expected)
    finally:
        release.set()


def test_close_cancels_drop_callback(listing, tmp_path, qtbot, monkeypatch, qapp):
    folder = _folder(tmp_path)
    entered, release, _ = _gate(monkeypatch, folder)
    try:
        assert _drop(listing, [folder])
        qtbot.waitUntil(entered.is_set)
        timer = threading.Timer(0.02, release.set)
        timer.start()
        listing.close()
        qtbot.waitUntil(release.is_set)
        listing._dnd_scanner.shutdown()
        qapp.processEvents()
        assert listing.paths() == []
    finally:
        release.set()


def test_scanner_refusal_unblocks_later_drop_without_gui_filesystem_work(listing, tmp_path, qtbot, monkeypatch):
    from spacr.qt import dnd_handlers
    folder = _folder(tmp_path)
    original_scanner = dnd_handlers._scanner_for
    class Refused:
        def __init__(self, fail=False):
            self.fail = fail
        def submit(self, collect, landed):
            if self.fail:
                raise RuntimeError('deliberately refused worker')
            return False
    for scanner in (None, Refused(), Refused(True)):
        monkeypatch.setattr(dnd_handlers, '_scanner_for', lambda owner: scanner)
        assert _drop(listing, [folder])
        assert listing.paths() == []
        assert listing._pending_path_drops == []
    monkeypatch.setattr(dnd_handlers, '_scanner_for', original_scanner)
    assert _drop(listing, [folder])
    qtbot.waitUntil(lambda: listing.paths() == [str(folder / 'a.csv'), str(folder / 'b.csv')])


def test_unreadable_folder_keeps_resolved_files_and_does_not_block_next_drop(
        listing, tmp_path, qtbot, monkeypatch):
    first = tmp_path / 'first.csv'
    last = tmp_path / 'last.csv'
    for path in (first, last):
        path.write_bytes(b'value\n1\n')
    unreadable = _folder(tmp_path, 'unreadable')
    original = os.listdir
    main = threading.get_ident()
    failed_threads = []

    def listdir(path):
        if os.fspath(path) == str(unreadable):
            failed_threads.append(threading.get_ident())
            raise PermissionError('folder access revoked')
        return original(path)

    monkeypatch.setattr(files.os, 'listdir', listdir)
    values, contents = [], []
    listing.value_changed.connect(lambda: values.append((threading.get_ident(), listing.paths())))
    listing.contents_changed.connect(lambda: contents.append((threading.get_ident(), listing.paths())))
    assert _drop(listing, [first, unreadable])
    qtbot.waitUntil(lambda: listing.paths() == [str(first)])
    assert failed_threads and all(thread != main for thread in failed_threads)
    assert values == contents == [(main, [str(first)])]
    assert _drop(listing, [last])
    qtbot.waitUntil(lambda: listing.paths() == [str(first), str(last)])
    assert values == contents == [(main, [str(first)]), (main, [str(first), str(last)])]
    assert first.read_bytes() == last.read_bytes() == b'value\n1\n'


def test_single_file_drop_replaces_once_without_starting_a_folder_worker(
        qtbot, tmp_path, monkeypatch):
    from spacr.qt import dnd_handlers

    paths = [tmp_path / name for name in ('previous.csv', 'first.csv', 'chosen.csv')]
    for path in paths:
        path.write_bytes(b'value\n2\n')
    widget = files.FilePathListWidget(value=str(paths[0]), kind='table', single=True)
    qtbot.addWidget(widget)
    widget.show()
    monkeypatch.setattr(dnd_handlers, '_scanner_for',
                        lambda owner: pytest.fail('single-file choice started folder enumeration'))
    values, contents = [], []
    widget.value_changed.connect(lambda: values.append(widget.get_value()))
    widget.contents_changed.connect(lambda: contents.append(widget.get_value()))
    assert _drop(widget, paths[1:])
    assert widget.get_value() == str(paths[-1])
    assert widget.paths() == [str(paths[-1])]
    assert widget._single_line.text() == str(paths[-1])
    assert values == contents == [str(paths[-1])]
    assert _drop(widget, [paths[-1]])
    assert values == contents == [str(paths[-1])]
    assert all(path.read_bytes() == b'value\n2\n' for path in paths)


class _InlineScanner:
    """Run a drop's folder walk on the calling thread, now or when released.

    The real scanner runs ``collect`` on a QThread, which coverage cannot
    see; this one runs the same closure here so each branch is measured.
    """

    def __init__(self, deferred=False):
        self.deferred = deferred
        self.queued = []

    def submit(self, collect, landed):
        if self.deferred:
            self.queued.append((collect, landed))
        else:
            landed(collect())
        return True

    def release(self):
        for collect, landed in self.queued:
            landed(collect())
        self.queued.clear()


def test_a_drop_expands_folders_in_place_and_keeps_files_as_given(listing, tmp_path, monkeypatch):
    from spacr.qt import dnd_handlers
    folder = _folder(tmp_path)
    loose = tmp_path / 'loose.csv'
    loose.write_text('value\n3\n', encoding="utf-8")
    monkeypatch.setattr(dnd_handlers, '_scanner_for', lambda owner: _InlineScanner())
    changes = []
    listing.contents_changed.connect(lambda: changes.append(listing.paths()))
    assert _drop(listing, [loose, folder])
    expected = [str(loose), str(folder / 'a.csv'), str(folder / 'b.csv')]
    assert listing.paths() == expected
    assert changes == [expected]
    assert listing._pending_path_drops == []


def test_a_walk_that_breaks_keeps_what_it_found_and_frees_the_queue(listing, tmp_path, monkeypatch, caplog):
    import logging
    from spacr.qt import dnd_handlers
    folder = _folder(tmp_path)
    first = tmp_path / 'first.csv'
    first.write_text('value\n1\n', encoding="utf-8")

    def broken(path, kind):
        raise RuntimeError('the folder walk failed')

    monkeypatch.setattr(dnd_handlers, '_scanner_for', lambda owner: _InlineScanner())
    monkeypatch.setattr(files.FilePathListWidget, '_folder_members_for_kind', staticmethod(broken))
    with caplog.at_level(logging.ERROR, logger=files.LOG.name):
        assert _drop(listing, [first, folder])
    assert 'the folder walk failed' in caplog.text
    assert listing.paths() == [str(first)]
    assert listing._pending_path_drops == []


def test_a_drop_that_lands_after_the_list_was_cleared_adds_nothing(listing, tmp_path, monkeypatch):
    from spacr.qt import dnd_handlers
    folder = _folder(tmp_path)
    scanner = _InlineScanner(deferred=True)
    monkeypatch.setattr(dnd_handlers, '_scanner_for', lambda owner: scanner)
    assert _drop(listing, [folder])
    listing.clear()
    changes = []
    listing.contents_changed.connect(lambda: changes.append(listing.paths()))
    scanner.release()
    assert listing.paths() == [] and changes == []
    assert listing._pending_path_drops == []


def test_a_probe_answer_for_a_list_being_torn_down_is_ignored(listing, monkeypatch):
    calls = []

    def gone():
        calls.append('refresh')
        raise RuntimeError('Internal C++ object already deleted')

    monkeypatch.setattr(listing, '_refresh_hint', gone)
    assert listing._path_probe_redraw('/data/plate1.csv', True) is None
    assert calls == ['refresh']
    assert listing.paths() == []


def test_a_list_outliving_the_probe_source_is_still_destroyed_cleanly(qtbot, monkeypatch):
    import shiboken6
    widget = files.FilePathListWidget(kind='table')
    released = []

    class _GoneSignal:
        def disconnect(self, slot):
            released.append(slot)
            raise RuntimeError('Internal C++ object already deleted')

    redraw = widget._path_probe_redraw
    path_probe.probes.answered.disconnect(redraw)
    monkeypatch.setattr(path_probe, 'probes', type('Gone', (), {'answered': _GoneSignal()})())
    shiboken6.delete(widget)
    assert released == [redraw]


def test_a_table_with_no_download_buttons_has_nothing_to_align(qtbot, monkeypatch):
    table = files.PairedFileTableWidget()
    qtbot.addWidget(table)
    monkeypatch.setattr(table, 'DOWNLOAD_BUTTONS', ())
    monkeypatch.setattr(table, '_owning_screen', lambda: object())
    assert table.align_download_buttons() is False
