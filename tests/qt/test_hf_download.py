"""Tests for :mod:`spacr.qt.hf_download` — the demo-dataset downloader.

Everything here runs **offline**. The two externalities (``requests`` for
the transport and ``huggingface_hub.list_repo_files`` for the manifest)
are stubbed; the module's own URL building, streaming, integrity check,
worker loop, cancellation and Qt thread hand-off are exercised for real.

Three regressions are pinned here:

* the download flow drove the ``QProgressDialog`` from the worker thread
  (Qt printed "Recursive repaint detected" and then segfaulted);
* pressing Cancel could not reach the worker while it was downloading;
* a stream that broke mid-file left a truncated image on the final path.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid       # ships with PySide6

from spacr.qt import hf_download as hf


# ---------------------------------------------------------------------------
# Transport stubs
# ---------------------------------------------------------------------------

class FakeResponse:
    """Just enough of ``requests.Response`` for :func:`_download_one`."""

    def __init__(self, chunks=(), headers=None, status_error=None):
        self._chunks = list(chunks)
        self.headers = dict(headers or {})
        self._status_error = status_error
        self.iter_kwargs = []

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def iter_content(self, chunk_size=None):
        self.iter_kwargs.append(chunk_size)
        for chunk in self._chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


@pytest.fixture
def fake_get(monkeypatch):
    """Install a stub ``requests.get`` and hand back the call log."""
    calls = []

    def _install(response):
        def _get(url, **kwargs):
            calls.append((url, kwargs))
            return response
        monkeypatch.setattr("requests.get", _get)
        return calls
    return _install


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

def test_demo_endpoints_match_the_tk_downloader():
    assert hf.DATASET_REPO == "einarolafsson/toxo_mito"
    assert hf.DATASET_SUB == "plate1"
    assert hf.SETTINGS_REPO == "einarolafsson/spacr_settings"


def test_download_result_carries_both_paths():
    r = hf.DownloadResult(dataset_path=Path("/a/plate1"),
                          settings_path=Path("/a/settings"))
    assert r.dataset_path == Path("/a/plate1")
    assert r.settings_path == Path("/a/settings")


# ---------------------------------------------------------------------------
# _list_files
# ---------------------------------------------------------------------------

def test_list_files_keeps_only_the_requested_subfolder(monkeypatch):
    seen = []

    def fake_list(repo_id, repo_type=None):
        seen.append((repo_id, repo_type))
        return ["plate1/a.tif", "plate1/b.tif", "plate2/c.tif",
                "README.md", "settings.csv"]
    monkeypatch.setattr("huggingface_hub.list_repo_files", fake_list)

    out = hf._list_files("einarolafsson/toxo_mito", "plate1")
    assert out == ["plate1/a.tif", "plate1/b.tif"]
    assert seen == [("einarolafsson/toxo_mito", "dataset")]


def test_list_files_with_no_subfolder_takes_only_csvs(monkeypatch):
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda repo_id, repo_type=None: ["a.csv", "b.json", "sub/c.csv",
                                         "README.md"])
    assert hf._list_files("einarolafsson/spacr_settings", "") == \
        ["a.csv", "sub/c.csv"]


def test_list_files_returns_empty_when_the_repo_has_nothing_matching(
        monkeypatch):
    monkeypatch.setattr("huggingface_hub.list_repo_files",
                        lambda repo_id, repo_type=None: ["README.md"])
    assert hf._list_files("repo/x", "plate1") == []
    assert hf._list_files("repo/x", "") == []


# ---------------------------------------------------------------------------
# _content_length
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("headers,expected", [
    ({"Content-Length": "1024"}, 1024),
    ({"Content-Length": "0"}, 0),
    ({}, None),
    ({"Content-Length": "not-a-number"}, None),
    ({"Content-Length": None}, None),
])
def test_content_length_parsing(headers, expected):
    assert hf._content_length(FakeResponse(headers=headers)) == expected


def test_content_length_tolerates_a_response_without_headers():
    class Bare:
        pass
    assert hf._content_length(Bare()) is None


# ---------------------------------------------------------------------------
# _download_one — request shape
# ---------------------------------------------------------------------------

def test_download_one_builds_the_resolve_url_and_streams(fake_get, tmp_path):
    resp = FakeResponse([b"abc", b"", b"de"])
    calls = fake_get(resp)

    out = hf._download_one("einarolafsson/toxo_mito",
                           "plate1/img_A01_C1.tif", tmp_path)

    url, kwargs = calls[0]
    assert url == ("https://huggingface.co/datasets/einarolafsson/toxo_mito"
                   "/resolve/main/plate1/img_A01_C1.tif?download=true")
    assert kwargs == {"stream": True, "timeout": 30}
    assert resp.iter_kwargs == [1 << 15]

    # Flattened into the destination dir under its basename only.
    assert out == tmp_path / "img_A01_C1.tif"
    assert out.read_bytes() == b"abcde"        # empty chunk skipped
    assert list(tmp_path.iterdir()) == [out]   # no ".part" left behind


def test_download_one_honours_a_matching_content_length(fake_get, tmp_path):
    fake_get(FakeResponse([b"12345"], headers={"Content-Length": "5"}))
    out = hf._download_one("r/x", "a.tif", tmp_path)
    assert out.read_bytes() == b"12345"


def test_download_one_writes_an_empty_file_for_an_empty_body(fake_get,
                                                             tmp_path):
    fake_get(FakeResponse([], headers={"Content-Length": "0"}))
    out = hf._download_one("r/x", "empty.tif", tmp_path)
    assert out.exists() and out.read_bytes() == b""


# ---------------------------------------------------------------------------
# _download_one — failure paths
# ---------------------------------------------------------------------------

def test_download_one_propagates_an_http_error_and_writes_nothing(fake_get,
                                                                  tmp_path):
    fake_get(FakeResponse(status_error=RuntimeError("404 Client Error")))
    with pytest.raises(RuntimeError, match="404"):
        hf._download_one("r/x", "missing.tif", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_one_rejects_a_truncated_body(fake_get, tmp_path):
    """Checksum-mismatch equivalent: fewer bytes than Content-Length."""
    fake_get(FakeResponse([b"only-5"[:5]],
                          headers={"Content-Length": "4096"}))
    with pytest.raises(IOError) as exc:
        hf._download_one("r/x", "big.tif", tmp_path)
    assert "Truncated download for big.tif" in str(exc.value)
    assert "wrote 5 bytes but the server declared 4096" in str(exc.value)
    assert list(tmp_path.iterdir()) == [], "the short file must not survive"


def test_download_one_rejects_an_over_long_body(fake_get, tmp_path):
    fake_get(FakeResponse([b"0123456789"], headers={"Content-Length": "4"}))
    with pytest.raises(IOError):
        hf._download_one("r/x", "big.tif", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_one_discards_the_partial_file_when_the_stream_breaks(
        fake_get, tmp_path):
    """BUG (fixed): the body was streamed straight onto the final path, so
    a connection dropped halfway left a truncated .tif that looked like a
    completed download to every later stage of the pipeline."""
    boom = ConnectionError("Connection broken: IncompleteRead")
    fake_get(FakeResponse([b"A" * 4096, b"B" * 4096, boom],
                          headers={"Content-Length": "99999"}))
    with pytest.raises(ConnectionError):
        hf._download_one("r/x", "half.tif", tmp_path)
    assert not (tmp_path / "half.tif").exists()
    assert list(tmp_path.iterdir()) == []


def test_download_one_keeps_an_existing_good_file_when_a_retry_fails(
        fake_get, tmp_path):
    """The .part staging also means a failed re-download cannot clobber the
    copy already on disk."""
    good = tmp_path / "a.tif"
    good.write_bytes(b"GOOD-IMAGE-BYTES")
    fake_get(FakeResponse([b"junk", ConnectionError("dropped")]))
    with pytest.raises(ConnectionError):
        hf._download_one("r/x", "a.tif", tmp_path)
    assert good.read_bytes() == b"GOOD-IMAGE-BYTES"


def test_download_one_reraises_when_the_staging_file_cannot_be_removed(
        fake_get, tmp_path):
    """A directory sitting where the .part file belongs breaks both the
    write and the cleanup; the original error must still surface."""
    (tmp_path / "a.tif.part").mkdir()
    fake_get(FakeResponse([b"x"]))
    with pytest.raises(IsADirectoryError):
        hf._download_one("r/x", "a.tif", tmp_path)
    assert (tmp_path / "a.tif.part").is_dir()
    assert not (tmp_path / "a.tif").exists()


# ---------------------------------------------------------------------------
# _HFDownloadWorker — driven synchronously on the calling thread
# ---------------------------------------------------------------------------

class WorkerHarness:
    """Runs the worker in-process and records every signal it emits."""

    def __init__(self, dest):
        self.worker = hf._HFDownloadWorker(dest)
        self.progress = []
        self.info = []
        self.finished = []
        self.worker.progress.connect(
            lambda n, d, t: self.progress.append((n, d, t)))
        self.worker.info.connect(self.info.append)
        self.worker.finished.connect(
            lambda ok, ds, st, err: self.finished.append((ok, ds, st, err)))

    def run(self):
        self.worker.run()
        assert len(self.finished) == 1, "worker must finish exactly once"
        return self.finished[0]


@pytest.fixture
def stub_repos(monkeypatch):
    """Stub the manifest + per-file download; return the download log."""
    downloaded = []

    def _install(dataset_files, settings_files, on_download=None):
        monkeypatch.setattr(
            hf, "_list_files",
            lambda repo, sub: list(dataset_files) if sub
            else list(settings_files))

        def fake_download(repo, name, dest_dir):
            downloaded.append((repo, name, dest_dir))
            if on_download is not None:
                on_download(name)
            p = Path(dest_dir) / Path(name).name
            p.write_bytes(b"IMG:" + name.encode())
            return p
        monkeypatch.setattr(hf, "_download_one", fake_download)
        return downloaded
    return _install


def test_worker_downloads_both_repos_into_their_own_folders(qapp, tmp_path,
                                                            stub_repos):
    log = stub_repos(["plate1/a.tif", "plate1/b.tif"], ["measure.csv"])
    h = WorkerHarness(tmp_path)
    ok, ds, st, err = h.run()

    assert (ok, err) == (True, "")
    assert Path(ds) == tmp_path / "plate1"
    assert Path(st) == tmp_path / "settings"
    assert (tmp_path / "plate1" / "a.tif").read_bytes() == b"IMG:plate1/a.tif"
    assert (tmp_path / "plate1" / "b.tif").exists()
    assert (tmp_path / "settings" / "measure.csv").read_bytes() == \
        b"IMG:measure.csv"

    # Each repo's files went to that repo's folder.
    assert [(repo, name) for repo, name, _ in log] == [
        (hf.DATASET_REPO, "plate1/a.tif"),
        (hf.DATASET_REPO, "plate1/b.tif"),
        (hf.SETTINGS_REPO, "measure.csv"),
    ]
    assert [d for _, _, d in log] == [tmp_path / "plate1"] * 2 + \
        [tmp_path / "settings"]


def test_worker_progress_counts_every_file_across_both_repos(qapp, tmp_path,
                                                             stub_repos):
    stub_repos(["plate1/a.tif", "plate1/b.tif"], ["m.csv"])
    h = WorkerHarness(tmp_path)
    h.run()
    assert h.progress == [
        ("plate1/a.tif", 0, 3),
        ("plate1/b.tif", 1, 3),
        ("m.csv", 2, 3),
        ("done", 3, 3),
    ]
    assert h.info == ["Listing files on Hugging Face…",
                      "Found 3 files to download."]


def test_worker_reports_an_empty_manifest_as_a_failure(qapp, tmp_path,
                                                       stub_repos):
    stub_repos([], [])
    h = WorkerHarness(tmp_path)
    ok, ds, st, err = h.run()
    assert ok is False
    assert (ds, st) == ("", "")
    assert err == ("No files to download from the Hugging Face "
                   "repositories.")
    assert h.progress == []
    # Bailing out early must not scatter empty folders around.
    assert not (tmp_path / "plate1").exists()


def test_worker_surfaces_a_no_network_failure(qapp, tmp_path, monkeypatch):
    """A DNS failure comes out as a sentence, not as the urllib3 text.

    The exception string used to be forwarded verbatim; it now goes through
    ``explain_download_failure`` — see the offline-degradation block at the
    bottom of this module for what each failure class is turned into.
    """
    def offline(repo, sub):
        raise ConnectionError(
            "HTTPSConnectionPool(host='huggingface.co', port=443): "
            "Name or service not known")
    monkeypatch.setattr(hf, "_list_files", offline)
    h = WorkerHarness(tmp_path)
    ok, ds, st, err = h.run()
    assert ok is False
    assert (ds, st) == ("", "")
    assert "Could not reach huggingface.co" in err
    assert "internet connection" in err


def test_worker_surfaces_a_mid_download_failure(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "_list_files",
                        lambda repo, sub: ["plate1/a.tif"] if sub else ["m.csv"])

    def explode(repo, name, dest_dir):
        raise IOError(f"Truncated download for {name}: wrote 5 bytes "
                      f"but the server declared 4096.")
    monkeypatch.setattr(hf, "_download_one", explode)

    h = WorkerHarness(tmp_path)
    ok, ds, st, err = h.run()
    assert ok is False
    assert "Truncated download for plate1/a.tif" in err
    # It failed on the very first file, so only that one was announced.
    assert h.progress == [("plate1/a.tif", 0, 2)]


def test_worker_stops_in_the_dataset_loop_when_cancelled(qapp, tmp_path,
                                                         stub_repos):
    log = stub_repos(["plate1/a.tif", "plate1/b.tif"], ["m.csv"])
    h = WorkerHarness(tmp_path)
    h.worker.cancel()
    ok, ds, st, err = h.run()
    assert (ok, ds, st, err) == (False, "", "", "Cancelled by user.")
    assert log == [], "nothing should be fetched after a pre-emptive cancel"


def test_worker_stops_in_the_settings_loop_when_cancelled(qapp, tmp_path,
                                                          stub_repos):
    """Cancel raised while the last dataset file downloads is honoured at
    the top of the settings loop."""
    holder = {}

    def on_download(name):
        if name == "plate1/b.tif":
            holder["worker"].cancel()

    log = stub_repos(["plate1/a.tif", "plate1/b.tif"], ["m.csv", "n.csv"],
                     on_download=on_download)
    h = WorkerHarness(tmp_path)
    holder["worker"] = h.worker
    ok, ds, st, err = h.run()
    assert (ok, err) == (False, "Cancelled by user.")
    assert [name for _, name, _ in log] == ["plate1/a.tif", "plate1/b.tif"]
    assert not (tmp_path / "settings" / "m.csv").exists()


def test_worker_cancel_flag_starts_false(qapp, tmp_path):
    w = hf._HFDownloadWorker(tmp_path)
    assert w._cancel is False
    assert w._dest == tmp_path
    w.cancel()
    assert w._cancel is True


def test_worker_accepts_a_string_destination(qapp, tmp_path, stub_repos):
    stub_repos(["plate1/a.tif"], [])
    h = WorkerHarness(str(tmp_path))
    ok, ds, st, err = h.run()
    assert ok is True
    assert Path(ds) == tmp_path / "plate1"


# ---------------------------------------------------------------------------
# download_toxo_mito_demo — the full Qt flow
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_flow(qtbot, tmp_path, monkeypatch):
    """Run download_toxo_mito_demo end-to-end against stubbed transport."""
    state = {}

    def _run(dataset_files=("plate1/a.tif",), settings_files=("m.csv",),
             on_download=None, list_files=None):
        if list_files is None:
            def list_files(repo, sub):
                return list(dataset_files) if sub else list(settings_files)
        monkeypatch.setattr(hf, "_list_files", list_files)

        threads = []

        def fake_download(repo, name, dest_dir):
            threads.append(threading.get_ident())
            if on_download is not None:
                on_download(name)
            p = Path(dest_dir) / Path(name).name
            p.write_bytes(b"IMG")
            return p
        monkeypatch.setattr(hf, "_download_one", fake_download)

        parent = QWidget()
        qtbot.addWidget(parent)
        outcome = []
        cb_threads = []

        def on_done(result, err):
            cb_threads.append(threading.get_ident())
            outcome.append((result, err))

        hf.download_toxo_mito_demo(parent, tmp_path, on_done)
        state.update(parent=parent, outcome=outcome,
                     worker_threads=threads, cb_threads=cb_threads,
                     dlg=parent._hf_download_dialog)
        return state

    yield _run
    # Never leave a live QThread behind, whatever the test did.
    thread = getattr(state.get("parent"), "_hf_download_thread", None)
    if isinstance(thread, QThread) and thread.isRunning():
        thread.quit()
        thread.wait(5000)


def test_demo_download_completes_and_reports_local_paths(qtbot, tmp_path,
                                                         demo_flow):
    s = demo_flow(dataset_files=("plate1/a.tif", "plate1/b.tif"),
                  settings_files=("m.csv",))
    qtbot.waitUntil(lambda: bool(s["outcome"]), timeout=10000)

    result, err = s["outcome"][0]
    assert err == ""
    assert result.dataset_path == tmp_path / "plate1"
    assert result.settings_path == tmp_path / "settings"
    assert (tmp_path / "plate1" / "a.tif").read_bytes() == b"IMG"
    assert (tmp_path / "plate1" / "b.tif").exists()
    assert (tmp_path / "settings" / "m.csv").exists()


def test_demo_download_runs_off_the_gui_thread_and_calls_back_on_it(
        qtbot, tmp_path, demo_flow):
    """BUG (fixed): the handlers were plain closures, so Qt gave them a
    DIRECT connection and they drove the QProgressDialog from the worker
    thread — "QWidget::repaint: Recursive repaint detected", then SIGSEGV.

    The dialog spies below assert every touch happens on the GUI thread.
    """
    gui_thread = threading.get_ident()
    s = demo_flow(dataset_files=("plate1/a.tif",), settings_files=("m.csv",))

    dlg = s["dlg"]
    touches = []
    for name in ("setLabelText", "setValue", "setMaximum"):
        original = getattr(dlg, name)

        def spy(*args, _orig=original):
            touches.append(threading.get_ident())
            return _orig(*args)
        setattr(dlg, name, spy)

    qtbot.waitUntil(lambda: bool(s["outcome"]), timeout=10000)

    assert s["worker_threads"], "the download never ran"
    assert all(t != gui_thread for t in s["worker_threads"]), \
        "the download blocked the GUI thread"
    assert touches, "the progress dialog was never updated"
    assert set(touches) == {gui_thread}, \
        "the progress dialog was touched from a non-GUI thread"
    assert s["cb_threads"] == [gui_thread]


def test_demo_download_releases_its_retained_objects_when_done(
        qtbot, tmp_path, demo_flow):
    s = demo_flow()
    parent = s["parent"]
    assert isinstance(parent._hf_download_thread, QThread)
    assert isinstance(parent._hf_download_worker, hf._HFDownloadWorker)
    assert isinstance(parent._hf_download_ui, hf._HFDownloadUI)

    qtbot.waitUntil(lambda: bool(s["outcome"]), timeout=10000)

    for attr in ("_hf_download_thread", "_hf_download_worker",
                 "_hf_download_dialog", "_hf_download_ui"):
        assert not hasattr(parent, attr), f"{attr} was never released"
    # deleteLater has been processed: the dialog's C++ half is gone, so a
    # stuck modal cannot outlive the download.
    assert not isValid(s["dlg"])


def test_demo_download_reports_a_network_failure_through_the_callback(
        qtbot, tmp_path, demo_flow):
    def offline(repo, sub):
        raise ConnectionError("Max retries exceeded with url: /api/datasets")

    s = demo_flow(list_files=offline)
    qtbot.waitUntil(lambda: bool(s["outcome"]), timeout=10000)

    result, err = s["outcome"][0]
    assert result is None
    # `err` is what `_on_e2e_demo` puts in a QMessageBox, so it is the
    # explained text rather than the transport's own words.
    assert "Could not reach huggingface.co" in err
    assert "Max retries exceeded" not in err
    assert s["worker_threads"] == []
    assert not isValid(s["dlg"]), "the progress dialog outlived the failure"


def test_demo_download_cancel_button_reaches_the_worker_mid_flight(
        qtbot, tmp_path, demo_flow):
    """BUG (fixed): ``canceled`` was queued to the worker's event loop,
    which is blocked for the whole of ``run()`` — so the cancel was only
    delivered once the download it meant to abort had already finished."""
    started = threading.Event()
    release = threading.Event()

    def block_on_first(name):
        if name == "plate1/a.tif":
            started.set()
            assert release.wait(20), "test never released the worker"

    s = demo_flow(dataset_files=("plate1/a.tif", "plate1/b.tif"),
                  settings_files=("m.csv",),
                  on_download=block_on_first)

    qtbot.waitUntil(started.is_set, timeout=10000)
    s["dlg"].canceled.emit()          # exactly what the Cancel button does
    release.set()

    qtbot.waitUntil(lambda: bool(s["outcome"]), timeout=10000)
    result, err = s["outcome"][0]
    assert result is None
    assert err == "Cancelled by user."
    assert len(s["worker_threads"]) == 1, \
        "the worker kept downloading after Cancel"
    assert not (tmp_path / "plate1" / "b.tif").exists()


def test_demo_download_creates_the_destination_directory(qtbot, tmp_path,
                                                         monkeypatch):
    monkeypatch.setattr(hf, "_list_files", lambda repo, sub: [])
    parent = QWidget()
    qtbot.addWidget(parent)
    dest = tmp_path / "does" / "not" / "exist"
    outcome = []
    hf.download_toxo_mito_demo(parent, dest,
                               lambda r, e: outcome.append((r, e)))
    assert dest.is_dir()
    qtbot.waitUntil(lambda: bool(outcome), timeout=10000)
    assert outcome[0][0] is None
    assert outcome[0][1].startswith("No files to download")


# ---------------------------------------------------------------------------
# _HFDownloadUI defensive path
# ---------------------------------------------------------------------------

def test_finish_handler_still_calls_back_when_the_dialog_is_already_gone(
        qtbot):
    """If the dialog's C++ half has been torn down, setValue raises. The
    completion callback is the important part and must survive it."""
    class DeadDialog:
        closed = False

        def maximum(self):
            return 1

        def setValue(self, value):
            raise RuntimeError("Internal C++ object already deleted.")

        def reset(self):
            pass

        def close(self):
            DeadDialog.closed = True

        def deleteLater(self):
            pass

    owner = QWidget()
    qtbot.addWidget(owner)
    thread = QThread(owner)          # never started
    worker = hf._HFDownloadWorker(Path("."))
    calls = []
    ui = hf._HFDownloadUI(DeadDialog(), thread, worker, owner,
                          lambda r, e: calls.append((r, e)))

    ui.on_finished(True, "/tmp/plate1", "/tmp/settings", "")

    qtbot.waitUntil(lambda: bool(calls), timeout=5000)
    result, err = calls[0]
    assert err == ""
    assert result.dataset_path == Path("/tmp/plate1")
    assert result.settings_path == Path("/tmp/settings")
    assert DeadDialog.closed is True


# ---------------------------------------------------------------------------
# Offline degradation — the one demo in the Demos menu that needs the network
# ---------------------------------------------------------------------------
#
# The other six entries under Demos are synthetic and run with no network at
# all. This one downloads a real plate, so it is the only one that can fail for
# a reason outside spaCR, and the only one whose failure message the user has
# to be able to act on. What it used to hand the dialog was ``str(exc)``: for
# the ordinary offline case, 300 characters of nested urllib3 that never say
# "you are offline" and never say what to do instead.


def test_offline_failure_names_the_network_and_points_at_the_synthetic_demos():
    """A ConnectionError must not reach the user as a urllib3 dump."""
    import requests

    exc = requests.exceptions.ConnectionError(
        "HTTPSConnectionPool(host='huggingface.co', port=443): Max retries "
        "exceeded with url: /api/datasets/x (Caused by NewConnectionError("
        "'<urllib3.connection.HTTPSConnection object at 0x7f00>: Failed to "
        "establish a new connection: [Errno 101] Network is unreachable'))")
    message = hf.explain_download_failure(exc)

    assert "huggingface.co" in message
    assert "internet connection" in message
    assert "synthetic" in message and "no network" in message
    assert "urllib3" not in message and "MaxRetryError" not in message


def test_missing_huggingface_hub_says_which_package_and_how_to_install_it():
    message = hf.explain_download_failure(
        ImportError("huggingface_hub is not installed: No module named "
                    "'huggingface_hub'"))
    assert "pip install huggingface_hub" in message
    assert "synthetic" in message


def test_a_truncated_transfer_says_nothing_partial_was_kept():
    message = hf.explain_download_failure(IOError(
        "Truncated download for plate1/a.tif: wrote 10 bytes but the server "
        "declared 4096."))
    assert "Truncated download" in message
    assert "re-running" in message


def test_explain_survives_an_environment_without_requests(monkeypatch):
    """The requests import is local; a broken env must still get a message.

    Not hypothetical: ``requests`` is a dependency *of* huggingface_hub, so an
    environment that cannot import one frequently cannot import the other —
    which is exactly the environment this function has to describe.
    """
    import builtins

    real_import = builtins.__import__

    def _no_requests(name, *args, **kwargs):
        if name == "requests":
            raise ImportError("No module named 'requests'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_requests)
    message = hf.explain_download_failure(RuntimeError("some transport failure"))
    assert "some transport failure" in message
    assert "synthetic" in message


def test_list_files_names_the_missing_package(monkeypatch):
    """``_list_files`` must not let a bare ModuleNotFoundError stand.

    ``explain_download_failure`` keys the install instructions off ImportError,
    and the raised text is what gets embedded in the dialog.
    """
    import builtins

    real_import = builtins.__import__

    def _no_hub(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_hub)
    with pytest.raises(ImportError, match="huggingface_hub is not installed"):
        hf._list_files("someone/dataset", "plate1")


def test_the_download_worker_emits_the_friendly_text(monkeypatch, tmp_path):
    """End of the wire: what ``_on_download_done`` is handed when offline."""
    import requests

    def _boom(*_a, **_k):
        raise requests.exceptions.ConnectionError(
            "HTTPSConnectionPool(host='huggingface.co', port=443): "
            "MaxRetryError: Network is unreachable")

    monkeypatch.setattr(hf, "_list_files", _boom)

    worker = hf._HFDownloadWorker(tmp_path)
    seen = []
    worker.finished.connect(lambda ok, ds, st, err: seen.append((ok, err)))
    worker.run()

    assert len(seen) == 1, seen
    ok, err = seen[0]
    assert ok is False
    assert "Could not reach huggingface.co" in err
    assert "MaxRetryError" not in err
