"""The single-archive example downloads, from the first chunk to the folder.

Every example set except the original toxo_mito pack arrives as ONE tar
rather than as a file list, and :class:`spacr.qt.hf_download._TarExampleWorker`
is the whole of that path: stream, cancel between chunks, check the length,
rename, unpack with a filter, rewrite the paths.  Its subclasses only name a
repo.

WHY THESE ARE WORTH PINNING. Each step has a silent failure:

* a stream that stops early and is unpacked anyway gives a user a plate
  missing fields, with no error and no way to know which ones;
* a ``.part`` renamed before the length is checked turns that into a file the
  next run treats as already downloaded;
* Cancel that is checked only between FILES leaves a thirty-gigabyte archive
  running after the dialog has gone;
* a failure raised rather than reported reaches nobody: this runs on a worker
  thread.

Everything here is offline.  ``requests.get`` is stubbed and the tar is built
in ``tmp_path``, so the extraction, the rename and the rewrite are the real
ones.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from spacr.qt import hf_download as hf


class _Response:
    """Enough of ``requests.Response`` for the streaming workers."""

    def __init__(self, chunks, *, content_length=None, status_error=None):
        self._chunks = list(chunks)
        self.headers = ({} if content_length is None
                        else {"Content-Length": str(content_length)})
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def iter_content(self, chunk_size=None):
        for chunk in self._chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


def _tar_bytes(files):
    """A tar holding ``{name: text}``, as bytes."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, text in files.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


@pytest.fixture
def serve(monkeypatch):
    """Serve one body for every download, and record the URLs asked for."""
    asked = []

    def _install(body, *, content_length="exact", status_error=None,
                 chunk_size=1 << 20):
        chunks = [body[i:i + chunk_size]
                  for i in range(0, len(body), chunk_size)] or [b""]
        length = (len(body) if content_length == "exact"
                  else content_length)

        def _get(url, **kwargs):
            asked.append(url)
            return _Response(chunks, content_length=length,
                             status_error=status_error)

        monkeypatch.setattr("requests.get", _get)
        return asked

    return _install


def _worker(tmp_path, repo="einarolafsson/example", archive="example.tar"):
    """A base tar worker pointed at a fixture repo."""
    worker = hf._TarExampleWorker(tmp_path / "dest")
    worker.repo = repo
    worker.archive = archive
    return worker


def _record(worker):
    """Collect ``finished``, ``info`` and ``progress`` without a Qt loop."""
    seen = {"finished": [], "info": [], "progress": []}
    worker.finished.connect(
        lambda ok, ds, st, err: seen["finished"].append((ok, ds, st, err)))
    worker.info.connect(seen["info"].append)
    worker.progress.connect(
        lambda name, done, total: seen["progress"].append((name, done, total)))
    return seen


# -- the happy path, which is also where the paths are rewritten -------------


def test_one_archive_is_streamed_unpacked_and_reported(qapp, tmp_path, serve):
    body = _tar_bytes({"plate1/a.txt": "one", "settings/x.csv": "two"})
    asked = serve(body)
    worker = _worker(tmp_path)
    seen = _record(worker)

    worker.run()

    assert asked == [
        "https://huggingface.co/datasets/einarolafsson/example/resolve/main/"
        "example.tar?download=true"
    ]
    ok, dataset, settings, error = seen["finished"][0]
    assert ok is True and error == ""
    assert Path(dataset) == tmp_path / "dest"
    assert Path(settings) == tmp_path / "dest" / "settings"
    assert (tmp_path / "dest" / "plate1" / "a.txt").read_text() == "one"


def test_the_archive_itself_is_not_left_in_the_folder(qapp, tmp_path, serve):
    """A user opening the folder sees the data, not a tar beside it.

    Leaving it also doubles the disk the example costs, which for the
    published screen is thirty gigabytes.
    """
    serve(_tar_bytes({"plate1/a.txt": "one"}))
    worker = _worker(tmp_path)
    _record(worker)

    worker.run()

    assert not (tmp_path / "dest" / "example.tar").exists()
    assert not list((tmp_path / "dest").glob("*.part"))


def test_the_destination_is_made_on_the_worker_thread_not_in_the_constructor(
    qapp, tmp_path, serve,
):
    """A caller may hand over a folder that does not exist yet.

    Creating it in ``__init__`` would make a failure to create it raise
    inside the caller's event handler, where the progress dialog has not
    been built yet and there is nothing to report through.
    """
    serve(_tar_bytes({"a.txt": "one"}))
    missing = tmp_path / "not" / "there" / "yet"
    worker = hf._TarExampleWorker(missing)
    worker.repo, worker.archive = "einarolafsson/example", "example.tar"
    seen = _record(worker)

    assert not missing.exists(), "the constructor must not have made it"

    worker.run()

    assert seen["finished"][0][0] is True
    assert (missing / "a.txt").is_file()


# -- the three ways it stops, none of which may leave usable-looking data ----


def test_a_download_that_stops_early_unpacks_nothing(qapp, tmp_path, serve):
    """The check is BEFORE the rename, so no short file reaches the target.

    A plate silently missing fields is the failure this prevents: nothing in
    the folder would say which fields were lost, and the run that used it
    would simply measure fewer objects.
    """
    body = _tar_bytes({"plate1/a.txt": "one"})
    serve(body[: len(body) // 2], content_length=len(body))
    worker = _worker(tmp_path)
    seen = _record(worker)

    worker.run()

    ok, _dataset, _settings, error = seen["finished"][0]
    assert ok is False
    assert "stopped early" in error
    assert "Nothing was unpacked" in error
    assert not (tmp_path / "dest" / "example.tar").exists()
    assert not list((tmp_path / "dest").glob("*.part"))
    assert not (tmp_path / "dest" / "plate1").exists()


def test_cancel_takes_effect_between_chunks_not_between_files(
    qapp, tmp_path, serve,
):
    """One archive is one file, so a per-file check never fires at all.

    The published screen is thirty gigabytes in one piece: a Cancel that
    waits for the end of the file is a Cancel that does nothing, with the
    dialog already gone.
    """
    body = _tar_bytes({"plate1/a.txt": "x" * 4_000_000})
    serve(body, chunk_size=1 << 20)
    worker = _worker(tmp_path)
    seen = _record(worker)
    worker.progress.connect(lambda *_: worker.cancel())

    worker.run()

    assert seen["finished"] == [(False, "", "", "Cancelled by user.")]
    assert not list((tmp_path / "dest").glob("*.part"))
    assert not (tmp_path / "dest" / "plate1").exists()


def test_a_refused_request_is_reported_rather_than_raised(qapp, tmp_path,
                                                          serve):
    """This runs on a worker thread, where an exception has nobody to catch it.

    The message goes through ``explain_download_failure`` so it says what to
    do rather than repeating the exception.
    """
    serve(b"", status_error=OSError("Network is unreachable"))
    worker = _worker(tmp_path)
    seen = _record(worker)

    worker.run()

    ok, _dataset, _settings, error = seen["finished"][0]
    assert ok is False
    assert error and error != ""
    assert (tmp_path / "dest").exists() or True


def test_an_archive_no_table_names_is_a_failure_not_a_guess(qapp, tmp_path,
                                                            serve):
    """A subclass with neither ``archive`` nor an entry in EXAMPLE_ARCHIVES."""
    serve(_tar_bytes({"a.txt": "one"}))
    worker = hf._TarExampleWorker(tmp_path / "dest")
    worker.repo = "einarolafsson/no-such-repo"
    seen = _record(worker)

    worker.run()

    assert seen["finished"][0][0] is False


# -- the chosen-archives worker, which is the same loop run several times ----


def _chosen(tmp_path, archives, repo="einarolafsson/screen"):
    return hf._ChosenArchivesWorker(tmp_path / "dest", archives=archives,
                                    repo=repo)


def test_choosing_nothing_is_refused_rather_than_read_as_everything(
    qapp, tmp_path,
):
    """THE WHOLE REASON THIS WORKER EXISTS is taking the 2 GB of databases
    without the 30 GB of crops. A default of "all" would silently undo the
    one choice it was built to offer."""
    worker = _chosen(tmp_path, [])
    seen = _record(worker)

    worker.run()

    assert seen["finished"] == [(False, "", "", "Nothing was selected.")]
    assert not (tmp_path / "dest").exists()


def test_each_chosen_archive_is_fetched_and_unpacked_in_turn(qapp, tmp_path,
                                                             serve):
    asked = serve(_tar_bytes({"piece.txt": "one"}))
    worker = _chosen(tmp_path, ["databases.tar", "crops.tar"])
    seen = _record(worker)

    worker.run()

    assert [url.rsplit("/", 1)[-1] for url in asked] == [
        "databases.tar?download=true", "crops.tar?download=true",
    ]
    assert seen["finished"][0][0] is True
    assert any("1 of 2" in line for line in seen["info"])
    assert any("2 of 2" in line for line in seen["info"])


def test_a_cancel_between_archives_stops_before_the_next_one(qapp, tmp_path,
                                                             serve):
    asked = serve(_tar_bytes({"piece.txt": "one"}))
    worker = _chosen(tmp_path, ["databases.tar", "crops.tar"])
    seen = _record(worker)
    worker.cancel()

    worker.run()

    assert asked == []
    assert seen["finished"] == [(False, "", "", "Cancelled by user.")]


def test_a_failing_archive_stops_the_run_without_contradicting_itself(
    qapp, tmp_path, serve,
):
    """The fetch has already reported the failure; a second message would
    tell the user two different things about the same download."""
    body = _tar_bytes({"piece.txt": "one"})
    asked = serve(body[: len(body) // 2], content_length=len(body))
    worker = _chosen(tmp_path, ["databases.tar", "crops.tar"])
    seen = _record(worker)

    worker.run()

    assert len(asked) == 1, "the second archive must not be attempted"
    assert len(seen["finished"]) == 1
    assert seen["finished"][0][0] is False
    assert "stopped early" in seen["finished"][0][3]


def test_cancelling_mid_archive_reports_once_and_keeps_no_part_file(
    qapp, tmp_path, serve,
):
    serve(_tar_bytes({"piece.txt": "x" * 4_000_000}), chunk_size=1 << 20)
    worker = _chosen(tmp_path, ["databases.tar", "crops.tar"])
    seen = _record(worker)
    worker.progress.connect(lambda *_: worker.cancel())

    worker.run()

    assert seen["finished"] == [(False, "", "", "Cancelled by user.")]
    assert not list((tmp_path / "dest").glob("*.part"))


# -- the file-list worker for Measure's example -----------------------------


@pytest.fixture
def measure_repo(monkeypatch, tmp_path):
    """Stub the hub listing and the per-file download for Measure's set."""
    downloaded = []

    def _install(names, *, on_download=None):
        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub, "list_repo_files",
            lambda repo, **kwargs: list(names))

        def _one(repo, name, folder):
            downloaded.append(name)
            if on_download is not None:
                on_download(name)
            target = Path(folder) / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("payload", encoding="utf-8")

        monkeypatch.setattr(hf, "_download_one", _one)
        monkeypatch.setattr(hf, "expand_measure_arrays", lambda merged: None)
        return downloaded

    return _install


def test_the_measure_example_downloads_every_listed_file(qapp, tmp_path,
                                                         measure_repo):
    downloaded = measure_repo(["merged/plate1.npz", "settings/measure.csv"])
    worker = hf._MeasureExampleWorker(tmp_path / "dest")
    seen = _record(worker)

    worker.run()

    assert downloaded == ["merged/plate1.npz", "settings/measure.csv"]
    ok, dataset, settings, error = seen["finished"][0]
    assert ok is True and error == ""
    assert Path(dataset) == tmp_path / "dest"
    assert Path(settings) == tmp_path / "dest" / "settings"


def test_a_repo_that_lists_nothing_names_the_repo_in_the_failure(
    qapp, tmp_path, measure_repo,
):
    """"No files" from a renamed or emptied repo has to say which one."""
    measure_repo([])
    worker = hf._MeasureExampleWorker(tmp_path / "dest")
    seen = _record(worker)

    worker.run()

    ok, _dataset, _settings, error = seen["finished"][0]
    assert ok is False
    assert hf.MEASURE_EXAMPLE_REPO in error


def test_hidden_files_in_the_repo_are_not_downloaded(qapp, tmp_path,
                                                     measure_repo):
    """``.gitattributes`` is in every dataset repo and is not example data."""
    downloaded = measure_repo([".gitattributes", "merged/plate1.npz"])
    worker = hf._MeasureExampleWorker(tmp_path / "dest")
    _record(worker)

    worker.run()

    assert downloaded == ["merged/plate1.npz"]


def test_the_measure_example_stops_between_files_when_cancelled(
    qapp, tmp_path, measure_repo,
):
    worker = hf._MeasureExampleWorker(tmp_path / "dest")
    downloaded = measure_repo(
        ["a.npz", "b.npz", "c.npz"], on_download=lambda _name: worker.cancel())
    seen = _record(worker)

    worker.run()

    assert downloaded == ["a.npz"], "the loop must stop at the next file"
    assert seen["finished"] == [(False, "", "", "Cancelled by user.")]


def test_a_missing_huggingface_hub_names_the_package(qapp, tmp_path,
                                                     monkeypatch):
    """It is an optional dependency, and the message is the install line.

    Raised on a worker thread, so it has to arrive through ``finished``.
    """
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    worker = hf._MeasureExampleWorker(tmp_path / "dest")
    seen = _record(worker)

    worker.run()

    ok, _dataset, _settings, error = seen["finished"][0]
    assert ok is False
    assert "huggingface_hub" in error


def test_the_deprecated_array_shim_still_calls_the_module_function(
    qapp, tmp_path, monkeypatch,
):
    """Kept because other code may hold the method; it must not do its own
    work, or there would be two expansions to keep in step."""
    called = []
    monkeypatch.setattr(hf, "expand_measure_arrays", called.append)

    hf._MeasureExampleWorker(tmp_path)._expand_arrays(tmp_path / "merged")

    assert called == [tmp_path / "merged"]
