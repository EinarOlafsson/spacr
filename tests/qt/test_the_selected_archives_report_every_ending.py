"""Selected screen downloads preserve the choice and report exactly one outcome.

Drive run() on the test thread: no network, GPU, or live QThread is needed
to exercise streaming, cancellation, and the completion signal.
"""
import pytest

from spacr.qt import hf_download as hf


@pytest.fixture
def download(tmp_path, monkeypatch):
    """Keep the streaming loop real; record transport and extraction effects."""
    requests = []
    extracted = []
    prepared = []
    responses = {}

    class Response:
        def __init__(self, chunks, length):
            self.chunks = chunks
            self.headers = {} if length is None else {"Content-Length": str(length)}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 1 << 20
            yield from self.chunks

    def get(url, **kwargs):
        requests.append((url, kwargs))
        archive = url.rsplit("/", 1)[1].split("?", 1)[0]
        return responses[archive]

    def extract(archive, destination):
        extracted.append((archive.name, archive.read_bytes(), destination))

    monkeypatch.setattr("requests.get", get)
    monkeypatch.setattr(hf, "extract_example_archive", extract)
    monkeypatch.setattr(hf, "make_the_example_paths_absolute", prepared.append)

    def make(archives=("database.tar", "crops.tar")):
        worker = hf._ChosenArchivesWorker(
            tmp_path / "screen", archives=archives, repo="test/screen")
        seen = {"finished": [], "progress": [], "info": []}
        for name, events in seen.items():
            getattr(worker, name).connect(lambda *args, events=events: events.append(args))
        return worker, seen

    return make, Response, responses, requests, extracted, prepared


@pytest.mark.parametrize("length", [None, 3])
def test_only_selected_archives_are_fetched_in_order(download, tmp_path, length):
    make, Response, responses, requests, extracted, prepared = download
    responses.update({
        "database.tar": Response([b"", b"db", b"1"], length),
        "crops.tar": Response([b"png"], 3),
    })
    worker, seen = make()
    worker.run()
    root = tmp_path / "screen"
    assert requests == [
        (f"https://huggingface.co/datasets/test/screen/resolve/main/{name}?download=true",
         {"stream": True, "timeout": 30})
        for name in ("database.tar", "crops.tar")
    ]
    assert extracted == [("database.tar", b"db1", root), ("crops.tar", b"png", root)]
    assert prepared == [root]
    assert seen["finished"] == [(True, str(root), str(root / "settings"), "")]
    assert seen["progress"][-1] == ("done", 1, 1)
    assert seen["info"] == [
        ("Downloading database.tar (1 of 2)…",),
        ("Downloading crops.tar (2 of 2)…",),
        ("Preparing the files…",),
    ]
    assert list(root.iterdir()) == [], "transport archives must be removed after extraction"


@pytest.mark.parametrize("cancelled", [False, True])
def test_empty_or_cancelled_selection_never_starts_a_request(download, cancelled):
    make, _, _, requests, extracted, prepared = download
    worker, seen = make() if cancelled else make(())
    if cancelled:
        worker.cancel()
    worker.run()
    error = "Cancelled by user." if cancelled else "Nothing was selected."
    assert seen["finished"] == [(False, "", "", error)]
    assert requests == extracted == prepared == []
    assert seen["progress"] == []


def test_cancellation_during_a_stream_removes_the_partial_and_stops(download, tmp_path):
    make, Response, responses, requests, extracted, prepared = download
    worker, seen = make()

    def chunks():
        yield b"first chunk"
        worker.cancel()
        yield b"must not be written"

    responses["database.tar"] = Response(chunks(), None)
    worker.run()
    assert len(requests) == 1
    assert seen["finished"] == [(False, "", "", "Cancelled by user.")]
    assert extracted == prepared == []
    assert list((tmp_path / "screen").iterdir()) == []
    assert ("done", 1, 1) not in seen["progress"]


def test_cancelling_after_one_archive_does_not_fetch_the_next(download, monkeypatch):
    make, Response, responses, requests, _, prepared = download
    worker, seen = make()
    responses["database.tar"] = Response([b"db"], 2)
    extracted = []

    def extract(archive, destination):
        extracted.append(archive.name)
        worker.cancel()

    monkeypatch.setattr(hf, "extract_example_archive", extract)
    worker.run()
    assert extracted == ["database.tar"]
    assert len(requests) == 1
    assert prepared == []
    assert seen["finished"] == [(False, "", "", "Cancelled by user.")]


@pytest.mark.parametrize("expected", [2, 4])
def test_wrong_length_is_never_unpacked_or_reported_as_success(download, tmp_path, expected):
    make, Response, responses, requests, extracted, prepared = download
    responses["database.tar"] = Response([b"abc"], expected)
    worker, seen = make()
    worker.run()
    assert len(requests) == 1
    assert len(seen["finished"]) == 1
    ok, dataset, settings, error = seen["finished"][0]
    assert (ok, dataset, settings) == (False, "", "")
    assert error
    assert extracted == prepared == []
    assert list((tmp_path / "screen").iterdir()) == []
    assert ("done", 1, 1) not in seen["progress"]


@pytest.mark.parametrize("stage", ["request", "extraction", "preparation"])
def test_a_failure_stops_the_run_and_emits_one_explained_failure(download, monkeypatch, stage):
    make, Response, responses, requests, extracted, prepared = download
    responses["database.tar"] = Response([b"db"], 2)
    responses["crops.tar"] = Response([b"png"], 3)
    reached = []

    def fail(*args, **kwargs):
        reached.append(stage)
        raise RuntimeError(f"{stage} failed")

    if stage == "request":
        monkeypatch.setattr("requests.get", fail)
    else:
        name = "extract_example_archive" if stage == "extraction" else "make_the_example_paths_absolute"
        monkeypatch.setattr(hf, name, fail)
    monkeypatch.setattr(hf, "explain_download_failure", lambda error: f"Explained: {error}")
    worker, seen = make()
    worker.run()
    assert reached == [stage]
    assert seen["finished"] == [(False, "", "", f"Explained: {stage} failed")]
    assert len(requests) == {"request": 0, "extraction": 1, "preparation": 2}[stage]
    assert len(extracted) == (2 if stage == "preparation" else 0)
    assert prepared == []
    assert ("done", 1, 1) not in seen["progress"]


def test_progress_reports_megabytes_before_completion(download):
    make, Response, responses, _, _, _ = download
    chunk = b"x" * (1 << 20)
    responses["database.tar"] = Response([chunk, chunk], len(chunk) * 2)
    worker, seen = make(("database.tar",))
    worker.run()
    assert seen["progress"] == [("database.tar", 1, 2), ("database.tar", 2, 2), ("done", 1, 1)]
    assert len(seen["finished"]) == 1 and seen["finished"][0][0] is True
