"""Item 288: the model zoo's listing code against what the network hands it.

The zoo reads bioimage.io's collection, the community uploads on Hugging
Face and its own caches, all best effort: a corrupt cache, a changed
schema, a server that will not say how big a file is, a folder with no
checkpoint -- each means fewer rows or an honest note, never a zoo that
fails to open. What is pinned here is the answer in each case. No network
is used: the transports are replaced by stand-ins at their seams.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from spacr import model_zoo as mz


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home directory of our own, so the zoo's caches start empty."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_a_scorecard_without_a_holdout_set_names_none():
    entry = mz.ModelEntry(key="k", name="n",
                          metrics={"f1": {"finetuned": 0.8, "vanilla": 0.6}})
    lines = entry.scorecard_lines()
    assert lines and not any("hold-out set" in line for line in lines)


@pytest.mark.parametrize("uri, card", [
    ("https://huggingface.co/datasets/only_owner", ""),
    ("https://huggingface.co/einar/models/resolve/main/x.pth",
     "https://huggingface.co/einar/models"),
    ("https://example.org/x.pth", ""),
])
def test_the_model_card_is_derived_only_from_a_whole_hub_address(uri, card):
    assert mz.ModelEntry(key="k", name="n", uri=uri).model_card_url == card


def test_without_qt_or_its_application_the_gui_thread_answer_is_no(
        monkeypatch):
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", None)
    assert mz._on_the_qt_gui_thread() is False


def test_an_application_that_cannot_be_asked_is_not_the_gui_thread(
        monkeypatch):
    def refuse():
        raise RuntimeError("application torn down")

    fake = SimpleNamespace(
        QCoreApplication=SimpleNamespace(instance=refuse),
        QThread=SimpleNamespace(currentThread=lambda: None))
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", fake)
    assert mz._on_the_qt_gui_thread() is False


def test_only_a_model_manifest_can_be_cellpose_sam():
    manifest = {"type": "dataset", "name": "cellpose-sam training images"}
    assert mz._looks_like_cellpose_sam(manifest) is False


def test_a_corrupt_collection_cache_offline_is_no_collection(home):
    cache = mz._bioimageio_cache("bioimageio_children.json")
    cache.parent.mkdir(parents=True)
    cache.write_text("{not json")
    assert mz._bioimageio_payload(1.0, None, allow_network=False) == \
        (None, False)
    assert mz.bioimageio_entries(allow_network=False) == []


def test_a_sizes_cache_that_is_not_a_mapping_is_no_sizes(home):
    cache = mz._bioimageio_cache("bioimageio_weight_sizes.json")
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps([1, 2, 3]))
    assert mz._bioimageio_sizes() == {}


class _Response:
    def __init__(self, headers, status=200):
        self.headers = headers
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("headers, status, size", [
    ({"Content-Range": "bytes 0-0/1234"}, 206, 1234),
    ({"Content-Length": "999"}, 200, 999),
    ({"Content-Length": "1"}, 206, 0),
    ({}, 200, 0),
])
def test_a_weights_file_is_sized_by_what_the_server_says(
        monkeypatch, headers, status, size):
    import urllib.request

    asked = []

    def urlopen(request, timeout):
        asked.append(request.headers.get("Range"))
        return _Response(headers, status)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert mz._weights_size("https://x/w.pth", 1.0) == size
    assert asked == ["bytes=0-0"]


def test_sizes_that_cannot_be_cached_are_still_returned(home, monkeypatch):
    (home / ".spacr").write_text("a file where the cache folder goes")
    monkeypatch.setattr(mz, "_weights_size", lambda uri, timeout: 4321)
    sizes = mz._measure_bioimageio_weights(["https://x/w.pth"], {}, 1.0)
    assert sizes == {"https://x/w.pth": 4321}


@pytest.mark.parametrize("payload", [{"items": "not a list"}, 42, None])
def test_a_collection_of_an_unknown_shape_makes_no_rows(payload):
    assert mz._bioimageio_rows(payload, {}) == []


def test_a_known_network_with_no_weights_file_is_refused_in_words():
    row = mz.ModelEntry(key="k", name="n", uri="")
    reason = mz._bioimageio_refusal({}, row, 0, known=True)
    assert reason == "its package names no weights file to download."


def test_no_weights_source_is_no_address():
    assert mz._bioimageio_uri("happy-elephant", "") == ""


def test_a_corrupt_community_cache_offline_lists_nothing(home):
    cache = home / ".spacr" / "community_models.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("{not json")
    assert mz.community_entries(allow_network=False) == []


def test_community_folders_without_a_checkpoint_are_left_out(home,
                                                             monkeypatch):
    prefix = mz.COMMUNITY_PREFIX
    files = [f"{prefix}good/submission.json", f"{prefix}good/model.pth",
             f"{prefix}readme_only/README.md"]

    class Api:
        def list_repo_files(self, repo):
            return list(files)

    def download(repo, name):
        raise OSError("hub unreachable")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=Api, hf_hub_download=download))
    rows = mz.community_entries(allow_network=True)
    assert [row.name for row in rows] == ["model.pth"]
    assert rows[0].trained_by == "a spaCR user", "no submission record read"
    cached = json.loads((home / ".spacr" / "community_models.json")
                        .read_text())
    assert [r["folder"] for r in cached] == ["good"]


def test_rows_are_grouped_under_every_heading_and_filtered_by_them():
    rows = [mz.ModelEntry(key="a", name="a", source="community"),
            mz.ModelEntry(key="b", name="b", source="bundled")]
    groups = mz.group_by_source(rows)
    assert list(groups) == list(mz.ZOO_SOURCES)
    assert [r.key for r in groups["spaCR community"]] == ["a"]
    assert [r.key for r in groups["spaCR"]] == ["b"]
    assert groups["bioimage.io"] == []
    assert [r.key for r in mz.entries_from_sources(rows, ["spaCR"])] == ["b"]


def test_a_download_dropped_after_the_last_byte_is_not_resumed(monkeypatch):
    import requests

    class Dropping:
        def iter_content(self, chunk_size):
            yield b"abcd"
            raise requests.exceptions.ConnectionError("ended prematurely")

    fetched = []
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: fetched.append(k) or None)
    chunks = mz._resumed_chunks("https://x/w.pth", Dropping(), total=4,
                                timeout=1, chunk_size=2)
    assert next(chunks) == b"abcd"
    with pytest.raises(requests.exceptions.ConnectionError):
        next(chunks)
    assert fetched == [], "nothing is left to fetch, so nothing is asked"


def test_a_hub_that_cannot_be_listed_gives_no_community_rows(home,
                                                             monkeypatch):
    class Api:
        def list_repo_files(self, repo):
            raise OSError("hub unreachable")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=Api, hf_hub_download=None))
    assert mz.community_entries(allow_network=True) == []
    assert not (home / ".spacr" / "community_models.json").exists()
