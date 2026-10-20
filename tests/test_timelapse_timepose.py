"""The version-agnostic Timepose release loader and tracker (item 711).

No release is registered yet. A release lives in a Hugging Face repository,
so every test here registers a fake one and replaces the download and the
network: a fake downloader writes small files whose sha256 the test pins, and a fake adapter stands in for the model. No
token, network or GPU is needed.
"""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import model_zoo
from spacr import timelapse as tl

FILES = {
    "weights/w.pt": b"weights",
    "fakepose/__init__.py": b"",
    "fakepose/infer.py": (b"CONTRACT = 'spacr.timepose/1'\n"
                          b"def load_model(folder, device):\n"
                          b"    return {'folder': folder, 'device': device}\n"),
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record(key="timepose_test", contract="spacr.timepose/1"):
    return {"key": key, "kind": "timepose", "name": key,
            "repo_id": "someone/private-timepose", "revision": "a" * 40,
            "entry": "fakepose.infer", "contract": contract,
            "weights": "weights/w.pt", "display_name": "Timepose test",
            "files": {name: _sha(data) for name, data in FILES.items()}}


def _downloader(content=None, calls=None):
    def download(repo_id, filename, revision, repo_type, local_dir, token):
        if calls is not None:
            calls.append((repo_id, filename, revision, token))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((content or FILES)[filename])
        return str(path)
    return download


@pytest.fixture
def release(monkeypatch):
    record = _record()
    monkeypatch.setattr(model_zoo, "BUNDLED_REMOTE_MODELS",
                        tuple(model_zoo.BUNDLED_REMOTE_MODELS) + (record,))
    monkeypatch.setattr(model_zoo, "_TIMEPOSE_VERIFIED", {})
    monkeypatch.setattr(model_zoo, "_TIMEPOSE_MODELS", {})
    return record


def _movie():
    """Four frames: object 1 moves right; object 2 divides into 3 and 4 at t=2."""
    masks = np.zeros((4, 32, 32), np.int32)
    for t in range(4):
        masks[t, 4:8, 2 + t:6 + t] = 1
    masks[0, 20:26, 10:16] = 2
    masks[1, 20:26, 10:16] = 2
    masks[2, 18:22, 10:14] = 3
    masks[2, 24:28, 10:14] = 4
    masks[3, 18:22, 11:15] = 3
    masks[3, 24:28, 11:15] = 4
    edges = [[0, 1, 1], [1, 1, 1], [2, 1, 1], [0, 2, 2], [1, 2, 3], [1, 2, 4],
             [2, 3, 3], [2, 4, 4]]
    return masks, edges


class FakeAdapter:
    CONTRACT = "spacr.timepose/1"

    def __init__(self, edges):
        self.edges = edges
        self.calls = []

    def segment_track_movie(self, model, images, invert=False, masks=None):
        self.calls.append(dict(images=np.asarray(images), invert=invert, masks=masks))
        own = np.asarray(masks) if masks is not None else _movie()[0]
        return {"masks": own, "edges": self.edges, "divisions": [(1, 2, 3, 4)]}


def test_edges_become_tracks_with_a_division():
    masks, edges = _movie()
    tracked, parents = tl._edges_to_tracked_stack(masks, edges)
    assert tracked.shape == masks.shape
    mover = {int(np.unique(tracked[t][masks[t] == 1])[0]) for t in range(4)}
    assert len(mover) == 1
    mother = int(np.unique(tracked[1][masks[1] == 2])[0])
    daughters = {int(np.unique(tracked[2][masks[2] == k])[0]) for k in (3, 4)}
    assert len(daughters) == 2 and mother not in daughters
    assert {parents[d] for d in daughters} == {mother}
    for k in (3, 4):
        assert np.unique(tracked[3][masks[3] == k]).tolist() == \
            np.unique(tracked[2][masks[2] == k]).tolist()


def test_two_parents_for_one_child_are_refused():
    masks, _ = _movie()
    with pytest.raises(ValueError):
        tl._edges_to_tracked_stack(masks, [[0, 1, 1], [0, 2, 1]])


def test_the_backend_matches_the_trackastra_table(tmp_path, monkeypatch):
    masks, edges = _movie()
    adapter = FakeAdapter(edges)
    monkeypatch.setattr(model_zoo, "_load_timepose",
                        lambda key, device=None: (adapter, object()))
    src = tmp_path / "run" / "masks"
    src.mkdir(parents=True)
    images = np.random.default_rng(0).random((4, 32, 32, 2)).astype(np.float32)
    stack = tl._timepose_track_cells(src=str(src), name="field1", mode="timepose_test",
                                     batch_filenames=[f"f{t}.tif" for t in range(4)],
                                     object_type="cell", masks=masks, images=images,
                                     invert=True)
    assert np.asarray(stack).shape == masks.shape
    sent = adapter.calls[0]
    assert sent["images"].shape == (4, 2, 32, 32)
    assert sent["invert"] is True
    assert np.array_equal(sent["masks"], masks)
    table = pd.read_csv(tmp_path / "run" / "tracks" / "timepose_test_tracks_cell_field1.csv")
    expected = list(tl._relabelled_stack_to_tracks_df(np.asarray(stack)).columns)
    assert list(table.columns[:len(expected)]) == expected
    assert {"parent_track_id", "parent_track_id_source"} <= set(table.columns)
    daughters = table[table["parent_track_id"] > 0]
    assert daughters["track_id"].nunique() == 2
    assert set(daughters["parent_track_id_source"]) == {"timepose"}


def test_own_masks_ask_the_model_to_segment(tmp_path, monkeypatch):
    masks, edges = _movie()
    adapter = FakeAdapter(edges)
    monkeypatch.setattr(model_zoo, "_load_timepose",
                        lambda key, device=None: (adapter, object()))
    src = tmp_path / "masks"
    src.mkdir()
    tl._timepose_track_cells(src=str(src), name="f", batch_filenames=None,
                             object_type="cell", masks=np.zeros_like(masks),
                             images=np.ones((4, 32, 32), np.float32), own_masks=True)
    assert adapter.calls[0]["masks"] is None


def test_images_are_required(tmp_path):
    masks, _ = _movie()
    with pytest.raises(ValueError):
        tl._timepose_track_cells(src=str(tmp_path / "m"), name="f", batch_filenames=None,
                                 object_type="cell", masks=masks, images=None)


def test_no_token_names_the_fix(release, tmp_path, monkeypatch):
    monkeypatch.setattr(model_zoo, "_timepose_token", lambda token=None: None)
    with pytest.raises(model_zoo._TimeposeAccessError) as caught:
        model_zoo._timepose_files(release["key"], dest=tmp_path)
    assert "hf auth login" in str(caught.value)
    assert release["repo_id"] in str(caught.value)


def test_no_access_says_so(release, tmp_path, monkeypatch):
    monkeypatch.setattr(model_zoo, "_timepose_token", lambda token=None: "hf_x")
    RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})

    def refuse(**kwargs):
        raise RepositoryNotFoundError("401")
    with pytest.raises(model_zoo._TimeposeAccessError) as caught:
        model_zoo._timepose_files(release["key"], dest=tmp_path, downloader=refuse)
    assert "cannot read" in str(caught.value)


def test_a_wrong_checksum_deletes_the_file_and_loads_nothing(release, tmp_path, monkeypatch):
    monkeypatch.setattr(model_zoo, "_timepose_token", lambda token=None: "hf_x")
    bad = dict(FILES, **{"fakepose/infer.py": b"import os\n"})
    with pytest.raises(model_zoo.ChecksumMismatch):
        model_zoo._load_timepose(release["key"], device="cpu", dest=tmp_path,
                                 downloader=_downloader(bad))
    folder = tmp_path / release["key"] / release["revision"][:12]
    assert not (folder / "fakepose" / "infer.py").exists()
    assert not model_zoo._TIMEPOSE_MODELS


def test_a_verified_release_loads_through_its_adapter(release, tmp_path, monkeypatch):
    monkeypatch.setattr(model_zoo, "_timepose_token", lambda token=None: "hf_x")
    calls = []
    adapter, model = model_zoo._load_timepose(release["key"], device="cpu", dest=tmp_path,
                                              downloader=_downloader(calls=calls))
    assert adapter.CONTRACT == "spacr.timepose/1"
    assert model["device"] == "cpu"
    assert {c[1] for c in calls} == set(FILES)
    assert {c[2] for c in calls} == {release["revision"]}
    calls.clear()
    model_zoo._TIMEPOSE_MODELS.clear()
    model_zoo._load_timepose(release["key"], device="cpu", dest=tmp_path,
                             downloader=_downloader(calls=calls))
    assert calls == []
    for name in list(sys.modules):
        if name.startswith("_spacr_timepose_test"):
            sys.modules.pop(name)


def test_no_release_is_offered_until_one_is_registered():
    from spacr.settings import ALPHA_FEATURES
    from spacr.settings_spec import convert_settings_dict_for_gui

    records = [r for r in model_zoo.BUNDLED_REMOTE_MODELS if r.get("kind") == "timepose"]
    assert records == []
    assert 711 not in ALPHA_FEATURES
    spec = convert_settings_dict_for_gui({"timelapse_mode": "trackastra"})
    assert not [m for m in spec["timelapse_mode"][1] if str(m).startswith("timepose")]
    with pytest.raises(model_zoo.ModelZooError):
        model_zoo._timepose_record("timepose_v1")
