"""Item 504: the bioimage.io category carries every Cellpose-SAM and Cellpose
3 model bioimage.io publishes, and a downloaded one runs.

The collection below is trimmed from bioimage.io's own, read on 2026-09-25
from https://hypha.aicell.io/bioimage-io/artifacts/bioimage.io/children:
nine Cellpose models in 277 items, of which

* four Cellpose 3 packages (``CellPoseWrapper`` / ``CPnetBioImageIO``) whose
  weights are Cellpose 3 checkpoints -- famous-fish's ``cyto3.pth`` is
  byte-for-byte Cellpose 3's own cyto3;
* one more (happy-elephant) whose published weights are a 1596-byte
  stand-in, its code downloading the real checkpoint from GitHub at run
  time;
* two Cellpose-SAM models, one byte-for-byte ``cpsam_v2``;
* two Cellpose-DINO models, which Cellpose 4 builds only with ``dinov3``.

What is pinned:

1. the listing: each kind recognised, the licence and what it was trained on
   on every row, and a row spaCR cannot run says so and offers no download;
2. the cache: a stale collection is still listed offline, a refresh measures
   the weights files and a stand-in is refused before anyone downloads it;
3. a download: fetched and checksummed by the zoo's own fetch, a stand-in
   refused, and a dropped connection resumed;
4. the model setting it becomes, and the route that setting takes through
   Mask generation: a Cellpose 3 checkpoint takes exactly the route and the
   arguments a stock Cellpose 3 model takes, a Cellpose-SAM checkpoint is
   loaded by spaCR's own Cellpose 4.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import pytest

import spacr._segmentation_backends as SB
import spacr.object as O
from spacr import model_zoo as zoo
import tests.test_object_tstack_wiring as _wiring

fake_model = _wiring.fake_model
force_cpu = _wiring.force_cpu
_base_settings = _wiring._base_settings
_write_npz = _wiring._write_npz

FILES = "https://hypha.aicell.io/bioimage-io/artifacts/{}/files/{}"


def _model(alias, name, arch, source, *, sha="", licence="BSD-3-Clause",
           tags=("cellpose",), description="", kwargs=None, **extra):
    architecture = {"callable": arch}
    if kwargs is not None:
        architecture["kwargs"] = kwargs
    manifest = {"type": "model", "name": name, "description": description,
                "license": licence, "tags": list(tags),
                "authors": [{"name": "Ada"}, {"name": "Grace"}],
                "weights": {"pytorch_state_dict": {
                    "source": source, "sha256": sha,
                    "architecture": architecture}}}
    manifest.update(extra)
    return {"alias": alias, "manifest": manifest}


CYTO3 = _model("famous-fish", "CellPose(cyto3)", "CellPoseWrapper",
               "cyto3.pth", sha="2DC3087A", description="CellPose 'cyto3' model",
               kwargs={"model_type": "cyto3", "diam_mean": None,
                       "flow_threshold": 0.4, "cellprob_threshold": 0,
                       "channels": [0, 0], "gpu": False})
MUSCLE = _model("thoughtful-chipmunk", "OC1 Project 6 Cellpose",
                "CellPoseWrapper", "cellpose_model", sha="a8c0",
                description="Analysis of the fiber profile of skeletal muscle",
                kwargs={"flow_threshold": 0.7})
STANDIN = _model("happy-elephant", "OC1 Project 11 Cellpose", "CellPoseWrapper",
                 "model_weights.pth", sha="02b8", licence="CC-BY-NC-4.0",
                 description="Segmentation of Epithelial Cells")
SAM = _model("idealistic-eagle", "Cellpose-SAM", "CPnetBioImageIO", "cpsam",
             sha="0F1CC3F7", tags=("instance-segmentation", "cellpose"),
             description="Cellpose-SAM, the generalist")
DINO = _model("famous-sheep", "CellposeDINO ViT-L 2D Microscopy Instance "
              "Segmenter", "CellposeInstanceLabelsModel",
              "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpdino",
              sha="fad0", tags=("cellpose", "cellpose-dino"),
              description="CellposeDINO with a DINOv3 ViT-L backbone",
              training_data={"id": "10.25378/janelia.13270466"})
ODD = _model("odd-otter", "Cellpose flavoured StarDist", "StarDistNet",
             "w.pth", tags=("cellpose", "stardist"))
NOTEBOOK = {"alias": "polished-t-shirt", "manifest": {
    "type": "application", "name": "Cellpose (2D and 3D) - ZeroCostDL4Mic",
    "tags": ["Cellpose"]}}
UNET = _model("plain-unet", "UNet 2D nuclei", "UNet2d", "unet.pth",
              tags=("nuclei",))

COLLECTION = [CYTO3, MUSCLE, STANDIN, SAM, DINO, ODD, NOTEBOOK, UNET]


@pytest.fixture(autouse=True)
def _sandboxed(tmp_path, monkeypatch):
    """Own home, so no real cache counts, and no backend environments."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "backends"))
    monkeypatch.setattr(zoo, "_dinov3_available", lambda: False)
    return tmp_path


def _cache(tmp_path, payload=COLLECTION, age_hours=0.0, sizes=None):
    folder = tmp_path / "home" / ".spacr"
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / "bioimageio_children.json"
    cache.write_text(json.dumps(payload))
    if age_hours:
        then = time.time() - age_hours * 3600
        os.utime(cache, (then, then))
    if sizes is not None:
        (folder / "bioimageio_weight_sizes.json").write_text(json.dumps(sizes))
    return cache


def _rows(**kwargs):
    return {e.key: e for e in zoo.bioimageio_entries(**kwargs)}


# ===========================================================================
# 1. The listing
# ===========================================================================

def test_each_cellpose_model_is_listed_once_and_nothing_else(tmp_path):
    _cache(tmp_path)
    rows = _rows()
    assert set(rows) == {
        "bioimageio_cellpose_cyto3", "bioimageio_oc1_project_6_cellpose",
        "bioimageio_oc1_project_11_cellpose", "bioimageio_cellpose_sam",
        "bioimageio_cellposedino_vit_l_2d_microscopy_instance_segmenter",
        "bioimageio_cellpose_flavoured_stardist"}
    assert {e.source for e in rows.values()} == {"bioimage.io"}


def test_the_kinds_are_the_cellpose_that_runs_them(tmp_path):
    _cache(tmp_path)
    rows = _rows()
    assert rows["bioimageio_cellpose_cyto3"].kind == "cellpose3"
    assert rows["bioimageio_oc1_project_6_cellpose"].kind == "cellpose3"
    assert rows["bioimageio_cellpose_sam"].kind == "cellpose", (
        "Cellpose-SAM is spaCR's own Cellpose 4's, not Cellpose 3's")
    assert rows["bioimageio_cellpose_sam"].name == "cellpose_sam"
    assert rows["bioimageio_cellpose_cyto3"].name == "cellpose_cyto3.pth"


def test_every_row_carries_its_licence_and_what_it_was_trained_on(tmp_path):
    _cache(tmp_path)
    rows = _rows()
    for row in rows.values():
        assert row.licence, row.key
        assert row.trained_on and row.trained_on != zoo.UNKNOWN, row.key
        assert row.trained_by == "Ada, Grace"
    assert rows["bioimageio_oc1_project_11_cellpose"].licence == "CC-BY-NC-4.0"
    assert rows["bioimageio_oc1_project_6_cellpose"].trained_on == (
        "OC1 Project 6 Cellpose — Analysis of the fiber profile of skeletal "
        "muscle")
    dino = rows["bioimageio_cellposedino_vit_l_2d_microscopy_instance_segmenter"]
    assert dino.trained_on.startswith(
        "CellposeDINO ViT-L 2D Microscopy Instance Segmenter — trained on "
        "https://doi.org/10.25378/janelia.13270466.")
    described = rows["bioimageio_cellpose_sam"].describe()
    assert "licence    BSD-3-Clause" in described
    assert "trained on Cellpose-SAM — Cellpose-SAM, the generalist" in described


def test_a_runnable_row_downloads_its_weights_with_their_hash(tmp_path):
    _cache(tmp_path)
    rows = _rows()
    cyto3 = rows["bioimageio_cellpose_cyto3"]
    assert cyto3.uri == FILES.format("famous-fish", "cyto3.pth")
    assert cyto3.sha256 == "2dc3087a"
    sam = rows["bioimageio_cellpose_sam"]
    assert sam.uri == FILES.format("idealistic-eagle", "cpsam")
    assert sam.sha256 == "0f1cc3f7"
    for row in (cyto3, sam):
        assert zoo._bioimageio_cannot_run(row) == ""


def test_the_use_text_no_longer_sends_anyone_to_segmentation_backend(tmp_path):
    _cache(tmp_path)
    rows = _rows()
    cyto3 = rows["bioimageio_cellpose_cyto3"].notes[0]
    assert "segmentation_backend" not in cyto3
    assert "exactly as a stock Cellpose 3 model does" in cyto3
    assert "cellpose3:<its path> in the object's model setting" in cyto3
    sam = rows["bioimageio_cellpose_sam"].notes[0]
    assert "spaCR's own Cellpose 4" in sam and "its path" in sam


def test_the_packages_own_run_settings_are_reported_not_used(tmp_path):
    _cache(tmp_path)
    notes = _rows()["bioimageio_cellpose_cyto3"].notes
    assert notes[1] == (
        "Its bioimage.io package runs it with flow_threshold=0.4, "
        "cellprob_threshold=0, channels=[0, 0]; spaCR runs it with the "
        "object's Mask generation settings instead, as it runs every other "
        "model.")


def test_a_dino_model_says_it_cannot_run_without_dinov3(tmp_path, monkeypatch):
    _cache(tmp_path)
    key = "bioimageio_cellposedino_vit_l_2d_microscopy_instance_segmenter"
    dino = _rows()[key]
    why = zoo._bioimageio_cannot_run(dino)
    assert why.startswith(zoo._CANNOT_RUN)
    assert "dinov3" in why and dino.uri == ""
    assert dino.notes[1] == "bioimage.io model famous-sheep"
    monkeypatch.setattr(zoo, "_dinov3_available", lambda: True)
    ready = _rows()[key]
    assert zoo._bioimageio_cannot_run(ready) == ""
    assert ready.uri == ("https://huggingface.co/mouseland/cellpose-sam/"
                         "resolve/main/cpdino")


def test_a_network_that_is_no_cellpose_says_so(tmp_path):
    _cache(tmp_path)
    odd = _rows()["bioimageio_cellpose_flavoured_stardist"]
    why = zoo._bioimageio_cannot_run(odd)
    assert "(StarDistNet)" in why and "neither Cellpose 3's nor" in why
    assert odd.uri == ""


def test_a_stand_in_weights_file_is_refused_before_download(tmp_path):
    standin = FILES.format("happy-elephant", "model_weights.pth")
    _cache(tmp_path, sizes={standin: 1596,
                            FILES.format("famous-fish", "cyto3.pth"): 26566255})
    rows = _rows()
    row = rows["bioimageio_oc1_project_11_cellpose"]
    why = zoo._bioimageio_cannot_run(row)
    assert "1596-byte stand-in" in why and row.uri == ""
    assert row.size_bytes == 1596
    cyto3 = rows["bioimageio_cellpose_cyto3"]
    assert cyto3.size_bytes == 26566255
    assert zoo._bioimageio_cannot_run(cyto3) == ""


def test_other_rows_never_say_they_cannot_run():
    plain = zoo.ModelEntry(key="k", name="n", notes=("anything",))
    assert zoo._bioimageio_cannot_run(plain) == ""
    assert zoo._bioimageio_cannot_run(types.SimpleNamespace()) == ""


# ===========================================================================
# 2. The cache
# ===========================================================================

def test_a_stale_collection_is_still_listed_offline(tmp_path):
    _cache(tmp_path, age_hours=zoo.BIOIMAGEIO_CACHE_HOURS + 5)
    assert "bioimageio_cellpose_cyto3" in _rows()


def test_nothing_cached_and_no_network_is_no_rows():
    assert zoo.bioimageio_entries() == []


def test_an_unreachable_bioimageio_falls_back_to_the_stale_list(
        tmp_path, monkeypatch):
    _cache(tmp_path, age_hours=zoo.BIOIMAGEIO_CACHE_HOURS + 5)
    monkeypatch.setattr(zoo, "_weights_size", lambda uri, timeout: 0)
    rows = _rows(allow_network=True, url=(tmp_path / "gone.json").as_uri())
    assert "bioimageio_cellpose_sam" in rows


def test_a_refresh_fetches_the_collection_and_measures_the_weights(
        tmp_path, monkeypatch):
    served = tmp_path / "served.json"
    served.write_text(json.dumps(COLLECTION))
    asked = []

    def measure(uri, timeout):
        asked.append(uri)
        return 1596 if "happy-elephant" in uri else 26_000_000

    monkeypatch.setattr(zoo, "_weights_size", measure)
    rows = _rows(allow_network=True, url=served.as_uri())
    cache = tmp_path / "home" / ".spacr" / "bioimageio_children.json"
    assert json.loads(cache.read_text()) == COLLECTION
    assert sorted(asked) == sorted([
        FILES.format("famous-fish", "cyto3.pth"),
        FILES.format("thoughtful-chipmunk", "cellpose_model"),
        FILES.format("happy-elephant", "model_weights.pth"),
        FILES.format("idealistic-eagle", "cpsam")]), (
        "only rows that could be downloaded are measured")
    assert "stand-in" in zoo._bioimageio_cannot_run(
        rows["bioimageio_oc1_project_11_cellpose"])
    asked.clear()
    again = zoo.bioimageio_entries()
    assert asked == [] and "stand-in" in zoo._bioimageio_cannot_run(
        {e.key: e for e in again}["bioimageio_oc1_project_11_cellpose"]), (
        "the measurement is remembered for the offline listing")


def test_the_catalogue_carries_the_rows_without_the_network(tmp_path,
                                                           monkeypatch):
    _cache(tmp_path)
    monkeypatch.setattr(zoo, "shared_catalogue", lambda *a, **k: [])
    listed = [e for e in zoo.catalogue(remote=True, block=False,
                                       include_plugins=False)
              if e.source == "bioimage.io"]
    assert {e.kind for e in listed} == {"cellpose", "cellpose3"}
    assert len(listed) == 6


# ===========================================================================
# 3. The download
# ===========================================================================

def _served_row(tmp_path, key, payload):
    """A listed row whose weights are served from a local file."""
    weights = tmp_path / "served" / "weights.bin"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(payload)
    _cache(tmp_path)
    row = _rows()[key]
    from dataclasses import replace
    return replace(row, uri=str(weights),
                   sha256=hashlib.sha256(payload).hexdigest())


def test_a_download_is_checksummed_and_named_after_the_model(tmp_path):
    payload = os.urandom(zoo._BIOIMAGEIO_MIN_WEIGHTS + 10)
    row = _served_row(tmp_path, "bioimageio_cellpose_cyto3", payload)
    path = zoo.fetch(row, tmp_path / "models")
    assert path == tmp_path / "models" / "cellpose_cyto3.pth"
    assert path.read_bytes() == payload


def test_a_stand_in_that_slipped_through_is_not_installed(tmp_path):
    row = _served_row(tmp_path, "bioimageio_cellpose_cyto3", b"x" * 1596)
    with pytest.raises(zoo.ModelZooError, match="too small to be a Cellpose"):
        zoo.fetch(row, tmp_path / "models")
    assert not any((tmp_path / "models").iterdir())


class _Dropping:
    """A ``requests`` response that sends some bytes, then drops."""

    def __init__(self, blocks, drop, status=200, headers=None):
        self.blocks, self.drop = blocks, drop
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1):
        import requests

        yield from self.blocks
        if self.drop:
            raise requests.exceptions.ChunkedEncodingError("ended prematurely")


def test_a_dropped_download_resumes_where_it_stopped(monkeypatch):
    import requests

    asked = []

    def get(url, stream=True, timeout=None, headers=None):
        asked.append(headers)
        if headers is None:
            return _Dropping([b"abc", b"def"], drop=True)
        assert headers == {"Range": "bytes=6-"}
        return _Dropping([b"ghi"], drop=False, status=206,
                         headers={"content-range": "bytes 6-8/9"})

    monkeypatch.setattr(requests, "get", get)
    chunks, _total = zoo.open_uri("https://example.org/w")
    assert b"".join(chunks) == b"abcdefghi"
    assert asked == [None, {"Range": "bytes=6-"}]


def test_a_server_that_will_not_resume_still_fails(monkeypatch):
    import requests

    def get(url, stream=True, timeout=None, headers=None):
        if headers is None:
            return _Dropping([b"abc"], drop=True)
        return _Dropping([b"abcdef"], drop=False, status=200)

    monkeypatch.setattr(requests, "get", get)
    chunks, _total = zoo.open_uri("https://example.org/w")
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        b"".join(chunks)


# ===========================================================================
# 4. The model setting it becomes, and its route
# ===========================================================================

def _downloaded(tmp_path, key):
    payload = os.urandom(zoo._BIOIMAGEIO_MIN_WEIGHTS + 10)
    row = _served_row(tmp_path, key, payload)
    return row, zoo.fetch(row, tmp_path / "models")


def test_a_downloaded_cellpose3_model_is_a_cellpose3_setting(tmp_path):
    _row, path = _downloaded(tmp_path, "bioimageio_cellpose_cyto3")
    value = SB._cellpose3_value(str(path))
    assert value == f"cellpose3:{path}"
    assert SB._cellpose3_choice(value) == str(path)
    assert SB._cellpose3_model(value, object_type="cell") == str(path)
    assert SB._cellpose3_is_chosen({"cell_model_name": value})


def test_a_deleted_download_is_refused_not_run_as_cyto3(tmp_path):
    _row, path = _downloaded(tmp_path, "bioimageio_cellpose_cyto3")
    value = SB._cellpose3_value(str(path))
    path.unlink()
    with pytest.raises(FileNotFoundError, match="not there"):
        SB._cellpose3_model(value, object_type="cell")


def test_a_downloaded_cellpose_sam_model_is_its_path(tmp_path):
    from spacr.utils import _resolve_cellpose_pretrained

    _row, path = _downloaded(tmp_path, "bioimageio_cellpose_sam")
    assert SB._cellpose3_choice(str(path)) is None
    assert not SB._cellpose3_is_chosen({"cell_model_name": str(path)})
    assert _resolve_cellpose_pretrained(str(path), object_type="cell") == str(path)


def _spy_backend(monkeypatch):
    seen = []

    def _eval(x, **kwargs):
        seen[-1]["eval"].append(dict(
            kwargs, images=[hashlib.sha256(np.ascontiguousarray(i)).hexdigest()
                            for i in x]))
        masks = [np.zeros(np.shape(i)[:2], np.uint16) for i in x]
        return masks, [[None] * 4 for _ in x], None

    def _load(name, **kwargs):
        seen.append(dict(kwargs, name=name, eval=[]))
        return types.SimpleNamespace(eval=_eval)

    def _no_cellpose(*args, **kwargs):
        raise AssertionError("Cellpose 4 was built for a Cellpose 3 object")

    monkeypatch.setattr(SB, "_load_backend", _load)
    monkeypatch.setattr(O, "cp_models",
                        types.SimpleNamespace(CellposeModel=_no_cellpose))
    return seen


def test_a_bioimageio_cellpose3_model_takes_the_stock_models_route(
        tmp_path, monkeypatch):
    """Same backend, same load, same images, same eval keywords as cyto3:
    only the model named differs."""
    _row, path = _downloaded(tmp_path, "bioimageio_cellpose_cyto3")
    seen = _spy_backend(monkeypatch)
    for value, where in (("cellpose3:cyto3", "stock"),
                         (SB._cellpose3_value(str(path)), "bioimageio")):
        src = tmp_path / where / "plate" / "masks"
        _write_npz(src, (1, 24, 24, 2))
        O.generate_cellpose_masks_sam(
            str(src), _base_settings(src, cell_model_name=value), "cell")
    stock, bioimageio = seen
    assert stock["name"] == bioimageio["name"] == "cellpose3"
    assert stock.pop("model_name") == "cellpose3:cyto3"
    assert bioimageio.pop("model_name") == f"cellpose3:{path}"
    assert stock == bioimageio


def test_a_bioimageio_cellpose_sam_model_is_loaded_by_cellpose4(
        tmp_path, monkeypatch, fake_model):
    _row, path = _downloaded(tmp_path, "bioimageio_cellpose_sam")

    def _no_backend(*args, **kwargs):
        raise AssertionError("a Cellpose-SAM model went to a backend")

    monkeypatch.setattr(SB, "_load_backend", _no_backend)
    src = tmp_path / "plate" / "masks"
    _write_npz(src, (1, 24, 24, 2))
    O.generate_cellpose_masks_sam(
        str(src), _base_settings(src, cell_model_name=str(path)), "cell")
    assert fake_model["model"].pretrained_model == str(path)
