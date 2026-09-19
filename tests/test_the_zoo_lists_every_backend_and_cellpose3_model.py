"""Item 423 in the model zoo: backend rows with real states, Cellpose 3's
models, bioimage.io's Cellpose 3 checkpoints, and a licence on each row.

Maintainer's decision, 2026-09-19: "Isolated env per backend! But with the
addition of adding cellpose 3 and its cyto, nucleus, and cyto2 and cyto3
models. The cellpose 3 backend will allow more models from the biomass.io to
be added to the model zoo." ("biomass.io" is bioimage.io.)

The bioimage.io manifests below are trimmed copies of the real ones,
read from https://hypha.aicell.io/bioimage-io/artifacts/bioimage.io/children
on 2026-09-19: famous-fish (CellPose(cyto3), a ``CellPoseWrapper``),
philosophical-panda (Cellpose's own ``CPnetBioImageIO`` export), and
idealistic-eagle (Cellpose-SAM, the Cellpose 4 kind).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import spacr._segmentation_backends as SB
from spacr import model_zoo as zoo


@pytest.fixture(autouse=True)
def _sandboxed(tmp_path, monkeypatch):
    """Own backends folder and home, so no real environment or cache counts."""
    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "backends"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(SB, "_PROBED", {})
    monkeypatch.setattr(SB, "_importable", lambda module: False)
    return tmp_path


def _finish(tmp_path, name):
    env = tmp_path / "backends" / name
    python = Path(SB._env_python(str(env)))
    python.parent.mkdir(parents=True)
    python.write_text("")
    SB._write_marker(str(env), {"backend": name})
    return env


_CYTO3 = {
    "alias": "famous-fish",
    "manifest": {
        "type": "model", "name": "CellPose(cyto3)",
        "description": "Cellpose cyto3 on fluorescent cells",
        "license": "BSD-3-Clause",
        "authors": [{"name": "Ada"}, {"name": "Grace"}, "not a mapping"],
        "tags": ["Cellpose", "Cell Segmentation"],
        "weights": {"pytorch_state_dict": {
            "source": "cyto3.pth", "sha256": "2DC3087A",
            "architecture": {"callable": "CellPoseWrapper"}}},
    },
}

_EXPORT = {
    "alias": "philosophical-panda",
    "manifest": {
        "type": "model", "name": "Cellpose Plant Nuclei ResNet",
        "license": "MIT", "tags": ["cellpose", "nuclei"],
        "weights": {"pytorch_state_dict": {
            "source": "https://mirror/cp_state_dict", "sha256": "26c2",
            "architecture": {"callable": "CPnetBioImageIO"}}},
    },
}

_SAM = {
    "alias": "idealistic-eagle",
    "manifest": {
        "type": "model", "name": "Cellpose-SAM",
        "tags": ["cellpose"], "license": "BSD-3-Clause",
        "weights": {"pytorch_state_dict": {
            "source": "cpsam.pth",
            "architecture": {"callable": "CellPoseWrapper"}}},
    },
}


def test_kinds_name_the_cellpose3_kind():
    assert "cellpose3" in zoo.KINDS
    zoo.ModelEntry(key="k", name="n", kind="cellpose3")


def test_every_backend_is_listed_with_its_state_reason_and_licence(tmp_path):
    rows = {e.key: e for e in zoo.installable_backend_entries()}
    assert set(rows) == {"cellpose3_v1", "dinocell_v1", "samcell_v1"}
    cellpose3 = rows["cellpose3_v1"]
    assert (cellpose3.kind, cellpose3.source) == ("backend", "installable")
    assert cellpose3.uri == "backend:cellpose3" and cellpose3.path == ""
    assert cellpose3.licence == "BSD-3-Clause"
    assert cellpose3.notes[0].startswith("installable: installs into an "
                                         "environment of its own")
    assert "Howard Hughes Medical Institute" in cellpose3.notes[1]
    assert rows["samcell_v1"].licence == rows["dinocell_v1"].licence == "MIT"

    env = _finish(tmp_path, "cellpose3")
    ready = {e.key: e for e in zoo.installable_backend_entries()}
    assert ready["cellpose3_v1"].source == "installed"
    assert ready["cellpose3_v1"].path == str(env)
    assert zoo.INSTALLABLE_BACKENDS["cellpose3"] == (
        "Cellpose 3", "backend:cellpose3", "cellpose",
        SB._SPECS["cellpose3"].blurb)


def test_one_installed_inside_spacr_has_no_environment_to_point_at(
        monkeypatch):
    monkeypatch.setattr(SB, "_importable", lambda module: True)
    samcell = {e.key: e for e in zoo.installable_backend_entries()}[
        "samcell_v1"]
    assert samcell.source == "installed" and samcell.path == ""


def test_cellpose3s_models_are_listed_and_usable_once_it_is_installed(
        tmp_path):
    rows = zoo._cellpose3_model_entries()
    assert [e.name for e in rows] == ["cyto3", "cyto2", "cyto", "nuclei"]
    assert {e.kind for e in rows} == {"cellpose3"}
    assert all(e.path == "" and "needs the Cellpose 3 backend" in e.notes[0]
               for e in rows)
    assert rows[3].trained_on == (
        "Cellpose's nucleus model. Runs through the Cellpose 3 backend.")
    assert {e.licence for e in rows} == {"BSD-3-Clause"}
    _finish(tmp_path, "cellpose3")
    ready = zoo._cellpose3_model_entries()
    assert [e.path for e in ready] == ["cyto3", "cyto2", "cyto", "nuclei"]
    assert all(e.notes == () for e in ready)


@pytest.mark.parametrize("manifest, expected", [
    (_CYTO3["manifest"], ("cyto3.pth", "2dc3087a")),
    (_EXPORT["manifest"], ("https://mirror/cp_state_dict", "26c2")),
    (_SAM["manifest"], None),
    ({"type": "application", "name": "Cellpose notebook"}, None),
    ({"type": "model", "weights": "not a mapping"}, None),
    ({"type": "model", "weights": {"pytorch_state_dict": {
        "source": "x.pth", "architecture": "not a mapping"}}}, None),
    ({"type": "model", "weights": {"pytorch_state_dict": {
        "source": "x.pth", "architecture": {"callable": "UNet2D"}}}}, None),
    ({"type": "model", "weights": {"pytorch_state_dict": {
        "architecture": {"callable": "CellPoseWrapper"}}}}, None),
])
def test_only_the_two_cellpose3_architectures_count(manifest, expected):
    assert zoo._cellpose3_weights(manifest) == expected


def test_a_bioimageio_cellpose3_row_downloads_its_checkpoint_with_its_hash():
    row = zoo._cellpose3_download_entry(
        "famous-fish", "cellpose_cyto3", "CellPose(cyto3)",
        _CYTO3["manifest"], "Ada", "cyto3.pth", "2dc3087a")
    assert row.uri == ("https://hypha.aicell.io/bioimage-io/artifacts/"
                       "famous-fish/files/cyto3.pth")
    assert (row.name, row.kind, row.source) == (
        "cellpose_cyto3.pth", "cellpose3", "bioimage.io")
    assert row.sha256 == "2dc3087a" and row.licence == "BSD-3-Clause"
    assert "segmentation_backend to cellpose3" in row.notes[0]
    bare = zoo._cellpose3_download_entry(
        "a", "slug", "t", {}, "", "https://x/cellpose_model", "")
    assert (bare.uri, bare.name, bare.licence) == (
        "https://x/cellpose_model", "slug.pth", "")
    kept = zoo._cellpose3_download_entry("a", "slug", "t", {}, "", "w.pt", "")
    assert kept.name == "slug.pt"


def test_the_bioimageio_listing_offers_both_kinds_from_one_cache(tmp_path):
    cache = tmp_path / "home" / ".spacr" / "bioimageio_children.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps([_CYTO3, _EXPORT, _SAM, "not an item",
                                 {"alias": "", "manifest": _CYTO3["manifest"]},
                                 {"alias": "odd", "manifest": ["a list"]},
                                 {"alias": "unet", "manifest": {
                                     "type": "model", "name": "UNet 2D"}}]))
    rows = {e.key: e for e in zoo.bioimageio_entries()}
    assert set(rows) == {"bioimageio_cellpose_cyto3",
                         "bioimageio_cellpose_plant_nuclei_resnet",
                         "cellpose_sam"}
    assert rows["bioimageio_cellpose_cyto3"].trained_by == "Ada, Grace"
    assert rows["bioimageio_cellpose_plant_nuclei_resnet"].uri == (
        "https://mirror/cp_state_dict")
    assert rows["cellpose_sam"].kind == "cellpose", (
        "a Cellpose-SAM model is spaCR's own Cellpose's, not Cellpose 3's")


def test_the_catalogue_lists_them_without_the_network(tmp_path):
    kinds = [e.kind for e in zoo.catalogue(remote=False,
                                           include_plugins=False)]
    assert kinds.count("cellpose3") == 4
    assert kinds.count("backend") == 3


def test_a_row_names_the_backend_it_needs():
    assert zoo._backend_for(SimpleNamespace(uri="backend:dinocell",
                                            kind="backend")) == "dinocell"
    assert zoo._backend_for(SimpleNamespace(uri="https://x",
                                            kind="cellpose3")) == "cellpose3"
    assert zoo._backend_for(SimpleNamespace(uri="", kind="cellpose")) == ""


def test_the_licence_travels_on_the_row_and_into_its_card():
    entry = zoo._entry_from_mapping({"name": "m.pth", "license": "CC-BY-4.0",
                                     "sha256": "ab"})
    assert entry.licence == "CC-BY-4.0"
    assert "  licence    CC-BY-4.0" in entry.describe().splitlines()
    british = zoo._entry_from_mapping({"name": "m.pth", "licence": "MIT",
                                       "sha256": "ab"})
    assert british.licence == "MIT"
    assert "licence" not in zoo.ModelEntry(key="k", name="n").describe()


def test_the_real_listing_carries_the_real_licences():
    """The licences were read from each release, not assumed: the texts at
    the tags and on PyPI, 2026-09-19."""
    specs = SB._SPECS
    assert specs["cellpose3"].licence == "BSD-3-Clause"
    assert specs["dinocell"].licence == "MIT"
    assert specs["samcell"].licence == "MIT"
    assert "Apache-2.0" in specs["samcell"].licence_note
    assert os.path.basename(specs["cellpose3"].homepage) == "cellpose"
