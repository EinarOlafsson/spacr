"""The vvvv gamma project written by the alpha vvvv export (item N685).

The generated ``.vl`` is checked against documents vvvv gamma 7.4 itself
wrote: two help patches shipped in its VL.Skia package and the project
spaCR generates for a sample field, opened and saved again by vvvv 7.4.1
(``tests/data/vvvv``). A real vvvv run is not possible in CI; the
maintainer steps are in the N685 item notes.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ElementTree
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr import tabular
from spacr.qt import mask_engine as engine

DATA = Path(__file__).resolve().parents[1] / "data" / "vvvv"
OFFICIAL = ("vl_skia_example_bouncing_ball.vl", "vl_skia_howto_draw_text.vl")
SAVED = "spacr_field_01_saved_by_vvvv_7_4.vl"
DIGITS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
FIELD_COLUMNS = [
    "label", "centroid_x", "centroid_y", "area", "bbox_x0", "bbox_y0",
    "bbox_x1", "bbox_y1", "mean_intensity_channel_1",
    "mean_intensity_channel_2", "class"]
FIELD_NAMES = {"labels": "field_01_labels.png",
               "outlines": "field_01_outlines.png",
               "objects": "field_01_objects.csv",
               "image": "field_01_image.png"}
P = "{property}"


def parse(text: str):
    return ElementTree.fromstring(text.lstrip("﻿").encode("utf-8"))


def load(name: str):
    return parse((DATA / name).read_text(encoding="utf-8-sig"))


def generated(columns=FIELD_COLUMNS, width=400, height=300, stem="field_01"):
    names = {key: value.replace("field_01", stem)
             for key, value in FIELD_NAMES.items()}
    return engine._vvvv_project_document(stem, width, height, names, columns)


def decode_id(text: str) -> bytes:
    """vvvv's id codec: two 64-bit halves, 11 base-62 digits each."""
    halves = []
    for part in (text[:11], text[11:]):
        value = 0
        for char in part:
            value = value * 62 + DIGITS.index(char)
        assert value < 2 ** 64, text
        halves.append(value.to_bytes(8, "big"))
    return b"".join(halves)


def ids(root):
    return [element.get("Id") for element in root.iter() if element.get("Id")]


def node_name(node):
    ref = node.find(P + "NodeReference")
    choices = [choice.get("Name") for choice in ref.findall("Choice")]
    return choices[-1]


def links(root):
    return [element.get("Ids").split(",") for element in root.iter("Link")]


def structure(root):
    """Everything but layout: tags, attributes and text, in order."""
    rows = []
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        attributes = {key: value for key, value in element.attrib.items()
                      if key != "Bounds"}
        rows.append((element.tag, tuple(sorted(attributes.items())),
                     (element.text or "").strip()))
    return rows


@pytest.mark.parametrize("name", OFFICIAL + (SAVED,))
def test_id_codec_reads_every_id_vvvv_wrote(name):
    for text in ids(load(name)):
        raw = decode_id(text)
        assert raw[7] >> 4 == 4 and raw[8] >> 6 == 2, text


def test_generated_ids_are_unique_version_4_guids():
    root = parse(generated())
    found = ids(root)
    assert len(found) == len(set(found)) > 300
    for text in found:
        assert re.fullmatch(r"[A-Za-z0-9]{22}", text)
        raw = decode_id(text)
        assert raw[7] >> 4 == 4 and raw[8] >> 6 == 2


def test_document_matches_what_vvvv_saved():
    """vvvv 7.4.1 opened the generated project and saved it unchanged."""
    ours = parse(generated())
    saved = load(SAVED)
    assert structure(ours) == structure(saved)


def test_document_shape_follows_official_examples():
    ours = parse(generated())
    for name in OFFICIAL:
        official = load(name)
        assert ours.tag == official.tag == "Document"
        assert ours.get("Version") == official.get("Version") == "0.128"
        app = [node for node in ours.iter("Node")
               if node.get("Name") == "Application"]
        reference = [node for node in official.iter("Node")
                     if node.get("Name") == "Application"]
        assert len(app) == len(reference) == 1
        choice = app[0].find(P + "NodeReference/Choice")
        expected = reference[0].find(P + "NodeReference/Choice")
        assert choice.attrib == expected.attrib
        for root in (ours, official):
            patch = [node for node in root.iter("Node")
                     if node.get("Name") == "Application"][0].find("Patch")
            parts = {child.get("Name"): child.get("Id")
                     for child in patch.findall("Patch")}
            fragments = {child.get("Patch") for child in
                         patch.find("ProcessDefinition").findall("Fragment")}
            assert fragments == {parts["Create"], parts["Update"]}
    locations = {dep.get("Location") for dep in ours.iter("NugetDependency")}
    assert {"VL.CoreLib", "VL.Skia"} <= locations
    assert ours.get("LanguageVersion") == engine._VVVV_LANGUAGE_VERSION


def test_nodes_and_pins_exist_in_vvvv_documents():
    """Every node our patch uses appears, with the same pins, in a file
    vvvv wrote; the official examples confirm the shared ones."""
    ours = parse(generated(columns=FIELD_COLUMNS[:-1]))
    saved = load(SAVED)
    known = {}
    for root in [saved] + [load(name) for name in OFFICIAL]:
        for node in root.iter("Node"):
            if node.find(P + "NodeReference") is None or \
                    node.get("Name") == "Application":
                continue
            known.setdefault(node_name(node), set()).update(
                pin.get("Name") for pin in node.findall("Pin"))
    for node in ours.iter("Node"):
        if node.get("Name") == "Application":
            continue
        name = node_name(node)
        assert name in known, name
        assert {pin.get("Name") for pin in node.findall("Pin")} <= known[name]
    official = Counter()
    for name in OFFICIAL:
        official.update(node_name(node) for node in load(name).iter("Node")
                        if node.find(P + "NodeReference") is not None)
    assert {"Renderer", "Circle", "Text", "FontAndParagraph"} <= set(official)


def test_links_join_existing_pins_and_pads():
    root = parse(generated())
    endpoints = set(ids(root))
    outputs = {pin.get("Id") for pin in root.iter("Pin")
               if pin.get("Kind") in ("OutputPin", "StateOutputPin")}
    inputs = {pin.get("Id") for pin in root.iter("Pin")
              if pin.get("Kind") in ("InputPin", "StateInputPin")}
    pads = {pad.get("Id") for pad in root.iter("Pad")}
    points = {point.get("Id") for point in root.iter("ControlPoint")}
    sinks = Counter()
    for source, sink in links(root):
        assert source in endpoints and sink in endpoints
        assert source in outputs | pads | points
        assert sink in inputs | points
        sinks[sink] += 1
    assert max(sinks.values()) == 1


def test_paths_are_relative_and_name_the_exported_files():
    root = parse(generated())
    paths = sorted(pad.get("Value") for pad in root.iter("Pad")
                   if pad.find(P + "TypeAnnotation/Choice").get("Name") == "Path")
    assert paths == sorted([FIELD_NAMES["image"], FIELD_NAMES["outlines"],
                            FIELD_NAMES["objects"], "manifest.json"])
    assert not any(re.match(r"^([A-Za-z]:|/|\\)", path) for path in paths)


def test_table_columns_drive_the_slice_indices():
    root = parse(generated())
    region = [node for node in root.iter("Node")
              if node.find(P + "NodeReference") is not None
              and node_name(node) == "ForEach"][0]
    indices = sorted(int(pad.get("Value")) for pad in region.iter("Pad")
                     if pad.get("Comment") == "Index")
    assert indices == [0, 1, 2, FIELD_COLUMNS.index("class")]
    plain = parse(generated(columns=FIELD_COLUMNS[:-1]))
    names = Counter(node_name(node) for node in plain.iter("Node")
                    if node.find(P + "NodeReference") is not None)
    assert names["Text"] == 1 and names["GetSlice"] == 3


def test_fit_maps_image_pixels_to_the_normalized_window():
    root = parse(generated(width=400, height=300))
    values = {pad.get("Comment"): pad.get("Value") for pad in root.iter("Pad")}
    sx, sy = (float(v) for v in values["Scaling"].split(","))
    tx, ty = (float(v) for v in values["Translation"].split(","))
    for x, y, nx, ny in ((0, 0, -4 / 3, -1), (400, 300, 4 / 3, 1),
                         (200, 150, 0, 0)):
        assert x * sx + tx == pytest.approx(nx, abs=1e-6)
        assert y * sy + ty == pytest.approx(ny, abs=1e-6)


def test_same_field_writes_the_same_document():
    assert generated() == generated()
    assert generated() != generated(stem="other")


def test_export_writes_project_readme_and_display_image(tmp_path):
    labels = np.zeros((30, 40), dtype=np.uint16)
    labels[5:15, 5:15] = 1
    labels[18:28, 20:35] = 2
    image = np.zeros((30, 40, 2), dtype=np.uint16)
    image[..., 0] = np.arange(40, dtype=np.uint16) * 100
    image[..., 1] = 7
    folder = Path(engine._export_vvvv(tmp_path, "well_A1.tif", labels, image,
                                     channel_names=["dapi", "gfp"],
                                     classes={1: "a", 2: "b"}))
    project = folder / "well_A1.vl"
    root = parse(project.read_text(encoding="utf-8-sig"))
    assert project.read_bytes().startswith(b"\xef\xbb\xbf<?xml")
    paths = {pad.get("Value") for pad in root.iter("Pad")
             if pad.find(P + "TypeAnnotation/Choice").get("Name") == "Path"}
    assert paths <= {p.name for p in folder.iterdir()}
    header = tabular.read_table(folder / "well_A1_objects.csv",
                                canonicalise=False).columns
    region = [node for node in root.iter("Node")
              if node.find(P + "NodeReference") is not None
              and node_name(node) == "ForEach"][0]
    class_index = max(int(pad.get("Value")) for pad in region.iter("Pad")
                      if pad.get("Comment") == "Index")
    assert header[class_index] == "class"
    display = imageio.imread(folder / "well_A1_image.png")
    assert display.shape == (30, 40, 3) and display.dtype == np.uint8
    assert display[..., 0].max() == 255 and display[..., 2].max() == 0
    readme = (folder / "README.txt").read_text(encoding="utf-8")
    assert "well_A1.vl" in readme and engine._VVVV_GAMMA_VERSION in readme
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    assert manifest["files"]["project"] == "well_A1.vl"
    assert manifest["files"]["image"] == "well_A1_image.png"
    assert manifest["vvvv_gamma"] == engine._VVVV_GAMMA_VERSION
    first_id = manifest["export_id"]
    stamp = project.stat().st_mtime_ns
    labels[labels == 2] = 0
    engine._export_vvvv(tmp_path, "well_A1.tif", labels, image,
                       channel_names=["dapi", "gfp"], classes={1: "a"})
    assert project.stat().st_mtime_ns == stamp
    again = json.loads((folder / "manifest.json").read_text("utf-8"))
    assert again["export_id"] != first_id and again["objects"] == 1


def test_export_without_image_still_writes_a_project(tmp_path):
    folder = Path(engine._export_vvvv(tmp_path, "bare.png",
                                     np.zeros((8, 8), dtype=np.uint16)))
    assert not (folder / "bare_image.png").exists()
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    assert "image" not in manifest["files"]
    parse((folder / "bare.vl").read_text(encoding="utf-8-sig"))


def test_publish_waits_out_a_reader_holding_the_file(tmp_path, monkeypatch):
    calls = []
    real = tabular._publish

    def flaky(target, write):
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError("in use")
        return real(target, write)

    monkeypatch.setattr(tabular, "_publish", flaky)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    publish = engine._vvvv_publisher(tabular)
    target = tmp_path / "x.txt"
    publish(str(target), lambda pending: Path(pending).write_text("ok"))
    assert target.read_text() == "ok" and len(calls) == 3
