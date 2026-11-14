"""Table-driven tests of ``spacr.napari_bridge``: reading a field, naming the window, spotting labels.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: :func:`spacr.napari_bridge.read_image` (each format is
read exactly as stored; nothing there is a ``FileNotFoundError``),
:func:`spacr.napari_bridge.open_in_napari` (window title and layers, with
napari stood up at its import seam) and
:func:`spacr.napari_bridge._is_labels_layer` (the layer's own answer, then
its class name, then its array).  napari itself is never imported.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from spacr import napari_bridge as nb


class _Layer:
    """What a duck-typed viewer hands back from ``add_*``."""

    def __init__(self, data, name, kind, rest):
        self.data = np.asarray(data)
        self.name = name
        self._type_string = kind
        self.rest = rest


class _Viewer:
    """Anything with ``add_image`` and ``add_labels`` is a viewer to the bridge."""

    def __init__(self, title=""):
        self.title = title
        self.layers = []

    def add_image(self, *, data, name, **rest):
        return self._add(data, name, "image", rest)

    def add_labels(self, *, data, name, **rest):
        return self._add(data, name, "labels", rest)

    def _add(self, data, name, kind, rest):
        layer = _Layer(data, name, kind, rest)
        self.layers.append(layer)
        return layer


def test_read_image_returns_each_format_exactly_as_stored(subtests, tmp_path):
    """TIFF keeps its bit depth, .npy its dtype, Pillow formats their values; absences raise."""
    import tifffile
    from PIL import Image

    u16 = (np.arange(64, dtype=np.uint16) * 1000).reshape(8, 8)
    u8 = np.arange(64, dtype=np.uint8).reshape(8, 8)
    f32 = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    tifffile.imwrite(str(tmp_path / "field.tif"), u16)
    tifffile.imwrite(str(tmp_path / "upper.TIFF"), u16[:4])
    np.save(tmp_path / "field.npy", (u16 // 2).astype(np.uint16))
    np.save(tmp_path / "float.npy", f32)
    Image.fromarray(u8).save(tmp_path / "field.png")
    cases = [
        # (case id, path, expected array)
        ("16-bit tiff", tmp_path / "field.tif", u16),
        ("upper-case .TIFF suffix", str(tmp_path / "upper.TIFF"), u16[:4]),
        ("uint16 npy", tmp_path / "field.npy", (u16 // 2).astype(np.uint16)),
        ("float npy", tmp_path / "float.npy", f32),
        ("8-bit png through Pillow", tmp_path / "field.png", u8),
    ]
    for case_id, path, expected in cases:
        with subtests.test(msg=case_id):
            got = nb.read_image(path)
            assert got.dtype == expected.dtype
            assert got.shape == expected.shape
            np.testing.assert_array_equal(got, expected)
    for case_id, path in [("missing file", tmp_path / "nothing.tif"),
                          ("a directory", tmp_path)]:
        with subtests.test(msg=case_id):
            with pytest.raises(FileNotFoundError) as caught:
                nb.read_image(path)
            assert str(caught.value) == str(path)


@pytest.fixture
def fake_napari(monkeypatch):
    """Stand napari up at its import seam; ``made`` records each viewer it constructs."""
    module = types.ModuleType("napari")
    made = []

    class Viewer(_Viewer):
        def __init__(self, title=""):
            super().__init__(title=title)
            made.append(self)

    module.Viewer = Viewer
    module.run = lambda: None
    monkeypatch.setitem(sys.modules, "napari", module)
    return made


def test_open_in_napari_titles_the_window_and_adds_the_layers(subtests, tmp_path,
                                                             fake_napari, monkeypatch):
    """A new viewer is titled by the caller, the mask file, or 'spaCR mask'; a given one is reused."""
    from spacr.mask_io import save_mask

    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[1:4, 1:4] = 1
    mask[5:7, 5:7] = 2
    mask_file = str(save_mask(str(tmp_path / "field.tif"), mask))
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)
    from_disk = nb.load_handoff(mask_file)
    in_memory = nb.MaskHandoff(mask=np.zeros((4, 4), dtype=np.uint16))
    with_image = nb.MaskHandoff(mask=mask, mask_path=mask_file, image=image,
                                name="cells", scale=(0.5, 0.5))
    mine = _Viewer(title="already open")
    cases = [
        # (case id, handoff, viewer, title, expected title, expected layers, new viewer)
        ("titled after the mask file", from_disk, None, "", "field.tif",
         [("labels", nb.LABELS_LAYER_NAME)], True),
        ("an explicit title wins", from_disk, None, "correct this one",
         "correct this one", [("labels", nb.LABELS_LAYER_NAME)], True),
        ("an in-memory mask still gets a title", in_memory, None, "", "spaCR mask",
         [("labels", nb.LABELS_LAYER_NAME)], True),
        ("image under the labels, at its scale", with_image, None, "", "field.tif",
         [("image", nb.IMAGE_LAYER_NAME), ("labels", "cells")], True),
        ("a given viewer is added to, not retitled", from_disk, mine, "ignored",
         "already open", [("labels", nb.LABELS_LAYER_NAME)], False),
    ]
    for case_id, handoff, viewer, title, want_title, want_layers, new in cases:
        with subtests.test(msg=case_id):
            fake_napari.clear()
            mine.layers.clear()
            got = nb.open_in_napari(handoff, viewer=viewer, title=title)
            assert fake_napari == ([got] if new else [])
            if not new:
                assert got is mine
            assert got.title == want_title
            assert [(layer._type_string, layer.name) for layer in got.layers] == want_layers
            labels = got.layers[-1]
            np.testing.assert_array_equal(labels.data, handoff.mask)
            assert labels.data is not handoff.mask
            assert labels.rest["opacity"] == 0.6
            scale = [0.5, 0.5] if handoff.scale else None
            assert all(layer.rest.get("scale") == scale for layer in got.layers)
            if handoff.image is not None:
                np.testing.assert_array_equal(got.layers[0].data, image)

    with subtests.test(msg="without napari and without a viewer it is refused"):
        monkeypatch.setitem(sys.modules, "napari", None)
        with pytest.raises(nb.NapariExtraMissing) as caught:
            nb.open_in_napari(in_memory)
        assert str(caught.value) == nb.missing_napari_message("napari")


def test_is_labels_layer_asks_the_layer_then_its_name_then_its_array(subtests):
    """``_type_string`` wins, then 'label'/'image' in the class name, then a 2-/3-D integer array."""
    def anonymous(data):
        return type("Anonymous", (), {"data": data})()

    def named(class_name, data, **attributes):
        return type(class_name, (), {"data": data, **attributes})()

    ints = np.zeros((4, 4), dtype=np.uint16)
    cases = [
        # (case id, layer, is a labels layer)
        ("declares labels", named("Thing", ints, _type_string="labels"), True),
        ("declares labels, any case and padding",
         named("Thing", ints, _type_string="  Labels "), True),
        ("declares image although its class says labels",
         named("LabelsLayer", ints, _type_string="image"), False),
        ("declares points", named("Thing", ints, _type_string="points"), False),
        ("empty declaration falls through to the name",
         named("SomethingLabelsLayer", ints, _type_string=""), True),
        ("class named after labels", named("SomethingLabelsLayer", ints), True),
        ("class named after labels holding floats",
         named("LabelLayer", np.zeros((4, 4))), True),
        ("class named after an image holding integers", named("SomeImageLayer", ints),
         False),
        ("anonymous 2-D integer array", anonymous(ints), True),
        ("anonymous 3-D integer array", anonymous(np.zeros((2, 4, 4), np.int32)), True),
        ("anonymous list of ints", anonymous([[1, 0], [0, 2]]), True),
        ("anonymous float array", anonymous(np.zeros((4, 4))), False),
        ("anonymous 1-D integer array", anonymous(np.zeros(4, np.int32)), False),
        ("anonymous 4-D integer array", anonymous(np.zeros((4, 4, 3, 2), np.uint16)),
         False),
        ("no array at all", anonymous(None), False),
        ("no data attribute", type("Bare", (), {})(), False),
    ]
    for case_id, layer, expected in cases:
        with subtests.test(msg=case_id):
            assert nb._is_labels_layer(layer) is expected
