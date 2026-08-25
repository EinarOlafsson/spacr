"""The napari bridge with the pieces it usually has taken away.

napari is an optional extra, so the module's honest paths are the ones where
it is missing, where the image beside the mask is in a format that is not a
TIFF, and where the layer handed back does not announce what it is. Each of
those is driven here, and the round trip is asserted on the array rather than
on a call.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from spacr import napari_bridge as nb


class FakeLayer:
    """A layer that names its own kind, the way napari's layers do."""

    def __init__(self, data, name, kind):
        self.data = np.asarray(data)
        self.name = name
        self._type_string = kind


class FakeViewer:
    """Anything with ``add_image`` and ``add_labels`` will do."""

    def __init__(self, title=""):
        self.title = title
        self.layers = []
        self.ran = False

    def add_image(self, *, data, name, **rest):
        return self._add(data, name, "image")

    def add_labels(self, *, data, name, **rest):
        return self._add(data, name, "labels")

    def _add(self, data, name, kind):
        layer = FakeLayer(data, name, kind)
        self.layers.append(layer)
        return layer


@pytest.fixture
def mask_file(tmp_path):
    """A small labelled mask on disk, through spaCR's own writer."""
    from spacr.mask_io import save_mask

    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[1:4, 1:4] = 1
    mask[5:7, 5:7] = 2
    return str(save_mask(str(tmp_path / "field.tif"), mask))


@pytest.fixture
def fake_napari(monkeypatch):
    """Stand napari up at its import seam, so the two functions that need it run.

    The extra is not installed in this environment, and those two functions
    are the only ones in the module that touch it.
    """
    module = types.ModuleType("napari")
    made = {}

    class Viewer(FakeViewer):
        def __init__(self, title=""):
            super().__init__(title=title)
            made["viewer"] = self

    def run():
        made["ran"] = True

    module.Viewer = Viewer
    module.run = run
    monkeypatch.setitem(sys.modules, "napari", module)
    return made


# ---------------------------------------------------------------------------
# the optional extra
# ---------------------------------------------------------------------------

def test_requiring_napari_returns_the_module_when_it_is_there(fake_napari):
    """The one import site hands the module back to its caller."""
    assert nb.require_napari() is sys.modules["napari"]


def test_the_missing_message_names_the_module_that_was_actually_missing():
    """A transitive failure should not blame napari itself."""
    text = nb.missing_napari_message("qtpy")

    assert "qtpy" in text
    assert nb.missing_napari_message("") == nb.missing_napari_message("napari")


def test_availability_is_answered_without_importing_anything():
    """The greying-out question never raises, whatever the answer is."""
    from importlib.util import find_spec

    assert nb.napari_available() is (find_spec("napari") is not None)


# ---------------------------------------------------------------------------
# reading the field
# ---------------------------------------------------------------------------

def test_a_file_that_is_not_there_is_a_file_not_found(tmp_path):
    """The error names the path rather than failing inside a decoder."""
    with pytest.raises(FileNotFoundError) as caught:
        nb.read_image(tmp_path / "nothing.tif")

    assert "nothing.tif" in str(caught.value)


def test_a_directory_is_not_an_image(tmp_path):
    """``isfile`` is the test, so a folder refuses the same way."""
    with pytest.raises(FileNotFoundError):
        nb.read_image(tmp_path)


def test_a_numpy_field_is_read_by_numpy(tmp_path):
    """``.npy`` keeps its dtype and its values exactly."""
    array = (np.arange(64, dtype=np.uint16) * 500).reshape(8, 8)
    path = tmp_path / "field.npy"
    np.save(path, array)

    got = nb.read_image(path)

    assert got.dtype == np.uint16
    assert np.array_equal(got, array)


def test_a_png_field_is_read_through_pillow(tmp_path):
    """Anything else goes through Pillow, unrescaled and unreordered."""
    from PIL import Image

    array = np.arange(64, dtype=np.uint8).reshape(8, 8)
    path = tmp_path / "field.png"
    Image.fromarray(array).save(path)

    got = nb.read_image(path)

    assert got.shape == (8, 8)
    assert np.array_equal(got, array)


def test_a_tiff_field_keeps_its_bit_depth(tmp_path):
    """A 16-bit field must not come back as eight."""
    import tifffile

    array = (np.arange(64, dtype=np.uint16) * 1000).reshape(8, 8)
    path = tmp_path / "field.tif"
    tifffile.imwrite(str(path), array)

    got = nb.read_image(path)

    assert got.dtype == np.uint16
    assert np.array_equal(got, array)


# ---------------------------------------------------------------------------
# opening it
# ---------------------------------------------------------------------------

def test_opening_without_a_viewer_makes_one_titled_after_the_mask(
        fake_napari, mask_file):
    """The window title is the field's filename, not a generic one."""
    handoff = nb.load_handoff(mask_file)

    viewer = nb.open_in_napari(handoff)

    assert viewer is fake_napari["viewer"]
    assert viewer.title == "field.tif"
    assert [layer.name for layer in viewer.layers] == [handoff.name]


def test_an_explicit_title_wins(fake_napari, mask_file):
    """A caller that names the window gets the name it asked for."""
    viewer = nb.open_in_napari(nb.load_handoff(mask_file),
                               title="correct this one")

    assert viewer.title == "correct this one"


def test_an_in_memory_mask_still_gets_a_window_title(fake_napari):
    """With no path there is still a name on the window."""
    handoff = nb.MaskHandoff(mask=np.zeros((4, 4), dtype=np.uint16))

    viewer = nb.open_in_napari(handoff)

    assert viewer.title == "spaCR mask"


def test_the_event_loop_is_napari_s_own(fake_napari):
    """``run_event_loop`` is one call, and it is napari's."""
    nb.run_event_loop()

    assert fake_napari["ran"] is True


def test_a_blocking_correction_runs_the_loop_before_taking_the_mask_back(
        fake_napari, mask_file):
    """``block=True`` is "correct it, then take it back", in that order."""
    order = []
    module = sys.modules["napari"]
    real_run = module.run

    def note_run():
        order.append("ran")
        real_run()

    module.run = note_run

    class Painting(FakeViewer):
        def add_labels(self, *, data, name, **rest):
            layer = super().add_labels(data=data, name=name, **rest)
            order.append("painted")
            layer.data[1, 1] = 7
            return layer

    result = nb.correct_mask(mask_file, viewer=Painting(), block=True)

    assert order == ["painted", "ran"]
    assert result.changed_pixels == 1
    assert result.written is True


# ---------------------------------------------------------------------------
# recognising a labels layer
# ---------------------------------------------------------------------------

def test_a_layer_that_says_what_it_is_is_believed():
    """napari's own layers carry ``_type_string``, so it is asked first."""
    assert nb._is_labels_layer(FakeLayer(np.zeros((4, 4), int), "m", "labels"))
    assert not nb._is_labels_layer(
        FakeLayer(np.zeros((4, 4), int), "i", "image"))


def test_a_layer_named_after_labels_is_taken_at_its_word():
    """The class name is the fallback when nothing declares the type."""
    class SomethingLabelsLayer:
        data = np.zeros((4, 4), dtype=np.uint16)

    assert nb._is_labels_layer(SomethingLabelsLayer())


def test_a_layer_named_after_an_image_is_refused():
    """A 16-bit image is integer too, so the name settles it first."""
    class SomeImageLayer:
        data = np.zeros((4, 4), dtype=np.uint16)

    assert not nb._is_labels_layer(SomeImageLayer())


def test_a_nameless_layer_falls_back_to_the_shape_of_its_array():
    """Last resort: a 2- or 3-D integer array is mask-shaped."""
    class Anonymous:
        def __init__(self, data):
            self.data = data

    assert nb._is_labels_layer(Anonymous(np.zeros((4, 4), dtype=np.uint16)))
    assert nb._is_labels_layer(Anonymous(np.zeros((2, 4, 4), dtype=np.int32)))
    assert not nb._is_labels_layer(Anonymous(np.zeros((4, 4), dtype=float)))
    assert not nb._is_labels_layer(
        Anonymous(np.zeros((4, 4, 3, 2), dtype=np.uint16)))


def test_a_layer_with_no_array_at_all_is_not_a_mask():
    """Nothing to look at means the answer is no, not an exception."""
    class Empty:
        data = None

    assert not nb._is_labels_layer(Empty())


# ---------------------------------------------------------------------------
# describing the correction
# ---------------------------------------------------------------------------

def test_the_description_counts_removals_and_reshapes_separately(mask_file):
    """Each kind of edit is named, so a status bar is worth reading."""
    from spacr.mask_io import load_mask

    before = load_mask(mask_file)
    after = np.array(before, copy=True)
    after[after == 2] = 0                    # one object removed
    after[1, 3] = 0                          # one object reshaped
    after[7, 0] = 3                          # one object added

    result = nb.write_back(mask_file, after, original=before)

    text = result.describe()
    assert "1 object(s) added" in text
    assert "1 removed" in text
    assert "1 reshaped" in text
    assert "field.tif" in text
    assert result.touched == (1, 2, 3)
    assert bool(result) is True


def test_an_unchanged_round_trip_describes_itself_as_such(mask_file):
    """Nothing written is stated, not implied by an empty line."""
    from spacr.mask_io import load_mask

    before = load_mask(mask_file)

    result = nb.write_back(mask_file, before, original=before)

    assert result.describe() == ("The mask came back unchanged; nothing was "
                                 "written.")
    assert bool(result) is False
