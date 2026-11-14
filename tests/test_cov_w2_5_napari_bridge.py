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

# ---------------------------------------------------------------------------
# opening it
# ---------------------------------------------------------------------------

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
