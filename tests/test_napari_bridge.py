"""``A18`` — the napari round trip, asserted with no napari installed.

That is not a compromise: it is the design. Everything either side of napari
— what is handed over, what is accepted back, what is written, what is
recorded — is a function over plain arrays and a duck-typed viewer, so the
part that can lose data is exercised for real, deterministically, in a
process that never starts a second Qt stack. Only three functions actually
need napari (:func:`~spacr.napari_bridge.require_napari`,
:func:`~spacr.napari_bridge.open_in_napari`,
:func:`~spacr.napari_bridge.run_event_loop`), and each is tested through its
missing-dependency path, which is the path most users will meet.

Round-trip fidelity is what this file is mostly about:

* a hand-edited label comes back with **the same value on the same pixels**;
* the array is **not transposed** — the test mask is deliberately not square,
  because on a square one a stray ``.T`` is invisible;
* a label too large for ``uint16`` is **refused**, not wrapped. ``uint16``
  silently turns 70000 into 4464, which would rename a cell.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from spacr import napari_bridge as nb
from spacr.curation import CurationLog, is_curated
from spacr.mask_io import load_mask, save_mask


# ---------------------------------------------------------------------------
# A field, and a viewer shaped like napari's
# ---------------------------------------------------------------------------

def _mask() -> np.ndarray:
    """A 7x11 mask — NOT square, so a stray transpose cannot hide."""
    mask = np.zeros((7, 11), dtype=np.uint16)
    mask[1:3, 2:5] = 41           # a blocky object with a memorable label
    mask[5, 8:10] = 900           # a second one, far away
    mask[0, 10] = 7               # a corner pixel: catches a flip
    return mask


def _field(tmp_path, mask=None):
    """Write a mask and an image, and return their paths."""
    mask = _mask() if mask is None else mask
    mask_path = str(tmp_path / "plate1_A01_f1_mask.tif")
    save_mask(mask_path, mask)
    image = (np.arange(mask.size, dtype=np.uint16)
             .reshape(mask.shape) * 3)
    image_path = str(tmp_path / "plate1_A01_f1.tif")
    save_mask(image_path, image)
    return mask_path, image_path


class FakeLayer:
    """A napari layer, as far as this bridge is concerned.

    ``_type_string`` is not invented for the test: it is the attribute
    napari's own layers carry (``"image"``, ``"labels"``, ``"points"``), and
    it is what :func:`spacr.napari_bridge.labels_from_viewer` asks when it has
    to tell a 16-bit image apart from a label mask — which no heuristic over
    the values can do, since both are 2-D integer arrays.
    """

    def __init__(self, data, name, kind):
        self.data = np.asarray(data)
        self.name = str(name)
        self._type_string = str(kind)


class FakeViewer:
    """A napari viewer, as far as this bridge is concerned.

    Duck-typed rather than mocked: :func:`spacr.napari_bridge.add_to_viewer`
    calls the same ``add_image`` / ``add_labels`` it would call on the real
    thing, with the same keyword arguments, and
    :func:`~spacr.napari_bridge.labels_from_viewer` reads the layer back the
    same way. What is not exercised is napari's own rendering, which is
    napari's problem.

    The one behaviour that had to be faithful is that a layer **keeps the
    array it was handed and is painted in place**, because that is how
    napari's brush works — and it is what caught the bridge handing the
    layer spaCR's own copy of the mask, which would have made the "before"
    get painted too and every diff come out zero.
    """

    def __init__(self):
        self.layers = []
        self.closed = False

    def add_image(self, **kwargs):
        return self._add(kwargs, "image")

    def add_labels(self, **kwargs):
        return self._add(kwargs, "labels")

    def _add(self, kwargs, kind):
        layer = FakeLayer(kwargs["data"], kwargs["name"], kind)
        self.layers.append(layer)
        return layer

    def close(self):
        self.closed = True


def _opened(tmp_path, mask=None):
    """A field handed to a fake viewer. Returns (handoff, viewer, paths)."""
    mask_path, image_path = _field(tmp_path, mask)
    handoff = nb.load_handoff(mask_path, image_path)
    viewer = FakeViewer()
    nb.add_to_viewer(viewer, handoff)
    return handoff, viewer, (mask_path, image_path)


# ---------------------------------------------------------------------------
# Handing the field over
# ---------------------------------------------------------------------------

def test_the_mask_is_read_with_spacrs_own_reader(tmp_path):
    mask_path, image_path = _field(tmp_path)
    handoff = nb.load_handoff(mask_path, image_path)
    assert handoff.mask.dtype == np.uint16
    assert np.array_equal(handoff.mask, load_mask(mask_path))
    assert handoff.labels == (7, 41, 900)
    assert handoff.image is not None and handoff.image.shape == (7, 11)


def test_a_field_can_be_handed_over_with_no_image(tmp_path):
    mask_path, _ = _field(tmp_path)
    handoff = nb.load_handoff(mask_path)
    assert handoff.image is None
    assert [spec["kind"] for spec in nb.layer_specs(handoff)] == ["labels"]


def test_the_layer_specs_hand_over_the_array_unchanged(tmp_path):
    """No reordering on the way out, and the names the way back looks for."""
    mask_path, image_path = _field(tmp_path)
    handoff = nb.load_handoff(mask_path, image_path)
    image_spec, labels_spec = nb.layer_specs(handoff)
    assert image_spec["kind"] == "image"
    assert image_spec["name"] == nb.IMAGE_LAYER_NAME
    assert labels_spec["kind"] == "labels"
    assert labels_spec["name"] == nb.LABELS_LAYER_NAME
    assert np.array_equal(labels_spec["data"], handoff.mask)
    assert labels_spec["data"].shape == (7, 11)


def test_a_scale_is_passed_through_when_the_field_is_calibrated(tmp_path):
    mask_path, _ = _field(tmp_path)
    handoff = nb.load_handoff(mask_path, scale=(0.65, 0.65))
    assert nb.layer_specs(handoff)[0]["scale"] == [0.65, 0.65]


def test_add_to_viewer_makes_one_layer_per_spec(tmp_path):
    handoff, viewer, _ = _opened(tmp_path)
    assert [layer.name for layer in viewer.layers] == ["image", "mask"]
    assert np.array_equal(viewer.layers[1].data, handoff.mask)


def test_the_handoff_describes_itself_for_a_status_bar(tmp_path):
    mask_path, _ = _field(tmp_path)
    text = nb.load_handoff(mask_path).describe()
    assert "3 object(s)" in text and "7x11" in text


# ---------------------------------------------------------------------------
# ROUND-TRIP FIDELITY — the whole feature
# ---------------------------------------------------------------------------

def test_a_hand_edited_label_comes_back_with_the_same_value_on_the_same_pixels(tmp_path):
    """The assertion the feature exists to satisfy.

    Object 41 is extended by one pixel in "napari". What comes back — and
    what lands on disk — is 41 on exactly that pixel, with every other pixel
    of the field untouched.
    """
    handoff, viewer, (mask_path, _) = _opened(tmp_path)
    before = handoff.mask.copy()

    viewer.layers[1].data[3, 2] = 41          # the hand edit
    corrected = nb.labels_from_viewer(viewer)

    assert corrected.dtype == np.uint16
    assert corrected[3, 2] == 41
    expected = before.copy()
    expected[3, 2] = 41
    assert np.array_equal(corrected, expected)

    result = nb.write_back(mask_path, corrected, original=before)
    on_disk = load_mask(mask_path)
    assert np.array_equal(on_disk, expected)
    assert on_disk.dtype == np.uint16
    # Every other label is exactly where it was.
    assert on_disk[5, 8] == 900 and on_disk[0, 10] == 7
    assert result.changed_pixels == 1
    assert result.altered == (41,) and result.added == () and result.removed == ()


def test_painting_in_napari_does_not_paint_over_spacrs_own_before(tmp_path):
    """A real bug this suite caught, pinned so it cannot come back.

    napari's brush edits the layer's array IN PLACE. Handing the layer
    ``handoff.mask`` itself therefore meant the "before" spaCR holds got
    painted along with the copy the user was editing — so the diff in
    :func:`spacr.napari_bridge.write_back` came out zero for every
    correction, silently, and nothing was ever written or recorded.
    """
    handoff, viewer, (mask_path, _) = _opened(tmp_path)
    assert viewer.layers[1].data is not handoff.mask

    viewer.layers[1].data[3, 2] = 41
    assert handoff.mask[3, 2] == 0, (
        "the handoff's mask was painted on; it is the 'before' the diff is "
        "taken against and must survive the editing session untouched")
    assert np.array_equal(handoff.mask, load_mask(mask_path))


def test_the_mask_is_not_transposed_anywhere_in_the_round_trip(tmp_path):
    """A non-square field, carried out and back with nothing reordered.

    napari's 2-D array axes are ``(row, column)``, numpy's order and spaCR's
    order, so the correct amount of transposing is none. This is the test
    that fails the day somebody "fixes" an orientation they saw in a viewer.
    """
    handoff, viewer, (mask_path, _) = _opened(tmp_path)
    assert handoff.mask.shape == (7, 11)
    assert viewer.layers[1].data.shape == (7, 11)
    corrected = nb.labels_from_viewer(viewer)
    assert corrected.shape == (7, 11)
    assert np.array_equal(corrected, handoff.mask)
    # And the asymmetric corner survives, which a flip would move.
    assert corrected[0, 10] == 7 and corrected[0, 0] == 0


def test_every_label_value_survives_the_round_trip_exactly(tmp_path):
    """No renumbering, no relabelling by connectivity, no gaps closed."""
    mask = np.zeros((5, 9), dtype=np.uint16)
    mask[0, 0] = 3
    mask[2, 2] = 60000            # near the top of uint16
    mask[4, 8] = 12345
    handoff, viewer, (mask_path, _) = _opened(tmp_path, mask)
    corrected = nb.labels_from_viewer(viewer)
    assert sorted(int(v) for v in np.unique(corrected) if v) == [
        3, 12345, 60000]
    assert np.array_equal(corrected, mask)


def test_a_deleted_object_comes_back_as_a_deletion(tmp_path):
    handoff, viewer, (mask_path, _) = _opened(tmp_path)
    viewer.layers[1].data[viewer.layers[1].data == 900] = 0
    result = nb.write_back(mask_path, nb.labels_from_viewer(viewer),
                           original=handoff.mask)
    assert result.removed == (900,)
    assert 900 not in load_mask(mask_path)


def test_a_new_object_comes_back_as_an_addition(tmp_path):
    handoff, viewer, (mask_path, _) = _opened(tmp_path)
    viewer.layers[1].data[6, 0:3] = 55
    result = nb.write_back(mask_path, nb.labels_from_viewer(viewer),
                           original=handoff.mask)
    assert result.added == (55,)
    assert result.changed_pixels == 3
    assert set(result.touched) == {55}


def test_a_3d_mask_round_trips_too(tmp_path):
    mask = np.zeros((3, 4, 6), dtype=np.uint16)
    mask[1, 2, 3] = 17
    path = str(tmp_path / "zstack_mask.npy")
    np.save(path, mask)
    handoff = nb.load_handoff(path)
    viewer = FakeViewer()
    nb.add_to_viewer(viewer, handoff)
    viewer.layers[0].data[0, 0, 0] = 17
    result = nb.write_back(path, nb.labels_from_viewer(viewer),
                           original=mask)
    back = load_mask(path)
    assert back.shape == (3, 4, 6)
    assert back[1, 2, 3] == 17 and back[0, 0, 0] == 17
    assert result.changed_pixels == 1


# ---------------------------------------------------------------------------
# The refusals — where "doing something sensible" would lose data
# ---------------------------------------------------------------------------

def test_a_label_too_large_for_uint16_is_refused_not_wrapped():
    """``np.uint16(70000)`` is 4464. Silently renaming a cell is the worst
    thing this module could do, so it does not do it."""
    mask = np.zeros((4, 4), dtype=np.int64)
    mask[1, 1] = 70000
    with pytest.raises(nb.MaskFidelityError) as caught:
        nb.to_spacr_mask(mask)
    assert "70000" in str(caught.value)
    assert "4464" in str(caught.value)       # what the wrap would have done


def test_a_label_at_the_uint16_ceiling_is_accepted():
    """The boundary is a refusal one past it, not one before."""
    mask = np.zeros((2, 2), dtype=np.int64)
    mask[0, 0] = nb.MAX_LABEL
    assert nb.to_spacr_mask(mask)[0, 0] == nb.MAX_LABEL


def test_a_negative_label_is_refused():
    mask = np.zeros((3, 3), dtype=np.int32)
    mask[1, 1] = -2
    with pytest.raises(nb.MaskFidelityError, match="negative"):
        nb.to_spacr_mask(mask)


def test_a_float_layer_with_fractional_values_is_refused():
    """That is an image layer; taking it back would ruin the mask."""
    with pytest.raises(nb.MaskFidelityError, match="fractional"):
        nb.to_spacr_mask(np.array([[0.0, 1.5], [2.0, 3.0]]))


def test_whole_numbered_floats_are_accepted():
    """napari hands some layers back as float64 with integral values."""
    out = nb.to_spacr_mask(np.array([[0.0, 4.0], [9.0, 0.0]]))
    assert out.dtype == np.uint16 and out[1, 0] == 9


def test_an_rgb_layer_is_refused_rather_than_squeezed():
    with pytest.raises(nb.MaskFidelityError, match="2-D or 3-D"):
        nb.to_spacr_mask(np.zeros((2, 2, 3, 4), dtype=np.uint8))


def test_a_mask_whose_shape_changed_is_refused(tmp_path):
    """spaCR will not resize a mask to fit."""
    mask_path, _ = _field(tmp_path)
    with pytest.raises(nb.MaskFidelityError, match="will not resize"):
        nb.write_back(mask_path, np.zeros((3, 3), dtype=np.uint16))


def test_taking_back_a_viewer_with_no_such_layer_says_which_are_there(tmp_path):
    handoff, viewer, _ = _opened(tmp_path)
    viewer.layers[1].name = "renamed"
    viewer.layers.append(
        FakeLayer(np.zeros((7, 11), np.uint16), "second", "labels"))
    with pytest.raises(nb.MaskFidelityError) as caught:
        nb.labels_from_viewer(viewer)
    assert "renamed" in str(caught.value) and "second" in str(caught.value)


def test_a_renamed_lone_labels_layer_is_still_found(tmp_path):
    """A user who renamed the layer has not thereby thrown their work away."""
    handoff, viewer, _ = _opened(tmp_path)
    viewer.layers[1].name = "my corrections"
    assert np.array_equal(nb.labels_from_viewer(viewer), handoff.mask)


def test_an_empty_viewer_is_a_refusal_not_a_crash():
    with pytest.raises(nb.MaskFidelityError, match="none"):
        nb.labels_from_viewer(FakeViewer())


# ---------------------------------------------------------------------------
# The correction is recorded, the same way spaCR records its own
# ---------------------------------------------------------------------------

def test_a_correction_is_appended_to_the_curation_ledger(tmp_path):
    handoff, viewer, (mask_path, _) = _opened(tmp_path)
    assert is_curated(mask_path) is False

    viewer.layers[1].data[3, 2] = 41
    result = nb.write_back(mask_path, nb.labels_from_viewer(viewer),
                           original=handoff.mask)

    assert is_curated(mask_path) is True
    assert result.log_path == f"{mask_path}.curation.json"
    log = CurationLog.read_beside(mask_path)
    assert len(log) == 1
    edit = log.edits[0]
    assert edit.kind == nb.EDIT_KIND
    assert edit.n_changed == 1
    assert edit.detail["via"] == "napari"
    assert edit.detail["altered"] == [41]
    assert "curated by hand" in log.describe()


def test_a_ledger_the_brush_already_wrote_is_appended_to_not_replaced(tmp_path):
    """Append-only across tools: the brush's history is not overwritten."""
    handoff, viewer, (mask_path, _) = _opened(tmp_path)
    existing = CurationLog(mask_path, source="spacr-qt curation")
    existing.append("paint", 41, n_changed=12, radius=3.0)
    existing.write_beside(mask_path)

    viewer.layers[1].data[3, 2] = 41
    nb.write_back(mask_path, nb.labels_from_viewer(viewer),
                  original=handoff.mask)

    log = CurationLog.read_beside(mask_path)
    assert [edit.kind for edit in log.edits] == ["paint", nb.EDIT_KIND]
    assert log.source == "spacr-qt curation"


def test_a_round_trip_that_changed_nothing_writes_nothing(tmp_path):
    """The rule ``MaskCuration.end_stroke`` applies to a stroke that missed.

    A ledger padded with no-op entries is one nobody reads, and rewriting the
    file would move its mtime and make every downstream artifact look stale
    for no reason.
    """
    handoff, viewer, (mask_path, _) = _opened(tmp_path)
    before = os.stat(mask_path).st_mtime_ns

    result = nb.write_back(mask_path, nb.labels_from_viewer(viewer),
                           original=handoff.mask)

    assert bool(result) is False
    assert result.written is False and result.log_path == ""
    assert os.stat(mask_path).st_mtime_ns == before
    assert is_curated(mask_path) is False
    assert "unchanged" in result.describe()


def test_a_preview_computes_the_diff_without_writing(tmp_path):
    handoff, viewer, (mask_path, _) = _opened(tmp_path)
    viewer.layers[1].data[3, 2] = 41
    result = nb.write_back(mask_path, nb.labels_from_viewer(viewer),
                           original=handoff.mask, write=False)
    assert result.changed_pixels == 1 and result.written is False
    assert is_curated(mask_path) is False
    assert np.array_equal(load_mask(mask_path), handoff.mask)


def test_the_ledger_goes_beside_the_file_that_was_actually_written(tmp_path):
    """``save_mask`` resolves a bare stem, and the ledger must follow it.

    ``log_path_for`` keys on the full name including the extension, so a
    ledger written for ``foo`` would be a second, orphaned history next to
    the one the brush writes for ``foo.tif``.
    """
    mask = _mask()
    stem = str(tmp_path / "bare_stem")
    save_mask(stem, mask)
    corrected = mask.copy()
    corrected[3, 2] = 41
    result = nb.write_back(stem, corrected)
    assert result.mask_path.endswith(".tif")
    assert result.log_path == f"{stem}.tif.curation.json"
    assert is_curated(f"{stem}.tif") is True


def test_the_diff_is_taken_against_what_is_on_disk_when_not_given(tmp_path):
    mask_path, _ = _field(tmp_path)
    corrected = load_mask(mask_path).copy()
    corrected[6, 6] = 41
    result = nb.write_back(mask_path, corrected)
    assert result.changed_pixels == 1 and result.altered == (41,)


def test_extra_detail_reaches_the_ledger_entry(tmp_path):
    handoff, viewer, (mask_path, _) = _opened(tmp_path)
    viewer.layers[1].data[3, 2] = 41
    nb.write_back(mask_path, nb.labels_from_viewer(viewer),
                  original=handoff.mask, extra={"reviewer": "EO"})
    assert CurationLog.read_beside(mask_path).edits[0].detail["reviewer"] == "EO"


def test_the_result_describes_itself_for_a_status_bar(tmp_path):
    handoff, viewer, (mask_path, _) = _opened(tmp_path)
    viewer.layers[1].data[6, 0:3] = 55
    result = nb.write_back(mask_path, nb.labels_from_viewer(viewer),
                           original=handoff.mask)
    text = result.describe()
    assert "3 pixel(s)" in text and "1 object(s) added" in text
    assert "curation.json" in text


def test_a_handoff_reports_whether_its_mask_was_already_curated(tmp_path):
    mask_path, _ = _field(tmp_path)
    assert nb.load_handoff(mask_path).curated is False
    log = CurationLog(mask_path, source="spacr-qt curation")
    log.append("paint", 41, n_changed=5)
    log.write_beside(mask_path)
    assert nb.load_handoff(mask_path).curated is True
    assert "already curated" in nb.load_handoff(mask_path).describe()


# ---------------------------------------------------------------------------
# napari is optional, and its absence is an instruction
# ---------------------------------------------------------------------------

def test_the_missing_extra_names_the_pip_command_not_a_traceback():
    message = nb.missing_napari_message("napari")
    assert 'pip install "spacr[napari]"' in message
    assert "napari" in message
    # And it says what to do instead, because most people do not need it.
    assert "Curate" in message


@pytest.mark.skipif(nb.napari_available(),
                    reason="napari is installed in this environment")
def test_require_napari_raises_an_import_error_carrying_that_message():
    with pytest.raises(nb.NapariExtraMissing) as caught:
        nb.require_napari()
    assert 'pip install "spacr[napari]"' in str(caught.value)
    # An ImportError subclass, so `except ImportError` callers keep working.
    assert isinstance(caught.value, ImportError)


@pytest.mark.skipif(nb.napari_available(),
                    reason="napari is installed in this environment")
def test_opening_without_napari_refuses_with_the_same_message(tmp_path):
    mask_path, _ = _field(tmp_path)
    with pytest.raises(nb.NapariExtraMissing):
        nb.open_in_napari(nb.load_handoff(mask_path))
    with pytest.raises(nb.NapariExtraMissing):
        nb.run_event_loop()


def test_open_in_napari_accepts_a_viewer_and_then_needs_no_napari(tmp_path):
    """The seam the GUI screen and these tests both go through."""
    mask_path, image_path = _field(tmp_path)
    viewer = FakeViewer()
    returned = nb.open_in_napari(nb.load_handoff(mask_path, image_path),
                                 viewer=viewer)
    assert returned is viewer
    assert [layer.name for layer in viewer.layers] == ["image", "mask"]


def test_correct_mask_drives_the_whole_round_trip_through_a_given_viewer(tmp_path):
    """``block=False`` with a viewer in hand needs no event loop and no napari.

    The viewer paints the moment the mask arrives, which is the fake's way of
    being "the user corrected it while the window was open".
    """
    class EditingViewer(FakeViewer):
        def add_labels(self, **kwargs):
            layer = super().add_labels(**kwargs)
            layer.data[3, 2] = 41
            return layer

    mask_path, image_path = _field(tmp_path)
    result = nb.correct_mask(mask_path, image_path, viewer=EditingViewer(),
                             block=False)
    assert result.changed_pixels == 1 and result.written is True
    assert result.mask_path.endswith(".tif")
    assert load_mask(mask_path)[3, 2] == 41
    assert is_curated(mask_path) is True


def test_napari_available_never_raises():
    assert nb.napari_available() in (True, False)


# ---------------------------------------------------------------------------
# The import cost, which is the reason for every lazy import in the module
# ---------------------------------------------------------------------------

def test_importing_the_bridge_imports_neither_napari_nor_a_second_qt_stack():
    """Module scope must stay clean, or a settings panel pays for napari."""
    code = (
        "import sys, spacr.napari_bridge;"
        "bad=[m for m in ('napari','qtpy','PyQt5','PyQt6','PySide2','vispy',"
        "'magicgui','torch','spacr.utils') if m in sys.modules];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]", out.stdout


def test_no_import_of_napari_sits_outside_a_function():
    """The rule, checked against the source rather than against a run.

    A lazy import that is only lazy on the paths a test happens to take is
    not lazy.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(nb))
    module_scope = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_scope += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_scope.append(node.module.split(".")[0])
    assert "napari" not in module_scope, module_scope


def test_it_is_reachable_as_a_lazy_submodule_of_the_package():
    import spacr

    assert "napari_bridge" in spacr._SUBMODULES
    assert spacr.napari_bridge is nb


def test_the_napari_extra_is_declared_in_setup_py():
    """An undeclared import is an ImportError on somebody else's machine."""
    import ast
    from pathlib import Path

    setup = Path(nb.__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(setup.read_text(encoding="utf-8"))
    extras = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            for keyword in node.keywords:
                if keyword.arg == "extras_require":
                    extras = ast.literal_eval(keyword.value)
    assert extras is not None
    assert nb.NAPARI_EXTRA in extras
    assert any(spec.startswith("napari") for spec in extras[nb.NAPARI_EXTRA])
    # Deliberately not aggregated: `all` must not install a second GUI stack.
    assert not any(spec.startswith("napari") for spec in extras.get("all", []))
