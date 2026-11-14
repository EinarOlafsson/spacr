"""Opening, meshing and saving a native 3-D mask, and every refusal on the way.

``load_volume_image_and_mask`` reads a spatial TIFF or MRC volume with its
physical spacing (OME sizes in micrometres, ImageJ resolution, an MRC voxel
size in nanometres, or pixels when the file says nothing) and restores a
saved geometry sidecar bound to the exact source and mask pixels.  Each way
a file can be wrong -- incomplete or unsupported spacing, a zero resolution,
a volume past the voxel budget, a source that changes while it is opened, a
sidecar with foreign geometry or broken anchor controls -- is refused with
the sentence the user reads.  ``VolumeMaskSession.geometry_for_label``
returns saved geometry as is and meshes a label exactly, or refuses it; and
``save_volume_mask`` refuses before it writes anything it would have to
take back.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.qt import mask_engine as engine
from spacr.qt.widgets.gate_spec import BoxGate, EditableVolumeGate, GateSet
from spacr.tiff_io import write_tiff

pytestmark = pytest.mark.qt

_ZYX = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)


def _box(limits, name="900"):
    return EditableVolumeGate.from_gate(
        BoxGate.from_limits(name, ("X", "Y", "Z"), limits))


def _source(folder, name="source.tif"):
    write_tiff(folder / name, _ZYX,
               metadata={"axes": "ZYX", "voxel_spacing": [1., 1., 1.],
                         "unit": "um"})


def _saved(folder):
    """A source with one saved 2x2x2 box object and its sidecar record."""
    _source(folder)
    session = engine.load_volume_image_and_mask(str(folder), "source.tif")
    session.apply_geometry(900, _box([(0, 2)] * 3))
    engine.save_volume_mask(session)
    sidecar = Path(session.mask_path + ".volume.json")
    return session, sidecar, json.loads(sidecar.read_text(encoding="utf-8"))


def _ome(folder, name, **sizes):
    tifffile.imwrite(folder / name, np.zeros((3, 4, 5), dtype=np.uint16),
                     ome=True, metadata={"axes": "ZYX", **sizes})


def test_load_volume_image_and_mask_reads_each_files_spacing(tmp_path,
                                                             subtests):
    """Each source opens with its exact axes, spacing, unit and an empty mask."""
    folder = tmp_path
    _source(folder)
    _ome(folder, "plain.ome.tif")
    _ome(folder, "sized.ome.tif", PhysicalSizeX=500., PhysicalSizeXUnit="nm",
         PhysicalSizeY=0.5, PhysicalSizeZ=0.002, PhysicalSizeZUnit="mm")
    mrcfile = pytest.importorskip("mrcfile")
    with mrcfile.new(str(folder / "volume.mrc"), overwrite=True) as handle:
        handle.set_data(np.zeros((3, 4, 5), dtype=np.float32))
        handle.voxel_size = (20.0, 30.0, 40.0)
    # case, filename, shape, spacing, unit
    cases = [
        ("tiff voxel_spacing metadata", "source.tif", (4, 5, 6),
         (1., 1., 1.), "um"),
        ("OME without physical sizes is in pixels", "plain.ome.tif",
         (3, 4, 5), (1., 1., 1.), "pixel"),
        ("OME sizes convert to micrometres", "sized.ome.tif", (3, 4, 5),
         (2., .5, .5), "um"),
        ("MRC voxel size in nanometres", "volume.mrc", (3, 4, 5),
         (4., 3., 2.), "nm"),
    ]
    for case, name, shape, spacing, unit in cases:
        with subtests.test(msg=case):
            session = engine.load_volume_image_and_mask(str(folder), name)
            assert session.axes == "ZYX"
            assert session.spacing == pytest.approx(spacing)
            assert session.unit == unit
            assert session.mask.shape == shape
            assert session.mask.dtype == np.uint16 and not session.mask.any()
            assert session.source_path == str(folder / name)
            assert session.mask_path == engine.mask_save_path(str(folder), name)
            assert session.geometries == {}
            assert session._saved_mask_sha256 is None
            assert session._saved_geometry_sha256 is None


def test_load_volume_image_and_mask_refuses_each_unreadable_source(
        tmp_path, monkeypatch, subtests):
    """Each malformed volume is refused with its message and cause."""
    _ome(tmp_path, "two_axes.ome.tif", PhysicalSizeX=1., PhysicalSizeY=1.)
    _ome(tmp_path, "parsecs.ome.tif", PhysicalSizeX=1.,
         PhysicalSizeXUnit="pc", PhysicalSizeY=1., PhysicalSizeZ=1.)
    tifffile.imwrite(tmp_path / "imagej.tif",
                     np.zeros((3, 4, 5), dtype=np.uint16), imagej=True,
                     resolution=((0, 1), (1, 1)),
                     metadata={"axes": "ZYX", "spacing": 2.})
    _source(tmp_path)
    # case, filename, message fragment, cause type
    cases = [
        ("OME missing the Z size", "two_axes.ome.tif",
         "OME volumes need complete supported physical spacing", KeyError),
        ("OME in an unsupported unit", "parsecs.ome.tif",
         "OME volumes need complete supported physical spacing", KeyError),
        ("ImageJ zero resolution", "imagej.tif",
         "Voxel spacing must contain three positive finite distances", None),
    ]
    for case, name, fragment, cause in cases:
        with subtests.test(msg=case):
            with pytest.raises(ValueError) as raised:
                engine.load_volume_image_and_mask(str(tmp_path), name)
            assert fragment in str(raised.value)
            if cause is not None:
                assert isinstance(raised.value.__cause__, cause)

    with subtests.test(msg="a TIFF over the voxel budget"):
        with monkeypatch.context() as patch:
            patch.setattr(engine.VolumeMaskSession, "MAX_VOXELS", 10)
            with pytest.raises(ValueError,
                               match="This TIFF exceeds the native volume "
                                     "memory budget"):
                engine.load_volume_image_and_mask(str(tmp_path), "source.tif")

    with subtests.test(msg="a source that changes while it is opened"):
        original = engine._yolo_source_hash
        seen = []

        def drifting(path):
            seen.append(Path(path).name)
            digest = original(path)
            return digest if len(seen) == 1 else "changed" + digest

        with monkeypatch.context() as patch:
            patch.setattr(engine, "_yolo_source_hash", drifting)
            with pytest.raises(ValueError,
                               match="The source volume changed while "
                                     "opening it"):
                engine.load_volume_image_and_mask(str(tmp_path), "source.tif")
        assert seen == ["source.tif", "source.tif"]
        assert not (tmp_path / "masks").exists()


def test_load_volume_image_and_mask_restores_or_refuses_a_saved_sidecar(
        tmp_path, subtests):
    """A sidecar restores geometry and controls exactly, or a bad one is refused."""
    with subtests.test(msg="restored"):
        folder = tmp_path / "restored"
        folder.mkdir()
        session, sidecar, record = _saved(folder)
        reopened = engine.load_volume_image_and_mask(str(folder), "source.tif")
        assert reopened.geometries == session.geometries
        assert list(reopened.geometries) == [900]
        assert reopened._editor_controls == {"anchor_density": 2,
                                             "show_anchors": True}
        assert int((reopened.mask == 900).sum()) == 8
        assert set(np.unique(reopened.mask)) == {0, 900}
        assert reopened._saved_mask_sha256 == session._saved_mask_sha256
        assert reopened._saved_geometry_sha256 == session._saved_geometry_sha256
        assert record["version"] == 1 and record["axes"] == "ZYX"

    box = GateSet([BoxGate.from_limits("900", ("X", "Y", "Z"),
                                       [(0, 2)] * 3)]).to_dict()
    # case, record changes, message fragment
    cases = [
        ("geometry that is not an editable volume", {"geometry": box},
         "The volume sidecar contains invalid object geometry"),
        ("controls that are not a mapping", {"editor_controls": []},
         "The saved volume anchor controls are invalid"),
        ("controls missing a key", {"editor_controls": {"anchor_density": 2}},
         "The saved volume anchor controls are invalid"),
        ("a density past 8", {"editor_controls": {
            "anchor_density": 9, "show_anchors": True}},
         "The saved volume anchor controls are invalid"),
        ("a float density", {"editor_controls": {
            "anchor_density": 2.0, "show_anchors": True}},
         "The saved volume anchor controls are invalid"),
        ("an integer for a switch", {"editor_controls": {
            "anchor_density": 2, "show_anchors": 1}},
         "The saved volume anchor controls are invalid"),
        ("a sidecar of another version", {"version": 2},
         "The saved volume geometry does not match these source and mask "
         "pixels"),
    ]
    for index, (case, changes, fragment) in enumerate(cases):
        with subtests.test(msg=f"refused: {case}"):
            folder = tmp_path / f"case{index}"
            folder.mkdir()
            _session, sidecar, record = _saved(folder)
            record.update(changes)
            sidecar.write_text(json.dumps(record), encoding="utf-8")
            with pytest.raises(ValueError) as raised:
                engine.load_volume_image_and_mask(str(folder), "source.tif")
            assert fragment in str(raised.value)


def _mrc(folder, data, voxel_size=None, name="volume.mrc"):
    mrcfile = pytest.importorskip("mrcfile")
    path = folder / name
    with mrcfile.new(str(path), overwrite=True) as handle:
        handle.set_data(data)
        if voxel_size is not None:
            handle.voxel_size = voxel_size
    return path


def test_read_mrc_volume_opens_a_volume_or_says_why_not(tmp_path,
                                                        monkeypatch,
                                                        subtests):
    """Each MRC file gives the exact (image, axes, spacing, unit) or a reason."""
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    # case, data, voxel size, spacing, unit
    cases = [
        ("no voxel size opens in pixels", data, 0.0, (1., 1., 1.), "pixel"),
        ("a voxel size opens in nanometres", data, (20.0, 30.0, 40.0),
         (4., 3., 2.), "nm"),
        ("a partial voxel size is not physical", data, (20.0, 0.0, 40.0),
         (1., 1., 1.), "pixel"),
        ("integer voxels keep their dtype", data.astype(np.int16), None,
         (1., 1., 1.), "pixel"),
    ]
    for index, (case, pixels, size, spacing, unit) in enumerate(cases):
        with subtests.test(msg=case):
            path = _mrc(tmp_path, pixels, size, name=f"v{index}.mrc")
            image, axes, got_spacing, got_unit = engine._read_mrc_volume(path)
            np.testing.assert_array_equal(image, pixels)
            assert image.dtype == pixels.dtype
            assert axes == "ZYX"
            assert got_spacing == pytest.approx(spacing)
            assert got_unit == unit

    # case, data, MAX_VOXELS, message fragment
    refusals = [
        ("one section", np.zeros((1, 4, 5), np.float32), None,
         "This MRC file holds one section, not a volume"),
        ("beyond the budget", np.zeros((3, 4, 5), np.float32), 10,
         "This volume exceeds the native volume memory budget"),
        ("complex numbers", np.zeros((3, 4, 5), np.complex64), None,
         "3D masks require a finite numeric spatial TIFF"),
    ]
    for index, (case, pixels, budget, fragment) in enumerate(refusals):
        with subtests.test(msg=f"refused: {case}"):
            path = _mrc(tmp_path, pixels, name=f"r{index}.mrc")
            with monkeypatch.context() as patch:
                if budget is not None:
                    patch.setattr(engine.VolumeMaskSession, "MAX_VOXELS",
                                  budget)
                with pytest.raises(ValueError) as raised:
                    engine._read_mrc_volume(path)
            assert fragment in str(raised.value)

    with subtests.test(msg="refused: mrcfile is not installed"):
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "mrcfile", None)
            with pytest.raises(ValueError) as raised:
                engine._read_mrc_volume(tmp_path / "v0.mrc")
        assert 'pip install "spacr[tem]"' in str(raised.value)
        assert isinstance(raised.value.__cause__, ImportError)


def test_save_volume_mask_refuses_before_writing_what_it_cannot_keep(
        tmp_path, monkeypatch, subtests):
    """Each refused save leaves no mask, sidecar, staging folder or lock behind."""
    empty = engine.VolumeMaskSession(np.zeros((5, 5, 5)),
                                     np.zeros((5, 5, 5), dtype=np.uint16))
    for case, session in [("a session without a source", empty),
                          ("not a session", "not a session")]:
        with subtests.test(msg=case):
            with pytest.raises(ValueError,
                               match="The source volume changed; no mask was "
                                     "saved"):
                engine.save_volume_mask(session)

    def _opened(name):
        folder = tmp_path / name
        folder.mkdir()
        _source(folder)
        return folder, engine.load_volume_image_and_mask(str(folder),
                                                         "source.tif")

    with subtests.test(msg="a mask path that is the source"):
        folder, session = _opened("over")
        before = (folder / "source.tif").read_bytes()
        session.mask_path = session.source_path
        with pytest.raises(ValueError,
                           match="The mask output must not overwrite the "
                                 "source volume"):
            engine.save_volume_mask(session)
        assert (folder / "source.tif").read_bytes() == before
        assert not (folder / "source.tif.volume.json.lock").exists()

    with subtests.test(msg="another save holds the lock"):
        folder, session = _opened("locked")
        lock = Path(session.mask_path + ".volume.json.lock")
        lock.parent.mkdir(parents=True)
        lock.write_bytes(b"")
        with pytest.raises(ValueError,
                           match="This volume is already being saved"):
            engine.save_volume_mask(session)
        assert lock.exists() and not Path(session.mask_path).exists()
        assert sorted(p.name for p in lock.parent.iterdir()) == [lock.name]

    with subtests.test(msg="a failed first publication takes the mask back"):
        folder, session = _opened("failed")
        session.apply_geometry(900, _box([(0, 1)] * 3))
        original = engine.os.replace

        def reject(source, destination):
            if Path(source).name == "geometry.json":
                raise OSError("injected geometry publication failure")
            return original(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(engine.os, "replace", reject)
            with pytest.raises(OSError, match="injected geometry publication"):
                engine.save_volume_mask(session)
        target = Path(session.mask_path)
        assert sorted(p.name for p in target.parent.iterdir()) == []
        assert session._saved_mask_sha256 is None
        assert session._saved_geometry_sha256 is None

    with subtests.test(msg="the same save, unhindered, publishes both files"):
        written = engine.save_volume_mask(session)
        assert written == session.mask_path
        assert sorted(p.name for p in Path(written).parent.iterdir()) == [
            "source.tif", "source.tif.volume.json"]
        assert session._saved_mask_sha256 == engine._yolo_source_hash(
            Path(written))


def test_geometry_for_label_returns_saved_geometry_or_an_exact_mesh(subtests):
    """A saved geometry is returned as is; a label is meshed exactly or refused."""
    single = np.zeros((3, 3, 3), dtype=np.uint16)
    single[1, 1, 1] = 5
    session = engine.VolumeMaskSession(np.zeros(single.shape), single)

    with subtests.test(msg="one voxel is a closed cube"):
        gate = session.geometry_for_label(5)
        assert gate.name == "5"
        assert len(gate.vertices) == 8 and len(gate.faces) == 12
        assert gate.columns == ("X", "Y", "Z")
        assert session.geometries == {}, "previewing a mesh saved it"

    with subtests.test(msg="a saved geometry is what the label shows"):
        saved = object()
        session.geometries[5] = saved
        assert session.geometry_for_label(5) is saved
        session.geometries.clear()

    for case, label in [("an absent label", 6), ("label zero", 0),
                        ("a negative label", -1), ("a boolean", True),
                        ("a float", 5.0)]:
        with subtests.test(msg=f"refused: {case}"):
            with pytest.raises(ValueError,
                               match="Choose an existing positive object "
                                     "label"):
                session.geometry_for_label(label)
            assert session.geometries == {}

    with subtests.test(msg="refused: too much boundary detail"):
        speckled = np.zeros((24, 24, 24), dtype=np.uint16)
        speckled[::2, ::2, ::2] = 7
        dense = engine.VolumeMaskSession(np.zeros(speckled.shape), speckled)
        with pytest.raises(ValueError, match="too much boundary detail"):
            dense.geometry_for_label(7)
        assert dense.geometries == {}
        np.testing.assert_array_equal(dense.mask, speckled)
