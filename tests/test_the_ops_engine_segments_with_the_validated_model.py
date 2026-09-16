"""The objects phase: which Cellpose model, which diameter, and what numbering refuses.

372 PART 14-L validated well A1 (1,280,002 objects) with a ``CellposeModel``
built with no ``pretrained_model`` at all -- the library's own default -- so
the settings' ``cpsam`` has to mean that default, not an explicit load of
different weights. V19 asked that ``ops_gpu`` off keep the work on the CPU.
V11b found ``number(strict=True)`` refusing groups that no window saw whole;
the engine numbers without them and keeps the refusal in the report, so the
dropped groups are counted rather than silent.

Cellpose is replaced where the model is not the subject: the construction is
recorded against the installed ``CellposeModel.__init__`` signature, and
segmentation is a threshold with the installed ``eval`` signature. Composing,
sewing, numbering and the store are the shipped code.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
models = pytest.importorskip("cellpose.models")

from scipy import ndimage

import spacr.accelerator
from spacr import ops_engine
from spacr.ops_store import read_table, row_count, write_table


@pytest.fixture
def built(monkeypatch):
    """Replace ``CellposeModel`` with a recorder that binds like the real one.

    :param monkeypatch: pytest's monkeypatch.
    :returns: the list each construction's explicitly passed arguments are
        appended to. An argument the installed signature does not accept
        raises TypeError, as the real constructor would.
    """
    signature = inspect.signature(models.CellposeModel.__init__)
    calls = []

    class _RecordingModel:
        """A CellposeModel that loads no weights and remembers how it was built."""

        def __init__(self, *args, **kwargs):
            """Bind against the installed signature and record what was passed.

            :param args: positional arguments, as the engine passed them.
            :param kwargs: keyword arguments, as the engine passed them.
            """
            bound = signature.bind(self, *args, **kwargs)
            calls.append({name: value for name, value in bound.arguments.items()
                          if name != "self"})

    monkeypatch.setattr(models, "CellposeModel", _RecordingModel)
    return calls


def _the_card_must_not_be_asked():
    """Stand in for :func:`spacr.accelerator.cellpose_kwargs` where it must not run."""
    raise AssertionError("ops_gpu is off, but the accelerator was consulted")


@pytest.mark.parametrize("name", [None, "", "cpsam", "  cpsam "])
def test_the_default_model_name_builds_the_librarys_own_default(built, monkeypatch, name):
    """No ``pretrained_model`` is passed, so Cellpose loads the weights the well was validated with.

    With ``ops_gpu`` off the model is built for the CPU and the accelerator is
    never asked.
    """
    monkeypatch.setattr(spacr.accelerator, "cellpose_kwargs", _the_card_must_not_be_asked)

    ops_engine._cellpose_model({"cellpose_model": name}, gpu=False)

    assert built == [{"gpu": False}]


def test_another_model_name_is_loaded_by_that_name(built, monkeypatch):
    """A custom model, named by path in the settings, is what Cellpose is asked to load."""
    monkeypatch.setattr(spacr.accelerator, "cellpose_kwargs", _the_card_must_not_be_asked)

    ops_engine._cellpose_model({"cellpose_model": " /models/ops_nuclei "}, gpu=False)

    assert built == [{"gpu": False, "pretrained_model": "/models/ops_nuclei"}]


def test_on_the_card_the_model_takes_the_machines_accelerator_arguments(built, monkeypatch):
    """``gpu``, ``device`` and ``use_bfloat16`` come from one resolver, together.

    They have to agree (cellpose branches on ``gpu`` before it reads
    ``device``), so the engine passes the accelerator's three and does not
    spell out its own.
    """
    device = object()
    monkeypatch.setattr(spacr.accelerator, "cellpose_kwargs",
                        lambda: {"gpu": True, "device": device, "use_bfloat16": False})

    ops_engine._cellpose_model({}, gpu=True)
    ops_engine._cellpose_model({"cellpose_model": "cyto3"}, gpu=True)

    card = {"gpu": True, "device": device, "use_bfloat16": False}
    assert built == [card, {**card, "pretrained_model": "cyto3"}]


class _ThresholdModel:
    """A stand-in for Cellpose: nuclei are the composite's pixels above 4,000."""

    def __init__(self):
        """Start with no segmentation calls recorded."""
        self.diameters = []

    def eval(self, image, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
             niter=None, augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        """Label one composed window and record the diameter it was asked for.

        The installed cellpose's eval parameters are written out rather than
        taken as ``**kwargs``, so an argument cellpose removed fails here too
        (tests/test_cellpose_api_contract.py).

        :param image: one composed window.
        :param diameter: what the engine passed.
        :returns: ``(masks, flows, styles)``, masks first as Cellpose returns them.
        """
        self.diameters.append(diameter)
        return ndimage.label(np.asarray(image) > 4000)[0], None, None


def _lay_well(tmp_path, monkeypatch, canvas, *, tile, step, model):
    """Cut ``canvas`` into two side-by-side nuclear tiles and store where they sit.

    The placements go into ``ops_geometry`` as the stitch phase leaves them,
    so the objects phase runs on its own the way a re-run does.

    :param tmp_path: the test's folder.
    :param monkeypatch: pytest's monkeypatch.
    :param canvas: the well's nuclear image.
    :param tile: the tile edge.
    :param step: how far the second tile sits to the right of the first.
    :param model: the segmenter the engine is given.
    :returns: ``(settings, db)``.
    """
    raw = tmp_path / "raw" / "c1"
    raw.mkdir(parents=True)
    places = {0: (0, 0), 1: (0, step)}
    for site, (top, left) in places.items():
        tifffile.imwrite(raw / f"10X_c1_A1_DAPI_Site-{site}.tif",
                         canvas[top:top + tile, left:left + tile].astype(np.uint16))
    out = tmp_path / "out"
    out.mkdir()
    db = str(out / "measurements.db")
    write_table(db, "ops_geometry", pd.DataFrame([{
        "plate": "plate", "well": "A1", "cycle": 1, "site": site,
        "y": float(y), "x": float(x), "origin_y": 0.0, "origin_x": 0.0,
        "tile_height": tile, "tile_width": tile} for site, (y, x) in places.items()]))

    def model_for(settings, gpu):
        """The engine's model hook, with its real parameters.

        :param settings: the run's settings.
        :param gpu: whether the card was asked for.
        :returns: the stand-in.
        """
        assert gpu is False and settings["plate"] == "plate"
        return model

    monkeypatch.setattr(ops_engine, "_cellpose_model", model_for)
    settings = {"genotype_source": str(tmp_path / "raw"), "dst_root": str(out),
                "plate": "plate", "ops_gpu": False}
    return settings, db


def test_the_diameter_setting_reaches_cellpose_and_the_windows_are_counted_out(
        tmp_path, monkeypatch, capsys):
    """A diameter typed into the settings arrives as a float on every window.

    A well of 120 windows reports its progress every 50, which is what the
    operator watches during the 593 s of the validated well. Only the objects
    phase was asked for, so nothing is stitched or decoded, and every nucleus
    is one object numbered 1..N.
    """
    monkeypatch.setattr(ops_engine, "_WINDOW", 16)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 4)
    canvas = np.full((96, 176), 500.0)
    nuclei = [(y, x) for y in range(10, 90, 20) for x in range(10, 170, 20)]
    for y, x in nuclei:
        canvas[y:y + 3, x:x + 3] = 8000.0
    model = _ThresholdModel()
    settings, db = _lay_well(tmp_path, monkeypatch, canvas, tile=96, step=80, model=model)

    report = ops_engine.run_ops({**settings, "cellpose_diameter": "12"},
                                phases=("objects",))["wells"]["A1"]

    objects = report["objects"]
    assert "stitch" not in report and "decode" not in report
    assert objects["windows"] == 120 and objects["empty_windows"] == 0
    assert objects["objects"] == len(nuclei) and objects["strict_refusal"] == ""
    assert len(model.diameters) == 120
    assert all(type(value) is float and value == 12.0 for value in model.diameters)
    printed = capsys.readouterr().out
    assert "OPS: A1 objects: 50 of 120 windows" in printed
    assert "OPS: A1 objects: 100 of 120 windows" in printed
    assert sorted(read_table(db, "ops_objects")["object_id"]) == list(
        range(1, len(nuclei) + 1))
    assert row_count(db, "ops_barcodes") is None


def test_an_object_no_window_saw_whole_is_dropped_and_the_refusal_is_reported(
        tmp_path, monkeypatch):
    """A bar longer than the window overlap is clipped by both windows that see it.

    Strict numbering refuses it; the engine numbers the well without it, so a
    fragment is never counted as a nucleus, and the refusal and the count of
    such groups go into the report. The nucleus that one window saw whole is
    the well's only object. No diameter is set, so Cellpose is left to
    choose one.
    """
    monkeypatch.setattr(ops_engine, "_WINDOW", 64)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 8)
    canvas = np.full((64, 112), 500.0)
    canvas[10:16, 10:16] = 8000.0
    canvas[30:38, 40:100] = 8000.0
    model = _ThresholdModel()
    settings, db = _lay_well(tmp_path, monkeypatch, canvas, tile=64, step=48, model=model)

    objects = ops_engine.run_ops(settings, phases=("objects",))["wells"]["A1"]["objects"]

    assert objects["windows"] == 2 and objects["observations"] == 3
    assert objects["objects"] == 1 and objects["all_clipped_groups"] == 1
    assert objects["strict_refusal"].startswith(
        "1 group(s) of observations were clipped by every window that saw them")
    assert model.diameters == [None, None]
    stored = read_table(db, "ops_objects")
    assert list(stored["object_id"]) == [1]
    assert (stored["centroid_y"].iloc[0], stored["centroid_x"].iloc[0]) == (12.5, 12.5)


def test_a_well_in_which_segmentation_finds_no_nuclei_is_refused(tmp_path, monkeypatch):
    """An empty object list would let every later phase run and produce nothing.

    The refusal names the well, and no ``ops_objects`` table is written for
    the readiness gate to find.
    """
    monkeypatch.setattr(ops_engine, "_WINDOW", 64)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 8)
    canvas = np.full((64, 112), 500.0)
    settings, db = _lay_well(tmp_path, monkeypatch, canvas, tile=64, step=48,
                             model=_ThresholdModel())

    with pytest.raises(ValueError, match="well A1: segmentation found no nuclei"):
        ops_engine.run_ops(settings, phases=("objects",))
    assert row_count(db, "ops_objects") is None
    assert not os.path.exists(os.path.join(settings["dst_root"], "A1", "ops_report.json"))
