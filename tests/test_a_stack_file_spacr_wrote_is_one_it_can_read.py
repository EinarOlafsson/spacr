"""A Mask run on a macOS external volume died reading a file spaCR never wrote.

GitHub #121 (auto-filed) and #117, spaCR 1.5.0.8 on an Apple M4, plate on
``/Volumes/jk-ummi``::

    object.py:790, in generate_cellpose_masks_sam
        with np.load(path) as data:
    ValueError: This file contains pickled (object) data. If you trust the
    file you can load it unsafely using the `allow_pickle=` keyword argument

The message says "pickled", and nothing spaCR writes into ``masks/`` is
pickled: the normalised archive holds ``data`` (float32) and ``filenames``
(a unicode array). The #117 log names the file that actually failed one
stage earlier -- ``stack/._test_N06_5_1.npy`` -- and that is an AppleDouble
sidecar. macOS writes one, ``._<name>``, beside every file that carries
extended attributes on a volume that cannot store them natively (exFAT, FAT,
many SMB shares). It keeps the ``.npy`` / ``.npz`` ending of the file it
shadows, so every ``os.listdir(...) if name.endswith('.npz')`` in the mask
path picked it up, and ``np.load`` treats any file that is neither ``.npy``
nor a zip as a pickle.

``allow_pickle=True`` is therefore not a fix: it turns the ValueError into an
UnpicklingError, and it would let any file dropped into the folder execute
code at load. These tests pin the real fix -- the mask path's listings leave
dotfiles out -- the way the user reached it: a test-mode Mask run over a
plate whose volume writes a sidecar beside every file.
"""
from __future__ import annotations

import ast
import builtins
import inspect
import io
import os
import pickle
import re
import struct
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call


def _appledouble_bytes() -> bytes:
    """The 4096-byte AppleDouble header macOS writes for a file's metadata.

    Magic ``0x00051607``, version 2, the ``Mac OS X`` filler, and two
    entries (Finder info, resource fork) -- the layout of the ``._`` files
    macOS leaves on an exFAT or SMB volume.
    """
    header = struct.pack(">II16sH", 0x00051607, 0x00020000,
                         b"Mac OS X        ", 2)
    header += struct.pack(">III", 9, 50, 3760)
    header += struct.pack(">III", 2, 3810, 286)
    return header + b"\x00" * (4096 - len(header))


APPLEDOUBLE = _appledouble_bytes()
#: numpy's refusal of a pickled payload, in both of its wordings: numpy 2.5
#: says "This file contains pickled (object) data", and numpy 1.26, which the
#: CI runners install, says "Cannot load file containing pickled data when
#: allow_pickle=False".
PICKLE_MESSAGE = re.compile(r"pickled (?:\(object\) )?data")


def _sidecar(path) -> str:
    """``dir/._name`` for ``dir/name``."""
    path = os.fspath(path)
    return os.path.join(os.path.dirname(path), "._" + os.path.basename(path))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def macos_volume(monkeypatch):
    """Make a folder behave like a macOS external volume.

    Inside the returned root, as macOS does on a volume without native
    extended attributes: every file opened for writing gets a ``._<name>``
    AppleDouble sidecar, a rename carries the sidecar along, and a delete
    removes it. Paths outside the root are untouched.

    :returns: a callable taking the root folder to emulate.
    """
    roots = []
    real_open = builtins.open
    real_replace = os.replace
    real_rename = os.rename
    real_remove = os.remove

    def inside(path) -> bool:
        if not isinstance(path, (str, os.PathLike)):
            return False
        full = os.path.realpath(os.fspath(path))
        return any(full.startswith(root + os.sep) for root in roots)

    def shadowed(path) -> bool:
        return inside(path) and not os.path.basename(
            os.fspath(path)).startswith("._")

    def fake_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if any(flag in mode for flag in "wax+") and shadowed(file):
            side = _sidecar(file)
            if not os.path.exists(side):
                with real_open(side, "wb") as out:
                    out.write(APPLEDOUBLE)
        return handle

    def follow(src, dst):
        if shadowed(src) and os.path.exists(_sidecar(src)):
            real_replace(_sidecar(src), _sidecar(dst))

    def fake_replace(src, dst, *args, **kwargs):
        real_replace(src, dst, *args, **kwargs)
        follow(src, dst)

    def fake_rename(src, dst, *args, **kwargs):
        real_rename(src, dst, *args, **kwargs)
        follow(src, dst)

    def fake_remove(path, *args, **kwargs):
        real_remove(path, *args, **kwargs)
        if shadowed(path) and os.path.exists(_sidecar(path)):
            real_remove(_sidecar(path))

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(io, "open", fake_open)
    monkeypatch.setattr(os, "replace", fake_replace)
    monkeypatch.setattr(os, "rename", fake_rename)
    monkeypatch.setattr(os, "remove", fake_remove)
    monkeypatch.setattr(os, "unlink", fake_remove)

    def mount(root):
        roots.append(os.path.realpath(os.fspath(root)))
        return root

    return mount


@pytest.fixture
def fake_cellpose(monkeypatch):
    """Replace Cellpose with a model that draws two square objects per field.

    :returns: list of the ``pretrained_model`` values models were built with.
    """
    import torch
    import spacr.object as sobj
    import spacr.plot as splot

    built = []

    class _Model:
        def __init__(self, gpu=False, pretrained_model="cpsam", device=None,
                     **_ignored):
            built.append(pretrained_model)

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=None,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis, z_axis=z_axis,
                                     do_3D=do_3D,
                                     stitch_threshold=stitch_threshold)
            masks, flows = [], []
            for image in x:
                height, width = np.asarray(image).shape[:2]
                mask = np.zeros((height, width), dtype=np.uint16)
                mask[4:14, 4:14] = 1
                mask[20:30, 20:30] = 2
                masks.append(mask)
                flows.append(np.zeros((height, width), dtype=np.float32))
            return masks, flows, None

    cpu = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sobj.accelerator, "cellpose_kwargs",
                        lambda: {"gpu": False, "device": cpu})
    monkeypatch.setattr(sobj, "cp_models",
                        types.SimpleNamespace(CellposeModel=_Model))
    monkeypatch.setattr(splot, "plot_cellpose4_output",
                        lambda *args, **kwargs: None)
    return built


def _stack_folder(root: Path, fields=("plate1_A01_1", "plate1_A01_2")) -> Path:
    """A ``stack/`` of four-channel fields, saved the way preprocessing does."""
    stack = root / "stack"
    stack.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for field in fields:
        array = rng.integers(100, 4000, size=(40, 40, 4)).astype(np.uint16)
        np.save(stack / f"{field}.npy", array)
    return stack


def _normalize_settings(**over):
    from spacr.settings import set_default_settings_preprocess_img_data

    settings = set_default_settings_preprocess_img_data({
        "src": "unused", "timelapse": False, "randomize": False,
        "batch_size": 50, "plot": False,
    })
    settings.update(over)
    return settings


def _mask_settings(src, **over):
    settings = {
        "src": str(src), "cell_channel": 3, "nucleus_channel": 0,
        "pathogen_channel": 2, "channels": [0, 1, 2, 3],
        "cell_diameter": 30, "nucleus_diameter": 30,
        "pathogen_diameter": 30, "plot": False, "verbose": False,
        "save": True, "batch_size": 50, "timelapse": False,
        "seg_qc": "off", "resume": False,
    }
    settings.update(over)
    return settings


class TestTheArchiveHoldsNoObjectArray:
    """The premise of #121 as filed, measured against the writer."""

    def test_the_normalised_archive_loads_without_pickle(self, tmp_path):
        """What ``concatenate_and_normalize`` writes is plain arrays."""
        from spacr.io import concatenate_and_normalize

        stack = _stack_folder(tmp_path)
        concatenate_and_normalize(str(stack), channels=[0, 3, 2],
                                  save_dtype=np.float32,
                                  settings=_normalize_settings())

        archives = sorted((tmp_path / "masks").glob("*.npz"))
        assert [a.name for a in archives] == ["stack_0_norm.npz"]
        with np.load(archives[0], allow_pickle=False) as data:
            assert data["data"].dtype == np.float32
            assert data["filenames"].dtype.kind == "U"
            assert sorted(data["filenames"].tolist()) == [
                "plate1_A01_1.npy", "plate1_A01_2.npy"]

    def test_a_sidecar_is_what_numpy_calls_pickled(self, tmp_path):
        """The exact message of #121, from an AppleDouble file."""
        side = tmp_path / "._stack_0_norm.npz"
        side.write_bytes(APPLEDOUBLE)
        with pytest.raises(ValueError, match=PICKLE_MESSAGE):
            np.load(side)

    def test_allowing_pickle_would_not_have_fixed_it(self, tmp_path):
        """``allow_pickle=True`` trades one error for another."""
        side = tmp_path / "._stack_0_norm.npz"
        side.write_bytes(APPLEDOUBLE)
        with pytest.raises(pickle.UnpicklingError):
            np.load(side, allow_pickle=True)


class TestTheListingLeavesDotfilesOut:
    """The helper every mask-path listing goes through."""

    def test_order_is_kept_and_dotfiles_are_dropped(self, tmp_path, monkeypatch):
        from spacr.io import _listdir_visible

        names = ["b.npy", "._b.npy", "a.npz", ".DS_Store",
                 ".spacr_tmp_x1.npy", "c.txt"]
        monkeypatch.setattr(os, "listdir", lambda folder: list(names))
        assert _listdir_visible(str(tmp_path)) == ["b.npy", "a.npz", "c.txt"]

    def test_a_missing_folder_still_raises(self, tmp_path):
        from spacr.io import _listdir_visible

        with pytest.raises(FileNotFoundError):
            _listdir_visible(str(tmp_path / "absent"))


MASK_PATH_LISTINGS = {
    "spacr.io": (
        "_rename_and_organize_image_files", "_merge_channels",
        "_normalized_npz_field_ids", "_publish_v1_normalized_archives",
        "_concatenate_and_normalize_impl",
        "_create_movies_from_npy_per_channel", "preprocess_img_data",
        "_load_and_concatenate_arrays", "convert_separate_files_to_yokogawa",
        "convert_to_yokogawa",
    ),
    "spacr.object": (
        "generate_cellpose_masks_sam", "generate_cellpose_masks",
        "generate_organelle_masks_sam",
    ),
    "spacr.core": ("_overlay_candidates", "preprocess_generate_masks"),
    "spacr.utils": (
        "check_mask_folder", "_run_test_mode", "adjust_cell_masks",
        "cleanup_pipeline_folders",
    ),
    "spacr.plot": ("plot_arrays",),
}


def _bare_listdir_lines(module_name, function_name):
    """Lines in one function that call ``os.listdir`` directly."""
    import importlib

    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    source = textwrap.dedent(inspect.getsource(function))
    first = inspect.getsourcelines(function)[1]
    found = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "listdir"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"):
            found.append(first + node.lineno - 1)
    return found


@pytest.mark.parametrize(
    "module_name,function_name",
    [(m, f) for m, names in MASK_PATH_LISTINGS.items() for f in names])
def test_no_mask_path_function_lists_a_folder_with_bare_listdir(
        module_name, function_name):
    """Every listing on the mask path skips dotfiles, not just the one that crashed.

    The raw-image listing learnt to skip ``._`` files long ago (``io.py``,
    ``_rename_and_organize_image_files``) and none of the listings after it
    did, which is how #121 got through.
    Directory-name listings are included: they cost nothing to route through
    the helper, and a rule with exceptions is the rule the next site slips
    past.

    This pins the functions it names; it does not find new ones. The
    listings that do not call ``os.listdir`` from ``spacr.io``'s reach are
    held by behaviour tests below instead: ``seg_qc._iter_masks``,
    ``illumination._merged_files``, ``_v1_v2_bridge.v2_mask_source``,
    ``pipeline_v2.FilenameMapper.discover`` (``Path.iterdir``),
    ``utils.generate_image_path_map`` (``os.walk``) and ``validate._listdir``.
    ``seg_qc`` is tested to import no torch and ``spacr.io`` imports it at
    module level, which is why those modules filter inline.
    """
    lines = _bare_listdir_lines(module_name, function_name)
    assert not lines, (
        f"{module_name}.{function_name} calls os.listdir directly at "
        f"line(s) {lines}; list through spacr.io._listdir_visible so a macOS "
        f"._ sidecar or an interrupted .spacr_tmp_ write is not read as data")


def test_the_mask_generator_reads_past_a_sidecar(tmp_path, fake_cellpose):
    """#121 at the line it crashed on, with the archive the pipeline writes."""
    from spacr.io import concatenate_and_normalize
    from spacr.object import generate_cellpose_masks_sam

    stack = _stack_folder(tmp_path)
    mask_src = concatenate_and_normalize(str(stack), channels=[0, 3, 2],
                                         save_dtype=np.float32,
                                         settings=_normalize_settings())
    (Path(mask_src) / "._stack_0_norm.npz").write_bytes(APPLEDOUBLE)

    generate_cellpose_masks_sam(mask_src, _mask_settings(tmp_path), "cell")

    written = sorted(p.name for p in
                     (Path(mask_src) / "cell_mask_stack").iterdir())
    assert written == ["plate1_A01_1.npy", "plate1_A01_2.npy"]
    assert fake_cellpose == ["cpsam"]


def test_the_normaliser_does_not_count_a_sidecar_as_a_field(tmp_path, capsys):
    """The first failure in the #117 log: ``._test_N06_5_1.npy`` in ``stack/``."""
    from spacr.io import concatenate_and_normalize

    stack = _stack_folder(tmp_path)
    (stack / "._plate1_A01_1.npy").write_bytes(APPLEDOUBLE)

    concatenate_and_normalize(str(stack), channels=[0, 3, 2],
                              save_dtype=np.float32,
                              settings=_normalize_settings())

    out = capsys.readouterr().out
    assert not PICKLE_MESSAGE.search(out)
    assert "RUN INCOMPLETE" not in out
    with np.load(tmp_path / "masks" / "stack_0_norm.npz") as data:
        assert sorted(data["filenames"].tolist()) == [
            "plate1_A01_1.npy", "plate1_A01_2.npy"]


def test_segmentation_qc_does_not_score_a_sidecar(tmp_path):
    """QC scores what ``spacr.object`` wrote and nothing beside it.

    Before the fix one good mask and its sidecar scored "FAIL - 1 of 2
    fields failed (50%): fix the segmentation before running Measure".
    """
    from spacr.seg_qc import _iter_masks

    np.save(tmp_path / "plate1_A01_1.npy", np.zeros((8, 8), np.uint16))
    (tmp_path / "._plate1_A01_1.npy").write_bytes(APPLEDOUBLE)

    names = [name for name, load in _iter_masks(str(tmp_path))]
    assert names == ["plate1_A01_1"]


def test_the_illumination_estimate_reads_past_a_sidecar(tmp_path):
    """With illumination correction on, the Mask run estimates from ``stack/``.

    ``prepare_segmentation_illumination`` hands ``stack/`` to
    :func:`spacr.illumination.estimate_illumination`, which listed it bare:
    the sidecar was read as a field of its own plate, ``._plate1``.
    """
    from spacr.illumination import _merged_files, estimate_illumination

    stack = _stack_folder(tmp_path, fields=("plate1_A01_1", "plate1_A01_2",
                                            "plate1_A01_3"))
    for field in list(stack.glob("*.npy")):
        Path(_sidecar(field)).write_bytes(APPLEDOUBLE)

    grouped = _merged_files(str(stack))
    assert sorted(grouped) == ["plate1"]
    assert all(not os.path.basename(p).startswith(".")
               for paths in grouped.values() for p in paths)
    assert estimate_illumination(str(stack), channels=[0]) is not None


def _cq1_plate(root: Path, fields=(5, 6)) -> Path:
    """Raw CQ1 tiffs named like the #117 plate, four channels per field."""
    import tifffile

    root.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for field in fields:
        for channel in (1, 2, 3, 4):
            image = rng.integers(100, 3000, size=(48, 48)).astype(np.uint16)
            image[6:16, 6:16] += 5000
            image[24:34, 24:34] += 5000
            tifffile.imwrite(
                root / f"W0318F{field:04d}T0001Z000C{channel}.tif", image)
    return root


def test_a_test_mode_mask_run_on_a_macos_volume_completes(
        tmp_path, macos_volume, fake_cellpose, capsys):
    """The #117 run, reproduced on a volume that writes ``._`` sidecars.

    Settings are the reporter's where they matter: CQ1 names, channels
    ``[0, 1, 2, 3]``, cell 3 / nucleus 0 / pathogen 2, test mode on one image
    set, cells adjusted with the nucleus and pathogen masks, segmentation QC
    in report mode. The raw tiffs carry sidecars too, as a Finder copy leaves
    them.
    """
    from spacr.core import preprocess_generate_masks

    plate = _cq1_plate(tmp_path / "Projection")
    for tif in list(plate.glob("*.tif")):
        Path(_sidecar(tif)).write_bytes(APPLEDOUBLE)
    macos_volume(tmp_path)

    settings = {
        "src": str(plate), "metadata_type": "cq1", "custom_regex": None,
        "channels": [0, 1, 2, 3], "cell_channel": 3, "nucleus_channel": 0,
        "pathogen_channel": 2, "cell_diameter": 30, "nucleus_diameter": 30,
        "pathogen_diameter": 30, "magnification": 20, "test_mode": True,
        "test_images": 1, "preprocess": True, "masks": True, "save": True,
        "plot": False, "verbose": False, "n_jobs": 1, "batch_size": 50,
        "adjust_cells": True, "seg_qc": "report", "randomize": True,
        "keep_intermediate": True, "consolidate": False, "timelapse": False,
        "pipeline_style": "v1",
    }
    preprocess_generate_masks(settings)

    out = capsys.readouterr().out
    assert not PICKLE_MESSAGE.search(out)
    assert "RUN INCOMPLETE" not in out
    qc_lines = [line for line in out.splitlines()
                if line.startswith("Segmentation QC (")]
    assert len(qc_lines) == 3, qc_lines
    assert all(" of 1 fields" in line for line in qc_lines), qc_lines

    test_dir = plate / "test"
    assert (test_dir / "stack" / "._test_N06_5_1.npy").exists() or \
        (test_dir / "stack" / "._test_N06_6_1.npy").exists(), (
        "the emulated volume wrote no sidecar beside the stack, so this "
        "test is not exercising #117")
    merged = sorted(p.name for p in (test_dir / "merged").iterdir()
                    if p.suffix == ".npy" and not p.name.startswith("."))
    assert len(merged) == 1 and merged[0].startswith("test_N06_")
    array = np.load(test_dir / "merged" / merged[0])
    assert array.shape[-1] == 4 + 3
    for role in ("cell", "nucleus", "pathogen"):
        masks = [p.name for p in
                 (test_dir / "masks" / f"{role}_mask_stack").iterdir()
                 if not p.name.startswith(".")]
        assert masks == merged, role


SEPARATE_FILES_REGEX = (
    r"(?P<plateID>.*)_(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)"
    r"_w(?P<chanID>\d)\.tif")


def _separate_files_plate(root: Path, wells=("B03", "C07")) -> Path:
    """Per-channel tiffs named ``exp1_<well>_s1_w<c>.tif``, each with a sidecar."""
    import tifffile

    root.mkdir(parents=True)
    rng = np.random.default_rng(2)
    for well in wells:
        for channel in (1, 2, 3, 4):
            image = rng.integers(100, 3000, size=(48, 48)).astype(np.uint16)
            image[6:16, 6:16] += 5000
            image[24:34, 24:34] += 5000
            path = root / f"exp1_{well}_s1_w{channel}.tif"
            tifffile.imwrite(path, image)
            Path(_sidecar(path)).write_bytes(APPLEDOUBLE)
    return root


def _converted_wells(folder: Path):
    """The well tokens of the Yokogawa-named files a conversion wrote."""
    return sorted({p.name.split("_")[1] for p in folder.glob("plate*_*.tif")})


def test_the_regex_conversion_keeps_its_wells_past_sidecars(tmp_path, capsys):
    """``metadata_type='auto'`` with a custom regex, on a folder with sidecars.

    The regex conversion used to read ``._exp1_B03_s1_w1.tif`` as a tiff and
    raise, and the Mask run's silent fallback to the regex-less conversion
    then numbered the wells itself: B03 and C07 came out as A03 and A04.
    """
    from spacr.io import convert_separate_files_to_yokogawa

    plate = _separate_files_plate(tmp_path / "plate")
    convert_separate_files_to_yokogawa(str(plate), SEPARATE_FILES_REGEX)

    assert _converted_wells(plate) == ["B03", "C07"]
    import pandas as pd
    log = pd.read_csv(plate / "rename_log.csv")
    assert not log["Original File(s)"].str.contains(r"\._").any()


def test_the_plain_conversion_spends_no_well_on_a_sidecar(tmp_path, capsys):
    """``metadata_type='auto'`` without a regex: one image, one sidecar.

    It used to report "RUN INCOMPLETE - 1 of 2" for the sidecar, stamp the
    converted folder incomplete, and hand the image the second well, A02.
    """
    import tifffile
    from spacr.errors import run_is_complete
    from spacr.io import convert_to_yokogawa

    folder = tmp_path / "plate"
    folder.mkdir()
    tifffile.imwrite(folder / "img1.tif", np.ones((8, 8), np.uint16))
    (folder / "._img1.tif").write_bytes(APPLEDOUBLE)

    ledger = convert_to_yokogawa(str(folder))

    assert ledger.n_failed == 0 and ledger.n_attempted == 1
    assert sorted(p.name for p in folder.glob("plate*.tif")) == [
        "plate1_A01_T0001F001L01C01.tif"]
    assert "RUN INCOMPLETE" not in capsys.readouterr().out
    assert run_is_complete(str(folder / "rename_log.csv"))


def test_an_auto_metadata_mask_run_on_a_macos_volume_keeps_its_wells(
        tmp_path, macos_volume, fake_cellpose, capsys):
    """The Mask run with ``metadata_type='auto'`` and a regex, as a user sets it.

    Before the fix the regex conversion raised on the first sidecar and the
    run fell back, without saying why, to the regex-less conversion. That
    gave every file a well of its own ("RUN INCOMPLETE - 8 of 16 items
    failed" for the sidecars), so each well held one channel and the run
    stopped with "IndexError: index 3 is out of bounds for axis 3 with size
    1". With one channel per well the same fallback finishes instead, on
    wells the plate never had: see the two tests above.
    """
    from spacr.core import preprocess_generate_masks

    plate = _separate_files_plate(tmp_path / "plate")
    macos_volume(tmp_path)

    settings = {
        "src": str(plate), "metadata_type": "auto",
        "custom_regex": SEPARATE_FILES_REGEX,
        "channels": [0, 1, 2, 3], "cell_channel": 3, "nucleus_channel": 0,
        "pathogen_channel": 2, "cell_diameter": 30, "nucleus_diameter": 30,
        "pathogen_diameter": 30, "magnification": 20, "test_mode": False,
        "preprocess": True, "masks": True, "save": True, "plot": False,
        "verbose": False, "n_jobs": 1, "batch_size": 50,
        "adjust_cells": True, "seg_qc": "report", "randomize": False,
        "keep_intermediate": True, "consolidate": False, "timelapse": False,
        "pipeline_style": "v1",
    }
    preprocess_generate_masks(settings)

    out = capsys.readouterr().out
    assert "not a TIFF file" not in out
    assert "RUN INCOMPLETE" not in out
    assert not PICKLE_MESSAGE.search(out)
    merged = sorted(p.name for p in (plate / "merged").iterdir()
                    if p.suffix == ".npy" and not p.name.startswith("."))
    assert [name.split("_")[1] for name in merged] == ["B03", "C07"], merged
    import pandas as pd
    log = pd.read_csv(plate / "rename_log.csv")
    assert not log["Original File(s)"].str.contains(r"\._").any()


def test_consolidating_a_macos_volume_copies_no_sidecar(tmp_path):
    """``consolidate=True`` flattens sub-folders into ``consolidated/``.

    ``sub/._img1.tif`` used to be copied as ``sub_._img1.tif``: the
    rename moved the dot off the front, so no listing downstream could tell
    it was a sidecar any more. A volume's own hidden folders
    (``.Spotlight-V100``, ``.Trashes``) were walked and copied too.
    """
    from spacr.utils import copy_images_to_consolidated, generate_image_path_map

    (tmp_path / "sub").mkdir()
    (tmp_path / ".Spotlight-V100").mkdir()
    for name in ("sub/img1.tif", "top.tif", ".Spotlight-V100/store.tif"):
        (tmp_path / name).write_bytes(b"image")
    for name in ("sub/._img1.tif", "._top.tif"):
        (tmp_path / name).write_bytes(APPLEDOUBLE)

    image_map = generate_image_path_map(str(tmp_path))
    assert sorted(os.path.basename(v) for v in image_map.values()) == [
        "sub_img1.tif", "top.tif"]

    copy_images_to_consolidated(image_map, str(tmp_path))
    assert sorted(os.listdir(tmp_path / "consolidated")) == [
        "sub_img1.tif", "top.tif"]


def _v2_merged(root: Path, fields=("plate1_A01_1", "plate1_A01_2")) -> Path:
    """A v2 ``merged/``: one image plane and one cell mask plane per field."""
    import json

    merged = root / "merged"
    merged.mkdir(parents=True)
    for field in fields:
        stack = np.zeros((40, 40, 2), np.uint16)
        stack[4:14, 4:14, 1] = 1
        stack[20:30, 20:30, 1] = 2
        path = merged / f"stack_{field}.npy"
        np.save(path, stack)
        Path(_sidecar(path)).write_bytes(APPLEDOUBLE)
    (merged / "channel_order.json").write_text(json.dumps(
        {"image_channels": ["cell"], "mask_channels": ["cell"]}))
    return merged


def test_the_v2_scorecard_does_not_score_a_sidecar(tmp_path, capsys):
    """``pipeline_style='v2'`` scores ``merged/`` through ``v2_mask_source``.

    It listed the folder bare, so each sidecar became a field of its own
    and failed to load: "FAIL - 2 of 4 fields failed (50%): fix the
    segmentation before running Measure" about a plate with nothing wrong.
    """
    from spacr._v1_v2_bridge import v2_mask_source
    from spacr.core import _score_v2_masks

    merged = _v2_merged(tmp_path)
    assert sorted(v2_mask_source(merged, "cell")) == [
        "plate1_A01_1", "plate1_A01_2"]

    result = _score_v2_masks(tmp_path, {"seg_qc": "report", "verbose": True})
    assert len(result["field_qcs"]) == 2
    assert "FAIL" not in capsys.readouterr().out


def _yokogawa_plate(root: Path, fields=(1, 2)) -> Path:
    """Raw CellVoyager tiffs, channels 0 and 1 per field, each with a sidecar."""
    import tifffile

    root.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for field in fields:
        for channel in (0, 1):
            image = rng.integers(100, 3000, size=(48, 48)).astype(np.uint16)
            image[6:16, 6:16] += 5000
            image[24:34, 24:34] += 5000
            path = root / (f"plate1_A01_T0001F{field:03d}L01A01Z01"
                           f"C{channel:02d}.tif")
            tifffile.imwrite(path, image)
            Path(_sidecar(path)).write_bytes(APPLEDOUBLE)
    return root


def test_the_v2_discovery_does_not_take_a_sidecar_for_an_image(tmp_path):
    """v2 lists the raw folder itself, with ``Path.iterdir``.

    The CellVoyager pattern starts ``(?P<plateID>.*)_``, so
    ``._plate1_A01_...tif`` matched as plate ``._plate1`` and
    ``stream_originals_to_stack`` then died reading it: "not a TIFF file".
    """
    from spacr.pipeline_v2 import FilenameMapper

    plate = _yokogawa_plate(tmp_path / "plate")
    mapper = FilenameMapper.discover(plate, metadata_type="auto")

    assert len(mapper.records) == 4
    assert {record.plate for record in mapper.records} == {"plate1"}


class _TwoSquaresCellpose:
    """A Cellpose double for the v2 path that draws two squares per image."""

    def __init__(self, *args, **kwargs):
        self.pretrained_model = None

    def eval(self, images, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
             niter=None, augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        masks = []
        for image in images:
            mask = np.zeros(np.asarray(image).shape[:2], dtype=np.uint16)
            mask[6:16, 6:16] = 1
            mask[24:34, 24:34] = 2
            masks.append(mask)
        return masks, None, None


def test_a_v2_mask_run_on_a_macos_volume_completes(
        tmp_path, macos_volume, monkeypatch, capsys):
    """The Mask run with ``pipeline_style='v2'`` on a volume that writes sidecars.

    Before the fix it stopped at the first raw sidecar ("not a TIFF
    file"); with discovery fixed alone, the scorecard still counted each
    ``merged/._stack_*.npy`` as a failed field.
    """
    import spacr.accelerator as accelerator
    from spacr.core import preprocess_generate_masks

    monkeypatch.setattr("cellpose.models.CellposeModel", _TwoSquaresCellpose)
    monkeypatch.setattr(accelerator, "is_gpu", lambda: False)
    plate = _yokogawa_plate(tmp_path / "plate")
    macos_volume(tmp_path)

    preprocess_generate_masks({
        "src": str(plate), "pipeline_style": "v2", "metadata_type": "auto",
        "custom_regex": None, "channels": [0, 1], "cell_channel": 1,
        "nucleus_channel": 0, "pathogen_channel": None,
        "cell_diameter": 30, "nucleus_diameter": 30, "seg_qc": "report",
        "verbose": True, "plot": False, "save": True,
    })

    out = capsys.readouterr().out
    assert "not a TIFF file" not in out
    assert "FAIL" not in out
    stacks = sorted(p.name for p in (plate / "merged").glob("stack_*.npy"))
    assert len(stacks) == 2, stacks
    assert (plate / "merged" / f"._{stacks[0]}").exists(), (
        "the emulated volume wrote no sidecar beside the v2 stack, so this "
        "test is not exercising the scorecard's listing")
    card = (plate / "qc" / "segmentation_qc_cell.csv").read_text()
    assert "._" not in card


def test_the_preflight_counts_no_sidecar(tmp_path):
    """The Mask run's preflight lists ``stack/`` and ``merged/`` too.

    Sorted, every ``._`` name comes before every field, so with three
    fields the three sidecars were the three files ``_peek_planes`` tried:
    no plane count, and a file count of six.
    """
    from spacr.validate import _peek_planes

    stack = _stack_folder(tmp_path, fields=("plate1_A01_1", "plate1_A01_2",
                                            "plate1_A01_3"))
    for field in list(stack.glob("*.npy")):
        Path(_sidecar(field)).write_bytes(APPLEDOUBLE)

    planes, example, count = _peek_planes(str(stack))
    assert (planes, count) == (4, 3)
    assert not example.startswith(".")
