"""A second Mask run on a plate folder must not trust what the first one left.

GitHub #118 and #124, spaCR 1.5.0.8, both from the same user.

#118: a Mask run in test mode on a plate folder spaCR had already processed
died with ``No image stacks were produced from <plate>/test ... It holds no
images but does hold sub-folders (orig, stack) — if those are plates, point
src at one of them``. ``orig/`` and ``stack/`` are spaCR's own folders, so
the advice sent the user looking for a mistake they had not made.

#124: a run killed during normalisation left ``masks/stack_4_norm.npz`` cut
short; the next run reused it and died with ``zipfile.BadZipFile``.

Every test here builds the state a real earlier run leaves, by running the
real preprocessing and then doing to its output what a kill or the end-of-run
cleanup does, and then runs again the way the user did.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call

ROOT = Path(__file__).resolve().parents[1]


def _settings(src, **over):
    """Mask settings for the eight-TIFF CellVoyager plate, with defaults filled."""
    from spacr.settings import set_default_settings_preprocess_generate_masks

    settings = {
        "src": str(src), "metadata_type": "cellvoyager", "custom_regex": None,
        "channels": [0, 1], "nucleus_channel": 0, "cell_channel": 1,
        "pathogen_channel": None, "organelle_channel": None, "plot": False,
        "batch_size": 1, "test_mode": False, "timelapse": False,
        "normalize": True, "randomize": False, "verbose": False,
    }
    settings.update(over)
    return set_default_settings_preprocess_generate_masks(settings)


def _preprocess(src, **over):
    from spacr.io import preprocess_img_data
    return preprocess_img_data(_settings(src, **over))


def _stack_names(folder):
    return sorted(p.name for p in Path(folder).glob("*.npy"))


def _archive_fields(masks):
    """Load every archive the way the segmenter does and list its fields."""
    fields = []
    for path in sorted(Path(masks).glob("*.npz")):
        with np.load(path) as data:
            assert data["data"].shape[0] == len(data["filenames"])
            fields.extend(str(name) for name in data["filenames"])
    return sorted(fields)


@pytest.fixture
def plate(yokogawa_cellvoyager_dir):
    """A plate folder of eight raw TIFFs: two wells, two fields, two channels."""
    return yokogawa_cellvoyager_dir["src"]


@pytest.fixture
def quiet_plots(monkeypatch):
    import spacr.plot as PLOT
    monkeypatch.setattr(PLOT, "plot_arrays", lambda *a, **k: None)


def _finish_like_cleanup(src, keep_original):
    """Leave ``src`` as a finished run's cleanup does: merged/ built, the rest gone."""
    from spacr.utils import cleanup_pipeline_folders

    merged = Path(src) / "merged"
    merged.mkdir()
    for name in _stack_names(Path(src) / "stack"):
        np.save(merged / name, np.zeros((4, 4, 4), np.float32))
    cleanup_pipeline_folders(str(src), keep_intermediate=False,
                             keep_original=keep_original, verbose=False)


# ---------------------------------------------------------------------------
# #118: a folder spaCR already preprocessed
# ---------------------------------------------------------------------------

def test_a_rerun_on_a_finished_plate_builds_its_stacks_again_from_orig(plate):
    """keep_original_images on: orig/ survives the first run, stack/ does not.

    A second run -- new segmentation settings, same plate -- found no images
    directly in the folder and stopped with advice to point src at orig/ or
    stack/ "if those are plates". The raw images are right there in orig/.
    """
    _preprocess(plate)
    assert len(_stack_names(plate / "stack")) == 4
    assert len(list((plate / "orig").glob("*.tif"))) == 8
    _finish_like_cleanup(plate, keep_original=True)
    assert not (plate / "stack").exists() and not (plate / "masks").exists()

    _, out_src = _preprocess(plate)

    assert out_src == str(plate)
    assert len(_stack_names(plate / "stack")) == 4
    assert len(list((plate / "orig").glob("*.tif"))) == 8
    assert not list(plate.glob("*.tif"))
    assert _archive_fields(plate / "masks") == _stack_names(plate / "stack")


def test_a_rerun_on_a_plate_with_nothing_left_says_what_it_found(plate):
    """keep_original_images off: only merged/ survives. There is nothing to
    preprocess, and the message has to say THAT, not that orig/ and stack/
    might be plates. It must not create an empty orig/ and stack/ either:
    those were the sub-folders #118's message then pointed at."""
    _preprocess(plate)
    _finish_like_cleanup(plate, keep_original=False)
    before = sorted(p.name for p in plate.iterdir())

    with pytest.raises(FileNotFoundError) as excinfo:
        _preprocess(plate)

    message = str(excinfo.value)
    assert "No image stacks were produced" in message
    assert "already processed" in message
    assert "merged/ holds 4 merged field(s)" in message
    assert "point src at a copy of the raw images" in message
    assert "point src at one of them" not in message
    assert sorted(p.name for p in plate.iterdir()) == before


def test_issue_118_test_mode_on_a_plate_whose_raw_images_are_gone(
        plate, quiet_plots):
    """#118's own settings: test_mode on, test_images 1.

    The first run had save_original_images off, so the raw images were deleted
    once stack/ was written, and it stopped before normalising. Test mode
    samples raw images, found none, and #118's exact message followed:
    ``... <plate>/test ... sub-folders (orig, stack) — if those are plates``.
    The field stacks the first run wrote are what test mode can use.
    """
    _preprocess(plate, save_original_images=False)
    assert not list(plate.glob("*.tif")) and not (plate / "orig").exists()
    shutil.rmtree(plate / "masks")

    _, out_src = _preprocess(plate, test_mode=True, test_images=1)

    test = plate / "test"
    assert out_src == str(test)
    assert len(_stack_names(test / "stack")) == 1
    assert _stack_names(test / "stack")[0] in _stack_names(plate / "stack")
    assert _archive_fields(test / "masks") == _stack_names(test / "stack")
    assert len(_stack_names(plate / "stack")) == 4


def test_test_mode_on_a_plate_with_nothing_left_names_the_plate(
        plate, quiet_plots):
    _preprocess(plate)
    _finish_like_cleanup(plate, keep_original=False)

    with pytest.raises(FileNotFoundError) as excinfo:
        _preprocess(plate, test_mode=True, test_images=1)

    message = str(excinfo.value)
    assert f"Test mode copies a sample of the raw images in {plate}" in message
    assert f"{plate} is a folder spaCR has already processed" in message
    assert "point src at one of them" not in message
    assert not (plate / "test" / "orig").exists()
    assert not (plate / "test" / "stack").exists()


def test_a_pattern_that_matches_nothing_leaves_the_images_alone(tmp_path):
    """Images that no field could be read from were deleted outright with
    save_original_images off, and moved into orig/ beside an empty stack/
    with it on -- the state #118's message then misread as plates."""
    for keep in (False, True):
        folder = tmp_path / f"keep_{keep}"
        folder.mkdir()
        for name in ("img_01.tif", "img_02.tif"):
            (folder / name).write_bytes(b"not a field")

        with pytest.raises(FileNotFoundError) as excinfo:
            _preprocess(folder, save_original_images=keep)

        assert sorted(p.name for p in folder.iterdir()) == [
            "img_01.tif", "img_02.tif"]
        message = str(excinfo.value)
        assert "spaCR found 2 image file(s)" in message
        assert "could be read as a field" in message
        assert "metadata_type='cellvoyager'" in message


def test_a_rerun_after_a_kill_while_writing_stack_finishes_the_plate(plate):
    """Killed while writing stack/: two fields written, one of them cut short,
    and the raw images still in the plate folder, because they are moved
    only after every stack is written. The rerun used to take stack/ as it
    found it -- one good field, one it could not load -- and never built
    the other two."""
    _preprocess(plate)
    stack = plate / "stack"
    names = _stack_names(stack)
    for tif in (plate / "orig").glob("*.tif"):
        tif.rename(plate / tif.name)
    shutil.rmtree(plate / "orig")
    shutil.rmtree(plate / "masks")
    for name in names[2:]:
        (stack / name).unlink()
    whole = (stack / names[1]).read_bytes()
    (stack / names[1]).write_bytes(whole[: len(whole) // 3])

    _preprocess(plate)

    assert _stack_names(stack) == names
    for name in names:
        from spacr.resume import validate_merged_field
        assert validate_merged_field(str(stack / name)) == (True, "done")
    assert (stack / (names[1] + ".damaged")).exists()
    assert not list(plate.glob("*.tif"))
    assert len(list((plate / "orig").glob("*.tif"))) == 8
    assert _archive_fields(plate / "masks") == names


def test_a_stack_named_by_another_scheme_is_not_doubled(plate, capsys):
    """A stack/ none of whose names these images would produce (an older
    spaCR, or channel folders) is left alone rather than filled in beside,
    which would put every field on the plate in it twice."""
    stack = plate / "stack"
    stack.mkdir()
    rng = np.random.default_rng(1)
    np.save(stack / "plate1_A01_1_.npy",
            rng.integers(1, 4000, (8, 8, 2), dtype=np.uint16))

    _preprocess(plate)

    assert _stack_names(stack) == ["plate1_A01_1_.npy"]
    assert "another naming scheme" in capsys.readouterr().out
    assert len(list(plate.glob("*.tif"))) == 8


# ---------------------------------------------------------------------------
# #124: an archive a killed run cut short
# ---------------------------------------------------------------------------

def _killed_during_normalisation(plate):
    """Normalise the plate, then leave masks/ as a kill during it would:
    stack_2 cut short mid-write, stack_3 never started."""
    _preprocess(plate)
    masks = plate / "masks"
    assert sorted(p.name for p in masks.glob("*.npz")) == [
        f"stack_{i}_norm.npz" for i in range(4)]
    whole = (masks / "stack_2_norm.npz").read_bytes()
    (masks / "stack_2_norm.npz").write_bytes(whole[: len(whole) // 2])
    (masks / "stack_3_norm.npz").unlink()
    with pytest.raises(zipfile.BadZipFile):
        np.load(masks / "stack_2_norm.npz")
    return masks


def test_issue_124_a_truncated_archive_is_set_aside_and_its_fields_rebuilt(
        plate, capsys):
    masks = _killed_during_normalisation(plate)
    capsys.readouterr()

    _, out_src = _preprocess(plate)

    assert out_src == str(plate)
    assert _archive_fields(masks) == _stack_names(plate / "stack")
    assert (masks / "stack_2_norm.npz.damaged").exists()
    out = capsys.readouterr().out
    assert "Checked 3 normalised archive(s)" in out
    assert "2 whole, 1 damaged" in out
    assert "stack_2_norm.npz (not a complete zip archive" in out
    assert "2 field(s) in stack/ are in no whole archive" in out
    assert "stack_3_norm.npz, stack_4_norm.npz" in out
    assert "Found existing masks folder. Skipping preprocessing" in out


def test_a_damaged_archive_is_rebuilt_from_orig_when_stack_is_gone(plate):
    masks = _killed_during_normalisation(plate)
    fields = _stack_names(plate / "stack")
    shutil.rmtree(plate / "stack")

    _preprocess(plate)

    assert _stack_names(plate / "stack") == fields
    assert _archive_fields(masks) == fields


def test_a_damaged_archive_with_nothing_to_rebuild_from_is_an_error(plate):
    """No stack/ and no raw images: the fields of the damaged archive cannot
    come back, and going on would segment a plate short of them in silence."""
    masks = _killed_during_normalisation(plate)
    shutil.rmtree(plate / "stack")
    shutil.rmtree(plate / "orig")

    with pytest.raises(FileNotFoundError) as excinfo:
        _preprocess(plate)

    message = str(excinfo.value)
    assert "stack_2_norm.npz" in message
    assert "holds no field stacks to rebuild them from" in message
    assert (masks / "stack_2_norm.npz.damaged").exists()


def test_a_damaged_stack_beside_whole_archives_is_built_again_from_orig(
        plate, capsys):
    """The archives are whole and preprocessing is skipped, but merged/ is
    built from stack/, so a field stack cut short still has to come back --
    byte for byte what the first run wrote."""
    from spacr.resume import validate_merged_field

    _preprocess(plate)
    stack = plate / "stack"
    name = _stack_names(stack)[0]
    whole = np.load(stack / name)
    written = (stack / name).read_bytes()
    (stack / name).write_bytes(written[: len(written) // 2])
    capsys.readouterr()

    _preprocess(plate)

    assert validate_merged_field(str(stack / name)) == (True, "done")
    np.testing.assert_array_equal(np.load(stack / name), whole)
    assert (stack / (name + ".damaged")).exists()
    out = capsys.readouterr().out
    assert f"1 damaged, set aside as <name>.damaged: {name} (truncated)" in out
    assert "Found existing masks folder. Skipping preprocessing" in out


def test_no_archive_is_added_to_a_set_made_with_other_channels(plate, capsys):
    """Channels changed between the runs: an archive normalised with the new
    ones beside the old would give the segmenter a set it cannot index."""
    masks = _killed_during_normalisation(plate)
    capsys.readouterr()

    _preprocess(plate, nucleus_channel=0, cell_channel=0)

    assert sorted(p.name for p in masks.glob("*.npz")) == [
        "stack_0_norm.npz", "stack_1_norm.npz"]
    assert ("hold 2 channel(s) and these settings select 1"
            in capsys.readouterr().out)


def test_an_illumination_corrected_set_is_rebuilt_whole(plate):
    """Correction records completion for the published set as a whole, so a
    gap is not patched: the caller preprocesses again from stack/."""
    from spacr.io import _resume_normalized_archives

    _killed_during_normalisation(plate)
    settings = _settings(plate, illumination_correction=True)
    assert _resume_normalized_archives(settings, str(plate), [0, 1]) is False
    assert (plate / "masks" / "stack_2_norm.npz.damaged").exists()


@pytest.fixture
def fake_cellpose(monkeypatch):
    """CPU stand-in for the Cellpose model: one square object per field."""
    import torch
    import spacr.object as O
    import spacr.plot as PL

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(PL, "plot_cellpose4_output", lambda *a, **k: None)

    class _Model:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self, x=None, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
            check_cellpose_eval_call(x, channel_axis,
                                     z_axis=kwargs.get("z_axis"),
                                     do_3D=kwargs.get("do_3D", False))
            masks = []
            for image in x:
                mask = np.zeros(np.asarray(image).shape[:2], np.uint16)
                mask[20:60, 20:60] = 1
                masks.append(mask)
            return masks, [np.zeros_like(m, np.float32) for m in masks], None, None

    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(CellposeModel=_Model))


def test_issue_124_the_mask_run_finishes_after_a_kill(plate, fake_cellpose):
    """The user's path: run Mask again after the kill. It died in the
    segmenter with zipfile.BadZipFile; it now segments every field."""
    from spacr.core import preprocess_generate_masks

    _killed_during_normalisation(plate)

    preprocess_generate_masks(_settings(
        plate, preprocess=True, masks=True, save=True, adjust_cells=False,
        keep_intermediate=True, keep_original_images=True, n_jobs=1,
        cell_diameter=40, nucleus_diameter=20, magnification=20))

    fields = _stack_names(plate / "stack")
    assert _stack_names(plate / "merged") == fields
    for object_type in ("cell", "nucleus"):
        assert _stack_names(plate / "masks" / f"{object_type}_mask_stack") == fields


def test_issue_118_the_mask_run_in_test_mode_finishes(plate, fake_cellpose,
                                                     quiet_plots):
    """#118 through the user's entry point, with its own test-mode settings:
    it died in preprocess_img_data before any segmentation ran."""
    from spacr.core import preprocess_generate_masks

    _preprocess(plate, save_original_images=False)
    shutil.rmtree(plate / "masks")

    preprocess_generate_masks(_settings(
        plate, preprocess=True, masks=True, save=True, adjust_cells=False,
        test_mode=True, test_images=1, keep_intermediate=True,
        keep_original_images=True, n_jobs=1, cell_diameter=40,
        nucleus_diameter=20, magnification=20))

    test = plate / "test"
    fields = _stack_names(test / "stack")
    assert len(fields) == 1
    assert _stack_names(test / "merged") == fields
    assert len(_stack_names(plate / "stack")) == 4


# ---------------------------------------------------------------------------
# the root of #124: an archive is written whole or not at all
# ---------------------------------------------------------------------------

_STALLING_WRITER = textwrap.dedent("""
    import os, sys, time
    import numpy as np
    marker = sys.argv[2]

    def stall(file, *args, **kwargs):
        handle = open(file, 'wb') if isinstance(file, (str, os.PathLike)) else file
        handle.write(b'PK\\x03\\x04' + b'\\0' * 4096)
        handle.flush()
        open(marker, 'w').close()
        time.sleep(120)

    np.savez_compressed = stall
    import spacr.io as IO
    stack = sys.argv[1]
    settings = {'timelapse': False, 'randomize': False, 'batch_size': 4,
                'lower_percentile': 2, 'nucleus_channel': 0, 'cell_channel': 1,
                'nucleus_background': 100, 'nucleus_signal_to_noise': 10,
                'remove_background_nucleus': False, 'cell_background': 100,
                'cell_signal_to_noise': 10, 'remove_background_cell': False,
                'plot': False}
    IO.concatenate_and_normalize(stack, [0, 1], settings=settings)
""")


def test_a_run_killed_while_writing_an_archive_leaves_no_archive(tmp_path):
    """SIGKILL, not an exception: no ``finally`` runs. The old writer opened
    ``masks/stack_0_norm.npz`` itself, so the kill left the start of an
    archive under the final name, which is #124."""
    stack = tmp_path / "plate" / "stack"
    stack.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(2):
        np.save(stack / f"plate_A01_{i + 1}_1.npy",
                rng.integers(0, 4000, (16, 16, 2), dtype=np.uint16))
    marker = tmp_path / "writing"

    child = subprocess.Popen(
        [sys.executable, "-c", _STALLING_WRITER, str(stack), str(marker)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 180
        while not marker.exists():
            if child.poll() is not None:
                pytest.fail("writer exited before writing: "
                            + child.stderr.read().decode()[-2000:])
            if time.monotonic() > deadline:
                pytest.fail("writer never reached the archive write")
            time.sleep(0.2)
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)

    masks = tmp_path / "plate" / "masks"
    assert not list(masks.glob("*.npz"))
    leftovers = [p.name for p in masks.iterdir()]
    assert len(leftovers) == 1 and leftovers[0].endswith(".partial")

    from spacr.io import _sweep_partial_writes
    assert _sweep_partial_writes(str(masks)) == leftovers
    assert not list(masks.iterdir())
