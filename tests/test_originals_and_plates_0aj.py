"""Item 0aj P1b: the original images, letter + number wells, and one plate per project.

keep_original_images (the existing key):
- KEEP (default): once a field is committed to fov.ome.zarr and verified,
  its originals are MOVED into project/orig/.
- DON'T KEEP: each original is deleted only after its field is committed
  AND every stored plane is verified byte for byte against that original.
A run cut short between commit and verify, or between verify and delete,
never loses an image, and a rerun finishes the job.

Preflight: wells that are not letter + number stop Mask before anything
changes. A source of several plates is split into plate folders (never
overwriting, sizes and counts checked) only on request; headless runs stop
with the dry-run list unless split_plates is on.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from tests.test_a_rerun_trusts_nothing_a_killed_run_left import (  # noqa: E402,F401
    _RUN, _settings, fake_cellpose,
)

ROOT = Path(__file__).resolve().parents[1]


def _tiffs(folder):
    return sorted(p.name for p in Path(folder).glob("*.tif"))


def _captured(manifest):
    """``{field file name: [plane per channel]}``, read from the TIFFs before the run."""
    fields = {}
    for entry in sorted(manifest, key=lambda e: (e["well"], e["field"], e["channel"])):
        name = f"plate1_{entry['well']}_{int(entry['field'])}_1.npy"
        fields.setdefault(name, []).append(tifffile.imread(entry["path"]))
    return fields


def _assert_store_holds(store, captured):
    from spacr.image_store import _image_entries, read_image

    entries = _image_entries(store)
    assert sorted(entries) == sorted(captured)
    for name, planes in captured.items():
        image = read_image(Path(store) / entries[name]["key"])
        assert image.shape[0] == len(planes)
        for stored, original in zip(image, planes):
            assert stored.dtype == original.dtype
            assert np.array_equal(stored, original)


def _run(plate, **over):
    from spacr.core import preprocess_generate_masks

    run = dict(_RUN, keep_intermediate=False)
    run.update(over)
    preprocess_generate_masks(_settings(plate, preprocess=True, **run))


# ---------------------------------------------------------------------------
# keep_original_images
# ---------------------------------------------------------------------------

def test_keep_moves_the_originals_to_orig_only_after_the_fields_are_verified(
        yokogawa_cellvoyager_dir, fake_cellpose, monkeypatch):
    import spacr.image_store as IS

    plate = yokogawa_cellvoyager_dir["src"]
    names = _tiffs(plate)
    captured = _captured(yokogawa_cellvoyager_dir["manifest"])
    seen = {}
    real = IS._settle_originals

    def settle(project, **kwargs):
        # First called once preprocessing committed the fields: every
        # original is still in src.
        seen.setdefault("in_src", _tiffs(project))
        seen.setdefault("committed", sorted(IS._image_entries(IS._merged_store_path(project))))
        return real(project, **kwargs)

    monkeypatch.setattr(IS, "_settle_originals", settle)
    _run(plate)
    assert seen["in_src"] == names and seen["committed"] == sorted(captured)
    assert _tiffs(plate) == []
    assert _tiffs(plate / "orig") == names
    _assert_store_holds(plate / "fov.ome.zarr", captured)
    report = IS.verify_project(plate)
    assert report["mismatches"] == [] and report["originals_compared"] == 8


def test_dont_keep_deletes_each_original_after_its_planes_match_the_store(
        yokogawa_cellvoyager_dir, fake_cellpose):
    from spacr.image_store import verify_store

    plate = yokogawa_cellvoyager_dir["src"]
    captured = _captured(yokogawa_cellvoyager_dir["manifest"])
    _run(plate, keep_original_images=False)
    assert _tiffs(plate) == [] and _tiffs(plate / "orig") == []
    _assert_store_holds(plate / "fov.ome.zarr", captured)
    assert verify_store(plate)["mismatches"] == []


def test_the_older_save_original_images_off_also_waits_for_the_store(
        yokogawa_cellvoyager_dir, fake_cellpose, monkeypatch):
    import spacr.image_store as IS

    plate = yokogawa_cellvoyager_dir["src"]
    captured = _captured(yokogawa_cellvoyager_dir["manifest"])
    seen = {}
    real = IS._settle_originals

    def settle(project, **kwargs):
        seen.setdefault("in_src", len(_tiffs(project)))
        seen.setdefault("keep", kwargs.get("keep"))
        return real(project, **kwargs)

    monkeypatch.setattr(IS, "_settle_originals", settle)
    _run(plate, save_original_images=False)
    assert seen == {"in_src": 8, "keep": False}
    assert _tiffs(plate) == [] and _tiffs(plate / "orig") == []
    _assert_store_holds(plate / "fov.ome.zarr", captured)


def test_a_run_stopped_between_commit_and_verify_loses_nothing_and_resumes(
        yokogawa_cellvoyager_dir, fake_cellpose, monkeypatch):
    import spacr.image_store as IS

    plate = yokogawa_cellvoyager_dir["src"]
    names = _tiffs(plate)
    captured = _captured(yokogawa_cellvoyager_dir["manifest"])

    def stopped(*_a, **_k):
        raise KeyboardInterrupt("stopped between commit and verify")

    with monkeypatch.context() as patch:
        patch.setattr(IS, "_field_checksum_problems", stopped)
        with pytest.raises(KeyboardInterrupt):
            _run(plate, keep_original_images=False)
    # The fields are committed; not one original moved or went.
    assert sorted(IS._image_entries(plate / "fov.ome.zarr")) == sorted(captured)
    assert _tiffs(plate) == names
    _run(plate, keep_original_images=False, resume=True)
    assert _tiffs(plate) == [] and _tiffs(plate / "orig") == []
    _assert_store_holds(plate / "fov.ome.zarr", captured)


def test_a_run_stopped_between_verify_and_delete_loses_nothing_and_resumes(
        yokogawa_cellvoyager_dir, fake_cellpose, monkeypatch):
    import spacr.image_store as IS

    plate = yokogawa_cellvoyager_dir["src"]
    captured = _captured(yokogawa_cellvoyager_dir["manifest"])
    deleted = []
    real = IS._delete_original

    def delete_three(path):
        if len(deleted) == 3:
            raise KeyboardInterrupt("stopped between verify and delete")
        real(path)
        deleted.append(path)

    with monkeypatch.context() as patch:
        patch.setattr(IS, "_delete_original", delete_three)
        with pytest.raises(KeyboardInterrupt):
            _run(plate, keep_original_images=False)
    assert len(_tiffs(plate)) == 5
    _assert_store_holds(plate / "fov.ome.zarr", captured)
    _run(plate, keep_original_images=False, resume=True)
    assert _tiffs(plate) == [] and _tiffs(plate / "orig") == []
    _assert_store_holds(plate / "fov.ome.zarr", captured)


def _committed_project(plate, monkeypatch):
    """A finished Mask run whose originals are still in src (the settle step skipped)."""
    import spacr.image_store as IS

    with monkeypatch.context() as patch:
        patch.setattr(IS, "_settle_originals", lambda *a, **k: {"problems": []})
        _run(plate, keep_original_images=False)
    assert len(_tiffs(plate)) == 8


def test_a_process_killed_while_deleting_loses_nothing_and_a_rerun_finishes(
        yokogawa_cellvoyager_dir, fake_cellpose, monkeypatch):
    from spacr.image_store import _settle_originals

    plate = yokogawa_cellvoyager_dir["src"]
    captured = _captured(yokogawa_cellvoyager_dir["manifest"])
    _committed_project(plate, monkeypatch)
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(ROOT)!r})
        import spacr.image_store as IS
        real, done = IS._delete_original, []
        def kill_after_two(path):
            if len(done) == 2:
                os._exit(9)
            real(path)
            done.append(path)
        IS._delete_original = kill_after_two
        IS._settle_originals({str(plate)!r}, keep=False)
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT),
                            env=dict(os.environ, PYTHONPATH=str(ROOT)), timeout=300)
    assert result.returncode == 9
    assert len(_tiffs(plate)) == 6
    report = _settle_originals(plate, keep=False)
    assert report["problems"] == [] and len(report["deleted"]) == 6
    assert _tiffs(plate) == []
    _assert_store_holds(plate / "fov.ome.zarr", captured)


def test_a_process_killed_while_moving_loses_nothing_and_a_rerun_finishes(
        yokogawa_cellvoyager_dir, fake_cellpose, monkeypatch):
    from spacr.image_store import _settle_originals

    plate = yokogawa_cellvoyager_dir["src"]
    names = _tiffs(plate)
    _committed_project(plate, monkeypatch)
    # A move cut between its link and its unlink: the file is under both names.
    (plate / "orig").mkdir(exist_ok=True)
    os.link(plate / names[0], plate / "orig" / names[0])
    report = _settle_originals(plate, keep=True)
    assert report["problems"] == []
    assert _tiffs(plate) == [] and _tiffs(plate / "orig") == names


def test_a_field_that_does_not_match_its_originals_keeps_them_in_place(
        yokogawa_cellvoyager_dir, fake_cellpose, monkeypatch):
    import zarr

    from spacr.image_store import _image_entries, _settle_originals

    plate = yokogawa_cellvoyager_dir["src"]
    _committed_project(plate, monkeypatch)
    store = plate / "fov.ome.zarr"
    name, entry = sorted(_image_entries(store).items())[0]
    array = zarr.open_array(str(store / entry["key"] / "0"), mode="r+")
    array[0, 3, 3] = int(array[0, 3, 3]) ^ 1
    report = _settle_originals(plate, keep=False)
    assert any(name in problem for problem in report["problems"])
    # The three good fields' originals went; the bad field's two stayed.
    assert len(report["deleted"]) == 6 and len(_tiffs(plate)) == 2


def test_a_projected_channel_is_kept_in_orig_even_when_keeping_is_off(tmp_path):
    from spacr.image_store import _settle_originals

    with pytest.MonkeyPatch.context() as patch:
        import spacr.image_store as IS
        patch.setattr(IS, "_image_entries", lambda store: {
            "plate1_A01_1_1.npy": {"key": "A/1/1", "image": True,
                                   "plane_sha256": ["x"]}})
        patch.setattr(IS, "read_image", lambda path: np.zeros((1, 4, 4), np.uint16))
        patch.setattr(IS, "_field_checksum_problems", lambda *a: [])
        (tmp_path / "a.tif").write_bytes(b"1")
        (tmp_path / "b.tif").write_bytes(b"2")
        patch.setattr(IS, "_original_files", lambda *a: (
            {"plate1_A01_1_1.npy": {"1": [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")]}},
            ["1"], []))
        report = _settle_originals(tmp_path, keep=False)
    assert report["deleted"] == [] and len(report["kept"]) == 1
    assert _tiffs(tmp_path / "orig") == ["a.tif", "b.tif"]


def test_move_without_overwrite_never_replaces_a_different_file(tmp_path):
    from spacr.image_store import _move_without_overwrite

    (tmp_path / "a").write_bytes(b"one")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a").write_bytes(b"two")
    with pytest.raises(FileExistsError):
        _move_without_overwrite(tmp_path / "a", tmp_path / "d" / "a")
    assert (tmp_path / "a").read_bytes() == b"one"
    assert (tmp_path / "d" / "a").read_bytes() == b"two"
    (tmp_path / "d" / "a").write_bytes(b"one")
    assert _move_without_overwrite(tmp_path / "a", tmp_path / "d" / "a") == "already there"
    assert not (tmp_path / "a").exists()


# ---------------------------------------------------------------------------
# Preflight: letter + number wells
# ---------------------------------------------------------------------------

def _write(folder, names, rng):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        tifffile.imwrite(str(folder / name), rng.integers(0, 4000, (64, 64)).astype(np.uint16))


CQ1_REGEX = (r"(?P<wellID>W\d{4})F(?P<fieldID>\d{4})T(?P<timeID>\d{4})"
             r"Z(?P<sliceID>\d{3})C(?P<chanID>\d)")
OPERA_REGEX = r"(?P<wellID>r\d\dc\d\d)f(?P<fieldID>\d\d)p01-ch(?P<chanID>\d)sk1fk1fl1"


@pytest.mark.parametrize("names,regex,well", [
    (["W0017F0004T0001Z000C1.tif", "W0017F0004T0001Z000C2.tif"], CQ1_REGEX, "W0017"),
    (["r01c01f01p01-ch1sk1fk1fl1.tif", "r01c01f01p01-ch2sk1fk1fl1.tif"], OPERA_REGEX, "r01c01"),
])
def test_wells_that_are_not_letter_and_number_stop_mask_before_any_change(
        tmp_path, names, regex, well):
    from spacr.core import preprocess_generate_masks

    rng = np.random.default_rng(0)
    src = tmp_path / "plate"
    _write(src, names, rng)
    before = sorted(os.listdir(src))
    settings = dict(src=str(src), metadata_type="custom", custom_regex=regex,
                    channels=[0, 1], nucleus_channel=0, cell_channel=1,
                    pathogen_channel=None, plot=False)
    with pytest.raises(ValueError) as stopped:
        preprocess_generate_masks(dict(settings))
    message = str(stopped.value)
    assert well in message and "Convert the file names first" in message
    assert sorted(os.listdir(src)) == before


def test_letter_and_number_wells_pass_and_cq1_ids_pass_after_conversion(
        yokogawa_cellvoyager_dir, yokogawa_cq1_dir):
    from spacr.io import _well_name_refusal

    assert _well_name_refusal(str(yokogawa_cellvoyager_dir["src"]),
                              {"metadata_type": "cellvoyager"}) is None
    assert _well_name_refusal(str(yokogawa_cq1_dir["src"]), {"metadata_type": "cq1"}) is None
    # The npy format has no plate layout to refuse for.
    assert _well_name_refusal(str(yokogawa_cq1_dir["src"]),
                              {"metadata_type": "custom", "custom_regex": CQ1_REGEX,
                               "mask_image_format": "npy"}) is None


# ---------------------------------------------------------------------------
# Several plates in one source folder
# ---------------------------------------------------------------------------

def _two_plates(tmp_path):
    rng = np.random.default_rng(1)
    src = tmp_path / "screen"
    names = [f"{plate}_{well}_T0001F00{field}L01A01Z01C0{chan}.tif"
             for plate in ("plateA", "plateB") for well in ("A01", "B02")
             for field in (1, 2) for chan in (1, 2)]
    _write(src, names, rng)
    return src, names


def test_several_plates_stop_a_headless_run_with_the_dry_run_list(tmp_path, capsys):
    from spacr.core import preprocess_generate_masks

    src, names = _two_plates(tmp_path)
    with pytest.raises(ValueError) as stopped:
        preprocess_generate_masks(_settings(src, **_RUN))
    assert "2 plates" in str(stopped.value) and "split_plates" in str(stopped.value)
    out = capsys.readouterr().out
    assert "Dry run, nothing moved yet" in out and "plateA" in out and "plateB" in out
    assert _tiffs(src) == sorted(names)


def test_split_plates_moves_each_plate_into_its_own_project(
        tmp_path, fake_cellpose, capsys):
    from spacr.core import preprocess_generate_masks
    from spacr.image_store import _store_entries

    src, names = _two_plates(tmp_path)
    sizes = {name: (src / name).stat().st_size for name in names}
    preprocess_generate_masks(_settings(src, split_plates=True, **dict(_RUN, keep_intermediate=False)))
    out = capsys.readouterr().out
    assert out.index("Dry run, nothing moved yet") < out.index("Split 16 files into 2 plate folders")
    assert _tiffs(src) == []
    assert (src / "spacr_plate_split_plan.tsv").is_file()
    for plate in ("plateA", "plateB"):
        project = src / plate
        assert _tiffs(project) == []
        moved = _tiffs(project / "orig")
        assert moved == sorted(n for n in names if n.startswith(plate))
        assert all((project / "orig" / n).stat().st_size == sizes[n] for n in moved)
        assert len(_store_entries(project / "fov.ome.zarr")) == 4
    assert not (src / "fov.ome.zarr").exists()


def test_a_split_never_overwrites_and_moves_nothing_when_a_target_is_taken(tmp_path):
    from spacr.io import _plate_split_plan, _split_plates

    src, names = _two_plates(tmp_path)
    taken = src / "plateB" / names[-1]
    taken.parent.mkdir()
    taken.write_bytes(b"someone else's file")
    plan = _plate_split_plan(str(src), {"metadata_type": "cellvoyager"})
    with pytest.raises(RuntimeError, match="already exists"):
        _split_plates(str(src), plan, printer=lambda line: None)
    assert _tiffs(src) == sorted(names)
    assert taken.read_bytes() == b"someone else's file"


def test_a_split_cut_short_is_finished_by_running_it_again(tmp_path):
    from spacr.io import _plate_sources

    src, names = _two_plates(tmp_path)
    # Cut short: one file linked into its plate folder but not yet unlinked.
    (src / "plateA").mkdir()
    os.link(src / names[0], src / "plateA" / names[0])
    folders = _plate_sources([str(src)], {"metadata_type": "cellvoyager", "split_plates": True},
                             printer=lambda line: None)
    assert folders == [str(src / "plateA"), str(src / "plateB")]
    assert _tiffs(src) == []
    assert len(_tiffs(src / "plateA")) == 8 and len(_tiffs(src / "plateB")) == 8
    # A rerun on the emptied source runs the recorded plate folders.
    assert _plate_sources([str(src)], {"metadata_type": "cellvoyager"},
                          printer=lambda line: None) == folders


def test_the_gui_asks_before_it_splits_the_plates(tmp_path, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt.screens.app_screen import AppScreen

    src, names = _two_plates(tmp_path)
    asked = {}

    def answer(box):
        asked["text"] = box.text()
        asked["details"] = box.detailedText()
        return asked["reply"]

    monkeypatch.setattr(QMessageBox, "exec", answer)
    settings = {"src": str(src), "metadata_type": "cellvoyager"}
    asked["reply"] = QMessageBox.Cancel
    assert AppScreen._confirm_plate_split(None, settings) is False
    assert "one folder per plate" in asked["text"] and "Batch Runner" in asked["text"]
    assert "plateA" in asked["details"] and "Dry run" in asked["details"]
    assert "split_plates" not in settings and _tiffs(src) == sorted(names)
    asked["reply"] = QMessageBox.Yes
    assert AppScreen._confirm_plate_split(None, settings) is True
    assert settings["split_plates"] is True
    # Nothing moves until the run does it.
    assert _tiffs(src) == sorted(names)
    one = {"src": str(tmp_path / "none"), "metadata_type": "cellvoyager"}
    assert AppScreen._confirm_plate_split(None, one) is True
