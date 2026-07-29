"""CPU-only coverage for the tail of spacr.utils: pipeline-folder cleanup,
path normalisation, image consolidation and metadata/outlier helpers.

Covers ``cleanup_pipeline_folders``, ``delete_intermedeate_files``,
``filter_and_save_csv``, ``extract_tar_bz2_files``,
``calculate_shortest_distance``, ``format_path_for_system``,
``normalize_src_path``, ``generate_image_path_map``,
``copy_images_to_consolidated``, ``correct_metadata`` and
``remove_outliers_by_group``.

Everything here is filesystem/DataFrame work on tmp_path — no network, no CUDA,
no plotting.
"""
from __future__ import annotations

import os
import io
import tarfile
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _no_figures():
    """Guarantee no matplotlib figure survives a test in this module."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _npy(path, name):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, name), np.zeros((2, 2), np.uint16))


def _pipeline_tree(root, merged=("a.npy",), stack=("a.npy",), orig=True):
    """Build a minimal post-mask run folder: merged/ + stack/ + masks/ [+ orig/]."""
    root = str(root)
    os.makedirs(root, exist_ok=True)
    for name in merged:
        _npy(os.path.join(root, "merged"), name)
    for name in stack:
        _npy(os.path.join(root, "stack"), name)
    os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    _npy(os.path.join(root, "masks", "cell_mask_stack"), "a.npy")
    if orig:
        os.makedirs(os.path.join(root, "orig"), exist_ok=True)
        with open(os.path.join(root, "orig", "raw.tif"), "w") as fh:
            fh.write("raw")
    return root


# ===========================================================================
# cleanup_pipeline_folders — verbose branches + numeric channel folders
# ===========================================================================

def test_cleanup_verbose_reports_empty_merged_and_keeps_everything(tmp_path, capsys):
    """merged/ present but holding no .npy → nothing may be deleted, and the
    verbose branch says so."""
    from spacr.utils import cleanup_pipeline_folders

    root = _pipeline_tree(tmp_path / "p1", merged=(), stack=("a.npy",))
    os.makedirs(os.path.join(root, "merged"), exist_ok=True)  # empty dir

    deleted = cleanup_pipeline_folders(root, verbose=True)

    assert deleted == []
    out = capsys.readouterr().out
    assert "merged/ is empty" in out
    assert os.path.isdir(os.path.join(root, "stack"))
    assert os.path.isdir(os.path.join(root, "masks"))
    assert os.path.isdir(os.path.join(root, "orig"))


def test_cleanup_without_merged_folder_is_a_verbose_noop(tmp_path, capsys):
    """No merged/ at all → refuse to touch anything and say why."""
    from spacr.utils import cleanup_pipeline_folders

    root = str(tmp_path / "p1")
    _npy(os.path.join(root, "stack"), "a.npy")
    os.makedirs(os.path.join(root, "orig"), exist_ok=True)

    deleted = cleanup_pipeline_folders(root, verbose=True)

    assert deleted == []
    assert "no merged/ folder" in capsys.readouterr().out
    assert os.path.isdir(os.path.join(root, "stack"))
    assert os.path.isdir(os.path.join(root, "orig"))


def test_cleanup_verbose_reports_unmerged_field_count(tmp_path, capsys):
    """stack/ holds 2 fields that never reached merged/ → guard fires and the
    verbose message names how many fields are missing."""
    from spacr.utils import cleanup_pipeline_folders

    root = _pipeline_tree(tmp_path / "p1", merged=("a.npy",),
                          stack=("a.npy", "b.npy", "c.npy"))

    deleted = cleanup_pipeline_folders(root, verbose=True, keep_original=True)

    assert deleted == []
    out = capsys.readouterr().out
    assert "keeping stack/ + masks/" in out
    assert "2 field(s)" in out
    assert os.path.isdir(os.path.join(root, "stack"))


def test_cleanup_removes_numeric_channel_folders_only(tmp_path):
    """The per-channel folders 1/, 2/, 10/ go with stack/ + masks/; non-numeric
    folders and same-named *files* are left alone."""
    from spacr.utils import cleanup_pipeline_folders

    root = _pipeline_tree(tmp_path / "p1")
    for d in ("1", "2", "10"):
        _npy(os.path.join(root, d), "chan.npy")
    os.makedirs(os.path.join(root, "measurements"), exist_ok=True)
    with open(os.path.join(root, "3"), "w") as fh:       # a FILE named '3'
        fh.write("not a folder")

    deleted = cleanup_pipeline_folders(root, verbose=True)

    basenames = sorted(os.path.basename(p) for p in deleted)
    assert basenames == ["1", "10", "2", "masks", "orig", "stack"]
    for d in ("1", "2", "10", "stack", "masks", "orig"):
        assert not os.path.exists(os.path.join(root, d))
    assert os.path.isdir(os.path.join(root, "merged"))
    assert os.path.isdir(os.path.join(root, "measurements"))
    assert os.path.isfile(os.path.join(root, "3"))


def test_cleanup_keep_original_retains_orig_but_drops_stack(tmp_path):
    from spacr.utils import cleanup_pipeline_folders

    root = _pipeline_tree(tmp_path / "p1")
    deleted = cleanup_pipeline_folders(root, keep_original=True, verbose=False)

    assert os.path.join(root, "orig") not in deleted
    assert os.path.isdir(os.path.join(root, "orig"))
    assert not os.path.exists(os.path.join(root, "stack"))


# ===========================================================================
# delete_intermedeate_files
# ===========================================================================
#
# The guard counts the FILES in merged/ against those in stack/ and only
# deletes the intermediates once merged/ is at least as populated. (It used to
# compare the lengths of the two path STRINGS, which differ by exactly one
# character for every conceivable src, so the body was unreachable; the tests
# below drive the real body directly now that it is live.)

def _intermediate_tree(root, orig=True):
    root = str(root)
    for d in ("merged", "stack", "masks", "1", "2", "10"):
        _npy(os.path.join(root, d), "a.npy")
    if orig:
        os.makedirs(os.path.join(root, "orig"), exist_ok=True)
        with open(os.path.join(root, "orig", "raw.tif"), "w") as fh:
            fh.write("raw")
    return root


def test_delete_intermedeate_files_removes_intermediates(tmp_path):
    """With merged/ built and orig/ present, stack/ + the per-channel folders
    must be removed."""
    from spacr.utils import delete_intermedeate_files

    root = _intermediate_tree(tmp_path / "p1")
    delete_intermedeate_files({"src": root})

    assert not os.path.exists(os.path.join(root, "stack"))
    assert not os.path.exists(os.path.join(root, "masks"))
    assert not os.path.exists(os.path.join(root, "1"))
    assert os.path.isdir(os.path.join(root, "merged"))
    assert os.path.isdir(os.path.join(root, "orig"))


def test_delete_intermedeate_files_body_deletes_stack_and_channels(tmp_path, capsys):
    """Deletion body: stack/, masks/ and every numeric channel folder go;
    merged/ and orig/ stay, and each removal is reported once."""
    from spacr.utils import delete_intermedeate_files

    root = _intermediate_tree(tmp_path / "p1")
    delete_intermedeate_files({"src": root})

    for d in ("stack", "masks", "1", "2", "10"):
        assert not os.path.exists(os.path.join(root, d)), d
    assert os.path.isdir(os.path.join(root, "merged"))
    assert os.path.isfile(os.path.join(root, "orig", "raw.tif"))
    out = capsys.readouterr().out
    assert out.count("Deleted ") == 5


def test_delete_intermedeate_files_requires_orig_backup(tmp_path, capsys):
    """No orig/ backup → refuse to delete and say which path is missing."""
    from spacr.utils import delete_intermedeate_files

    root = _intermediate_tree(tmp_path / "p1", orig=False)
    delete_intermedeate_files({"src": root})

    assert os.path.isdir(os.path.join(root, "stack"))
    assert os.path.isdir(os.path.join(root, "1"))
    out = capsys.readouterr().out
    assert os.path.join(root, "orig") in out
    assert "does not exist" in out


def test_delete_intermedeate_files_missing_src_dir(tmp_path, capsys):
    from spacr.utils import delete_intermedeate_files

    missing = str(tmp_path / "not_there")
    delete_intermedeate_files({"src": missing})

    out = capsys.readouterr().out
    assert f"{missing} does not exist." in out
    assert not os.path.exists(missing)


def test_delete_intermedeate_files_reports_oserror(tmp_path, monkeypatch, capsys):
    """A failing rmtree is reported per-path and does not abort the loop."""
    import spacr.utils as su
    from spacr.utils import delete_intermedeate_files

    root = _intermediate_tree(tmp_path / "p1")
    def _boom(path):
        raise OSError("device busy")

    monkeypatch.setattr(su, "shutil", types.SimpleNamespace(rmtree=_boom))

    delete_intermedeate_files({"src": root})

    out = capsys.readouterr().out
    assert out.count("could not be deleted") == 5
    assert "device busy" in out
    # nothing was actually removed
    assert os.path.isdir(os.path.join(root, "stack"))


def test_delete_intermedeate_files_without_src_key(tmp_path, capsys):
    """The 'no src key' guard: injected via a mapping whose __contains__ lies
    (a plain dict would already have raised KeyError on the first join)."""
    from spacr.utils import delete_intermedeate_files

    root = _intermediate_tree(tmp_path / "p1")
    class _NoSrcKey(dict):
        def __contains__(self, key):
            return False

    delete_intermedeate_files(_NoSrcKey(src=root))

    out = capsys.readouterr().out
    assert "No 'src' key in settings dictionary." in out
    assert os.path.isdir(os.path.join(root, "stack"))


# ===========================================================================
# filter_and_save_csv
# ===========================================================================

def test_filter_and_save_csv_keeps_tails_only(tmp_path, capsys):
    from spacr.utils import filter_and_save_csv

    src = tmp_path / "in.csv"
    dst = tmp_path / "out.csv"
    pd.DataFrame({"score": [-5.0, -0.5, 0.0, 2.0, 9.0],
                  "name": list("abcde")}).to_csv(src, index=False)

    filter_and_save_csv(str(src), str(dst), "score",
                        upper_threshold=1.0, lower_threshold=-1.0)

    assert dst.is_file()
    out_df = pd.read_csv(dst)
    assert sorted(out_df["score"].tolist()) == [-5.0, 2.0, 9.0]
    assert sorted(out_df["name"].tolist()) == ["a", "d", "e"]
    assert list(out_df.columns) == ["score", "name"]
    assert f"Filtered DataFrame saved to {dst}" in capsys.readouterr().out


def test_filter_and_save_csv_can_produce_empty_output(tmp_path):
    from spacr.utils import filter_and_save_csv

    src = tmp_path / "in.csv"
    dst = tmp_path / "out.csv"
    pd.DataFrame({"score": [0.0, 0.1, -0.2]}).to_csv(src, index=False)

    filter_and_save_csv(str(src), str(dst), "score", 5.0, -5.0)

    assert pd.read_csv(dst).empty


# ===========================================================================
# extract_tar_bz2_files
# ===========================================================================

def _make_tar_bz2(folder, stem, payload=b"hello spacr"):
    member = os.path.join(folder, f"{stem}_member.txt")
    with open(member, "wb") as fh:
        fh.write(payload)
    archive = os.path.join(folder, f"{stem}.tar.bz2")
    with tarfile.open(archive, "w:bz2") as tar:
        tar.add(member, arcname="inner.txt")
    os.remove(member)
    return archive


def test_extract_tar_bz2_files_rejects_non_folder(tmp_path):
    from spacr.utils import extract_tar_bz2_files

    f = tmp_path / "plain.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="not a valid folder"):
        extract_tar_bz2_files(str(f))


def test_extract_tar_bz2_files_extracts_into_named_subfolder(tmp_path, capsys):
    from spacr.utils import extract_tar_bz2_files

    folder = tmp_path / "arch"
    folder.mkdir()
    _make_tar_bz2(str(folder), "sample", payload=b"hello spacr")
    (folder / "ignore_me.txt").write_text("not an archive")

    extract_tar_bz2_files(str(folder))

    extracted = folder / "sample" / "inner.txt"
    assert extracted.is_file()
    assert extracted.read_bytes() == b"hello spacr"
    # the non-archive file was untouched and got no folder of its own
    assert not (folder / "ignore_me").exists()
    assert "Extracted: sample.tar.bz2" in capsys.readouterr().out


def test_extract_tar_bz2_files_reports_corrupt_archive(tmp_path, capsys):
    """A truncated/garbage .tar.bz2 is reported, not raised."""
    from spacr.utils import extract_tar_bz2_files

    folder = tmp_path / "arch"
    folder.mkdir()
    (folder / "broken.tar.bz2").write_bytes(b"this is not bzip2 data at all")

    extract_tar_bz2_files(str(folder))

    out = capsys.readouterr().out
    assert "Failed to extract broken.tar.bz2" in out
    # the destination folder is created before the failure, and stays empty
    assert (folder / "broken").is_dir()
    assert os.listdir(folder / "broken") == []


def test_extract_tar_bz2_files_rejects_path_traversal(tmp_path, capsys):
    from spacr.utils import extract_tar_bz2_files

    folder = tmp_path / "arch"
    folder.mkdir()
    archive = folder / "unsafe.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        member = tarfile.TarInfo("../escape.txt")
        payload = b"must stay inside the extraction folder"
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))

    extract_tar_bz2_files(str(folder))

    assert not (folder / "escape.txt").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert "Failed to extract unsafe.tar.bz2" in capsys.readouterr().out


# ===========================================================================
# calculate_shortest_distance
# ===========================================================================

def test_calculate_shortest_distance_edge_to_edge_and_clamped():
    from spacr.utils import calculate_shortest_distance

    df = pd.DataFrame({
        "pathogen_channel_0_centroid_weighted-0": [0.0, 0.0],
        "pathogen_channel_0_centroid_weighted-1": [0.0, 0.0],
        "nucleus_channel_0_centroid_weighted-0": [30.0, 3.0],
        "nucleus_channel_0_centroid_weighted-1": [40.0, 4.0],
        "pathogen_feret_diameter_max": [10.0, 100.0],
        "nucleus_feret_diameter_max": [20.0, 100.0],
    })

    out = calculate_shortest_distance(df, "pathogen", "nucleus")

    col = "pathogen_nucleus_shortest_distance"
    assert col in out.columns
    # row 0: centroid distance 50 - (5 + 10) = 35
    assert out[col].iloc[0] == pytest.approx(35.0)
    # row 1: centroid distance 5 - (50 + 50) < 0 → clamped to 0
    assert out[col].iloc[1] == pytest.approx(0.0)
    assert (out[col] >= 0).all()
    # the helper mutates and returns the same frame
    assert out is df


# ===========================================================================
# format_path_for_system
# ===========================================================================

def test_format_path_for_system_windows_branch(monkeypatch, capsys):
    import spacr.utils as su

    monkeypatch.setattr(su.platform, "system", lambda: "Windows")
    got = su.format_path_for_system("C:/data/plate1/img.tif")

    assert "/" not in got
    assert got.endswith("img.tif")
    assert "\\" in got
    assert "Path not found" in capsys.readouterr().out


def test_format_path_for_system_unsupported_os_raises(monkeypatch):
    import spacr.utils as su

    monkeypatch.setattr(su.platform, "system", lambda: "Plan9")
    with pytest.raises(ValueError, match="Unsupported OS: Plan9"):
        su.format_path_for_system("/data/img.tif")


def test_format_path_for_system_reports_existing_path(tmp_path, monkeypatch, capsys):
    import spacr.utils as su

    monkeypatch.setattr(su.platform, "system", lambda: "Linux")
    p = tmp_path / "img.tif"
    p.write_text("x")
    got = su.format_path_for_system(str(p).replace(os.sep, "\\"))

    assert got == str(p)
    assert f"Found path: {p}" in capsys.readouterr().out


# ===========================================================================
# normalize_src_path
# ===========================================================================

def test_normalize_src_path_parses_list_literal():
    from spacr.utils import normalize_src_path

    got = normalize_src_path("['/data/plate1', '/data/plate2']")
    assert got == ["/data/plate1", "/data/plate2"]
    assert all(isinstance(x, str) for x in got)


def test_normalize_src_path_non_string_list_literal_stays_a_string():
    """A literal list of NON-strings is not a valid src list → kept verbatim."""
    from spacr.utils import normalize_src_path

    assert normalize_src_path("[1, 2, 3]") == "[1, 2, 3]"


def test_normalize_src_path_plain_path_unchanged():
    from spacr.utils import normalize_src_path

    assert normalize_src_path("/data/plate1") == "/data/plate1"
    assert normalize_src_path(["/a", "/b"]) == ["/a", "/b"]


def test_normalize_src_path_rejects_other_types():
    from spacr.utils import normalize_src_path

    with pytest.raises(ValueError, match="Invalid type for 'src': int"):
        normalize_src_path(123)


# ===========================================================================
# generate_image_path_map + copy_images_to_consolidated
# ===========================================================================

def _nested_image_tree(root):
    (root / "sub1").mkdir()
    (root / "sub1" / "sub2").mkdir()
    (root / "sub1" / "a.tif").write_bytes(b"AAA")
    (root / "sub1" / "sub2" / "b.png").write_bytes(b"BBB")
    (root / "sub1" / "notes.txt").write_text("skip me")
    return root


def test_generate_image_path_map_embeds_folder_hierarchy(tmp_path):
    from spacr.utils import generate_image_path_map

    root = _nested_image_tree(tmp_path)
    mapping = generate_image_path_map(str(root))

    assert set(mapping) == {
        str(root / "sub1" / "a.tif"),
        str(root / "sub1" / "sub2" / "b.png"),
    }
    assert mapping[str(root / "sub1" / "a.tif")] == str(root / "sub1_a.tif")
    assert (mapping[str(root / "sub1" / "sub2" / "b.png")]
            == str(root / "sub1_sub2_b.png"))


def test_generate_image_path_map_honours_valid_extensions(tmp_path):
    from spacr.utils import generate_image_path_map

    (tmp_path / "a.tif").write_bytes(b"A")
    (tmp_path / "b.png").write_bytes(b"B")
    mapping = generate_image_path_map(str(tmp_path), valid_extensions=("png",))

    assert list(mapping) == [str(tmp_path / "b.png")]


def test_copy_images_to_consolidated_copies_and_renames(tmp_path, capsys):
    from spacr.utils import copy_images_to_consolidated, generate_image_path_map

    root = _nested_image_tree(tmp_path)
    mapping = generate_image_path_map(str(root))

    copy_images_to_consolidated(mapping, str(root))

    dst = root / "consolidated"
    assert sorted(os.listdir(dst)) == ["sub1_a.tif", "sub1_sub2_b.png"]
    assert (dst / "sub1_a.tif").read_bytes() == b"AAA"
    assert (dst / "sub1_sub2_b.png").read_bytes() == b"BBB"
    # originals are copies, not moves
    assert (root / "sub1" / "a.tif").is_file()
    out = capsys.readouterr().out
    assert "Consolidating images" in out
    assert "2/2" in out


def test_copy_images_to_consolidated_creates_folder_for_empty_map(tmp_path, capsys):
    from spacr.utils import copy_images_to_consolidated

    copy_images_to_consolidated({}, str(tmp_path))

    assert (tmp_path / "consolidated").is_dir()
    assert os.listdir(tmp_path / "consolidated") == []
    assert capsys.readouterr().out == ""


# ===========================================================================
# correct_metadata
# ===========================================================================

def test_correct_metadata_strips_double_p_plate_prefix():
    from spacr.utils import correct_metadata

    df = pd.DataFrame({"plateID": ["pp1", "p2"], "prcfo": ["pp1_A01_1_o1", "x"]})
    out = correct_metadata(df)

    assert out["plateID"].tolist() == ["p1", "p2"]
    assert out["prcfo"].tolist() == ["p1_A01_1_o1", "x"]


def test_correct_metadata_promotes_object_and_plate_columns():
    from spacr.utils import correct_metadata

    df = pd.DataFrame({"object_name": ["o1", "o2"], "plate": ["p1", "p1"]})
    out = correct_metadata(df)

    assert out["objectID"].tolist() == ["o1", "o2"]
    assert out["plateID"].tolist() == ["p1", "p1"]
    # source columns are kept, not renamed
    assert "object_name" in out.columns and "plate" in out.columns


def test_correct_metadata_renames_short_legacy_names():
    from spacr.utils import correct_metadata

    df = pd.DataFrame({"row": ["A"], "col": [1], "field": [2], "v": [0.5]})
    out = correct_metadata(df)

    assert set(out.columns) == {"rowID", "columnID", "fieldID", "v"}
    assert out["rowID"].tolist() == ["A"]
    assert out["columnID"].tolist() == [1]
    assert out["fieldID"].tolist() == [2]


def test_correct_metadata_handles_name_suffixed_legacy_columns():
    """plate_name/row_name/column_name are the older export naming."""
    from spacr.utils import correct_metadata

    df = pd.DataFrame({"plate_name": ["p1"], "row_name": ["A"],
                       "column_name": [7], "v": [0.1]})
    out = correct_metadata(df)

    assert out["plateID"].tolist() == ["p1"]
    assert out["rowID"].tolist() == ["A"]
    assert out["columnID"].tolist() == [7]
    assert "row_name" not in out.columns and "column_name" not in out.columns
    # plate_name is copied (not renamed), so it survives alongside plateID
    assert "plate_name" in out.columns


def test_correct_metadata_renames_column_variant():
    from spacr.utils import correct_metadata

    df = pd.DataFrame({"column": [3], "v": [1]})
    out = correct_metadata(df)

    assert "columnID" in out.columns and "column" not in out.columns
    assert out["columnID"].tolist() == [3]


def test_correct_metadata_field_name_produces_single_field_id():
    from spacr.utils import correct_metadata

    df = pd.DataFrame({"field_name": ["1", "2"], "v": [1, 2]})
    out = correct_metadata(df)

    assert list(out.columns).count("fieldID") == 1
    assert out["fieldID"].tolist() == ["1", "2"]


def test_correct_metadata_passthrough_without_legacy_columns():
    from spacr.utils import correct_metadata

    df = pd.DataFrame({"rowID": ["A"], "columnID": [1], "fieldID": [1]})
    out = correct_metadata(df)

    assert list(out.columns) == ["rowID", "columnID", "fieldID"]
    assert len(out) == 1


# ===========================================================================
# remove_outliers_by_group
# ===========================================================================

def _outlier_frame():
    base = list(np.linspace(10.0, 11.0, 12))
    return pd.DataFrame({
        "grp": ["a"] * 13 + ["b"] * 13,
        "val": base + [500.0] + base + [-500.0],
    })


def test_remove_outliers_by_group_zscore_drops_extremes():
    from spacr.utils import remove_outliers_by_group

    df = _outlier_frame()
    out = remove_outliers_by_group(df, "grp", "val", method="zscore", threshold=2.0)

    assert 500.0 not in out["val"].values
    assert -500.0 not in out["val"].values
    assert len(out) == 24
    assert set(out["grp"]) == {"a", "b"}


def test_remove_outliers_by_group_zscore_threshold_keeps_everything():
    from spacr.utils import remove_outliers_by_group

    df = _outlier_frame()
    out = remove_outliers_by_group(df, "grp", "val", method="zscore", threshold=50.0)

    assert len(out) == len(df)
    assert 500.0 in out["val"].values


def test_remove_outliers_by_group_iqr_is_per_group():
    """Group 'b' has a wide spread, so a value that is an outlier in 'a' is
    perfectly normal in 'b'."""
    from spacr.utils import remove_outliers_by_group

    df = pd.DataFrame({
        "grp": ["a"] * 5 + ["b"] * 5,
        "val": [1.0, 1.1, 1.2, 1.3, 50.0] + [0.0, 25.0, 50.0, 75.0, 100.0],
    })
    out = remove_outliers_by_group(df, "grp", "val", method="iqr", threshold=1.5)

    a_vals = out.loc[out["grp"] == "a", "val"].tolist()
    b_vals = out.loc[out["grp"] == "b", "val"].tolist()
    assert 50.0 not in a_vals
    assert len(a_vals) == 4
    assert b_vals == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_remove_outliers_by_group_rejects_unknown_method():
    from spacr.utils import remove_outliers_by_group

    df = _outlier_frame()
    with pytest.raises(ValueError, match="method must be 'iqr' or 'zscore'"):
        remove_outliers_by_group(df, "grp", "val", method="mad")
