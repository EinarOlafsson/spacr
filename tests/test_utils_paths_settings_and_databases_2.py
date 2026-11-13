"""Table-driven tests for the cleanup, path, archive/download, metadata,
mask and plotting helpers of ``spacr.utils`` (part 2 of
``test_utils_paths_settings_and_databases.py``).

Test-suite audit batch 5, group D: each test below replaces several older
coverage-chasing tests of one unit with one case table run through pytest's
``subtests`` fixture. Everything is CPU-only and offline.
"""
from __future__ import annotations

import inspect
import io
import os
import subprocess
import tarfile
import threading
import types

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from spacr import utils as U  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _npy(path, name):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, name), np.zeros((2, 2), np.uint16))


# ===========================================================================
# cleanup_pipeline_folders / delete_intermedeate_files
# ===========================================================================

def _pipeline_tree(root, merged=("a.npy",), stack=("a.npy",), orig=True):
    """A post-mask run folder: merged/ + stack/ + masks/ [+ orig/]."""
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


def test_cleanup_pipeline_folders_deletes_intermediates_only_after_a_full_merge(
        tmp_path, subtests, capsys):
    """Once merged/ holds every field of stack/, stack/, masks/, numbered
    channel folders and (unless kept) orig/ are deleted and returned, folders
    already gone are skipped, and anything else stays; with merged/ missing,
    empty or behind, nothing is deleted and verbose mode says why."""
    def numeric_folders(root):
        _pipeline_tree(root)
        for d in ("1", "2", "10"):
            _npy(os.path.join(root, d), "chan.npy")
        os.makedirs(os.path.join(root, "measurements"))
        with open(os.path.join(root, "3"), "w") as fh:  # a FILE named '3'
            fh.write("not a folder")
        return {"verbose": True}

    def stack_already_gone(root):
        os.makedirs(os.path.join(root, "merged"))
        np.save(os.path.join(root, "merged", "field1.npy"), np.zeros((4, 4), np.uint16))
        os.makedirs(os.path.join(root, "masks"))
        np.save(os.path.join(root, "masks", "field1.npy"), np.zeros((4, 4), np.uint16))
        os.makedirs(os.path.join(root, "1"))
        return {}

    def keep_original(root):
        _pipeline_tree(root)
        return {"keep_original": True, "verbose": False}

    def empty_merged(root):
        _pipeline_tree(root, merged=())
        os.makedirs(os.path.join(root, "merged"))
        return {"verbose": True}

    def merged_behind(root):
        _pipeline_tree(root, stack=("a.npy", "b.npy", "c.npy"))
        return {"verbose": True, "keep_original": True}

    def no_merged(root):
        _npy(os.path.join(root, "stack"), "a.npy")
        os.makedirs(os.path.join(root, "orig"))
        return {"verbose": True}

    cases = [
        # build, deleted basenames, kept entries, printed
        (numeric_folders, ["1", "10", "2", "masks", "orig", "stack"],
         ["merged", "measurements", "3"], []),
        (stack_already_gone, ["1", "masks"], ["merged/field1.npy"], []),
        (keep_original, ["masks", "stack"], ["orig", "merged"], []),
        (empty_merged, [], ["stack", "masks", "orig"], ["merged/ is empty"]),
        (merged_behind, [], ["stack", "masks", "orig"],
         ["keeping stack/ + masks/", "2 field(s)"]),
        (no_merged, [], ["stack", "orig"], ["no merged/ folder"]),
    ]
    for build, deleted_names, kept, printed in cases:
        with subtests.test(case=build.__name__):
            root = str(tmp_path / build.__name__)
            os.makedirs(root)
            kwargs = build(root)
            capsys.readouterr()
            deleted = U.cleanup_pipeline_folders(root, **kwargs)
            assert sorted(os.path.basename(p) for p in deleted) == deleted_names
            assert all(os.path.dirname(p) == root for p in deleted)
            for name in deleted_names:
                assert not os.path.exists(os.path.join(root, name))
            for entry in kept:
                assert os.path.exists(os.path.join(root, entry)), entry
            out = capsys.readouterr().out
            for fragment in printed:
                assert fragment in out
            if build is numeric_folders:
                assert os.path.isfile(os.path.join(root, "3"))


def _intermediate_tree(root, orig=True):
    root = str(root)
    for d in ("merged", "stack", "masks", "1", "2", "10"):
        _npy(os.path.join(root, d), "a.npy")
    if orig:
        os.makedirs(os.path.join(root, "orig"), exist_ok=True)
        with open(os.path.join(root, "orig", "raw.tif"), "w") as fh:
            fh.write("raw")
    return root


def test_delete_intermedeate_files_removes_stack_masks_and_channels_when_safe(
        tmp_path, subtests, monkeypatch, capsys):
    """With merged/ complete and an orig/ backup, stack/, masks/ and every
    numbered channel folder are deleted (each reported once, failures per
    path); without a backup, a src, a 'src' key or a complete merge nothing
    is deleted."""
    intermediates = ("stack", "masks", "1", "2", "10")

    class _NoSrcKey(dict):
        def __contains__(self, key):
            return False

    def merged_lags(root):
        os.makedirs(os.path.join(root, "merged"))
        np.save(os.path.join(root, "merged", "a.npy"), np.zeros((2, 2), np.uint16))
        os.makedirs(os.path.join(root, "stack"))
        for name in ("a.npy", "b.npy", "c.npy"):
            np.save(os.path.join(root, "stack", name), np.zeros((2, 2), np.uint16))
        os.makedirs(os.path.join(root, "orig"))
        os.makedirs(os.path.join(root, "1"))

    cases = [
        # label, build, settings factory, rmtree fails, deleted, printed fragments
        ("complete_run", _intermediate_tree, lambda r: {"src": r}, False, True,
         []),
        ("rmtree_fails", _intermediate_tree, lambda r: {"src": r}, True, False,
         ["device busy"]),
        ("no_orig_backup", lambda r: _intermediate_tree(r, orig=False), lambda r: {"src": r},
         False, False, ["does not exist"]),
        ("no_src_key", _intermediate_tree, lambda r: _NoSrcKey(src=r), False, False,
         ["No 'src' key in settings dictionary."]),
        ("merged_lags", merged_lags, lambda r: {"src": r}, False, False, []),
    ]
    for label, build, settings, rmtree_fails, deleted, printed in cases:
        with subtests.test(case=label), monkeypatch.context() as mp:
            root = str(tmp_path / label)
            build(root)
            if rmtree_fails:
                def boom(path):
                    raise OSError("device busy")
                mp.setattr(U, "shutil", types.SimpleNamespace(rmtree=boom))
            capsys.readouterr()
            U.delete_intermedeate_files(settings(root))
            out = capsys.readouterr().out
            present = [d for d in intermediates if os.path.exists(os.path.join(root, d))]
            if deleted:
                assert present == []
                assert out.count("Deleted ") == 5
                assert os.path.isdir(os.path.join(root, "merged"))
                assert os.path.isfile(os.path.join(root, "orig", "raw.tif"))
            else:
                assert "stack" in present and "1" in present
                assert "Deleted " not in out
            for fragment in printed:
                assert fragment in out
            if label == "rmtree_fails":
                assert out.count("could not be deleted") == 5
            if label == "no_orig_backup":
                assert os.path.join(root, "orig") in out

    with subtests.test(case="missing_src"):
        missing = str(tmp_path / "not_there")
        capsys.readouterr()
        U.delete_intermedeate_files({"src": missing})
        assert f"{missing} does not exist." in capsys.readouterr().out
        assert not os.path.exists(missing)


# ===========================================================================
# format_path_for_system / normalize_src_path
# ===========================================================================

def test_format_path_for_system_uses_the_os_separator_and_reports_existence(
        tmp_path, subtests, monkeypatch, capsys):
    """Paths are rewritten with the platform's separator and reported as
    found or not found; an unknown OS raises."""
    existing = tmp_path / "img.tif"
    existing.write_text("x", encoding="utf-8")
    cases = [
        ("linux_existing", "Linux", str(existing).replace(os.sep, "\\"), str(existing),
         f"Found path: {existing}"),
        ("windows_missing", "Windows", "C:/data/plate1/img.tif", "C:\\data\\plate1\\img.tif",
         "Path not found"),
    ]
    for label, system, given, expected, printed in cases:
        with subtests.test(case=label), monkeypatch.context() as mp:
            mp.setattr(U.platform, "system", lambda _s=system: _s)
            capsys.readouterr()
            got = U.format_path_for_system(given)
            assert got.endswith("img.tif")
            if system == "Windows":
                assert "/" not in got and "\\" in got
            else:
                assert got == expected
            assert printed in capsys.readouterr().out

    with subtests.test(case="unsupported_os"), monkeypatch.context() as mp:
        mp.setattr(U.platform, "system", lambda: "Plan9")
        with pytest.raises(ValueError, match="Unsupported OS: Plan9"):
            U.format_path_for_system("data/img.tif")


def test_normalize_src_path_returns_a_string_or_a_list_of_strings(subtests):
    """A path string or list passes through, a stringified list of strings
    (what a settings CSV round trip produces) becomes a real list, a literal
    list of non-strings stays a string, and any other type is refused."""
    plate1 = os.path.join("data", "plate1")
    plate2 = os.path.join("data", "plate2")
    cases = [
        ("plain_path", plate1, plate1),
        ("list", ["a", "b"], ["a", "b"]),
        ("stringified_list", repr([plate1, plate2]), [plate1, plate2]),
        ("non_string_literal", "[1, 2, 3]", "[1, 2, 3]"),
    ]
    for label, given, expected in cases:
        with subtests.test(case=label):
            got = U.normalize_src_path(given)
            assert got == expected and type(got) is type(expected)
            if isinstance(got, list):
                assert all(isinstance(x, str) for x in got)
    for bad in (3, 123):
        with subtests.test(case=f"int_{bad}"):
            with pytest.raises(ValueError, match="Invalid type for 'src': int"):
                U.normalize_src_path(bad)


# ===========================================================================
# extract_tar_bz2_files / generate_image_path_map / download_models
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


def test_extract_tar_bz2_files_unpacks_each_archive_into_its_own_folder(tmp_path, subtests,
                                                                        capsys):
    """Each <stem>.tar.bz2 is extracted into <stem>/ and reported; other files
    are ignored; a corrupt archive or one that escapes its folder is reported,
    not raised; a non-folder argument is refused."""
    def good(folder):
        _make_tar_bz2(str(folder), "sample", payload=b"hello spacr")
        (folder / "ignore_me.txt").write_text("not an archive", encoding="utf-8")

    def traversal(folder):
        with tarfile.open(folder / "unsafe.tar.bz2", "w:bz2") as tar:
            member = tarfile.TarInfo("../escape.txt")
            payload = b"must stay inside the extraction folder"
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))

    def corrupt(folder):
        (folder / "broken.tar.bz2").write_bytes(b"this is not bzip2 data at all")

    def check_good(folder, out):
        assert (folder / "sample" / "inner.txt").read_bytes() == b"hello spacr"
        assert not (folder / "ignore_me").exists()
        assert "Extracted: sample.tar.bz2" in out

    def check_traversal(folder, out):
        assert not (folder / "escape.txt").exists()
        assert not (folder.parent / "escape.txt").exists()
        assert "Failed to extract unsafe.tar.bz2" in out

    def check_corrupt(folder, out):
        assert "Failed to extract broken.tar.bz2" in out
        assert (folder / "broken").is_dir() and os.listdir(folder / "broken") == []

    for build, check in ((good, check_good), (traversal, check_traversal),
                         (corrupt, check_corrupt)):
        with subtests.test(case=build.__name__):
            folder = tmp_path / build.__name__ / "arch"
            folder.mkdir(parents=True)
            build(folder)
            capsys.readouterr()
            assert U.extract_tar_bz2_files(str(folder)) is None
            check(folder, capsys.readouterr().out)

    with subtests.test(case="not_a_folder"):
        f = tmp_path / "plain.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="not a valid folder"):
            U.extract_tar_bz2_files(str(f))


def test_generate_image_path_map_flattens_the_tree_into_prefixed_names(tmp_path, subtests,
                                                                       capsys):
    """Every image below the root maps to <root>/<sub1>_<sub2>_<name>,
    non-image files and extensions not asked for are skipped, and
    copy_images_to_consolidated copies (not moves) them into consolidated/."""
    root = tmp_path / "tree"
    (root / "sub1" / "sub2").mkdir(parents=True)
    (root / "sub1" / "a.tif").write_bytes(b"AAA")
    (root / "sub1" / "sub2" / "b.png").write_bytes(b"BBB")
    (root / "sub1" / "notes.txt").write_text("skip me", encoding="utf-8")
    full = {str(root / "sub1" / "a.tif"): str(root / "sub1_a.tif"),
            str(root / "sub1" / "sub2" / "b.png"): str(root / "sub1_sub2_b.png")}

    with subtests.test(case="default_extensions"):
        assert U.generate_image_path_map(str(root)) == full

    with subtests.test(case="png_only"):
        flat = tmp_path / "flat"
        flat.mkdir()
        (flat / "a.tif").write_bytes(b"A")
        (flat / "b.png").write_bytes(b"B")
        assert list(U.generate_image_path_map(str(flat), valid_extensions=("png",))) == \
            [str(flat / "b.png")]

    with subtests.test(case="consolidate"):
        capsys.readouterr()
        U.copy_images_to_consolidated(U.generate_image_path_map(str(root)), str(root))
        dst = root / "consolidated"
        assert sorted(os.listdir(dst)) == ["sub1_a.tif", "sub1_sub2_b.png"]
        assert (dst / "sub1_a.tif").read_bytes() == b"AAA"
        assert (dst / "sub1_sub2_b.png").read_bytes() == b"BBB"
        assert (root / "sub1" / "a.tif").is_file()
        out = capsys.readouterr().out
        assert "Consolidating images" in out and "2/2" in out


class _FakeResponse:
    def __init__(self, payload=b"weights-bytes", status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self.payload), chunk_size):
            yield self.payload[i:i + chunk_size]


def test_download_models_fetches_every_listed_file_once(tmp_path, subtests, monkeypatch):
    """Models land in <package>/resources/models streamed in full from the
    dataset's resolve URL; a folder that already holds files is returned
    without touching the network, while an existing empty folder is filled."""
    cases = [
        # label, pre-existing files (None = no folder), listed files, payload, kwargs
        ("fresh", None, ["nested/cyto.pth", "nuc.pth"], b"AB" * 6000,
         dict(repo_id="me/models", retries=3, delay=0)),
        ("empty_folder", [], ["cellpose.pth"], b"weights", {}),
        ("already_present", ["already.pth"], None, None,
         dict(repo_id="me/models", retries=2, delay=0)),
    ]
    for label, existing, listed_files, payload, kwargs in cases:
        with subtests.test(case=label), monkeypatch.context() as mp:
            pkg = tmp_path / label / "spacr"
            pkg.mkdir(parents=True)
            local = pkg / "resources" / "models"
            mp.setattr(U, "spacr_path", str(pkg / "__init__.py"))
            slept, test_thread, original_sleep = [], threading.get_ident(), U.time.sleep
            mp.setattr(U.time, "sleep", lambda s: slept.append(s)
                       if threading.get_ident() == test_thread else original_sleep(s))
            if existing is not None:
                local.mkdir(parents=True)
                for name in existing:
                    (local / name).write_bytes(b"x")
            listed, urls = [], []

            def list_files(repo_id, repo_type=None, _files=listed_files):
                if _files is None:
                    raise AssertionError("must not hit the network when models exist")
                listed.append(repo_id)
                return list(_files)

            def get(url, stream=False, _payload=payload):
                if _payload is None:
                    raise AssertionError("must not hit the network when models exist")
                urls.append(url)
                return _FakeResponse(_payload)

            mp.setattr(U, "list_repo_files", list_files)
            mp.setattr(U.requests, "get", get)
            assert U.download_models(**kwargs) == str(local)
            assert slept == []
            if listed_files is None:
                assert os.listdir(local) == ["already.pth"]
                continue
            repo = kwargs.get("repo_id", "einarolafsson/models")
            assert listed == [repo]
            assert sorted(os.listdir(local)) == sorted(os.path.basename(f) for f in listed_files)
            for f in listed_files:
                assert (local / os.path.basename(f)).read_bytes() == payload
            assert urls == [f"https://huggingface.co/datasets/{repo}/resolve/main/{f}?download=true"
                            for f in listed_files]
            # a full folder is not re-fetched
            assert U.download_models(**kwargs) == str(local)
            assert listed == [repo]


# ===========================================================================
# get_cuda_version
# ===========================================================================

def test_get_cuda_version_reads_the_nvcc_release_or_returns_none(subtests, monkeypatch):
    """'release 11.8' in nvcc's output gives '118'; no release token, a
    failing or a missing nvcc gives None."""
    def output(text):
        return lambda *a, **k: text

    def raises(exc):
        def boom(*a, **k):
            raise exc
        return boom

    cases = [
        ("release_11_8", output(b"nvcc: NVIDIA (R) Cuda compiler driver\n"
                                b"Cuda compilation tools, release 11.8, V11.8.89\n"), "118"),
        ("no_release_token", output(b"garbage output\n"), None),
        ("nvcc_fails", raises(subprocess.CalledProcessError(1, ["nvcc", "--version"])), None),
        ("nvcc_missing", raises(FileNotFoundError("nvcc")), None),
    ]
    for label, check_output, expected in cases:
        with subtests.test(case=label), monkeypatch.context() as mp:
            mp.setattr(subprocess, "check_output", check_output)
            assert U.get_cuda_version() == expected
    with subtests.test(case="this_machine"):
        version = U.get_cuda_version()
        assert version is None or version.isdigit()


# ===========================================================================
# correct_metadata / remove_outliers_by_group / group_feature_class
# ===========================================================================

def test_correct_metadata_keeps_one_column_per_metadata_key(subtests):
    """Legacy spellings (plate/row/col/column/field and their _name forms)
    are renamed to plateID/rowID/columnID/fieldID, object_name is copied to
    objectID, a doubled 'pp' plate prefix is stripped from plateID and
    prcfo, and a frame already in the vocabulary passes through."""
    cases = [
        ("short_legacy", {"row": ["A"], "col": [1], "field": [2], "v": [0.5]},
         {"rowID": ["A"], "columnID": [1], "fieldID": [2], "v": [0.5]}),
        ("name_suffixed", {"plate_name": ["p1"], "row_name": ["A"], "column_name": [7],
                           "v": [0.1]},
         {"plateID": ["p1"], "rowID": ["A"], "columnID": [7], "v": [0.1]}),
        ("column_variant", {"column": [3], "v": [1]}, {"columnID": [3], "v": [1]}),
        ("field_name", {"field_name": ["1", "2"], "v": [1, 2]},
         {"fieldID": ["1", "2"], "v": [1, 2]}),
        ("object_and_plate", {"object_name": ["o1", "o2"], "plate": ["p1", "p1"]},
         {"object_name": ["o1", "o2"], "plateID": ["p1", "p1"], "objectID": ["o1", "o2"]}),
        ("double_p_prefix", {"plateID": ["pp1", "p2"], "prcfo": ["pp1_A01_1_o1", "x"]},
         {"plateID": ["p1", "p2"], "prcfo": ["p1_A01_1_o1", "x"]}),
        ("passthrough", {"rowID": ["A"], "columnID": [1], "fieldID": [1]},
         {"rowID": ["A"], "columnID": [1], "fieldID": [1]}),
    ]
    for label, given, expected in cases:
        with subtests.test(case=label):
            out = U.correct_metadata(pd.DataFrame(given))
            assert sorted(out.columns) == sorted(expected)
            assert len(set(out.columns)) == len(out.columns)
            assert {c: out[c].tolist() for c in out.columns} == expected


def test_remove_outliers_by_group_filters_within_each_group(subtests):
    """Rows outside the IQR fence or z-score threshold of their OWN group are
    dropped; a generous threshold keeps everything; other methods raise."""
    base = list(np.linspace(10.0, 11.0, 12))
    extremes = pd.DataFrame({"grp": ["a"] * 13 + ["b"] * 13,
                             "val": base + [500.0] + base + [-500.0]})
    spread = pd.DataFrame({"grp": ["a"] * 5 + ["b"] * 5,
                           "val": [1.0, 1.1, 1.2, 1.3, 50.0] + [0.0, 25.0, 50.0, 75.0, 100.0]})
    cases = [
        ("iqr_per_group", spread, "iqr", 1.5,
         {"a": [1.0, 1.1, 1.2, 1.3], "b": [0.0, 25.0, 50.0, 75.0, 100.0]}),
        ("zscore_drops_extremes", extremes, "zscore", 2.0, {"a": base, "b": base}),
        ("zscore_keeps_all", extremes, "zscore", 50.0,
         {"a": base + [500.0], "b": base + [-500.0]}),
    ]
    for label, df, method, threshold, expected in cases:
        with subtests.test(case=label):
            out = U.remove_outliers_by_group(df, "grp", "val", method=method,
                                             threshold=threshold)
            for group, values in expected.items():
                assert out.loc[out["grp"] == group, "val"].tolist() == pytest.approx(values)
            assert len(out) == sum(len(v) for v in expected.values())
    with subtests.test(case="unknown_method"):
        with pytest.raises(ValueError, match="method must be 'iqr' or 'zscore'"):
            U.remove_outliers_by_group(extremes, "grp", "val", method="mad")


def test_group_feature_class_labels_each_feature_by_its_group(subtests):
    """Each feature is labelled with every group it names, joined by '-'
    (compartments by default); unmatched features stay None except for the
    'channel' grouping, where they are 'morphology'; the input is returned."""
    cases = [
        ("default_compartments",
         ["cell_area", "nucleus_cell_overlap", "pathogen_channel_1_mean", "plate_wide_measure"],
         {}, "compartment", ["cell", "cell-nucleus", "pathogen", None]),
        ("channel_fills_morphology",
         ["cell_channel_0_mean", "cell_area", "nucleus_channel_1_mean"],
         {"feature_groups": ["channel_0", "channel_1"], "name": "channel"}, "channel",
         ["channel_0", "morphology", "channel_1"]),
        ("custom_name", ["cell_area", "blob_count"], {"feature_groups": ["cell"], "name": "grp"},
         "grp", ["cell", None]),
    ]
    for label, features, kwargs, column, expected in cases:
        with subtests.test(case=label):
            df = pd.DataFrame({"feature": features,
                               "importance": np.arange(len(features), dtype=float)})
            out = U.group_feature_class(df, **kwargs)
            assert out is df
            assert out[column].tolist() == expected
            assert out[column].isna().sum() == expected.count(None)


# ===========================================================================
# control_filelist / convert_and_relabel_masks / mask_object_count
# ===========================================================================

def test_control_filelist_selects_files_by_well_column_or_row(tmp_path, subtests):
    """Files whose well column (default 01/02) or row letter is in the
    requested values are listed; no match gives an empty list."""
    for n in ("plate1_A01_f1.tif", "plate1_B02_f1.tif", "plate1_C03_f1.tif"):
        (tmp_path / n).write_text("", encoding="utf-8")
    cases = [
        ("default_columns", {}, ["plate1_A01_f1.tif", "plate1_B02_f1.tif"]),
        ("column_03", {"mode": "columnID", "values": ["03"]}, ["plate1_C03_f1.tif"]),
        ("column_99", {"mode": "columnID", "values": ["99"]}, []),
        ("rows_A_C", {"mode": "rowID", "values": ["A", "C"]},
         ["plate1_A01_f1.tif", "plate1_C03_f1.tif"]),
    ]
    for label, kwargs, expected in cases:
        with subtests.test(case=label):
            assert sorted(U.control_filelist(str(tmp_path), **kwargs)) == expected


def test_convert_and_relabel_masks_rewrites_int64_masks_as_uint16(tmp_path, subtests, capsys):
    """int64 .npy masks are relabelled to consecutive uint16 labels with the
    geometry kept; other dtypes and non-.npy files are skipped untouched; a
    mask with more objects than uint16 holds is left as it was and reported."""
    def big_labels():
        m = np.zeros((32, 32), dtype=np.int64)
        m[2:8, 2:8] = 70000
        m[20:26, 20:26] = 90000
        return m

    def one_label():
        m = np.zeros((8, 8), dtype=np.int64)
        m[1:3, 1:3] = 100000
        return m

    def too_many():
        m = np.zeros((512, 512), dtype=np.int64)
        m[::2, ::2] = 1  # 65536 isolated pixels: one more than uint16 holds
        return m

    def uint8():
        m = np.zeros((16, 16), dtype=np.uint8)
        m[2:6, 2:6] = 7
        return m

    cases = [
        # label, mask, converted?, labels after, printed, not printed
        ("big_labels", big_labels(), True, [0, 1, 2], "Converted m.npy", None),
        ("one_label", one_label(), True, [0, 1], "Converted m.npy", None),
        ("non_int64", uint8(), False, None, "Skipping m.npy", None),
        ("too_many_objects", too_many(), False, None, "Error: Relabeling failed for m.npy",
         "Converted m.npy"),
    ]
    for label, mask, converted, labels, printed, absent in cases:
        with subtests.test(case=label):
            folder = tmp_path / label
            folder.mkdir()
            np.save(str(folder / "m.npy"), mask)
            (folder / "notes.txt").write_text("ignore me", encoding="utf-8")
            capsys.readouterr()
            U.convert_and_relabel_masks(str(folder))
            out = np.load(str(folder / "m.npy"))
            if converted:
                assert out.dtype == np.uint16
                assert sorted(np.unique(out).tolist()) == labels
                np.testing.assert_array_equal(out != 0, mask != 0)
            else:
                assert out.dtype == mask.dtype
                np.testing.assert_array_equal(out, mask)
            assert (folder / "notes.txt").read_text(encoding="utf-8") == "ignore me"
            printed_out = capsys.readouterr().out
            assert printed in printed_out
            if absent:
                assert absent not in printed_out


def test_mask_object_count_counts_labelled_objects(subtests):
    """The number of non-zero labels, including after _filter_cp_masks
    merged a touching satellite into its neighbour."""
    two = np.zeros((10, 10), dtype=np.int32)
    two[1:3, 1:3] = 1
    two[5:7, 5:7] = 2
    for label, mask, expected in [("two_objects", two, 2),
                                  ("empty", np.zeros((5, 5), dtype=np.int32), 0)]:
        with subtests.test(case=label):
            assert U.mask_object_count(mask) == expected

    with subtests.test(case="after_merge_filter"):
        mask = np.zeros((32, 32), dtype=np.int32)
        mask[4:20, 4:20] = 1
        mask[8:11, 20:23] = 2  # small satellite -> merged at threshold 0.66
        out = U._filter_cp_masks(
            [mask], [[np.zeros(mask.shape, dtype=np.float32)]],
            filter_size=False, filter_intensity=False, minimum_size=0, maximum_size=10 ** 6,
            remove_border_objects=False, merge=True,
            batch=[np.zeros((*mask.shape, 2), dtype=np.float32)], plot=False, figuresize=2)
        assert U.mask_object_count(out[0]) == 1
        assert int((out[0] > 0).sum()) == 16 * 16 + 9


# ===========================================================================
# normalize_to_dtype and the plot helpers that display through it
# ===========================================================================

def test_normalize_to_dtype_stretches_each_channel_to_the_target_range(subtests):
    """Each channel is stretched between its percentiles (or given
    per-channel bounds) onto the target dtype's range; an unknown dtype falls
    back to the input dtype and an all-zero channel stays flat."""
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 4000, size=(16, 16, 2)).astype(np.uint16)
    varied = (np.random.default_rng(1).random((16, 16, 2)) * 4000).astype(np.uint16)
    half_empty = np.zeros((8, 8, 2), dtype=np.uint16)
    half_empty[..., 1] = (np.arange(64).reshape(8, 8) * 100 + 1).astype(np.uint16)
    bounds = [(10, 3000), (5, 3500)]
    cases = [
        # label, array, kwargs, output dtype, output max, full-range channels, flat channels
        ("unknown_dtype_uses_input", noisy, {"new_dtype": "uint32"}, np.uint16, 65535, [0, 1], []),
        ("default", varied, {}, np.uint16, 65535, [0, 1], []),
        # the values are scaled to the uint8 range but kept in the input's dtype
        ("uint8_range", varied, {"new_dtype": np.uint8}, np.uint16, 255, [0, 1], []),
        ("uint8_by_name", varied, {"new_dtype": "uint8"}, np.uint16, 255, [0, 1], []),
        ("uint16_by_name", varied, {"new_dtype": "uint16"}, np.uint16, 65535, [0, 1], []),
        ("intensity_bounds", varied, {"percentile_list": bounds}, np.uint16, 65535, [], []),
        ("all_zero_channel", half_empty, {"p1": 2, "p2": 98}, np.uint16, 65535, [1], [0]),
    ]
    for label, arr, kwargs, dtype, top, full, flat in cases:
        with subtests.test(case=label):
            with np.errstate(invalid="ignore", divide="ignore"):
                out = U.normalize_to_dtype(arr, **kwargs)
            assert out.shape == arr.shape
            assert out.dtype == dtype
            assert out.max() <= top
            for c in full:
                assert out[..., c].min() == 0
                assert out[..., c].max() == top
            for c in flat:
                assert out[..., c].min() == out[..., c].max()
            if "percentile_list" in kwargs:
                for c, (lo, hi) in enumerate(bounds):
                    expected = np.clip((arr[..., c].astype(float) - lo) / (hi - lo), 0, 1) * top
                    assert np.abs(out[..., c].astype(float) - expected).max() <= 1.0


def test_plot_helpers_display_the_normalize_to_dtype_result(tmp_path, subtests, monkeypatch,
                                                            capsys):
    """plot_arrays shows one panel per channel of normalize_to_dtype(p1, p2),
    and _normalize_and_outline without overlay returns the 0-100 percentile
    normalisation of the intensity channels only."""
    from spacr.plot import _get_colours_merged, _normalize_and_outline, plot_arrays

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    with subtests.test(case="plot_arrays"):
        plt.close("all")
        arr = np.random.default_rng(0).integers(0, 5000, size=(32, 32, 3)).astype(np.uint16)
        path = tmp_path / "stack3.npy"
        np.save(path, arr)
        capsys.readouterr()
        plot_arrays(str(path), figuresize=6, cmap="gray", normalize=True, q1=1, q2=99)
        assert f"Image path: {path}" in capsys.readouterr().out
        figs = plt.get_fignums()
        assert len(figs) == 1
        axes = plt.figure(figs[0]).axes
        assert len(axes) == 3
        expected = U.normalize_to_dtype(array=arr, p1=1, p2=99)
        for c, ax in enumerate(axes):
            assert ax.get_title() == f"Channel {c}"
            assert not ax.axison
            shown = np.asarray(ax.images[0].get_array())
            assert shown.shape == (32, 32)
            assert np.array_equal(shown, expected[:, :, c])
        assert not np.array_equal(expected, arr)

    with subtests.test(case="normalize_and_outline"):
        h = w = 48
        yy, xx = np.mgrid[:h, :w]
        stack = np.zeros((h, w, 6), dtype=np.uint16)
        stack[:, :, 0] = (yy * 37 + xx * 11 + 5).astype(np.uint16)
        stack[:, :, 1] = (xx * 53 + 7).astype(np.uint16)
        stack[:, :, 2] = ((yy + xx) * 23 + 3).astype(np.uint16)
        cell = np.zeros((h, w), dtype=np.uint16)
        cell[(yy - 12) ** 2 + (xx - 12) ** 2 <= 81] = 1
        cell[(yy - 34) ** 2 + (xx - 34) ** 2 <= 81] = 2
        nucleus = np.zeros((h, w), dtype=np.uint16)
        nucleus[(yy - 12) ** 2 + (xx - 12) ** 2 <= 16] = 1
        nucleus[(yy - 34) ** 2 + (xx - 34) ** 2 <= 16] = 2
        pathogen = np.zeros((h, w), dtype=np.uint16)
        pathogen[(yy - 9) ** 2 + (xx - 15) ** 2 <= 4] = 1
        stack[:, :, 3], stack[:, :, 4], stack[:, :, 5] = cell, nucleus, pathogen
        expected = U.normalize_to_dtype(array=stack.copy(), p1=0, p2=100)[:, :, [0, 1, 2]]
        overlayed, image, outlines = _normalize_and_outline(
            image=stack, remove_background=False, normalize=False,
            normalization_percentiles=[2, 98], overlay=False, overlay_chans=[0, 1, 2],
            mask_dims=[3, 4, 5], outline_colors=_get_colours_merged("rgb"), outline_thickness=2)
        assert overlayed == [] and outlines == []
        assert image.shape == (h, w, 3) and image.dtype == np.uint16
        np.testing.assert_array_equal(image, expected)


# ===========================================================================
# remove_canvas / plot_clusters_grid / augment_images
# ===========================================================================

def test_remove_canvas_makes_black_pixels_transparent(subtests):
    """Grayscale and RGB images become float RGBA in [0, 1] whose alpha is 1
    exactly where a pixel is non-black; other modes raise."""
    from PIL import Image

    gray = np.zeros((4, 4), np.uint8)
    gray[1:3, 1:3] = 200
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[0, 0] = (255, 0, 0)
    block = np.zeros((4, 4, 3), np.uint8)
    block[1:3, 1:3] = 100
    cases = [
        ("gray_block", Image.fromarray(gray, mode="L"), gray > 0, (1, 1), 1.0),
        ("gray_2x2", Image.fromarray(np.array([[0, 100], [200, 0]], np.uint8), "L"),
         np.array([[False, True], [True, False]]), (1, 0), 1.0),
        ("rgb_pixel", Image.fromarray(rgb, mode="RGB"), rgb.any(-1), (0, 0), 1.0),
        ("rgb_block", Image.fromarray(block, "RGB"), block.any(-1), (1, 1), 100 / 255),
    ]
    for label, image, opaque, probe, value in cases:
        with subtests.test(case=label):
            out = U.remove_canvas(image)
            assert out.shape == (*opaque.shape, 4)
            np.testing.assert_array_equal(out[..., 3], opaque.astype(out.dtype))
            assert out[probe][0] == pytest.approx(value)
    with subtests.test(case="rgba_refused"):
        with pytest.raises(ValueError, match="Unsupported image mode"):
            U.remove_canvas(Image.fromarray(np.zeros((4, 4, 4), np.uint8), mode="RGBA"))


def test_plot_clusters_grid_draws_one_panel_per_cluster_without_noise(tmp_path, subtests,
                                                                      monkeypatch, capsys):
    """Label -1 is DBSCAN noise: it never becomes a cluster; each real
    cluster gets a titled panel of image_nr thumbnails; all-noise labels
    print 'No clusters found.' and draw nothing."""
    from PIL import Image

    paths = []
    for i in range(7):
        arr = np.zeros((8, 8), np.uint8)
        arr[2:6, 2:6] = 50 + 10 * i
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(arr, mode="L").save(p)
        paths.append(str(p))
    kwargs = dict(image_nr=2, figuresize=3, black_background=False, verbose=False)

    for label, colors in (("noise_only_named_colours", ["r"]),
                          ("noise_only_rgba", np.array([[0.1, 0.2, 0.3, 1.0]]))):
        with subtests.test(case=label):
            plt.close("all")
            capsys.readouterr()
            assert U.plot_clusters_grid(embedding=np.zeros((3, 2)), labels=np.array([-1, -1, -1]),
                                        image_paths=paths[:3], colors=colors, **kwargs) is None
            assert "No clusters found." in capsys.readouterr().out
            assert plt.get_fignums() == []

    with subtests.test(case="real_grid"):
        colors = np.array([[0.9, 0.1, 0.1, 1.0], [0.1, 0.9, 0.1, 1.0], [0.1, 0.1, 0.9, 1.0]])
        fig = U.plot_clusters_grid(embedding=np.zeros((7, 2)),
                                   labels=np.array([-1, 0, 0, 0, 1, 1, 1]),
                                   image_paths=paths, colors=colors, **kwargs)
        assert isinstance(fig, plt.Figure)
        assert [len(a.child_axes) for a in fig.axes] == [2, 2]
        assert sorted(t.get_text() for t in fig.texts) == ["Cluster 0", "Cluster 1"]

    with subtests.test(case="grid_receives_clusters_only"), monkeypatch.context() as mp:
        captured = {}
        sentinel = plt.figure()

        def fake_plot_grid(cluster_images, colors, figuresize, black_background, verbose,
                           theme_colors=None):
            captured["labels"] = sorted(cluster_images)
            captured["sizes"] = [len(cluster_images[k]) for k in sorted(cluster_images)]
            captured["theme_colors"] = theme_colors
            return sentinel

        mp.setattr(U, "plot_grid", fake_plot_grid)
        fig = U.plot_clusters_grid(embedding=np.zeros((4, 2)), labels=np.array([-1, 0, 0, 1]),
                                   image_paths=paths[:4], colors=["r", "g", "b"], **kwargs)
        assert fig is sentinel
        assert captured == {"labels": [0, 1], "sizes": [2, 1], "theme_colors": None}


def test_augment_images_writes_six_orientations_per_image(tmp_path, subtests, monkeypatch):
    """Each image is saved as original, both flips and three rotations next
    to whatever the destination already holds; no images writes nothing."""
    import cv2

    monkeypatch.setattr(U, "cpu_count", lambda: 2)
    suffixes = ["flip_hor", "flip_ver", "original", "rot_180", "rot_270", "rot_90"]

    def write(path, seed):
        img = (np.random.default_rng(seed).random((16, 16, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(path), img)
        return str(path)

    with subtests.test(case="no_images"):
        dst = tmp_path / "empty_run"
        dst.mkdir()
        assert U.augment_images([], str(dst)) is None
        assert list(dst.iterdir()) == []

    with subtests.test(case="existing_destination"):
        src, dst = tmp_path / "src", tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (dst / "already_here.txt").write_text("kept", encoding="utf-8")
        U.augment_images([write(src / "cell.png", 0)], str(dst))
        assert sorted(p.name for p in dst.iterdir()) == sorted(
            ["already_here.txt"] + [f"cell_{s}.png" for s in suffixes])
        assert (dst / "already_here.txt").read_text(encoding="utf-8") == "kept"
        original = cv2.imread(str(src / "cell.png"))
        np.testing.assert_array_equal(cv2.imread(str(dst / "cell_original.png")), original)
        np.testing.assert_array_equal(cv2.imread(str(dst / "cell_flip_hor.png")),
                                      cv2.flip(original, 1))

    with subtests.test(case="new_destination_two_images"):
        paths = [write(tmp_path / f"img{i}.png", i) for i in range(2)]
        dst = tmp_path / "aug"
        U.augment_images(paths, str(dst))
        assert sorted(p.name for p in dst.glob("*.png")) == sorted(
            f"img{i}_{s}.png" for i in range(2) for s in suffixes)


# ===========================================================================
# calculate_activation_correlations / classification_metrics
# ===========================================================================

def test_calculate_activation_correlations_scores_every_channel_pair(subtests):
    """For each input/activation channel pair: Pearson r on the jointly
    finite pixels (NaN when none) and Manders M1/M2 per threshold (NaN when
    no pixel passes both); 3-D maps get a channel axis and smaller maps are
    resized up to the input without mutating the caller's tensor."""
    torch = pytest.importorskip("torch")
    from scipy.stats import pearsonr

    rng = np.random.default_rng(3)

    def tensor(a):
        return torch.tensor(a, dtype=torch.float32)

    def stat_cols(out):
        return [c for c in out.columns if c != "file_name"]

    def three_d(out, inputs, maps):
        assert [c for c in out.columns if c.endswith("_pearsons")] == [
            "channel_0_activation_0_pearsons", "channel_1_activation_0_pearsons"]
        assert len(out.columns) == 1 + 2 * 1 * (1 + 2 * 3)
        assert np.all(np.isfinite(out[stat_cols(out)].to_numpy()))

    def resized(out, inputs, maps):
        assert len(out.columns) == 1 + 3 * 1 * (1 + 2 * 3)
        vals = out[[c for c in out.columns if c.endswith("_pearsons")]].to_numpy()
        assert vals.shape == (2, 3)
        assert np.all(np.isfinite(vals)) and np.all(np.abs(vals) <= 1.0)
        assert maps.shape == (2, 1, 8, 8)

    def all_nan(out, inputs, maps):
        assert stat_cols(out)
        assert out[stat_cols(out)].isna().all().all()

    def manders_nan(threshold):
        def check(out, inputs, maps):
            assert out.loc[0, "channel_0_activation_0_pearsons"] == pytest.approx(-1.0)
            assert np.isnan(out.loc[0, f"channel_0_activation_0_{threshold}_M1"])
            assert np.isnan(out.loc[0, f"channel_0_activation_0_{threshold}_M2"])
        return check

    def agreeing(out, inputs, maps):
        assert out.loc[0, "channel_0_activation_0_50_M1"] == pytest.approx(1.0)
        assert out.loc[0, "channel_0_activation_0_50_M2"] == pytest.approx(1.0)

    def infinite(out, inputs, maps):
        assert np.isnan(out.loc[0, "channel_0_activation_0_pearsons"])
        assert np.isnan(out.loc[0, "channel_0_activation_0_15_M1"])
        assert np.isnan(out.loc[0, "channel_0_activation_0_75_M2"])

    def partial_nan(out, inputs, maps):
        expected, _ = pearsonr(inputs.flatten().numpy()[1:], maps.flatten().numpy()[1:])
        assert out.loc[0, "channel_0_activation_0_pearsons"] == pytest.approx(expected, abs=1e-6)
        assert np.isfinite(out.loc[0, "channel_0_activation_0_50_M1"])

    partial = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).clone()
    partial[0, 0, 0, 0] = float("nan")
    cases = [
        # label, inputs, maps, names, kwargs, check
        ("three_d_maps", tensor(rng.random((2, 2, 10, 10))), tensor(rng.random((2, 10, 10))),
         ["a.png", "b.png"], {}, three_d),
        ("resized_maps", tensor(rng.random((2, 3, 16, 16))), tensor(rng.random((2, 1, 8, 8))),
         ["a.png", "b.png"], {}, resized),
        ("all_nan_two_channels", torch.full((1, 2, 8, 8), float("nan")), torch.ones((1, 2, 8, 8)),
         ["nan.png"], {}, all_nan),
        ("all_nan_one_channel", torch.full((1, 1, 2, 2), float("nan")), torch.ones((1, 1, 2, 2)),
         ["empty.png"], {"manders_thresholds": [15, 75]}, all_nan),
        ("disjoint_75", torch.tensor([[[[1.0, 2.0]]]]), torch.tensor([[[[2.0, 1.0]]]]),
         ["disjoint.png"], {"manders_thresholds": [75]}, manders_nan(75)),
        ("disjoint_50", torch.tensor([[[[0.0, 1.0]]]]), torch.tensor([[[[1.0, 0.0]]]]),
         ["anti.png"], {"manders_thresholds": [50]}, manders_nan(50)),
        ("agreeing_50", torch.tensor([[[[0.0, 1.0]]]]), torch.tensor([[[[0.0, 1.0]]]]),
         ["same.png"], {"manders_thresholds": [50]}, agreeing),
        ("infinite_activation", torch.arange(36, dtype=torch.float32).reshape(1, 1, 6, 6),
         torch.full((1, 1, 6, 6), float("inf")), ["inf.png"], {}, infinite),
        ("partial_nan", partial, tensor(np.random.default_rng(11).random((1, 1, 4, 4))),
         ["partial.png"], {}, partial_nan),
    ]
    for label, inputs, maps, names, kwargs, check in cases:
        with subtests.test(case=label):
            out = U.calculate_activation_correlations(inputs, maps, names, **kwargs)
            assert list(out["file_name"]) == names
            assert len(out) == len(names)
            check(out, inputs, maps)


def test_classification_metrics_reports_one_row_per_epoch(subtests):
    """One row indexed by the epoch with accuracy, per-class accuracy (NaN
    for a class with no rows), PR-AUC (NaN for a single class, with the 0.5
    default threshold) and the loss; mismatched lengths raise."""
    torch = pytest.importorskip("torch")
    cases = [
        # label, labels, probs, loss, epoch, expected values (NaN-checked by name)
        ("two_classes", [0, 0, 1, 1, 1, 0], [0.1, 0.4, 0.6, 0.9, 0.8, 0.2], 0.5, 1,
         {"accuracy": 1.0, "neg_accuracy": 1.0, "pos_accuracy": 1.0, "prauc": 1.0,
          "loss": 0.5}),
        ("all_negative_one_wrong", [0, 0, 0], [0.10, 0.20, 0.60], 0.25, 7,
         {"accuracy": 2 / 3, "neg_accuracy": 2 / 3, "pos_accuracy": np.nan, "prauc": np.nan,
          "optimal_threshold": 0.5, "loss": 0.25}),
        ("all_negative_all_right", [0, 0], [0.01, 0.02], 0.0, 0,
         {"accuracy": 1.0, "neg_accuracy": 1.0, "pos_accuracy": np.nan}),
        ("all_positive", [1, 1, 1], [0.6, 0.7, 0.8], 0.1, 2,
         {"prauc": np.nan, "optimal_threshold": 0.5}),
    ]
    for label, labels, probs, loss, epoch, expected in cases:
        with subtests.test(case=label):
            out = U.classification_metrics(labels, probs, torch.tensor(loss), epoch=epoch)
            assert list(out.index) == [str(epoch)]
            for column, value in expected.items():
                if np.isnan(value):
                    assert np.isnan(out[column].iloc[0]), column
                else:
                    assert out[column].iloc[0] == pytest.approx(value), column
    with subtests.test(case="length_mismatch"):
        with pytest.raises(ValueError):
            U.classification_metrics([0, 1], [0.5], torch.tensor(0.1), epoch=0)


# ===========================================================================
# dense_mask_channel_positions and the mask generators that read it
# ===========================================================================

def test_dense_mask_channel_positions_follow_role_order(subtests):
    """Each numeric role channel maps to its position on the merged stack in
    MASK_CHANNEL_ROLE_ORDER (not sorted order); None or non-numeric channels
    are absent; the object generators delegate to the shared helper."""
    from spacr import object as OBJ

    for role in ("nucleus", "cell", "pathogen", "organelle"):
        assert f"{role}_channel" in U.MASK_CHANNEL_ROLE_ORDER
    cases = [
        ("role_order_not_sorted", {"nucleus_channel": 2, "cell_channel": 0,
                                   "organelle_channel": 1}, {2: 0, 0: 1, 1: 2}),
        ("none_pathogen", {"nucleus_channel": 2, "cell_channel": 0, "organelle_channel": 1,
                           "pathogen_channel": None}, {2: 0, 0: 1, 1: 2}),
        ("numeric_string", {"nucleus_channel": "2", "cell_channel": 0, "pathogen_channel": None,
                            "organelle_channel": 1}, {2: 0, 0: 1, 1: 2}),
        ("two_roles", {"nucleus_channel": 0, "cell_channel": 1}, {0: 0, 1: 1}),
        ("not_a_number", {"nucleus_channel": "later", "cell_channel": 0}, {0: 0}),
        ("none_and_uncoercible", {"nucleus_channel": None, "cell_channel": "not a number",
                                  "pathogen_channel": 1}, {1: 0}),
    ]
    for label, settings, expected in cases:
        with subtests.test(case=label):
            assert U.dense_mask_channel_positions(settings) == expected
    with subtests.test(case="generators_use_the_shared_helper"):
        for function in (OBJ.generate_cellpose_masks_sam, OBJ.generate_cellpose_masks):
            source = inspect.getsource(function)
            assert "if _raw in _dense:" not in source
            assert source.count("_fill_cellpose_channel_positions(settings)") == 1
        helper = inspect.getsource(OBJ._fill_cellpose_channel_positions)
        assert "if raw in dense:" not in helper
        assert helper.count("settings[derived] = position") == 1


def test_sam_mask_generation_verbose_prints_settings_and_dense_channels(tmp_path, monkeypatch,
                                                                       capsys):
    """Verbose generate_cellpose_masks_sam displays the stringified settings
    table once and prints the channel list resolved through
    dense_mask_channel_positions (role order: nucleus before cell)."""
    import torch
    import spacr.object as O
    from tests.cellpose_api_contract import (MISSING_CHANNEL_AXIS, configured_eval_arguments,
                                             emulate_pretrained_model, eval_arguments,
                                             init_arguments)
    from tests.conftest import check_cellpose_eval_call

    cpu = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(O.accelerator, "is_gpu", lambda: False)
    monkeypatch.setattr(O.accelerator, "cellpose_kwargs", lambda: {"gpu": False, "device": cpu})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    class _Model:
        """CellposeModel double declaring the installed 4.0.7 signature."""

        def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                     diam_mean=None, device=None, nchan=None, use_bfloat16=True):
            self.gpu = gpu
            self.pretrained_model = pretrained_model
            self.device = device  # a device routes eval through the GPU governor
            self.init_kwargs = init_arguments(locals())
            self.loaded_model = emulate_pretrained_model(pretrained_model, model_type)

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True, invert=False,
                 rescale=None, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
                 do_3D=False, anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=256, compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis)
            eval_arguments(locals())
            configured_eval_arguments(locals())
            masks, flows = [], []
            for im in (np.asarray(i) for i in x):
                m = np.zeros(im.shape[:2], dtype=np.uint16)
                m[2:8, 2:8] = 1
                m[12:18, 12:18] = 2
                masks.append(m)
                flows.append(np.zeros(im.shape[:2], dtype=np.float32))
            return masks, flows, None

    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(CellposeModel=_Model))
    src = tmp_path / "stack"
    src.mkdir()
    data = np.random.default_rng(0).integers(0, 4000, size=(2, 32, 32, 2)).astype(np.uint16)
    np.savez(src / "batch1.npz", data=data,
             filenames=np.array([f"plate1_A01_{i + 1}.npy" for i in range(2)]))

    def settings():
        return {"src": str(src), "cell_channel": 0, "nucleus_channel": 1,
                "pathogen_channel": None, "magnification": 20, "batch_size": 50,
                "verbose": True, "plot": False, "save": True, "timelapse": False, "n_jobs": 1,
                "cell_min_split_area": 0, "nucleus_min_split_area": 0,
                "pathogen_min_split_area": 0}

    shown = []
    monkeypatch.setattr(O, "display", lambda obj, *a, **k: shown.append(obj))
    O.generate_cellpose_masks_sam(str(src), settings(), "cell")

    assert len(shown) == 1
    df = shown[0]
    assert list(df.columns) == ["setting_key", "setting_value"]
    assert "src" in set(df["setting_key"])
    assert all(isinstance(v, str) for v in df["setting_value"])
    resolved = settings()
    dense = U.dense_mask_channel_positions(resolved)
    for role in ("nucleus", "cell", "pathogen"):
        raw = resolved.get(f"{role}_channel")
        if raw is not None and int(raw) in dense:
            resolved[f"cellpose_{role}_channel"] = dense[int(raw)]
    _extract, expected = U._get_cellpose_channels(resolved)
    assert str(expected["cell"]) in capsys.readouterr().out
