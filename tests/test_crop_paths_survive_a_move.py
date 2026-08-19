"""A screen that moved computer still finds its crops.

`png_list.png_path` is absolute and written at crop time. The maintainer's
rule: "if a source folder moves computer the paths still work as long as the
folder structure is upheld". Measured on the TSG101 screen -- 0 of 60,816
recorded paths existed, 60,816 of 60,816 existed once rebuilt under the plate
folder the database was opened from.

The safety half is tested as hard as the feature: a path is only ever
rewritten to somewhere a file ACTUALLY IS.
"""
import os

import pandas as pd
import pytest

from spacr.portable_paths import (reroot_column, reroot_crop_path,
                                  source_root_for_database)


@pytest.fixture()
def moved(tmp_path):
    """A crop that exists under `new/`, recorded as living under `old/`."""
    tail = os.path.join("data", "single_nucleus", "plate1_H19", "cell_png")
    new = tmp_path / "new" / tail
    new.mkdir(parents=True)
    (new / "crop.png").write_bytes(b"x")
    recorded = str(tmp_path / "old" / tail / "crop.png")
    return recorded, str(tmp_path / "new"), str(new / "crop.png")


def test_a_recorded_path_is_rebuilt_under_the_folder_it_lives_in_now(moved):
    recorded, root, real = moved
    assert not os.path.exists(recorded)

    assert reroot_crop_path(recorded, root) == real


def test_a_path_that_still_exists_is_left_exactly_as_it_is(tmp_path):
    here = tmp_path / "data" / "x.png"
    here.parent.mkdir(parents=True)
    here.write_bytes(b"x")

    # Even with a root that could rebuild it, an existing path is untouched.
    assert reroot_crop_path(str(here), str(tmp_path / "elsewhere")) == str(here)


def test_a_path_that_cannot_be_found_is_returned_UNCHANGED(tmp_path):
    # The safety rule. Rewriting to somewhere equally absent would make the
    # error name a folder the user never chose.
    recorded = "/gone/plate1/data/single_nucleus/crop.png"

    assert reroot_crop_path(recorded, str(tmp_path)) == recorded


def test_a_root_that_itself_contains_a_data_component_still_resolves(tmp_path):
    tail = os.path.join("data", "single_nucleus", "crop.png")
    new = tmp_path / "new" / tail
    new.parent.mkdir(parents=True)
    new.write_bytes(b"x")
    # The RECORDED root has its own `data` component, so splitting on the
    # first one would keep `proj/plate1/data` in the tail.
    recorded = f"/nas/data/proj/plate1/{tail}"

    assert reroot_crop_path(recorded, str(tmp_path / "new")) == str(new)


def test_reroot_column_reports_how_many_moved_and_leaves_the_rest(moved):
    recorded, root, real = moved
    frame = pd.DataFrame({"png_path": [recorded, "", None, "/still/gone.png"]})

    moved_count = reroot_column(frame, "png_path", root)

    assert moved_count == 1
    assert frame["png_path"].iloc[0] == real
    assert frame["png_path"].iloc[3] == "/still/gone.png"


def test_a_column_that_is_not_there_is_not_an_error():
    # The PNG route and the merged route carry different path columns, so a
    # caller must be able to ask for both.
    assert reroot_column(pd.DataFrame({"a": [1]}), "png_path", "/tmp") == 0


def test_the_root_of_a_database_is_the_plate_folder_that_holds_data():
    got = source_root_for_database("/screens/plate1/measurements/measurements.db")

    assert got == "/screens/plate1"


def test_no_root_means_no_rewrite():
    assert reroot_crop_path("/gone/data/x.png", "") == "/gone/data/x.png"


def test_the_dataset_generator_and_the_montage_share_ONE_rule(tmp_path):
    """`io`'s generator and the montage must resolve a moved path identically.

    The rule was a nested local inside `io.generate_dataset`, reachable only
    from that generator -- so a screen that had moved computer had its crops
    resolved for a training set and NOT for the montage, which showed 60,816
    dead paths. Two copies of a rule is how they come to disagree, so this
    pins that there is one.
    """
    import inspect

    from spacr import io

    source = inspect.getsource(io)
    assert "from .portable_paths import reroot_crop_path" in source, (
        "io re-implemented the rebuild instead of sharing it")
    # And the shared rule is the one that decides: a rebuild that lands
    # nowhere leaves the recorded path alone.
    assert reroot_crop_path("/gone/plate1/data/x.png", str(tmp_path)) == \
        "/gone/plate1/data/x.png"


# ---------------------------------------------------------------- robustness
# "just make it work robustly and dont change the database" -- 2026-08-19.
# A caller holds whichever root it happens to have, so all of them resolve.


@pytest.fixture()
def screen(tmp_path):
    """A screen at `new/`, with crops recorded as living under `old/`."""
    tail = os.path.join("data", "single_nucleus", "plate1_H19", "cell_png")
    crop = tmp_path / "new" / "plate1" / tail / "crop.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"x")
    db = tmp_path / "new" / "plate1" / "measurements" / "measurements.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    recorded = str(tmp_path / "old" / "run7" / "plate1" / tail / "crop.png")
    return {"recorded": recorded, "crop": str(crop), "db": str(db),
            "plate": str(tmp_path / "new" / "plate1"),
            "measurements": str(db.parent),
            "screen": str(tmp_path / "new")}


@pytest.mark.parametrize("root_key", ["plate", "measurements", "db", "screen"])
def test_every_root_a_caller_might_hold_resolves(screen, root_key):
    got = reroot_crop_path(screen["recorded"], screen[root_key])

    assert got == screen["crop"], f"the {root_key} root did not resolve"


def test_a_root_with_nothing_under_it_changes_nothing(tmp_path, screen):
    assert reroot_crop_path(screen["recorded"], str(tmp_path)) == \
        screen["recorded"]


def test_the_database_file_is_never_written(screen):
    import hashlib

    before = hashlib.md5(open(screen["db"], "rb").read()).hexdigest()
    reroot_crop_path(screen["recorded"], screen["db"])
    after = hashlib.md5(open(screen["db"], "rb").read()).hexdigest()

    assert before == after


def test_a_column_resolves_once_and_reuses_the_prefix(screen, monkeypatch):
    """60,816 rows must not cost 60,816 filesystem searches."""
    import pandas as pd

    from spacr import portable_paths

    calls = {"n": 0}
    real = portable_paths._reroot_with_prefix

    def counted(path, root):
        calls["n"] += 1
        return real(path, root)

    monkeypatch.setattr(portable_paths, "_reroot_with_prefix", counted)
    frame = pd.DataFrame({"png_path": [screen["recorded"]] * 40})

    moved = portable_paths.reroot_column(frame, "png_path", screen["plate"])

    assert moved == 40
    assert calls["n"] == 1, (
        f"searched {calls['n']} times for one prefix; the map is not reused")


def test_a_root_that_resolves_nothing_searches_each_FOLDER_once(screen,
                                                                monkeypatch):
    import pandas as pd

    from spacr import portable_paths

    calls = {"n": 0}
    real = portable_paths._reroot_with_prefix

    def counted(path, root):
        calls["n"] += 1
        return real(path, root)

    monkeypatch.setattr(portable_paths, "_reroot_with_prefix", counted)
    frame = pd.DataFrame({"png_path": [screen["recorded"]] * 40})

    portable_paths.reroot_column(frame, "png_path", screen["screen"] + "/nope")

    assert calls["n"] == 1, (
        "an unresolvable folder was searched again for every row in it")


def test_the_crop_source_falls_back_to_the_verifying_resolver(screen):
    """`reanchor_path` rewrites on structure and never checks the disk."""
    from spacr.crops import PngCropSource

    got = PngCropSource(root=screen["screen"]).resolve(screen["recorded"])

    assert got == screen["crop"]
    assert os.path.exists(got)
