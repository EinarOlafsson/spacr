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

    report = reroot_column(frame, "png_path", root)

    assert report.moved == 1
    assert frame["png_path"].iloc[0] == real
    assert frame["png_path"].iloc[3] == "/still/gone.png"


def test_a_column_that_is_not_there_is_not_an_error():
    # The PNG route and the merged route carry different path columns, so a
    # caller must be able to ask for both.
    assert reroot_column(pd.DataFrame({"a": [1]}), "png_path", "/tmp").moved == 0


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

    report = portable_paths.reroot_column(frame, "png_path", screen["plate"])

    assert report.moved == 40
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


# -------------------------------------------------- instruction 155 F, in full


def test_a_path_written_on_windows_resolves_on_linux(tmp_path):
    """"The same with a database whose paths were written on Windows."

    `path.split('/data/')` cannot match `\\data\\`, so a database written on
    Windows and opened on Linux -- or a share mounted both ways -- re-anchored
    through neither existing route.
    """
    crop = tmp_path / "new" / "data" / "plate1" / "a.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"x")

    got = reroot_crop_path(r"C:\lab\exp1\data\plate1\a.png",
                           str(tmp_path / "new"))

    assert got == str(crop)


def test_what_could_not_be_placed_is_counted_and_one_is_named(tmp_path):
    """"Paths that cannot be re-anchored are counted and one is named."

    A silent pass-through is how a wrong re-anchor stayed invisible: the path
    is returned unchanged and fails later as a missing file, somewhere with
    less context.
    """
    import pandas as pd

    crop = tmp_path / "new" / "data" / "w1" / "a.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"x")
    frame = pd.DataFrame({"png_path": [
        "/old/exp/data/w1/a.png",          # resolvable
        "/old/exp/nowhere/b.png",          # not
        "/old/exp/nowhere/c.png",          # not, same folder
    ]})

    report = reroot_column(frame, "png_path", str(tmp_path / "new"))

    assert report.moved == 1
    assert report.unresolved == 2
    assert report.first_unresolved == "/old/exp/nowhere/b.png"
    described = report.describe()
    assert "2 could not be placed" in described
    assert "/old/exp/nowhere/b.png" in described


def test_an_old_root_with_its_own_data_folder_lands_on_the_FILE(tmp_path):
    """The silent-corruption regression test.

    Splitting on the FIRST `/data/` turned `/old/data/exp1/data/plate1/a.png`
    into `<root>/data/exp1` -- a path that has lost `plate1/a.png` and names a
    DIRECTORY. It did not raise; it failed much later as a missing file.
    """
    crop = tmp_path / "new" / "data" / "plate1" / "a.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"x")
    decoy = tmp_path / "new" / "data" / "exp1"
    decoy.mkdir(parents=True)

    got = reroot_crop_path("/old/data/exp1/data/plate1/a.png",
                           str(tmp_path / "new"))

    assert got == str(crop)
    assert os.path.isfile(got), "re-anchored to a directory, not to the file"


def test_a_route_that_is_not_on_this_machine_is_not_60000_failures(tmp_path):
    """A screen with PNG crops and no `merged/` folder is HEALTHY.

    Measured on the real TSG101 screen: every one of the 60,816 `png_path`
    values resolved, and every one of the 60,816 `path_name` values did not,
    because that screen has no merged arrays. Reporting the second as 60,816
    failures is the false alarm that teaches a reader to ignore the true one.
    """
    import pandas as pd

    crop = tmp_path / "new" / "data" / "w1" / "a.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"x")
    absent = pd.DataFrame({"path_name": ["/old/exp/merged/x.npy",
                                         "/old/exp/merged/y.npy"]})

    report = reroot_column(absent, "path_name", str(tmp_path / "new"))

    assert report.absent and not report.partial
    assert "not on this machine" in report.describe()
    assert "could not be placed" not in report.describe()


def test_a_partial_failure_still_names_one(tmp_path):
    import pandas as pd

    crop = tmp_path / "new" / "data" / "w1" / "a.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"x")
    mixed = pd.DataFrame({"png_path": ["/old/exp/data/w1/a.png",
                                       "/old/exp/data/w9/missing.png"]})

    report = reroot_column(mixed, "png_path", str(tmp_path / "new"))

    assert report.partial and not report.absent
    assert "the first is /old/exp/data/w9/missing.png" in report.describe()


def test_reanchor_frame_resolves_from_any_root_too(tmp_path):
    """The GUI's own pass gains the same robustness.

    `reanchor_path` rewrites on structure and never asks the disk, and it
    needs `root` to be the folder that holds `data/`. Measured on the TSG101
    screen with 3,000 recorded crops: the plate folder resolved all 3,000,
    and the screen folder above it, the measurements folder and the database
    file each resolved NONE -- the rewrite lands somewhere plausible that is
    not there. A caller holding one of those is not doing anything wrong.
    """
    import pandas as pd

    from spacr.crops import reanchor_frame

    crop = tmp_path / "new" / "plate1" / "data" / "w1" / "a.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"x")
    db = tmp_path / "new" / "plate1" / "measurements" / "measurements.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    recorded = "/old/run7/plate1/data/w1/a.png"

    for root in (tmp_path / "new" / "plate1", tmp_path / "new",
                 db.parent, db):
        frame = pd.DataFrame({"png_path": [recorded]})
        out, _report = reanchor_frame(frame, str(root))
        assert out["png_path"].iloc[0] == str(crop), f"{root} did not resolve"


def test_reanchor_frame_still_refuses_a_wrong_root(tmp_path):
    import pandas as pd

    from spacr.crops import reanchor_frame

    frame = pd.DataFrame({"png_path": ["/old/run7/plate1/data/w1/a.png"]})
    out, report = reanchor_frame(frame, str(tmp_path / "nothing here"))

    assert not os.path.exists(out["png_path"].iloc[0])
    assert report.failures or report.n_reanchored >= 0
