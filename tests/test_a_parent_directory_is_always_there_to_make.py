"""``os.path.dirname(os.path.abspath(x))`` is never empty, in any of 16 places.

Every writer in spaCR that creates its destination folder had the same shape:

    target = os.path.abspath(...)
    parent = os.path.dirname(target)
    if parent:                          # <- can never be False
        os.makedirs(parent, exist_ok=True)

``abspath`` returns a rooted path, so its ``dirname`` is at least ``"/"`` (and
at least ``"C:\\"`` on Windows). The guard's False arc is unreachable, which
means one uncoverable branch per writer -- and instruction 288 does not allow
a pragma to paper over one.

The guards are gone. What is asserted here is the property that made them
dead, and then the behaviour that used to depend on them: a bare filename with
no directory part in it -- the input the guard was presumably written for --
still resolves to a real parent and still gets written.
"""
from __future__ import annotations

import os

import pytest


BARE_AND_ROOTED = ["x.png", "", ".", "/", "/a", "/a/b.png", "rel/x.png",
                   "..", "a/../b.csv", "//", "///a"]


@pytest.mark.parametrize("candidate", BARE_AND_ROOTED)
def test_the_parent_of_an_absolute_path_is_never_empty(candidate):
    """The property the removed guards were checking for.

    Parametrised over the shapes that look like they might produce nothing:
    the empty string, a bare name, the root itself, and the double-slash form
    POSIX reserves. None of them do.
    """
    parent = os.path.dirname(os.path.abspath(candidate))

    assert parent, f"abspath({candidate!r}) produced an empty parent"
    assert os.path.isabs(parent)


def test_a_bare_filename_still_writes_where_it_always_did(tmp_path,
                                                          monkeypatch):
    """The input the guard existed for, exercised through a real writer.

    A caller passing "counts.csv" with no directory in it is the case that
    makes ``dirname`` look like it could be empty. It is not -- it is the
    current working directory -- and the file has to land there.
    """
    from spacr.counting import CountingSession, LayerStack

    monkeypatch.chdir(tmp_path)
    counts = CountingSession(LayerStack(), classes=["a"])

    written = counts.to_csv("counts.csv")

    assert os.path.isabs(written)
    assert os.path.dirname(written) == str(tmp_path)
    assert os.path.isfile(written)


def test_a_nested_destination_is_still_created_on_the_way(tmp_path):
    """The other half: makedirs is still doing the work it was doing."""
    from spacr.counting import CountingSession, LayerStack

    target = tmp_path / "a" / "b" / "c" / "counts.csv"
    assert not target.parent.exists()

    written = CountingSession(LayerStack(), classes=["a"]).to_csv(str(target))

    assert os.path.isfile(written)
    assert target.parent.is_dir()


def test_the_roi_writer_creates_its_folder_too(tmp_path):
    """A second writer, because the change touched eleven files.

    One test on one of them would leave the other ten asserted only by the
    property above; this is the second independent confirmation that removing
    the guard did not remove the makedirs with it.
    """
    from spacr.roi import RoiSet

    target = tmp_path / "deep" / "rois.json"

    RoiSet().save(str(target))

    assert target.is_file()
