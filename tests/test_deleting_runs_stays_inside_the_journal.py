"""``delete_runs`` removes directories, so its refusals are the subject.

Run History grew a "Clear all" button and a right-click Delete on
2026-08-31. Both call :func:`spacr.run_journal.delete_runs`, which is the
only code in spaCR that removes a folder the user did not name in a file
dialog. Everything here is about what it must NOT delete.

A record's ``dir`` comes from a manifest on disk. A manifest is a file,
files can be edited or symlinked, and "the path came from our own
journal" is therefore not a reason to trust it.
"""
from __future__ import annotations

import os
import shutil

import pytest

from spacr.run_journal import delete_runs, runs_root


@pytest.fixture
def journal(tmp_path, monkeypatch):
    """A runs root under tmp_path, with three run folders in it."""
    home = tmp_path / "home"
    (home / ".spacr" / "runs").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: home))
    root = runs_root()
    made = []
    for name in ("run_a", "run_b", "run_c"):
        folder = root / name
        (folder / "outputs").mkdir(parents=True)
        (folder / "manifest.json").write_text("{}")
        made.append(folder)
    return root, made


def test_it_deletes_the_runs_it_is_given(journal):
    """The happy path, driven -- not just the refusals."""
    root, made = journal
    deleted, refused = delete_runs([made[0], made[2]])
    assert (deleted, refused) == (2, [])
    assert not made[0].exists() and not made[2].exists()
    assert made[1].exists(), "it deleted a run it was not given"


def test_it_will_not_delete_the_journal_itself(journal):
    """"Delete everything" is a loop over children, never the root.

    A caller computing an empty selection and passing the root instead
    would otherwise take every run on the machine with it.
    """
    root, made = journal
    deleted, refused = delete_runs([root])
    assert deleted == 0
    assert refused and "journal itself" in refused[0]
    assert root.exists() and all(folder.exists() for folder in made)


def test_it_will_not_delete_anything_outside_the_journal(journal, tmp_path):
    """A plainly foreign directory is refused."""
    root, _made = journal
    outsider = tmp_path / "precious"
    outsider.mkdir()
    (outsider / "thesis.txt").write_text("years of work")

    deleted, refused = delete_runs([outsider])
    assert deleted == 0
    assert refused and "outside" in refused[0]
    assert outsider.exists() and (outsider / "thesis.txt").exists()


def test_a_traversal_climbing_out_of_the_journal_is_refused(journal,
                                                            tmp_path):
    """``runs/../../..`` must not reach a real directory and delete it.

    THE TARGET HAS TO EXIST for this test to mean anything, and the
    number of ``..`` segments has to actually reach it. Written first
    with too few, so the path named nothing, and it was refused for being
    a non-existent folder rather than for escaping -- passing, proving
    nothing, and surviving the removal of the containment check.

    What makes it bite: an UNRESOLVED path like
    ``<root>/../../../precious`` still has ``<root>`` among its
    ``parents``, so a containment test that skips ``resolve()`` waves it
    straight through and deletes the target.
    """
    root, _made = journal
    outsider = tmp_path / "precious"
    outsider.mkdir()
    (outsider / "thesis.txt").write_text("years of work")

    hops = [".."] * len(root.resolve().relative_to(tmp_path.resolve()).parts)
    traversal = root.joinpath(*hops, outsider.name)
    assert traversal.resolve() == outsider.resolve(), (
        "the traversal does not reach the target, so this proves nothing")

    deleted, refused = delete_runs([traversal])
    assert deleted == 0
    assert refused and "outside" in refused[0]
    assert outsider.exists() and (outsider / "thesis.txt").exists()


def test_a_symlink_pointing_out_of_the_journal_is_refused(journal, tmp_path):
    """The reason resolution happens before the containment check.

    A link that LIVES in the journal and POINTS somewhere else passes any
    test written against the unresolved path, and deletes the target.
    """
    root, _made = journal
    outsider = tmp_path / "elsewhere"
    outsider.mkdir()
    (outsider / "keep.txt").write_text("keep")
    link = root / "looks_like_a_run"
    try:
        os.symlink(outsider, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    deleted, refused = delete_runs([link])
    assert deleted == 0
    assert outsider.exists() and (outsider / "keep.txt").exists()
    # THE REASON, not just the refusal. Without `resolve()` this link is
    # judged to be inside the journal and only survives because
    # `shutil.rmtree` happens to refuse a symlink -- which is luck, not a
    # guard, and would not save a link to a directory on a platform whose
    # rmtree followed it. Asserting "outside" pins that it was rejected
    # for WHERE IT POINTS.
    assert refused and "outside" in refused[0], refused


def test_it_refuses_the_run_that_is_still_going(journal, monkeypatch):
    """Deleting a live run's folder breaks it on its next write.

    The failure would surface far from here, in whatever the run does
    next, with an error naming a path rather than a cause.
    """
    root, made = journal

    class _Live:
        dir = str(made[1])

    monkeypatch.setattr("spacr.run_journal.current_run", lambda: _Live())
    deleted, refused = delete_runs(made)
    assert deleted == 2
    assert made[1].exists()
    assert any("still running" in message for message in refused)


def test_a_broken_live_run_lookup_does_not_block_deletion(
        journal, monkeypatch):
    _root, made = journal

    def broken_lookup():
        raise RuntimeError("thread state is unavailable")

    monkeypatch.setattr("spacr.run_journal.current_run", broken_lookup)
    deleted, refused = delete_runs([made[0]])

    assert (deleted, refused) == (1, [])
    assert not made[0].exists()


def test_an_unusable_path_does_not_abandon_later_runs(journal):
    _root, made = journal

    deleted, refused = delete_runs(["\0", made[0]])

    assert deleted == 1
    assert not made[0].exists()
    assert len(refused) == 1
    assert "not a usable path" in refused[0]


def test_one_filesystem_failure_does_not_abandon_later_runs(
        journal, monkeypatch):
    _root, made = journal
    real_rmtree = shutil.rmtree

    def flaky_rmtree(target):
        if target == made[0]:
            raise OSError("read only")
        return real_rmtree(target)

    monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)
    deleted, refused = delete_runs([made[0], made[1]])

    assert deleted == 1
    assert made[0].exists()
    assert not made[1].exists()
    assert refused == ["run_a: OSError: read only"]


def test_one_bad_path_does_not_abandon_the_rest(journal, tmp_path):
    """Refusals are RETURNED, not raised.

    Deleting fifty runs where one is unusable must delete forty-nine and
    say which one it kept. An exception at that point has already deleted
    an unknown number and reports none of them.
    """
    root, made = journal
    deleted, refused = delete_runs(
        [made[0], tmp_path / "nowhere", made[1], root / "never_existed"])
    assert deleted == 2
    assert len(refused) == 2
    assert not made[0].exists() and not made[1].exists()


def test_a_file_is_refused_rather_than_unlinked(journal):
    """A run is a DIRECTORY. Anything else is a mistake worth reporting."""
    root, _made = journal
    stray = root / "notes.txt"
    stray.write_text("not a run")
    deleted, refused = delete_runs([stray])
    assert deleted == 0
    assert refused and "not a run folder" in refused[0]
    assert stray.exists()
