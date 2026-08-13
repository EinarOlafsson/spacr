"""The provenance baseline covers where a run can WRITE, and nowhere else.

Instruction 47, found while chasing the 45-second Qt test it names.

``_capture_initial_provenance`` used to choose its inventory root as::

    root = path if path.is_dir() else path.parent

so any path-valued setting naming a FILE dragged in the whole directory that
file sits in. Measured before the fix: ``model_path`` pointing at one
checkpoint among 20,000 stat'ed 20,001 paths, of which 20,000 could not
possibly change -- ``model_path`` is an input and the run writes nothing
beside it. On the machine this was found on, a file candidate under ``/tmp``
walked 475,250 paths and cost 44 seconds cold.

AND EVERY RUN PAID IT. The baseline is taken whether or not hashing was asked
for, because it is what tells the final pass which files the run created.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from spacr import run_journal as rj


@pytest.fixture
def journal(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(rj, "runs_root", lambda: root)
    return root


@pytest.fixture
def crowded(tmp_path):
    """A folder holding one interesting file and many uninteresting ones."""
    folder = tmp_path / "models"
    folder.mkdir()
    for index in range(50):
        (folder / f"other_{index}.pth").write_bytes(b"x")
    chosen = folder / "chosen.pth"
    chosen.write_bytes(b"x")
    return folder, chosen


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------

def test_an_input_file_inventories_itself_and_not_its_neighbours(
        journal, crowded):
    folder, chosen = crowded
    with rj.open_run("mask", {"model_path": str(chosen)}) as run:
        baseline = dict(run._baseline)
    assert list(baseline) == [str(chosen)], (
        f"inventoried {len(baseline)} paths for one input file")


def test_a_folder_setting_still_inventories_the_folder(journal, crowded):
    """The narrowing must not cost the case the baseline exists for."""
    folder, _chosen = crowded
    with rj.open_run("mask", {"src": str(folder)}) as run:
        baseline = dict(run._baseline)
    assert len(baseline) == 51


def test_an_output_file_still_watches_its_directory(journal, crowded):
    """A run CAN write siblings beside an output file -- a report beside its
    figures, a tar beside its manifest -- so the directory is right there."""
    folder, _chosen = crowded
    with rj.open_run("mask", {"report_path": str(folder / "report.html")}) as run:
        baseline = dict(run._baseline)
    assert len(baseline) == 51


@pytest.mark.parametrize("key,expected_dir", [
    ("model_path", False),
    ("csv", False),
    ("dst", True),
    ("report_path", True),
    ("tar_path", True),
])
def test_output_keys_take_the_directory_and_input_keys_do_not(key, expected_dir,
                                                              tmp_path):
    target = tmp_path / "a_file.bin"
    target.write_bytes(b"x")
    root = rj.Run._inventory_root(key, target, False)
    assert (root is not None) is expected_dir, key


def test_a_directory_is_always_the_root_whatever_the_key(tmp_path):
    folder = tmp_path / "d"
    folder.mkdir()
    assert rj.Run._inventory_root("model_path", folder, False) == folder


def test_the_output_only_flag_wins_over_the_key_name(tmp_path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"x")
    assert rj.Run._inventory_root("model_path", target, True) == tmp_path


# ---------------------------------------------------------------------------
# The two passes have to agree
# ---------------------------------------------------------------------------

def test_a_narrowed_baseline_does_not_report_neighbours_as_outputs(
        journal, crowded, monkeypatch):
    """If the final pass walked wider than the baseline, every sibling would
    look like a file this run created."""
    folder, chosen = crowded
    monkeypatch.setattr(rj.Run, "hashing_enabled", lambda self: True)
    with rj.open_run("mask", {"model_path": str(chosen)}) as run:
        pass
    assert not run.output_hashes, sorted(run.output_hashes)[:5]


def test_a_changed_input_file_is_still_caught_as_an_output(
        journal, crowded, monkeypatch):
    """Narrowing must not make a modification invisible."""
    folder, chosen = crowded
    monkeypatch.setattr(rj.Run, "hashing_enabled", lambda self: True)
    with rj.open_run("mask", {"model_path": str(chosen)}):
        time.sleep(0.01)
        chosen.write_bytes(b"rewritten by the run")
    # The run object is the context value; re-open to read what it recorded.
    manifests = sorted(journal.glob("*/manifest.json"))
    assert manifests
    import json
    manifest = json.loads(manifests[-1].read_text())
    assert str(chosen) in manifest["output_hashes"]


# ---------------------------------------------------------------------------
# The backstop
# ---------------------------------------------------------------------------

def test_a_huge_directory_is_truncated_and_says_so(journal, tmp_path,
                                                   monkeypatch):
    """A source folder can hold a million crops. Truncating in silence would
    make the manifest claim an inventory it does not have."""
    monkeypatch.setattr(rj, "INVENTORY_BUDGET", 10)
    folder = tmp_path / "many"
    folder.mkdir()
    for index in range(40):
        (folder / f"f{index}.png").write_bytes(b"x")
    with rj.open_run("mask", {"src": str(folder)}) as run:
        baseline = dict(run._baseline)
        warnings = list(run.provenance_warnings)
    assert len(baseline) == 10
    assert any("not complete" in w for w in warnings), warnings


def test_the_budget_warning_is_recorded_once_per_root(journal, tmp_path,
                                                      monkeypatch):
    monkeypatch.setattr(rj, "INVENTORY_BUDGET", 5)
    folder = tmp_path / "many"
    folder.mkdir()
    for index in range(20):
        (folder / f"f{index}.png").write_bytes(b"x")
    with rj.open_run("mask", {"src": str(folder), "dst": str(folder)}) as run:
        warnings = [w for w in run.provenance_warnings if "not complete" in w]
    assert len(warnings) == 1, warnings


# ---------------------------------------------------------------------------
# The cost, which is the reason any of this changed
# ---------------------------------------------------------------------------

def test_naming_one_file_does_not_scale_with_its_neighbours(journal, tmp_path):
    """The property, stated as a property rather than as a stopwatch.

    A timing assertion would be flaky on a loaded machine; the invariant
    that actually matters is that the work does not grow with the number of
    files the run cannot touch.
    """
    counts = []
    for population in (10, 200):
        folder = tmp_path / f"pop{population}"
        folder.mkdir()
        for index in range(population):
            (folder / f"n{index}.pth").write_bytes(b"x")
        chosen = folder / "chosen.pth"
        chosen.write_bytes(b"x")
        with rj.open_run("mask", {"model_path": str(chosen)}) as run:
            counts.append(len(run._baseline))
    assert counts == [1, 1], counts
