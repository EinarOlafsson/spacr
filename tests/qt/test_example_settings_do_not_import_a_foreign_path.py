"""A shipped example settings file must not point at the publisher's machine.

Reported 2026-09-01 from a home machine: pressing the load button set the
source to ``/home/carruthers/datasets/plate1``, a user that machine has never
heard of. The cause is on disk in the downloaded cache::

    gen_masks_settings.csv:34     src,/home/carruthers/datasets/plate1
    crop_measure_settings.csv:14  src,/home/carruthers/datasets/plate1/merged

Both lines matter. The first is why the path is wrong; the SECOND is why
simply substituting the destination is not the fix -- the Measure set points at
a subfolder, and collapsing it to the plate root would quietly measure the
wrong directory rather than fail.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from spacr.qt.screens.app_screen import AppScreen

reanchor = AppScreen.reanchor_example_paths


@pytest.fixture
def plate(tmp_path):
    dest = tmp_path / "plate1"
    (dest / "merged").mkdir(parents=True)
    return dest


def test_the_reported_path_is_rehomed(plate):
    """The exact value from the user's machine."""
    out = reanchor({"src": "/home/carruthers/datasets/plate1"}, plate)
    assert out["src"] == str(plate)


def test_a_subfolder_keeps_its_tail(plate):
    """``/merged`` is the whole reason a blunt substitution is wrong."""
    out = reanchor({"src": "/home/carruthers/datasets/plate1/merged"}, plate)
    assert out["src"] == str(plate / "merged")
    assert Path(out["src"]).is_dir(), "the re-homed path must actually exist"


def test_a_deep_path_keeps_every_component(plate):
    out = reanchor(
        {"db_path": "/home/carruthers/datasets/plate1/measurements/measurements.db"},
        plate)
    assert out["db_path"] == str(plate / "measurements" / "measurements.db")


def test_a_path_that_already_resolves_here_is_left_alone(plate):
    """A user who edited src to their own data must not have it undone."""
    mine = plate / "merged"
    out = reanchor({"src": str(mine)}, plate)
    assert out["src"] == str(mine)


def test_values_that_are_not_paths_are_untouched(plate):
    given = {"agg_type": "mean", "channels": "[0,1]", "rel": "settings/x.csv",
             "blank": "", "n": "5"}
    assert reanchor(dict(given), plate) == given


def test_an_absolute_path_with_no_shared_anchor_is_not_guessed(plate):
    """A wrong path that LOOKS local is worse than one that is obviously not.

    There is nothing to hang the tail on here, so re-pointing at the
    destination would be a guess. It is left for the user to see.
    """
    out = reanchor({"model": "/opt/models/cellpose.pth"}, plate)
    assert out["model"] == "/opt/models/cellpose.pth"


def test_the_input_mapping_is_not_modified(plate):
    given = {"src": "/home/carruthers/datasets/plate1"}
    reanchor(given, plate)
    assert given["src"] == "/home/carruthers/datasets/plate1"


def test_without_the_fix_the_foreign_path_would_be_applied(plate):
    """Not vacuous: the raw value really is the bad one.

    If the CSV were already local this whole guard would be asserting against
    nothing, so the untreated value is checked to be the reported defect.
    """
    raw = "/home/carruthers/datasets/plate1/merged"
    assert not Path(raw).exists(), (
        "this test machine really has a /home/carruthers tree; "
        "pick a different foreign path for the fixture")
    assert reanchor({"src": raw}, plate)["src"] != raw


# ---------------------------------------------------------------------------
# Container-valued settings. Found by an audit of the first version of this
# fix, which tested ``isinstance(value, str)`` and skipped everything else --
# and so skipped exactly the two modules that have no local-path fallback.
# ---------------------------------------------------------------------------

def test_a_list_valued_source_is_rehomed(plate):
    """Classify's ``src`` is list-valued.

    ``utils.load_settings`` turns a CSV cell starting with ``[`` into a real
    Python list, and Classify's loader never writes a local path of its own --
    so a skipped list left the publisher's path as the panel's only truth.
    """
    out = reanchor({"src": ["/home/carruthers/datasets/plate1"]}, plate)
    assert out["src"] == [str(plate)]


def test_a_tuple_of_paths_is_rehomed_and_stays_a_tuple(plate):
    """Regression's count_data/score_data/paired_data are container-valued."""
    given = ("/home/carruthers/datasets/plate1/a.csv",
             "/home/carruthers/datasets/plate1/b.csv")
    out = reanchor({"paired_data": given}, plate)
    assert out["paired_data"] == (str(plate / "a.csv"), str(plate / "b.csv"))
    assert isinstance(out["paired_data"], tuple)


def test_nested_containers_are_walked(plate):
    out = reanchor({"src": [["/home/carruthers/datasets/plate1/x"]]}, plate)
    assert out["src"] == [[str(plate / "x")]]


def test_non_string_values_survive_untouched(plate):
    given = {"n": 5, "flag": True, "nothing": None, "ratio": 0.5}
    assert reanchor(dict(given), plate) == given
