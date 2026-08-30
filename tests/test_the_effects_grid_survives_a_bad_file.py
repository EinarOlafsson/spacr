"""Reading and writing the sweep's effects grid, including every way it fails.

The grid is a convenience: it lets multivariate montage selection work across
application sessions. Everything in this file is the code that decides the
convenience is unavailable, and all of it is silent by design -- the montage
must still be drawn from a single score when the grid cannot be read.

Silent failure paths are the ones that most need a test, because nothing else
will ever tell you they stopped working: a raise here would be noticed
immediately, and a wrong ``None`` would not be noticed at all.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# effects_grid_from_results — every route to None
# ---------------------------------------------------------------------------

def test_a_grid_that_will_not_parse_is_treated_as_absent(tmp_path, monkeypatch):
    """Lines 304-309: the read raises and the answer is None, not a traceback.

    A truncated or half-written CSV is what a sweep killed mid-write leaves.
    The failure is injected rather than staged with a corrupt file, because
    pandas parses almost anything -- a file of random bytes comes back as a
    one-column frame, which is a DIFFERENT branch. Forcing the raise is the
    only way to reach the handler this test is about.

    The montage can be drawn without the grid, so taking the run down over a
    file it can do without would be the wrong trade -- the caller's own
    message already says what the absence costs.
    """
    from spacr import cell_montage

    (tmp_path / cell_montage.EFFECTS_GRID_FILE).write_text("guide,area\ng1,0.5\n")

    def refuse(*_args, **_kwargs):
        raise pd.errors.ParserError("unexpected end of file")

    monkeypatch.setattr(cell_montage.pd, "read_csv", refuse)

    assert cell_montage.effects_grid_from_results(str(tmp_path)) is None


def test_an_empty_grid_is_treated_as_absent(tmp_path):
    """Line 311: a file that parsed but holds nothing is still no grid."""
    from spacr.cell_montage import EFFECTS_GRID_FILE, effects_grid_from_results

    pd.DataFrame().to_csv(tmp_path / EFFECTS_GRID_FILE)

    assert effects_grid_from_results(str(tmp_path)) is None


def test_no_path_and_no_file_are_both_absent(tmp_path):
    """The two guards above, so the ones below are reached deliberately."""
    from spacr.cell_montage import effects_grid_from_results

    assert effects_grid_from_results("") is None
    assert effects_grid_from_results(None) is None
    assert effects_grid_from_results(str(tmp_path)) is None      # no file


def test_a_readable_grid_comes_back_with_its_terms_reduced_to_guides(tmp_path):
    """Line 312, and the reason the index is rewritten.

    The sweep writes statsmodels terms as its index -- ``C(grna)[T.g1]`` --
    and the montage looks guides up by bare name. Reading a grid without
    reducing the index would match nothing, silently, and the montage would
    fall back as though there had been no grid at all.
    """
    from spacr.cell_montage import EFFECTS_GRID_FILE, effects_grid_from_results

    frame = pd.DataFrame({"area": [0.5, -0.2]},
                         index=["C(grna)[T.g1]", "g2"])
    frame.to_csv(tmp_path / EFFECTS_GRID_FILE)

    grid = effects_grid_from_results(str(tmp_path))

    assert grid is not None
    assert list(grid.index) == ["g1", "g2"]


def test_a_file_path_is_read_from_its_own_folder(tmp_path):
    """The ``isdir`` branch: callers pass either the folder or a file in it."""
    from spacr.cell_montage import EFFECTS_GRID_FILE, effects_grid_from_results

    pd.DataFrame({"area": [0.5]}, index=["g1"]).to_csv(
        tmp_path / EFFECTS_GRID_FILE)
    results = tmp_path / "results.csv"
    results.write_text("anything")

    assert effects_grid_from_results(str(results)) is not None


# ---------------------------------------------------------------------------
# write_effects_grid — nothing to write, and nowhere to write it
# ---------------------------------------------------------------------------

def test_nothing_worth_writing_writes_nothing(tmp_path):
    """Line 325: the empty string means "no grid was filed"."""
    from spacr.cell_montage import write_effects_grid

    assert write_effects_grid(None, str(tmp_path)) == ""
    assert write_effects_grid(pd.DataFrame(), str(tmp_path)) == ""
    assert write_effects_grid(pd.DataFrame({"a": [1]}), "") == ""


def test_a_folder_that_cannot_be_written_is_not_fatal(tmp_path, monkeypatch):
    """Lines 330-333: a sweep that produced its answer has not failed.

    The grid is filed BESIDE the run, and a read-only or full destination must
    cost the convenience, not the result the sweep just computed.
    """
    from spacr import cell_montage

    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(cell_montage.os, "makedirs", refuse)

    assert cell_montage.write_effects_grid(
        pd.DataFrame({"a": [1.0]}, index=["g1"]), str(tmp_path)) == ""


def test_a_written_grid_can_be_read_back(tmp_path):
    """The round trip, which is the whole point of persisting it."""
    from spacr.cell_montage import effects_grid_from_results, write_effects_grid

    effects = pd.DataFrame({"area": [0.5, -0.2]}, index=["g1", "g2"])
    written = write_effects_grid(effects, str(tmp_path))

    assert written and os.path.isfile(written)
    back = effects_grid_from_results(str(tmp_path))
    assert back is not None
    assert list(back.index) == ["g1", "g2"]


# ---------------------------------------------------------------------------
# _guide_of_term — line 387, a term that is not a factor level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("term, expected", [
    ("C(grna)[T.g1]", "g1"),        # the statsmodels factor spelling
    ("grna[g2]", "g2"),             # the plain bracketed spelling
    ("g3", "g3"),                   # THE LINE: no brackets, returned as-is
    ("", ""),
])
def test_a_term_without_brackets_is_already_a_guide_name(term, expected):
    """Line 387.

    The grid's index mixes spellings because it is written by whichever fit
    produced it. A bare guide name must survive unchanged -- and this is the
    common case for a sweep over guides rather than over a factor.
    """
    from spacr.cell_montage import _guide_of_term

    assert _guide_of_term(term) == expected
