"""Guards for the four repairs that closed the last of instruction 100's xfails.

Each xfail that came off carried exactly one assertion, and each of those
assertions can be satisfied by a repair that is wrong in a different way --
by hardcoding the number the test looks for, by folding every key rather than
the ones that were folded on the way in, by deleting a feature outright
instead of tying it to the opt-in.

These tests pin the half the xfail did not: what the repair must NOT do.
The sweep-runs half lives in ``test_xfail_backlog_item_100_sweep_runs``,
apart so its Qt skip cannot take these three modules' guards with it.
"""
from __future__ import annotations

import sys

import pandas as pd
import pytest

import spacr.hit_attribution as ha
import spacr.remote_execution as rx
import spacr.validate as V
from spacr.remote_execution import RemoteExecutionError


# ---------------------------------------------------------------------------
# remote_execution._run_command -- the timeout sentence
# ---------------------------------------------------------------------------

def test_a_timeout_above_the_floor_still_names_its_own_value():
    """The floor is a floor, not the number the sentence always prints.

    The sub-second case is satisfied by writing "1s" unconditionally, which
    would be wrong for every ordinary timeout; this is the other side of it.
    """
    with pytest.raises(RemoteExecutionError) as caught:
        rx._run_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=2.0)
    assert "timed out after 2s" in str(caught.value)


# ---------------------------------------------------------------------------
# validate._normalize_app -- folding the caller's key
# ---------------------------------------------------------------------------

def test_an_unregistered_hyphenated_key_passes_through_in_its_own_spelling():
    """The fold reaches a registered alias; it does not rewrite a miss.

    ``_normalize_app`` hands an unknown key back so the caller can name it in
    an error. Folding unconditionally would report a key the caller never
    typed.
    """
    assert "no-such-app-anywhere" not in V.APP_ALIASES
    assert "no_such_app_anywhere" not in V.APP_ALIASES
    assert V._normalize_app("no-such-app-anywhere") == "no-such-app-anywhere"


def test_an_alias_spelled_with_a_hyphen_wins_over_its_folded_twin(monkeypatch):
    """The exact key is tried first, so a hyphenated entry stays reachable."""
    monkeypatch.setitem(V.APP_ALIASES, "probe-thing", "exact_entry")
    monkeypatch.setitem(V.APP_ALIASES, "probe_thing", "folded_entry")

    assert V._normalize_app("probe-thing") == "exact_entry"
    assert V._normalize_app("probe_thing") == "folded_entry"


# ---------------------------------------------------------------------------
# hit_attribution._default_features -- the score and what is derived from it
# ---------------------------------------------------------------------------

def _small_frame():
    return pd.DataFrame({
        "XGBoost_score": [0.1, 0.9, 0.4],
        "candidate_rank": [3, 1, 2],
        "candidate_percentile": [0.25, 1.0, 0.5],
        "cell_area": [10.0, 11.0, 12.0],
    })


def test_the_score_transforms_come_back_when_the_score_is_asked_for():
    """They are excluded WITH the score, not deleted from the feature set.

    Dropping ``candidate_rank`` unconditionally would also satisfy the xfail
    that came off, and would quietly remove a feature the caller explicitly
    opted into.
    """
    features = ha._default_features(_small_frame(), "XGBoost_score",
                                    include_score=True)

    assert "XGBoost_score" in features
    assert "candidate_rank" in features
    assert "candidate_percentile" in features


def test_without_the_score_only_the_morphology_survives():
    """The opt-out excludes the score and every monotone transform of it."""
    features = ha._default_features(_small_frame(), "XGBoost_score",
                                    include_score=False)

    assert features == ["cell_area"]


def test_every_score_derived_column_is_one_build_hit_cell_frame_writes():
    """The constant and the writer name the same columns.

    A rename on one side only would put the score back among the defaults
    under a name nothing recognises.
    """
    import inspect

    # Non-empty first: an empty tuple would satisfy the loop below without
    # checking anything, and would ALSO put both transforms back among the
    # default features -- a vacuous pass exactly where the leak reopens.
    assert set(ha.SCORE_DERIVED_COLUMNS) == {"candidate_rank",
                                             "candidate_percentile"}
    source = inspect.getsource(ha.build_hit_cell_frame)
    for column in ha.SCORE_DERIVED_COLUMNS:
        assert f'frame["{column}"]' in source
