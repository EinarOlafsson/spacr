"""A Poisson or horseshoe response must still be a count when it is fitted.

Found by a worker on instruction 100's backlog and fixed here for the direct
call path.

`process_scores` correctly SUMS the well for the count families -- Poisson and
horseshoe model the number of positive objects in a well, not their average --
and then applied `transform` to that sum anyway. The default transform is
`'log'`, so the integer count left as a float and `_validate_poisson_response`
refused it:

    at the very END of a run that had already read both CSVs and fitted
    nothing at all.

Neither count family could be started, from Tk, from Qt or from the CLI.

`settings.py` now clears `transform` for these families before a run, which
covers every GUI and CLI path. This is the second line of defence and the one
that covers a DIRECT `regression()` or `process_scores()` call, which never
passes through the settings layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _wells(n_wells=12, per_well=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(n_wells):
        for obj in range(per_well):
            rows.append({
                "plateID": "plate1",
                "rowID": f"r{well // 4 + 1}",
                "columnID": f"c{well % 4 + 1}",
                "objectID": f"o{obj}",
                "pred": int(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("regression_type", ["poisson", "horseshoe"])
def test_the_response_stays_a_whole_number(regression_type):
    """The defect, stated as the thing the validator was about to say."""
    from spacr.ml import process_scores

    frame, column = process_scores(
        _wells(), "pred", plate="plate1", min_cell_count=5,
        agg_type="mean", transform="log", regression_type=regression_type)

    assert column == "pred", (
        f"the response was renamed to {column!r}: a transform was applied to "
        "a count and the count model will refuse it at the end of the run")
    values = frame[column].to_numpy(dtype=float)
    assert np.allclose(values, np.round(values)), (
        f"{regression_type} response is not integral: {values[:5]}")
    assert (values >= 0).all()


@pytest.mark.parametrize("regression_type", ["poisson", "horseshoe"])
def test_the_validator_that_used_to_refuse_it_now_accepts(regression_type):
    """Through the real check, not a stand-in for it.

    "does not raise" is the whole behaviour here, so the assertions pin what
    was actually handed to it -- a validator that accepted an empty column
    would pass a bare call and mean nothing.
    """
    from spacr.ml import _validate_poisson_response, process_scores

    frame, column = process_scores(
        _wells(seed=3), "pred", plate="plate1", min_cell_count=5,
        agg_type="mean", transform="log", regression_type=regression_type)

    response = frame[column]
    assert len(response) > 0, "nothing was validated"
    _validate_poisson_response(response)             # must not raise

    # And it still refuses what it is for, so the acceptance above is real.
    with pytest.raises(Exception):
        _validate_poisson_response(response.astype(float) + 0.5)


def test_an_ordinary_model_still_gets_its_transform():
    """The fix must not reach past the count families. `log` on a continuous
    well mean is the ordinary case and is why the default exists."""
    from spacr.ml import process_scores

    frame, column = process_scores(
        _wells(seed=1), "pred", plate="plate1", min_cell_count=5,
        agg_type="mean", transform="log", regression_type="ols")

    assert column == "log_pred", f"the transform was dropped: got {column!r}"
    assert "log_pred" in frame.columns


def test_no_transform_asked_for_is_still_no_transform():
    from spacr.ml import process_scores

    frame, column = process_scores(
        _wells(seed=2), "pred", plate="plate1", min_cell_count=5,
        agg_type="mean", transform=None, regression_type="poisson")

    assert column == "pred"
    assert not any(name.startswith("log_") for name in frame.columns)


def test_it_says_it_ignored_the_transform(capsys):
    """Silently dropping a setting the user set is how a run produces a
    result nobody can explain."""
    from spacr.ml import process_scores

    process_scores(_wells(seed=4), "pred", plate="plate1", min_cell_count=5,
                   agg_type="mean", transform="log",
                   regression_type="poisson")

    printed = capsys.readouterr().out
    assert "transform" in printed and "poisson" in printed, printed
