"""``scan_parameters``' progress/cancel hook — the one entry point the GUI needed.

:func:`spacr.power_model.scan_parameters` is minutes long at any realistic
design, and until this hook existed there was no way to see where it was or to
stop it: the whole sweep was one opaque call. The Power / Design screen drives
it from a worker thread, so it needs both.

The hook is deliberately tiny, and these tests pin the two things that make it
safe to add to a function whose output people quote:

* **it changes no number.** A sweep run with a hook and the same sweep run
  without one produce identical rows, identical ``run_key``s and identical
  metrics. If the hook could perturb the seeding or the grid order, every
  result produced through the GUI would be a different result from the same
  call typed at a prompt.
* **only ``False`` stops it.** A callback that returns ``0``, ``""`` or
  ``None`` has almost certainly returned something incidental, and abandoning
  a five-minute sweep on that is not a decision to infer.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from spacr import power_model as pm


#: A design small enough that four fits take about a second, and real enough
#: that the fit actually runs rather than being refused.
TINY = dict(
    n_genes_in_library=30,
    gene_abundance_alpha=5.0,
    gene_hit_rate=0.2,
    n_wells_per_screen=40,
    well_abundance_factor_mu=4.0,
    well_abundance_factor_var=1.0,
    imaging_n_cells_per_well_var=None,
    class_pos_mu=0.6,
    class_pos_var=0.005,
    class_neg_mu=0.1,
    class_neg_var=0.005,
    sequencing_n_cells_per_well_lambda=300.0,
    pcr_factor_mu=1.0,
    pcr_factor_var=0.3,
    n_reads_per_well=8000,
    read_depth_cv=0.0,
)

FAST_FIT = dict(n_steps=120, n_draws=32)


def _sweep(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pm.scan_parameters(
            imaging_n_cells_per_well_mu=[40.0, 80.0],
            n_replicates=2,
            backend="torch",
            seed=3,
            fit_kwargs=FAST_FIT,
            **{**TINY, **kwargs},
        )


def test_the_hook_sees_every_fit_once_in_order():
    seen = []
    frame = _sweep(on_point=lambda point: seen.append(
        (point["index"], point["total"], point["point_index"],
         point["replicate"], point["resumed"])))
    assert len(frame) == 4
    assert [row[0] for row in seen] == [1, 2, 3, 4]
    assert {row[1] for row in seen} == {4}
    assert [row[2] for row in seen] == [1, 1, 2, 2]
    assert [row[3] for row in seen] == [0, 1, 0, 1]
    assert not any(row[4] for row in seen)


def test_the_hook_is_handed_the_row_it_just_finished():
    rows = []
    frame = _sweep(on_point=lambda point: rows.append(dict(point["row"])))
    assert len(rows) == len(frame)
    assert [row["run_key"] for row in rows] == list(frame["run_key"])
    assert all(row["status"] in {"ok", "not_converged", "failed"}
               for row in rows)


def test_a_hook_changes_no_number_at_all():
    """The load-bearing property: watching a sweep cannot alter it.

    Same seed, same grid, same backend — so a run started from the GUI and
    the identical call typed at a Python prompt must agree row for row. Were
    the hook able to touch the seeding or the grid order, every number the
    screen reports would be unreproducible from the library.
    """
    watched = _sweep(on_point=lambda point: None)
    plain = _sweep()
    assert list(watched["run_key"]) == list(plain["run_key"])
    assert list(watched["status"]) == list(plain["status"])
    assert np.allclose(watched["model_auroc"].astype(float),
                       plain["model_auroc"].astype(float), equal_nan=True)
    assert np.allclose(watched["model_ap"].astype(float),
                       plain["model_ap"].astype(float), equal_nan=True)


def test_returning_false_stops_the_sweep_and_keeps_what_it_had():
    frame = _sweep(on_point=lambda point: False)
    assert len(frame) == 1
    assert frame.attrs["cancelled"] is True
    assert frame.attrs["n_planned"] == 4
    # The row it did finish is a real result, not a placeholder.
    assert frame["status"].iloc[0] in {"ok", "not_converged", "failed"}


def test_a_short_frame_can_be_told_from_a_small_grid():
    """`n_planned` is why a cancelled sweep does not read as a finished one.

    Without it, four rows out of four and four rows out of twenty-seven look
    the same to anything downstream — and a curve drawn from the second while
    labelled like the first shows a design falling off a cliff it never
    reached.
    """
    full = _sweep(on_point=lambda point: None)
    assert full.attrs["cancelled"] is False
    assert full.attrs["n_planned"] == len(full) == 4
    stopped = _sweep(on_point=lambda point: point["index"] < 2)
    assert stopped.attrs["cancelled"] is True
    assert stopped.attrs["n_planned"] == 4
    assert len(stopped) == 2


@pytest.mark.parametrize("verdict", [None, 0, "", [], True, 1])
def test_only_an_exact_false_stops_it(verdict):
    """Falsy is not the same as "stop". Abandoning minutes of work is a
    decision the caller has to make explicitly."""
    assert len(_sweep(on_point=lambda point: verdict)) == 4


def test_a_raising_hook_aborts_rather_than_being_swallowed():
    """A broken progress reporter is a bug in the caller.

    Swallowing it would leave a long sweep running with nothing watching it
    and no way to stop it — which is exactly the state the hook exists to
    prevent.
    """
    def _boom(point):
        raise RuntimeError("the progress bar exploded")

    with pytest.raises(RuntimeError, match="exploded"):
        _sweep(on_point=_boom)


def test_the_sweep_is_unchanged_when_no_hook_is_passed():
    """The default path keeps its contract, attrs included."""
    frame = _sweep()
    assert len(frame) == 4
    assert frame.attrs["cancelled"] is False
    assert frame.attrs["n_planned"] == 4


def test_a_resumed_row_is_reported_too(tmp_path):
    """A progress bar must not run backwards on the second attempt.

    Rows restored from a progress file are finished fits like any other; a
    hook that only saw freshly-computed ones would show a resumed sweep
    starting at 60 % and counting to 40 %.
    """
    progress = tmp_path / "scan.tsv"
    first = _sweep(progress_file=str(progress), on_point=lambda point: None)
    assert len(first) == 4

    seen = []
    second = _sweep(progress_file=str(progress),
                    on_point=lambda point: seen.append(point["resumed"]))
    assert len(second) == 4
    assert seen == [True, True, True, True]
    assert list(second["run_key"]) == list(first["run_key"])
