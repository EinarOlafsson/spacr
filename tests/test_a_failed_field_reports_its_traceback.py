"""A field that fails inside the Measure worker brings its traceback home.

The parent told the user "worker traceback in ~/.spacr/logs/spacr.log" and the
named log held nothing about it -- reported 2026-09-01 against
`plate1_E02_20_1.npy`. `_measure_crop_core` runs in a multiprocessing.Pool
worker, so the parent's logging configuration is not that process's:
`traceback.print_exc()` there writes to a worker stderr nobody reads, and a
`RunLedger` opened there is a second ledger in a second process.

So the one thing needed to fix the field was the one thing thrown away, and the
message pointed at where it wasn't.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import spacr.measure as measure


def test_the_worker_returns_five_things_now():
    """The fifth is the traceback text."""
    source = inspect.getsource(measure._measure_crop_core)
    assert "return index, average_time, cells, figs, error_text" in source


def test_the_success_path_carries_an_empty_error():
    """A field that worked must not look like one that failed."""
    source = inspect.getsource(measure._measure_crop_core)
    assert 'error_text = ""' in source


def test_the_traceback_is_text_not_an_exception():
    """An exception is not always picklable, and a field failing with an
    unpicklable error would fail AGAIN on the way back -- losing the first
    failure, which is the one that matters."""
    source = inspect.getsource(measure._measure_crop_core)
    assert "traceback.format_exception" in source


def test_the_worker_no_longer_opens_its_own_ledger():
    """A RunLedger opened in a pool worker is a second ledger in a second
    process; the parent's is the one the run reads."""
    source = inspect.getsource(measure._measure_crop_core)
    assert "RunLedger('_measure_crop_core')" not in source


def test_the_bare_debug_print_is_gone():
    """`print('main', e)` labelled every field failure "main"."""
    source = inspect.getsource(measure._measure_crop_core)
    assert "print('main'" not in source


def test_the_parent_records_what_the_worker_sent():
    source = Path(measure.__file__).read_text(encoding="utf-8")
    assert "result[4] if len(result) > 4 else" in source, (
        "the parent does not read the traceback the worker returns")
    assert "exc=detail" in source, (
        "the traceback is read and then not recorded")
    # The old message survives only in the comment explaining why it was
    # wrong. What matters is that no `exc=` argument still carries it.
    assert "exc='field failed inside _measure_crop_core '" not in source


def test_a_worker_that_sends_nothing_still_says_so():
    """A four-tuple from an older run must not record an empty reason."""
    source = Path(measure.__file__).read_text(encoding="utf-8")
    assert "worker returned no traceback" in source


def test_the_old_four_tuple_is_still_processable():
    """`process_measure_crop_results` is called with saved partial results, so
    it must not start raising on a shape it accepted yesterday."""
    source = inspect.getsource(measure.process_measure_crop_results)
    assert "result[:4]" in source

    # And it actually runs: a four-tuple with no figures is a no-op, not a
    # ValueError about unpacking.
    measure.process_measure_crop_results([(0, 1.0, 3, None)], {"src": "."})
    measure.process_measure_crop_results([(0, 1.0, 3, None, "")], {"src": "."})
    measure.process_measure_crop_results([None], {"src": "."})
