"""The failure reporter survives its own helpers failing.

The whole value of this module is that it never replaces the exception it was
asked to describe. That promise is only worth anything when the pieces it calls
are themselves broken, so each helper is driven into an exception here: the
resource ledger, the design summary, and the write to disk.
"""
from __future__ import annotations

import os

import pytest

from spacr import regression_failure


def boom(*args, **kwargs):
    raise RuntimeError('the helper is broken')


def test_a_broken_resource_ledger_leaves_the_rest_of_the_report_intact():
    """When the per-stage cost readings cannot be collected they are omitted.

    The costs are a nicety beside the traceback; a report that refused to print
    because the resource module raised would lose the actual failure.
    """
    import spacr.fit_resources as fit_resources

    original = fit_resources.describe_resources
    fit_resources.describe_resources = boom
    try:
        text = regression_failure.describe_failure(
            ValueError('singular matrix'), stage='fitting')
    finally:
        fit_resources.describe_resources = original

    assert 'THE REGRESSION FAILED.' in text
    assert 'WHAT IT COST, PER STAGE' not in text
    assert 'ValueError: singular matrix' in text
    assert 'TRACEBACK' in text


def test_a_reporter_that_breaks_still_names_the_original_failure(monkeypatch):
    """If the report itself raises, the one-line fallback still carries the error.

    A diagnostic that crashed while explaining a crash would leave the user with
    a traceback pointing at the reporter and nothing about the fit.
    """
    monkeypatch.setattr(regression_failure, '_design_lines', boom)
    text = regression_failure.describe_failure(
        KeyError('prc'), stage='building the design')
    assert text == "THE REGRESSION FAILED: KeyError: 'prc'\n"


def test_a_report_that_cannot_be_written_returns_none_rather_than_raising(
        tmp_path):
    """An unwritable destination yields None, so the caller still prints it.

    The failure report is written beside the run; when the run folder is a file
    or is read-only, refusing to write must not become a second exception on top
    of the one being reported.
    """
    blocker = tmp_path / 'results'
    blocker.write_text('not a directory')

    out = regression_failure.write_failure_report(
        blocker / 'run', RuntimeError('out of memory'), stage='fitting')
    assert out is None
    assert blocker.read_text() == 'not a directory'


def test_no_destination_folder_is_reported_as_no_file_written(tmp_path):
    """An empty res_folder returns None without creating anything.

    A failure early enough to have no results folder is still reported to the
    console; the writer says so by returning None rather than inventing a path.
    """
    assert regression_failure.write_failure_report(
        '', ValueError('boom')) is None
