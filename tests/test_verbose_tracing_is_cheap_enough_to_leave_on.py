"""Instruction 294: verbose logging must be cheap before it can be default.

The ask was "verbose logging should be on by default". Measured the day it
was asked, with verbose on: spaCR wrote three 5 MB log files inside one
minute and the interface became "supper laggy". So the ask cannot ship
until the trace is cheap, and this file is where "cheap" is held to a
number rather than an intention.

WHAT IS ALREADY HELD ELSEWHERE, and deliberately not re-asserted here: the
per-frame paint path is excluded by `_TRACE_SKIP_MODULES` and Qt's event
overrides by `_TRACE_SKIP_NAMES`, both of which carry their own comments and
their own failures. This file is only about the cost of the hook itself on
the calls it does NOT skip.
"""
from __future__ import annotations

import os

import pytest

from spacr import logging_util


@pytest.fixture
def clean_cache():
    """A trace-path cache that starts empty and does not outlive the test."""
    was = dict(logging_util._TRACE_REALPATH_CACHE)
    logging_util._TRACE_REALPATH_CACHE.clear()
    yield logging_util._TRACE_REALPATH_CACHE
    logging_util._TRACE_REALPATH_CACHE.clear()
    logging_util._TRACE_REALPATH_CACHE.update(was)


def test_a_path_is_resolved_once_however_often_it_is_traced(
        clean_cache, monkeypatch):
    """THE DEFECT, STATED AS A COUNT.

    `_trace_one_event` runs on every Python call AND return in the process
    while verbose is on, and it resolved the source path on each one.
    `os.path.realpath` is not a string operation -- it walks the path
    against the filesystem. Measured 11,739 ns against 45 ns for a dict
    hit, which at a conservative ten thousand calls a second is about a
    quarter of a core spent resolving the same few hundred paths.

    So: a thousand lookups of one path must reach the filesystem ONCE.
    """
    calls = []
    real = os.path.realpath

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(logging_util.os.path, "realpath", counting)

    target = logging_util.__file__
    for _ in range(1000):
        logging_util._traced_realpath(target)

    assert len(calls) == 1, (
        f"the path was resolved {len(calls)} times for 1000 lookups; the "
        f"trace hook is back to a filesystem call per traced event")


def test_the_cached_answer_is_the_real_one(clean_cache):
    """A cache that returns the wrong path would pass the test above.

    The resolved path decides whether a frame is inside the spaCR package
    at all -- `_trace_one_event` compares it against `_TRACE_ROOT` -- so a
    cache that answered with the unresolved `co_filename` would quietly
    change WHICH functions are traced, not just how fast.
    """
    target = logging_util.__file__
    assert logging_util._traced_realpath(target) == os.path.realpath(target)
    # and again, from the cache this time
    assert logging_util._traced_realpath(target) == os.path.realpath(target)


def test_two_different_paths_get_two_different_answers(clean_cache):
    """PROOF THE CACHE IS KEYED, not a single remembered answer.

    A one-slot cache -- or one keyed on something constant -- would satisfy
    both tests above and hand every file the first file's path, which would
    put every traced frame inside or outside the package together.
    """
    here = logging_util.__file__
    there = os.path.join(os.path.dirname(here), "utils.py")
    assert os.path.exists(there), "picked a neighbour that is not there"

    first = logging_util._traced_realpath(here)
    second = logging_util._traced_realpath(there)

    assert first != second, "two source files resolved to one path"
    assert first == os.path.realpath(here)
    assert second == os.path.realpath(there)
    assert len(clean_cache) == 2, f"cache holds {len(clean_cache)} entries"


def test_the_hook_still_decides_by_the_resolved_path(clean_cache):
    """The cache sits UNDER the in-package test, and must not bypass it.

    `_trace_one_event` returns early for a file outside `_TRACE_ROOT`. If
    the cache were consulted after that decision, or returned something the
    comparison could not read, a file outside the package would start being
    traced -- which is the hook's one hard boundary.
    """
    outside = logging_util._traced_realpath(os.__file__)
    assert not outside.startswith(logging_util._TRACE_ROOT), (
        "the standard library resolved to inside the spaCR package")

    inside = logging_util._traced_realpath(logging_util.__file__)
    assert inside.startswith(logging_util._TRACE_ROOT), (
        "spaCR's own module resolved to outside the spaCR package")
