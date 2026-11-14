"""Run one behaviour case and hand back what it observed.

A case in the spacr.ml test files is an ordinary function (or a method of a
small class) written against ordinary pytest fixtures.  Its body records, in
order, the value each of its checks looked at; ``observe`` resolves the
fixtures the case asks for by name, calls it, and returns that list.  The
test then states the expectation for the row::

    observed = observe(case, request)
    assert observed == expected

The case body keeps its own secondary asserts, so a failure inside the case
still names the line.
"""
from __future__ import annotations

import inspect


def observe(case, request):
    """Call ``case`` with its fixtures and return the list of recorded values."""
    if isinstance(case, tuple):
        owner, name = case
        func = getattr(owner(), name)
    else:
        func = case
    kwargs = {}
    for name in inspect.signature(func).parameters:
        kwargs[name] = request.getfixturevalue(name)
    seen = func(**kwargs)
    return [] if seen is None else seen


def keep(value):
    """A copy of ``value`` as it is now, so a later change by the case cannot rewrite the record."""
    import copy

    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 - an uncopyable value is recorded as it is
        return value
