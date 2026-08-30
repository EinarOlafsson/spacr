"""What ``check_model`` reports for a model name that does not exist.

The answer today is: nothing, and this file pins that rather than asserting
what the code was clearly meant to do. ``check_model`` tries
``from .model_zoo import KNOWN_MODELS`` inside a bare ``except Exception``, and
``KNOWN_MODELS`` does not exist -- the name appears nowhere in the package
except at that import. So ``known`` is always the empty set, ``if known and
name not in known`` is always False, and the "is not a model spaCR knows"
complaint can never be emitted.

Recorded in instruction 310. When the check is given a real source of model
names, these tests must be UPDATED to the new expectation rather than deleted:
they are the record of what the behaviour was, and the reason the change is a
fix rather than a regression.
"""
from __future__ import annotations

import pytest


def test_the_registry_the_check_asks_for_does_not_exist():
    """The root cause, asserted directly so a fix is visible here first.

    This is spaCR's commonest defect shape inverted: not finished code with no
    caller, but a caller reading a name nothing defines -- and a bare except
    turning that into silence rather than an error.
    """
    with pytest.raises(ImportError):
        from spacr.model_zoo import KNOWN_MODELS       # noqa: F401


def test_an_unknown_model_name_earns_no_complaint_today():
    """Current behaviour, pinned.

    A user typing 'maxvit_tt' gets no complaint from the checker -- the
    validation that exists for exactly this typo cannot fire. The failure is
    deferred to training, where it surfaces as a torchvision AttributeError
    long after the click.
    """
    from spacr.model_check import check_model

    report = check_model({"model_type": "maxvit_tt", "classes": ["a", "b"]})

    assert not [p for p in report.problems
                if "not a model spaCR knows" in p]


def test_a_known_model_name_also_earns_no_complaint():
    """The contrast that shows the check is silent either way.

    If the branch were live, this test and the one above would differ. That
    they agree is the evidence that nothing is being checked.
    """
    from spacr.model_check import check_model

    report = check_model({"model_type": "maxvit_t", "classes": ["a", "b"]})

    assert not [p for p in report.problems
                if "not a model spaCR knows" in p]


def test_no_model_at_all_is_still_reported():
    """The check that DOES work, so the silence above is specific.

    An empty model_type is caught and named, which is why the unknown-name
    case reads as an oversight rather than a deliberate leniency.
    """
    from spacr.model_check import check_model

    report = check_model({"model_type": "", "classes": ["a", "b"]})

    assert report.ok is False
    assert any("no model is chosen" in p for p in report.problems)


def test_the_check_never_raises_whatever_it_is_given():
    """The docstring's promise: this runs from a click.

    "A dialog that crashes the screen is a worse answer than one that says
    what is wrong" -- so every shape of nonsense must come back as a report.
    """
    from spacr.model_check import check_model

    for settings in ({}, {"model_type": None}, {"model_type": 3},
                     {"model_type": "maxvit_t", "classes": None},
                     {"model_type": "maxvit_t", "classes": []}):
        report = check_model(settings)
        assert hasattr(report, "problems")
