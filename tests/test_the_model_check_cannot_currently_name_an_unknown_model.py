"""``check_model`` validates names against the real TorchVision vocabulary."""
from __future__ import annotations

import builtins


def test_the_checker_reads_an_existing_model_registry():
    from spacr.model_check import _known_builtin_models

    known = _known_builtin_models()
    assert "maxvit_t" in known
    assert "resnet50" in known
    assert "maxvit_tt" not in known


def test_the_checker_keeps_a_lightweight_registry_without_torchvision(
        monkeypatch):
    """An optional import failure cannot turn validation off again."""
    from spacr.model_check import _known_builtin_models

    real_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "torchvision":
            raise ImportError("torchvision is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)

    known = _known_builtin_models()
    assert "maxvit_t" in known
    assert "maxvit_tt" not in known


def test_a_broken_live_registry_does_not_disable_the_curated_one(monkeypatch):
    """A third-party registry failure is contained by the click-time checker."""
    from torchvision import models

    from spacr.model_check import _known_builtin_models

    def broken(*args, **kwargs):
        raise RuntimeError("registry is broken")

    monkeypatch.setattr(models, "list_models", broken)

    assert "resnet50" in _known_builtin_models()


def test_an_unknown_model_name_is_refused_before_training():
    """A typo is named by the checker, not deferred to model construction."""
    from spacr.model_check import check_model

    report = check_model({"model_type": "maxvit_tt", "classes": ["a", "b"]})

    assert report.ok is False
    assert [p for p in report.problems if "not a model spaCR knows" in p]


def test_a_known_model_name_earns_no_model_name_complaint():
    """The registry distinguishes a real backbone from the typo above."""
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
