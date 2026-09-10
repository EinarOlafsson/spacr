"""Widget specs degrade to the curated list rather than to nothing.

Two guards keep a settings screen openable when the inputs are not what the
name-keyed tables assume:

* The torchvision zoo is read out of an ALREADY-IMPORTED module, so what
  sits in ``sys.modules`` is not under this module's control. A namespace it
  cannot walk must fall back to the curated model list -- a combo box with no
  entries is a setting the user can no longer set.
* The value-keyed table disambiguates two meanings of one setting name by
  looking at the value in hand. A value that is not text carries no such
  hint, so the lookup has to decline and let the name-keyed table decide.
"""
from __future__ import annotations

import sys
import types

import pytest

from spacr import settings_spec


class _UnreadableNamespace(types.ModuleType):
    """A module object whose ``__dict__`` cannot be walked."""

    @property
    def __dict__(self):  # noqa: D401 - the point is that it raises
        raise RuntimeError("namespace unavailable")


def test_an_unreadable_torchvision_namespace_falls_back_to_the_curated_list(
        monkeypatch):
    """A zoo that cannot be walked still offers every curated model.

    The combo box is the only way to choose ``model_type``; an empty one
    would make the setting unreachable from the GUI.
    """
    monkeypatch.setitem(sys.modules, "torchvision.models",
                        _UnreadableNamespace("torchvision.models"))

    names = settings_spec._torchvision_model_names()

    assert names == list(settings_spec._TORCHVISION_MODELS_CURATED)
    assert "resnet50" in names


def test_an_empty_torchvision_namespace_falls_back_too(monkeypatch):
    """A loaded-but-empty module contributes nothing and is not trusted."""
    monkeypatch.setitem(sys.modules, "torchvision.models",
                        types.ModuleType("torchvision.models"))

    assert settings_spec._torchvision_model_names() == list(
        settings_spec._TORCHVISION_MODELS_CURATED)


def test_a_loaded_torchvision_namespace_adds_its_own_models(monkeypatch):
    """A real zoo is merged with the curated list, never replaced by it."""
    module = types.ModuleType("torchvision.models")
    module.a_brand_new_net = lambda: None
    module._private = lambda: None
    monkeypatch.setitem(sys.modules, "torchvision.models", module)

    names = settings_spec._torchvision_model_names()

    assert "a_brand_new_net" in names
    assert "_private" not in names
    assert "resnet50" in names


def test_a_non_text_value_declines_the_value_keyed_table():
    """A numeric ``level`` carries no vocabulary hint, so the lookup declines.

    Returning a spec here would pin the setting to whichever vocabulary was
    listed first, which is how one plot's ``level`` ends up offering the other
    plot's choices.
    """
    assert settings_spec._value_special_cases("level", 3) is None


def test_a_text_value_still_picks_its_vocabulary():
    """The same key with a listed text value does resolve to a spec."""
    key = next(iter(settings_spec._VALUE_SPECIAL_CASES))
    vocabulary, _spec = settings_spec._VALUE_SPECIAL_CASES[key][0]
    word = sorted(vocabulary)[0]

    resolved = settings_spec._value_special_cases(key, word.upper())

    assert resolved is not None
    kind, options, default = resolved
    assert default == word
    stored = [option[0] if isinstance(option, tuple) else option
              for option in options]
    assert word in stored
    assert kind


def test_a_key_that_is_not_in_the_table_declines():
    """An unrelated setting name never reaches the value-keyed vocabularies."""
    assert settings_spec._value_special_cases("nothing_like_it", "words") is None
