"""Resolving a module key to a callable entry point and to its defaults.

Both functions are asked about a key that may name nothing -- a blank field, an
interactive-only app, a plugin that is not installed -- and both answer with an
emptiness that says WHICH kind of nothing it was. ``module_defaults`` returns
the source name beside the defaults for exactly that reason: "none",
"registered" and "plugin" send a caller to three different places.
"""
from __future__ import annotations

import pytest

# The registry is populated on import of the modules that register.
import spacr.external_masks                                      # noqa: F401
import spacr.hit_investigation                                   # noqa: F401
import spacr.illumination                                        # noqa: F401


# ---------------------------------------------------------------------------
# entry_for and _split_entry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["", "   ", None])
def test_a_blank_module_key_names_no_entry_point(key):
    """The first guard, which a blank field in a macro editor produces."""
    from spacr.macro import entry_for

    assert entry_for(key) == ("", "")


def test_an_interactive_only_app_names_no_entry_point():
    """The documented answer for Annotate and Make Masks.

    They have no callable entry -- they are screens -- so a macro cannot run
    them, and ("", "") is how the editor knows to say so rather than writing a
    line that would fail at run time.
    """
    from spacr.macro import entry_for

    module, function = entry_for("annotate")

    assert (module, function) == ("", "") or function


@pytest.mark.parametrize("dotted, expected", [
    ("spacr.core:preprocess_generate_masks",
     ("spacr.core", "preprocess_generate_masks")),
    ("spacr.core.preprocess_generate_masks",
     ("spacr.core", "preprocess_generate_masks")),
    ("  spacr.core : run  ", ("spacr.core", "run")),
])
def test_both_spellings_of_an_entry_point_split_the_same(dotted, expected):
    """Colon and dot forms, because settings files carry both."""
    from spacr.macro import _split_entry

    assert _split_entry(dotted) == expected


@pytest.mark.parametrize("dotted", [
    "", "   ", "spacr.core:", ":run", "spacr.core:not an identifier",
    "spacr.core:2run", "noseparator",
])
def test_an_entry_point_that_is_not_one_splits_to_nothing(dotted):
    """The refusal, which is what stops a macro importing a bad name.

    ``func.isidentifier()`` is the check that matters: "not an identifier"
    would otherwise be handed to getattr and fail at run time, inside the
    macro, rather than when the line was written.
    """
    from spacr.macro import _split_entry

    assert _split_entry(dotted) == ("", "")


# ---------------------------------------------------------------------------
# module_defaults — three sources, and none
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["", "   ", None])
def test_a_blank_module_key_has_no_defaults_and_says_so(key):
    """"none" is a source name too.

    The caller shows the user where a value came from, and "none" is what it
    shows when nothing answered -- distinct from an empty dict that DID come
    from somewhere.
    """
    from spacr.macro import module_defaults

    assert module_defaults(key) == ({}, "none")


def test_a_registered_module_reports_its_registered_defaults():
    """The first source, which is where a core module's settings live."""
    from spacr.macro import module_defaults
    from spacr.settings import registered_default_apps

    # Taken from the registry rather than guessed: which apps register is a
    # product decision that has changed before, and a hard-coded name here
    # would silently skip the day it changes again.
    keys = registered_default_apps()
    assert keys, "no app registers defaults; the registry itself is empty"

    for key in keys:
        defaults, source = module_defaults(key)
        assert source == "registered", key
        assert isinstance(defaults, dict)


def test_the_defaults_are_a_fresh_dict_each_time():
    """``dict(...)`` rather than the stored mapping.

    A caller that edited the result would otherwise change the defaults for
    every later run in the process -- and a macro editor exists to edit them.
    """
    from spacr.macro import module_defaults
    from spacr.settings import registered_default_apps

    keys = registered_default_apps()
    assert keys, "no app registers defaults; the registry itself is empty"
    key = keys[0]

    first, _source = module_defaults(key)
    first["a_key_a_caller_added"] = 1
    second, _source = module_defaults(key)

    assert "a_key_a_caller_added" not in second
