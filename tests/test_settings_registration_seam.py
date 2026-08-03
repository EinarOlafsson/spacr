"""The defaults seam: `register_defaults` / `defaults_for`.

Item 0.1. `spacr/settings.py` is 3900 lines of defaults factories, types,
tooltips and categories, and every new module used to have to append to
all four -- in the one file six parallel workstreams need at the same
time.

A module registers its own instead. These tests hold the seam to the
same standard as the file it relieves: a registered key must be typed,
tooltipped and categorised for real (proved by running the shipped
`check_settings` over it, not by reading the dict back), a rejected
contribution must leave nothing behind, and none of the ~50 existing
`set_default_*` functions may change in any way.
"""
from __future__ import annotations

import pytest

import spacr.settings as S


# ---------------------------------------------------------------------------
# Fixtures — the shared tables are global, so every test puts them back
# ---------------------------------------------------------------------------

@pytest.fixture
def defaults_sandbox():
    """Restore the registry and the four shared declaration tables."""
    registry = dict(S._DEFAULTS_REGISTRY)
    types_ = dict(S.expected_types)
    tips = dict(S.tooltips)
    cats = {name: list(keys) for name, keys in S.categories.items()}
    descs = dict(S.descriptions)
    yield
    S._DEFAULTS_REGISTRY.clear()
    S._DEFAULTS_REGISTRY.update(registry)
    S.expected_types.clear()
    S.expected_types.update(types_)
    S.tooltips.clear()
    S.tooltips.update(tips)
    S.categories.clear()
    S.categories.update(cats)
    S.descriptions.clear()
    S.descriptions.update(descs)


class _Var:
    """The `.get()`-shaped stand-in the settings panel hands `check_settings`."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _vars_dict(**values):
    return {key: (None, None, _Var(value), None)
            for key, value in values.items()}


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------

def test_nothing_is_registered_until_a_module_registers():
    """The seam holds only what registers itself.

    The existing `set_default_*` functions are deliberately NOT
    auto-registered: they are reached through the dispatch in
    `gui_core.setup_settings_panel` and
    `qt.screens.settings_model.resolve_default_settings`, and mirroring
    them here would create a second answer to "what are Mask's
    defaults?" that could disagree with the first.
    """
    assert not S.has_registered_defaults("mask")
    assert not S.has_registered_defaults("measure")
    with pytest.raises(KeyError, match="no defaults registered"):
        S.defaults_for("mask")


def test_a_registered_factory_is_what_defaults_for_returns(defaults_sandbox):
    def factory(settings):
        settings = dict(settings)
        settings.setdefault("src", "")
        settings.setdefault("seam_bins", 32)
        return settings

    assert S.register_defaults("seam_probe", factory) is factory
    assert S.has_registered_defaults("seam_probe")
    assert "seam_probe" in S.registered_default_apps()
    assert S.defaults_for("seam_probe") == {"src": "", "seam_bins": 32}


def test_the_seed_settings_reach_the_factory(defaults_sandbox):
    """Same `settings` argument every `set_default_*` in this file takes."""
    S.register_defaults("seam_probe",
                        lambda settings: dict(settings, seam_bins=32))
    assert S.defaults_for("seam_probe", {"src": "/data"}) == {
        "src": "/data", "seam_bins": 32}
    # ...and it is copied on the way in, so a caller's dict is not edited.
    seed = {"src": "/data"}
    S.defaults_for("seam_probe", seed)
    assert seed == {"src": "/data"}


def test_every_caller_gets_its_own_dict(defaults_sandbox):
    """A factory that hands out one shared dict would let one module's
    screen edit another's defaults."""
    shared = {"seam_bins": 32}
    S.register_defaults("seam_probe", lambda settings=None: shared)

    first = S.defaults_for("seam_probe")
    first["seam_bins"] = 999
    assert S.defaults_for("seam_probe") == {"seam_bins": 32}
    assert shared == {"seam_bins": 32}


@pytest.mark.parametrize("factory, expected", [
    (lambda settings: dict(settings, shape="positional"),
     {"shape": "positional"}),
    (lambda settings=None: {"shape": "optional"}, {"shape": "optional"}),
    (lambda: {"shape": "no argument"}, {"shape": "no argument"}),
    (lambda *args, **kw: {"shape": "star"}, {"shape": "star"}),
])
def test_all_three_factory_shapes_in_this_file_work(factory, expected,
                                                    defaults_sandbox):
    """`set_default_plot_merge_settings()` takes nothing;
    `get_measure_crop_settings(settings=None)` takes an optional dict;
    `set_default_analyze_screen(settings)` requires one. All three are
    valid factories, resolved by signature rather than by calling and
    retrying on TypeError."""
    S.register_defaults("seam_probe", factory)
    assert S.defaults_for("seam_probe") == expected


def test_a_factory_that_does_not_return_a_dict_is_refused(defaults_sandbox):
    S.register_defaults("seam_probe", lambda settings=None: ["src"])
    with pytest.raises(TypeError, match="expected dict"):
        S.defaults_for("seam_probe")


def test_two_modules_cannot_quietly_claim_one_key(defaults_sandbox):
    S.register_defaults("seam_probe", lambda s=None: {"a": 1})
    with pytest.raises(ValueError, match="already registered"):
        S.register_defaults("seam_probe", lambda s=None: {"b": 2})
    assert S.defaults_for("seam_probe") == {"a": 1}

    S.register_defaults("seam_probe", lambda s=None: {"b": 2}, replace=True)
    assert S.defaults_for("seam_probe") == {"b": 2}


def test_a_non_callable_or_keyless_registration_is_refused(defaults_sandbox):
    with pytest.raises(TypeError, match="not callable"):
        S.register_defaults("seam_probe", {"src": ""})
    with pytest.raises(ValueError, match="need an app key"):
        S.register_defaults("", lambda s=None: {})
    # A refused registration leaves no trace. Asserted per key rather than
    # as "the registry is empty": real modules register at import time --
    # spacr.runctx does -- and a seam that exists to be used must not make
    # its own test fail the moment something uses it.
    registered = S.registered_default_apps()
    assert "seam_probe" not in registered
    assert "" not in registered


def test_unregister_removes_the_factory(defaults_sandbox):
    S.register_defaults("seam_probe", lambda s=None: {"a": 1})
    assert S.unregister_defaults("seam_probe") is True
    assert S.unregister_defaults("seam_probe") is False
    assert not S.has_registered_defaults("seam_probe")


# ---------------------------------------------------------------------------
# The declarations a module ships with its defaults
# ---------------------------------------------------------------------------

def test_a_registered_type_is_live_in_check_settings(defaults_sandbox):
    """The point of the merge: the shipped validator honours it.

    Without a type the key is not merely untyped -- `check_settings`
    reports "not found in expected types" and drops it, so a module that
    shipped defaults but no types would silently lose every setting the
    user edited.
    """
    before, errors = S.check_settings(_vars_dict(seam_bins="32"),
                                      S.expected_types)
    assert "seam_bins" not in before
    assert any("not found in expected types" in e for e in errors)

    S.register_defaults("seam_probe", lambda s=None: {"seam_bins": 32},
                        expected_types={"seam_bins": int})

    after, errors = S.check_settings(_vars_dict(seam_bins="32"),
                                     S.expected_types)
    assert after["seam_bins"] == 32
    assert not errors


def test_registered_tooltips_categories_and_description_are_merged(
        defaults_sandbox):
    S.register_defaults(
        "seam_probe", lambda s=None: {"seam_bins": 32},
        expected_types={"seam_bins": int},
        tooltips={"seam_bins": "(int) - Histogram bins. Default 32."},
        categories={"General": ["seam_bins"],
                    "Seam Probe": ["seam_bins"]},
        description="A registered module, for the test.")

    assert S.tooltips["seam_bins"].startswith("(int)")
    assert "seam_bins" in S.categories["General"]
    assert S.categories["Seam Probe"] == ["seam_bins"]
    assert S.descriptions["seam_probe"] == "A registered module, for the test."
    # An existing category keeps everything it already had.
    assert "src" in S.categories["Paths"]


def test_a_key_is_not_added_to_the_same_category_twice(defaults_sandbox):
    S.register_defaults("seam_a", lambda s=None: {},
                        categories={"General": ["seam_bins"]})
    S.register_defaults("seam_b", lambda s=None: {},
                        categories={"General": ["seam_bins", "seam_other"]})
    assert S.categories["General"].count("seam_bins") == 1
    assert S.categories["General"].count("seam_other") == 1


def test_redeclaring_another_modules_type_or_tooltip_is_refused(
        defaults_sandbox):
    """A module may add to the shared tables. It may not rewrite them."""
    with pytest.raises(ValueError, match="already declared"):
        S.register_defaults("seam_probe", lambda s=None: {},
                            expected_types={"src": int})
    with pytest.raises(ValueError, match="redefines the tooltip"):
        S.register_defaults("seam_probe", lambda s=None: {},
                            tooltips={"src": "mine now"})
    assert "mask" in S.descriptions
    with pytest.raises(ValueError, match="already has a description"):
        S.register_defaults("mask", lambda s=None: {},
                            description="not the mask description")
    assert not S.has_registered_defaults("mask")
    # Re-declaring the SAME type is fine — two modules may share a key.
    S.register_defaults("seam_probe", lambda s=None: {},
                        expected_types={"src": S.expected_types["src"]})
    assert S.has_registered_defaults("seam_probe")


def test_a_rejected_contribution_leaves_nothing_behind(defaults_sandbox):
    """Half-registered is harder to diagnose than failed-at-import."""
    with pytest.raises(ValueError):
        S.register_defaults(
            "seam_probe", lambda s=None: {"seam_bins": 32},
            expected_types={"seam_bins": int, "src": int},
            tooltips={"seam_bins": "(int) - Histogram bins."},
            categories={"General": ["seam_bins"]})

    assert not S.has_registered_defaults("seam_probe")
    assert "seam_bins" not in S.tooltips
    assert "seam_bins" not in S.categories["General"]
    assert S.expected_types["src"] == (str, list)


# ---------------------------------------------------------------------------
# What must not have changed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "set_default_settings_preprocess_generate_masks",
    "get_measure_crop_settings",
    "deep_spacr_defaults",
    "set_default_umap_image_settings",
    "get_perform_regression_default_settings",
    "set_default_plot_merge_settings",
])
def test_the_existing_defaults_factories_still_work_untouched(name):
    """Adding a seam must not disturb the ~50 factories already here."""
    fn = getattr(S, name)
    try:
        produced = fn({})
    except TypeError:
        produced = fn()
    assert isinstance(produced, dict) and produced
    assert not S.has_registered_defaults(name)


def test_the_qt_panel_still_resolves_every_built_in_module():
    """The dispatch the seam does not replace is still the one in use."""
    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import resolve_default_settings

    for key, *_rest in APPS:
        assert isinstance(resolve_default_settings(key), dict)
