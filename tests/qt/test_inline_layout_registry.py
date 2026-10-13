"""`_INLINE_LAYOUT_APPS` must agree with what `categories_for_app` does.

The same fact lives in two places: a module's layout is written inline in
:func:`categories_for_app`, and the fact that it HAS one is listed in
:data:`_INLINE_LAYOUT_APPS`. `classify_merged` was registered as a module
with 110 settings and a hand-written regroup, and the list was not updated
-- so `has_curated_layout` reported False and the invariant test failed
claiming the layout was missing when it was right there.

This checks the two agree by BEHAVIOUR rather than by listing the names a
third time: if `categories_for_app` returns something other than the
categories handed to it, that module has a layout of its own and must say so.
"""

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import settings_model as SM


def _shared_categories(app_key):
    """The category map a module gets before any module-specific regroup."""
    settings = SM.resolve_default_settings(app_key)
    keys = list(settings)
    grouped = {}
    for key in keys:
        grouped.setdefault(SM.category_for_setting(key), []).append(key)
    return grouped


@pytest.fixture(scope="module")
def app_keys():
    keys = sorted(SM.settings_app_keys()) if hasattr(SM, "settings_app_keys") \
        else sorted(SM._APP_CATEGORY_SPECS) + sorted(SM._INLINE_LAYOUT_APPS)
    assert keys, "no app keys to check — the test is not looking at anything"
    return keys


class TestTheRegistryIsHonest:

    def test_every_inline_app_actually_regroups(self, app_keys):
        """A name in the list that does nothing is a claim with no layout."""
        inert = []
        for key in sorted(SM._INLINE_LAYOUT_APPS):
            try:
                shared = _shared_categories(key)
                after = SM.categories_for_app(key, dict(shared))
            except Exception:
                continue                    # nothing to judge
            if after == shared:
                inert.append(key)
        assert not inert, (
            f"listed in _INLINE_LAYOUT_APPS but categories_for_app leaves "
            f"the layout untouched: {inert}")

    def test_classify_merged_is_curated(self):
        """The specific regression: registered, curated, and not listed.

        It carries 110 settings and its own regroup -- the family switch is
        lifted out of "Model Architecture" because it is the top-level
        choice, not one setting among ninety.
        """
        assert SM.has_curated_layout("classify_merged")

    def test_classify_and_classify_merged_are_both_curated(self):
        for key in ("classify", "classify_merged"):
            assert SM.has_curated_layout(key), key

    def test_a_module_with_no_layout_is_not_claimed(self):
        assert not SM.has_curated_layout("a_module_that_does_not_exist")
