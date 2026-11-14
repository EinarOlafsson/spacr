"""How the correction module resolves a method name, or refuses one.

The degenerate-family tests of the correction primitives live in
``tests/test_multiple_testing_degenerate_families.py``.
"""

from __future__ import annotations

import pytest

from spacr import multiple_testing as mt

# ---------------------------------------------------------------------------
# canonical_method: the statsmodels-spelling fallback
# ---------------------------------------------------------------------------

def test_a_statsmodels_spelling_resolves_without_help_from_the_alias_table(
        monkeypatch):
    """The fallback loop, not the alias shortcut, has to do this on its own.

    ``holm-sidak`` is listed twice -- once in :data:`_ALIASES` and once as the
    method's ``statsmodels_name``. The alias answers first, which would hide a
    broken fallback until somebody added a method and forgot its alias. The
    alias is removed here so the fallback is the only thing left to answer.
    """
    aliases = dict(mt._ALIASES)
    aliases.pop("holm-sidak")
    aliases.pop("holm_sidak", None)
    monkeypatch.setattr(mt, "_ALIASES", aliases)
    assert mt.canonical_method("holm-sidak") == "holm_sidak"
    assert mt.canonical_method("Simes-Hochberg") == "simes_hochberg"


def test_a_method_nobody_implements_is_refused_with_the_inventory():
    """A silent fallback would apply a correction the user did not choose."""
    with pytest.raises(ValueError) as excinfo:
        mt.canonical_method("fdr_martian")
    message = str(excinfo.value)
    assert "fdr_martian" in message
    for key in ("fdr_bh", "bonferroni", "storey"):
        assert key in message
