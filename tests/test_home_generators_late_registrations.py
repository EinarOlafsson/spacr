"""Where the Home-variant generators put an app no categorisation table names.

``spacr/resources/home/versions/_generators/common.py`` holds hand-written
band tables for the candidate Home screens, while apps register themselves
from their own modules. ``_with_late_registrations`` files a newly registered
app into the declared fallback band, refuses a fallback band that does not
exist (a renamed band would silently switch the net off), and refuses a table
that still names a retired app (its tile would open nothing).

The module is imported by its package path (a namespace package inside the
package data); importing it puts the repository root first on ``sys.path``,
so ``sys.path`` is put back afterwards.
"""
from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def common():
    """The generators' ``common`` module, imported as the generators do."""
    saved_path = list(sys.path)
    try:
        yield importlib.import_module(
            "spacr.resources.home.versions._generators.common")
    finally:
        sys.path[:] = saved_path


def test_late_registrations_land_in_the_fallback_band_or_are_refused(
        common, monkeypatch, subtests):
    """Unfiled apps join the fallback band; a missing band or a retired key raises, naming both."""
    monkeypatch.setattr(common, "all_keys",
                        lambda: ["mask", "measure", "brand_new"])
    cases = [
        ("an unfiled app joins the fallback band",
         [("Prepare", ["mask"]), ("Review", ["measure"])], "Review",
         [("Prepare", ["mask"]), ("Review", ["measure", "brand_new"])]),
        ("an app somebody filed stays where it was put",
         [("Prepare", ["mask", "brand_new"]), ("Review", ["measure"])],
         "Review",
         [("Prepare", ["mask", "brand_new"]), ("Review", ["measure"])]),
        ("a full table needs no fallback band at all",
         [("Prepare", ["mask", "brand_new", "measure"])], "Nowhere",
         [("Prepare", ["mask", "brand_new", "measure"])]),
        ("a fallback band that does not exist",
         [("Prepare", []), ("Analyse", [])], "Prepear",
         "the fallback band 'Prepear' is not one of ['Prepare', 'Analyse'], "
         "so 3 uncategorised app(s) could not be placed: "
         "['brand_new', 'mask', 'measure']"),
        ("a table that still names retired apps",
         [("Prepare", ["mask", "cell_extract"]),
          ("Review", ["measure", "sequencing"])], "Review",
         "categorisation still presents retired app keys: "
         "['cell_extract', 'sequencing']. Remove each folded module from the "
         "Home table and route it through its host screen instead."),
    ]
    for case_id, table, fallback, expected in cases:
        with subtests.test(msg=case_id):
            literal = [(title, list(keys)) for title, keys in table]
            if isinstance(expected, str):
                with pytest.raises(AssertionError) as caught:
                    common._with_late_registrations(table, fallback=fallback)
                assert str(caught.value) == expected
            else:
                result = common._with_late_registrations(table,
                                                         fallback=fallback)
                assert result == expected
                common.check_coverage(result)
            assert table == literal, "the literal table is never mutated"
