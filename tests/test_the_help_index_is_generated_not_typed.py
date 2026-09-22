"""The Help search index is derived, and its ranking puts the name first.

Instruction 422. The instruction's own argument for this feature is also the
argument against getting it wrong: "a map that is hand-maintained will drift
the first time a setting is renamed, at which point the search box sends users
to the wrong place -- worse than not having it, because a search result is
believed."

So the two things worth pinning without a display are:

* the generated tables are GENERATED -- ``spacr/qt/help_api_index.py`` carries
  the sentence that says so, and its rows are consistent with the artefacts
  they were built from rather than with a list somebody typed;
* the ranking puts an exact name above a word of a name above a description
  hit, because the ordering is the difference between a search box and a
  lottery.

No widget is built here. ``spacr.qt.help_index`` imports no Qt at module
scope, on purpose, and this file is the test that keeps that true.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacr.qt.help_index import (
    HelpEntry, KIND_ORDER, build_index, provider_names, register_provider,
    score, search,
)

ROOT = Path(__file__).resolve().parents[1]


def entry(kind="setting", title="", subtitle="", description=""):
    """One entry, spelled out at each call site rather than fixtured."""
    return HelpEntry(kind=kind, title=title, subtitle=subtitle,
                     description=description)


def test_importing_the_index_does_not_import_qt():
    """The index must be buildable on a worker, so it may not need a display.

    ``help_index`` is imported at the top of this module; if it pulled Qt in
    it would already be in ``sys.modules``. The providers import their
    registries inside themselves, which is what makes that possible.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import spacr.qt.help_index; "
        "print('PySide6.QtWidgets' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=str(ROOT),
                         capture_output=True, timeout=300)
    assert out.stdout.decode().strip().endswith("False"), out.stderr.decode()


def test_an_exact_name_beats_a_word_of_a_name_beats_a_description():
    """The three bands, in the only order that makes the box usable.

    A description hit that outranked a name hit would bury a setting under
    the twenty settings whose help text happens to mention it.
    """
    exact = entry(title="diameter")
    word = entry(title="cell_diameter")
    prefix = entry(title="diameter_estimate_n_fields")
    described = entry(title="magnification",
                      description="Used with the cell diameter setting.")
    terms = ["diameter"]
    ranked = [score(e, terms) for e in (exact, word, prefix, described)]
    assert all(value is not None for value in ranked), ranked
    assert ranked[0] > ranked[1] >= ranked[2] > ranked[3]
    assert [e.title for e in search([described, prefix, word, exact],
                                    "diameter")][0] == "diameter"


def test_a_second_word_narrows_rather_than_widens():
    """Terms are ANDed: typing more must never return more."""
    rows = [entry(title="animation_blur"), entry(title="animation_speed")]
    assert len(search(rows, "animation")) == 2
    assert [e.title for e in search(rows, "animation blur")] == ["animation_blur"]


def test_a_description_hit_is_a_hit():
    """A user who does not know the name is the user who needs this most."""
    rows = [entry(title="merge_edge_pathogen_cells",
                  description="Merge cells that are touching the image edge.")]
    assert [e.title for e in search(rows, "touching")] == [
        "merge_edge_pathogen_cells"]


def test_letters_scattered_too_far_apart_are_not_a_match():
    """The ceiling that stopped a forty-character symbol matching seven letters.

    Without it, ``clldiam`` matched ``save_filename_map`` -- every letter
    present, in order, spread over the whole name -- and outranked
    ``cell_diameter``, which is the one the user meant.
    """
    rows = [entry(title="cell_diameter"),
            entry(kind="api", title="spacr.qt.folder_metadata.save_filename_map")]
    assert [e.title for e in search(rows, "clldiam")] == ["cell_diameter"]


def test_one_kind_cannot_fill_the_list():
    """Ten thousand API symbols must not crowd out the 39 modules."""
    rows = [entry(kind="api", title=f"spacr.mask.helper_{i}") for i in range(40)]
    rows.append(entry(kind="module", title="Mask"))
    found = search(rows, "mask", limit=12)
    assert found[0].kind == "module"
    assert sum(1 for e in found if e.kind == "api") <= 8


def test_a_provider_that_raises_costs_only_its_own_rows():
    """A broken kind must not take the search field down with it."""
    def boom():
        raise RuntimeError("no")

    register_provider("_broken_for_the_test", boom)
    register_provider("_fine_for_the_test",
                      lambda: [entry(kind="module", title="Still here")])
    try:
        rows = build_index(["_broken_for_the_test", "_fine_for_the_test"])
    finally:
        from spacr.qt import help_index

        help_index._PROVIDERS.pop("_broken_for_the_test", None)
        help_index._PROVIDERS.pop("_fine_for_the_test", None)
    assert [e.title for e in rows] == ["Still here"]


def test_every_shipped_kind_has_a_provider():
    """Adding a kind is registering a provider, not editing a switch."""
    assert set(KIND_ORDER) <= set(provider_names())


def test_the_generated_module_says_it_is_generated():
    """A file somebody might otherwise edit by hand has to say so."""
    text = (ROOT / "spacr" / "qt" / "help_api_index.py").read_text("utf-8")
    assert "Generated -- do not edit" in text.splitlines()[0]
    assert "tools/build_help_search_index.py" in text


def test_the_api_rows_are_the_published_symbols():
    """``API_ENTRIES`` is the API manifest, not a selection from it.

    The manifest is what the published pages are built from, so a symbol in
    one and not the other is a result that opens nothing, or a page nothing
    can find.
    """
    from spacr.qt.help_api_index import API_ENTRIES

    manifest = (ROOT / "docs" / "source" / "_static" / "i18n" / "api"
                / "en.json")
    if not manifest.is_file():
        pytest.skip("the API manifest has not been built in this tree")
    published = set(json.loads(manifest.read_text("utf-8"))["symbols"])
    indexed = {symbol for symbol, _summary in API_ENTRIES}
    assert indexed <= published
    private = {s for s in published
               if any(p.startswith("_") and p != "__main__"
                      for p in s.split("."))}
    assert published - private == indexed


def test_every_setting_consumer_is_addressable():
    """A consumer that Sphinx cannot address would be a dead link."""
    from spacr.qt.help_api_index import SETTING_CONSUMERS

    for key, consumers in SETTING_CONSUMERS.items():
        assert consumers, key
        for module, qualname in consumers:
            assert module.startswith("spacr."), (key, module)
            assert not qualname.startswith("_"), (key, qualname)
            assert "." not in qualname, (key, qualname)


def test_every_preference_row_says_what_it_does():
    """A row with no description can only be found by its exact caption.

    Measured while this was built: reading the finished dialog found a
    description for none of the 121 rows, because the dialog moves every
    tooltip into its hint strip. The generator catches them on the way past,
    and this is what says it still does.
    """
    from spacr.qt.help_api_index import PREFERENCE_ENTRIES

    assert len(PREFERENCE_ENTRIES) > 100
    missing = [label for label, _t, _o, tip in PREFERENCE_ENTRIES if not tip]
    assert missing == []
    for _label, _title, object_name, _tip in PREFERENCE_ENTRIES:
        assert object_name.startswith("PreferencesTab"), object_name


def test_every_word_the_index_writes_itself_can_be_translated():
    """A result list in English inside a Korean window is a half-built feature.

    Most of what a row shows is a NAME -- a setting key, a dotted symbol, a
    module caption -- and names are not translated. The rest is chrome this
    module writes: "Module" on 39 rows, "API reference" on more than ten
    thousand. Chrome has to carry the English template it was built from, so
    the row can be rendered in the reader's language when it is drawn.
    """
    from spacr.qt.help_index import (
        DESCRIPTION_READS, SUBTITLE_API, SUBTITLE_MODULE, SUBTITLE_PREFERENCE,
        rendered_description, rendered_subtitle,
    )

    own = (SUBTITLE_MODULE, SUBTITLE_API, SUBTITLE_PREFERENCE.split("{")[0],
           DESCRIPTION_READS.split("{")[0])
    entries = build_index()
    assert len(entries) > 1000
    untranslatable = [
        e for e in entries
        if any(word in e.subtitle for word in own) and not e.subtitle_source
    ]
    assert untranslatable == [], [e.subtitle for e in untranslatable[:5]]

    def stub(text, **values):
        rendered = f"<{text}>"
        return rendered.format(**values) if values else rendered

    for kind in ("module", "preference", "api"):
        one = next(e for e in entries if e.kind == kind)
        assert rendered_subtitle(one, stub).startswith("<"), kind
        assert rendered_subtitle(one, None) == one.subtitle, kind
        assert rendered_description(one, None) == one.description, kind


def test_a_row_whose_translation_fails_is_still_a_row():
    """The catalog is not allowed to take the result list down with it."""
    from spacr.qt.help_index import (
        SUBTITLE_PREFERENCE, rendered_description, rendered_subtitle,
    )

    row = HelpEntry(kind="preference", title="PNG resolution",
                    subtitle="Preferences ▸ Figures",
                    description="How large a saved figure is.",
                    subtitle_source=SUBTITLE_PREFERENCE,
                    subtitle_values=(("tab", "Figures"),))

    def explode(_text, **_values):
        raise RuntimeError("no catalog")

    assert rendered_subtitle(row, explode) == "Preferences ▸ Figures"
    assert rendered_description(row, explode) == "How large a saved figure is."

    wrong = HelpEntry(kind="module", title="Mask", subtitle="Module",
                      subtitle_source="{nobody} put this here")
    assert rendered_subtitle(wrong, None) == "{nobody} put this here"
    assert rendered_subtitle(
        HelpEntry(kind="module", title="Mask", subtitle="Module",
                  subtitle_source="{a} and {b}",
                  subtitle_values=(("a", "one"),)), None) == "Module"


def test_a_registry_that_will_not_answer_costs_only_the_rows_it_owns():
    """The shipped providers' own guards, not a stand-in for them.

    The guarantee the module claims is that a registry going wrong loses its
    own rows and nothing else. The test above proves it for a provider
    somebody registered; these are the four registries the shipped providers
    read, each refused in turn.
    """
    from spacr.qt import help_index
    from spacr.qt.screens import settings_model

    def boom(*_a, **_k):
        raise RuntimeError("the registry is not answering")

    saved = {name: getattr(settings_model, name)
             for name in ("get_tooltips", "get_categories",
                          "resolve_default_settings", "categories_for_app")}
    try:
        for name in saved:
            setattr(settings_model, name, boom)
        assert help_index.setting_entries() == []
        setattr(settings_model, "resolve_default_settings",
                saved["resolve_default_settings"])
        rows = help_index.setting_entries()
        assert rows, "the defaults answered, so there should be rows"
        assert all(e.subtitle and "▸" not in e.subtitle for e in rows[:20])
    finally:
        for name, value in saved.items():
            setattr(settings_model, name, value)


def test_a_module_that_cannot_say_whether_it_is_visible_is_still_offered():
    """Silence from the maturity filter is not a reason to hide a module."""
    from spacr.qt import app as qt_app
    from spacr.qt.help_index import module_entries

    def boom(_key):
        raise RuntimeError("no preference store here")

    saved = qt_app.app_is_visible
    try:
        qt_app.app_is_visible = boom
        rows = module_entries()
    finally:
        qt_app.app_is_visible = saved
    assert len(rows) == len(qt_app.APPS)


def test_the_generated_tables_being_absent_is_survivable(monkeypatch):
    """A tree where the generator has not run yet still opens.

    ``help_api_index`` is generated, so it can be missing in a checkout that
    has not built it. The three providers that read it answer with no rows
    instead of stopping the window.
    """
    import sys

    from spacr.qt import help_index

    monkeypatch.setitem(sys.modules, "spacr.qt.help_api_index", None)
    assert help_index.api_entries() == []
    assert help_index.preference_entries() == []
    assert help_index._settings_each_symbol_reads() == {}


def test_a_symbol_only_the_consumer_table_knows_is_offered_anyway():
    """A function that reads a setting but is not in the published manifest.

    It is still the answer to "what reads this setting", so it gets a row of
    its own, built from the consumer table alone.
    """
    from spacr.qt import help_index

    saved = help_index._settings_each_symbol_reads
    try:
        help_index._settings_each_symbol_reads = lambda: {
            "spacr.nowhere.invented": ["cell_diameter", "cell_min_size"],
        }
        rows = help_index.api_entries()
    finally:
        help_index._settings_each_symbol_reads = saved
    invented = [e for e in rows if e.title == "spacr.nowhere.invented"]
    assert len(invented) == 1
    row = invented[0]
    assert row.subtitle.endswith("cell_diameter, cell_min_size")
    assert row.description == "Reads cell_diameter, cell_min_size."
    assert row.payload["reads"] == "cell_diameter cell_min_size"


def test_a_tab_that_cannot_be_asked_about_is_not_offered():
    """A preference row whose page might not exist is the kind's dead link.

    The Fractal page is built only while that backdrop is on. If the answer
    cannot be got at all, the rows stay out: offering a row that opens
    Preferences and then cannot find its own tab is worse than not offering
    it.
    """
    from spacr.qt import help_index, theme

    assert help_index._tab_exists("PreferencesTabGeneral") is True
    saved = theme.spaceout_enabled
    try:
        theme.spaceout_enabled = lambda: True
        assert help_index._tab_exists("PreferencesTabFractal") is True
        theme.spaceout_enabled = lambda: False
        assert help_index._tab_exists("PreferencesTabFractal") is False

        def boom():
            raise RuntimeError("no theme store")

        theme.spaceout_enabled = boom
        assert help_index._tab_exists("PreferencesTabFractal") is False
    finally:
        theme.spaceout_enabled = saved


def test_an_empty_query_matches_nothing_and_an_entry_knows_its_own_haystack():
    """The two edges of the matcher, which every other test types past."""
    from spacr.qt.help_index import _subsequence_span

    one = entry(kind="setting", title="cell_diameter", subtitle="Mask ▸ Cells",
                description="How wide a cell is.")
    assert one.haystack == (
        "cell_diameter\nmask ▸ cells\nhow wide a cell is.")
    assert search([one], "   ") == []
    assert _subsequence_span("", "cell_diameter") == 0


def test_every_band_of_the_ranking_is_a_band():
    """The five bands, each shown by a query that lands in it and no other.

    The ordering between them is the difference between a search box and a
    lottery, and a band nothing exercises is a band nobody would notice
    breaking.
    """
    from spacr.qt.help_index import _term_score

    key = entry(kind="setting", title="cell_diameter",
                subtitle="Mask ▸ Cells", description="How wide a cell is.")
    dotted = entry(kind="api", title="spacr.core.preprocess_generate_masks")
    assert _term_score("cell_diameter", key) == 100.0
    assert _term_score("diameter", key) == 80.0
    assert _term_score("masks", dotted) == 80.0
    assert 60.0 < _term_score("cell_d", key) < 70.0
    assert 50.0 < _term_score("preproc", dotted) < 60.0
    assert _term_score("iamet", key) == 40.0
    assert _term_score("cldmt", key) is not None
    assert _term_score("wide", key) == 10.0
    assert _term_score("umap", key) is None

    far = entry(kind="api", title="a" + "x" * 30 + "bc")
    assert _term_score("abc", far) is None, "the fuzzy band has no ceiling"


def test_the_list_stops_at_the_limit_and_skips_a_hidden_module():
    """Two ends of what the user is handed: how many rows, and whose."""
    from spacr.qt import app as qt_app
    from spacr.qt.help_index import module_entries

    rows = [entry(kind="api", title=f"spacr.core.thing_{n}") for n in range(9)]
    assert len(search(rows, "thing", limit=3, per_kind=None)) == 3

    hidden = qt_app.APPS[0][0]
    saved = qt_app.app_is_visible
    try:
        qt_app.app_is_visible = lambda key: key != hidden
        offered = module_entries()
    finally:
        qt_app.app_is_visible = saved
    assert len(offered) == len(qt_app.APPS) - 1
    assert hidden not in [e.payload["app"] for e in offered]
