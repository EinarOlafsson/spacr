"""Finding a setting, and meeting a module a few settings at a time.

Offscreen, against rendered geometry: every assertion here is made about
form rows that a real ``AppScreen`` built, not about a dictionary that
stands in for one. That distinction is the reason the strip exists — the
question "is this setting on screen?" is answered by the QFormLayout, and a
model-level test would pass while a collapsed section hid every match.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QFormLayout

import spacr.settings as S
from spacr.qt.screens.settings_model import (
    CURATION_THRESHOLD,
    SettingsWidgets,
    categories_for_app,
    essential_keys,
    get_categories,
    has_curated_layout,
    needs_curated_layout,
    resolve_default_settings,
)
from spacr.qt.settings_search import (
    ALL,
    ESSENTIALS,
    SettingsSearchBar,
    disclosure_for,
    forget_disclosure,
    install,
)


@pytest.fixture(autouse=True)
def _clean_disclosure():
    """No test inherits another's Essentials/All choice."""
    forget_disclosure()
    yield
    forget_disclosure()


@pytest.fixture
def mask_screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    return screen


# ---------------------------------------------------------------------------
# 1. Search — the point is the description, not the key
# ---------------------------------------------------------------------------

def test_search_finds_a_key_by_its_tooltip_text(qtbot):
    """The word the user knows is in the description, not in the name.

    ``merge_edge_pathogen_cells`` is what a user is looking for when they
    say "a parasite is straddling two cells and they got fused". The word
    "straddling" appears nowhere in the key and nowhere in the label — only
    in the description — so a search that indexed either of those alone
    would return nothing and the setting would stay unreachable.
    """
    model = SettingsWidgets("measure")
    model.build_sections()

    key = "merge_edge_pathogen_cells"
    assert key in model._widgets
    assert "straddling" not in key
    assert "straddling" not in model._label_for(key).lower()
    assert "straddling" in model.plain_tooltip_for(key).lower()

    assert model.keys_matching("straddling") == [key]


def test_search_terms_narrow_rather_than_widen(qtbot):
    """A second word means "and also this", not "or anything like it"."""
    model = SettingsWidgets("mask")
    model.build_sections()
    one = set(model.keys_matching("cell"))
    two = set(model.keys_matching("cell diameter"))
    assert two, "two terms found nothing at all"
    assert two < one


def test_an_empty_query_matches_everything(qtbot):
    """Clearing the box restores the form without a special case."""
    model = SettingsWidgets("measure")
    model.build_sections()
    assert set(model.keys_matching("")) == set(model._widgets)
    assert set(model.keys_matching("   ")) == set(model._widgets)


def test_search_hides_the_rendered_row_not_just_the_field(mask_screen):
    """A filtered-out row takes its label with it.

    ``Section`` builds the label side as a wrapper widget it does not hand
    back, so hiding the field alone leaves a stranded label pointing at
    nothing. The check is made against the QFormLayout because that is the
    only thing that knows the two halves belong together.
    """
    bar = install(mask_screen)
    assert bar is not None
    bar.set_level(ALL)
    bar.set_query("merge pathogens")

    section, field = bar._index["merge_pathogens"]
    form = section.findChild(QFormLayout)
    assert form.isRowVisible(field)

    other_section, other_field = bar._index["n_jobs"]
    other_form = other_section.findChild(QFormLayout)
    assert not other_form.isRowVisible(other_field)


def test_a_narrowing_search_opens_the_sections_it_kept(mask_screen):
    """Telling a user there are three matches and then hiding all three
    behind collapsed headings is worse than not filtering at all."""
    bar = install(mask_screen)
    bar.set_level(ALL)
    assert not any(s.is_expanded() for s in mask_screen._settings_sections)

    bar.set_query("merge pathogens")
    holding = bar._index["merge_pathogens"][0]
    assert holding.is_expanded()
    # `isHidden`, not `isVisible`: the screen itself is never shown in an
    # offscreen test, which makes every descendant invisible for a reason
    # that has nothing to do with the filter.
    assert not holding.isHidden()
    emptied = bar._index["n_jobs"][0]
    assert emptied.isHidden()

    bar.set_query("")
    assert not holding.is_expanded(), (
        "clearing the box left the form splayed open instead of restoring it")


def test_a_search_that_matches_nothing_says_what_to_do(mask_screen):
    bar = install(mask_screen)
    bar.set_level(ALL)
    bar.set_query("zzzzz-no-such-setting")
    assert bar.visible_keys() == []
    text = bar.count_text().lower()
    assert "no setting matches" in text
    assert "clear" in text


# ---------------------------------------------------------------------------
# 2. Differs from default
# ---------------------------------------------------------------------------

def test_modified_reports_nothing_on_a_freshly_built_panel(qtbot):
    model = SettingsWidgets("measure")
    model.build_sections()
    assert model.modified_keys() == []


def test_modified_reports_exactly_what_changed(mask_screen):
    bar = install(mask_screen)
    bar.set_level(ALL)
    mask_screen._settings_model.set_value_for_key("n_jobs", 3)

    bar.set_modified_only(True)
    assert bar.visible_keys() == ["n_jobs"]
    assert "modified only" in bar.count_text()

    bar.set_modified_only(False)
    assert len(bar.visible_keys()) == len(bar.indexed_keys())


def test_modified_uses_the_same_equality_as_the_diff_dialog(qtbot):
    """A value round-tripped through CSV is not an edit.

    The run journal stores ``channels`` as the string ``"[0, 1, 2]"`` where
    a fresh panel holds the list. If this filter used ``==`` it would call
    that an edit while the settings diff called it unchanged, and the two
    views of one run would disagree.
    """
    from spacr.qt.settings_diff import _values_equal
    assert _values_equal("[0, 1, 2]", [0, 1, 2])

    model = SettingsWidgets("mask")
    model.build_sections()
    model.set_value_for_key("channels", "[0, 1, 2, 3]")
    assert "channels" not in model.modified_keys()


# ---------------------------------------------------------------------------
# 3. Progressive disclosure
# ---------------------------------------------------------------------------

def test_essentials_is_where_a_first_visit_starts(mask_screen):
    bar = install(mask_screen)
    assert bar.level() == ESSENTIALS
    shown = len(bar.visible_keys())
    total = len(bar.indexed_keys())
    assert 0 < shown < total
    assert f"of {total} settings" in bar.count_text()
    assert "All settings" in bar.count_text()


def test_all_settings_restores_every_row(mask_screen):
    bar = install(mask_screen)
    bar.set_level(ALL)
    assert len(bar.visible_keys()) == len(bar.indexed_keys())


def test_the_choice_is_remembered_per_module(mask_screen, qtbot):
    bar = install(mask_screen)
    bar.set_level(ALL)
    assert disclosure_for("mask") == ALL
    assert disclosure_for("measure") == ESSENTIALS, (
        "one module's choice leaked into another's")

    from spacr.qt.screens.app_screen import AppScreen
    second = AppScreen("mask")
    qtbot.addWidget(second)
    assert install(second).level() == ALL


def test_essentials_narrows_a_search_rather_than_fighting_it(mask_screen):
    """The two filters compose. With both on, what is shown is the
    intersection — never a match Essentials excluded, and never an
    essential the query did not match."""
    bar = install(mask_screen)
    bar.set_level(ESSENTIALS)
    essentials = set(bar.visible_keys())
    bar.set_query("channel")
    shown = set(bar.visible_keys())
    assert shown <= essentials
    assert shown <= set(mask_screen._settings_model.keys_matching("channel"))


def test_essential_keys_are_derived_from_the_layout(qtbot):
    """Derived, so a layout change cannot leave the essentials behind.

    The first group of a curated layout is by construction the "what you
    must set" group, and it is taken whole.
    """
    cats = categories_for_app("mask", get_categories())
    first_group = list(cats.values())[0]
    derived = essential_keys("mask", cats)
    assert set(first_group) <= set(derived)


@pytest.mark.parametrize("app_key", [
    "mask", "measure", "timelapse", "classify", "regression", "invasion",
    "cellpose_masks", "barcode_qc",
])
def test_essentials_is_a_real_subset_for_every_big_module(app_key):
    """Not empty (nothing to start from) and not everything (no disclosure)."""
    total = resolve_default_settings(app_key)
    essentials = [k for k in essential_keys(app_key) if k in total]
    assert essentials, f"{app_key} has no essential settings at all"
    assert len(essentials) < len(total), (
        f"{app_key} calls every one of its {len(total)} settings essential")


# ---------------------------------------------------------------------------
# 4. Curated layouts — the parametrised guard
# ---------------------------------------------------------------------------
#
# This is the test that fails by name when a module is added with real
# settings and no layout of its own. Falling back to the shared category map
# is not a layout: that map is keyed by what a setting IS (a path, a plot
# option, "Advanced"), not by what the module does with it, which is how
# Cellpose Masks came to render thirteen knobs under one "Cellpose" heading.

def _all_app_keys():
    from spacr.qt.app import APPS
    keys = [row[0] for row in APPS]
    # Reachable from the Tk GUI and the CLI, absent from the Qt home grid.
    keys.append("cellpose_all")
    return sorted(set(keys))


@pytest.mark.parametrize("app_key", _all_app_keys())
def test_every_module_with_real_settings_has_a_curated_layout(app_key):
    if not needs_curated_layout(app_key):
        pytest.skip(
            f"{app_key} renders {len(resolve_default_settings(app_key))} "
            f"settings, at or under the {CURATION_THRESHOLD} that can be one "
            "group whatever it is called")
    assert has_curated_layout(app_key), (
        f"{app_key} renders {len(resolve_default_settings(app_key))} "
        "settings with no layout of its own, so they land in whatever shared "
        "buckets they happen to fall into. Add an entry to "
        "`_APP_CATEGORY_SPECS` in spacr/qt/screens/settings_model.py."
    )


@pytest.mark.parametrize("app_key", sorted(
    __import__("spacr.qt.screens.settings_model", fromlist=["x"])
    ._APP_CATEGORY_SPECS))
def test_a_curated_layout_accounts_for_every_setting(app_key):
    """No module drops a setting off the end of its own layout.

    A key named in no group falls into the trailing "Other" bucket, which is
    not a heading anyone chose — it is the absence of one.
    """
    from spacr.qt.screens.settings_model import (_APP_HIDDEN_CATEGORIES,
                                                 _APP_HIDDEN_KEYS)
    defaults = resolve_default_settings(app_key)
    hidden = _APP_HIDDEN_CATEGORIES.get(app_key, set())
    # A key a module deliberately never shows is not one it "dropped off
    # the end". Timelapse keeps `timelapse` in its settings, at True, and
    # renders no control for it -- the module IS the timelapse one. The
    # distinction this test cares about is between a key nobody placed and
    # a key somebody decided not to place.
    never_shown = _APP_HIDDEN_KEYS.get(app_key, set())
    used = set(never_shown)
    for name, keys in categories_for_app(app_key, S.categories).items():
        if name in hidden:
            continue
        used.update(k for k in keys if k in defaults)
    assert not (set(defaults) - used), (
        f"{app_key} leaves {sorted(set(defaults) - used)} out of every group")


def test_a_layout_may_name_a_key_the_shared_map_never_heard_of():
    """A literal key in a spec outranks the shared category map.

    Barcode QC and Illumination register settings that are in no shared
    category at all. Filtering literals against that map sent all of Barcode
    QC's checks to the trailing "Other" bucket — the exact thing a layout
    exists to prevent.
    """
    rendered = categories_for_app("barcode_qc", S.categories)
    defaults = resolve_default_settings("barcode_qc")
    placed = {k for keys in rendered.values() for k in keys if k in defaults}
    assert "target_grnas_per_well" in placed
    assert set(defaults) <= placed


# ---------------------------------------------------------------------------
# 5. Installation and geometry
# ---------------------------------------------------------------------------

def test_the_strip_sits_outside_the_scroll_area(mask_screen):
    """A search box that scrolls away with its own results is a search box
    you have to scroll back up to reach."""
    bar = install(mask_screen)
    scroll = mask_screen._settings_scroll
    assert not scroll.isAncestorOf(bar)
    assert bar.parentWidget() is scroll.parentWidget()


def test_the_strip_is_a_thin_band_above_the_form_not_over_it(qtbot):
    """Rendered geometry, because nothing else catches this.

    Re-parenting the scroll area into the new container HIDES it -- that is
    what `setParent` does -- and a layout skips hidden children. So the
    QVBoxLayout saw one visible child, left the scroll area at the full-pane
    geometry it had as a splitter pane, and centred the strip on top of it:
    two widgets drawing over each other down the whole settings column.
    Every behavioural assertion in this file still passed, because form-row
    visibility knows nothing about where the row is on screen.
    """
    # Pinned, because this test measures pixels. The font scale is
    # persisted in QSettings, so it is ambient state rather than anything
    # this file sets: whatever the last run left behind decides it. At 1.5
    # the styled pane's padding scales with it and the strip sits at y=11,
    # so this test passed alone and failed after any other qt file -- and
    # the blame landed on whichever test drew the short straw.
    from PySide6.QtWidgets import QApplication

    from spacr.qt import theme
    from spacr.qt.preferences import get_font_scale, set_font_scale
    _scale = get_font_scale()
    set_font_scale(1.0)
    # The scale is baked into the stylesheet when it is built, and the app
    # stylesheet is applied once per session. Changing the preference alone
    # leaves the already-applied 1.5 padding on the pane, so it has to be
    # rebuilt for the new scale to reach the layout.
    _app = QApplication.instance()
    _app.setStyleSheet(theme.stylesheet())

    from spacr.qt.app import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1500, 950)
    window.show()
    qtbot.waitExposed(window)
    window._on_nav_selected("mask")
    qtbot.wait(300)

    screen = window._screens["mask"]
    bar = screen._settings_search
    scroll = screen._settings_scroll
    assert bar is not None

    try:
        _assert_thin_band(bar, scroll)
    finally:
        set_font_scale(_scale)
        _app.setStyleSheet(theme.stylesheet())


def _assert_thin_band(bar, scroll):
    assert bar.isVisible() and scroll.isVisible()
    assert not bar.geometry().intersects(scroll.geometry()), (
        f"the strip {bar.geometry()} overlaps the form {scroll.geometry()}")
    assert bar.y() == 0, "the strip is not at the top of the pane"
    assert scroll.y() == bar.height()
    # Two rows of controls, not a share of the pane.
    assert 0 < bar.height() < 120
    assert scroll.height() > 4 * bar.height()


def test_installing_twice_adds_one_strip(mask_screen):
    first = install(mask_screen)
    assert install(mask_screen) is first
    assert len(mask_screen.findChildren(SettingsSearchBar)) == 1


def test_a_screen_without_a_settings_form_is_left_alone(qtbot):
    from PySide6.QtWidgets import QWidget
    assert install(QWidget()) is None


def test_every_registered_widget_qss_block_reaches_the_stylesheet(qapp):
    """A bad palette key in a QSS block is invisible until someone looks.

    ``registered_widget_qss`` catches whatever a block raises, logs it and
    moves on -- which is right, since an unstyled widget must not take the
    window down. The cost is that a typo makes the block silently vanish:
    three blocks in this batch asked for ``palette["text_dim"]``, which the
    palette spells ``fg_dim``, and all three shipped with no stylesheet at
    all. The first one was found because Home started painting an opaque
    slab over the ambient backdrop; the other two were found by looking.

    So: render the real stylesheet and check every registered name is in
    it. It costs one call and closes the whole class.
    """
    import spacr.qt
    from spacr.qt import theme
    spacr.qt.register_self_registering_modules()

    names = theme.widget_qss_names()
    assert names, "no widget QSS blocks are registered at all"
    css = theme.stylesheet()
    missing = sorted(n for n in names if n not in css)
    assert not missing, (
        "these registered QSS blocks raised while rendering and were "
        f"swallowed, so their widgets ship unstyled: {missing}")


# ---------------------------------------------------------------------------
# 6. Fixed-alphabet multi-select
# ---------------------------------------------------------------------------

def test_train_channels_offers_exactly_r_g_b(qtbot):
    """The pipeline maps letters to planes with three membership tests and
    drops anything else in silence, so the control must not be able to
    express anything else."""
    from spacr.qt.screens.settings_model import _AlphabetSelect
    model = SettingsWidgets("classify")
    model.build_sections()
    widget = model._widgets["train_channels"]
    qtbot.addWidget(widget)
    assert isinstance(widget, _AlphabetSelect)
    assert widget.choices() == ("r", "g", "b")
    assert widget.get_value() == ["r", "g", "b"]
    assert model.collect()["train_channels"] == ["r", "g", "b"]


def test_the_alphabet_control_always_emits_canonical_order(qtbot):
    """``['b','r']`` and ``['r','b']`` select the same two planes but write
    two different model directories, so click order must not survive."""
    from spacr.qt.screens.settings_model import _AlphabetSelect
    widget = _AlphabetSelect(key="train_channels", default=[],
                             choices=(("r", "Red"), ("g", "Green"),
                                      ("b", "Blue")))
    qtbot.addWidget(widget)
    widget.set_value(["b", "r"])
    assert widget.get_value() == ["r", "b"]


def test_off_alphabet_values_cannot_be_selected(qtbot):
    from spacr.qt.screens.settings_model import _AlphabetSelect
    widget = _AlphabetSelect(key="train_channels", default=["r", "x", "red"],
                             choices=(("r", "Red"), ("g", "Green"),
                                      ("b", "Blue")))
    qtbot.addWidget(widget)
    assert widget.get_value() == ["r"]


def test_the_alphabet_control_reads_a_csv_string(qtbot):
    """Settings CSVs and the Live Preview both hand back text."""
    from spacr.qt.screens.settings_model import _AlphabetSelect
    widget = _AlphabetSelect(key="train_channels", default=None,
                             choices=(("r", "Red"), ("g", "Green"),
                                      ("b", "Blue")))
    qtbot.addWidget(widget)
    widget.set_value("['r', 'g']")
    assert widget.get_value() == ["r", "g"]
    widget.set_value("g,b")
    assert widget.get_value() == ["g", "b"]


def test_train_channels_round_trips_through_set_value_for_key(qtbot):
    model = SettingsWidgets("classify")
    model.build_sections()
    assert model.set_value_for_key("train_channels", ["g"])
    assert model.collect()["train_channels"] == ["g"]
