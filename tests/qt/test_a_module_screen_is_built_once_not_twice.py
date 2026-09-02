"""Opening a module must not do the same work twice.

Building the Mask screen cost 2.7 s, and it was not one slow step -- it was
the same steps repeated. The screen restyled its whole widget tree twice for
a page colour that never showed, swept every descendant twice to make the
layout containers transparent, asked the preference store what language the
interface was in 3,516 times to render 1,538 settings, and scanned all 1,538
widgets once per row to answer "which setting is this field".

These tests assert the SHAPE that made it slow is gone -- how many times each
thing happens -- rather than how long it takes, because a wall-clock
assertion on a shared machine measures the neighbours. Each one fails
against the code as it was.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QLabel, QWidget

from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens import settings_model as sm


def _make_screen(qtbot, app_key: str) -> AppScreen:
    scr = AppScreen(app_key)
    qtbot.addWidget(scr)
    return scr


# ---------------------------------------------------------------------------
# The page colour is decided once
# ---------------------------------------------------------------------------

class TestThePageColourIsNotAppliedAndThenWithdrawn:
    """`_install_ambient` used to sync the page before recording the backdrop.

    `page_fill` answers "does this screen have to paint its own page?" by
    reading `self._ambient`. Assigning the backdrop AFTER the sync meant the
    sync ran against a screen that still looked backdrop-less: it set
    `QPalette.Window` and an `AppScreen { background-color: ... }` stylesheet
    on a widget the animation was about to cover, and the unconditional sync
    at the end of `__init__` -- by then seeing the backdrop -- took both off
    again. Nobody ever saw the colour, and the price was two full re-polishes
    of a 1,538-widget tree.
    """

    def test_a_screen_with_a_backdrop_styles_itself_at_most_once(
            self, qtbot, monkeypatch):
        """One `setStyleSheet` on the screen itself while it is built."""
        from spacr.qt import preferences

        preferences.set_ambient_enabled(True)
        calls = []
        real = AppScreen.setStyleSheet

        def _record(self, sheet):
            calls.append(sheet)
            return real(self, sheet)

        monkeypatch.setattr(AppScreen, "setStyleSheet", _record)
        scr = _make_screen(qtbot, "mask")

        assert scr._ambient is not None, (
            "this test is about the ambient path; the preference was set "
            "but no backdrop was installed")
        # Exactly the shape of the bug: a colour applied and then cleared is
        # two calls whose net effect is nothing.
        assert len(calls) <= 1, (
            f"the screen restyled itself {len(calls)} times while opening: "
            f"{calls!r}")

    def test_the_backdrop_is_recorded_before_the_page_is_resolved(
            self, qtbot, monkeypatch):
        """`page_fill` sees the backdrop the first time it is asked."""
        from spacr.qt import preferences

        preferences.set_ambient_enabled(True)
        seen = []
        real = AppScreen.page_fill

        def _record(self):
            seen.append(self._ambient is not None)
            return real(self)

        monkeypatch.setattr(AppScreen, "page_fill", _record)
        scr = _make_screen(qtbot, "mask")

        assert scr._ambient is not None
        assert seen, "page_fill was never called"
        assert all(seen), (
            "page_fill was asked while `_ambient` was still None, so it "
            "answered with a flat page colour for a screen that has a "
            "backdrop")

    def test_a_screen_without_a_backdrop_still_paints_its_page(
            self, qtbot):
        """The ordering fix must not cost the no-backdrop screen its colour."""
        from spacr.qt import preferences

        preferences.set_ambient_enabled(False)
        scr = _make_screen(qtbot, "mask")

        assert scr._ambient is None
        colour = scr.page_fill()
        # An image theme legitimately has no flat page colour; the assertion
        # is that whatever `page_fill` decides is what got applied.
        wanted = None if colour is None else colour.name()
        assert scr._page_applied == wanted

    def test_a_palette_event_waits_until_the_backdrop_is_decided(
            self, qtbot, monkeypatch):
        """A nested settings-build event must not resolve a partial screen.

        The flag is `_backdrops_ready`. This test set
        `_ambient_install_ready`, which the screen has never had under
        that name, so it created an unused attribute, the real guard
        stayed True, and the test failed on the behaviour it was written
        to protect. Setting a flag that does not exist is silent in
        Python, which is why this went unnoticed.
        """
        scr = _make_screen(qtbot, "mask")
        calls = []
        monkeypatch.setattr(scr, "_sync_page_palette",
                            lambda: calls.append("page"))

        assert hasattr(scr, "_backdrops_ready"), (
            "the guard has been renamed again; this test is asserting "
            "nothing until it is pointed at the new name")
        scr._backdrops_ready = False
        scr.changeEvent(QEvent(QEvent.PaletteChange))

        assert calls == []

    def test_a_palette_event_is_honoured_once_the_backdrop_is_decided(
            self, qtbot, monkeypatch):
        """The other side, so the guard cannot pass by never firing."""
        scr = _make_screen(qtbot, "mask")
        calls = []
        monkeypatch.setattr(scr, "_sync_page_palette",
                            lambda: calls.append("page"))

        scr._backdrops_ready = True
        scr.changeEvent(QEvent(QEvent.PaletteChange))

        assert calls == ["page"]


# ---------------------------------------------------------------------------
# The transparency sweep runs once
# ---------------------------------------------------------------------------

class TestTheContainersAreSweptOnceNotTwice:
    """`clear_container_surfaces` walks every descendant with `findChildren`.

    On the Mask screen that is ~8,200 widgets, and it was run twice: once by
    the ambient install and once unconditionally afterwards, over a tree
    nothing had been added to in between.
    """

    def test_the_ambient_path_sweeps_the_tree_once(self, qtbot, monkeypatch):
        from spacr.qt import preferences, theme

        preferences.set_ambient_enabled(True)
        calls = []
        real = theme.clear_container_surfaces
        monkeypatch.setattr(
            theme, "clear_container_surfaces",
            lambda root, *a, **k: (calls.append(root), real(root, *a, **k))[1])

        scr = _make_screen(qtbot, "mask")
        assert scr._ambient is not None
        assert len(calls) == 1, (
            f"the tree was swept {len(calls)} times while opening")

    def test_a_screen_with_no_backdrop_is_still_swept(
            self, qtbot, monkeypatch):
        """Skipping the second pass must not skip the only pass."""
        from spacr.qt import preferences, theme

        preferences.set_ambient_enabled(False)
        calls = []
        real = theme.clear_container_surfaces
        monkeypatch.setattr(
            theme, "clear_container_surfaces",
            lambda root, *a, **k: (calls.append(root), real(root, *a, **k))[1])

        scr = _make_screen(qtbot, "mask")
        assert scr._ambient is None
        assert len(calls) == 1, (
            "a screen with the ambient preference off was never swept, so "
            "every layout container keeps the opaque window fill")

    @pytest.mark.parametrize("app_key", ["mask", "map_barcodes"])
    @pytest.mark.parametrize("enabled", [True, False])
    def test_a_second_sweep_would_tag_nothing_new(
            self, qtbot, app_key, enabled):
        """The skip's real precondition, asserted rather than assumed.

        Dropping the second pass is only safe while the first one has
        already reached every container. Running the sweep AGAIN on the
        finished screen and finding nothing new is exactly that claim --
        and it holds for the DNA-rain screen too, which sweeps before it
        parents its backdrop and therefore still needs both passes.
        """
        from spacr.qt import preferences, theme

        preferences.set_ambient_enabled(enabled)
        scr = _make_screen(qtbot, app_key)

        def tagged():
            return {w for w in [scr] + scr.findChildren(QWidget)
                    if w.property(theme.TRANSPARENT_PROPERTY)}

        before = tagged()
        assert before, "no container was made transparent at all"
        theme.clear_container_surfaces(scr)
        assert tagged() == before, (
            "sweeping again tags containers the build missed, so skipping "
            "the second pass leaves an opaque slab on the page")

    def test_the_settings_column_shows_the_page_through(self, qtbot):
        """The named containers this screen tags by hand, both ways round."""
        from spacr.qt import preferences, theme

        for enabled in (True, False):
            preferences.set_ambient_enabled(enabled)
            scr = _make_screen(qtbot, "mask")
            for name in ("_settings_scroll", "_settings_content",
                         "_body_splitter", "_runtime_wrap"):
                widget = getattr(scr, name, None)
                if widget is None:
                    continue
                assert widget.property(theme.TRANSPARENT_PROPERTY), (
                    f"{name} still paints its own background with ambient="
                    f"{enabled}, so it covers the page")


# ---------------------------------------------------------------------------
# The language is resolved once per build
# ---------------------------------------------------------------------------

class TestTheLanguageIsAskedOncePerBuild:
    """`_language_code` answers the same question for every setting.

    It reaches `QSettings` through `preferences.get_language`, and building
    the Mask panel asked 3,516 times -- more than twice per setting.
    """

    def test_a_scope_reads_the_preference_once(self, monkeypatch):
        from spacr.qt import preferences

        reads = []
        real = preferences.get_language
        monkeypatch.setattr(
            preferences, "get_language",
            lambda: (reads.append(1), real())[1])

        with sm.language_resolved_once():
            codes = [sm._language_code() for _ in range(50)]

        assert len(set(codes)) == 1
        assert len(reads) == 1, (
            f"50 identical questions read the preference {len(reads)} times")

    def test_outside_a_scope_the_answer_is_never_stale(self, monkeypatch):
        """A cache with no invalidation would freeze the UI language."""
        from spacr.qt import preferences

        answers = iter(["en", "sv", "ko"])
        monkeypatch.setattr(preferences, "get_language", lambda: next(answers))
        assert sm._language_code() == "en"
        assert sm._language_code() == "sv"
        assert sm._language_code() == "ko"

    def test_a_scope_that_closes_drops_its_cache(self, monkeypatch):
        from spacr.qt import preferences

        monkeypatch.setattr(preferences, "get_language", lambda: "sv")
        with sm.language_resolved_once():
            assert sm._language_code() == "sv"
        monkeypatch.setattr(preferences, "get_language", lambda: "ko")
        with sm.language_resolved_once():
            assert sm._language_code() == "ko"

    def test_a_nested_scope_does_not_drop_the_outer_cache(self, monkeypatch):
        """A screen wraps its panel and `build_sections` wraps itself."""
        from spacr.qt import preferences

        reads = []
        real = preferences.get_language
        monkeypatch.setattr(
            preferences, "get_language",
            lambda: (reads.append(1), real())[1])

        with sm.language_resolved_once():
            sm._language_code()
            with sm.language_resolved_once():
                sm._language_code()
            sm._language_code()

        assert len(reads) == 1
        assert sm._LANGUAGE_SCOPE is None, "the scope leaked past its block"

    def test_an_exception_still_closes_the_scope(self, monkeypatch):
        with pytest.raises(ValueError):
            with sm.language_resolved_once():
                raise ValueError("boom")
        assert sm._LANGUAGE_SCOPE is None
        assert sm._TRANSLATION_MEMO is None

    def test_building_a_panel_asks_far_less_than_once_per_setting(
            self, qtbot, monkeypatch):
        """The measurement the whole change exists for."""
        from spacr.qt import preferences

        reads = []
        real = preferences.get_language
        monkeypatch.setattr(
            preferences, "get_language",
            lambda: (reads.append(1), real())[1])

        scr = _make_screen(qtbot, "mask")
        rows = len(scr._settings_model._widgets)
        assert rows >= 90, "expected the complete mask settings panel"
        # It used to be more than twice the number of settings.
        assert len(reads) < rows, (
            f"{len(reads)} language reads for {rows} settings")


class TestTheTranslationMemoAnswersTheSameThingTheCatalogsDo:
    """A memo that changed a string would be a mistranslation, not a speedup."""

    KEYS = ("src", "cell_channel", "nucleus_diameter", "plate", "n_jobs",
            "organelle_cellprob_threshold", "verbose")

    @pytest.mark.parametrize("language", ["en", "sv", "ko"])
    def test_a_tooltip_is_the_same_inside_and_outside_a_scope(self, language):
        for key in self.KEYS:
            plain = sm.format_tooltip("", "mask", key, language)
            with sm.language_resolved_once():
                scoped = sm.format_tooltip("", "mask", key, language)
            assert scoped == plain, f"{key} in {language}"

    @pytest.mark.parametrize("language", ["en", "sv", "ko"])
    def test_a_hint_is_the_same_inside_and_outside_a_scope(self, language):
        for key in self.KEYS:
            plain = sm.plain_tooltip("A description.", "mask", key, language)
            with sm.language_resolved_once():
                scoped = sm.plain_tooltip(
                    "A description.", "mask", key, language)
            assert scoped == plain, f"{key} in {language}"

    def test_two_settings_in_one_scope_keep_their_own_names(self):
        """The memo is keyed on the setting, not shared across the panel."""
        with sm.language_resolved_once():
            names = {key: sm._translated_setting_name(key, "sv", "mask")
                     for key in self.KEYS}
        assert len(set(names.values())) == len(self.KEYS), names

    def test_the_evaluation_and_umap_doc_keys_still_route(self):
        """Hoisting the two literal sets must not change where a key lands."""
        assert "/classifier_evaluation/" in sm.api_docs_url(
            "classify", "nested_cv_inner_folds")
        assert "/hyperparam/" in sm.api_docs_url("umap", "n_trials")
        assert "/batch_correction/" in sm.api_docs_url("mask", "batch_size_x")


# ---------------------------------------------------------------------------
# A field finds its setting through an index, not a scan
# ---------------------------------------------------------------------------

class TestAFieldFindsItsSettingWithoutScanning:
    """The panel used to walk all of `_widgets` once per row.

    1,538 rows against 1,538 widgets is over a million identity comparisons
    to answer 1,538 questions a dictionary answers outright.
    """

    def test_the_index_agrees_with_the_scan_it_replaced(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        widgets = scr._settings_model._widgets
        index = scr._widget_key_index()
        assert len(index) == len(widgets)
        for key, widget in widgets.items():
            scanned = next(k for k, w in widgets.items() if w is widget)
            assert index[id(widget)] == scanned

    def test_every_row_label_still_names_its_setting(self, qtbot):
        """What the scan was FOR: the label carries the key and the help."""
        scr = _make_screen(qtbot, "mask")
        widgets = scr._settings_model._widgets
        labelled = {}
        for label in scr._hint_map:
            key = label.property("settingKey")
            assert key in widgets, f"{key!r} is not a setting"
            labelled[key] = label
        assert set(labelled) == set(widgets)
        for key, label in labelled.items():
            assert widgets[key]._spacr_setting_label is label
            assert "<a href=" in scr._html_tip_map[label]

    def test_the_index_is_rebuilt_when_the_model_gains_a_setting(
            self, qtbot):
        """A stale index would quietly stop finding rows."""
        from PySide6.QtWidgets import QLineEdit

        scr = _make_screen(qtbot, "mask")
        first = scr._widget_key_index()
        extra = QLineEdit()
        scr._settings_model._widgets["a_setting_added_later"] = extra
        second = scr._widget_key_index()
        assert second is not first
        assert second[id(extra)] == "a_setting_added_later"

    def test_a_widget_under_two_names_keeps_the_first(self, qtbot):
        """The scan stopped at the first match; the index must too."""
        from PySide6.QtWidgets import QLineEdit

        scr = _make_screen(qtbot, "mask")
        shared = QLineEdit()
        scr._settings_model._widgets["aaa_first_name"] = shared
        scr._settings_model._widgets["zzz_second_name"] = shared
        assert scr._widget_key_index()[id(shared)] == "aaa_first_name"

    def test_key_of_still_answers_with_the_empty_string(self, qtbot):
        """`_key_of` is the older, string-valued spelling of the question."""
        from PySide6.QtWidgets import QLineEdit

        scr = _make_screen(qtbot, "mask")
        for key, widget in scr._settings_model._widgets.items():
            assert scr._key_of(widget) == key
        stranger = QLineEdit()
        qtbot.addWidget(stranger)
        assert scr._key_of(stranger) == ""

    def test_a_field_the_model_does_not_own_is_answered_with_none(
            self, qtbot):
        from PySide6.QtWidgets import QLineEdit

        scr = _make_screen(qtbot, "mask")
        stranger = QLineEdit()
        qtbot.addWidget(stranger)
        assert scr._key_of_field(stranger) is None

    def test_a_stale_index_is_rebuilt_rather_than_believed(self, qtbot):
        """An `id` is unique only among live objects."""
        from PySide6.QtWidgets import QLineEdit

        scr = _make_screen(qtbot, "mask")
        widgets = scr._settings_model._widgets
        key, widget = next(iter(widgets.items()))
        assert scr._key_of_field(widget) == key
        # Same number of settings, different widget behind the same key --
        # a length-only stamp cannot see this, so the check must.
        replacement = QLineEdit()
        qtbot.addWidget(replacement)
        widgets[key] = replacement
        assert scr._key_of_field(replacement) == key
        assert scr._key_of_field(widget) is None


# ---------------------------------------------------------------------------
# The hover filter answers a non-hover event without asking anything else
# ---------------------------------------------------------------------------

class _CountingLabel(QLabel):
    """A label that records how often its properties are read."""

    def __init__(self):
        super().__init__("counted")
        self.property_reads = 0

    def property(self, name):  # noqa: A003 - Qt API
        self.property_reads += 1
        return super().property(name)


class TestTheHoverFilterIgnoresWhatIsNotAHover:
    """The filter is installed on every settings label.

    Building the Mask panel put 14,472 events through it -- polish, style
    change, palette change, show -- before the pointer had moved at all, and
    each one paid for two module lookups and a `QObject.property` round trip
    to answer a question about hovering.
    """

    def test_a_style_change_costs_no_property_lookup(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        label = _CountingLabel()
        qtbot.addWidget(label)
        label.property_reads = 0
        scr.eventFilter(label, QEvent(QEvent.StyleChange))
        scr.eventFilter(label, QEvent(QEvent.Polish))
        scr.eventFilter(label, QEvent(QEvent.PaletteChange))
        assert label.property_reads == 0, (
            f"{label.property_reads} property lookups for three events "
            f"nobody hovered")

    def test_the_filter_still_lets_other_events_through(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        label = QLabel("plain")
        qtbot.addWidget(label)
        assert scr.eventFilter(label, QEvent(QEvent.StyleChange)) is False

    def test_a_hover_still_writes_the_hint_strip(self, qtbot):
        """The early return must not swallow the events that matter.

        Delivered through ``QApplication.sendEvent`` rather than by calling
        ``eventFilter``: the filter chain is what the early return is IN, so
        a test that calls the method directly would still pass if the
        filter had stopped being installed at all.
        """
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QEnterEvent
        from PySide6.QtWidgets import QApplication

        scr = _make_screen(qtbot, "mask")
        label = next(iter(scr._hint_map))
        scr._hint_strip.setText("")
        pos = QPointF(1.0, 1.0)
        QApplication.sendEvent(label, QEnterEvent(pos, pos, pos))
        qtbot.wait(10)
        assert scr._hint_strip.text(), "hovering a setting wrote nothing"
        assert "https://" in scr._hint_strip.text()

        QApplication.sendEvent(label, QEvent(QEvent.Leave))
        qtbot.wait(10)
        assert scr._hint_strip.text() == scr._default_hint()

    def test_leaving_puts_the_default_prompt_back(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        label = next(iter(scr._hint_map))
        scr.eventFilter(label, QEvent(QEvent.Enter))
        scr.eventFilter(label, QEvent(QEvent.Leave))
        assert scr._hint_strip.text() == scr._default_hint()

    def test_a_category_header_still_writes_its_own_blurb(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        headers = [w for w in scr.findChildren(QWidget)
                   if w.property("settingsCategory")]
        assert headers, "no category header carries its name"
        header = headers[0]
        scr.eventFilter(header, QEvent(QEvent.Enter))
        assert scr._category_hint.text()
        scr.eventFilter(header, QEvent(QEvent.Leave))


# ---------------------------------------------------------------------------
# Correctness was not traded for speed
# ---------------------------------------------------------------------------

class TestTheApplicabilityRefreshStillLands:
    """The rule the item names: a panel must never open showing settings for
    objects that are not in the run.

    The refresh is posted through ``QTimer.singleShot``, so anything that
    changes WHEN widgets are built has to keep it landing before the user
    sees the panel.
    """

    @staticmethod
    def _roles(model):
        """Every object the panel offers, named by its channel setting."""
        return sorted(key[: -len("_channel")] for key in model._widgets
                      if key.endswith("_channel"))

    def test_a_fresh_mask_panel_shows_one_row_per_object(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        model = scr._settings_model
        # Let the deferred refresh through; a probe that never spins the
        # loop sees a stale panel.
        qtbot.wait(50)

        hidden = set(model.keys_hidden_by_the_run())
        assert hidden, "nothing was hidden, so the refresh never landed"

        settings = model._object_visibility_settings()
        roles = self._roles(model)
        assert roles, "the mask panel offers no objects at all"
        assert all(settings.get(f"{role}_channel") is None for role in roles), (
            "this test is about a panel whose channels are all None")

        shown = {key for key in model._widgets if key not in hidden}
        for role in roles:
            mine = sorted(key for key in shown
                          if key == role or key.startswith(f"{role}_"))
            if role == "cell":
                # Cell is the reference object every other family is
                # configured against. Its controls stay reachable even when
                # the channel is not chosen yet, so a fresh run can be
                # configured without first rebuilding the form.
                assert f"{role}_channel" in mine
                assert f"{role}_diameter" in mine
                continue
            assert mine in ([], [f"{role}_channel"]), (
                f"{role} has no channel set, so the panel should show its "
                f"channel row and nothing else; it shows {mine}")

        # And at least one object really is on screen -- an assertion that
        # every object is hidden would pass vacuously above.
        offered = [role for role in roles
                   if f"{role}_channel" in shown]
        assert offered == list(sm.CHANNELLED_OBJECTS), (
            f"the offered mask roles {offered} no longer match the canonical "
            f"channelled roles {list(sm.CHANNELLED_OBJECTS)}")

    def test_no_object_detail_row_survives_a_channel_of_none(self, qtbot):
        """The failure mode the deferral must never reintroduce."""
        scr = _make_screen(qtbot, "mask")
        model = scr._settings_model
        qtbot.wait(50)
        hidden = set(model.keys_hidden_by_the_run())
        shown = {key for key in model._widgets if key not in hidden}
        assert shown, "the panel hid everything"

        for role in self._roles(model):
            if role == "cell":
                continue
            for suffix in ("diameter", "CP_prob", "FT", "background",
                           "Signal_to_noise"):
                key = f"{role}_{suffix}"
                if key in model._widgets:
                    assert key not in shown, (
                        f"{key} is shown although {role} has no channel")

    def test_the_rows_are_findable_by_key_after_the_refresh(self, qtbot):
        """`_widgets` is how several checks find a row; it must still hold
        every one, shown or hidden."""
        scr = _make_screen(qtbot, "mask")
        model = scr._settings_model
        qtbot.wait(50)
        hidden = set(model.keys_hidden_by_the_run())
        assert hidden <= set(model._widgets), (
            "a row was hidden that `_widgets` has never heard of")
        for key in hidden:
            assert model._widgets[key] is not None


class TestEveryModuleStillOpens:
    """The build-scope wrapper sits in front of every module's panel."""

    @pytest.mark.parametrize(
        "app_key", ["mask", "measure", "classify", "umap", "map_barcodes"])
    def test_the_panel_has_rows_and_sections(self, qtbot, app_key):
        scr = _make_screen(qtbot, app_key)
        assert scr._settings_model._widgets, f"{app_key} built no settings"
        assert scr._settings_sections, f"{app_key} built no sections"
        assert sm._LANGUAGE_SCOPE is None, "the build scope leaked"
