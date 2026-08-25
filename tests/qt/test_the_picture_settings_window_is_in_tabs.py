"""The picture-settings window is divided into tabs, and the cap says its cost.

ONE LONG FORM IS NOT A PANEL. Twenty-eight controls in a single column make
the reader scroll past every question they are not asking to reach the one
they are. The module screens already group their settings into categories;
this window reads the same way, off the same table, so a setting is in one
place and one place only.

TWO NUMBERS ARE ASKED FOR AS TWO NUMBERS. The percentile window used to be a
text box holding a bracketed pair -- a parsing problem handed to the user,
who could type `[1 99]` and get a fallback instead of the picture they
configured.

AND A CAP IS A DECISION ABOUT WHERE A LIMIT SITS. How many pages the reader
has to walk, how much memory the tab holds while they do, and how long the
cut takes are on screen nowhere else, so the cap control carries them.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFormLayout, QSpinBox, QTabWidget  # noqa: E402

from spacr.picture_settings import (ALL_KEYS, LOAD_IMAGES,  # noqa: E402
                                    STREAM_IMAGES, categories,
                                    category_of, montage_cap_cost)
from spacr.qt.widgets.percentile_pair import PercentilePair    # noqa: E402
from spacr.qt.widgets.picture_settings_dialog import (         # noqa: E402
    PictureSettingsDialog, picture_defaults)

pytestmark = pytest.mark.qt


@pytest.fixture
def dialog(qtbot):
    made = PictureSettingsDialog(mode=LOAD_IMAGES)
    qtbot.addWidget(made)
    return made


class TestTheTableCoversEverySetting:

    def test_every_offered_setting_is_on_exactly_one_tab(self):
        """A control on no tab is a control the user cannot reach."""
        placed = [key for _title, keys in categories() for key in keys]

        assert sorted(placed) == sorted(ALL_KEYS)
        assert len(placed) == len(set(placed))

    def test_a_setting_the_table_has_never_heard_of_still_gets_a_tab(self,
                                                                    monkeypatch):
        """Omission must not hide a setting added later."""
        import spacr.picture_settings as module

        monkeypatch.setattr(module, "ALL_KEYS",
                            tuple(ALL_KEYS) + ("something_new",))

        assert module.category_of("something_new") == module.UNGROUPED_TITLE

    def test_a_retired_key_leaves_no_empty_row(self, monkeypatch):
        """The spec names keys; ``ALL_KEYS`` decides which of them exist."""
        import spacr.picture_settings as module

        monkeypatch.setattr(module, "ALL_KEYS", ("percentiles",))
        placed = [key for _t, keys in module.categories() for key in keys]

        assert placed == ["percentiles"]


class TestTheWindowIsTabbed:

    def test_the_settings_are_shown_in_tabs(self, dialog):
        assert dialog.findChild(QTabWidget) is not None
        assert len(dialog.tab_titles()) == len(categories())

    def test_the_tabs_are_the_ones_the_table_names(self, dialog):
        assert dialog.tab_titles() == tuple(t for t, _k in categories())

    def test_every_control_is_still_built(self, dialog):
        """Grouping must not lose a setting."""
        assert set(dialog._editors) == set(ALL_KEYS)
        assert set(dialog._labels) == set(ALL_KEYS)

    def test_each_control_reports_the_tab_it_is_on(self, dialog):
        for key in ALL_KEYS:
            assert dialog.tab_of(key) == category_of(key)

    def test_every_tab_lays_its_settings_out_as_a_form(self, dialog):
        """The label-left/field-right shape the rest of spaCR uses."""
        forms = dialog.findChildren(QFormLayout)

        assert len(forms) == len(categories())

    def test_a_named_tab_can_be_brought_to_the_front(self, dialog):
        assert dialog.show_tab("Outline") is True
        assert dialog.tab_titles()[dialog._tabs.currentIndex()] == "Outline"

    def test_asking_for_a_tab_that_is_not_there_says_so(self, dialog):
        assert dialog.show_tab("Nothing like a tab") is False

    def test_a_tab_holding_greyed_settings_says_how_many(self, dialog):
        """Behind a tab, a greyed control's reason has to be findable."""
        dialog.set_mode(STREAM_IMAGES)
        index = dialog.tab_titles().index(category_of("image_type"))

        assert "not used by the chosen image source" in \
            dialog._tabs.tabToolTip(index)

    def test_a_tab_this_mode_uses_entirely_says_nothing(self, dialog):
        dialog.set_mode(LOAD_IMAGES)
        index = dialog.tab_titles().index("Outline")

        assert dialog._tabs.tabToolTip(index) == ""

    def test_a_tab_stays_selectable_when_every_setting_on_it_is_greyed(
            self, dialog):
        """A tab that cannot be opened is a reason that cannot be read."""
        dialog.set_mode(LOAD_IMAGES)
        index = dialog.tab_titles().index("Channels")

        assert dialog._tabs.isTabEnabled(index)


class TestThePercentilesAreTwoFields:

    def test_the_control_is_a_pair_of_numeric_fields(self, dialog):
        editor = dialog._editors["percentiles"]

        assert isinstance(editor, PercentilePair)
        assert editor.low().maximum() <= 100.0
        assert editor.high().minimum() >= 0.0

    def test_it_opens_on_the_shipped_window(self, dialog):
        assert dialog.values()["percentiles"] == [2, 98]
        assert picture_defaults()["percentiles"] == [2, 98]

    def test_the_stored_value_is_still_a_pair_not_a_sentence(self, dialog):
        """Every settings file on disk holds two numbers; so does this."""
        value = dialog.values()["percentiles"]

        assert isinstance(value, list) and len(value) == 2

    def test_a_bracketed_value_from_an_old_settings_file_is_migrated(self,
                                                                    qtbot):
        made = PictureSettingsDialog(mode=LOAD_IMAGES,
                                     values={"percentiles": "[1 99]"})
        qtbot.addWidget(made)

        assert made.values()["percentiles"] == [1, 99]

    def test_the_low_field_cannot_be_pushed_past_the_high_one(self, qtbot):
        """An inverted window is not a value the panel can be left holding."""
        pair = PercentilePair([2, 98])
        qtbot.addWidget(pair)

        pair.low().setValue(99.0)

        low, high = pair.value()
        assert low <= high

    def test_the_high_field_cannot_be_pushed_below_the_low_one(self, qtbot):
        pair = PercentilePair([10, 90])
        qtbot.addWidget(pair)

        pair.high().setValue(1.0)

        low, high = pair.value()
        assert low <= high

    def test_setting_a_whole_new_window_is_not_clamped_by_the_old_one(self,
                                                                     qtbot):
        """Each field's range is pinned to the other, so both have to move."""
        pair = PercentilePair([40, 60])
        qtbot.addWidget(pair)

        pair.set_value([80, 95])

        assert pair.value() == [80, 95]

    def test_a_whole_percentile_stays_a_whole_number(self, qtbot):
        """The annotator ships integers; answering floats rewrites every file."""
        pair = PercentilePair([2, 98])
        qtbot.addWidget(pair)

        assert pair.value() == [2, 98]

    def test_a_fractional_percentile_is_kept(self, qtbot):
        """spaCR's own normalisation walks 99, 99.9, 99.99."""
        pair = PercentilePair([0.5, 99.9])
        qtbot.addWidget(pair)

        assert pair.value() == [0.5, 99.9]

    def test_changing_a_field_announces_the_pair(self, qtbot):
        pair = PercentilePair([2, 98])
        qtbot.addWidget(pair)
        heard = []
        pair.changed.connect(heard.append)

        pair.low().setValue(5.0)

        assert heard[-1] == [5, 98]


class TestTheCapSaysWhatItCosts:

    def test_the_cost_names_pages_memory_and_time(self):
        said = montage_cap_cost(10000)

        assert "pages" in said
        assert "MB" in said
        assert "s to cut" in said

    def test_more_objects_means_more_pages_and_more_memory(self):
        """The sentence has to move with the number or it is decoration."""
        small = montage_cap_cost(300)
        large = montage_cap_cost(10000)

        assert small != large
        assert "10,000 objects" in large
        assert "300 objects" in small

    @pytest.mark.parametrize("nothing", [0, -1, None, "x"])
    def test_a_cap_that_is_not_a_count_says_nothing(self, nothing):
        assert montage_cap_cost(nothing) == ""

    def test_the_cap_control_carries_the_cost(self, dialog):
        assert "objects is" in dialog._labels["cap"].toolTip()

    def test_the_cost_follows_the_number_being_edited(self, dialog):
        cap = dialog._editors["cap"]
        assert isinstance(cap, QSpinBox)

        cap.setValue(min(5000, cap.maximum()))

        assert f"{min(5000, cap.maximum()):,} objects is" in \
            dialog._labels["cap"].toolTip()

    def test_trying_three_caps_leaves_one_sentence_not_three(self, dialog):
        cap = dialog._editors["cap"]
        for value in (400, 900, 1500):
            cap.setValue(min(value, cap.maximum()))

        assert dialog._labels["cap"].toolTip().count("objects is") == 1
