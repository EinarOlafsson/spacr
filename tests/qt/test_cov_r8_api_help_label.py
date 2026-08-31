"""`ApiHelpLabel` -- a label whose hover carries a documentation link.

NO TEST FILE MENTIONED THIS MODULE. It was one of four in the package
that nothing referenced, and the only one of the four that is code
rather than a translation table: 63 statements at 20.55%, reached only
incidentally by whatever imported it.

What is worth holding here is mostly about restraint. A label with no
module to link to must NOT fall back to the documentation index -- that
is a link answering a question the reader did not ask. And the hover
filter must be installed exactly once, because Qt keeps a LIST of event
filters and calls each installation separately, so a second one pops two
tooltips for a single hover.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import Qt

from spacr.qt.widgets.api_help_label import ApiHelpLabel

pytestmark = pytest.mark.qt


@pytest.fixture()
def label(qtbot):
    def make(text="Cell diameter", app_key="mask"):
        widget = ApiHelpLabel(text, app_key)
        qtbot.addWidget(widget)
        return widget
    return make


class TestWhatTheLabelSpeaksFor:

    def test_a_label_with_a_module_records_it_as_a_property(self, label):
        """The property is what the QSS and the tooltip machinery read."""
        lbl = label(app_key="mask")
        assert lbl.property("moduleApiAppKey") == "mask"
        assert lbl.text() == "Cell diameter"

    def test_a_label_with_no_module_links_to_nothing(self, label):
        """THE RESTRAINT. `format_tooltip` would fall back to the index.

        A link to the documentation index answers no question the reader
        asked, so a label with nothing to link to shows its description
        and stops.
        """
        lbl = label(app_key="")
        assert lbl.url() == ""
        assert lbl.help_html() == "Cell diameter"
        assert lbl.property("moduleApiAppKey") in (None, "")

    def test_a_description_is_escaped_when_there_is_no_link(self, label):
        """It becomes rich text, so it has to stop being markup first."""
        lbl = label(text="width < height & <b>bold</b>", app_key="")
        assert lbl.help_html() == (
            "width &lt; height &amp; &lt;b&gt;bold&lt;/b&gt;")

    def test_setting_the_module_later_rebuilds_the_help(self, label):
        lbl = label(app_key="")
        assert lbl.url() == ""
        lbl.set_api_app_key("mask")
        assert lbl.property("moduleApiAppKey") == "mask"
        assert lbl.url(), "the label gained a module but not a link"

    def test_clearing_the_module_takes_the_link_away_again(self, label):
        lbl = label(app_key="mask")
        assert lbl.url()
        lbl.set_api_app_key("")
        assert lbl.url() == ""
        assert lbl.property("moduleApiAppKey") is None, (
            "an empty key must clear the property, not store an empty string")


class TestTheUrlOverride:

    def test_an_override_replaces_the_link_and_keeps_the_description(
            self, label):
        lbl = label(app_key="mask")
        lbl.set_url("https://example.invalid/custom")
        assert lbl.url() == "https://example.invalid/custom"
        assert "Cell diameter" in lbl.help_html()

    def test_an_override_equal_to_the_current_link_changes_nothing(
            self, label):
        """The early return, and it is not merely an optimisation.

        Appending an identical link would put the same URL on the label
        twice, one under the other.
        """
        lbl = label(app_key="mask")
        before = lbl.help_html()
        lbl.set_url(lbl.url())
        assert lbl.help_html() == before

    def test_an_empty_override_restores_the_module_link(self, label):
        lbl = label(app_key="mask")
        original = lbl.url()
        lbl.set_url("https://example.invalid/custom")
        assert lbl.url() != original
        lbl.set_url("")
        assert lbl.url() == original

    def test_an_override_is_escaped_into_the_href(self, label):
        """A URL is attribute content, so quotes have to be neutralised."""
        lbl = label(app_key="mask")
        lbl.set_url('https://example.invalid/a"onmouseover="x')
        assert '"onmouseover="' not in lbl.help_html()
        assert "&quot;" in lbl.help_html()

    def test_changing_the_module_drops_a_stale_override(self, label):
        """The override belonged to the old module's documentation."""
        lbl = label(app_key="mask")
        lbl.set_url("https://example.invalid/custom")
        lbl.set_api_app_key("measure")
        assert lbl.url() != "https://example.invalid/custom"


class TestRetranslation:

    def test_a_language_change_rebuilds_the_help(self, label):
        lbl = label(app_key="mask")
        lbl.retranslate_dynamic_content("de")
        assert lbl.help_html(), "the help was emptied by a language change"

    def test_a_language_change_drops_an_override(self, label):
        """The override's caption was written in the previous language."""
        lbl = label(app_key="mask")
        lbl.set_url("https://example.invalid/custom")
        lbl.retranslate_dynamic_content("de")
        assert lbl.url() != "https://example.invalid/custom"

    def test_a_falsy_language_means_no_language_was_named(self, label):
        """`None` and "" both mean "use whatever is active"."""
        lbl = label(app_key="mask")
        lbl.retranslate_dynamic_content("")
        assert lbl._language is None
        lbl.retranslate_dynamic_content(None)
        assert lbl._language is None


class TestTheHoverAffordances:

    def test_the_tooltip_carries_the_same_html_as_the_property(self, label):
        """The tooltip string is what the accessibility tree reads out."""
        lbl = label(app_key="mask")
        assert lbl.toolTip() == lbl.help_html()

    def test_the_tooltip_does_not_time_out(self, label):
        """-1 keeps the sticky popup up long enough to click the link."""
        lbl = label(app_key="mask")
        assert lbl.toolTipDuration() == -1

    def test_the_cursor_says_there_is_something_to_read(self, label):
        lbl = label(app_key="mask")
        assert lbl.cursor().shape() == Qt.WhatsThisCursor

    def test_the_hover_filter_is_installed_once_however_often_it_refreshes(
            self, label, monkeypatch):
        """TWO FILTERS POP TWO TOOLTIPS for one hover.

        Qt keeps a list of event filters and calls each installation
        separately, so the label removes its filter before installing it
        again. Asserted by counting installs against removals rather than
        by hovering, which cannot be observed offscreen.
        """
        lbl = label(app_key="mask")
        first = lbl._help_filter
        assert first is not None

        installs, removals = [], []
        monkeypatch.setattr(type(lbl), "installEventFilter",
                            lambda self, f: installs.append(f))
        monkeypatch.setattr(type(lbl), "removeEventFilter",
                            lambda self, f: removals.append(f))
        lbl.set_url("https://example.invalid/one")
        lbl.set_url("https://example.invalid/two")
        lbl.retranslate_dynamic_content("de")

        assert lbl._help_filter is first, "a second filter object was built"
        assert installs == [first, first, first]
        assert removals == [first, first, first], (
            "an install without a matching removal leaves two filters")
