"""A toggle in the action row must not starve the settings column.

The bug this defends against was invisible for as long as every toggle
said "Live". It appeared the moment one said "Hyperparameter search".
"""

import pytest

from PySide6.QtWidgets import QHBoxLayout, QWidget

from spacr.qt.widgets.ai_toggle_label import ELIDE_ABOVE_PX, AiToggleLabel


@pytest.fixture
def toggle(qt_theme_applied, qtbot):
    """A toggle that is SHOWN.

    Qt defers the resize event for a widget that has never been shown, so
    a hidden one keeps its full text however small you make it and every
    elision assertion passes for the wrong reason. Ask me how I know.
    """
    def make(text):
        widget = AiToggleLabel(text=text)
        qtbot.addWidget(widget)
        widget.show()
        qt_theme_applied.processEvents()
        return widget
    return make


def _squeeze(widget, width, qapp):
    """Resize a shown widget and let Qt deliver the event."""
    widget.resize(width, widget.sizeHint().height())
    qapp.processEvents()


class TestMinimumWidth:

    def test_a_long_toggle_cannot_demand_its_full_text(self, toggle):
        """QLabel reports the full text width as its minimum, which makes
        the text a hard floor for every ancestor layout.

        Measured before the cap: "Hyperparameter search" asked for 281px,
        held the action row at 1109px, and left 60px for the whole settings
        column at a 1200px window -- a 290px card in a 50px viewport.
        """
        long = toggle("Hyperparameter search")
        assert long.sizeHint().width() > ELIDE_ABOVE_PX, (
            "the premise of this test is that the text is wider than the cap")
        assert long.minimumSizeHint().width() == ELIDE_ABOVE_PX

    def test_a_short_toggle_is_left_alone(self, toggle):
        """"Live" and "AI" were never the problem; capping them would
        shrink a hit target for no reason."""
        short = toggle("Live")
        assert short.minimumSizeHint().width() == short.sizeHint().width()
        assert short.minimumSizeHint().width() <= ELIDE_ABOVE_PX

    def test_the_size_hint_still_asks_for_the_whole_text(self, toggle):
        """Only the MINIMUM is capped. A row with room shows the full
        label; capping the hint too would elide it even on a wide window.
        """
        long = toggle("Hyperparameter search")
        assert long.sizeHint().width() > long.minimumSizeHint().width()

    def test_a_row_of_toggles_can_shrink_below_the_sum_of_their_texts(
            self, qt_theme_applied, qtbot):
        """The property that actually matters, stated on a real layout."""
        host = QWidget()
        qtbot.addWidget(host)
        row = QHBoxLayout(host)
        for text in ("Live", "Hyperparameter search", "AI"):
            row.addWidget(AiToggleLabel(text=text))
        natural = sum(row.itemAt(i).widget().sizeHint().width()
                      for i in range(row.count()))
        assert host.minimumSizeHint().width() < natural


class TestElision:

    def test_the_full_text_shows_when_there_is_room(self, toggle,
                                                    qt_theme_applied):
        long = toggle("Hyperparameter search")
        _squeeze(long, long.sizeHint().width(), qt_theme_applied)
        assert long.displayed_text() == "Hyperparameter search"

    def test_it_elides_when_squeezed_instead_of_clipping(self, toggle,
                                                         qt_theme_applied):
        """Clipping is QLabel's default and cuts a word mid-glyph. An
        ellipsis at least says that something was left out."""
        long = toggle("Hyperparameter search")
        _squeeze(long, ELIDE_ABOVE_PX, qt_theme_applied)
        assert "…" in long.displayed_text()

    def test_text_returns_the_logical_label_not_what_fits(self, toggle,
                                                          qt_theme_applied):
        """Callers ask what the toggle SAYS, not what survived the layout.

        Two existing tests compare this against "Live", and the AppScreen
        identifies its switches by it.
        """
        long = toggle("Hyperparameter search")
        _squeeze(long, ELIDE_ABOVE_PX, qt_theme_applied)
        assert long.text() == "Hyperparameter search"

    def test_a_new_translation_replaces_the_stored_text(self, toggle):
        """The language switch calls setText with a fresh translation, so
        the stored copy has to follow rather than be captured once."""
        widget = toggle("Live")
        widget.setText("Hyperparameter search")
        assert widget.text() == "Hyperparameter search"

    def test_re_eliding_does_not_eat_the_stored_text(self, toggle,
                                                     qt_theme_applied):
        """Without the guard, each elision would be stored as the new
        logical text and the label would shorten one character at a time."""
        long = toggle("Hyperparameter search")
        full = long.sizeHint().width()
        for width in (ELIDE_ABOVE_PX, 60, ELIDE_ABOVE_PX, full):
            _squeeze(long, width, qt_theme_applied)
        assert long.text() == "Hyperparameter search"
        assert long.displayed_text() == "Hyperparameter search"

    def test_toggling_still_works_while_elided(self, toggle,
                                               qt_theme_applied):
        long = toggle("Hyperparameter search")
        _squeeze(long, ELIDE_ABOVE_PX, qt_theme_applied)
        long.setChecked(True)
        assert long.isChecked() and long.text() == "Hyperparameter search"
