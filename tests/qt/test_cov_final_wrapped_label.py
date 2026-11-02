"""The empty state's wrapped label, fixed in width or narrowing with its column."""

from spacr.qt.widgets.empty_state import _WrappedLabel

SENTENCE = "Pick a source folder so the first field of view can be shown here"


def test_a_fixed_label_keeps_its_width_and_its_own_hints(qapp):
    label = _WrappedLabel(SENTENCE, wrap_width=300)
    assert label.width() == 300 and label.minimumWidth() == 300
    assert label.maximumWidth() == 300
    assert label.sizeHint().width() <= 300
    assert label.minimumSizeHint().width() <= 300


def test_a_narrowing_label_asks_for_its_widest_width(qapp):
    label = _WrappedLabel(SENTENCE, wrap_width=300, narrows=True)
    assert label.maximumWidth() == 300
    assert label.minimumWidth() == 0
    assert label.sizeHint().width() == 300
    assert label.sizeHint().height() > 0


def test_a_narrowing_label_needs_only_its_longest_word(qapp):
    label = _WrappedLabel(SENTENCE, wrap_width=300, narrows=True)
    longest = max(label.fontMetrics().horizontalAdvance(word)
                  for word in SENTENCE.split())
    assert longest <= label.minimumSizeHint().width() < 300
    assert label.minimumSizeHint().width() <= longest + 40


def test_a_narrowing_label_without_words_still_has_a_hint(qapp):
    label = _WrappedLabel("", wrap_width=300, narrows=True)
    assert label.minimumSizeHint().width() <= 300


def test_an_empty_state_without_a_subtitle_has_no_subtitle_label(qapp):
    from PySide6.QtWidgets import QLabel

    from spacr.qt.widgets.empty_state import EmptyState

    panel = EmptyState(title="Nothing here yet")
    assert panel.findChild(QLabel, "SubtitleSmall") is None
    assert panel.findChild(QLabel, "TitleHeading") is not None
