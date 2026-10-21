"""A model window without a fitted class set must not schedule phantom edits."""
from spacr.qt.screens import annotate


def test_unfitted_controls_do_not_emit_or_schedule_threshold_changes(qtbot):
    dialog = annotate._SuggestModelDialog()
    qtbot.addWidget(dialog)
    delivered = []
    dialog.thresholds_changed.connect(delivered.append)
    assert dialog._slider is None and dialog._low is None and dialog._high is None
    dialog._show_slider_value()
    dialog._on_band(False, emit=False)
    dialog._filling = True
    dialog._schedule()
    assert not dialog._debounce.isActive()
    assert delivered == []
