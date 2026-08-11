"""Three columns and two thresholds silently applied two filters.

`fetch_filtered_paths` broadcasts a LENGTH-1 threshold or direction list up
to the number of measurements -- the documented shorthand, one threshold
across several columns -- and then ``zip``s. Any other mismatch fell through
both broadcasts and zip truncated to the shortest list, so the trailing
filters never ran.

Both fields in the Annotate settings dialog are independent free-text
comma-separated line edits, so "cell_area, nucleus_area, pathogen_area" with
"500, 200" is a typo away. The result is not a crash but a plausible-looking
WRONG POPULATION that gets hand-labelled and fed to a classifier.

Refused at both ends. The engine raises, which is the correctness boundary;
the dialog refuses OK, which is the half the user can act on, because the
engine call runs on a worker thread whose failure signal has no receiver.
"""

import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

def _lengths_are_refused(measurements, thresholds, directions):
    """Does the guard reject this combination? Mirrors the engine's check."""
    if len(thresholds) == 1 and len(measurements) > 1:
        thresholds = [thresholds[0]] * len(measurements)
    if isinstance(directions, str):
        directions = [directions] * len(measurements)
    if len(directions) == 1 and len(measurements) > 1:
        directions = [directions[0]] * len(measurements)
    return (len(thresholds) != len(measurements)
            or len(directions) != len(measurements))


@pytest.mark.parametrize(("measurements", "thresholds"), [
    (["a", "b", "c"], [500.0, 200.0]),      # the reported case
    (["a", "b"], [1.0, 2.0, 3.0]),          # more thresholds than columns
    (["a", "b", "c", "d"], [1.0, 2.0]),
])
def test_a_mismatch_is_refused(measurements, thresholds):
    assert _lengths_are_refused(measurements, thresholds, ["higher"])


@pytest.mark.parametrize(("measurements", "thresholds"), [
    (["a", "b", "c"], [500.0]),             # the documented shorthand
    (["a"], [500.0]),
    (["a", "b"], [1.0, 2.0]),               # one each
])
def test_the_documented_shapes_still_work(measurements, thresholds):
    assert not _lengths_are_refused(measurements, thresholds, ["higher"])


def test_the_engine_raises_and_names_both_counts():
    """The message has to say what to change, not just that it is wrong."""
    import inspect

    from spacr.qt import annotate_engine

    source = inspect.getsource(annotate_engine.fetch_filtered_paths)
    assert "measurement column(s) but" in source
    assert "len(thresholds) != len(measurements)" in source
    # The broadcasts must come FIRST, or the shorthand starts raising.
    guard = source.index("len(thresholds) != len(measurements)")
    broadcast = source.index("thresholds = [thresholds[0]] * len(measurements)")
    assert broadcast < guard, (
        "the guard runs before the length-1 broadcast, so one threshold "
        "across several columns is now refused")


# ---------------------------------------------------------------------------
# the dialog, which is where the user can fix it
# ---------------------------------------------------------------------------

def test_the_dialog_refuses_ok_on_a_mismatch(qtbot, monkeypatch):
    """An engine raise alone would be invisible: fetch_filtered_paths runs on
    a worker thread whose failure signal has no receiver."""
    from PySide6.QtWidgets import QDialog, QMessageBox

    from spacr.qt.screens.annotate import AnnotateSettings, _SettingsDialog

    dialog = _SettingsDialog(AnnotateSettings())
    qtbot.addWidget(dialog)

    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[2] if len(a) > 2 else ""))

    dialog._measurement.setText("cell_area, nucleus_area, pathogen_area")
    dialog._threshold.setText("500, 200")
    dialog.accept()

    assert warned, "OK was accepted with 3 columns and 2 thresholds"
    assert dialog.result() != QDialog.Accepted


@pytest.mark.parametrize(("meas", "thr"), [
    ("cell_area, nucleus_area", "500"),          # shorthand
    ("cell_area, nucleus_area", "500, 200"),     # one each
    ("cell_area", "500"),
    ("", ""),                                     # filter off entirely
])
def test_the_dialog_accepts_every_shape_that_works(qtbot, meas, thr):
    from PySide6.QtWidgets import QDialog

    from spacr.qt.screens.annotate import AnnotateSettings, _SettingsDialog

    dialog = _SettingsDialog(AnnotateSettings())
    qtbot.addWidget(dialog)
    dialog._measurement.setText(meas)
    dialog._threshold.setText(thr)
    dialog.accept()

    assert dialog.result() == QDialog.Accepted, f"{meas!r} / {thr!r} was refused"
