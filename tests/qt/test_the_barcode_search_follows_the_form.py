"""The barcode search follows the form, and only Apply writes to it.

Asked for as "the automatic settings mode should be like live mode" and then
answered, when the two readings were put side by side, as "Live search, Apply
by click": the search re-runs whenever a setting it reads changes, and the
form's values change only when Apply is pressed, so nothing typed is
overwritten.

Each test below pins one half of that sentence, plus the two things the
sentence does not say but the form forces: a path field emits on every
keystroke, so the re-run has to wait for the typing to stop; and a setting the
search does not read must not start one.

The searches run unthreaded, so one that starts also finishes before control
returns, and `search_finished` counts them exactly. The pause before a live
search is shortened for speed -- how long it is does not matter to any claim
here, only that edits inside it collapse into one search.
"""

from __future__ import annotations

import shutil

from spacr.qt.screens import map_barcodes

# The fixture and the screen builder the coincidence tests already use, so a
# change to how a Map Barcodes screen is set up for a search is made once.
from tests.qt.test_a_coincidence_is_never_reported_as_a_finding import (  # noqa: F401
    _screen,
    sequencing_run,
)

#: The pause used in these tests, in milliseconds.
PAUSE = 60
#: Long enough for a scheduled live search to have fired, with margin.
SETTLE = PAUSE * 6


def _live(qtbot, sequencing_run):
    """A screen and panel with a completed first search and a short pause.

    :returns: the screen, the panel and the list `search_finished` appends to.
    """
    screen, panel = _screen(qtbot, sequencing_run)
    panel._live_timer.setInterval(PAUSE)
    finished = []
    panel.search_finished.connect(finished.append)
    panel.start_search()
    assert len(finished) == 1 and finished[0] is not None, (
        "the first search did not finish, so nothing after it means anything")
    return screen, panel, finished


def _copy_of(path, tmp_path, name):
    """The same reference table at a new path, which is a changed setting."""
    target = tmp_path / name
    shutil.copyfile(path, target)
    return str(target)


def test_the_search_watches_exactly_what_it_reads(qtbot, qt_theme_applied,
                                                  sequencing_run):
    """Every setting on the form the search reads is listened to.

    `barcode_set` has no field on this form, so it cannot be listened to;
    everything else the plan reads must be, or editing it would leave the
    findings describing a form that no longer exists.
    """
    screen, panel = _screen(qtbot, sequencing_run)
    widgets = screen._settings_model._widgets
    readable = [key for key in map_barcodes._LIVE_SEARCH_KEYS if key in widgets]

    assert {"src", "grna_csv", "row_csv", "column_csv",
            "target_sequence"} <= set(readable)
    assert len(panel._watched) == len(readable)


def test_changing_a_reference_table_searches_again_without_a_click(
        qtbot, qt_theme_applied, sequencing_run, tmp_path):
    """The live half: an input changes, and the findings follow it."""
    screen, panel, finished = _live(qtbot, sequencing_run)

    screen._settings_model.set_value_for_key(
        "grna_csv", _copy_of(sequencing_run["grna_csv"], tmp_path, "g2.csv"))

    qtbot.waitUntil(lambda: len(finished) == 2, timeout=SETTLE * 10)
    assert finished[1] is not None, "the live search did not complete"


def test_a_setting_the_search_does_not_read_starts_nothing(
        qtbot, qt_theme_applied, sequencing_run):
    """Compression and output settings cannot change what the reads say."""
    screen, _panel, finished = _live(qtbot, sequencing_run)

    screen._settings_model.set_value_for_key("comp_level", 7)
    screen._settings_model.set_value_for_key("offset_start", 3)
    qtbot.wait(SETTLE)

    assert len(finished) == 1


def test_a_burst_of_edits_is_one_search_not_one_per_keystroke(
        qtbot, qt_theme_applied, sequencing_run):
    """Typing the anchor a character at a time costs exactly one search.

    Undebounced, each character would start a search and the last to finish
    would win -- which need not be the one matching what is on the screen.
    """
    screen, _panel, finished = _live(qtbot, sequencing_run)
    anchor = str(screen._settings_model.collect()["target_sequence"])
    model = screen._settings_model

    for end in range(len(anchor) - 6, len(anchor) + 1):
        model.set_value_for_key("target_sequence", anchor[:end] + "A")
    qtbot.waitUntil(lambda: len(finished) >= 2, timeout=SETTLE * 10)
    qtbot.wait(SETTLE)

    assert len(finished) == 2, f"{len(finished) - 1} searches for one burst"


def test_an_edit_that_ends_where_it_started_starts_nothing(
        qtbot, qt_theme_applied, sequencing_run):
    """What is compared is the inputs, not whether a field emitted."""
    screen, _panel, finished = _live(qtbot, sequencing_run)
    model = screen._settings_model
    anchor = str(model.collect()["target_sequence"])

    model.set_value_for_key("target_sequence", anchor + "A")
    model.set_value_for_key("target_sequence", anchor)
    qtbot.wait(SETTLE)

    assert len(finished) == 1


def test_a_form_nobody_has_searched_is_left_alone(qtbot, qt_theme_applied,
                                                  sequencing_run, tmp_path):
    """Opening Map Barcodes and typing a folder is not a request to read it."""
    screen, panel = _screen(qtbot, sequencing_run)
    panel._live_timer.setInterval(PAUSE)
    finished = []
    panel.search_finished.connect(finished.append)

    screen._settings_model.set_value_for_key(
        "grna_csv", _copy_of(sequencing_run["grna_csv"], tmp_path, "g3.csv"))
    qtbot.wait(SETTLE)

    assert finished == []
    assert not panel.is_searching()


def test_a_live_search_never_overwrites_a_value_typed_by_hand(
        qtbot, qt_theme_applied, sequencing_run, tmp_path):
    """The Apply half: a finished live search proposes and writes nothing.

    A window typed by hand is left in the form while a live search completes
    with a different proposal for that same setting.
    """
    screen, panel, finished = _live(qtbot, sequencing_run)
    model = screen._settings_model
    model.set_value_for_key("offset_start", 0)

    model.set_value_for_key(
        "column_csv",
        _copy_of(sequencing_run["column_csv"], tmp_path, "c2.csv"))
    qtbot.waitUntil(lambda: len(finished) == 2, timeout=SETTLE * 10)

    proposed = {key: value for key, _present, value in panel.proposed_changes()}
    assert "offset_start" in proposed and proposed["offset_start"] != 0, (
        "the live search did not propose a different window, so this test "
        "proves nothing")
    assert model.collect()["offset_start"] == 0


def test_what_apply_writes_does_not_start_another_search(
        qtbot, qt_theme_applied, sequencing_run):
    """Apply's own writes came from this search, so repeating it is waste.

    It would also wipe the summary of what was just written off the screen.

    THE WRITE HAS TO BE TO A SETTING THE SEARCH READS, or this proves nothing:
    writing the read window starts no search whether or not Apply adopts what
    it wrote. `propose_map_barcodes_settings` does write such settings -- a
    reference table's path, from the table the search actually loaded -- but
    this fixture's form already names those exact paths, so its proposal never
    differs there. The proposed change to the anchor is therefore placed in the
    panel's pending changes directly, which is the one thing Apply reads.
    """
    screen, panel, finished = _live(qtbot, sequencing_run)
    model = screen._settings_model
    anchor = str(model.collect()["target_sequence"])
    assert "target_sequence" in map_barcodes._LIVE_SEARCH_KEYS
    panel._changes = (("target_sequence", anchor, anchor + "A"),)

    written = panel.apply_proposal()
    qtbot.wait(SETTLE)

    assert written == ("target_sequence",)
    assert model.collect()["target_sequence"] == anchor + "A"
    assert len(finished) == 1, "Apply's own write started a search"


def test_asking_to_watch_twice_does_not_double_the_searches(
        qtbot, qt_theme_applied, sequencing_run, tmp_path):
    """Re-arming after a form rebuild must not connect a field twice."""
    screen, panel, finished = _live(qtbot, sequencing_run)
    before = len(panel._watched)

    panel._watch_the_form()
    screen._settings_model.set_value_for_key(
        "row_csv", _copy_of(sequencing_run["row_csv"], tmp_path, "r2.csv"))
    qtbot.waitUntil(lambda: len(finished) >= 2, timeout=SETTLE * 10)
    qtbot.wait(SETTLE)

    assert len(panel._watched) == before
    assert len(finished) == 2


def test_a_path_list_announces_a_load_without_calling_it_an_edit(qtbot):
    """The widget contract the live search depends on.

    A settings file that replaces a reference table has to reach the search,
    so `set_value` announces `contents_changed`. It must NOT announce
    `value_changed`, which marks the screen dirty: a load that did would make
    every screen opened from a settings file look edited.
    """
    from spacr.qt.widgets.file_list import FilePathListWidget

    widget = FilePathListWidget(single=True)
    qtbot.addWidget(widget)
    contents, edits = [], []
    widget.contents_changed.connect(lambda: contents.append(1))
    widget.value_changed.connect(lambda: edits.append(1))

    widget.set_value("/x/barcodes_row.csv")
    assert (len(contents), len(edits)) == (1, 0)

    widget.set_value("/x/barcodes_row.csv")
    assert len(contents) == 1, "an unchanged value announced a change"

    widget.add_paths(["/x/other_row.csv"])
    assert (len(contents), len(edits)) == (2, 1), (
        "a user edit must still be an edit, and a change")
