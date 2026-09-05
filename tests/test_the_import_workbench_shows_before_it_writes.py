"""137 A, C and D: the drop table, the role dropdowns, and the live preview.

B -- inferring the regex from the names -- is
`tests/test_the_regex_is_worked_out_from_the_filenames.py`. What is tested
here is the half a user touches: files arrive by drop, each group is given a
MEANING from a dropdown rather than typed, and the answer is on screen before
anything is written.

THE FAULT ALL OF THIS EXISTS TO STOP is a regex that looked right and grouped
the files wrong -- so the assertions that matter most are the ones about what
is SAID: the unmatched files named rather than dropped in silence, and two
groups claiming one role refused on the spot rather than at run time.
"""
from __future__ import annotations

import os

import pytest

from spacr.import_plan import (CHANNEL_MEANINGS, REQUIRED, ROLES, ImportPlan,
                               group_names, plan, role_trouble)

CELLVOYAGER = (r"plate1_(?P<wellID>[A-Za-z0-9]+)_T0001F(?P<fieldID>\d+)"
               r"L01A01Z01C(?P<chanID>\d+)\.tif")


def _names(wells=("A01", "A02"), fields=(1, 2), channels=(1, 2)):
    return [f"plate1_{w}_T0001F{f:03d}L01A01Z01C{c:02d}.tif"
            for w in wells for f in fields for c in channels]


# --------------------------------------------------------------------------- #
#  The plan, headless
# --------------------------------------------------------------------------- #

class TestItSaysWhatWouldHappen:

    def test_every_file_gets_a_new_name(self):
        made = plan(_names(), CELLVOYAGER, plate="exp1")

        assert made.n_matched == 8
        assert all(row.after.endswith(".tif") for row in made.renamed)

    def test_the_new_name_is_the_import_s_OWN_rule(self):
        """Not a second opinion about it: `io._escaped_field_stem` is what
        writes the stack, down to the plate escaping that
        `schema.parse_field_stem` reads back."""
        from spacr.io import _escaped_field_stem

        made = plan(_names(), CELLVOYAGER, plate="exp_1")

        assert made.renamed[0].after == \
            _escaped_field_stem("exp_1", "A01", "001", "") + ".tif"

    def test_a_plate_with_an_underscore_is_escaped(self):
        """`exp_1_A01_1_1` is five components for a four-component grammar,
        and the plate could not be measured at all."""
        made = plan(_names(), CELLVOYAGER, plate="exp_1")

        assert "%5F" in made.renamed[0].after

    def test_the_tree_counts_at_every_level(self):
        made = plan(_names(), CELLVOYAGER, plate="exp1")

        tree = made.tree()
        assert list(tree) == ["exp1"]
        assert sorted(tree["exp1"]) == ["A01", "A02"]
        assert sum(tree["exp1"]["A01"]["001"].values()) == 2

    def test_the_tree_reads_as_text_with_the_counts_in_it(self):
        made = plan(_names(), CELLVOYAGER, plate="exp1")

        lines = made.tree_lines()
        assert any("2 well(s)" in line for line in lines)
        assert any("channel(s) 01, 02" in line for line in lines)

    def test_nothing_is_written(self, tmp_path):
        """`io._run_test_mode` answers this by COPYING real files. This
        answers it before the commitment, which is the point."""
        before = set(os.listdir(tmp_path))

        plan(_names(), CELLVOYAGER, plate=str(tmp_path))

        assert set(os.listdir(tmp_path)) == before


class TestAFileThatDoesNotMatchIsNamed:

    def test_it_is_listed_rather_than_dropped(self):
        names = _names() + ["a_stray_file.tif", "another.tif"]

        made = plan(names, CELLVOYAGER, plate="exp1")

        assert made.unmatched == ("a_stray_file.tif", "another.tif")

    def test_the_summary_counts_both_sides(self):
        """"412 of 480 matched" is an answer; 412 files appearing without
        comment is how half a plate goes missing."""
        made = plan(_names() + ["stray.tif"], CELLVOYAGER, plate="exp1")

        assert made.summary() == "8 of 9 file(s) matched; 1 did not."

    def test_an_empty_drop_says_so(self):
        assert "Drop images" in ImportPlan().summary()

    def test_a_half_typed_regex_is_not_a_traceback(self):
        """A user is in the middle of editing it; a preview must survive."""
        made = plan(_names(), r"(?P<wellID>[A-Z", plate="exp1")

        assert made.trouble
        assert "not a regex yet" in made.trouble[0]
        assert made.n_matched == 0
        assert len(made.unmatched) == 8

    def test_no_regex_at_all_matches_nothing_quietly(self):
        made = plan(_names(), "")

        assert made.n_matched == 0
        assert not made.trouble


class TestTheRolesAreAClosedSet:

    def test_the_required_three_are_the_ones_the_import_reads(self):
        assert set(REQUIRED) == {"wellID", "fieldID", "chanID"}

    def test_every_role_says_what_it_means(self):
        for value, why in ROLES:
            assert len(why) > 10, value

    def test_two_groups_claiming_one_role_is_refused_on_the_spot(self):
        """Not at run time, which is where it would otherwise appear -- as an
        import that read one group and silently ignored the other."""
        said = role_trouble({"a": "wellID", "b": "wellID", "c": "fieldID",
                             "d": "chanID"})

        assert any("a, b" in s and "wellID" in s for s in said)

    def test_a_missing_required_role_is_named(self):
        said = role_trouble({"a": "wellID"})

        assert any("fieldID" in s and "chanID" in s for s in said)

    def test_a_complete_assignment_is_silent(self):
        assert role_trouble({"a": "wellID", "b": "fieldID",
                             "c": "chanID"}) == ()

    def test_the_role_wins_over_the_group_name(self):
        """The dropdown is what the user actually said; the group may be
        called `g1`."""
        regex = r"plate1_(?P<g1>[A-Za-z0-9]+)_T0001F(?P<g2>\d+)L01A01Z01C(?P<g3>\d+)\.tif"

        made = plan(_names(), regex,
                    {"g1": "wellID", "g2": "fieldID", "g3": "chanID"},
                    plate="exp1")

        assert made.trouble == ()
        assert made.renamed[0].well == "A01"
        assert made.renamed[0].channel == "01"

    def test_the_channel_meanings_the_ask_named_are_offered(self):
        for wanted in ("channel 1", "channel 4", "cell", "nucleus"):
            assert wanted in CHANNEL_MEANINGS

    def test_the_groups_are_read_in_order(self):
        assert group_names(CELLVOYAGER) == ("wellID", "fieldID", "chanID")

    def test_a_broken_regex_has_no_groups_rather_than_raising(self):
        assert group_names("(?P<a>[A-Z") == ()


# --------------------------------------------------------------------------- #
#  The panel
# --------------------------------------------------------------------------- #

pytest.importorskip("PySide6")


@pytest.fixture
def images(tmp_path):
    folder = tmp_path / "exp1"
    folder.mkdir()
    for name in _names():
        (folder / name).write_bytes(b"")
    (folder / "notes.txt").write_text("not an image")
    return folder


@pytest.fixture
def panel(qtbot):
    from spacr.qt.widgets.import_workbench import ImportWorkbench

    widget = ImportWorkbench()
    qtbot.addWidget(widget)
    return widget


def _dropped(panel, paths):
    """`add_files`, waited out. Returns how many files are held.

    The walk runs on a worker: a dropped plate folder is a path the USER
    chose, usually on the microscope's share, and walking it on the GUI
    thread froze the whole application -- see
    `tests/qt/test_the_import_workbench_never_walks_on_the_gui_thread.py`.
    Nothing below cares when the files arrive, only that they all do, so the
    waiting lives here rather than in every assertion.
    """
    import time

    from PySide6.QtWidgets import QApplication

    panel.add_files(paths)
    deadline = time.monotonic() + 10.0
    while panel.is_scanning() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()
    return len(panel.files())


class TestFilesArriveByDrop:

    def test_a_folder_is_walked(self, panel, images):
        assert _dropped(panel, [str(images)]) == 8

    def test_a_file_that_is_not_an_image_is_left_out(self, panel, images):
        _dropped(panel, [str(images)])

        assert all(not f.endswith(".txt") for f in panel.files())

    def test_the_same_file_twice_is_once(self, panel, images):
        _dropped(panel, [str(images)])
        _dropped(panel, [str(images)])

        assert len(panel.files()) == 8

    def test_the_panel_accepts_drops(self, panel):
        assert panel.acceptDrops()

    def test_clearing_empties_it(self, panel, images):
        _dropped(panel, [str(images)])

        panel.set_files([])

        assert panel.files() == []


class TestTheRegexIsProposedAndEditable:

    def test_the_first_drop_proposes_one(self, panel, images):
        _dropped(panel, [str(images)])

        assert panel.regex.text().strip()
        assert "matches 8 of 8" in panel.evidence.text()

    def test_a_later_drop_does_not_overwrite_an_edited_one(self, panel,
                                                           images, tmp_path):
        _dropped(panel, [str(images)])
        panel.regex.setText(CELLVOYAGER)

        other = tmp_path / "more"
        other.mkdir()
        (other / _names()[0]).write_bytes(b"")
        _dropped(panel, [str(other)])

        assert panel.regex.text() == CELLVOYAGER

    def test_editing_it_redraws_without_a_button(self, panel, images):
        """A Test button would imply the answer is stale until pressed."""
        _dropped(panel, [str(images)])
        panel.regex.setText(CELLVOYAGER)
        before = panel.the_plan().n_matched

        panel.regex.setText(r"nothing_matches_this")

        assert before == 8
        assert panel.the_plan().n_matched == 0


class TestTheTableIsThePreview:

    @pytest.fixture
    def loaded(self, panel, images):
        _dropped(panel, [str(images)])
        panel.regex.setText(CELLVOYAGER)
        return panel

    def test_one_row_per_file_with_its_new_name(self, loaded):
        assert loaded.table.rowCount() == 8
        assert loaded.table.item(0, 1).text().endswith(".tif")

    def test_an_unmatched_file_is_a_row_that_says_no_match(self, panel,
                                                            images):
        (images / "stray.tif").write_bytes(b"")
        _dropped(panel, [str(images)])
        panel.regex.setText(CELLVOYAGER)

        texts = [panel.table.item(r, 1).text()
                 for r in range(panel.table.rowCount())]
        assert "no match" in texts

    def test_the_tree_is_shown_beside_it(self, loaded):
        said = loaded.tree.toPlainText()

        assert "well(s)" in said
        assert "channel(s)" in said

    def test_the_summary_names_both_counts(self, loaded):
        assert "8 of 8" in loaded.dropped.text()


class TestTheRoleDropdowns:

    @pytest.fixture
    def loaded(self, panel, images):
        _dropped(panel, [str(images)])
        panel.regex.setText(CELLVOYAGER)
        return panel

    def test_one_per_group(self, loaded):
        assert set(loaded.roles()) == {"wellID", "fieldID", "chanID"}

    def test_a_group_already_named_for_a_role_defaults_to_it(self, loaded):
        """A proposal whose groups are `wellID` should not make the user say
        so again."""
        assert loaded.roles()["wellID"] == "wellID"

    def test_nobody_types_a_group_name(self, loaded):
        from PySide6.QtWidgets import QComboBox

        for group in loaded.roles():
            assert isinstance(loaded._boxes[group], QComboBox)

    def test_changing_one_redraws_the_preview(self, loaded):
        box = loaded._boxes["chanID"]
        box.setCurrentIndex(box.findData(""))

        assert "chanID" in loaded.role_trouble.text()

    def test_two_groups_on_one_role_is_said_on_the_panel(self, loaded):
        box = loaded._boxes["fieldID"]
        box.setCurrentIndex(box.findData("wellID"))

        assert "wellID" in loaded.role_trouble.text()

    def test_a_complete_assignment_says_nothing(self, loaded):
        assert loaded.role_trouble.text() == ""

    def test_the_dropdowns_follow_the_regex(self, loaded):
        loaded.regex.setText(r"(?P<wellID>[A-Z]\d+)")

        assert set(loaded.roles()) == {"wellID"}


class TestTheDialogHandsBackTheRegex:

    def test_it_returns_the_box_minus_the_extension(self, qtbot, images):
        """See the next test for why the extension comes off."""
        from spacr.qt.widgets.import_workbench import ImportWorkbenchDialog

        dialog = ImportWorkbenchDialog([str(images)], CELLVOYAGER)
        qtbot.addWidget(dialog)

        assert dialog.chosen_regex() == CELLVOYAGER[:-len(r"\.tif")]

    def test_the_extension_is_stripped_because_get_regex_appends_one(
            self, qtbot, images):
        r"""`_get_regex` builds `f"({custom_regex}).{img_format}"`, so a
        pattern already ending in `\.tif` becomes `...\.tif..tif` and
        matches nothing -- silently."""
        from spacr.qt.widgets.import_workbench import ImportWorkbenchDialog

        dialog = ImportWorkbenchDialog([str(images)], CELLVOYAGER)
        qtbot.addWidget(dialog)

        assert not dialog.chosen_regex().endswith(".tif")

    def test_and_the_composed_pattern_matches_the_real_filenames(
            self, qtbot, images):
        """The acceptance criterion this instruction states: "the accepted
        regex reaches `_get_regex`'s 'custom' branch, so the headless path
        and `spacr-run` are unchanged". Matching is what that means."""
        import re

        from spacr.qt.widgets.import_workbench import ImportWorkbenchDialog
        from spacr.utils import _get_regex

        dialog = ImportWorkbenchDialog([str(images)], CELLVOYAGER)
        qtbot.addWidget(dialog)

        composed = re.compile(
            _get_regex("custom", "tif", dialog.chosen_regex()))
        matched = [n for n in _names() if composed.search(n)]
        assert len(matched) == 8
        assert composed.search(_names()[0]).group("wellID") == "A01"

    def test_a_pattern_that_never_had_one_is_left_alone(self, qtbot):
        from spacr.qt.widgets.import_workbench import ImportWorkbenchDialog

        dialog = ImportWorkbenchDialog([], r"(?P<wellID>[A-Z]\d+)")
        qtbot.addWidget(dialog)

        assert dialog.chosen_regex() == r"(?P<wellID>[A-Z]\d+)"


# --------------------------------------------------------------------------- #
#  The bug this wiring found: every saved regex matched nothing
# --------------------------------------------------------------------------- #

class TestASavedRegexActuallyMatches:
    """`_get_regex` appends the extension ITSELF.

    So a pattern already ending in `\\.tif` -- or in the
    `\\.(?:tif|tiff|png|jpg|jpeg)$` that `auto_detect_regex` returns --
    became `(...$)..tif`: an anchor with characters after it, which can never
    match. Measured through the real path on eight cellvoyager filenames:
    0 of 8, with no error anywhere and the pattern in the box looking exactly
    right.
    """

    def _composed(self, pattern):
        import re

        from spacr.utils import _get_regex

        return re.compile(_get_regex("custom", "tif", pattern))

    def test_the_detectors_own_answer_survives_get_regex(self):
        from spacr.import_plan import for_get_regex
        from spacr.qt.regex_detect import auto_detect_regex

        pattern, _label, _hits = auto_detect_regex(_names())

        composed = self._composed(for_get_regex(pattern))
        assert sum(1 for n in _names() if composed.search(n)) == 8

    def test_and_it_did_not_before(self):
        """The bug, pinned, so the trim cannot be removed as tidying."""
        from spacr.qt.regex_detect import auto_detect_regex

        pattern, _label, _hits = auto_detect_regex(_names())

        composed = self._composed(pattern)
        assert sum(1 for n in _names() if composed.search(n)) == 0

    def test_the_editor_saves_the_trimmed_one(self, qtbot):
        from spacr.qt.regex_editor import RegexEditorDialog

        dialog = RegexEditorDialog(_names())
        qtbot.addWidget(dialog)
        dialog._on_save()

        composed = self._composed(dialog.regex)
        assert sum(1 for n in _names() if composed.search(n)) == 8

    def test_the_groups_still_come_out(self, qtbot):
        from spacr.qt.regex_editor import RegexEditorDialog

        dialog = RegexEditorDialog(_names())
        qtbot.addWidget(dialog)
        dialog._on_save()

        got = self._composed(dialog.regex).search(_names()[0])
        assert got.group("wellID") == "A01"

    def test_a_pattern_with_no_extension_is_untouched(self):
        from spacr.import_plan import for_get_regex

        assert for_get_regex(r"(?P<wellID>[A-Z]\d+)") == r"(?P<wellID>[A-Z]\d+)"

    def test_every_spelling_of_the_tail_comes_off(self):
        from spacr.import_plan import for_get_regex

        head = r"(?P<wellID>[A-Z]\d+)"
        for tail in (r"\.tif", r"\.png", r"\.(?:tif|tiff|png|jpg|jpeg)$",
                     r"\.(tif|tiff|png|jpg|jpeg)$", ".tif"):
            assert for_get_regex(head + tail) == head, tail

    def test_the_editor_offers_the_workbench(self, qtbot):
        from spacr.qt.regex_editor import RegexEditorDialog

        dialog = RegexEditorDialog(_names())
        qtbot.addWidget(dialog)

        assert dialog._workbench_btn.text() == "Work it out from the files…"
