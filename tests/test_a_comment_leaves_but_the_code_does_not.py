"""Stripping the prose out of a module must not change what the module does.

`tools/extract_source_notes.py` edits 568 files. The property that makes that
safe is not that the edit was written carefully, it is that every file is
parsed before and after and refused unless the two ASTs are identical.
Comments never reach an AST, so an identical dump means the edit touched
comments AND NOTHING ELSE.

These tests hold that guard down, along with the four things that look like
comments and are not:

* a ``#`` inside a string literal, which a regex would eat and the tokenizer
  does not;
* ``#:``, Sphinx attribute docstrings, which render into the PUBLISHED API
  documentation -- deleting one deletes a paragraph of the manual, silently;
* the shebang and the PEP 263 coding declaration;
* ``# noqa`` / ``# type:`` / ``# pragma``, read by tools rather than people.

and the classification at the two ends where it has to be right: a row of
dashes carries nothing, a measured number carries everything.

Every test builds its own throwaway tree. Nothing here reads or writes
``spacr/``.
"""
from __future__ import annotations

import ast
import sys
import textwrap
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        return import_module("extract_source_notes")
    finally:
        sys.path.remove(tools_dir)


extract_source_notes = _tool()
esn = extract_source_notes


def write(tmp_path: Path, relative: str, body: str) -> Path:
    """Put ``body`` at ``relative`` under ``tmp_path``, dedented."""
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    return path


def blocks_of(source: str):
    """Classify ``source`` and hand back its blocks."""
    found, _preserved = esn.collect_blocks("m.py", textwrap.dedent(source))
    return [esn.classify(block) for block in found]


def only_block(source: str):
    found = blocks_of(source)
    assert len(found) == 1, f"expected one block, got {len(found)}: {found}"
    return found[0]


class TestTheASTGuard:
    """The guard is the whole safety story, so it is tested by breaking it."""

    def test_a_damaged_strip_is_refused_and_the_file_is_left_alone(
            self, tmp_path, monkeypatch):
        """A strip that removes a line of CODE must not reach the disk.

        `strip_source` is the only thing that edits source text, which is why
        it is the seam: replace it with one that also drops a statement and
        the guard has to notice, because in the real thing the damage would
        not be announced.
        """
        module = write(tmp_path, "pkg/m.py", """
            # a note about the value below, worth 4096 bytes of anyone's time
            VALUE = 4096


            def use():
                return VALUE
        """)
        before = module.read_text()

        def sabotage(source, blocks):
            kept = [line for line in source.splitlines(keepends=True)
                    if not line.lstrip().startswith("#")
                    and "return VALUE" not in line]
            return "".join(kept)

        monkeypatch.setattr(esn, "strip_source", sabotage)
        with pytest.raises(esn.NoteExtractionError, match="same AST"):
            esn.process_file(module, tmp_path, tmp_path / "docs" / "notes",
                             execute=True)
        assert module.read_text() == before, "the damaged file was written"
        assert not (tmp_path / "docs").exists(), "notes were written anyway"

    def test_a_truncated_string_is_caught_even_though_it_is_still_python(self):
        """The failure a regex would cause parses fine and is still damage.

        Cutting ``"a # b"`` at the ``#`` leaves valid Python whose constant
        has changed. Nothing but comparing the two trees notices that.
        """
        before = 'MARK = "a # b"\n'
        after = 'MARK = "a "\n'
        ast.parse(after)
        assert esn.verify_ast(before, after) is False

    def test_the_guard_is_not_reachable_from_the_command_line(self):
        """There is no flag that turns the check off.

        A flag would be reached for on the one file where it fired, which is
        the one file where it mattered, so the parser must not offer one.
        """
        parser = esn.build_parser()
        options = {action.dest for action in parser._actions}
        for escape in ("force", "no_verify", "skip_ast", "unsafe"):
            assert escape not in options

    def test_every_real_shape_of_comment_survives_the_round_trip(self, tmp_path):
        """The awkward cases, run through the real path rather than argued about."""
        module = write(tmp_path, "pkg/tricky.py", '''
            #!/usr/bin/env python3
            # -*- coding: utf-8 -*-
            """A docstring with a # in it, and 1024 rows of nothing."""
            import os  # noqa: F401

            #: A Sphinx attribute doc, 3 lines long
            #: and continued here.
            TABLE = {
                "a#b": 1,      # trailing, mentioning 12 ms
                'c': "# not a comment",
            }

            # ------------------------------------------------------------------
            # A ruled banner over a real note
            # ------------------------------------------------------------------
            #
            # This took 1.5 seconds per call because the lookup was linear.

            def f(x):  # type: ignore[no-untyped-def]
                s = f"value {x} # still not a comment"
                return s \\
                    + "#"
        ''')
        report = esn.process_file(module, tmp_path, tmp_path / "notes")
        source = module.read_text()
        stripped = esn.strip_source(
            source, [b for b in report.blocks])
        assert esn.verify_ast(source, stripped)
        assert report.ast_verified and not report.ast_failed
        assert '"a#b"' in stripped and '"# not a comment"' in stripped
        assert "# still not a comment" in stripped


class TestWhatIsNotAComment:
    def test_a_hash_inside_a_string_literal_survives(self, tmp_path):
        """A regex eats this. The tokenizer knows it is a string."""
        module = write(tmp_path, "m.py", """
            # a plain note
            PATTERN = "count # of rows"
            OTHER = 'a # b'
            TRIPLE = '''line one # two'''
        """)
        source = module.read_text()
        report = esn.process_file(module, tmp_path, tmp_path / "notes")
        stripped = esn.strip_source(source, report.blocks)
        assert '"count # of rows"' in stripped
        assert "'a # b'" in stripped
        assert "line one # two" in stripped
        assert "# a plain note" not in stripped

    def test_a_sphinx_attribute_docstring_survives(self, tmp_path):
        """``#:`` is published documentation, not a comment.

        There are 14,180 of them in ``spacr/``. A pass that removed them
        would delete that many paragraphs from the API docs without saying
        so, which is why they are counted in the report rather than merely
        skipped.
        """
        module = write(tmp_path, "m.py", """
            #: The number of rows a page holds.
            #: Chosen so a page fits one screen.
            PAGE = 40

            # an ordinary note about nothing
            OTHER = 1
        """)
        source = module.read_text()
        blocks, preserved = esn.collect_blocks("m.py", source)
        assert preserved["sphinx"] == 2
        stripped = esn.strip_source(source, [esn.classify(b) for b in blocks])
        assert "#: The number of rows a page holds." in stripped
        assert "#: Chosen so a page fits one screen." in stripped
        assert "an ordinary note" not in stripped
        assert esn.verify_ast(source, stripped)

    def test_a_sphinx_line_is_never_swallowed_by_the_block_around_it(self):
        """A ``#:`` line in the middle of prose splits the block, it does not
        join it: joining would carry the published line out of the source."""
        found = blocks_of("""
            # prose above
            #: published
            # prose below
            X = 1
        """)
        texts = [text for block in found for text in block.texts]
        assert "#: published" not in texts
        assert len(found) == 2

    @pytest.mark.parametrize("line,kind", [
        ("#!/usr/bin/env python3", "shebang"),
        ("# -*- coding: utf-8 -*-", "coding"),
        ("# noqa: E501", "noqa"),
        ("# type: ignore[arg-type]", "type"),
        ("# pragma: no cover", "pragma"),
        ("#: an attribute doc", "sphinx"),
    ])
    def test_the_directives_are_named_and_kept(self, line, kind):
        assert esn.classify_directive(line, 1) == kind

    def test_a_directive_is_only_a_directive_where_python_looks_for_it(self):
        """A coding declaration on line 90 is prose about encodings."""
        assert esn.classify_directive("# -*- coding: utf-8 -*-", 90) is None
        assert esn.classify_directive("#!/usr/bin/env python3", 7) is None

    def test_the_preserved_directives_are_counted_for_the_report(self, tmp_path):
        """The maintainer is shown that they survived, not told."""
        module = write(tmp_path, "m.py", """
            #!/usr/bin/env python3
            # -*- coding: utf-8 -*-
            import os  # noqa: F401
            import sys  # type: ignore

            #: an attribute doc
            X = 1  # pragma: no cover
        """)
        report = esn.process_file(module, tmp_path, tmp_path / "notes")
        assert report.preserved["shebang"] == 1
        assert report.preserved["coding"] == 1
        assert report.preserved["noqa"] == 1
        assert report.preserved["type"] == 1
        assert report.preserved["pragma"] == 1
        assert report.preserved["sphinx"] == 1
        assert report.lines_removed == 0


class TestTheClassifier:
    def test_a_divider_is_discarded(self):
        block = only_block("""
            # ---------------------------------------------------------------
            X = 1
        """)
        assert block.verdict == esn.DISCARD

    def test_a_ruled_section_label_is_discarded(self):
        """``# -- shutdown ------`` names a region of the file the file shows."""
        block = only_block("""
            # -- shutdown --------------------------------------------------
            def close():
                pass
        """)
        assert block.verdict == esn.DISCARD
        assert "banner" in block.reason

    def test_a_ruled_banner_carrying_a_real_note_is_not_a_banner(self):
        """The house style opens its longest notes with a rule. Those notes
        are the ones most worth keeping, so the rule must not condemn them."""
        block = only_block("""
            # ---------------------------------------------------------------
            # The pool is bounded
            # ---------------------------------------------------------------
            #
            # An unbounded pool reached 92 GB and the OOM killer took the
            # session with it.
            SIZE = 4
        """)
        assert block.verdict == esn.RETAIN

    def test_a_measurement_is_retained(self):
        block = only_block("""
            # Sleeping 50 ms here: anything shorter and the widget repaints
            # mid-drag.
            DELAY = 0.05
        """)
        assert block.verdict == esn.RETAIN
        assert "measured" in block.reason

    @pytest.mark.parametrize("comment", [
        "# 4096 was measured as the point the copy stops getting cheaper",
        "# Kept as a list rather than a set because order reaches the CSV",
        "# Do NOT hoist this: the import has a side effect at module scope",
        "# Caught on 2026-08-10, when the ratchet regressed unnoticed",
        "# See https://bugreports.qt.io/browse/QTBUG-12345",
        "# Accepts 'std' or 'sem'",
    ])
    def test_the_things_the_code_cannot_say_are_retained(self, comment):
        block = only_block(comment + "\nX = 1\n")
        assert block.verdict == esn.RETAIN, block.reason

    def test_a_numbered_section_heading_is_still_a_heading(self):
        """A bare number is a weak signal and loses to the banner test.

        ``# --- 19 ---`` numbers a section; it is not a constant somebody
        chose, and the generated modules are full of them.
        """
        block = only_block("""
            # ---------------------------------------------------------------
            # 19
            # ---------------------------------------------------------------
            def v19():
                pass
        """)
        assert block.verdict == esn.DISCARD

    def test_a_banner_carrying_a_measurement_is_kept_anyway(self):
        """...but a strong signal beats the banner test, in that order."""
        block = only_block("""
            # -- the 40 ms budget ------------------------------------------
            BUDGET = 0.04
        """)
        assert block.verdict == esn.RETAIN

    def test_a_restatement_of_the_line_below_is_discarded(self):
        block = only_block("""
            # Get the unique cell labels
            unique_cell_labels = get_unique(cell_labels)
        """)
        assert block.verdict == esn.DISCARD
        assert "restates" in block.reason

    def test_commented_out_code_is_discarded(self):
        block = only_block("""
            #if print_object_number:
            #    num_objects = mask_object_count(mask)
            #    print(num_objects)
            X = 1
        """)
        assert block.verdict == esn.DISCARD
        assert "commented-out" in block.reason

    def test_a_bare_marker_is_discarded_and_a_furnished_one_is_not(self):
        assert only_block("# TODO\nX = 1\n").verdict == esn.DISCARD
        furnished = only_block(
            "# TODO: this refits the whole model per well, which costs "
            "40 seconds on tsg101\nX = 1\n")
        assert furnished.verdict == esn.RETAIN

    def test_a_short_label_the_code_does_not_contain_is_unsure_not_discarded(
            self):
        """The classifier is allowed not to know, and says so.

        Deleting a measurement nobody can re-derive is far worse than carrying
        a dull line in a notes file, so the ambiguous case must not fall to
        DISCARD by default.
        """
        block = only_block("""
            # Belt and braces
            value = compute(other)
        """)
        assert block.verdict == esn.UNSURE


class TestTheNotesFile:
    def test_notes_are_grouped_by_the_scope_the_block_sat_in(self, tmp_path):
        """A flat dump in file order is not navigable. The grouping and the
        line number are how a reader gets from a line of code to its note."""
        module = write(tmp_path, "pkg/m.py", """
            class Loader:
                def read(self):
                    # Reads in 8 KB chunks because the NAS stalls on 1 MB
                    # reads under load.
                    return self._read(8192)

            def helper():
                # Retried 3 times: the mount comes back within a second.
                return None
        """)
        source = module.read_text()
        blocks, _ = esn.collect_blocks("pkg/m.py", source)
        notes = esn.render_notes(
            "pkg/m.py", [esn.classify(b) for b in blocks])
        assert "## Loader.read" in notes
        assert "## helper" in notes
        assert notes.index("## Loader.read") < notes.index("## helper")
        assert "### lines 3-4" in notes
        assert "### line 8" in notes
        assert "8 KB chunks" in notes

    def test_a_title_line_is_not_reflowed_into_the_prose_under_it(self):
        """Hand-wrapped source lines are reflowed for the page -- except the
        opening line, when it is a title rather than a first clause."""
        block = only_block("""
            # log10 read fraction
            # A well with zero reads gives 0/0, which is the case this
            # guards. It took 3 hours to find.
            X = 1
        """)
        paragraphs = esn._paragraphs(block)
        assert paragraphs[0] == "log10 read fraction"
        assert paragraphs[1].startswith("A well with zero reads")

    def test_a_wrapped_sentence_is_still_reflowed(self):
        """The title rule must not cut an ordinary wrapped sentence in half."""
        block = only_block("""
            # Bound outstanding work to one
            # pool-width batch, measured at 12 GB otherwise.
            X = 1
        """)
        assert esn._paragraphs(block) == [
            "Bound outstanding work to one pool-width batch, measured at "
            "12 GB otherwise."]

    def test_a_comment_above_a_def_belongs_to_that_def(self, tmp_path):
        """It is about the function, not about the module it sits in."""
        source = textwrap.dedent("""
            # Split out in 2026 after the inline version was profiled at
            # 40% of the run.
            @cache
            def slow():
                return 1
        """).lstrip("\n")
        blocks, _ = esn.collect_blocks("m.py", source)
        assert [b.scope for b in blocks] == ["slow"]


class TestTheDryRun:
    def test_the_dry_run_writes_nothing(self, tmp_path, capsys):
        """The default must be able to be run on the real tree by anyone."""
        module = write(tmp_path, "pkg/m.py", """
            # ---------------------------------------------------------------
            # Sized at 4096 after measuring the copy
            # ---------------------------------------------------------------
            SIZE = 4096
        """)
        before = module.read_text()
        notes_dir = tmp_path / "docs" / "notes"
        status = esn.run([tmp_path / "pkg"], tmp_path, notes_dir)
        out = capsys.readouterr().out
        assert status == 0
        assert module.read_text() == before, "the module was edited"
        assert not notes_dir.exists(), "a notes file was created"
        assert "DRY RUN" in out
        assert "nothing was written" in out

    def test_the_dry_run_still_verifies_every_file(self, tmp_path, capsys):
        """The report's number has to mean something, so the check runs even
        when nothing is written -- that is what makes the dry run a decision
        the maintainer can take without reading 568 diffs."""
        for name in ("a", "b", "c"):
            write(tmp_path, f"pkg/{name}.py", """
                # A note carrying 12 seconds of measured cost.
                X = 1
            """)
        esn.run([tmp_path / "pkg"], tmp_path, tmp_path / "notes")
        out = capsys.readouterr().out
        assert "files verified identical: 3 / 3" in out
        assert "files refused           : 0" in out

    def test_the_report_shows_a_sample_of_each_class(self, tmp_path, capsys):
        write(tmp_path, "pkg/m.py", """
            # ---------------------------------------------------------------
            # Measured at 40 ms per call on the tsg101 plate.
            VALUE = 1
            # Get the value
            value = get_value()
            # Belt and braces
            other = compute(thing)
        """)
        esn.run([tmp_path / "pkg"], tmp_path, tmp_path / "notes")
        out = capsys.readouterr().out
        for verdict in (esn.DISCARD, esn.RETAIN, esn.UNSURE):
            assert f"SAMPLE -- {verdict}" in out

    def test_execute_writes_the_module_and_its_notes_at_the_mirrored_path(
            self, tmp_path):
        """The path convention IS the discoverability mechanism: no pointer
        comment is left behind, so ``pkg/m.py`` must land at
        ``<notes>/pkg/m.md`` and nowhere else."""
        module = write(tmp_path, "pkg/m.py", """
            # Sized at 4096 after measuring: below that the copy dominates.
            SIZE = 4096
        """)
        notes_dir = tmp_path / "docs" / "notes"
        assert esn.run([tmp_path / "pkg"], tmp_path, notes_dir,
                       execute=True) == 0
        assert "#" not in module.read_text()
        assert ast.dump(ast.parse("SIZE = 4096\n")) == ast.dump(
            ast.parse(module.read_text()))
        note = notes_dir / "pkg" / "m.md"
        assert note.exists()
        assert "4096" in note.read_text()

    def test_a_missing_path_is_an_error_not_an_empty_report(self, tmp_path):
        assert esn.main(["--root", str(tmp_path), "nowhere"]) == 1

    def test_unsure_blocks_can_be_left_in_the_source(self, tmp_path):
        """``--unsure keep`` is the cautious run: only what the classifier
        was sure about moves."""
        module = write(tmp_path, "pkg/m.py", """
            # Belt and braces
            value = compute(other)
        """)
        source = module.read_text()
        report = esn.process_file(module, tmp_path, tmp_path / "notes",
                                  execute=True, unsure="keep")
        assert report.counts[esn.UNSURE] == 1
        assert module.read_text() == source
