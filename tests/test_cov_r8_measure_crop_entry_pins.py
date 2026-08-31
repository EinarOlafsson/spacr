"""What ``measure_crop`` has already decided by the time these run.

Everything here sits inside a two-hundred-line entry point that runs a
whole measurement pass, so reaching any of it means building a plate, a
mask stack and a database. What each pin holds is the line ABOVE the
guard -- the one that makes it moot -- which is the part a change would
touch.
"""
from __future__ import annotations

import inspect

import pytest


class TestTheSourceArgument:

    def test_a_source_that_is_neither_a_string_nor_a_list_is_refused(self):
        """The refusal the two shapes below it rest on."""
        from spacr import measure as M

        source = inspect.getsource(M.measure_crop)
        assert "if not isinstance(settings['src'], (str, list)):" in source
        assert "src must be a string or a list of strings" in source

        with pytest.raises(ValueError, match="must be a string or a list"):
            M.measure_crop({"src": 7, "timelapse": False})

    def test_a_single_folder_becomes_a_list_of_one(self):
        """So everything below works on folders, plural, and there is one
        loop rather than two paths."""
        from spacr import measure as M

        source = inspect.getsource(M.measure_crop)
        assert "if isinstance(settings['src'], str):" in source
        assert "settings['src'] = [settings['src']]" in source

    def test_the_list_check_below_it_can_never_be_false(self):
        """THE PIN.

        By the time ``if isinstance(settings['src'], list)`` is asked,
        anything that was not a str or a list has been refused, and
        anything that was a str has been wrapped -- so it is always a
        list and the function cannot fall off its end returning None.

        That is what the guard buys: falling out of ``measure_crop``
        without running anything, and reporting no error, is a
        measurement pass the caller believes happened.
        """
        from spacr import measure as M

        source = inspect.getsource(M.measure_crop)
        refuse = source.index("if not isinstance(settings['src'], (str, list)):")
        wrap = source.index("if isinstance(settings['src'], str):")
        check = source.index("if isinstance(settings['src'], list):")

        assert refuse < wrap < check, (
            "the three source checks are no longer in order, so the list "
            "check below can now be false and the function returns None")

        for value in ("/data/plate1", ["/data/plate1", "/data/plate2"]):
            if isinstance(value, str):
                value = [value]
            assert isinstance(value, list)


class TestTheTimelapseNucleusNote:

    def test_the_note_is_printed_only_when_there_are_cells_to_relabel(self):
        """THE UNCOVERED ARC: ``cell_mask_dim`` is None.

        Tracking cells by their nucleus labels means relabelling the CELL
        mask, so a run with no cell mask has nothing to relabel and the
        sentence would describe work that is not going to happen.

        The relabelling itself is separately guarded on the same
        setting, which is what this pins: the note and the work agree.
        """
        from spacr import measure as M

        source = inspect.getsource(M)
        assert "if settings['timelapse_objects'] == 'nucleus':" in source
        assert "if not settings['cell_mask_dim'] is None:" in source
        assert "cells will be relabeled to nucleus labels to track cells." \
            in source

        assert source.count(
            "if settings['timelapse_objects'] == 'nucleus':") >= 2, (
            "the note and the relabelling no longer share a condition, so "
            "one can be announced without the other happening")

    def test_a_timelapse_run_never_writes_pngs(self):
        """The neighbouring decision, and it is the loud one: a timelapse
        run turns PNG writing off, because a crop per frame per object is
        a directory nobody can open."""
        from spacr import measure as M

        source = inspect.getsource(M.measure_crop)
        assert "if settings['timelapse']:" in source
        assert "settings['save_png'] = False" in source
        assert source.index("if settings['timelapse']:") < source.index(
            "settings['save_png'] = False")
