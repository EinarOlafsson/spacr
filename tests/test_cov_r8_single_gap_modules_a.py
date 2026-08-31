"""One uncovered decision each, in five modules that are otherwise complete.

Grouped rather than split into five files because each is a single
branch and the reasoning per module is a paragraph, not a suite. Three
are live and driven; two are dead and pinned to the code that makes
them dead.
"""
from __future__ import annotations

import inspect
import os

import pytest


# ---------------------------------------------------------------------------
# spacr/qt/fractal_defaults.py -- clamp
# ---------------------------------------------------------------------------

class TestClamp:
    """The whole function was unexecuted: nothing called it under test.

    It is the guard between a preferences field and the fractal's speed
    and depth, so a value that slips past it is a fractal running at a
    number no slider can produce.
    """

    def test_a_value_inside_the_range_comes_back_unchanged(self):
        from spacr.qt.fractal_defaults import clamp

        assert clamp(2.5, 1.0, 6.0) == 2.5

    def test_below_the_floor_returns_the_floor(self):
        from spacr.qt.fractal_defaults import clamp

        assert clamp(-100.0, 1.0, 6.0) == 1.0

    def test_above_the_ceiling_returns_the_ceiling(self):
        from spacr.qt.fractal_defaults import clamp

        assert clamp(1e9, 1.0, 6.0) == 6.0

    def test_the_bounds_themselves_are_inside(self):
        from spacr.qt.fractal_defaults import clamp

        assert clamp(1.0, 1.0, 6.0) == 1.0
        assert clamp(6.0, 1.0, 6.0) == 6.0

    def test_the_defaults_it_exists_to_hold_are_within_their_own_range(self):
        from spacr.qt import fractal_defaults as fd

        assert fd.clamp(fd.DEFAULT_SPEED_MAX,
                        0.0, fd.DEFAULT_SPEED_MAX) == fd.DEFAULT_SPEED_MAX


# ---------------------------------------------------------------------------
# spacr/organelle_types.py -- a preset with no members
# ---------------------------------------------------------------------------

class TestExplainingAPresetThatNamesNothing:

    def test_custom_is_the_preset_with_no_members(self):
        from spacr.organelle_types import ORGANELLE_TYPES

        empty = [name for name, preset in ORGANELLE_TYPES.items()
                 if not preset.members]
        assert empty == ["custom"], (
            "the set of member-less presets changed; the 'covers:' line "
            "below is skipped for exactly these")

    def test_custom_explains_nothing_because_it_applies_nothing(self, capsys):
        """The public path cannot reach the member-less branch.

        ``custom`` is the only preset that names no structures, and it
        recommends nothing either -- so the explanation returns at its
        first line, before it would have printed a 'covers:'. That is
        why the branch below has to be driven directly.
        """
        from spacr.organelle_types import apply_preset, preset_for

        assert preset_for("custom", None) == {}, (
            "custom now recommends settings, so apply_preset reaches the "
            "explanation and this branch has a live public path")

        apply_preset({"organelle_type": "custom"}, explain=True)
        assert capsys.readouterr().out == ""

    def test_a_member_less_preset_prints_no_covers_line(self, capsys):
        """THE UNCOVERED BRANCH.

        Every preset that names structures carries a 'covers:' line.
        One that names none must omit the line rather than print an
        empty one -- "covers: " reads as a preset that covers nothing,
        which is a different claim from making no claim.
        """
        from spacr.organelle_types import OrganelleType, _explain

        nameless = OrganelleType(
            label="Nameless", members=(), method="log",
            morphology="spots", size_split=None, params={}, caveat="")

        _explain(nameless, 6, {"organelle_method": "log"}, {})
        printed = capsys.readouterr().out

        assert "Nameless" in printed
        assert "set    organelle_method" in printed
        assert "covers:" not in printed

    def test_a_preset_that_does_name_members_prints_them(self, capsys):
        from spacr.organelle_types import apply_preset, ORGANELLE_TYPES

        named = next(name for name, preset in ORGANELLE_TYPES.items()
                     if preset.members)
        apply_preset({"organelle_type": named}, explain=True)
        printed = capsys.readouterr().out

        assert "covers:" in printed


# ---------------------------------------------------------------------------
# spacr/resource_log.py -- private memory figures that are not integers
# ---------------------------------------------------------------------------

class TestMemoryMeasureFallback:
    """USS, then PSS, then RSS -- and what happens when none is a number."""

    def test_the_first_integer_measure_wins(self):
        from spacr import resource_log

        class Full:
            uss = 111
            pss = 222
            rss = 333

        class Process:
            def memory_full_info(self):
                return Full()

            def memory_info(self):        # pragma: no cover - not reached
                raise AssertionError("the private figures were available")

        value, measure = resource_log._memory(_PsutilStub(), Process())
        assert (value, measure) == (111, "uss")

    def test_a_full_info_with_no_integer_measure_falls_through_to_rss(self):
        """THE UNCOVERED ARC: the loop finishes without returning.

        A platform can hand back a full-info object whose private
        figures are None, or strings, rather than refusing outright.
        Treating those as a memory number would put "uss = None" in the
        resource log; falling through gives the resident figure, which
        every platform has.
        """
        from spacr import resource_log

        class Full:
            uss = None
            pss = "not a number"
            # rss deliberately absent

        class Info:
            rss = 4096

        class Process:
            def memory_full_info(self):
                return Full()

            def memory_info(self):
                return Info()

        value, measure = resource_log._memory(_PsutilStub(), Process())
        assert (value, measure) == (4096, "rss")

    def test_no_resident_figure_either_reports_nothing_rather_than_zero(self):
        from spacr import resource_log

        class Process:
            def memory_full_info(self):
                raise RuntimeError("permission denied")

            def memory_info(self):
                return type("Info", (), {"rss": None})()

        value, measure = resource_log._memory(_PsutilStub(), Process())
        assert (value, measure) == (None, None), (
            "an unknown memory figure must not read as zero bytes")


class _PsutilStub:
    class NoSuchProcess(Exception):
        pass


# ---------------------------------------------------------------------------
# spacr/portable_paths.py -- the climb never revisits a folder
# ---------------------------------------------------------------------------

class TestTheClimbNeverRepeatsItself:

    def test_the_climb_is_strictly_shorter_at_every_step(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)

        from spacr.portable_paths import candidate_roots
        climbed = candidate_roots(str(deep))

        assert climbed[0] == os.path.abspath(str(deep))
        assert len(set(climbed)) == len(climbed), "a folder was visited twice"
        lengths = [len(p) for p in climbed]
        assert lengths == sorted(lengths, reverse=True), (
            "the climb did not move strictly upwards")

    def test_a_file_is_climbed_from_its_folder(self, tmp_path):
        f = tmp_path / "plate.csv"
        f.write_text("x\n")

        from spacr.portable_paths import candidate_roots

        assert candidate_roots(str(f))[0] == os.path.abspath(str(tmp_path))

    def test_nothing_gives_nothing(self):
        from spacr.portable_paths import candidate_roots

        assert candidate_roots("") == ()

    def test_the_duplicate_guard_cannot_fire(self):
        """THE PIN.

        ``here`` is an absolute path, so never empty, and the loop stops
        the moment ``dirname`` returns the same path it was given -- so
        every step is strictly shorter than the last and cannot already
        be in the list. The check is cheap insurance, not a live path.

        This fails if the ``parent == here`` break is removed, which is
        what would make a repeat possible.
        """
        from spacr import portable_paths

        source = inspect.getsource(portable_paths.candidate_roots)
        assert "if parent == here:" in source and "break" in source, (
            "the climb no longer stops at the filesystem root, so it can "
            "revisit a folder and the duplicate guard is live")
