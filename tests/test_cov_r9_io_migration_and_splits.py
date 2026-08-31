"""Two io.py recovery paths: the plate-name migration, and the
leakage-safe split refusing a class it cannot divide.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr import io as IO
from spacr.schema import KEY_SEPARATOR, escape_field_stem_plate


class TestEscapingIsNeverANoOp:

    def test_a_stem_with_a_raw_separator_in_the_plate_always_changes(self):
        """THE PIN, for ``if safe == stem: continue``.

        The check above it only lets through stems with MORE than four
        components, and four are fixed -- plate, well, field, time -- so
        a fifth means the plate itself holds a raw separator. Escaping
        one cannot leave the name alone, which is why the skip never
        runs.

        Driven over every arity the walker can meet rather than argued,
        because "cannot" is the kind of claim that quietly stops being
        true.
        """
        for extra in range(1, 5):
            plate = KEY_SEPARATOR.join(["exp"] + [str(n) for n in
                                                  range(1, extra + 1)])
            stem = KEY_SEPARATOR.join([plate, "A01", "1", "1"])

            assert len(stem.split(KEY_SEPARATOR)) > 4
            assert escape_field_stem_plate(stem, timelapse=True) != stem, (
                f"{stem!r} escaped to itself, so the skip in "
                f"migrate_unescaped_plate_names is live and untested")

    def test_an_ordinary_plate_is_not_even_considered(self):
        """The guard above: four components is the whole grammar, so a
        plate with no separator never reaches the escape at all. That is
        what makes the migration safe to run over a folder that does not
        need it."""
        stem = KEY_SEPARATOR.join(["plate1", "A01", "1", "1"])

        assert len(stem.split(KEY_SEPARATOR)) == 4


class TestTheMigration:

    def _plate(self, tmp_path, stem):
        merged = tmp_path / "merged"
        merged.mkdir(parents=True, exist_ok=True)
        np.save(merged / f"{stem}.npy", np.zeros((2, 2), dtype=np.uint8))
        return merged

    def test_a_plate_with_a_separator_is_planned_and_renamed(self, tmp_path):
        merged = self._plate(tmp_path, "exp_1_A01_1_1")

        planned = IO.migrate_unescaped_plate_names(str(tmp_path))

        assert len(planned) == 1
        old, new = planned[0]
        assert old.endswith("exp_1_A01_1_1.npy")
        assert "%5F" in os.path.basename(new)
        assert not os.path.exists(old)
        assert os.path.exists(new)
        assert [p.name for p in merged.iterdir()] == [os.path.basename(new)]

    def test_a_dry_run_reports_without_moving_anything(self, tmp_path):
        merged = self._plate(tmp_path, "exp_1_A01_1_1")

        planned = IO.migrate_unescaped_plate_names(str(tmp_path),
                                                   dry_run=True)

        assert len(planned) == 1
        assert [p.name for p in merged.iterdir()] == ["exp_1_A01_1_1.npy"]

    def test_an_ordinary_plate_is_left_alone(self, tmp_path):
        merged = self._plate(tmp_path, "plate1_A01_1_1")

        assert IO.migrate_unescaped_plate_names(str(tmp_path)) == []
        assert [p.name for p in merged.iterdir()] == ["plate1_A01_1_1.npy"]

    def test_a_second_run_is_a_no_op(self, tmp_path):
        """The property the >4-component test buys, and the reason it is
        not "does escaping change the name": escaping is NOT idempotent,
        so a migration that re-escaped its own output would corrupt on
        its second run."""
        self._plate(tmp_path, "exp_1_A01_1_1")

        first = IO.migrate_unescaped_plate_names(str(tmp_path))
        second = IO.migrate_unescaped_plate_names(str(tmp_path))

        assert len(first) == 1
        assert second == []

    def test_an_occupied_destination_refuses_before_moving_anything(
            self, tmp_path):
        """A half-applied rename is worse than none."""
        merged = self._plate(tmp_path, "exp_1_A01_1_1")
        np.save(merged / "exp%5F1_A01_1_1.npy", np.zeros((2, 2), np.uint8))

        with pytest.raises(FileExistsError) as caught:
            IO.migrate_unescaped_plate_names(str(tmp_path))

        assert "refusing to migrate" in str(caught.value)
        assert (merged / "exp_1_A01_1_1.npy").exists(), (
            "the source was moved despite the refusal")

    def test_a_file_of_another_kind_is_not_touched(self, tmp_path):
        """The suffix filter, which is the walk's first skip.

        A folder holds sidecars and notes beside its arrays, and
        renaming one on a guess is how a migration loses a file.
        """
        merged = tmp_path / "merged"
        merged.mkdir(parents=True)
        (merged / "exp_1_A01_1_1.json").write_text("{}")
        (merged / "notes_about_exp_1_and_more.txt").write_text("hello")

        assert IO.migrate_unescaped_plate_names(str(tmp_path)) == []
        assert sorted(p.name for p in merged.iterdir()) == [
            "exp_1_A01_1_1.json", "notes_about_exp_1_and_more.txt"]


class TestTheLeakageSafeSplit:

    def _crops(self, tmp_path, wells, cls):
        folder = tmp_path / "crops"
        folder.mkdir(parents=True, exist_ok=True)
        made = []
        for index, well in enumerate(wells):
            name = f"plate1_{well}_1_1_cell_{index + 1}.png"
            path = folder / name
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 16)
            made.append(str(path))
        return made

    def test_a_class_confined_to_one_well_is_refused_by_name(self, tmp_path):
        """THE UNCOVERED ARC: the grouped split leaves a class empty.

        Every crop of the class comes from a single well, so the whole
        class lands on one side of a well-grouped split. Training on that
        would report a test score for a class the model never had to
        generalise across -- the exact leakage the grouping prevents --
        so it is refused, and the message says which class and what to
        change rather than naming the splitter's parameters.
        """
        confined = self._crops(tmp_path, ["A01"] * 6, "positive")
        spread = self._crops(tmp_path, ["B01", "B02", "B03", "B04",
                                        "B05", "B06"], "negative")

        with pytest.raises(ValueError) as caught:
            IO.generate_dataset_from_lists(
                str(tmp_path / "out"), [confined, spread],
                ["positive", "negative"], test_split=0.5, group_by="well")

        message = str(caught.value)
        assert "leakage-safe well-grouped split cannot put every class in " \
            "both train and test" in message
        assert "choose a finer level" in message, (
            "the refusal no longer says what to change, so a user is told "
            "the split failed and not how to make it work")
        assert "a random fallback would report memorisation as transfer" in \
            message, (
            "the refusal no longer says WHY it will not simply fall back, "
            "which is the question the next reader asks")

    def test_the_second_copy_of_that_refusal_is_never_the_one_that_raises(
            self):
        """THE PIN, for io.py's own "Leakage-safe ... leaves class empty".

        `grouped_split` refuses first, with a message of its own, so the
        check in `generate_dataset_from_lists` cannot be reached. Two
        copies of one rule, and the one a reader finds by grepping the
        caller is not the one that runs -- which is the whole reason to
        write this down.
        """
        import inspect

        caller = inspect.getsource(IO.generate_dataset_from_lists)
        assert "Leakage-safe {group_by}-grouped split leaves class" in caller

        from spacr.classifier_evaluation import grouped_split

        assert "cannot put every class in" in inspect.getsource(grouped_split), (
            "grouped_split no longer refuses an undividable class itself, so "
            "the copy in generate_dataset_from_lists is now live")

    def test_the_provenance_guard_cannot_fire_for_a_class_with_data(self):
        """THE PIN, for ``if grouped_splits is None: raise RuntimeError``.

        ``grouped_splits`` is None only when no class had any item -- and
        every class is then empty, so each one takes the ``continue``
        four lines above and no iteration reaches the guard. It is a
        provenance assertion, not a path.
        """
        import inspect

        source = inspect.getsource(IO.generate_dataset_from_lists)
        empty = source.index("if not data:")
        guard = source.index("if grouped_splits is None:", empty)

        assert empty < guard, (
            "the provenance guard now runs before the empty-class skip, so "
            "a dataset of only-empty classes raises instead of writing the "
            "empty folders the class list needs")
        assert "continue" in source[empty:guard]

    def test_only_empty_classes_write_their_folders_and_no_split(
            self, tmp_path):
        """DRIVEN, because that is the state the guard is asserting about:
        nothing to split, and the class folders still created so the tree
        matches the class list."""
        dst = tmp_path / "out"

        IO.generate_dataset_from_lists(str(dst), [[], []], ["a", "b"])

        for split in ("train", "test"):
            for cls in ("a", "b"):
                assert (dst / split / cls).is_dir()
        assert not (dst / ".spacr_split.json").exists(), (
            "a split report was written for a dataset with nothing in it")
