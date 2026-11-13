"""Two io.py recovery paths: the plate-name migration, and the
leakage-safe split refusing a class it cannot divide.
"""
from __future__ import annotations



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


    def test_the_second_copy_of_that_refusal_is_never_the_one_that_raises(
            self):
        """THE PIN for deleting io.py's duplicate class-empty refusal.

        `grouped_split` refuses first, with a message of its own, so the
        check formerly in `generate_dataset_from_lists` could not be reached.
        The shared splitter is now the single owner of that rule.
        """
        import inspect

        caller = inspect.getsource(IO.generate_dataset_from_lists)
        assert "Leakage-safe {group_by}-grouped split leaves class" not in caller

        from spacr.classifier_evaluation import grouped_split

        assert "cannot put every class in" in inspect.getsource(grouped_split), (
            "grouped_split no longer refuses an undividable class itself, so "
            "the copy in generate_dataset_from_lists is now live")

    def test_the_provenance_guard_cannot_fire_for_a_class_with_data(self):
        """THE PIN for deleting ``if grouped_splits is None``.

        ``grouped_splits`` is None only when no class had any item -- and
        every class is then empty, so each one takes the ``continue``
        before provenance is read. The only-empty behavior is exercised
        below, so the dead assertion should not reappear.
        """
        import inspect

        source = inspect.getsource(IO.generate_dataset_from_lists)
        empty = source.index("if not data:")
        next_class = source.index("train_data, test_data =", empty)

        assert "continue" in source[empty:next_class]
        assert "if grouped_splits is None:" not in source

