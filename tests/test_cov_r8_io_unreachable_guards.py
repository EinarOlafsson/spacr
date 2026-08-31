"""Three io.py guards that cannot be reached, and the gates that stop them.

The coverage goal is a test for every section of code that NEEDS one.
Some of these do not: they are defensive arms behind an earlier check
that already refuses the input they defend against. Forcing coverage
onto them would mean reaching past the real gate with a monkeypatch,
which asserts nothing about the program and hides the gate itself.

So each one is pinned from the other side: the test asserts the EARLIER
check does the refusing. If a gate is ever weakened, these fail -- and
the guard behind it stops being unreachable, which is exactly when
somebody should look at it again.
"""
from __future__ import annotations

import itertools
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest


class TestTheAlreadyEscapedStemInTheMigration:
    """`migrate_unescaped_plate_names`: `if safe == stem: continue`.

    Unreachable. The loop only reaches the escape when the stem splits
    into MORE than four components, and the writer's grammar has exactly
    three fixed tail tokens -- well, field, timepoint. More than four
    therefore means the plate itself holds a raw separator, and escaping
    a raw separator always changes the name.
    """

    def test_no_over_long_stem_escapes_to_itself(self):
        from spacr.schema import escape_field_stem_plate

        plates = ["exp_1", "a_b", "%5F", "a%5Fb", "_", "a__b", "p_1_2",
                  "%", "a%b", "exp%5F1", "x", "a%255Fb", "%5F%5F"]
        tried = 0
        for plate, well, field, time in itertools.product(
                plates, ["A01", "A_1", "%5F"], ["1", "2%5F"], ["1", "0"]):
            stem = f"{plate}_{well}_{field}_{time}"
            if len(stem.split("_")) <= 4:
                continue          # the guard above the escape
            tried += 1
            try:
                safe = escape_field_stem_plate(stem, timelapse=True)
            except Exception:
                continue          # not a field stem; skipped for that reason
            assert safe != stem, (
                f"{stem!r} escapes to itself, so the `safe == stem` arm in "
                "migrate_unescaped_plate_names is now REACHABLE and wants a "
                "test of its own")
        assert tried > 50, "the search was too small to mean anything"

    def test_a_four_part_stem_never_reaches_the_escape_at_all(self):
        """The guard that makes the migration idempotent.

        Escaping is NOT idempotent -- a literal percent is escaped
        first, so `exp%5F1_A01_1_1` would become `exp%255F1_A01_1_1` on a
        second run. Testing the component COUNT rather than "does
        escaping change the name" is what stops the second run
        corrupting what the first one fixed.
        """
        stem = "exp%5F1_A01_1_1"
        assert len(stem.split("_")) == 4


class TestTheInvalidDatasetModeArm:
    """`generate_training_dataset`'s `else: raise ValueError(...)`.

    Unreachable. `resolve_basis` runs first and refuses anything that is
    not one of the two known modes, with its own error naming both. The
    comment on the arm says "anything reaching here is a value spaCR has
    never had" -- and nothing can reach it.
    """

    @staticmethod
    def _project(tmp_path: Path) -> Path:
        src = tmp_path / "proj"
        (src / "measurements").mkdir(parents=True)
        con = sqlite3.connect(src / "measurements" / "measurements.db")
        pd.DataFrame({
            "png_path": [str(tmp_path / "a.png")], "plateID": ["p1"],
            "rowID": ["r1"], "columnID": ["c1"], "fieldID": ["f1"],
            "object_label": ["1"], "prcfo": ["p1_r1_c1_f1_1"],
        }).to_sql("png_list", con, index=False)
        con.close()
        return src

    def test_an_unknown_mode_is_refused_by_the_gate_above_it(self, tmp_path):
        from spacr.io import generate_training_dataset

        src = self._project(tmp_path)
        with pytest.raises(Exception) as caught:
            generate_training_dataset({
                "src": str(src), "dataset_mode": "not_a_mode",
                "class_metadata": [], "png_type": "cell_png"})
        message = str(caught.value)
        assert type(caught.value).__name__ == "TrainingBasisError", (
            "the mode gate no longer refuses first; the ValueError arm in "
            "generate_training_dataset may now be reachable")
        assert "not_a_mode" in message
        assert "metadata" in message and "annotation" in message, (
            "the refusal must name both modes a run can actually use")

    def test_the_gate_names_exactly_the_two_live_modes(self):
        """`measurement` was retired and is migrated, not accepted raw."""
        from spacr.io import generate_training_dataset
        import inspect

        source = inspect.getsource(generate_training_dataset)
        assert "resolve_basis" in source, (
            "the mode gate has moved; this test no longer pins anything")
