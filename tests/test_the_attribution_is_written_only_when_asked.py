"""173's last owed piece: an OPT-IN write of the attribution into png_list.

"Never automatic. Writing into measurements.db is not something a viewer does
behind the user, which is the rule the montage already keeps."

IT IS NOT A GENOTYPE, AND THE COLUMN NAMES SAY SO. The pooled design never
observed which cell carried what. `_attributed` is in every column name for
that reason, and the note is written into the database beside them so the
claim travels with the data rather than living in a docstring.
"""
from __future__ import annotations

import sqlite3

import pytest

from spacr.attribution_columns import (ATTRIBUTION_NOTE, COLUMNS, NOTE_KEY,
                                       AttributionWriteError, describe,
                                       rows_from, write)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE png_list (png_path TEXT, prcfo TEXT, cell_id TEXT)")
        connection.execute(
            "CREATE TABLE settings (setting_key TEXT PRIMARY KEY, "
            "setting_value TEXT)")
        connection.executemany(
            "INSERT INTO png_list VALUES (?, ?, ?)",
            [(f"/x/{i}.png", f"plate1_r1_c1_f1_o{i}", f"o{i}")
             for i in range(1, 6)])
    return str(path)


def _rows(keys, guide="000000_1"):
    class _Call:
        def __init__(self, guide, p, entropy):
            self.guide, self.probability, self.entropy = guide, p, entropy

    calls = [_Call(guide, 0.9, 0.3) for _ in keys]
    return rows_from(calls, keys, coverage=0.82)


class TestItRefusesUnlessAsked:

    def test_the_default_is_a_refusal(self, database):
        with pytest.raises(AttributionWriteError, match="opt-in"):
            write(database, _rows(["plate1_r1_c1_f1_o1"]))

    def test_the_refusal_says_why_rather_than_how(self, database):
        """"not something a viewer does behind them" is the reason, and a
        reader who only learns the flag will pass the flag."""
        with pytest.raises(AttributionWriteError) as raised:
            write(database, _rows(["plate1_r1_c1_f1_o1"]))

        assert "behind" in str(raised.value)

    def test_confirmed_writes(self, database):
        out = write(database, _rows(["plate1_r1_c1_f1_o1"]), confirmed=True)

        assert out["matched"] == 1


class TestWhatItWrites:

    def test_the_five_columns_are_added(self, database):
        write(database, _rows(["plate1_r1_c1_f1_o1"]), confirmed=True)

        with sqlite3.connect(database) as connection:
            present = {row[1] for row in
                       connection.execute("PRAGMA table_info('png_list')")}

        assert set(COLUMNS) <= present

    def test_every_column_name_says_it_is_an_attribution(self):
        """A reader who takes `grna_attributed` for an observation has been
        misled by the name alone."""
        for name in COLUMNS:
            assert "attribut" in name, name

    def test_the_values_land_on_the_right_row(self, database):
        keys = ["plate1_r1_c1_f1_o2"]
        write(database, _rows(keys), confirmed=True)

        with sqlite3.connect(database) as connection:
            got = dict(connection.execute(
                "SELECT prcfo, grna_attributed FROM png_list "
                "WHERE grna_attributed IS NOT NULL").fetchall())

        assert got == {"plate1_r1_c1_f1_o2": "000000_1"}

    def test_the_gene_comes_off_the_guide_when_none_is_given(self, database):
        write(database, _rows(["plate1_r1_c1_f1_o1"]), confirmed=True)

        with sqlite3.connect(database) as connection:
            gene = connection.execute(
                "SELECT gene_attributed FROM png_list WHERE prcfo = ?",
                ("plate1_r1_c1_f1_o1",)).fetchone()[0]

        assert gene == "000000"

    def test_an_ambiguous_cell_gets_no_gene(self, database):
        rows = _rows(["plate1_r1_c1_f1_o1"], guide="ambiguous")

        write(database, rows, confirmed=True)

        with sqlite3.connect(database) as connection:
            guide, gene = connection.execute(
                "SELECT grna_attributed, gene_attributed FROM png_list "
                "WHERE prcfo = ?", ("plate1_r1_c1_f1_o1",)).fetchone()

        assert guide == "ambiguous" and not gene

    def test_the_claim_is_written_into_the_database(self, database):
        """So it travels with the data rather than living in a docstring."""
        write(database, _rows(["plate1_r1_c1_f1_o1"]), confirmed=True)

        with sqlite3.connect(database) as connection:
            note = connection.execute(
                "SELECT setting_value FROM settings WHERE setting_key = ?",
                (NOTE_KEY,)).fetchone()[0]

        assert note == ATTRIBUTION_NOTE
        assert "not an observation" in note
        assert "pooled" in note


class TestDoingItTwice:
    """An attribution is something a user redoes with a different threshold."""

    def test_a_second_write_is_not_an_error(self, database):
        write(database, _rows(["plate1_r1_c1_f1_o1"]), confirmed=True)

        out = write(database, _rows(["plate1_r1_c1_f1_o1"]), confirmed=True)

        assert out["added"] == 0, "the columns are already there"
        assert out["matched"] == 1

    def test_a_second_write_replaces_the_first(self, database):
        write(database, _rows(["plate1_r1_c1_f1_o1"]), confirmed=True)
        write(database, _rows(["plate1_r1_c1_f1_o1"], guide="233460_4"),
              confirmed=True)

        with sqlite3.connect(database) as connection:
            guide = connection.execute(
                "SELECT grna_attributed FROM png_list WHERE prcfo = ?",
                ("plate1_r1_c1_f1_o1",)).fetchone()[0]

        assert guide == "233460_4"


class TestItFailsOutLoud:

    def test_a_database_with_no_png_list_says_which_tables_it_has(self,
                                                                  tmp_path):
        path = tmp_path / "empty.db"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE cell (object_label INTEGER)")

        with pytest.raises(AttributionWriteError, match="no png_list"):
            write(str(path), _rows(["x"]), confirmed=True)

    def test_a_missing_key_column_names_what_is_there(self, tmp_path):
        path = tmp_path / "nokey.db"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE png_list (png_path TEXT)")

        with pytest.raises(AttributionWriteError, match="png_path"):
            write(str(path), _rows(["x"]), confirmed=True)

    def test_nothing_to_write_is_not_an_error(self, database):
        assert write(database, [], confirmed=True) == {"matched": 0,
                                                       "added": 0}


class TestTheDescriptionForTheConfirmation:

    def test_it_counts_what_would_happen(self):
        rows = _rows(["a", "b"]) + _rows(["c"], guide="ambiguous")

        said = describe(rows)

        assert "3 cell(s)" in said
        assert "2 attributed" in said and "1 left ambiguous" in said

    def test_it_says_it_is_not_a_genotype(self):
        assert "not a" in describe(_rows(["a"])) and \
            "genotype" in describe(_rows(["a"]))
