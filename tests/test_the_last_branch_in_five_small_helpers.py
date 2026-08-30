"""Five more last branches, in small helpers the rest of spaCR leans on.

Each of these functions is called from several places and none of them is more
than a screenful, which is exactly why the missing arc survived: a short
function reads as obviously correct, and "obviously correct" is not a test.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# well_spec.control_block_wells — arc 70 -> 68, a blank or repeated well
# ---------------------------------------------------------------------------

def test_control_wells_are_de_duplicated_and_blanks_dropped():
    """The ``if text and text not in seen:`` branch not taken, both ways.

    The three control-block settings overlap by design -- a plate can declare
    the same column as both a negative and a positive block by mistake, and
    users leave trailing blanks in list fields constantly. A duplicate well
    here would be excluded from the regression twice, which is harmless, but a
    BLANK one would be matched against every well whose name starts with
    nothing, which is not.
    """
    from spacr.well_spec import CONTROL_BLOCK_SETTINGS, control_block_wells

    keys = list(CONTROL_BLOCK_SETTINGS)
    settings = {keys[0]: ["A01", "  ", "A01", "A02"]}
    if len(keys) > 1:
        settings[keys[1]] = ["A02", ""]          # repeat across two blocks

    assert control_block_wells(settings) == ["A01", "A02"]


def test_no_control_blocks_at_all_is_an_empty_list():
    """The ``continue`` above, which the de-duplication must not depend on."""
    from spacr.well_spec import control_block_wells

    assert control_block_wells({}) == []
    assert control_block_wells(None) == []


def test_a_single_well_need_not_be_wrapped_in_a_list():
    """A scalar is accepted, which is how the settings panel stores one well."""
    from spacr.well_spec import CONTROL_BLOCK_SETTINGS, control_block_wells

    key = list(CONTROL_BLOCK_SETTINGS)[0]
    assert control_block_wells({key: "B07"}) == ["B07"]


# ---------------------------------------------------------------------------
# png_list._object_id_int — arc 41 -> 43, an id with no 'o' prefix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("o12", 12),          # the prefixed spelling the strip exists for
    ("O12", 12),
    ("12", 12),           # THE ARC: no prefix, nothing stripped
    ("  7 ", 7),
    ("", None),
    ("cell", None),
    (None, None),
])
def test_an_object_id_is_read_with_or_without_its_o_prefix(value, expected):
    """The ``if text[:1] in ('o', 'O'):`` branch, both sides.

    Object ids are written both ways across spaCR's own tables -- ``o12`` in
    the crop file names and a bare integer in the measurement columns -- so
    the unprefixed form is not an edge case, it is half the inputs. Stripping
    a character that is not there would turn 12 into 2.
    """
    from spacr.png_list import _object_id_int

    assert _object_id_int(value) == expected


def test_a_numeric_object_id_keeps_its_value():
    """Integers and floats bypass the text path entirely."""
    from spacr.png_list import _object_id_int

    assert _object_id_int(12) == 12
    assert _object_id_int(12.0) == 12
    assert _object_id_int(float("nan")) is None


# ---------------------------------------------------------------------------
# png_list._merged_field_paths — arc 79 -> 70, the first table was empty
# ---------------------------------------------------------------------------

def test_an_empty_first_table_falls_through_to_the_next(tmp_path):
    """The ``if out:`` branch not taken, so the loop tries the next table.

    The tables are tried in preference order, and a screen that ran only part
    of the pipeline has the earlier table present but EMPTY. Breaking on the
    first table that merely exists would return nothing and report the screen
    as having no fields, when the answer was in the next table along.
    """
    from spacr.png_list import _merged_field_paths

    db = tmp_path / "measurements.db"
    conn = sqlite3.connect(str(db))
    try:
        for table in ("cell", "cytoplasm", "nucleus", "pathogen"):
            conn.execute(
                f'CREATE TABLE "{table}" (plateID TEXT, rowID TEXT, '
                f'columnID TEXT, fieldID TEXT, path_name TEXT, file_name TEXT)')
        # Deliberately leave the earlier tables empty and fill a later one.
        # cell and cytoplasm exist but are empty; pathogen has the answer.
        conn.execute('INSERT INTO pathogen VALUES ("p1","r1","c1","f1",'
                     '"/data/merged","p1_r1_c1_f1.npy")')
        conn.commit()
    finally:
        conn.close()

    found = _merged_field_paths(str(db))

    assert found == {("p1", "r1", "c1", "f1"): ("/data/merged",
                                                "p1_r1_c1_f1.npy")}


# ---------------------------------------------------------------------------
# portable_paths.RerootReport.describe — arc 180 -> 184, nothing unresolved
# ---------------------------------------------------------------------------

def test_a_clean_reroot_describes_only_what_it_moved():
    """The ``if self.unresolved:`` branch not taken.

    The happy case: every path was placed. The sentence must not acquire a
    trailing "0 could not be placed" clause, because a count of zero in a
    warning is what teaches users to stop reading the warnings.
    """
    from spacr.portable_paths import RerootReport

    line = RerootReport(column="png_path", moved=60816, unresolved=0,
                        root="/data/plate1").describe()

    assert "re-rooted 60,816 png_path value(s)" in line
    assert "could not be placed" not in line


def test_a_partial_reroot_names_the_first_path_it_could_not_place():
    """The taken side -- the actionable case the report exists for."""
    from spacr.portable_paths import RerootReport

    line = RerootReport(column="png_path", moved=10, unresolved=3,
                        first_unresolved="/old/a.png",
                        root="/data/plate1").describe()

    assert "re-rooted 10" in line
    assert "3 could not be placed" in line
    assert "/old/a.png" in line


def test_a_route_that_is_not_on_this_machine_says_so_instead():
    """The ``absent`` short-circuit above, which must win over the clause above."""
    from spacr.portable_paths import RerootReport

    line = RerootReport(column="path_name", moved=0, unresolved=60816,
                        first_unresolved="/old/a.npy",
                        root="/data/plate1").describe()

    assert "are not on this machine" in line
    assert "re-rooted" not in line


# ---------------------------------------------------------------------------
# settings_spec._cellpose_model_names — line 128, the cold fallback
# ---------------------------------------------------------------------------

def test_a_cold_process_offers_only_the_shipped_model():
    """The ``return ["cpsam"]`` a cold settings-panel build takes.

    It cannot be reached in this process: importing the test suite has already
    put ``spacr.settings`` in sys.modules, and that is the whole condition. So
    it is checked in a FRESH interpreter, which is also the only honest way to
    test something whose contract is "before anything heavy is imported" --
    the point is that neither Cellpose nor spacr.settings gets loaded, and a
    test that had already loaded them would prove nothing.
    """
    code = (
        "import sys;"
        "import spacr.settings_spec as s;"
        "names = s._cellpose_model_names();"
        "assert 'spacr.settings' not in sys.modules, 'settings was imported';"
        "assert 'cellpose.models' not in sys.modules, 'cellpose was imported';"
        "print(names)"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True,
                         cwd="/mnt/firecuda2/codex/repo/spacr")

    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "['cpsam']"


def test_the_cold_fallback_is_taken_whenever_neither_module_is_loaded(monkeypatch):
    """The same ``return ["cpsam"]``, reached in-process so it is measured.

    The subprocess test above proves the condition really does hold on a cold
    interpreter -- that neither module gets imported as a side effect. This one
    proves the BRANCH does what it says, by removing the two entries from
    sys.modules for the length of the call. Both are worth having: the first
    would still pass if the function had been rewritten to return something
    else, and the second would still pass if importing settings_spec had begun
    dragging Cellpose in behind it.
    """
    import spacr.settings_spec as settings_spec

    monkeypatch.delitem(sys.modules, "spacr.settings", raising=False)
    monkeypatch.delitem(sys.modules, "cellpose.models", raising=False)

    assert settings_spec._cellpose_model_names() == ["cpsam"]


def test_a_warm_process_asks_settings_for_the_full_model_list():
    """The taken side: once settings is loaded, the real list is used.

    Importing it here is what makes the condition false, which is exactly the
    state the docstring describes as "already loaded".
    """
    import spacr.settings                                    # noqa: F401
    import spacr.settings_spec as settings_spec

    names = settings_spec._cellpose_model_names()

    assert isinstance(names, list) and names
    assert "cpsam" in names
