"""Every join was left, so which cells survived was never a decision.

`_read_and_join_tables` joined cytoplasm, nucleus, pathogen, organelle and
png_list onto the cell table with ``how='left'`` throughout. That is one
answer applied to five questions that have different answers:

- a cell object with no NUCLEUS is not an uninfected cell, it is debris or a
  fragment at the image edge. Kept, it contributes a row whose nuclear
  measurements are all missing, and every ratio computed from them is NaN.
- a cell with no CROP cannot be classified, annotated or displayed. Kept, it
  reached the classifier as a row that could only be dropped later -- after
  the cell counts had been reported.
- a cell with no PATHOGEN is an uninfected cell, and in a screen it is
  usually the control population. Dropping it silently conditions every
  result on infection.

So nucleus and png_list are inner, pathogen and organelle are left, and the
last of those is reversible through a setting rather than a code edit.
"""

import inspect

import pytest

from spacr.io import _read_and_join_tables
from spacr.object_roles import CHILD_ROLES, JOIN_HOW, join_how


# ---------------------------------------------------------------------------
# the five answers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table,how", [
    ("nucleus", "inner"),
    ("png_list", "inner"),
    ("cytoplasm", "left"),
    ("pathogen", "left"),
    ("organelle", "left"),
])
def test_each_table_joins_the_way_its_biology_requires(table, how):
    assert join_how(table) == how


def test_the_defaults_are_not_all_the_same_answer():
    """The regression in one line: five 'left' is what this replaced."""
    assert len(set(JOIN_HOW.values())) > 1


def test_every_child_role_has_a_declared_direction():
    for role in CHILD_ROLES:
        assert role in JOIN_HOW


def test_an_undeclared_table_keeps_its_rows_rather_than_dropping_them():
    """An unknown table must not silently narrow the population."""
    assert join_how("mitochondria") == "left"


@pytest.mark.parametrize("spelling", ["Nucleus", " nucleus ", "NUCLEUS"])
def test_the_table_name_is_matched_however_it_is_spelled(spelling):
    assert join_how(spelling) == "inner"


# ---------------------------------------------------------------------------
# restricting to infected cells is deliberate
# ---------------------------------------------------------------------------

def test_keep_uninfected_false_turns_the_pathogen_join_inner():
    assert join_how("pathogen", keep_uninfected=False) == "inner"
    assert join_how("organelle", keep_uninfected=False) == "inner"


def test_it_does_not_touch_the_joins_that_are_not_about_infection():
    """A cell with no nucleus is not an uninfected cell."""
    for table in ("nucleus", "png_list"):
        assert join_how(table, keep_uninfected=False) == "inner"
    assert join_how("cytoplasm", keep_uninfected=False) == "left"


def test_uninfected_cells_are_kept_unless_asked_otherwise():
    """The default must not quietly condition a screen on infection."""
    assert join_how("pathogen") == "left"
    assert inspect.signature(join_how).parameters[
        "keep_uninfected"].default is True


# ---------------------------------------------------------------------------
# the setting reaches the reader
# ---------------------------------------------------------------------------

def test_the_reader_exposes_the_setting():
    parameters = inspect.signature(_read_and_join_tables).parameters
    assert "keep_uninfected" in parameters
    assert parameters["keep_uninfected"].default is True


def test_the_reader_asks_the_registry_rather_than_hard_coding_left():
    """A literal 'left' beside a merge is how the five answers became one."""
    source = inspect.getsource(_read_and_join_tables)
    assert "join_how(" in source
    assert "how='left'" not in source, (
        "a join direction is hard-coded again; put it in JOIN_HOW")
