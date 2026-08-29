"""Which measurements mean something for which organelle.

Instruction 72's sixth item, 2026-08-10: the sweep of organelle was asked
for "in measure and mask", and the measure half is the question of which
measurements make sense per type. Instruction 71 answered it in prose --
"YES for the punctate/vesicular families, where 'how many and how clustered'
is the phenotype. Probably meaningless for a reticular or cisternal organelle
that is one connected object per cell" -- and deferred the gate to the type
this instruction introduced.

NOTHING IS SWITCHED OFF. A family the type makes doubtful is still measured:
a number that vanished without being asked to is worse than one that comes
with a caveat, and this project already carries eleven phantom settings from
values nobody could see. What the type buys is that the run SAYS SO.
"""
from __future__ import annotations

import pytest

import spacr.settings as S
from spacr.organelle_types import ORGANELLE_TYPES


def _measure(**overrides):
    settings = {"organelle_mask_dim": 7, "spatial_measurements": True}
    settings.update(overrides)
    return settings


# ---------------------------------------------------------------------------
# The answer itself
# ---------------------------------------------------------------------------

def test_every_type_the_tool_offers_has_an_answer():
    """A type with no verdict would silently fall through as 'no comment'."""
    assert set(S.ORGANELLE_OBJECTS_PER_CELL) == set(ORGANELLE_TYPES)


def test_a_reticulum_is_one_object_and_a_punctum_is_many():
    """The two ends of the maintainer's own list, which is what decides it."""
    assert S.ORGANELLE_OBJECTS_PER_CELL["reticular"] == "one"
    assert S.ORGANELLE_OBJECTS_PER_CELL["punctate"] == "many"
    assert not S.organelle_counting_is_meaningful("reticular")
    assert S.organelle_counting_is_meaningful("punctate")


def test_a_category_that_mixes_both_says_so_rather_than_picking():
    """"Spherical" holds the nucleus and macromolecular condensates.

    One is a single object per cell and the other is dozens, so a verdict
    either way would be wrong for half the structures the maintainer listed
    under the name. The honest answer is that the structure decides, not the
    category -- the same finding as the type-to-morphology mapping, where
    two of the nine rows split on size.
    """
    assert S.ORGANELLE_OBJECTS_PER_CELL["spherical"] == "depends"
    assert S.ORGANELLE_OBJECTS_PER_CELL["toroidal"] == "depends"
    # Still measured: a maybe may not delete a real number.
    assert S.organelle_counting_is_meaningful("spherical")


def test_custom_makes_no_claim():
    """It recommends nothing, so it is told nothing about what was imaged."""
    assert S.ORGANELLE_OBJECTS_PER_CELL["custom"] == "unknown"
    assert S.organelle_measurement_caveats(_measure(organelle_type="custom")) == []


def test_every_one_per_cell_type_says_why_in_the_user_s_own_words():
    """"Meaningless" with no reason reads as a bug in the tool."""
    for name, verdict in S.ORGANELLE_OBJECTS_PER_CELL.items():
        if verdict in ("one", "depends"):
            reason = S.ORGANELLE_COUNT_REASONS.get(name, "")
            assert len(reason.split()) >= 12, (name, reason)


# ---------------------------------------------------------------------------
# What a run is told
# ---------------------------------------------------------------------------

def test_a_reticular_slot_is_warned_about_its_neighbour_counts():
    caveats = S.organelle_measurement_caveats(
        _measure(organelle_type="reticular", object_distances=True))
    settings_named = {name for _label, name, _why in caveats}
    assert settings_named == {"spatial_measurements", "object_distances"}
    assert all(label == "Organelle 1" for label, _n, _w in caveats)
    assert all("one connected mesh" in why for _l, _n, why in caveats)


def test_a_punctate_slot_is_told_nothing():
    """A run measuring what these families were designed for gets no
    paragraph telling it everything is fine."""
    assert S.organelle_measurement_caveats(
        _measure(organelle_type="punctate", object_distances=True)) == []


def test_a_family_that_is_off_is_not_warned_about():
    """The caveat is about numbers the run will actually produce."""
    assert S.organelle_measurement_caveats(_measure(
        organelle_type="reticular", spatial_measurements=False)) == []


def test_a_slot_with_no_mask_to_measure_is_not_warned_about():
    """Slots two to four are off by default and say nothing."""
    caveats = S.organelle_measurement_caveats({
        "organelle_mask_dim": 7, "organelle_type": "punctate",
        "organelleb_type": "reticular", "spatial_measurements": True,
    })
    assert caveats == []


def test_a_second_slot_with_a_mask_is_warned_about_by_its_own_name():
    caveats = S.organelle_measurement_caveats({
        "organelle_mask_dim": 7, "organelle_type": "punctate",
        "organelleb_mask_dim": 8, "organelleb_type": "cisternal",
        "spatial_measurements": True,
    })
    assert [label for label, _n, _w in caveats] == ["Organelle 2"]


def test_a_settings_file_written_before_the_type_existed_says_nothing():
    """It is not making a claim about what it imaged, so neither is spaCR."""
    assert S.organelle_measurement_caveats({
        "organelle_mask_dim": 7, "spatial_measurements": True}) == []


# ---------------------------------------------------------------------------
# It reaches the measure module rather than sitting in a table
# ---------------------------------------------------------------------------

def test_measure_offers_the_type_for_every_slot():
    """Measure cannot answer the question without being told what it is
    measuring, and the mask module's copy never reaches a measure run."""
    defaults = S.get_measure_crop_settings({"number_of_organelles": 4})
    from spacr.object_roles import ORGANELLE_ROLES
    for role in ORGANELLE_ROLES:
        assert defaults[f"{role}_type"] == S.DEFAULT_ORGANELLE_TYPE


def test_a_verbose_measure_run_prints_the_caveat(capsys):
    S.get_measure_crop_settings({
        "verbose": True, "organelle_mask_dim": 7,
        "organelle_type": "filamentous", "spatial_measurements": True,
    })
    printed = capsys.readouterr().out
    assert "Organelle 1" in printed
    assert "spatial_measurements" in printed
    assert "cytoskeleton" in printed


def test_a_quiet_measure_run_prints_nothing_about_it(capsys):
    S.get_measure_crop_settings({
        "organelle_mask_dim": 7, "organelle_type": "filamentous",
        "spatial_measurements": True,
    })
    assert "Organelle 1" not in capsys.readouterr().out


def test_the_type_is_on_the_measure_panel_and_not_in_a_leftover_bucket(qapp):
    """"Additional Settings" is not a heading anyone chose; it is the
    absence of one, and a setting that lands there was not placed.

    Read off the RENDERED panel rather than off `categories_for_app`: that
    function always builds the leftover bucket from the shared category map,
    and it is `build_sections` that drops the keys the module does not
    offer. Asserting on the map would have passed with every organelle type
    sitting in the bucket.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import (SettingsWidgets,
                                                 categories_for_app)

    layout = categories_for_app("measure", S.categories)
    assert "organelle_type" in layout["Mask & Channel Mapping"]

    sections = SettingsWidgets(
        "measure", current={"number_of_organelles": 1}).build_sections()
    titles = [title for title, _rows in sections]
    assert "Additional Settings" not in titles and "Other" not in titles
    mapping = dict(sections)["Mask & Channel Mapping"]
    assert any("Type" in label for label, _widget in mapping)


# ---------------------------------------------------------------------------
# And the run journal carries it, not only the console
# ---------------------------------------------------------------------------

@pytest.fixture
def scratch_run_logs(tmp_path, monkeypatch):
    """Per-run JSONL logs in a scratch folder, with the root logger open."""
    import logging

    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    root = logging.getLogger()
    before = root.level
    root.setLevel(logging.DEBUG)
    yield tmp_path / "logs"
    root.setLevel(before)


def test_the_run_journal_carries_the_caveat(scratch_run_logs):
    """The console is not where a batch is read back from.

    `explain_organelle_measurements` prints the sentence and stops there, so
    a measurements.db opened a week later carried the count-dependent columns
    with nothing beside them saying what they mean for a one-object-per-cell
    organelle. The measure run puts the same sentence on its own journal,
    stamped with the run id the database is stamped with, so
    `read_run_log(run_id)` gives them back together.
    """
    from spacr.measure import _record_organelle_caveats
    from spacr.runctx import read_run_log, run_context

    settings = {"organelle_mask_dim": 7, "organelle_type": "reticular",
                "spatial_measurements": True}

    with run_context("measure", {}) as run:
        recorded = _record_organelle_caveats(settings, run)
        run_id = run.run_id

    assert {setting for _label, setting, _why in recorded} == {
        "spatial_measurements"}

    journal = read_run_log(run_id, contains="[organelle]")
    assert journal, "the caveat never reached the run journal"
    said = " ".join(record["message"] for record in journal)
    assert "Organelle 1" in said
    assert "spatial_measurements" in said
    assert "one connected mesh" in said


def test_a_punctate_run_leaves_no_note_on_the_journal(scratch_run_logs):
    """Silent when there is nothing to say: a run measuring what these
    families were designed for is not given a paragraph."""
    from spacr.measure import _record_organelle_caveats
    from spacr.runctx import read_run_log, run_context

    with run_context("measure", {}) as run:
        assert _record_organelle_caveats(
            {"organelle_mask_dim": 7, "organelle_type": "punctate",
             "spatial_measurements": True}, run) == []
        run_id = run.run_id

    assert read_run_log(run_id, contains="[organelle]") == []


def test_the_measure_run_records_them_where_the_tables_are_written():
    """The call sits inside `measure_crop`, beside the ledger the tables are
    accounted on -- not in a helper nothing reaches."""
    import inspect

    from spacr import measure

    source = inspect.getsource(measure.measure_crop)
    assert "_record_organelle_caveats(settings, run)" in source
