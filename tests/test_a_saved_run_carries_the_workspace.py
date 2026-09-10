"""Instruction 180 — a run folder can carry what was OPEN around the run.

The run journal records what the pipeline was GIVEN. These tests are about
the other half: the databases attached, the montage generated, the volcano's
level and thresholds and selection — state that lives in widgets and, before
this, died with the process.

Two properties are load-bearing and each has its own test. NOTHING IS SKIPPED
SILENTLY: a file over the copy limit, a panel that raises, a database that
moved — each is named in the document or the report. And `off` writes NOTHING,
so a user who does not want this pays no bytes for it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacr import workspace


class Panel:
    """The contributor protocol, and nothing else."""

    def __init__(self, state=None):
        self.state = dict(state or {})
        self.applied = None
        self.answer = True

    def workspace_state(self):
        return dict(self.state)

    def apply_workspace_state(self, state):
        self.applied = dict(state)
        return self.answer


class LegacyVolcano:
    """The pair the regression panel already had before this existed."""

    def __init__(self, state):
        self._state = dict(state)
        self.applied = None

    def plot_state(self):
        return dict(self._state)

    def apply_plot_state(self, state):
        self.applied = dict(state)
        return True


@pytest.fixture(autouse=True)
def _no_leftover_providers():
    workspace.clear_providers()
    yield
    workspace.clear_providers()


# -- collecting -------------------------------------------------------------

def test_every_panel_contributes_its_own_slice_under_its_own_name():
    doc = workspace.collect({
        "volcano": Panel({"level": "gene", "colour_by": "condition"}),
        "montage": Panel({"coefficient": "GRA14"}),
    }, app_key="regression", saved="2026-08-19T00:00:00+00:00")

    assert doc["version"] == workspace.SCHEMA_VERSION
    assert doc["app_key"] == "regression"
    assert doc["sections"]["volcano"]["level"] == "gene"
    assert doc["sections"]["montage"]["coefficient"] == "GRA14"


def test_the_volcanos_existing_state_pair_is_taken_as_it_is():
    """One state model for the volcano, not a second one beside it."""
    panel = LegacyVolcano({"level": "grna", "threshold_multiplier": 2.5})
    doc = workspace.collect({"volcano": panel})
    assert doc["sections"]["volcano"]["threshold_multiplier"] == 2.5

    report = workspace.restore({"volcano": panel}, doc)
    assert report["restored"] == ["volcano"]
    assert panel.applied["level"] == "grna"


def test_a_provider_is_called_so_a_rebuilt_panel_is_the_one_asked():
    live = {"panel": Panel({"n": 1})}
    doc_before = workspace.collect({"p": lambda: live["panel"]})
    live["panel"] = Panel({"n": 2})
    doc_after = workspace.collect({"p": lambda: live["panel"]})
    assert doc_before["sections"]["p"]["n"] == 1
    assert doc_after["sections"]["p"]["n"] == 2


def test_one_panel_that_raises_does_not_cost_the_others_and_is_named():
    def angry():
        raise RuntimeError("Internal C++ object already deleted")

    doc = workspace.collect({"good": Panel({"a": 1}), "bad": angry})
    assert doc["sections"]["good"] == {"a": 1}
    assert "bad" not in doc["sections"]
    assert any(p["section"] == "bad" and "RuntimeError" in p["why"]
               for p in doc["problems"])


def test_tuples_and_paths_survive_the_trip_through_json(tmp_path):
    panel = Panel({"baseline": ("control", "gene"), "where": tmp_path})
    doc = workspace.collect({"p": panel})
    round_tripped = json.loads(json.dumps(doc))
    assert round_tripped["sections"]["p"]["baseline"] == ["control", "gene"]
    assert round_tripped["sections"]["p"]["where"] == str(tmp_path)


def test_a_state_that_contains_itself_costs_a_truncated_section_not_the_process():
    state = {"name": "loop"}
    state["me"] = state
    doc = workspace.collect({"p": Panel(state)})
    assert doc["sections"]["p"]["name"] == "loop"


# -- the file inventory -----------------------------------------------------

def test_a_path_a_panel_names_is_recorded_without_anyone_declaring_it(tmp_path):
    """The walk covers the key a panel gains and nobody remembers to declare."""
    db = tmp_path / "measurements.db"
    db.write_bytes(b"sqlite-ish")
    doc = workspace.collect({"databases": Panel({"attached": [str(db)]})})

    record = next(f for f in doc["files"] if f["path"] == str(db))
    assert record["exists"] is True
    assert record["size"] == len(b"sqlite-ish")
    assert len(record["sha256"]) == 64


def test_a_gene_name_is_not_asked_of_the_filesystem():
    doc = workspace.collect({"p": Panel({"genes": ["GRA14", "ROP18", "MYR1"]})})
    assert doc["files"] == []


def test_a_path_that_does_not_exist_is_not_recorded_as_a_file(tmp_path):
    doc = workspace.collect({"p": Panel({"db": str(tmp_path / "gone.db")})})
    assert doc["files"] == []


# -- reference vs copy ------------------------------------------------------

def test_off_writes_nothing_at_all(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    doc = workspace.collect({"p": Panel({"a": 1})})
    assert workspace.save(run, doc, mode="off") is None
    assert list(run.iterdir()) == []


def test_reference_records_the_database_but_does_not_copy_it(tmp_path):
    db = tmp_path / "measurements.db"
    db.write_bytes(b"x" * 4096)
    run = tmp_path / "run"
    doc = workspace.collect({"databases": Panel({"attached": [str(db)]})})
    workspace.save(run, doc, mode="reference")

    written = json.loads((run / workspace.DOC_NAME).read_text())
    assert written["mode"] == "reference"
    assert written["files"][0].get("copied") is None
    assert not (run / workspace.FILES_DIR).exists()


def test_copy_brings_the_bytes_in_and_the_copy_is_the_same_file(tmp_path):
    db = tmp_path / "measurements.db"
    db.write_bytes(b"the actual bytes")
    run = tmp_path / "run"
    doc = workspace.collect({"databases": Panel({"attached": [str(db)]})})
    workspace.save(run, doc, mode="copy")

    written = json.loads((run / workspace.DOC_NAME).read_text())
    copied = run / written["files"][0]["copied"]
    assert copied.read_bytes() == b"the actual bytes"


def test_a_file_over_the_limit_is_named_with_its_size_never_dropped(tmp_path):
    big = tmp_path / "screen.db"
    big.write_bytes(b"0" * (3 * 1024 * 1024))
    run = tmp_path / "run"
    doc = workspace.collect({"databases": Panel({"attached": [str(big)]})})
    workspace.save(run, doc, mode="copy", copy_limit_mb=1)

    record = json.loads((run / workspace.DOC_NAME).read_text())["files"][0]
    assert record["copied"] is None
    assert "3.0 MB" in record["skipped"] and "1 MB per-file limit" in record["skipped"]


def test_a_figure_the_session_made_is_carried_even_in_reference_mode(tmp_path):
    """It exists nowhere else — a reference to it would be a reference to nothing."""
    figure = tmp_path / "volcano.png"
    figure.write_bytes(b"PNG")
    run = tmp_path / "run"
    doc = workspace.collect({"figures": Panel({
        workspace.CARRY_KEY: [{"role": "figure", "path": str(figure)}],
    })})
    workspace.save(run, doc, mode="reference")

    record = json.loads((run / workspace.DOC_NAME).read_text())["files"][0]
    assert record["carry"] is True
    assert (run / record["copied"]).read_bytes() == b"PNG"


def test_a_carried_figure_is_carried_whatever_the_limit_says(tmp_path):
    figure = tmp_path / "montage.png"
    figure.write_bytes(b"P" * (2 * 1024 * 1024))
    run = tmp_path / "run"
    doc = workspace.collect({"figures": Panel({
        workspace.CARRY_KEY: [{"role": "figure", "path": str(figure)}],
    })})
    workspace.save(run, doc, mode="copy", copy_limit_mb=1)

    record = json.loads((run / workspace.DOC_NAME).read_text())["files"][0]
    assert record["copied"] and "skipped" not in record


def test_the_carry_list_is_not_handed_back_to_the_panel(tmp_path):
    figure = tmp_path / "f.png"
    figure.write_bytes(b"PNG")
    panel = Panel({"level": "gene",
                   workspace.CARRY_KEY: [{"role": "figure", "path": str(figure)}]})
    doc = workspace.collect({"figures": panel})
    workspace.restore({"figures": panel}, doc)
    assert panel.applied == {"level": "gene"}


# -- modes ------------------------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    (None, "reference"), ("off", "off"), ("copy", "copy"),
    ("REFERENCE", "reference"), (True, "reference"), (False, "off"),
    ("none", "off"), ("full", "copy"), ("nonsense", "reference"),
])
def test_a_typo_in_the_setting_costs_the_default_never_the_run(given, expected):
    assert workspace.resolve_mode(given) == expected


def test_the_mode_and_the_limit_come_from_the_settings_dict():
    settings = {"save_workspace": "copy", "workspace_copy_limit_mb": 64}
    assert workspace.mode_from_settings(settings) == "copy"
    assert workspace.copy_limit_from_settings(settings) == 64.0
    assert workspace.copy_limit_from_settings({"workspace_copy_limit_mb": "junk"}) == 512.0


# -- getting it back --------------------------------------------------------

def test_a_round_trip_puts_every_panel_back_where_it_was(tmp_path):
    run = tmp_path / "run"
    volcano = Panel({"level": "gene", "threshold_multiplier": 2.0})
    montage = Panel({"coefficient": "GRA14"})
    workspace.save(run, workspace.collect({"volcano": volcano, "montage": montage}))

    fresh_volcano, fresh_montage = Panel(), Panel()
    doc = workspace.load(run)
    report = workspace.restore({"volcano": fresh_volcano, "montage": fresh_montage},
                               doc, run_dir=run)

    assert report["restored"] == ["montage", "volcano"]
    assert fresh_volcano.applied["threshold_multiplier"] == 2.0
    assert fresh_montage.applied["coefficient"] == "GRA14"


def test_a_section_with_nothing_on_screen_to_own_it_is_named(tmp_path):
    doc = workspace.collect({"volcano": Panel({"level": "gene"})})
    report = workspace.restore({}, doc)
    assert report["restored"] == []
    assert report["skipped"] == [
        {"section": "volcano", "why": "nothing on screen owns it"}]


def test_a_panel_that_declines_is_reported_not_counted_as_restored():
    """A panel with no table yet cannot take a plot state, and says so."""
    panel = Panel()
    panel.answer = False
    doc = workspace.collect({"volcano": Panel({"level": "gene"})})
    report = workspace.restore({"volcano": panel}, doc)
    assert report["restored"] == []
    assert report["skipped"][0]["why"] == "the panel declined it"


def test_a_moved_database_is_a_named_failure_not_a_traceback(tmp_path):
    db = tmp_path / "measurements.db"
    db.write_bytes(b"data")
    run = tmp_path / "run"
    workspace.save(run, workspace.collect({"databases": Panel({"db": str(db)})}))
    db.unlink()

    report = workspace.restore({"databases": Panel()}, workspace.load(run), run_dir=run)
    missing = [f for f in report["files"] if f["state"] == "missing"]
    assert [f["path"] for f in missing] == [str(db)]
    assert "missing" in workspace.report_text(report)


def test_a_database_edited_since_the_run_is_called_changed_not_present(tmp_path):
    db = tmp_path / "measurements.db"
    db.write_bytes(b"before")
    run = tmp_path / "run"
    workspace.save(run, workspace.collect({"databases": Panel({"db": str(db)})}))
    db.write_bytes(b"after!")

    states = workspace.check_files(workspace.load(run), run_dir=run)
    assert states[0]["state"] == "changed"


def test_a_copied_file_is_found_in_the_bundle_when_the_original_is_gone(tmp_path):
    db = tmp_path / "measurements.db"
    db.write_bytes(b"data")
    run = tmp_path / "run"
    workspace.save(run, workspace.collect({"databases": Panel({"db": str(db)})}),
                   mode="copy")
    db.unlink()

    states = workspace.check_files(workspace.load(run), run_dir=run)
    assert states[0]["state"] == "carried"
    assert Path(states[0]["path"]).read_bytes() == b"data"


def test_load_takes_the_folder_or_the_document_itself(tmp_path):
    run = tmp_path / "run"
    workspace.save(run, workspace.collect({"p": Panel({"a": 1})}))
    assert workspace.load(run)["sections"]["p"] == {"a": 1}
    assert workspace.load(run / workspace.DOC_NAME)["sections"]["p"] == {"a": 1}
    assert workspace.load(tmp_path / "no-such-run") is None
    assert workspace.has_workspace(run) and not workspace.has_workspace(tmp_path)


# -- the registry -----------------------------------------------------------

def test_the_registry_is_how_gui_state_reaches_a_journal_that_cannot_import_qt():
    panel = Panel({"level": "gene"})
    workspace.register("volcano", lambda: panel)
    assert workspace.collect(workspace.providers())["sections"]["volcano"]["level"] == "gene"
    assert workspace.unregister("volcano") is True
    assert workspace.unregister("volcano") is False
    assert workspace.collect(workspace.providers())["sections"] == {}


def test_a_contributor_needs_a_name_and_a_provider_must_be_callable():
    with pytest.raises(ValueError):
        workspace.register("", lambda: None)
    with pytest.raises(TypeError):
        workspace.register("x", "not callable")


# -- saying what is in one --------------------------------------------------

def test_the_inventory_names_the_sections_the_files_and_what_was_skipped(tmp_path):
    big = tmp_path / "screen.db"
    big.write_bytes(b"0" * (2 * 1024 * 1024))
    run = tmp_path / "run"
    doc = workspace.collect({"databases": Panel({"db": str(big)}),
                             "volcano": Panel({"level": "gene"})},
                            app_key="regression")
    workspace.save(run, doc, mode="copy", copy_limit_mb=1)

    text = workspace.inventory_text(workspace.load(run), run_dir=run)
    assert "regression" in text
    assert "databases" in text and "volcano" in text
    assert "screen.db" in text and "skipped" in text
    assert "2.0 MB" in text


# -- the run journal writes it, and only when there is something to write ----

def test_a_command_line_run_that_opened_nothing_gets_no_empty_document(tmp_path):
    """`spacr mask` in a terminal gets the run folder it always got."""
    run = tmp_path / "run"
    run.mkdir()
    assert workspace.save_for_run(run, {}, app_key="mask") is None
    assert list(run.iterdir()) == []


def test_the_application_with_panels_open_gets_the_bundle(tmp_path):
    run = tmp_path / "run"
    workspace.register("volcano", lambda: Panel({"level": "gene"}))
    path = workspace.save_for_run(run, {}, app_key="regression")
    assert path == run / workspace.DOC_NAME
    assert workspace.load(run)["sections"]["volcano"]["level"] == "gene"


def test_off_in_the_settings_beats_panels_being_open(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    workspace.register("volcano", lambda: Panel({"level": "gene"}))
    assert workspace.save_for_run(run, {"save_workspace": "off"}) is None
    assert list(run.iterdir()) == []


def test_the_application_pushes_its_preference_down_to_the_journal(tmp_path):
    """QSettings cannot be read from a pipeline; the app hands the answer over."""
    assert workspace.default_mode() == "reference"
    workspace.set_default_mode("copy", copy_limit_mb=8)
    assert workspace.mode_from_settings({}) == "copy"
    assert workspace.copy_limit_from_settings({}) == 8.0

    # A scripted run outranks the application's preference.
    assert workspace.mode_from_settings({"save_workspace": "off"}) == "off"
    assert workspace.copy_limit_from_settings({"workspace_copy_limit_mb": 2}) == 2.0

    workspace.set_default_mode("off")
    run = tmp_path / "run"
    run.mkdir()
    workspace.register("volcano", lambda: Panel({"level": "gene"}))
    assert workspace.save_for_run(run, {}) is None


def test_a_run_writes_its_workspace_when_it_closes(tmp_path, monkeypatch):
    from spacr import run_journal

    monkeypatch.setattr(run_journal, "runs_root", lambda: tmp_path / "runs")
    workspace.register("volcano", lambda: Panel({"level": "gene",
                                                 "threshold_multiplier": 2.0}))
    with run_journal.open_run("regression", {"src": str(tmp_path)}) as run:
        run.set_status("success")

    doc = workspace.load(run.dir)
    assert doc["app_key"] == "regression"
    assert doc["sections"]["volcano"]["threshold_multiplier"] == 2.0


def test_a_panel_that_explodes_does_not_fail_a_run_that_produced_results(tmp_path, monkeypatch):
    from spacr import run_journal

    monkeypatch.setattr(run_journal, "runs_root", lambda: tmp_path / "runs")

    def explode():
        raise RuntimeError("Internal C++ object already deleted")

    workspace.register("volcano", explode)
    with run_journal.open_run("regression", {}) as run:
        run.set_status("success")

    assert run.status == "success"
    assert workspace.load(run.dir) is None      # no sections, no files, no file
