"""Item 931: a saved settings file reruns with the defaults it was written under.

WHY. Item 931 changed the default of ``ops_compose_blend`` from the plain
average to feathering. A settings file or run journal written before that
has no ``ops_compose_blend`` key, so without a record of its era it would
quietly rerun with the new default and compose a different mosaic. Every
file spaCR writes now carries ``spacr_settings_version``; reading a file
with an older stamp, or none, gives each key whose default changed since,
and that the file does not set, the default it had then
(``spacr.settings.DEFAULT_CHANGES``). A key the file sets always wins.
"""
from __future__ import annotations

import csv
import json
import logging

import pytest

from spacr import run_journal as rj
from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                            SETTINGS_VERSION_KEY, _fold_older_defaults,
                            _settings_file_version, _stamp_settings_version)


@pytest.fixture(autouse=True)
def _isolated_runs(monkeypatch, tmp_path):
    """Journals go to the test's folder, never to ~/.spacr/runs."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(rj, "runs_root", lambda: runs)
    yield


def _ops():
    from spacr.cli import resolve_module

    return resolve_module("ops")


def _old_journal(tmp_path, settings):
    """A run folder as spaCR wrote it before item 931: no stamp."""
    run = tmp_path / "runs" / "2026-11-01_120000_abcd1234__ops"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"app_key": "ops"}),
                                       encoding="utf-8")
    (run / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    return run


def _replay(run, monkeypatch):
    """Replay ``run`` through ``spacr repro`` and return what the entry got."""
    from spacr import cli_repro

    seen = {}

    def entry(app_key):
        def run_ops(settings):
            seen.update(settings)
        return run_ops

    monkeypatch.setattr(cli_repro, "_resolve_pipeline", entry)
    assert cli_repro.main([str(run)]) == 0
    return seen


def _csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerows(rows)
    return str(path)


def test_the_table_is_one_row_per_change_and_names_old_defaults():
    """Every row is at or below the current version, and each key it lists
    really did change: its old default is not today's."""
    from spacr.ops_settings import OPS_DEFAULTS

    assert DEFAULT_CHANGES[931] == {"ops_compose_blend": "average"}
    assert all(version <= SETTINGS_SCHEMA_VERSION for version in DEFAULT_CHANGES)
    assert SETTINGS_SCHEMA_VERSION == max(DEFAULT_CHANGES)
    assert OPS_DEFAULTS["ops_compose_blend"] != DEFAULT_CHANGES[931]["ops_compose_blend"]


def test_a_pre_931_journal_replays_with_average(tmp_path, monkeypatch, caplog):
    """THE CASE THIS EXISTS FOR. A journal from before 931 recorded every
    OPS key of its time and no ops_compose_blend; its replay composes as
    that run did, and the replay's own journal records it."""
    run = _old_journal(tmp_path, {"plate": "P1", "ops_window_overlap": 96,
                                  "ops_raster_overlap": 213})
    with caplog.at_level(logging.WARNING, logger="spacr.settings"):
        seen = _replay(run, monkeypatch)
    assert seen["ops_compose_blend"] == "average"
    assert SETTINGS_VERSION_KEY not in seen
    assert "ops_compose_blend='average'" in caplog.text
    replays = [d for d in (tmp_path / "runs").iterdir() if d != run]
    assert len(replays) == 1
    recorded = json.loads((replays[0] / "settings.json").read_text("utf-8"))
    assert recorded[SETTINGS_VERSION_KEY] == SETTINGS_SCHEMA_VERSION
    assert recorded["ops_compose_blend"] == "average"


def test_a_new_run_is_stamped_and_composes_with_feather(tmp_path, monkeypatch):
    """A run resolved today records the current version and feather, in both
    journal files, and its replay and a rerun of its settings stay feather."""
    from spacr.cli import resolve_settings
    from spacr.ops_engine import _compose_blend

    settings = resolve_settings(_ops(), None)
    assert _compose_blend(settings) == "feather"
    with rj.open_run("ops", settings) as run:
        pass
    recorded = json.loads((run.dir / "settings.json").read_text("utf-8"))
    assert recorded[SETTINGS_VERSION_KEY] == SETTINGS_SCHEMA_VERSION
    assert recorded["ops_compose_blend"] == "feather"
    with open(run.dir / "settings.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[1] == [SETTINGS_VERSION_KEY, str(SETTINGS_SCHEMA_VERSION)]
    assert SETTINGS_VERSION_KEY not in run.settings

    stamped_without_key = dict(recorded)
    del stamped_without_key["ops_compose_blend"]
    path = tmp_path / "stamped.json"
    path.write_text(json.dumps(stamped_without_key), encoding="utf-8")
    rerun = resolve_settings(_ops(), str(path))
    assert _compose_blend(rerun) == "feather"
    assert SETTINGS_VERSION_KEY not in rerun
    assert _replay(run.dir, monkeypatch)["ops_compose_blend"] == "feather"


def test_an_explicit_key_always_wins(tmp_path):
    """A key the file sets is a decision, whatever the file's era."""
    from spacr.cli import resolve_settings

    unstamped = _csv(tmp_path / "unstamped.csv",
                     [("plate", "P1"), ("ops_compose_blend", "feather")])
    assert resolve_settings(_ops(), unstamped)["ops_compose_blend"] == "feather"
    stamped = _csv(tmp_path / "stamped.csv",
                   [(SETTINGS_VERSION_KEY, str(SETTINGS_SCHEMA_VERSION)),
                    ("plate", "P1"), ("ops_compose_blend", "average")])
    assert resolve_settings(_ops(), stamped)["ops_compose_blend"] == "average"
    assert resolve_settings(_ops(), stamped,
                            ["ops_compose_blend=feather"])["ops_compose_blend"] == "feather"


def test_an_unstamped_minimal_file_gets_average_and_the_notice(tmp_path, caplog):
    """A hand-written file has no stamp and is read as the oldest version:
    it gets the old default, and the log says which and how to opt in."""
    from spacr.cli import resolve_settings

    path = _csv(tmp_path / "minimal.csv", [("plate", "P1")])
    with caplog.at_level(logging.WARNING, logger="spacr.settings"):
        resolved = resolve_settings(_ops(), path)
    assert resolved["ops_compose_blend"] == "average"
    assert f"no {SETTINGS_VERSION_KEY}" in caplog.text
    assert "ops_compose_blend='average'" in caplog.text
    assert f"{SETTINGS_VERSION_KEY} = {SETTINGS_SCHEMA_VERSION}" in caplog.text


def test_an_older_default_is_added_only_to_a_pipeline_that_reads_it():
    """Another pipeline's settings never gain an OPS key."""
    measure_like = {"src": "/data"}
    assert _fold_older_defaults(measure_like, 0, known={"src"}) == {}
    assert measure_like == {"src": "/data"}
    ops_like = {"plate": "P1"}
    applied = _fold_older_defaults(ops_like, 0, known={"plate", "ops_compose_blend"})
    assert applied == {"ops_compose_blend": ("average", 931)}
    assert ops_like["ops_compose_blend"] == "average"


def test_a_current_stamp_takes_no_older_default():
    settings = {"plate": "P1"}
    assert _fold_older_defaults(settings, SETTINGS_SCHEMA_VERSION) == {}
    assert "ops_compose_blend" not in settings


def test_the_stamp_is_written_first_and_read_back_as_a_number(tmp_path):
    """Every writer leads with the stamp; every reader removes it, so no
    pipeline sees it, and text from a CSV reads as the same number."""
    from spacr.cli import load_settings_file, write_settings_file

    stamped = _stamp_settings_version({"plate": "P1", SETTINGS_VERSION_KEY: 1})
    assert list(stamped) == [SETTINGS_VERSION_KEY, "plate"]
    assert stamped[SETTINGS_VERSION_KEY] == SETTINGS_SCHEMA_VERSION
    written = write_settings_file({"plate": "P1"}, tmp_path / "s.csv")
    with open(written[0], newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle))[1] == [SETTINGS_VERSION_KEY,
                                               str(SETTINGS_SCHEMA_VERSION)]
    loaded = load_settings_file(written[0])
    assert _settings_file_version(loaded) == SETTINGS_SCHEMA_VERSION
    assert SETTINGS_VERSION_KEY not in loaded
    assert _settings_file_version({SETTINGS_VERSION_KEY: "930"}) == 930
    assert _settings_file_version({}) == 0
    assert _settings_file_version({SETTINGS_VERSION_KEY: "x"}) == 0
