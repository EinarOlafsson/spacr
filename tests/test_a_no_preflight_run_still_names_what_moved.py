"""Item 288: what moved in a settings file is named even when nothing checks it.

``spacr-run <module> --settings old.csv`` migrates the file onto today's
names before it meets the defaults, and collects one notice per moved key
so the pre-flight report can name them. Two paths the existing tests in
``test_an_old_settings_file_keeps_its_values_through_spacr_run.py`` do not
take:

* ``--no-preflight`` skips the report the notices were collected for, so
  they are logged instead -- otherwise skipping the check would also skip
  telling the user their file used an old name.
* A key a fold removes but the doctor has no sentence for is still
  migrated, and is simply not named: the migration must not depend on the
  wording table being complete.
"""
from __future__ import annotations

import csv

from spacr import cli


def _file(tmp_path, rows):
    path = tmp_path / "settings.csv"
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows([("Key", "Value")] + list(rows))
    return str(path)


def test_a_no_preflight_run_logs_the_renamed_key(tmp_path, capsys,
                                                 monkeypatch):
    """The entry point is refused so nothing runs; the notice is what is
    under test."""
    started = []

    def refuse(module):
        started.append(module.key)
        raise cli.SettingsError("not importing the pipeline in a test")

    monkeypatch.setattr(cli, "import_entry", refuse)
    path = _file(tmp_path, [("src", str(tmp_path)), ("min_cell_count", "50")])
    rc = cli.main(["regression", "--settings", path, "--no-preflight"])
    out = capsys.readouterr()
    assert rc == cli.EXIT_USAGE
    assert started == ["regression"]
    assert "'min_cell_count' was renamed to 'min_cells_per_well'" in out.out
    assert "not importing the pipeline" in out.err


def test_a_fold_the_doctor_cannot_describe_still_migrates(tmp_path,
                                                          monkeypatch):
    import spacr.validate as validate

    monkeypatch.setattr(validate, "_check_retired_keys", lambda s: [])
    moved = []
    resolved = cli._under_todays_names(
        {"src": str(tmp_path), "min_cell_count": "50"}, moved)
    assert resolved["min_cells_per_well"] == "50"
    assert "min_cell_count" not in resolved
    assert moved == []
