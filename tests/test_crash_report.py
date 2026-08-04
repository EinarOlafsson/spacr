"""S6 — the crash reporter has to survive the crash it is reporting.

Every test here is about one of three properties, because those are the three
ways a crash reporter fails in the field:

**It must produce a file, always.** A collector that raises — a locked
database, an unreadable log, an optional dependency that fails to import, a bug
in a collector — may not stop the bundle, because the bundle *is* the evidence
and there is no second chance to gather it. ``TestNothingStopsTheReport`` breaks
each collector in turn and checks that a file still lands on disk with
everything else in it.

**A missing section must be visible.** The rule above is only defensible if the
failure is recorded, so every broken collector is checked to leave its
exception in ``manifest.json``. A section that vanished silently would be the
same class of bug as the ``except Exception: pass`` sites this run is auditing —
the difference is entirely in whether the output says so.

**It must compose, not re-collect.** ``spacr.doctor`` already runs 17 checks and
``spacr.runctx`` already writes a per-run JSONL keyed by run id. A second
implementation of either would drift from the one users actually run, so the
tests assert that the report's doctor rows are the doctor's rows and that the
run log in the bundle is the file ``runctx`` wrote, byte for byte.

The log directory is redirected with ``SPACR_LOG_DIR`` for every test, which is
the supported way (:func:`spacr.logging_util.log_dir`) and keeps the developer's
real ``~/.spacr/logs`` out of it.
"""
from __future__ import annotations

import json
import os
import sys
import zipfile

import pytest

from spacr import crashreport as cr


@pytest.fixture()
def logs(tmp_path, monkeypatch):
    """Point every spaCR log path at a private directory for this test."""
    root = tmp_path / "logs"
    root.mkdir()
    monkeypatch.setenv("SPACR_LOG_DIR", str(root))
    return root


def _bundle(path):
    """Return ``{name: text}`` for every file in a written bundle."""
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name).decode("utf-8")
                for name in archive.namelist()}


# ---------------------------------------------------------------------------
# The shape of the thing
# ---------------------------------------------------------------------------

def test_a_report_is_one_file_holding_the_four_things_it_promises(logs,
                                                                 tmp_path):
    """Log, settings, versions and the run — the whole point of S6."""
    (logs / "spacr.log").write_text("INFO an ordinary line\n", encoding="utf-8")
    settings = tmp_path / "measure.json"
    settings.write_text(json.dumps({"src": "/data/plate1", "cell_mask_dim": 1}),
                        encoding="utf-8")

    written = cr.write_crash_report(tmp_path / "report.zip",
                                    settings=str(settings), app="measure")

    files = _bundle(written)
    assert "summary.txt" in files and "manifest.json" in files
    assert "versions.json" in files                      # versions
    assert "settings.json" in files                      # settings
    assert "spacr.log" in files                          # log
    assert "doctor.txt" in files and "doctor.json" in files
    assert json.loads(files["settings.json"])["src"] == "/data/plate1"
    assert "an ordinary line" in files["spacr.log"]
    # The summary is first so a reader knows where to start.
    with zipfile.ZipFile(written) as archive:
        assert archive.namelist()[0] == "summary.txt"


def test_the_doctor_rows_in_the_bundle_are_the_doctors_own(logs, tmp_path):
    """Composed, not re-collected.

    If this module grew its own version of "is the GPU usable" it would drift
    from the one ``spacr-doctor`` prints, and a user would be told two
    different things by two commands about one machine.
    """
    from spacr import doctor

    report = cr.collect(checkout=str(tmp_path), probe_gpu=False)

    rows = json.loads(report.sections["doctor.json"])
    assert len(rows) == len(doctor.run_checks(
        doctor.Context(checkout=tmp_path, probe_gpu=False)))
    assert {row["check"] for row in rows} == {
        r.check for r in doctor.run_checks(
            doctor.Context(checkout=tmp_path, probe_gpu=False))}
    assert report.manifest["doctor_summary"]
    # ...and every non-PASS row still carries the fix the doctor attaches.
    for row in rows:
        if row["status"] not in ("PASS", "SKIP"):
            assert row["fix"], row


def test_the_run_log_in_the_bundle_is_the_file_runctx_wrote(logs, tmp_path):
    """Byte for byte, not a summary of it.

    A crash report that paraphrased the log would lose exactly the line nobody
    thought to keep.
    """
    import logging

    from spacr import runctx

    with runctx.run_context("measure", {"random_seed": 1}) as run:
        logging.getLogger("spacr.test").error("the failing line")
        run_id = run.run_id

    report = cr.collect(run_id, checkout=str(tmp_path))

    on_disk = open(runctx.run_log_path(run_id), encoding="utf-8").read()
    assert report.sections[f"run-{run_id}.jsonl"] == on_disk
    assert "the failing line" in report.sections["run-problems.txt"]
    assert report.run_id == run_id


def test_the_last_run_is_found_without_being_named(logs):
    """``find_last_run_id`` is what makes the command usable after a crash.

    A user who has just lost a pipeline does not know the run id, and asking
    for one is how a support tool goes unused.
    """
    import logging

    from spacr import runctx

    first = second = ""
    with runctx.run_context("mask", {}) as run:
        logging.getLogger("spacr.test").warning("first")
        first = run.run_id
    with runctx.run_context("measure", {}) as run:
        logging.getLogger("spacr.test").warning("second")
        second = run.run_id

    assert first != second
    assert cr.find_last_run_id() == second


def test_inside_a_run_the_active_id_wins_over_the_newest_file(logs):
    """The crash being reported is *this* run, not the one before it."""
    import logging

    from spacr import runctx

    with runctx.run_context("mask", {}) as first:
        logging.getLogger("spacr.test").warning("older")
        older = first.run_id
    with runctx.run_context("measure", {}) as active:
        logging.getLogger("spacr.test").warning("current")
        assert cr.find_last_run_id() == active.run_id != older


# ---------------------------------------------------------------------------
# Nothing stops the report
# ---------------------------------------------------------------------------

class TestNothingStopsTheReport:
    """A reporter that dies while reporting destroys the only evidence.

    Each test breaks one collector and asserts two things: the bundle is still
    written with every other section in it, and the failure is *named* in the
    manifest. The second half is what separates this from a swallowed error.
    """

    def test_a_collector_that_raises_becomes_a_manifest_entry(self, logs,
                                                              tmp_path,
                                                              monkeypatch):
        """Found a real hole in this module: ``run_checks`` used to be called
        beside the guard instead of inside it, so a ``spacr.doctor`` that
        failed to import took the whole bundle with it — including the log and
        the versions, which are what would have explained the import failure.
        """
        def explode(*args, **kwargs):
            raise RuntimeError("this collector is broken")

        monkeypatch.setattr(cr, "_package_versions", explode)
        monkeypatch.setattr(cr, "_redacted_environment", explode)

        report = cr.collect(checkout=str(tmp_path))

        assert set(report.problems) == {"versions.json", "environment.json"}
        entry = report.manifest["sections"]["versions.json"]
        assert entry["status"] == "failed"
        assert "this collector is broken" in entry["error"]
        assert "Traceback" in entry["traceback"]
        # ...and everything else is still there.
        assert "doctor.txt" in report.sections

    def test_a_doctor_that_will_not_even_import_is_a_manifest_entry(
            self, logs, tmp_path, monkeypatch):
        """The case that was actually broken.

        ``doctor.run_checks`` already turns a check that raises into an
        ``ERROR`` row, but importing the module and building its Context are
        outside that guarantee — and a broken environment is exactly when a
        crash report is being written.
        """
        import spacr

        # Both halves are needed: ``from . import doctor`` takes the attribute
        # off the already-imported package without consulting sys.modules, so
        # poisoning only sys.modules would leave the import working and this
        # test asserting nothing.
        monkeypatch.delattr(spacr, "doctor", raising=False)
        monkeypatch.setitem(sys.modules, "spacr.doctor", None)

        report = cr.collect(checkout=str(tmp_path))

        assert "doctor.txt" in report.problems
        assert "spacr.doctor" in \
            report.manifest["sections"]["doctor.txt"]["error"]
        # The json half says why it is absent rather than looking empty.
        assert "doctor.json" in report.problems
        assert "doctor.txt" in \
            report.manifest["sections"]["doctor.json"]["error"]
        # ...and the sections that do not depend on it survived.
        assert "versions.json" in report.sections

    def test_an_unreadable_log_does_not_lose_the_rest(self, logs, tmp_path,
                                                      monkeypatch):
        """The log is the most likely thing to be unreadable and the least
        likely to be the only thing worth having."""
        def unreadable(*args, **kwargs):
            raise PermissionError("spacr.log is not yours")

        monkeypatch.setattr(cr, "_tail", unreadable)
        (logs / "spacr.log").write_text("x\n", encoding="utf-8")

        written = cr.write_crash_report(tmp_path / "r.zip",
                                        checkout=str(tmp_path))

        files = _bundle(written)
        assert "versions.json" in files and "doctor.txt" in files
        manifest = json.loads(files["manifest.json"])
        assert manifest["sections"]["spacr.log"]["status"] == "failed"
        assert "not yours" in manifest["sections"]["spacr.log"]["error"]
        assert "FAILED to gather" in files["summary.txt"]

    def test_a_memory_error_in_a_collector_is_still_only_a_missing_section(
            self, logs, tmp_path, monkeypatch):
        """``BaseException``, not ``Exception``, and the reason is here.

        A ``MemoryError`` or a ``RecursionError`` out of one collector would
        otherwise take the whole bundle with it — including every section
        already gathered, which is the evidence that would have explained it.
        """
        def exhausted():
            raise MemoryError("no")

        report = cr.CrashReport(created_utc="now")
        cr._collect(report, "big.json", exhausted)

        assert report.manifest["sections"]["big.json"]["status"] == "failed"
        assert "MemoryError" in report.manifest["sections"]["big.json"]["error"]

    def test_ctrl_c_is_not_recorded_as_a_missing_section(self):
        """A user pressing Ctrl-C means stop, not "note that I did".

        The one exception to the rule above, and it has to be an exception:
        swallowing ``KeyboardInterrupt`` would make the reporter unkillable.
        """
        def interrupted():
            raise KeyboardInterrupt

        report = cr.CrashReport(created_utc="now")
        with pytest.raises(KeyboardInterrupt):
            cr._collect(report, "x.json", interrupted)

    def test_a_section_with_nothing_in_it_is_not_reported_as_a_failure(
            self, logs, tmp_path):
        """"There was no settings file" and "the settings could not be read"
        are different answers, and a reader who cannot tell them apart learns
        to ignore both."""
        report = cr.collect(checkout=str(tmp_path))

        assert report.problems == []
        assert "settings.json" in report.omitted
        assert report.manifest["sections"]["settings.json"]["status"] == "empty"

    def test_report_exception_never_replaces_the_users_exception(self, logs,
                                                                 tmp_path,
                                                                 monkeypatch,
                                                                 capsys):
        """Called from an ``except`` block that is about to re-raise.

        Raising here would replace the traceback the user needs with one about
        the reporting of it.
        """
        monkeypatch.setattr(cr, "write_crash_report", lambda *a, **k:
                            (_ for _ in ()).throw(OSError("disk full")))

        assert cr.report_exception(ValueError("the real bug")) == ""
        assert "could not write a crash report" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# What it must not leak, and what it must not be
# ---------------------------------------------------------------------------

def test_a_credential_shaped_variable_is_redacted_and_said_to_be(logs,
                                                                 tmp_path,
                                                                 monkeypatch):
    """A bundle goes into a public issue tracker.

    Redaction is by variable *name*, not by value: a heuristic over values
    would redact a plate path containing the word "key" and miss a token whose
    name nobody thought of. The names withheld are listed, so the user can see
    what was kept back and the maintainer can see that something was.
    """
    monkeypatch.setenv("SPACR_API_TOKEN", "hunter2-do-not-share")
    monkeypatch.setenv("SPACR_SRC", "/data/plate1")

    report = cr.collect(checkout=str(tmp_path))

    environment = json.loads(report.sections["environment.json"])
    assert environment["SPACR_API_TOKEN"] == "<redacted>"
    assert environment["SPACR_SRC"] == "/data/plate1"
    assert "SPACR_API_TOKEN" in report.manifest["redacted_environment"]
    assert "hunter2-do-not-share" not in report.sections["environment.json"]


def test_an_unrelated_environment_variable_is_dropped_rather_than_included(
        logs, tmp_path, monkeypatch):
    """A full ``os.environ`` is mostly the user's shell, and is where an
    accidental disclosure comes from. Only the prefixes that explain "works
    here, not there" are reported."""
    monkeypatch.setenv("MY_COMPANY_INTERNAL_HOST", "prod-db-01.internal")

    report = cr.collect(checkout=str(tmp_path))

    assert "MY_COMPANY_INTERNAL_HOST" not in report.sections["environment.json"]


def test_a_huge_log_is_tailed_and_the_bundle_says_how_much_went(logs,
                                                                tmp_path,
                                                                monkeypatch):
    """An attachment nobody can upload is not evidence.

    The *tail* is kept, because the cause of a crash is at the end; the byte
    counts are recorded, because a truncated section that looked short would
    be read as "the run logged almost nothing".
    """
    monkeypatch.setattr(cr, "MAX_LOG_BYTES", 4096)
    line = "x" * 99 + "\n"
    (logs / "spacr.log").write_text(line * 500, encoding="utf-8")   # 50 kB

    report = cr.collect(checkout=str(tmp_path))

    kept = report.sections["spacr.log"]
    assert len(kept.encode()) <= 4096
    assert report.manifest["main_log"]["truncated"] is True
    assert report.manifest["main_log"]["total_bytes"] == 50000
    assert report.manifest["main_log"]["dropped_bytes"] > 45000
    # The partial first line is dropped rather than shown as a whole one.
    assert kept.splitlines()[0] == "x" * 99


def test_the_traceback_of_the_reported_exception_goes_in_verbatim(logs,
                                                                  tmp_path):
    """The one thing a maintainer looks at first."""
    try:
        raise ValueError("measure_crop could not read the merged array")
    except ValueError as exc:
        report = cr.collect(exception=exc, checkout=str(tmp_path),
                            note="ran measure on plate1 after a resume")

    assert "measure_crop could not read the merged array" in \
        report.sections["traceback.txt"]
    assert "Traceback (most recent call last)" in report.sections["traceback.txt"]
    assert "after a resume" in report.sections["note.txt"]


def test_the_run_status_of_the_project_database_is_included(logs, tmp_path):
    """Whether the last run *finished* is a different question from whether
    something crashed just now — and it is the one that matters when a user
    reports numbers rather than a traceback."""
    from spacr.errors import RunLedger

    db = tmp_path / "measurements.db"
    ledger = RunLedger("measure_crop")
    with ledger.item("plate1_A01_1", stage="measure"):
        pass
    try:
        with ledger.item("plate1_A01_2", stage="measure"):
            raise RuntimeError("field 2 died")
    except RuntimeError:
        pass
    ledger.finalize(artifact=str(db))

    report = cr.collect(db=str(db), checkout=str(tmp_path))

    if "run-status.json" in report.sections:
        records = json.loads(report.sections["run-status.json"])
        assert any(record.get("n_failed") for record in records), records
    else:                                  # pragma: no cover - shape guard
        pytest.fail("the run status was not collected: "
                    f"{report.manifest['sections']['run-status.json']}")


def test_the_module_does_not_import_torch_to_be_imported(logs):
    """``import spacr.crashreport`` must be cheap.

    It is imported from an excepthook, in a process that may be out of memory
    and is certainly in trouble. Pulling in torch at import time would make the
    reporter the second thing to fail.
    """
    import subprocess

    probe = ("import sys; import spacr.crashreport; "
             "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True,
                         env={**os.environ, "SPACR_LOG_DIR": str(logs)})
    assert out.stdout.strip() == "False", out.stdout


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------

def test_the_command_writes_a_bundle_and_says_where(logs, tmp_path, capsys):
    """``python -m spacr.crashreport``, the whole path a user takes."""
    target = tmp_path / "bug.zip"

    assert cr.main(["-o", str(target), "--checkout", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert str(target) in printed
    assert "summary.txt" in printed          # what to open first
    assert target.is_file()
    assert "versions.json" in _bundle(target)


def test_a_directory_destination_gets_a_named_file_inside_it(logs, tmp_path):
    """So ``-o .`` does the obvious thing rather than writing a file called
    ``.``."""
    written = cr.write_crash_report(tmp_path, checkout=str(tmp_path))

    assert os.path.dirname(written) == str(tmp_path)
    assert os.path.basename(written).startswith("spacr-crashreport-")
    assert written.endswith(".zip")


def test_with_no_destination_it_lands_in_the_log_directory(logs, tmp_path):
    """The one directory spaCR knows exists and is writable on this machine."""
    written = cr.write_crash_report(checkout=str(tmp_path))

    assert os.path.dirname(written) == str(logs.resolve())


def test_the_command_returns_one_rather_than_raising_when_it_cannot_write(
        logs, tmp_path, capsys):
    """A crash reporter that traceback'd on a read-only output directory would
    be reporting its own crash instead of the user's."""
    unwritable = tmp_path / "nope" / "deep"
    unwritable.parent.mkdir()
    unwritable.parent.chmod(0o500)
    try:
        assert cr.main(["-o", str(unwritable / "r.zip"),
                        "--checkout", str(tmp_path)]) == 1
        assert "could not write" in capsys.readouterr().err
    finally:
        unwritable.parent.chmod(0o700)


def test_the_excepthook_reports_and_then_defers_to_the_previous_one(logs,
                                                                    tmp_path,
                                                                    monkeypatch,
                                                                    capsys):
    """Chained, not replaced: the user still sees the traceback they expect,
    with one line after it saying where the bundle is."""
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: seen.append(a))

    hook = cr.install_excepthook(tmp_path)
    try:
        error = ValueError("boom")
        hook(ValueError, error, None)
    finally:
        sys.excepthook = sys.__excepthook__

    assert len(seen) == 1 and seen[0][1] is error
    assert "wrote a crash report" in capsys.readouterr().err
    assert list(tmp_path.glob("spacr-crashreport-*.zip"))


def test_the_excepthook_stays_out_of_the_way_of_ctrl_c(logs, tmp_path,
                                                       monkeypatch):
    """Ctrl-C is not a crash, and writing a bundle for one is noise."""
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: seen.append(a))

    hook = cr.install_excepthook(tmp_path)
    try:
        hook(KeyboardInterrupt, KeyboardInterrupt(), None)
    finally:
        sys.excepthook = sys.__excepthook__

    assert len(seen) == 1
    assert not list(tmp_path.glob("spacr-crashreport-*.zip"))
