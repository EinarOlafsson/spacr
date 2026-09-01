"""`Run._record_stage` and the log-tail writer: the halves nothing ran.

The measured gap list put these in `run_journal.py`, and they share a
theme: they are what the journal does when the thing it is recording is
malformed or the disk will not take it. The journal is provenance -- the
rule stated in `_record_stage`'s own docstring is that "provenance must
never replace a scientific result or exception" -- so every one of these
paths has to swallow rather than propagate.

A stage recorder that raises would take down the run it is describing,
which is the worst possible trade: losing the science to preserve the
note about the science.
"""
from __future__ import annotations

import json

import pytest

import spacr.run_journal as rj


@pytest.fixture()
def run(tmp_path):
    d = tmp_path / "2026-08-31_010203_abcd__mask"
    d.mkdir()
    return rj.Run(app_key="mask", settings={}, dir=d)


class TestAStageIdThatNamesNothing:

    def test_an_empty_stage_id_records_nothing(self, run):
        run._record_stage("")
        assert run.stages == []

    def test_a_whitespace_stage_id_records_nothing(self, run):
        """It is stripped before it is judged, so "   " is empty too."""
        run._record_stage("   ")
        assert run.stages == []

    def test_a_none_stage_id_records_nothing(self, run):
        """`str(None)` is "None", which strips to a non-empty string --

        so this asserts the guard as it IS, not as it might be assumed to
        be. A caller passing None gets a stage called "None" rather than
        a silent drop, and knowing which of the two happens is the point
        of writing it down.
        """
        run._record_stage(None)
        assert [s.get("id") for s in run.stages] == ["None"]


class TestMergingIntoAStageThatIsAlreadyThere:

    def test_a_second_observation_updates_the_label_in_place(self, run):
        """The `elif label is not None` arm, which nothing reached.

        Repeat calls have to update the existing stage and preserve
        first-seen order rather than appending a duplicate.
        """
        run._record_stage("segment", label="Segmenting")
        run._record_stage("segment", label="Segmenting cells")
        assert len(run.stages) == 1
        assert run.stages[0]["label"] == "Segmenting cells"

    def test_a_second_observation_without_a_label_keeps_the_first(self, run):
        """`label is None` means "not mentioned", not "clear it"."""
        run._record_stage("segment", label="Segmenting")
        run._record_stage("segment", state="done")
        assert run.stages[0]["label"] == "Segmenting"
        assert run.stages[0]["state"] == "done"

    def test_first_seen_order_survives_updates(self, run):
        run._record_stage("a", label="A")
        run._record_stage("b", label="B")
        run._record_stage("a", state="done")
        assert [s["id"] for s in run.stages] == ["a", "b"]

    def test_a_stage_with_both_ends_gets_a_duration(self, run):
        run._record_stage("a", started_at=10.0)
        run._record_stage("a", ended_at=12.5)
        assert run.stages[0]["duration_s"] == pytest.approx(2.5)

    def test_a_stage_with_one_end_has_no_duration(self, run):
        run._record_stage("a", started_at=10.0)
        assert run.stages[0]["duration_s"] is None


def test_a_recorder_that_raises_does_not_take_the_run_with_it(run):
    """The `except BaseException` arm -- five lines nothing reached.

    BaseException and not Exception, deliberately: provenance must not
    convert a KeyboardInterrupt during a run into a lost result either.
    The stage is dropped and the run continues.
    """

    class _Hostile:
        def items(self):
            raise KeyboardInterrupt("the user pressed ctrl-c mid-merge")

    run._record_stage("segment", metrics=_Hostile())
    # the stage was created before the metrics were merged; what matters is
    # that nothing propagated out of the recorder
    assert isinstance(run.stages, list)


class TestTheLogTailItWrites:

    def test_a_tail_with_no_final_newline_gains_one(self, run,
                                                    monkeypatch, tmp_path):
        """Otherwise the first stage line is glued to the last log line.

        The stage evidence is appended directly after the tail, so a tail
        that does not end in a newline would run into it.
        """
        src = tmp_path / "app.log"
        src.write_text("first line\nlast line without a newline")
        monkeypatch.setattr(rj, "log_path", lambda: src, raising=False)
        import spacr.logging_util as lu

        monkeypatch.setattr(lu, "log_path", lambda: src)

        run._record_stage("segment", state="done")
        run._snapshot_log_tail(n=10)

        written = (run.dir / "log.txt").read_text()
        assert "last line without a newline\nFlowView stage" in written

    def test_nothing_to_write_writes_no_file_at_all(self, run, monkeypatch,
                                                    tmp_path):
        """An empty log.txt reads as "nothing was logged", which is a lie
        when there was simply nothing to copy yet."""
        import spacr.logging_util as lu

        monkeypatch.setattr(lu, "log_path", lambda: tmp_path / "absent.log")
        run._snapshot_log_tail(n=10)
        assert not (run.dir / "log.txt").exists()

    def test_a_disk_that_will_not_take_the_snapshot_is_warned_about(
            self, run, monkeypatch, tmp_path, caplog):
        """The `except Exception` arm around the write.

        Failing the run because its NOTE could not be written would be
        the exact inversion the docstring warns against.
        """
        src = tmp_path / "app.log"
        src.write_text("a line\n")
        import spacr.logging_util as lu

        monkeypatch.setattr(lu, "log_path", lambda: src)

        import pathlib

        def refuse(self, *_a, **_k):
            raise OSError("no space left on device")

        monkeypatch.setattr(pathlib.Path, "write_text", refuse)
        with caplog.at_level("WARNING"):
            run._snapshot_log_tail(n=10)
        assert "could not write run log snapshot" in caplog.text


def test_a_warnings_value_that_is_neither_a_list_nor_falsy_is_stringified(
        tmp_path, monkeypatch):
    """`search_runs`'s `elif values:` arm, and the one below it.

    NOT COVERED, and it cannot be: the third arc coverage reports here,
    `elif values:` evaluating FALSE, is unreachable. `values` is
    `manifest.get(key) or []`, so it is either the truthy original or an
    empty list -- and an empty list satisfies the `isinstance` above it.
    A falsy non-list can never reach the `elif`. This test drives the
    reachable half, so the arm itself is exercised.
    """
    d = tmp_path / "2026-08-31_010203_abcd__mask"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "app_key": "mask",
        "status": "done",
        "start_utc": "2026-08-31T01:02:03",
        "warnings": "one warning, as a bare string",
    }))
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    found = rj.search_runs()
    assert found, "the run was not found at all"
    joined = " ".join(str(v) for v in found[0].values())
    assert "one warning, as a bare string" in joined


class TestTheFileWalkerSkippingAnExcludedRoot:
    """`_iter_files` yields nothing for a path that is itself excluded."""

    def test_an_excluded_directory_yields_nothing(self, tmp_path):
        keep = tmp_path / "keep"
        keep.mkdir()
        (keep / "a.txt").write_text("a")
        assert list(rj._iter_files(keep, [])) == [keep / "a.txt"]
        assert list(rj._iter_files(keep, [keep])) == [], (
            "an excluded root was walked anyway")

    def test_an_excluded_file_yields_nothing(self, tmp_path):
        """The guard is checked before the is_file() branch below it."""
        f = tmp_path / "a.txt"
        f.write_text("a")
        assert list(rj._iter_files(f, [])) == [f]
        assert list(rj._iter_files(f, [f])) == []

    def test_a_path_under_an_excluded_root_yields_nothing(self, tmp_path):
        root = tmp_path / "project"
        inner = root / "nested"
        inner.mkdir(parents=True)
        (inner / "a.txt").write_text("a")
        assert list(rj._iter_files(inner, [root])) == []


class TestARunFolderThatCannotBeRead:
    """One bad folder must not empty the history, and must not be silent."""

    @staticmethod
    def _corrupt(root, name="2026-08-31_020000_bad__mask"):
        d = root / name
        d.mkdir()
        (d / "manifest.json").write_text("{ this is not json")
        return d

    @staticmethod
    def _good(root, name="2026-08-31_010000_good__mask"):
        d = root / name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({
            "app_key": "mask", "status": "done",
            "start_utc": "2026-08-31T01:00:00",
        }))
        return d

    def test_recent_runs_keeps_the_readable_ones_and_says_why(
            self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
        self._good(tmp_path)
        self._corrupt(tmp_path)
        with caplog.at_level("WARNING"):
            got = rj.recent_runs(10)
        assert [r["dir"].name for r in got] == ["2026-08-31_010000_good__mask"]
        assert "manifest.json could not be read" in caplog.text, (
            "the run vanished from the list with no trace anywhere")

    def test_journal_totals_skips_it_and_says_why(self, tmp_path,
                                                 monkeypatch, caplog):
        monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
        self._good(tmp_path)
        self._corrupt(tmp_path)
        with caplog.at_level("WARNING"):
            totals = rj.journal_totals()
        assert totals["total_runs"] == 1
        assert "could not be read" in caplog.text

    def test_a_folder_with_no_manifest_is_passed_over_quietly(
            self, tmp_path, monkeypatch):
        """Not an error: a run that is still starting has no manifest yet."""
        monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
        self._good(tmp_path)
        (tmp_path / "2026-08-31_030000_starting__mask").mkdir()
        assert rj.journal_totals()["total_runs"] == 1


class TestTheSettingsCsvReader:

    def test_an_ordinary_settings_csv_is_read(self, tmp_path):
        p = tmp_path / "settings.csv"
        p.write_text("Key,Value\ndiameter,30\nchannels,[0 1]\n")
        got = rj._read_settings_csv(p)
        assert got == {"diameter": "30", "channels": "[0 1]"}

    def test_a_row_with_no_value_reads_as_empty(self, tmp_path):
        p = tmp_path / "settings.csv"
        p.write_text("Key,Value\ndiameter\n")
        assert rj._read_settings_csv(p) == {"diameter": ""}

    def test_an_embedded_nul_byte_is_refused(self, tmp_path):
        """A NUL in a settings CSV means the file is damaged.

        Reading past it would put a corrupted value into the run's
        provenance, which is worse than refusing: the record would look
        like a faithful copy of what the user ran.
        """
        import csv

        p = tmp_path / "settings.csv"
        p.write_bytes(b"Key,Value\ndiameter,3\x000\n")
        with pytest.raises(csv.Error, match="NUL"):
            rj._read_settings_csv(p)


def test_a_logger_that_cannot_log_does_not_break_the_recorder(run,
                                                              monkeypatch):
    """The inner `except BaseException: pass` in `_record_stage`.

    The outer guard logs the failure; if the LOGGING itself fails there
    is nowhere left to complain, and complaining loudly would defeat the
    entire point of a guard whose job is to keep provenance from taking
    down a run.
    """

    class _Hostile:
        def items(self):
            raise RuntimeError("bad metrics")

    def refuse(*_a, **_k):
        raise RuntimeError("the log handler is gone too")

    monkeypatch.setattr(rj.LOG, "debug", refuse)

    run._record_stage("segment", metrics=_Hostile())   # must not raise

    # AND THE RUN IS STILL USABLE. That is the whole point of the guard:
    # provenance must not take a run down. A recorder left in a broken
    # state would pass "did not raise" and fail the next stage instead.
    run._record_stage("measure")
    assert [stage.get("id") for stage in run.stages] == [
        "segment", "measure",
    ], "the recorder did not accept the stage after its logger failed"


def test_a_module_outside_the_tallied_three_counts_only_as_a_run(
        tmp_path, monkeypatch):
    """`journal_totals` tallies mask, measure and classify by name.

    Every other module still counts toward `total_runs` -- the dashboard
    would otherwise under-report how much work the journal holds -- but
    gains no per-app counter of its own.
    """
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    for name, app in (("2026-08-31_010000_a__mask", "mask"),
                      ("2026-08-31_020000_b__regression", "regression")):
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({
            "app_key": app, "status": "done",
            "start_utc": "2026-08-31T01:00:00",
        }))
    totals = rj.journal_totals()
    assert totals["total_runs"] == 2
    assert totals["mask_runs"] == 1
    assert totals["measure_runs"] == 0
    assert totals["classify_runs"] == 0


def test_the_runs_root_is_made_on_first_access(tmp_path, monkeypatch):
    """`runs_root` creates ~/.spacr/runs rather than assuming it.

    Every other test in this file replaces it, so nothing here reached
    its three lines -- and a first-ever launch is exactly when they run.
    """
    import pathlib

    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path))
    expected = tmp_path / ".spacr" / "runs"
    assert not expected.exists()
    got = rj.runs_root()
    assert got == expected
    assert expected.is_dir(), "the journal root was not created"
    assert rj.runs_root() == expected      # idempotent on a second call
