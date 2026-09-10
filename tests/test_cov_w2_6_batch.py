"""A queue that survives a night nobody is watching -- the awkward paths.

The queue's whole job is bookkeeping under failure: a settings file that
went unreadable between adding a job and running it, a Stop pressed
mid-segmentation, a queue file on a share that has gone away. None of that
needs a GPU, so every case here runs the real queue with a runner that is
handed to it -- which is what the ``runner=`` parameter is for.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from spacr import batch
from spacr.batch import (
    ERROR,
    Job,
    Problem,
    Queue,
    QueueError,
    WARNING,
    load_queue,
    plan,
    resume_queue,
    run_queue,
    save_queue,
    validate_queue,
)
from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token
from spacr.cli import SettingsError


def _plate(tmp_path: Path, name: str = "plate1") -> str:
    """A folder that looks enough like a plate for the pre-flight to pass."""
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    (src / f"{name}_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"")
    return str(src)


def _settings_csv(tmp_path: Path, name: str, **values) -> str:
    path = tmp_path / name
    lines = ["Key,Value"] + [f"{k},{v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def mask_job(tmp_path):
    src = _plate(tmp_path)
    return Job(module="mask",
               settings=_settings_csv(tmp_path, "mask.csv", src=src, cell_channel=0))


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def test_a_timestamp_that_is_not_one_is_read_as_no_timestamp():
    """A hand-edited queue file can hold anything in ``started``; the queue
    must load it rather than raise while reporting a duration."""
    assert batch._parse_iso("last Tuesday") is None
    assert batch._parse_iso(None) is None
    assert batch._parse_iso(batch._now_iso()) is not None


def test_a_job_that_never_finished_has_no_duration():
    job = Job(module="mask", started=batch._now_iso())
    assert job.duration_s is None
    assert job.elapsed_s is None          # still pending, not running


def test_warnings_are_reported_beside_the_errors_not_instead_of_them():
    """A queue that will run but has something worth looking at is a
    different thing from one that cannot run."""
    text = batch.format_problems([
        Problem("mask-1", ERROR, "src does not exist", "fix it"),
        Problem("measure-1", WARNING, "no settings file", "give it one"),
    ])
    assert "WARNINGS — the queue will run" in text
    assert "no settings file" in text
    assert "src does not exist" in text


@pytest.mark.parametrize("value,expected", [
    ("/data/plate1", ["/data/plate1"]),
    (["/data/a", "/data/b"], ["/data/a", "/data/b"]),
    ("", []), (None, []), (17, []),
])
def test_src_is_normalised_to_a_list_however_it_was_written(value, expected):
    assert batch._src_values({"src": value}) == expected


def test_a_path_is_inside_the_folder_it_lives_under():
    assert batch._within("/data/plate1", "/data") is True
    assert batch._within("/data", "/data") is True
    assert batch._within("/data/plate1", "/other") is False
    assert batch._within("/database", "/data") is False


def test_a_path_that_cannot_be_resolved_is_not_inside_anything(monkeypatch):
    """This decides whether a missing input is deferred to an upstream job or
    reported as a misspelling tonight; a path the OS will not resolve is not
    evidence of either, and must not raise out of the pre-flight."""
    def _refuse(_path):
        raise ValueError("embedded null byte")

    monkeypatch.setattr(os.path, "abspath", _refuse)
    assert batch._within("/data/plate1", "/data") is False


def test_a_setting_that_is_a_number_in_a_string_is_not_a_path():
    """Keeps ``cell_mask_dim='4'`` -- a real type error -- from being
    mistaken for a path some upstream job might create."""
    assert batch._looks_like_path("4") is False
    assert batch._looks_like_path("") is False
    assert batch._looks_like_path(4) is False
    assert batch._looks_like_path("/data/plate1") is True
    assert batch._looks_like_path("~/plate1") is True
    assert batch._looks_like_path("settings.csv") is True


# --------------------------------------------------------------------------
# reading a hand-edited file
# --------------------------------------------------------------------------

def test_a_job_entry_that_is_not_an_object_says_what_one_looks_like():
    with pytest.raises(QueueError, match="not an object"):
        Job.from_dict(["mask"])


def test_a_job_entry_with_no_module_is_refused():
    with pytest.raises(QueueError, match='has no "module"'):
        Job.from_dict({"settings": "/data/mask.csv"})
    with pytest.raises(QueueError, match='has no "module"'):
        Job.from_dict({"module": "   "})


def test_a_status_nobody_defined_is_refused_by_name():
    with pytest.raises(QueueError, match="not one of"):
        Job.from_dict({"module": "mask", "status": "halfway"})


def test_one_dependency_written_as_a_bare_string_is_still_a_list():
    job = Job.from_dict({"module": "measure", "depends_on": "mask-1"})
    assert job.depends_on == ["mask-1"]


def test_a_queue_document_that_is_not_an_object_is_refused():
    with pytest.raises(QueueError, match="must hold an object"):
        Queue.from_dict([{"module": "mask"}])


def test_a_format_version_that_is_not_a_number_is_refused():
    with pytest.raises(QueueError, match="version number"):
        Queue.from_dict({"spacr_queue": "one", "jobs": []})


def test_a_queue_from_a_future_spacr_says_to_upgrade():
    with pytest.raises(QueueError, match="Upgrade spaCR"):
        Queue.from_dict({"spacr_queue": batch.QUEUE_FORMAT + 1, "jobs": []})


def test_a_queue_file_with_no_jobs_key_loads_as_an_empty_queue():
    assert len(Queue.from_dict({"spacr_queue": batch.QUEUE_FORMAT})) == 0


def test_a_jobs_key_that_is_not_a_list_is_refused():
    with pytest.raises(QueueError, match='"jobs" must be a list'):
        Queue.from_dict({"jobs": {"module": "mask"}})


def test_a_hand_written_job_gets_an_id_and_a_label_of_its_own():
    queue = Queue.from_dict({"jobs": [{"module": "mask"},
                                      {"module": "mask"}]})
    assert queue.ids == ["mask-1", "mask-2"]
    assert all(job.label for job in queue.jobs)


# --------------------------------------------------------------------------
# the queue file itself
# --------------------------------------------------------------------------

def test_a_write_that_completes_leaves_no_temporary_behind(tmp_path):
    target = batch._atomic_write(tmp_path / "queue.json", "{}\n")
    assert target.read_text() == "{}\n"
    assert [p.name for p in tmp_path.iterdir()] == ["queue.json"]


def test_a_failed_rename_reports_itself_rather_than_the_tidy_up(
        tmp_path, monkeypatch):
    """A half-written queue file would lose the record of which of twelve
    jobs had already run, so the rename failure is the news -- not a temp
    file that then could not be deleted either."""
    def _refuse_replace(src, dst):
        raise OSError(18, "Invalid cross-device link")

    def _refuse_unlink(self, *args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(os, "replace", _refuse_replace)
    monkeypatch.setattr(Path, "unlink", _refuse_unlink)
    with pytest.raises(OSError, match="cross-device"):
        batch._atomic_write(tmp_path / "queue.json", "{}\n")
    assert not (tmp_path / "queue.json").exists()


def test_a_folder_is_not_a_queue_file(tmp_path):
    (tmp_path / "queue.json").mkdir()
    with pytest.raises(QueueError, match="is a folder"):
        load_queue(tmp_path / "queue.json")


def test_a_queue_file_that_cannot_be_read_says_so_in_a_sentence(tmp_path,
                                                                monkeypatch):
    """An unattended runner should fail with a sentence, never a traceback."""
    path = tmp_path / "queue.json"
    path.write_text("{}")
    real = Path.read_text

    def _refuse(self, *args, **kwargs):
        if self == path:
            raise OSError(5, "Input/output error")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _refuse)
    with pytest.raises(QueueError, match="could not read"):
        load_queue(path)


def test_a_missing_queue_file_says_which_one(tmp_path):
    with pytest.raises(QueueError, match="queue file not found"):
        load_queue(tmp_path / "absent.json")


# --------------------------------------------------------------------------
# editing a queue
# --------------------------------------------------------------------------

def test_two_jobs_cannot_share_an_id(mask_job, tmp_path):
    queue = Queue()
    queue.add(mask_job)
    twin = Job(module="mask", settings=mask_job.settings, id=mask_job.id)
    with pytest.raises(QueueError, match="already in this queue"):
        queue.add(twin)


def test_removing_a_job_that_is_not_there_reports_that(mask_job):
    queue = Queue()
    queue.add(mask_job)
    assert queue.remove("nothing-1") is False
    assert queue.remove(mask_job.id) is True
    assert queue.ids == []


def test_moving_a_job_that_is_not_there_reports_that():
    assert Queue().move("nothing-1", -1) == -1


def test_moving_a_job_past_the_ends_clamps_rather_than_wraps(tmp_path):
    src = _plate(tmp_path)
    settings = _settings_csv(tmp_path, "mask.csv", src=src, cell_channel=0)
    queue = Queue()
    for _ in range(3):
        queue.add(Job(module="mask", settings=settings))
    assert queue.move("mask-3", -99) == 0
    assert queue.ids[0] == "mask-3"
    assert queue.move("mask-3", 99) == 2
    assert queue.ids[-1] == "mask-3"
    assert queue.move("mask-3", 0) == 2


def test_resetting_a_queue_clears_every_job_of_its_history(mask_job):
    queue = Queue()
    queue.add(mask_job)
    mask_job.status = batch.STATUS_FAILED
    mask_job.exit_code = 1
    mask_job.error = "it died"
    assert queue.reset() is queue
    assert mask_job.status == batch.STATUS_PENDING
    assert mask_job.exit_code is None
    assert mask_job.error == ""


# --------------------------------------------------------------------------
# settings resolution and dependency roots
# --------------------------------------------------------------------------

def test_a_job_naming_a_module_that_does_not_exist_cannot_be_resolved():
    with pytest.raises(SettingsError, match="unknown module"):
        batch.resolve_job_settings(Job(module="not_a_module"))


def test_upstream_roots_are_transitive_and_survive_a_broken_link(tmp_path):
    """Measure -> Classify(CV) -> Classify(ML) chains three deep and the
    folder is created by the job at the top."""
    src = _plate(tmp_path)
    mask = Job(module="mask", settings=_settings_csv(tmp_path, "mask.csv",
                                                     src=src, cell_channel=0), id="mask-1")
    measure = Job(module="measure", settings={"src": src}, id="measure-1",
                  depends_on=["mask-1", "mask-1", "gone-1"])
    queue = Queue(jobs=[mask, measure])
    roots = batch._upstream_roots(measure, queue)
    assert roots == [src]


def test_a_dependency_whose_settings_will_not_resolve_contributes_no_roots(
        tmp_path):
    broken = Job(module="mask", settings="/nowhere/missing.csv", id="mask-1")
    downstream = Job(module="measure", settings={"src": str(tmp_path)},
                     id="measure-1", depends_on=["mask-1"])
    queue = Queue(jobs=[broken, downstream])
    assert batch._upstream_roots(downstream, queue) == []


def test_a_job_with_no_dependencies_has_no_upstream_roots(mask_job):
    assert batch._upstream_roots(mask_job, Queue(jobs=[mask_job])) == []
    assert batch._upstream_roots(mask_job, None) == []


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def test_settings_that_are_neither_a_path_nor_a_mapping_are_refused():
    problems = batch.validate_job(Job(module="mask", settings=17, id="m-1"))
    assert any("a job takes a settings file path" in p.message
               for p in problems)
    assert any(p.is_error for p in problems)


def test_a_job_with_no_settings_is_warned_about_not_refused(tmp_path):
    problems = batch.validate_job(Job(module="mask", settings="", id="m-1"))
    warnings = [p for p in problems if not p.is_error]
    assert any("defaults alone" in p.message for p in warnings)


def test_dependency_rules_only_apply_inside_a_queue():
    job = Job(module="mask", id="m-1", depends_on=["nothing"])
    assert batch._dependency_problems(job, None) == []


def test_a_job_that_depends_on_itself_is_refused(mask_job):
    mask_job.id = "mask-1"
    mask_job.depends_on = ["mask-1"]
    queue = Queue(jobs=[mask_job])
    problems = batch._dependency_problems(mask_job, queue)
    assert any("depends on itself" in p.message for p in problems)


def test_a_dependency_that_is_not_in_the_queue_lists_the_ones_that_are(
        mask_job):
    mask_job.id = "mask-1"
    mask_job.depends_on = ["ghost-1"]
    queue = Queue(jobs=[mask_job])
    problems = batch._dependency_problems(mask_job, queue)
    assert any("not a job in this queue" in p.message for p in problems)
    assert any("mask-1" in (p.fix or "") for p in problems)


def test_a_dependency_that_comes_later_would_always_skip_the_job(tmp_path):
    src = _plate(tmp_path)
    settings = _settings_csv(tmp_path, "mask.csv", src=src, cell_channel=0)
    first = Job(module="mask", settings=settings, id="a-1",
                depends_on=["b-1"])
    second = Job(module="mask", settings=settings, id="b-1")
    queue = Queue(jobs=[first, second])
    problems = batch._dependency_problems(first, queue)
    assert any("comes *later*" in p.message for p in problems)


def test_an_empty_queue_is_a_warning_not_an_error():
    problems = validate_queue(Queue())
    assert len(problems) == 1
    assert not problems[0].is_error
    assert "nothing would run" in problems[0].message


def test_a_job_with_no_id_at_all_is_reported(tmp_path):
    queue = Queue(jobs=[Job(module="mask", settings={"src": _plate(tmp_path)})])
    problems = validate_queue(queue)
    assert any("has no id" in p.message and p.is_error for p in problems)


def test_a_dependency_cycle_is_reported_as_one(tmp_path):
    src = _plate(tmp_path)
    settings = _settings_csv(tmp_path, "mask.csv", src=src, cell_channel=0)
    a = Job(module="mask", settings=settings, id="a-1", depends_on=["b-1"])
    b = Job(module="mask", settings=settings, id="b-1", depends_on=["a-1"])
    problems = validate_queue(Queue(jobs=[a, b]))
    assert any("dependency cycle" in p.message for p in problems)


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

def test_the_plan_says_what_it_could_not_read_rather_than_omitting_it(
        tmp_path):
    queue = Queue(jobs=[Job(module="mask", settings="/nowhere/gone.csv",
                            id="mask-1", label="mask @ nowhere")])
    text = plan(queue)
    assert "settings unreadable" in text


def test_the_plan_shows_the_overrides_and_a_job_that_already_ran(tmp_path):
    src = _plate(tmp_path)
    job = Job(module="mask", settings=_settings_csv(tmp_path, "mask.csv",
                                                    src=src, cell_channel=0),
              id="mask-1", overrides=["diameter=30"],
              status=batch.STATUS_SUCCESS)
    text = plan(Queue(jobs=[job]))
    assert "--set diameter=30" in text
    assert f"status {batch.STATUS_SUCCESS}" in text
    assert src in text


def test_a_detailed_plan_renders_what_each_job_would_actually_do(tmp_path):
    src = _plate(tmp_path)
    job = Job(module="mask", settings=_settings_csv(tmp_path, "mask.csv",
                                                    src=src, cell_channel=0), id="mask-1")
    text = plan(Queue(jobs=[job]), detail=True)
    assert len(text.splitlines()) > len(
        plan(Queue(jobs=[job])).splitlines())


def test_a_detailed_plan_says_when_the_detail_is_unavailable(tmp_path,
                                                             monkeypatch):
    src = _plate(tmp_path)
    job = Job(module="mask", settings=_settings_csv(tmp_path, "mask.csv",
                                                    src=src, cell_channel=0), id="mask-1")
    from spacr import validate as validate_module

    def _refuse(settings, key):
        raise OSError(5, "the share went away")

    monkeypatch.setattr(validate_module, "describe_plan", _refuse)
    assert "plan unavailable" in plan(Queue(jobs=[job]), detail=True)


# --------------------------------------------------------------------------
# the runners
# --------------------------------------------------------------------------

def test_a_job_process_runs_and_its_output_lands_in_its_own_log(tmp_path,
                                                                monkeypatch):
    """One interleaved log from twelve overnight jobs is unreadable."""
    log = tmp_path / "logs" / "01_mask-1.log"
    monkeypatch.setattr(batch, "job_command", lambda job, settings, python=None:
                        [sys.executable, "-c", "print('segmenting')"])
    code = batch.subprocess_runner(Job(module="mask", id="mask-1"), "", str(log))
    assert code == 0
    text = log.read_text()
    assert "segmenting" in text
    assert "exit code 0" in text


def test_a_process_that_will_not_start_is_a_usage_failure(tmp_path,
                                                          monkeypatch):
    log = tmp_path / "01_mask-1.log"
    monkeypatch.setattr(batch, "job_command", lambda job, settings, python=None:
                        ["/no/such/interpreter"])
    code = batch.subprocess_runner(Job(module="mask", id="mask-1"), "",
                                   str(log))
    assert code == batch.EXIT_USAGE
    assert "could not start the job process" in log.read_text()


def test_a_running_job_is_polled_so_a_stop_can_be_honoured(tmp_path,
                                                           monkeypatch):
    """Without a token the queue blocks in wait(); with one it has to come
    up for air often enough to notice a Stop."""
    log = tmp_path / "01_mask-1.log"
    monkeypatch.setattr(batch, "job_command", lambda job, settings, python=None:
                        [sys.executable, "-c",
                         "import time; time.sleep(0.3); print('done')"])
    with installed_token(CancellationToken()):
        code = batch.subprocess_runner(Job(module="mask", id="mask-1"), "",
                                       str(log))
    assert code == 0
    assert "done" in log.read_text()


def test_a_stop_pressed_mid_job_stops_the_child_and_says_so(tmp_path,
                                                            monkeypatch):
    log = tmp_path / "01_mask-1.log"
    monkeypatch.setattr(batch, "job_command", lambda job, settings, python=None:
                        [sys.executable, "-c",
                         "import time; time.sleep(30)"])
    token = CancellationToken()
    with installed_token(token):
        token.cancel("the user pressed Stop")
        with pytest.raises(PipelineCancelled):
            batch.subprocess_runner(Job(module="mask", id="mask-1"), "",
                                    str(log))
    assert "cancelled safely" in log.read_text()


def test_the_in_process_runner_calls_the_cli_and_tees_its_output(tmp_path,
                                                                 monkeypatch):
    from spacr import cli

    log = tmp_path / "inprocess.log"
    seen = {}

    def _fake_main(argv):
        seen["argv"] = list(argv)
        print("running in this interpreter")
        return 0

    monkeypatch.setattr(cli, "main", _fake_main)
    job = Job(module="mask", id="mask-1", overrides=["diameter=30"])
    assert batch.inprocess_runner(job, "/data/mask.csv", str(log)) == 0
    assert seen["argv"] == ["mask", "--settings", "/data/mask.csv",
                            "--set", "diameter=30"]
    text = log.read_text()
    assert "in-process" in text
    assert "running in this interpreter" in text


def test_the_in_process_runner_honours_a_sys_exit_from_the_cli(tmp_path,
                                                               monkeypatch):
    from spacr import cli

    monkeypatch.setattr(cli, "main",
                        lambda argv: (_ for _ in ()).throw(SystemExit(2)))
    assert batch.inprocess_runner(Job(module="mask", id="m-1"), "",
                                  str(tmp_path / "log")) == 2

    monkeypatch.setattr(cli, "main",
                        lambda argv: (_ for _ in ()).throw(SystemExit("bad")))
    assert batch.inprocess_runner(Job(module="mask", id="m-1"), "",
                                  str(tmp_path / "log")) == 1


# --------------------------------------------------------------------------
# reading a job's own ledger back
# --------------------------------------------------------------------------

def test_a_run_status_sidecar_beside_the_plate_is_an_artifact(tmp_path):
    src = Path(_plate(tmp_path))
    sidecar = src / f"mask{batch.RUN_STATUS_SUFFIX}"
    sidecar.write_text("{}")
    found = [p.name for p in batch._status_artifacts({"src": str(src)})]
    assert sidecar.name in found


def test_an_artifact_with_no_new_stamp_contributes_nothing(tmp_path,
                                                           monkeypatch):
    """A measurements.db accumulates one row per stage, so 'what did THIS
    job record' is the difference, not the total."""
    src = Path(_plate(tmp_path))
    (src / "mask.run_status.json").write_text("{}")
    monkeypatch.setattr(batch, "_status_artifacts",
                        lambda settings: [src / "mask.run_status.json"])
    monkeypatch.setattr(batch, "read_run_status",
                        lambda artifact: [{"n_attempted": 5,
                                           "n_succeeded": 5}])
    settings = {"src": str(src)}
    before = batch._status_snapshot(settings)
    assert batch._collect_run_status(settings, before) is None


def test_a_stamp_that_attempted_nothing_is_empty_not_complete(tmp_path,
                                                              monkeypatch):
    """'complete' is the word the queue uses to decide there is nothing to
    re-run, and a stage that attempted no items has not earned it."""
    src = Path(_plate(tmp_path))
    artifact = src / "mask.run_status.json"
    artifact.write_text("{}")
    monkeypatch.setattr(batch, "_status_artifacts",
                        lambda settings: [artifact])
    monkeypatch.setattr(batch, "read_run_status",
                        lambda a: [{"n_attempted": 0, "n_succeeded": 0,
                                    "n_failed": 0}])
    out = batch._collect_run_status({"src": str(src)}, {})
    assert out["status"] == "empty"
    assert out["records"] == 1


def test_a_log_that_is_not_there_has_no_tail(tmp_path):
    assert batch._log_tail(str(tmp_path / "absent.log")) == []


def test_a_usage_failure_with_nothing_in_the_log_still_names_itself(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("\n\n")
    kind, message = batch.classify_failure(batch.EXIT_USAGE, str(log))
    assert kind == "ConfigurationError"
    assert "exit code 2" in message


# --------------------------------------------------------------------------
# running the queue
# --------------------------------------------------------------------------

def _ok_runner(code=0, text=""):
    def runner(job, settings_path, log_path):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(text)
        return code
    return runner


def test_a_queue_file_that_cannot_be_written_does_not_end_the_night(
        tmp_path, monkeypatch, caplog):
    """The run continues; it just could not be resumed from there."""
    src = _plate(tmp_path)
    job = Job(module="mask", settings=_settings_csv(tmp_path, "mask.csv",
                                                    src=src, cell_channel=0))
    queue = Queue(name="overnight")
    queue.add(job)

    def _refuse(queue_, path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(batch, "save_queue", _refuse)
    result = run_queue(queue, path=str(tmp_path / "q.json"),
                       runner=_ok_runner(), echo=False)
    assert result.queue.jobs[0].status == batch.STATUS_SUCCESS


def test_a_job_whose_settings_went_missing_fails_without_being_run(tmp_path):
    """It was addable when it was added; the file went away since."""
    src = _plate(tmp_path)
    good = Job(module="mask", settings=_settings_csv(tmp_path, "mask.csv",
                                                     src=src, cell_channel=0), id="mask-1")
    queue = Queue()
    queue.add(good)
    queue.jobs.append(Job(module="mask", settings="/nowhere/gone.csv",
                          id="mask-2", label="mask @ nowhere"))
    ran = []

    def runner(job, settings_path, log_path):
        ran.append(job.id)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("")
        return 0

    result = run_queue(queue, runner=runner, echo=False, force=True)
    assert ran == ["mask-1"]
    failed = queue.find("mask-2")
    assert failed.status == batch.STATUS_FAILED
    assert failed.exit_code == batch.EXIT_USAGE
    assert result.failed == [failed]


def test_an_unreadable_settings_file_can_stop_the_whole_queue(tmp_path):
    queue = Queue()
    queue.jobs.append(Job(module="mask", settings="/nowhere/gone.csv",
                          id="mask-1", label="mask @ nowhere"))
    queue.jobs.append(Job(module="mask", settings="/nowhere/also.csv",
                          id="mask-2", label="mask @ nowhere"))
    result = run_queue(queue, runner=_ok_runner(), on_error="stop",
                       echo=False, force=True)
    assert queue.find("mask-2").status == batch.STATUS_NOT_RUN
    assert 'on_error="stop"' in result.stopped_reason


def test_repeated_settings_failures_stop_the_queue_as_systematic(tmp_path):
    queue = Queue()
    for i in range(4):
        queue.jobs.append(Job(module="mask", settings=f"/nowhere/{i}.csv",
                              id=f"mask-{i}", label="mask @ nowhere"))
    result = run_queue(queue, runner=_ok_runner(),
                       max_consecutive_failures=2, echo=False, force=True)
    assert "in a row" in result.stopped_reason
    assert queue.find("mask-3").status == batch.STATUS_NOT_RUN


def test_a_stop_pressed_during_a_job_leaves_it_resumable(tmp_path):
    src = _plate(tmp_path)
    job = Job(module="mask", settings=_settings_csv(tmp_path, "mask.csv",
                                                    src=src, cell_channel=0))
    queue = Queue()
    queue.add(job)
    events = []

    def runner(job_, settings_path, log_path):
        raise PipelineCancelled("the user pressed Stop")

    with pytest.raises(PipelineCancelled):
        run_queue(queue, path=str(tmp_path / "q.json"), runner=runner,
                  on_progress=events.append, echo=False)
    assert job.status == batch.STATUS_NOT_RUN
    assert "run this job again to resume" in job.error
    assert any(e.event == "queue_stopped" for e in events)
    # ... and the file on disk says so, so a resume picks it up.
    assert load_queue(tmp_path / "q.json").jobs[0].status == \
        batch.STATUS_NOT_RUN


def test_ctrl_c_during_a_job_marks_it_interrupted_and_halts_the_rest(
        tmp_path):
    src = _plate(tmp_path)
    settings = _settings_csv(tmp_path, "mask.csv", src=src, cell_channel=0)
    queue = Queue()
    queue.add(Job(module="mask", settings=settings, id="mask-1"))
    queue.add(Job(module="mask", settings=settings, id="mask-2"))

    def runner(job, settings_path, log_path):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("")
        if job.id == "mask-1":
            raise KeyboardInterrupt
        return 0

    result = run_queue(queue, runner=runner, echo=False)
    assert queue.find("mask-1").status == batch.STATUS_FAILED
    assert "interrupted by the user" in queue.find("mask-1").error
    assert queue.find("mask-2").status == batch.STATUS_NOT_RUN
    assert "Ctrl-C" in result.stopped_reason


def test_the_summary_is_printed_when_the_caller_asks_for_it(tmp_path,
                                                            capsys):
    src = _plate(tmp_path)
    queue = Queue(name="overnight")
    queue.add(Job(module="mask", settings=_settings_csv(tmp_path, "mask.csv",
                                                        src=src, cell_channel=0)))
    run_queue(queue, runner=_ok_runner(), echo=True)
    assert "overnight" in capsys.readouterr().out


def test_a_queue_file_that_cannot_be_stamped_does_not_lose_the_result(
        tmp_path, monkeypatch, caplog):
    src = _plate(tmp_path)
    queue = Queue()
    queue.add(Job(module="mask", settings=_settings_csv(tmp_path, "mask.csv",
                                                        src=src, cell_channel=0)))
    from spacr.errors import RunLedger

    def _refuse(self, path):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(RunLedger, "stamp", _refuse)
    result = run_queue(queue, path=str(tmp_path / "q.json"),
                       runner=_ok_runner(), echo=False)
    assert result.ok is True


def test_a_dependency_that_is_not_in_the_queue_does_not_block(tmp_path):
    """`validate_queue` already reported it as an error; blocking on it here
    would report the same thing twice in different words."""
    job = Job(module="mask", id="mask-1", depends_on=["ghost-1"])
    assert batch._blocking_dependency(job, Queue(jobs=[job])) is None


# --------------------------------------------------------------------------
# resuming
# --------------------------------------------------------------------------

def test_a_module_with_no_field_checkpoints_is_not_given_resume():
    job = Job(module="ml_analyze", overrides=["a=1"])
    batch._enable_field_resume(job)
    assert job.override_args == ["a=1"]


def test_a_job_written_with_a_mapping_of_overrides_keeps_that_shape():
    job = Job(module="mask", overrides={"diameter": 30})
    batch._enable_field_resume(job)
    assert job.overrides == {"diameter": 30, "resume": True}


def test_an_existing_resume_override_is_replaced_not_duplicated():
    job = Job(module="measure", overrides=["resume=False", "diameter=30"])
    batch._enable_field_resume(job)
    assert job.override_args == ["diameter=30", "resume=True"]


def test_a_job_that_was_running_when_the_machine_died_is_resumed(tmp_path):
    src = _plate(tmp_path)
    settings = _settings_csv(tmp_path, "mask.csv", src=src, cell_channel=0)
    queue = Queue()
    queue.add(Job(module="mask", settings=settings, id="mask-1"))
    path = tmp_path / "q.json"
    queue.find("mask-1").status = batch.STATUS_RUNNING
    save_queue(queue, path)

    ran = []

    def runner(job, settings_path, log_path):
        ran.append(job.override_args)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("")
        return 0

    result = resume_queue(path, runner=runner, echo=False)
    assert ran == [["resume=True"]]
    assert result.queue.find("mask-1").status == batch.STATUS_SUCCESS


def test_a_queue_that_never_started_has_no_duration():
    result = batch.QueueResult(queue=Queue())
    assert result.duration_s is None
    assert result.ok is True


def test_a_child_that_ignores_a_polite_stop_is_killed(tmp_path, monkeypatch):
    """Terminate, wait, then kill: a wedged Cellpose process must not hold
    the queue open for the rest of the night."""
    log = tmp_path / "01_mask-1.log"
    monkeypatch.setattr(batch, "job_command", lambda job, settings, python=None:
                        [sys.executable, "-c",
                         "import signal, time; "
                         "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                         "time.sleep(60)"])
    token = CancellationToken()
    with installed_token(token):
        token.cancel("the user pressed Stop")
        with pytest.raises(PipelineCancelled):
            batch.subprocess_runner(Job(module="mask", id="mask-1"), "",
                                    str(log))
    assert "cancelled safely" in log.read_text()


def test_a_job_that_was_still_running_when_the_queue_halted_is_not_run(
        tmp_path):
    """`not_run` and `running` are different things, and a queue file that
    still says `running` would be resumed as a crashed job rather than as
    one that never started."""
    src = _plate(tmp_path)
    settings = _settings_csv(tmp_path, "mask.csv", src=src, cell_channel=0)
    queue = Queue()
    queue.add(Job(module="mask", settings=settings, id="mask-1"))
    queue.add(Job(module="mask", settings=settings, id="mask-2"))
    queue.find("mask-2").status = batch.STATUS_RUNNING

    def runner(job, settings_path, log_path):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("RuntimeError: the share went away\n")
        return 1

    result = run_queue(queue, runner=runner, on_error="stop", echo=False)
    assert queue.find("mask-1").status == batch.STATUS_FAILED
    assert queue.find("mask-2").status == batch.STATUS_NOT_RUN
    assert result.not_run == [queue.find("mask-2")]
