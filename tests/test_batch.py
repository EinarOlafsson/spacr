"""The batch / queue runner — :mod:`spacr.batch`.

Job execution is mocked throughout. A test that segments an image would be
testing cellpose, not the queue; what is actually load-bearing here is the
bookkeeping that decides whether twelve jobs survive a night nobody is
watching:

* a queue file round-trips, and a hand-written one loads;
* every invalid job is reported at once, and an invalid queue refuses to start;
* ``on_error='continue'`` runs the *independent* jobs after a failure;
* a job whose dependency failed is **skipped**, never run — the single most
  important behaviour in this module, because Measure after a failed Mask
  writes a database that looks like a real result;
* ``on_error='stop'`` leaves the rest ``not run``, which is not ``skipped``;
* the consecutive-failure threshold stops a systematic failure and says so;
* state is persisted after *every* transition and :func:`resume_queue` picks
  up exactly where the queue stopped;
* the queue file is written atomically, so a crash mid-write cannot truncate
  the record of what already ran;
* a job that exits 0 but stamped a partial ``run_status`` is reported as
  partial rather than as a success;
* identical failures are grouped;
* a GUI-only module is refused when the job is *added*, with the explanation;
* and importing the module costs no torch, no Qt and no cellpose.
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
    Job,
    Queue,
    QueueError,
    load_queue,
    plan,
    resume_queue,
    run_queue,
    save_queue,
    validate_queue,
)
from spacr.errors import RunLedger

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _plate(tmp_path: Path, name: str = "plate1") -> str:
    """A folder that looks enough like a plate for the pre-flight to pass."""
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    # One cellvoyager-named file: spacr.validate only lists names, never reads
    # pixels, so an empty file is a real input as far as validation goes.
    (src / f"{name}_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"")
    return str(src)


def _merged(src: str) -> str:
    """Give a plate the ``merged/*.npy`` that the measure pre-flight demands."""
    merged = Path(src) / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "field_1.npy").write_bytes(b"")
    return str(merged)


def _settings_csv(tmp_path: Path, name: str, **values) -> str:
    """Write a two-column ``Key,Value`` settings CSV, as the GUI does."""
    path = tmp_path / name
    lines = ["Key,Value"] + [f"{k},{v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class FakeRunner:
    """Stands in for ``spacr-run``: records the call, returns a canned code.

    :param codes: exit code per job id; ``default`` covers the rest.
    :param log_text: text written into each job's log, per job id.
    """

    def __init__(self, codes=None, default=0, log_text=None, hook=None):
        self.codes = dict(codes or {})
        self.default = default
        self.log_text = dict(log_text or {})
        self.hook = hook
        self.calls = []

    def __call__(self, job, settings_path, log_path) -> int:
        self.calls.append(job.id)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(
            self.log_text.get(job.id, f"# {job.id} ran\n"), encoding="utf-8")
        if self.hook is not None:
            self.hook(job, settings_path, log_path)
        return int(self.codes.get(job.id, self.default))

    @property
    def ran(self):
        """Ids of the jobs that were actually executed."""
        return list(self.calls)


@pytest.fixture
def plate(tmp_path):
    return _plate(tmp_path)


@pytest.fixture
def mask_measure_queue(tmp_path, plate):
    """The canonical overnight chain: Mask, then Measure on the same plate."""
    _merged(plate)
    queue = Queue(name="night")
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}))
    queue.add(Job(module="measure",
                  settings={"src": plate, "cell_mask_dim": 4, "channels": [0]},
                  depends_on=["mask-1"]))
    return queue


# ---------------------------------------------------------------------------
# the queue file
# ---------------------------------------------------------------------------


def test_queue_file_round_trips(tmp_path, mask_measure_queue):
    """Save, load, and get the same queue back — jobs, order, ids and all."""
    path = tmp_path / "night.queue.json"
    save_queue(mask_measure_queue, path)
    loaded = load_queue(path)

    assert loaded.name == mask_measure_queue.name
    assert loaded.ids == mask_measure_queue.ids == ["mask-1", "measure-1"]
    assert loaded.to_dict() == mask_measure_queue.to_dict()
    assert loaded.jobs[1].depends_on == ["mask-1"]


def test_queue_file_is_hand_editable(tmp_path, plate):
    """A user fixing job 9 at 3 a.m. opens the file in an editor, not a GUI.

    So the minimum a human would type must load: a module, a settings path,
    and nothing else. Ids, labels and bookkeeping are filled in.
    """
    _merged(plate)
    path = tmp_path / "hand.json"
    path.write_text(json.dumps({
        "spacr_queue": 1,
        "name": "hand-written",
        "jobs": [
            {"module": "mask", "settings": {"src": plate, "cell_channel": 0}},
            {"module": "measure",
             "settings": {"src": plate, "cell_mask_dim": 4},
             "depends_on": "mask-1"},
        ],
    }, indent=2), encoding="utf-8")

    queue = load_queue(path)
    assert queue.name == "hand-written"
    assert queue.ids == ["mask-1", "measure-1"]
    assert queue.jobs[1].depends_on == ["mask-1"]          # a bare string works
    assert queue.jobs[0].label                              # derived, not required
    assert all(job.status == batch.STATUS_PENDING for job in queue.jobs)
    assert not [p for p in validate_queue(queue) if p.is_error]


def test_load_queue_rejects_nonsense_with_a_sentence(tmp_path):
    """An unattended runner must fail with an explanation, not a traceback."""
    missing = tmp_path / "nope.json"
    with pytest.raises(QueueError) as excinfo:
        load_queue(missing)
    assert "not found" in str(excinfo.value)

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(QueueError) as excinfo:
        load_queue(broken)
    assert "not valid JSON" in str(excinfo.value)

    future = tmp_path / "future.json"
    future.write_text(json.dumps({"spacr_queue": 99, "jobs": []}), encoding="utf-8")
    with pytest.raises(QueueError) as excinfo:
        load_queue(future)
    assert "format 99" in str(excinfo.value)


def test_save_queue_is_atomic_and_never_truncates(tmp_path, mask_measure_queue,
                                                  monkeypatch):
    """Simulate a crash at the moment of the swap: the old file must survive.

    A queue file truncated halfway through a write is worse than no queue file
    at all — it is the record of which of twelve jobs already ran.
    """
    path = tmp_path / "night.queue.json"
    save_queue(mask_measure_queue, path)
    original = path.read_text(encoding="utf-8")
    assert json.loads(original)["jobs"]

    mask_measure_queue.jobs[0].status = batch.STATUS_SUCCESS

    def _die(*_a, **_k):
        raise OSError("the machine went down mid-write")

    monkeypatch.setattr(batch.os, "replace", _die)
    with pytest.raises(OSError):
        save_queue(mask_measure_queue, path)

    assert path.read_text(encoding="utf-8") == original    # intact, not truncated
    assert json.loads(path.read_text(encoding="utf-8"))    # and still parses
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name]
    assert leftovers == [], f"temp files left behind: {leftovers}"


# ---------------------------------------------------------------------------
# validation happens when the job is added
# ---------------------------------------------------------------------------


def test_interactive_only_module_is_rejected_when_added(plate):
    """'annotate' has no headless callable, and the message says why.

    Rejecting it at Add is the point: the alternative is a queue that looks
    fine all evening and dies on job 4 at 1 a.m.
    """
    queue = Queue()
    with pytest.raises(QueueError) as excinfo:
        queue.add(Job(module="annotate", settings={"src": plate}))
    text = str(excinfo.value)
    assert "GUI-only" in text
    assert "spacr-qt" in text          # the actual explanation from cli.py
    assert queue.jobs == []


def test_unknown_module_is_rejected_when_added(plate):
    queue = Queue()
    with pytest.raises(QueueError) as excinfo:
        queue.add(Job(module="maks", settings={"src": plate}))
    assert "unknown module" in str(excinfo.value)


def test_missing_settings_file_is_rejected_when_added(tmp_path):
    queue = Queue()
    with pytest.raises(QueueError) as excinfo:
        queue.add(Job(module="mask", settings=str(tmp_path / "gone.csv")))
    assert "not found" in str(excinfo.value)


def test_bad_override_is_rejected_when_added(tmp_path, plate):
    """``--set`` coercion is cli.py's, so a value it refuses the queue refuses."""
    settings = _settings_csv(tmp_path, "mask.csv", src=plate, cell_channel=0)
    queue = Queue()
    with pytest.raises(QueueError) as excinfo:
        queue.add(Job(module="mask", settings=settings,
                      overrides=["not_a_real_setting=3"]))
    assert "does not exist" in str(excinfo.value)

    ok = queue.add(Job(module="mask", settings=settings,
                       overrides=["verbose=True"]))
    assert ok.override_args == ["verbose=True"]


def test_validation_reports_every_invalid_job_at_once(tmp_path, plate):
    """Three broken jobs produce three errors, not one.

    One-at-a-time reporting is how a twelve-job queue takes twelve rounds of
    fixing.
    """
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}))
    queue.add(Job(module="maks", settings={"src": plate}, id="typo-module"),
              validate=False)
    queue.add(Job(module="mask", settings=str(tmp_path / "gone.csv"), id="no-file"),
              validate=False)
    queue.add(Job(module="mask", settings={"src": str(tmp_path / "plaet9")},
                  id="typo-src"), validate=False)

    problems = validate_queue(queue)
    errors = [p for p in problems if p.is_error]
    assert {p.job_id for p in errors} == {"typo-module", "no-file", "typo-src"}

    text = batch.format_problems(problems)
    for job_id in ("typo-module", "no-file", "typo-src"):
        assert job_id in text
    assert f"{len(errors)} error(s)" in text
    assert len(errors) >= 3


def test_an_invalid_queue_refuses_to_start(tmp_path, plate):
    """run_queue raises before the first job, listing everything wrong."""
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}))
    queue.add(Job(module="mask", settings={"src": str(tmp_path / "plaet9")},
                  id="typo-src"), validate=False)
    runner = FakeRunner()

    with pytest.raises(QueueError) as excinfo:
        run_queue(queue, runner=runner, echo=False)

    assert runner.ran == [], "the queue started despite an invalid job"
    assert "refusing to start" in str(excinfo.value)
    assert "typo-src" in str(excinfo.value)
    assert queue.jobs[0].status == batch.STATUS_PENDING


def test_force_runs_an_invalid_queue_anyway(tmp_path, plate):
    """The escape hatch exists, and is not the default."""
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": str(tmp_path / "plaet9")},
                  id="typo-src"), validate=False)
    runner = FakeRunner()
    result = run_queue(queue, runner=runner, force=True, echo=False)
    assert runner.ran == ["typo-src"]
    assert result.succeeded


def test_duplicate_ids_and_cycles_and_forward_dependencies_are_errors(plate):
    """Structural problems no run order can satisfy are caught up front."""
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.jobs.append(Job(module="mask", settings={"src": plate, "cell_channel": 0},
                          id="a", label="clone"))
    messages = " ".join(p.message for p in validate_queue(queue) if p.is_error)
    assert "share the id" in messages

    cyc = Queue()
    cyc.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    cyc.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="b",
                depends_on=["a"]))
    cyc.jobs[0].depends_on = ["b"]
    messages = " ".join(p.message for p in validate_queue(cyc) if p.is_error)
    assert "dependency cycle" in messages
    assert "comes *later* in the queue" in messages


def test_removing_a_job_drops_the_dependency_on_it(plate):
    """A dangling depends_on would silently skip everything behind it."""
    _merged(plate)
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.add(Job(module="measure", settings={"src": plate, "cell_mask_dim": 4},
                  id="b", depends_on=["a"]))
    assert queue.remove("a") is True
    assert queue.jobs[0].depends_on == []
    assert not [p for p in validate_queue(queue) if p.is_error]


# -- deferred validation ----------------------------------------------------


def test_a_chained_job_is_addable_before_its_input_exists(tmp_path, plate):
    """Measure behind Mask, on a plate with no merged/ yet, must be addable.

    Refusing it would refuse every Mask→Measure queue ever written, which is
    the whole use case. The error is downgraded to a deferred warning because
    an earlier job in *this* queue writes into that folder.
    """
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}))
    job = queue.add(Job(module="measure",
                        settings={"src": plate, "cell_mask_dim": 4},
                        depends_on=["mask-1"]))

    problems = batch.validate_job(job, queue)
    assert not [p for p in problems if p.is_error]
    deferred = [p for p in problems if "deferred" in p.message]
    assert deferred, "the missing merged/ folder should be reported as deferred"
    assert "merged" in deferred[0].message


def test_a_misspelled_src_is_still_an_error_on_a_chained_job(tmp_path, plate):
    """Deferring is not the same as not checking.

    Job 9's ``/data/plaet9`` is not written by any upstream job, so it stays
    an error and is found tonight rather than at 3 a.m.
    """
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}))
    with pytest.raises(QueueError) as excinfo:
        queue.add(Job(module="measure",
                      settings={"src": str(tmp_path / "plaet9"), "cell_mask_dim": 4},
                      depends_on=["mask-1"]))
    assert "does not exist" in str(excinfo.value)


def test_a_type_error_is_never_deferred(plate):
    """``cell_mask_dim='4'`` is wrong now and still wrong in six hours."""
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}))
    with pytest.raises(QueueError) as excinfo:
        queue.add(Job(module="measure",
                      settings={"src": plate, "cell_mask_dim": "4"},
                      depends_on=["mask-1"]))
    assert "cell_mask_dim" in str(excinfo.value)


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def test_a_failure_does_not_stop_the_independent_jobs(tmp_path, plate):
    """continue-on-error is what saves a night from one bad plate."""
    other = _plate(tmp_path, "plate2")
    queue = Queue(name="mixed")
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.add(Job(module="mask", settings={"src": other, "cell_channel": 0}, id="b"))
    queue.add(Job(module="mask", settings={"src": other, "cell_channel": 0}, id="c"))
    runner = FakeRunner(codes={"a": 1})

    result = run_queue(queue, runner=runner, on_error="continue",
                       max_consecutive_failures=0, echo=False)

    assert runner.ran == ["a", "b", "c"]
    assert [job.status for job in queue.jobs] == [
        batch.STATUS_FAILED, batch.STATUS_SUCCESS, batch.STATUS_SUCCESS]
    assert [job.id for job in result.failed] == ["a"]
    assert result.ok is False


def test_a_job_whose_dependency_failed_is_skipped_not_run(tmp_path, plate):
    """THE test. Measure after a failed Mask must not run at all.

    Running it produces a database that is empty or partial and looks exactly
    like a real result — and every downstream number computed from it is
    wrong without saying so. 'skipped' is its own status, and it names the
    upstream job.
    """
    _merged(plate)
    other = _plate(tmp_path, "plate2")
    queue = Queue(name="chain")
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="mask-a"))
    queue.add(Job(module="measure", settings={"src": plate, "cell_mask_dim": 4},
                  id="measure-a", depends_on=["mask-a"]))
    queue.add(Job(module="mask", settings={"src": other, "cell_channel": 0},
                  id="mask-b"))
    runner = FakeRunner(codes={"mask-a": 1})

    result = run_queue(queue, runner=runner, on_error="continue",
                       max_consecutive_failures=0, echo=False)

    assert "measure-a" not in runner.ran, "a job with a failed dependency was RUN"
    assert runner.ran == ["mask-a", "mask-b"]

    skipped = queue.find("measure-a")
    assert skipped.status == batch.STATUS_SKIPPED
    assert skipped.status != batch.STATUS_FAILED != batch.STATUS_NOT_RUN
    assert "mask-a" in skipped.error and "failed" in skipped.error
    assert skipped.exit_code is None

    summary = result.summary()
    assert "Skipped" in summary
    assert "measure-a" in summary
    assert [job.id for job in result.skipped] == ["measure-a"]


def test_skipping_is_transitive(tmp_path, plate):
    """A three-deep chain does not quietly restart in the middle."""
    _merged(plate)
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.add(Job(module="measure", settings={"src": plate, "cell_mask_dim": 4},
                  id="b", depends_on=["a"]))
    queue.add(Job(module="measure", settings={"src": plate, "cell_mask_dim": 4},
                  id="c", depends_on=["b"]))
    runner = FakeRunner(codes={"a": 1})

    run_queue(queue, runner=runner, max_consecutive_failures=0, echo=False)

    assert runner.ran == ["a"]
    assert queue.find("b").status == batch.STATUS_SKIPPED
    assert queue.find("c").status == batch.STATUS_SKIPPED
    assert "b" in queue.find("c").error


def test_on_error_stop_leaves_the_rest_not_run(tmp_path, plate):
    """'not run' is not 'skipped': nothing is wrong with these jobs."""
    other = _plate(tmp_path, "plate2")
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.add(Job(module="mask", settings={"src": other, "cell_channel": 0}, id="b"))
    queue.add(Job(module="mask", settings={"src": other, "cell_channel": 0}, id="c"))
    runner = FakeRunner(codes={"a": 1})

    result = run_queue(queue, runner=runner, on_error="stop", echo=False)

    assert runner.ran == ["a"]
    assert queue.find("a").status == batch.STATUS_FAILED
    assert queue.find("b").status == batch.STATUS_NOT_RUN
    assert queue.find("c").status == batch.STATUS_NOT_RUN
    assert not result.skipped, "nothing was skipped — the queue simply stopped"
    assert [job.id for job in result.not_run] == ["b", "c"]
    assert 'on_error="stop"' in result.stopped_reason
    assert "Not run" in result.summary()


def test_consecutive_failure_threshold_stops_and_says_why(tmp_path, plate):
    """Three failures in a row is a systematic problem, not three accidents.

    Continuing would spend the night repeating one mistake nine more times.
    """
    queue = Queue(name="systematic")
    for i in range(6):
        queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0},
                      id=f"j{i}"))
    runner = FakeRunner(default=1,
                        log_text={f"j{i}": "FileNotFoundError: /mnt/share is gone\n"
                                  for i in range(6)})

    result = run_queue(queue, runner=runner, on_error="continue",
                       max_consecutive_failures=3, echo=False)

    assert runner.ran == ["j0", "j1", "j2"]
    assert [job.status for job in queue.jobs[3:]] == [batch.STATUS_NOT_RUN] * 3
    assert "3 jobs failed in a row" in result.stopped_reason
    assert "systematic" in result.stopped_reason
    assert "max_consecutive_failures" in result.stopped_reason
    assert "STOPPED" in result.summary()


def test_a_success_resets_the_consecutive_counter(tmp_path, plate):
    """Two failures either side of a success are not a systematic failure."""
    queue = Queue()
    for i in range(5):
        queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0},
                      id=f"j{i}"))
    runner = FakeRunner(codes={"j0": 1, "j1": 1, "j2": 0, "j3": 1, "j4": 1})

    result = run_queue(queue, runner=runner, max_consecutive_failures=3, echo=False)

    assert runner.ran == ["j0", "j1", "j2", "j3", "j4"]
    assert result.stopped_reason == ""


def test_identical_failures_are_grouped_in_the_summary(tmp_path, plate):
    """Four jobs killed by one unmounted share are one problem, not four."""
    queue = Queue()
    for i in range(4):
        queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0},
                      id=f"j{i}"))
    shared = "Traceback (most recent call last):\nFileNotFoundError: /mnt/share is gone\n"
    runner = FakeRunner(default=1, log_text={f"j{i}": shared for i in range(4)})

    result = run_queue(queue, runner=runner, max_consecutive_failures=0, echo=False)

    summary = result.summary()
    assert "FileNotFoundError x4" in summary
    assert "/mnt/share is gone" in summary
    assert "(x4: j0, j1, j2, j3)" in summary
    assert summary.count("/mnt/share is gone") == 1, "the same failure printed 4 times"


def test_bad_settings_are_reported_as_a_configuration_failure(tmp_path, plate):
    """Exit code 2 from spacr-run means the run never started."""
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    runner = FakeRunner(codes={"a": 2}, log_text={"a": "error: src is empty.\n"})

    result = run_queue(queue, runner=runner, echo=False)

    assert queue.find("a").exit_code == 2
    assert "src is empty." in queue.find("a").error
    assert "ConfigurationError" in result.summary()


def test_progress_is_reported_incrementally(tmp_path, plate):
    """A GUI cannot show a seven-hour queue moving without this."""
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="b"))
    seen = []

    run_queue(queue, runner=FakeRunner(), on_progress=seen.append, echo=False)

    events = [p.event for p in seen]
    assert events[0] == "queue_started"
    assert events[-1] == "queue_finished"
    assert events.count("job_started") == 2
    assert events.count("job_finished") == 2
    started = [p for p in seen if p.event == "job_started"]
    assert [p.job_id for p in started] == ["a", "b"]
    assert [p.index for p in started] == [1, 2]
    assert all(p.total == 2 for p in started)


def test_a_progress_callback_that_raises_cannot_kill_the_queue(tmp_path, plate):
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))

    def _boom(_progress):
        raise RuntimeError("the GUI blew up")

    result = run_queue(queue, runner=FakeRunner(), on_progress=_boom, echo=False)
    assert result.succeeded


def test_a_runner_that_raises_is_a_job_failure_not_a_queue_crash(tmp_path, plate):
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="b"))

    def _explode(job, _sp, _lp):
        if job.id == "a":
            raise RuntimeError("the scheduler said no")
        Path(_lp).write_text("ok\n", encoding="utf-8")
        return 0

    result = run_queue(queue, runner=_explode, max_consecutive_failures=0, echo=False)
    assert queue.find("a").status == batch.STATUS_FAILED
    assert queue.find("b").status == batch.STATUS_SUCCESS
    assert result.failed


def test_stop_flag_halts_between_jobs(tmp_path, plate):
    """The GUI's Stop button never kills a job mid-write."""
    queue = Queue()
    for i in range(3):
        queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0},
                      id=f"j{i}"))
    runner = FakeRunner()

    result = run_queue(queue, runner=runner, stop_flag=lambda: True, echo=False)

    assert runner.ran == ["j0"]
    assert queue.find("j1").status == batch.STATUS_NOT_RUN
    assert "stopped by request" in result.stopped_reason


def test_on_error_must_be_continue_or_stop(plate):
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}))
    with pytest.raises(ValueError):
        run_queue(queue, on_error="panic", runner=FakeRunner(), echo=False)


# ---------------------------------------------------------------------------
# per-job logs
# ---------------------------------------------------------------------------


def test_every_job_writes_its_own_log_and_records_the_path(tmp_path, plate):
    """One interleaved log from twelve overnight jobs is unreadable."""
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="b"))
    runner = FakeRunner(log_text={"a": "A ran\n", "b": "B ran\n"})

    run_queue(queue, path=tmp_path / "q.json", runner=runner, echo=False)

    for job, text in ((queue.find("a"), "A ran"), (queue.find("b"), "B ran")):
        assert job.log_path
        assert os.path.isfile(job.log_path)
        assert Path(job.log_path).read_text(encoding="utf-8").strip() == text
    assert queue.find("a").log_path != queue.find("b").log_path


def test_an_inline_settings_dict_is_written_beside_the_log(tmp_path, plate):
    """Provenance: exactly what the job was given, beside what it printed."""
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    seen = {}

    def _runner(job, settings_path, log_path):
        seen["path"] = settings_path
        Path(log_path).write_text("ok\n", encoding="utf-8")
        return 0

    run_queue(queue, path=tmp_path / "q.json", runner=_runner, echo=False)

    assert seen["path"].endswith("a.settings.json")
    written = json.loads(Path(seen["path"]).read_text(encoding="utf-8"))
    assert written["src"] == plate
    assert written["cell_channel"] == 0


def test_a_settings_file_job_is_passed_the_file_itself(tmp_path, plate):
    settings = _settings_csv(tmp_path, "mask.csv", src=plate, cell_channel=0)
    queue = Queue()
    queue.add(Job(module="mask", settings=settings, id="a"))
    seen = {}

    def _runner(job, settings_path, log_path):
        seen["path"] = settings_path
        Path(log_path).write_text("ok\n", encoding="utf-8")
        return 0

    run_queue(queue, runner=_runner, echo=False)
    assert seen["path"] == settings


# ---------------------------------------------------------------------------
# persistence and resume
# ---------------------------------------------------------------------------


def test_state_is_persisted_after_every_transition(tmp_path, plate):
    """A machine that reboots mid-queue is resumed, not restarted.

    The runner reads the queue file back from disk while it runs, so what is
    asserted is what a crash would have left behind.
    """
    path = tmp_path / "q.json"
    queue = Queue()
    for i in range(3):
        queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0},
                      id=f"j{i}"))
    snapshots = []

    def _runner(job, _sp, log_path):
        Path(log_path).write_text("ok\n", encoding="utf-8")
        snapshots.append({j.id: j.status for j in load_queue(path).jobs})
        return 0

    run_queue(queue, path=path, runner=_runner, echo=False)

    assert snapshots[0] == {"j0": batch.STATUS_RUNNING, "j1": batch.STATUS_PENDING,
                            "j2": batch.STATUS_PENDING}
    assert snapshots[1] == {"j0": batch.STATUS_SUCCESS, "j1": batch.STATUS_RUNNING,
                            "j2": batch.STATUS_PENDING}
    assert snapshots[2] == {"j0": batch.STATUS_SUCCESS, "j1": batch.STATUS_SUCCESS,
                            "j2": batch.STATUS_RUNNING}
    final = load_queue(path)
    assert [j.status for j in final.jobs] == [batch.STATUS_SUCCESS] * 3
    assert all(j.log_path and j.started and j.finished for j in final.jobs)


def test_resume_picks_up_exactly_where_the_queue_stopped(tmp_path, plate):
    """The half of the night that already ran is not run again."""
    path = tmp_path / "q.json"
    other = _plate(tmp_path, "plate2")
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.add(Job(module="mask", settings={"src": other, "cell_channel": 0}, id="b"))
    queue.add(Job(module="mask", settings={"src": other, "cell_channel": 0}, id="c"))

    first = FakeRunner(codes={"b": 1})
    run_queue(queue, path=path, runner=first, on_error="stop", echo=False)
    assert first.ran == ["a", "b"]
    assert load_queue(path).find("c").status == batch.STATUS_NOT_RUN

    second = FakeRunner()
    result = resume_queue(path, runner=second, echo=False)

    assert second.ran == ["c"], "resume re-ran a job that had already settled"
    assert result.queue.find("a").status == batch.STATUS_SUCCESS
    assert result.queue.find("b").status == batch.STATUS_FAILED
    assert result.queue.find("c").status == batch.STATUS_SUCCESS


def test_resume_reruns_the_job_that_was_running_when_the_machine_died(tmp_path, plate):
    """Half a mask run is not a result, so it runs again."""
    path = tmp_path / "q.json"
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="b"))
    queue.jobs[0].status = batch.STATUS_SUCCESS
    queue.jobs[1].status = batch.STATUS_RUNNING
    queue.jobs[1].started = batch._now_iso()
    save_queue(queue, path)

    runner = FakeRunner()
    resume_queue(path, runner=runner, echo=False)

    assert runner.ran == ["b"]
    assert load_queue(path).find("b").status == batch.STATUS_SUCCESS


def test_resume_can_retry_failures(tmp_path, plate):
    """retry_failed also un-skips whatever was skipped because of them."""
    _merged(plate)
    path = tmp_path / "q.json"
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.add(Job(module="measure", settings={"src": plate, "cell_mask_dim": 4},
                  id="b", depends_on=["a"]))
    run_queue(queue, path=path, runner=FakeRunner(codes={"a": 1}),
              max_consecutive_failures=0, echo=False)
    assert queue.find("b").status == batch.STATUS_SKIPPED

    runner = FakeRunner()
    result = resume_queue(path, retry_failed=True, runner=runner, echo=False)

    assert runner.ran == ["a", "b"]
    assert result.ok is True


def test_running_without_a_path_still_works_but_says_where_the_logs_went(tmp_path,
                                                                        plate):
    """No queue file means no resume — but the run itself is unaffected."""
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    result = run_queue(queue, runner=FakeRunner(), echo=False)
    assert result.succeeded
    assert os.path.isdir(result.log_dir)
    assert result.log_dir in result.summary()


def test_the_queue_stamps_its_own_verdict_next_to_the_queue_file(tmp_path, plate):
    """A queue is a ledger of ledgers, and it stamps like one."""
    from spacr.errors import read_run_status

    path = tmp_path / "night.queue.json"
    queue = Queue(name="night")
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="b"))

    run_queue(queue, path=path, runner=FakeRunner(codes={"b": 1}),
              max_consecutive_failures=0, echo=False)

    records = read_run_status(path)
    assert records, "the queue file was never stamped"
    assert records[-1]["n_attempted"] == 2
    assert records[-1]["n_failed"] == 1
    assert records[-1]["status"] == "partial"


# ---------------------------------------------------------------------------
# run_status: a job that exits 0 is not necessarily a success
# ---------------------------------------------------------------------------


def test_a_job_that_exits_zero_but_stamped_partial_is_reported_as_partial(tmp_path,
                                                                         plate):
    """40 silently skipped fields exit 0 and look like a clean run.

    The queue is the last place that can say otherwise, so it reads each job's
    own run_status stamp back and reports it.
    """
    measurements = Path(plate) / "measurements"
    measurements.mkdir(parents=True, exist_ok=True)
    db = measurements / "measurements.db"

    def _stamp_partial(job, _sp, log_path):
        Path(log_path).write_text("done\n", encoding="utf-8")
        ledger = RunLedger("measure_crop")
        for i in range(344):
            ledger.record_success(f"well{i}")
        for i in range(40):
            ledger.record_failure(f"bad{i}", exc=ValueError("unreadable field"))
        ledger.stamp(db)
        return 0

    queue = Queue(name="partial")
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))

    result = run_queue(queue, runner=_stamp_partial, echo=False)

    job = queue.find("a")
    assert job.status == batch.STATUS_SUCCESS      # it really did exit 0
    assert job.exit_code == 0
    assert job.is_partial is True
    assert job.run_status["n_failed"] == 40
    assert job.run_status["n_succeeded"] == 344
    assert job.run_status["status"] == "partial"

    summary = result.summary()
    assert "PARTIAL" in summary
    assert "344" in summary and "384" in summary
    assert result.ok is False
    assert [j.id for j in result.partial] == ["a"]


def test_only_this_jobs_stamps_are_counted(tmp_path, plate):
    """A measurements.db accumulates a row per stage; the older ones are not ours."""
    measurements = Path(plate) / "measurements"
    measurements.mkdir(parents=True, exist_ok=True)
    db = measurements / "measurements.db"
    old = RunLedger("an earlier run")
    old.record_failure("well1", exc=ValueError("last week's problem"))
    old.stamp(db)

    def _clean(job, _sp, log_path):
        Path(log_path).write_text("done\n", encoding="utf-8")
        ledger = RunLedger("measure_crop")
        ledger.record_success("well1")
        ledger.stamp(db)
        return 0

    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    result = run_queue(queue, runner=_clean, echo=False)

    assert queue.find("a").run_status["n_failed"] == 0
    assert queue.find("a").is_partial is False
    assert result.ok is True


def test_a_job_that_stamps_nothing_reports_no_information(tmp_path, plate):
    """Never stamped is not the same as verified clean, and is not faked."""
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}, id="a"))
    run_queue(queue, runner=FakeRunner(), echo=False)
    assert queue.find("a").run_status is None
    assert queue.find("a").is_partial is False


# ---------------------------------------------------------------------------
# the summary is the deliverable
# ---------------------------------------------------------------------------


def test_the_summary_covers_ran_failed_skipped_and_how_long(tmp_path, plate):
    _merged(plate)
    other = _plate(tmp_path, "plate2")
    queue = Queue(name="overnight")
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0},
                  id="mask-a", label="plate1 mask"))
    queue.add(Job(module="measure", settings={"src": plate, "cell_mask_dim": 4},
                  id="measure-a", label="plate1 measure", depends_on=["mask-a"]))
    queue.add(Job(module="mask", settings={"src": other, "cell_channel": 0},
                  id="mask-b", label="plate2 mask"))
    queue.add(Job(module="mask", settings={"src": other, "cell_channel": 0},
                  id="mask-c", label="plate2 mask again"))

    result = run_queue(queue, path=tmp_path / "q.json",
                       runner=FakeRunner(codes={"mask-a": 1},
                                         log_text={"mask-a": "RuntimeError: CUDA OOM\n"}),
                       max_consecutive_failures=0, echo=False)
    summary = result.summary()

    assert "overnight" in summary
    for label in ("plate1 mask", "plate1 measure", "plate2 mask"):
        assert label in summary
    assert "RuntimeError x1" in summary
    assert "CUDA OOM" in summary
    assert "Skipped" in summary
    assert "measure-a" in summary
    assert queue.find("mask-b").duration_s is not None
    assert result.duration_s is not None


def test_plan_describes_the_run_order_and_the_concurrency_decision(mask_measure_queue):
    text = plan(mask_measure_queue)
    assert "mask-1" in text and "measure-1" in text
    assert "one at a time" in text
    assert "spacr.core.preprocess_generate_masks()" in text
    assert "after mask-1" in text
    assert "pre-flight" in text


def test_fmt_duration_reads_like_an_overnight_report():
    assert batch.fmt_duration(None) == "—"
    assert batch.fmt_duration(42.06) == "42.1s"
    assert batch.fmt_duration(432) == "7m 12s"
    assert batch.fmt_duration(3600 * 7 + 60 * 41) == "7h 41m"


# ---------------------------------------------------------------------------
# a queued job is a spacr-run invocation
# ---------------------------------------------------------------------------


def test_job_command_is_a_spacr_run_invocation(tmp_path, plate):
    settings = _settings_csv(tmp_path, "mask.csv", src=plate, cell_channel=0)
    job = Job(module="masks", settings=settings, id="a",     # an alias, on purpose
              overrides=["verbose=True"])
    cmd = batch.job_command(job, settings)

    assert cmd[0] == sys.executable
    assert cmd[1:4] == ["-m", "spacr.cli", "mask"]           # alias resolved
    assert cmd[4:6] == ["--settings", settings]
    assert cmd[6:] == ["--set", "verbose=True"]


def test_overrides_may_be_written_as_a_mapping(tmp_path, plate):
    """The hand-editable file allows {"diameter": 30}; --set wants a string."""
    job = Job(module="mask", settings={"src": plate}, overrides={"verbose": True})
    assert job.override_args == ["verbose=True"]
    assert job.to_dict()["overrides"] == ["verbose=True"]


def test_duplicating_a_job_gives_a_fresh_never_run_copy(tmp_path, plate):
    queue = Queue()
    first = queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0},
                          id="a"))
    first.status = batch.STATUS_SUCCESS
    first.exit_code = 0
    clone = first.copy()
    queue.add(clone)

    assert clone.id == "mask-1" and clone.id != first.id
    assert clone.status == batch.STATUS_PENDING
    assert clone.exit_code is None
    assert clone.settings == first.settings


def test_subprocess_runner_honours_the_cli_exit_code_contract(tmp_path):
    """The default runner really is ``spacr-run``, and 2 really means 2.

    Nothing is segmented: the settings point at a folder that does not exist,
    so the CLI's own pre-flight refuses to start and exits 2 long before any
    pipeline — or torch — is imported.
    """
    settings = _settings_csv(tmp_path, "mask.csv",
                             src=str(tmp_path / "does-not-exist"), cell_channel=0)
    job = Job(module="mask", settings=settings, id="a")
    log = tmp_path / "a.log"

    code = batch.subprocess_runner(job, settings, str(log))

    assert code == 2, log.read_text(encoding="utf-8")[-2000:]
    text = log.read_text(encoding="utf-8")
    assert "-m spacr.cli mask" in text            # the command is in the log
    assert "exit code 2" in text


# ---------------------------------------------------------------------------
# the import must stay cheap
# ---------------------------------------------------------------------------

_HEAVY = ("torch", "cellpose", "PySide6", "PyQt5", "PyQt6", "tkinter")


def test_import_pulls_no_torch_qt_or_cellpose():
    """A queue file must be readable and validatable without the pipeline stack.

    Checked in a fresh interpreter so nothing another test imported can mask a
    regression. This is what lets a login node plan a night's work.
    """
    code = (
        "import json, sys\n"
        "import spacr.batch\n"
        "print(json.dumps({m: (m in sys.modules) for m in %r}))\n" % (_HEAVY,)
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT),
             "HOME": "/tmp", "MPLBACKEND": "Agg"})
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    offenders = [m for m, present in loaded.items() if present]
    assert not offenders, f"spacr.batch imported heavy modules: {offenders}"


def test_validating_a_queue_pulls_no_torch_or_cellpose(tmp_path, plate):
    """Same again, for the part a user actually runs before bed."""
    path = tmp_path / "q.json"
    queue = Queue()
    queue.add(Job(module="mask", settings={"src": plate, "cell_channel": 0}))
    save_queue(queue, path)
    code = (
        "import json, sys\n"
        "from spacr.batch import load_queue, validate_queue, plan\n"
        f"q = load_queue({str(path)!r})\n"
        "assert not [p for p in validate_queue(q) if p.is_error]\n"
        "assert 'one at a time' in plan(q)\n"
        "print(json.dumps({m: (m in sys.modules) for m in %r}))\n" % (_HEAVY,)
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT),
             "HOME": "/tmp", "MPLBACKEND": "Agg"})
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    loaded = json.loads(proc.stdout.strip().splitlines()[-1])
    assert not [m for m, present in loaded.items() if present]
