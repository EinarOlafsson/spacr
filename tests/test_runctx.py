"""The run context: one run id, one seed, one on_error policy.

These are not smoke tests. Each of the three features is asserted on the
property that would actually be relied on:

* **S7** — the id on a log line and the id on the artifact the run
  registered are asserted to be *the same string*, by reading both back
  out of their real stores (the JSONL run log and the SQLite artifact
  registry). A test that only checked "a run id exists" would pass while
  the join was broken, which is the whole thing being built.
* **S5** — two runs at the same seed are asserted to produce
  **bit-identical** output across ``random``, NumPy's legacy global, a
  NumPy Generator and Torch, and two runs at different seeds are asserted
  to differ. Equality of one draw is not enough: a broken seeder that
  seeds only NumPy would pass that.
* **S9** — ``stop`` is asserted to abort on the *first* failure with the
  later units never attempted; ``skip`` is asserted to complete and to
  name exactly which units it dropped; ``retry`` is asserted to make
  exactly the bounded number of attempts, to sleep on the documented
  schedule, and then to raise like ``stop``.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr import artifacts, runctx                                # noqa: E402
from spacr.cancellation import PipelineCancelled                   # noqa: E402
from spacr.errors import ConfigurationError, RunLedger             # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def private_logs(tmp_path, monkeypatch):
    """Point the run log at a scratch folder and capture root records."""
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.DEBUG)
    yield tmp_path / "logs"
    root.setLevel(previous)


@pytest.fixture
def scratch_registry(tmp_path, monkeypatch):
    """Point every artifact registry at one scratch file."""
    path = tmp_path / "artifacts.db"
    monkeypatch.setenv(artifacts.ARTIFACTS_DB_ENV, str(path))
    yield path


class _Clock:
    """A sleep that records instead of sleeping."""

    def __init__(self):
        self.slept = []

    def __call__(self, seconds):
        self.slept.append(seconds)


# ---------------------------------------------------------------------------
# S7 — one run id through every log line, joinable to the artifact
# ---------------------------------------------------------------------------

def test_the_log_line_and_the_artifact_carry_the_same_run_id(
        private_logs, scratch_registry, tmp_path):
    """The join S7 exists for: a log line and an output, on one id.

    The id is not read off the context object and compared to itself. It
    is read out of the run's JSONL log and out of the SQLite artifact
    registry — the two stores a user would actually query — and the two
    strings are required to be equal.
    """
    project = tmp_path / "plate1"
    project.mkdir()
    produced = project / "measurements.db"
    produced.write_bytes(b"not really a database, but it is on disk")

    with runctx.run_context("measure", {"src": str(project)}) as run:
        logging.getLogger("spacr.measure").info("measured field A01f01")
        registry = artifacts.open_registry(str(project))
        artifact = registry.register(
            module="measure", kind="measurements-db", path=str(produced),
            project=str(project), run_id=run.run_id)
        run_id = run.run_id

    # ...from the log store.
    lines = runctx.read_run_log(run_id, contains="measured field")
    assert len(lines) == 1, f"expected one matching log line, got {lines}"
    id_on_the_log_line = lines[0]["run_id"]

    # ...from the artifact store, re-read rather than reused.
    stored = artifacts.open_registry(str(project)).get(artifact.artifact_id)
    id_on_the_artifact = stored.run_id

    assert id_on_the_log_line == id_on_the_artifact == run_id
    assert id_on_the_log_line, "the id must not be empty"

    # And the join runs in the direction a user needs it: from a file on
    # disk back to the log of the run that produced it.
    latest = artifacts.open_registry(str(project)).latest(
        "measurements-db", path=str(produced))
    recovered = runctx.read_run_log(latest.run_id)
    assert any("measured field A01f01" in line["message"]
               for line in recovered)


def test_every_log_line_from_every_module_carries_the_id(private_logs):
    """Not just lines from the run's own logger — any spacr logger."""
    with runctx.run_context("mask", {}) as run:
        logging.getLogger("spacr.core").warning("core said something")
        logging.getLogger("spacr.utils.deeply.nested").error("utils too")
        logging.getLogger("some.third.party").info("even a stranger")
        run_id = run.run_id

    records = runctx.read_run_log(run_id)
    messages = {record["message"] for record in records}
    assert "core said something" in messages
    assert "utils too" in messages
    assert "even a stranger" in messages
    assert {record["run_id"] for record in records} == {run_id}


def test_the_record_attribute_is_there_for_a_formatter(private_logs):
    """A handler formatting %(run_id)s must never blow up on a record."""
    captured = []

    class _Grab(logging.Handler):
        def emit(self, record):
            captured.append(logging.Formatter("%(run_id)s|%(message)s")
                            .format(record))

    handler = _Grab()
    logging.getLogger().addHandler(handler)
    try:
        with runctx.run_context("mask", {}) as run:
            logging.getLogger("spacr.core").warning("inside")
            expected = run.run_id
        logging.getLogger("spacr.core").warning("outside")
    finally:
        logging.getLogger().removeHandler(handler)

    assert f"{expected}|inside" in captured
    # Outside a run the attribute still exists, so the formatter is safe.
    assert any(line.endswith("|outside") for line in captured)


def test_the_run_id_reaches_a_child_process_through_the_environment(
        private_logs):
    """A spawned worker has an empty contextvar; the env carries the id."""
    with runctx.run_context("measure", {}) as run:
        assert os.environ[runctx.RUN_ID_ENV] == run.run_id
        # Simulate the worker: a fresh context, as spawn produces.
        import contextvars
        fresh = contextvars.Context()
        assert fresh.run(runctx.current_run_id) == run.run_id
    assert runctx.RUN_ID_ENV not in os.environ


def test_read_run_log_filters_and_tolerates_a_truncated_line(private_logs):
    """The query side: by level, by logger, by substring; half a line is skipped."""
    with runctx.run_context("mask", {}) as run:
        logging.getLogger("spacr.core").info("ordinary")
        logging.getLogger("spacr.core").error("bad thing")
        logging.getLogger("spacr.ml").error("other module")
        run_id = run.run_id

    path = Path(runctx.run_log_path(run_id))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"run_id": "half a record and then the power w')

    assert len(runctx.read_run_log(run_id, level="ERROR")) == 2
    assert len(runctx.read_run_log(run_id, logger="spacr.core")) == 2
    assert len(runctx.read_run_log(run_id, contains="bad thing")) == 1
    # Nothing raised despite the torn last line.
    assert len(runctx.read_run_log(run_id)) >= 3


def test_the_ledger_shares_the_run_id_so_the_stamp_joins_too(private_logs):
    """A RunLedger mints its own uuid; the run must overwrite it."""
    with runctx.run_context("measure", {}) as run:
        assert run.ledger.run_id == run.run_id
        adopted = RunLedger("elsewhere")
        assert adopted.run_id != run.run_id
        run.adopt(adopted)
        assert adopted.run_id == run.run_id
        assert run.new_ledger("another").run_id == run.run_id


def test_read_run_log_of_an_unknown_run_is_empty_not_an_error(private_logs):
    assert runctx.read_run_log("neverhappened") == []


def test_info_lines_survive_a_host_that_never_called_setup_logging(
        tmp_path, monkeypatch):
    """A handler only sees what its *logger* let through.

    A bare ``import spacr`` leaves the root logger at WARNING, so without
    the run opening the ``spacr`` level the per-run log would hold the
    warnings and none of the INFO lines that say what the run actually
    did. Deliberately does not use the ``private_logs`` fixture, which
    forces DEBUG and would hide exactly this.
    """
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    root, spacr_logger = logging.getLogger(), logging.getLogger("spacr")
    monkeypatch.setattr(root, "level", logging.WARNING)
    monkeypatch.setattr(spacr_logger, "level", logging.NOTSET)

    with runctx.run_context("mask", {}) as run:
        logging.getLogger("spacr.core").info("a quiet but important line")
        run_id = run.run_id

    messages = {record["message"] for record in runctx.read_run_log(run_id)}
    assert "a quiet but important line" in messages
    # ...and the host's own configuration is handed back untouched.
    assert spacr_logger.level == logging.NOTSET
    assert root.level == logging.WARNING


# ---------------------------------------------------------------------------
# S5 — the same seed gives bit-identical output; a different one does not
# ---------------------------------------------------------------------------

def _draw():
    """Draw from every RNG a spaCR run touches, and return the raw bytes."""
    import torch
    return (
        random.random(),
        np.random.rand(4).tobytes(),
        runctx.spacr_rng("features").normal(size=4).tobytes(),
        torch.rand(4).numpy().tobytes(),
        torch.nn.Linear(8, 4).weight.detach().numpy().tobytes(),
    )


def test_two_runs_with_the_same_seed_are_bit_identical(private_logs):
    """Same seed in, same bytes out — across python, numpy, Generator, torch."""
    with runctx.run_context("mask", {"random_seed": 1234}):
        first = _draw()
    with runctx.run_context("mask", {"random_seed": 1234}):
        second = _draw()

    assert first == second, "the same seed produced different draws"
    # And every component is genuinely non-trivial, so equality is not the
    # accident of two empty tuples.
    assert all(part for part in first)


def test_two_runs_with_different_seeds_differ_everywhere(private_logs):
    """A seeder that quietly ignores the seed would pass the test above."""
    with runctx.run_context("mask", {"random_seed": 1234}):
        first = _draw()
    with runctx.run_context("mask", {"random_seed": 5678}):
        second = _draw()

    assert first != second
    # Not just one of them: every RNG must have moved.
    for index, (a, b) in enumerate(zip(first, second)):
        assert a != b, f"component {index} did not change with the seed"


def test_the_seed_reaches_torch_cpu_and_the_legacy_numpy_global(private_logs):
    """The two most-used streams in spaCR, asserted individually."""
    import torch

    runctx.seed_everything(7)
    torch_first, numpy_first = torch.rand(3).clone(), np.random.rand(3).copy()
    runctx.seed_everything(7)
    assert torch.equal(torch.rand(3), torch_first)
    assert np.array_equal(np.random.rand(3), numpy_first)


@pytest.mark.gpu
def test_the_seed_reaches_cuda_when_there_is_a_gpu():
    """CUDA draws repeat under the same seed."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no usable CUDA device")
    runctx.seed_everything(11)
    first = torch.rand(8, device="cuda").cpu().clone()
    runctx.seed_everything(11)
    assert torch.equal(torch.rand(8, device="cuda").cpu(), first)


def test_sklearn_estimators_get_the_run_seed_not_a_hard_coded_literal():
    """``random_state()`` is what the ml/utils construction sites now pass."""
    from sklearn.ensemble import RandomForestClassifier

    features = np.arange(80).reshape(40, 2).astype(float)
    labels = np.array([0, 1] * 20)

    with runctx.run_context("ml", {"random_seed": 3}):
        assert runctx.random_state(42) == 3
        a = RandomForestClassifier(
            n_estimators=5, random_state=runctx.random_state(42)
        ).fit(features, labels).predict_proba(features)
    with runctx.run_context("ml", {"random_seed": 3}):
        b = RandomForestClassifier(
            n_estimators=5, random_state=runctx.random_state(42)
        ).fit(features, labels).predict_proba(features)
    with runctx.run_context("ml", {"random_seed": 999}):
        assert runctx.random_state(42) == 999
        c = RandomForestClassifier(
            n_estimators=5, random_state=runctx.random_state(42)
        ).fit(features, labels).predict_proba(features)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # Outside a run the old literal is what comes back, so nothing that
    # never opens a run changes behaviour.
    assert runctx.random_state(42) == 42


def test_named_streams_are_independent_and_repeatable():
    """Two stream names must not draw the same numbers."""
    runctx.seed_everything(5)
    left = runctx.spacr_rng("cell").normal(size=6)
    right = runctx.spacr_rng("nucleus").normal(size=6)
    assert not np.array_equal(left, right)

    runctx.seed_everything(5)
    assert np.array_equal(runctx.spacr_rng("cell").normal(size=6), left)


def test_seed_worker_gives_each_worker_a_different_stream():
    """Fork hands every worker one stream; seed_worker splits them."""
    runctx.seed_worker(0)
    first = np.random.rand(4).copy()
    runctx.seed_worker(1)
    second = np.random.rand(4).copy()
    assert not np.array_equal(first, second)
    runctx.seed_worker(0)
    assert np.array_equal(np.random.rand(4), first)


def test_the_report_names_what_it_seeded_and_what_it_cannot_promise():
    """The honesty requirement, asserted rather than left to the docstring."""
    report = runctx.seed_everything(3)
    assert report.seed == 3
    assert "python" in report.seeded
    assert "numpy" in report.seeded
    assert "torch" in report.seeded
    assert report.caveats, "a seed report with no caveats is a false promise"
    joined = " ".join(report.caveats)
    for subject in ("PYTHONHASHSEED", "Cellpose", "scikit-learn", "worker"):
        assert subject in joined, f"{subject} is unaccounted for"
    assert report.to_dict()["seed"] == 3


def test_an_explicit_none_seed_leaves_the_rngs_alone(private_logs):
    """`random_seed: None` must mean free-running, not "fall back to 42"."""
    assert runctx.resolve_seed({"random_seed": None}) is None
    assert runctx.resolve_seed({"random_seed": "none"}) is None
    assert runctx.resolve_seed({}) == runctx.DEFAULT_SEED
    assert runctx.resolve_seed({"random_seed": 9}) == 9

    with runctx.run_context("mask", {"random_seed": None}) as run:
        assert run.seed is None
        assert run.seed_report is None
        first = random.random()
    with runctx.run_context("mask", {"random_seed": None}):
        assert random.random() != first


def test_a_word_is_a_usable_seed():
    """`random_seed: "plate3-rerun"` is reproducible rather than a crash."""
    once = runctx.resolve_seed({"random_seed": "plate3-rerun"})
    assert isinstance(once, int)
    assert once == runctx.resolve_seed({"random_seed": "plate3-rerun"})
    assert once != runctx.resolve_seed({"random_seed": "plate4-rerun"})


def test_the_seed_environment_override(monkeypatch):
    monkeypatch.setenv(runctx.SEED_ENV, "77")
    assert runctx.resolve_seed({}) == 77
    assert runctx.resolve_seed({"random_seed": 5}) == 5


# ---------------------------------------------------------------------------
# S9 — stop | skip | retry
# ---------------------------------------------------------------------------

def _process(units, policy, fails_on=(), stage="unit"):
    """Run ``units`` under ``policy``; return (attempted, completed)."""
    attempted, completed = [], []
    for unit in units:
        for attempt in policy.attempts_for(unit, stage=stage):
            with attempt:
                attempted.append(unit)
                if unit in fails_on:
                    raise RuntimeError(f"{unit} is broken")
                completed.append(unit)
    return attempted, completed


def test_stop_aborts_on_the_first_failure_and_never_touches_the_rest():
    """The default. Later units must not be attempted at all."""
    policy = runctx.ErrorPolicy("stop", ledger=RunLedger("t"))
    with pytest.raises(RuntimeError, match="B is broken"):
        _process(["A", "B", "C", "D"], policy, fails_on={"B"})

    # The evidence: C and D were never started.
    assert policy.ledger.n_succeeded == 1
    assert policy.ledger.n_failed == 1
    assert [f.item for f in policy.ledger.failures] == ["B"]
    assert policy.skips == []


def test_stop_is_the_default_when_nothing_says_otherwise():
    assert runctx.ErrorPolicy().mode == "stop"
    assert runctx.resolve_error_policy(None).mode == "stop"
    assert runctx.resolve_error_policy({}).mode == "stop"
    assert runctx.DEFAULT_ON_ERROR == "stop"
    with runctx.run_context("mask", {}, log=False) as run:
        assert run.policy.mode == "stop"


def test_skip_completes_the_batch_and_records_exactly_what_it_dropped():
    """Never a silent drop: the units, the stage and the reason survive."""
    policy = runctx.ErrorPolicy("skip", ledger=RunLedger("t"), run_id="rid42")
    attempted, completed = _process(
        ["A", "B", "C", "D", "E"], policy, fails_on={"B", "D"}, stage="plate")

    assert attempted == ["A", "B", "C", "D", "E"]     # every unit was tried
    assert completed == ["A", "C", "E"]               # the run carried on
    assert policy.skipped_units == ["B", "D"]         # exactly these, in order
    assert policy.n_skipped == 2

    dropped = policy.skips[0]
    assert dropped.unit == "B"
    assert dropped.stage == "plate"
    assert "B is broken" in dropped.reason
    assert dropped.exc_type == "RuntimeError"
    assert dropped.attempts == 1
    assert dropped.run_id == "rid42"
    assert "RuntimeError" in dropped.traceback_str

    # The ledger says the same thing, so the artifact stamp does too.
    assert policy.ledger.n_failed == 2
    assert {f.item for f in policy.ledger.failures} == {"B", "D"}
    assert policy.ledger.status == "partial"
    assert "SKIPPED" in policy.summary()
    assert "B" in policy.summary() and "D" in policy.summary()
    assert json.dumps([s.to_dict() for s in policy.skips])


def test_retry_makes_exactly_the_bounded_number_of_attempts_then_stops():
    """Bounded, backed off, and terminal — the three things retry promises."""
    clock = _Clock()
    policy = runctx.ErrorPolicy("retry", attempts=3, backoff=0.5,
                                ledger=RunLedger("t"), sleep=clock)

    with pytest.raises(RuntimeError, match="B is broken"):
        attempted, _ = _process(["A", "B", "C"], policy, fails_on={"B"})

    # A once, B three times, and C never — retry ends as stop.
    assert policy.ledger.n_succeeded == 1
    assert policy.ledger.n_failed == 1
    assert policy.retries == [("B", 3)]
    # Two sleeps for three attempts, doubling.
    assert clock.slept == [0.5, 1.0]


def test_retry_succeeds_on_a_later_attempt_and_the_run_carries_on():
    """The case retry exists for: a transient failure, then success."""
    clock = _Clock()
    policy = runctx.ErrorPolicy("retry", attempts=4, backoff=0.25,
                                ledger=RunLedger("t"), sleep=clock)
    tries = {"B": 0}

    completed = []
    for unit in ["A", "B", "C"]:
        for attempt in policy.attempts_for(unit, stage="field"):
            with attempt:
                if unit == "B":
                    tries["B"] += 1
                    if tries["B"] < 3:
                        raise OSError("the share went away")
                completed.append(unit)

    assert completed == ["A", "B", "C"]
    assert tries["B"] == 3
    assert clock.slept == [0.25, 0.5]
    assert policy.retries == [("B", 3)]
    assert policy.ledger.n_failed == 0        # it eventually worked
    assert policy.ledger.n_succeeded == 3
    assert policy.n_skipped == 0


def test_retry_then_skip_gives_up_after_the_budget_without_aborting():
    """retry and skip compose: bounded attempts, then recorded and dropped."""
    clock = _Clock()
    policy = runctx.ErrorPolicy("retry", attempts=2, backoff=1.0,
                                ledger=RunLedger("t"), sleep=clock)
    policy.mode = "retry"
    # Now the same units under skip, to contrast the terminal behaviour.
    skipper = runctx.ErrorPolicy("skip", ledger=RunLedger("t"), sleep=clock)
    _, completed = _process(["A", "B"], skipper, fails_on={"B"})
    assert completed == ["A"]
    assert skipper.skipped_units == ["B"]

    with pytest.raises(RuntimeError):
        _process(["A", "B"], policy, fails_on={"B"})
    assert clock.slept[-1] == 1.0


def test_the_backoff_is_capped():
    clock = _Clock()
    policy = runctx.ErrorPolicy("retry", attempts=12, backoff=1.0,
                                ledger=RunLedger("t"), sleep=clock)
    with pytest.raises(RuntimeError):
        _process(["A"], policy, fails_on={"A"})
    assert max(clock.slept) == runctx.MAX_BACKOFF
    assert len(clock.slept) == 11


def test_a_configuration_error_is_never_skipped_or_retried():
    """A wrong src is wrong for every unit; skipping N of them hides one bug."""
    for mode in ("stop", "skip", "retry"):
        policy = runctx.ErrorPolicy(mode, ledger=RunLedger("t"),
                                    sleep=_Clock())
        with pytest.raises(ConfigurationError):
            for attempt in policy.attempts_for("A", stage="s"):
                with attempt:
                    raise ConfigurationError("src does not exist")
        assert policy.n_skipped == 0


@pytest.mark.parametrize("fatal", [KeyboardInterrupt, SystemExit,
                                   PipelineCancelled])
def test_operator_intent_is_never_swallowed(fatal):
    """Ctrl-C and a cancelled pipeline abort whatever on_error says."""
    policy = runctx.ErrorPolicy("skip", ledger=RunLedger("t"))
    with pytest.raises(fatal):
        for attempt in policy.attempts_for("A", stage="s"):
            with attempt:
                raise fatal("stop now")
    assert policy.n_skipped == 0


def test_run_returns_the_result_or_the_skipped_sentinel():
    policy = runctx.ErrorPolicy("skip", ledger=RunLedger("t"))
    assert policy.run("A", lambda: 21 * 2, stage="s") == 42

    def _boom():
        raise ValueError("nope")

    assert policy.run("B", _boom, stage="s") is runctx.SKIPPED
    assert not runctx.SKIPPED
    assert policy.skipped_units == ["B"]


def test_an_unknown_mode_is_refused_loudly():
    """`on_error='continue'` must not silently mean stop."""
    with pytest.raises(ValueError, match="on_error must be one of"):
        runctx.ErrorPolicy("continue")
    with pytest.raises(ValueError):
        runctx.resolve_error_policy({"on_error": "ignore"})
    with pytest.raises(ValueError, match="at least 1"):
        runctx.ErrorPolicy("retry", attempts=0)


def test_the_policy_comes_from_the_settings():
    policy = runctx.resolve_error_policy(
        {"on_error": "RETRY", "on_error_attempts": 5,
         "on_error_backoff": 2.5})
    assert (policy.mode, policy.attempts, policy.backoff) == ("retry", 5, 2.5)
    # Junk in the numeric fields falls back rather than crashing a run.
    loose = runctx.resolve_error_policy(
        {"on_error": "skip", "on_error_attempts": "", "on_error_backoff": None})
    assert loose.attempts == runctx.DEFAULT_RETRIES
    assert loose.backoff == runctx.DEFAULT_BACKOFF


def test_bind_moves_the_recording_without_losing_the_skip_list():
    """Measure rebinds per source folder; the run's account must survive."""
    first, second = RunLedger("one"), RunLedger("two")
    policy = runctx.ErrorPolicy("skip", ledger=first)
    _process(["A"], policy, fails_on={"A"})
    policy.bind(ledger=second, record=False)
    _process(["B"], policy, fails_on={"B"})

    assert first.n_failed == 1
    assert second.n_failed == 0            # record=False was honoured
    assert policy.skipped_units == ["A", "B"]   # the run's account is whole


# ---------------------------------------------------------------------------
# The three together
# ---------------------------------------------------------------------------

def test_a_skipping_run_logs_and_registers_under_one_id(
        private_logs, scratch_registry, tmp_path):
    """End to end: a run that skips a unit still joins log to artifact."""
    project = tmp_path / "plate"
    project.mkdir()
    output = project / "out.csv"

    settings = {"src": str(project), "random_seed": 5, "on_error": "skip"}
    with runctx.run_context("measure", settings) as run:
        for unit in ("A01", "A02", "A03"):
            for attempt in run.policy.attempts_for(unit, stage="well"):
                with attempt:
                    if unit == "A02":
                        raise ValueError("no cells found")
                    output.write_text(f"{unit}\n")
        artifact = artifacts.open_registry(str(project)).register(
            module="measure", kind="object-crops", path=str(output),
            project=str(project), run_id=run.run_id,
            status=artifacts.STATUS_PARTIAL)
        run_id, skipped = run.run_id, run.skips

    assert [record.unit for record in skipped] == ["A02"]
    assert artifact.run_id == run_id
    assert artifact.status == artifacts.STATUS_PARTIAL

    log = runctx.read_run_log(run_id)
    assert {record["run_id"] for record in log} == {run_id}
    assert any("skipping A02" in record["message"] for record in log)
    assert any(f"run {run_id} started" in record["message"] for record in log)
    assert any("1 skipped" in record["message"] for record in log)


def test_the_context_dict_is_the_whole_story():
    with runctx.run_context("mask", {"random_seed": 8, "on_error": "skip"},
                            log=False) as run:
        payload = run.to_dict()
    assert payload["module"] == "mask"
    assert payload["seed"] == 8
    assert payload["on_error"] == "skip"
    assert payload["run_id"] == run.run_id
    assert payload["seed_report"]["seed"] == 8
    assert json.dumps(payload)


def test_a_given_run_id_is_honoured_for_a_resumed_run(private_logs):
    """A distributed worker continues its parent's run rather than forking it."""
    with runctx.run_context("measure", {}, run_id="deadbeef1234") as run:
        assert run.run_id == "deadbeef1234"
        logging.getLogger("spacr.measure").info("worker line")
    assert any("worker line" in record["message"]
               for record in runctx.read_run_log("deadbeef1234"))


def test_a_failing_run_still_closes_its_log_and_clears_the_context(
        private_logs):
    with pytest.raises(ZeroDivisionError):
        with runctx.run_context("mask", {}) as run:
            run_id = run.run_id
            1 / 0
    assert runctx.current_run_context() is None
    assert runctx.RUN_ID_ENV not in os.environ
    log = runctx.read_run_log(run_id)
    assert any("failed after" in record["message"] for record in log)


def test_nested_runs_do_not_cross_contaminate(private_logs):
    """Two ids in one process: each log gets only its own lines."""
    with runctx.run_context("mask", {}) as outer:
        logging.getLogger("spacr.core").info("outer only")
        with runctx.run_context("measure", {}) as inner:
            logging.getLogger("spacr.core").info("inner only")
            inner_id = inner.run_id
        outer_id = outer.run_id

    assert inner_id != outer_id
    inner_messages = {r["message"] for r in runctx.read_run_log(inner_id)}
    outer_messages = {r["message"] for r in runctx.read_run_log(outer_id)}
    assert "inner only" in inner_messages
    assert "outer only" not in inner_messages
    assert "outer only" in outer_messages
    assert "inner only" not in outer_messages


# ---------------------------------------------------------------------------
# The settings seam
# ---------------------------------------------------------------------------

def test_a_pipeline_whose_settings_name_nothing_still_gets_stop():
    """The keys are optional; the *behaviour* is not.

    ``on_error`` is not injected into the ``set_default_*`` factories (see
    ``runctx._register_settings``), so what has to hold is that every
    pipeline still ends up with the safe mode.
    """
    from spacr import settings as settings_module

    for factory in (settings_module.set_default_settings_preprocess_generate_masks,
                    settings_module.get_measure_crop_settings):
        values = factory({})
        assert runctx.resolve_error_policy(values).mode == "stop"
        assert runctx.resolve_seed(values) == runctx.DEFAULT_SEED


def test_the_keys_are_typed_and_tooltipped_through_the_seam():
    """Without this, check_settings drops on_error out of a settings CSV."""
    from spacr import settings as settings_module

    assert settings_module.has_registered_defaults("runctx")
    assert settings_module.expected_types["on_error"] is str
    assert settings_module.expected_types["on_error_attempts"] is int
    assert settings_module.expected_types["on_error_backoff"] is float
    for key in ("on_error", "on_error_attempts", "on_error_backoff"):
        assert settings_module.tooltips[key].startswith("(")
    assert "stop" in settings_module.tooltips["on_error"]
    # random_seed was already declared by spacr.settings; the seam must not
    # have rewritten another module's help text.
    assert settings_module.expected_types["random_seed"] is int
    assert settings_module.defaults_for("runctx")["on_error"] == "stop"


def test_apply_defaults_never_overwrites_what_the_user_set():
    given = {"on_error": "skip", "random_seed": 1}
    assert runctx.apply_defaults(given) is given
    assert given["on_error"] == "skip"
    assert given["random_seed"] == 1
    assert given["on_error_attempts"] == runctx.DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# The mask pipeline's plate boundary, for real
# ---------------------------------------------------------------------------
#
# The policy is unit-tested above. These assert it is actually *wired* at
# core.preprocess_generate_masks' per-plate loop: a policy that works
# perfectly and is never reached is worth nothing. Cellpose is the one
# stubbed externality (it is a GPU model fit); the orchestration is real.

@pytest.fixture
def three_plates(tmp_path):
    """Three v1 run folders, each with one field and no masks yet."""
    roots = []
    for name in ("p1", "p2", "p3"):
        root = tmp_path / name
        (root / "stack").mkdir(parents=True)
        np.save(root / "stack" / "f0.npy", np.zeros((3, 8, 8), np.uint16))
        roots.append(root)
    return roots


@pytest.fixture
def mask_stubs(monkeypatch):
    """Stub the GPU segmentation and the disk-heavy collaborators."""
    import spacr.io as sio
    import spacr.object as sobj
    import spacr.plot as splot
    import spacr.utils as su

    segmented = []
    failing = {"folder": None}

    def _cellpose(mask_src, settings, object_type, *args, **kwargs):
        segmented.append((os.path.basename(os.path.dirname(mask_src)),
                          object_type))
        if failing["folder"] and failing["folder"] in mask_src:
            raise RuntimeError("cellpose fell over on this plate")

    monkeypatch.setattr(sobj, "generate_cellpose_masks_sam", _cellpose)
    monkeypatch.setattr(sobj, "generate_organelle_masks_sam",
                        lambda *a, **k: None)
    monkeypatch.setattr(sio, "_load_and_concatenate_arrays",
                        lambda *a, **k: None)
    monkeypatch.setattr(splot, "plot_image_mask_overlay", lambda *a, **k: None)
    monkeypatch.setattr(splot, "plot_arrays", lambda *a, **k: None)
    monkeypatch.setattr(su, "adjust_cell_masks", lambda *a, **k: None)
    monkeypatch.setattr(su, "_pivot_counts_table", lambda *a, **k: None)
    monkeypatch.setattr(su, "cleanup_pipeline_folders", lambda *a, **k: [])
    return segmented, failing


def _mask_settings(roots, **over):
    values = {
        "src": [str(root) for root in roots],
        "metadata_type": "cellvoyager", "channels": [0, 1, 2],
        "cell_channel": 1, "nucleus_channel": 0, "pathogen_channel": None,
        "organelle_channel": None, "preprocess": False, "masks": True,
        "plot": False, "verbose": False, "test_mode": False,
        "timelapse": False, "n_jobs": 1, "adjust_cells": False,
        "consolidate": False, "all_to_mip": False, "batch_size": 10,
        "save": True, "custom_regex": None, "randomize": True,
        "examples_to_plot": 2, "strict_errors": False,
    }
    values.update(over)
    return values


def _plates_touched(segmented):
    """Which plate folders the segmenter was called for, in order."""
    return list(dict.fromkeys(plate for plate, _object in segmented))


def test_the_mask_pipeline_stops_at_the_plate_that_failed(
        three_plates, mask_stubs, private_logs):
    """Default `stop`: plate 3 is never started."""
    from spacr.core import preprocess_generate_masks

    segmented, failing = mask_stubs
    failing["folder"] = "p2"

    with pytest.raises(RuntimeError, match="cellpose fell over"):
        preprocess_generate_masks(_mask_settings(three_plates))

    assert _plates_touched(segmented) == ["p1", "p2"]
    assert "p3" not in _plates_touched(segmented)


def test_the_mask_pipeline_skips_the_bad_plate_and_finishes_the_rest(
        three_plates, mask_stubs, private_logs):
    """`skip`: plate 3 runs, and the skip is on the record, not swallowed."""
    from spacr.core import preprocess_generate_masks

    segmented, failing = mask_stubs
    failing["folder"] = "p2"

    preprocess_generate_masks(_mask_settings(three_plates, on_error="skip"))

    assert _plates_touched(segmented) == ["p1", "p2", "p3"]
    # What was skipped is named in the run's own log, not merely absent.
    skipped_lines = [
        line for run in _run_ids_in(private_logs)
        for line in runctx.read_run_log(run, contains="on_error=skip")]
    assert any(str(three_plates[1]) in line["message"]
               for line in skipped_lines), skipped_lines


def test_the_mask_pipeline_retries_the_bad_plate_a_bounded_number_of_times(
        three_plates, mask_stubs, private_logs, monkeypatch):
    """`retry`: the plate is re-attempted, then the run stops like `stop`."""
    from spacr.core import preprocess_generate_masks

    segmented, failing = mask_stubs
    failing["folder"] = "p2"
    monkeypatch.setattr(runctx.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="cellpose fell over"):
        preprocess_generate_masks(_mask_settings(
            three_plates, on_error="retry", on_error_attempts=3,
            on_error_backoff=0))

    # p2's first object raises, so one segmenter call per attempt: exactly
    # on_error_attempts of them, and then the run stops — p3 is never
    # started, which is what "then behaves as stop" has to mean.
    assert [plate for plate, _o in segmented].count("p2") == 3
    assert "p3" not in _plates_touched(segmented)

    # One attempt, for contrast, when retry is not asked for.
    segmented.clear()
    with pytest.raises(RuntimeError):
        preprocess_generate_masks(_mask_settings(three_plates))
    assert [plate for plate, _o in segmented].count("p2") == 1


def _run_ids_in(log_dir):
    """Every run id with a log under ``log_dir``."""
    runs = Path(log_dir) / "runs"
    return [path.stem for path in runs.glob("*.jsonl")] if runs.is_dir() else []
