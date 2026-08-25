"""The run context's edges: no torch, a full disk, a seed that is a word.

The three guarantees this module makes -- one id on every line, one seed that
reaches everything, one error policy honoured at every boundary -- are each
held up by a failure path. The id survives a run log that cannot be opened.
The seed report says what it could NOT seed, which is only true if the
unavailable branches are real. The policy's retry loop has to do the right
thing when the body never ran at all.

Torch is present here and its CUDA is not, so the CUDA branches are reached
by putting a stand-in where the module documents itself as looking for one --
`sys.modules`, which it consults precisely so that taking a measurement never
imports Torch. The disk failures are reached by making the real write fail.
"""

import builtins
import json
import logging
import os
import sys
import types

import numpy as np
import pytest

from spacr import runctx
from spacr.runctx import (DEFAULT_SEED, ON_ERROR_RETRY, ON_ERROR_SKIP, SKIPPED,
                          ErrorPolicy, RunIdFilter, SeedReport, apply_defaults,
                          current_run_id, install_run_id_logging,
                          random_state, read_run_log, resolve_seed,
                          run_context, seed_everything, seed_worker, spacr_rng,
                          torch_generator, uninstall_run_id_logging)


@pytest.fixture
def scratch_logs(tmp_path, monkeypatch):
    """Per-run logs in a scratch folder, with the root logger open."""
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    root = logging.getLogger()
    before = root.level
    root.setLevel(logging.DEBUG)
    yield tmp_path / "logs"
    root.setLevel(before)


@pytest.fixture(autouse=True)
def seed_state_restored():
    """Seeding replaces a module global; put the old stream root back."""
    before = runctx._ROOT_SEED_SEQUENCE
    yield
    runctx._ROOT_SEED_SEQUENCE = before


# ---------------------------------------------------------------------------
# the id on every line
# ---------------------------------------------------------------------------

def test_a_record_with_no_run_id_gets_one_and_is_never_dropped():
    """The filter fills in the id; a record it cannot name still passes.

    Dropping a record would lose evidence, which is the opposite of what a
    run log is for.
    """
    record = logging.LogRecord("spacr.test", logging.INFO, __file__, 1,
                               "a line", (), None)
    assert not hasattr(record, "run_id")

    assert RunIdFilter("run-abc").filter(record) is True
    assert record.run_id == "run-abc"

    # an id already on the record is left alone
    assert RunIdFilter("run-xyz").filter(record) is True
    assert record.run_id == "run-abc"


def test_a_filter_with_no_id_of_its_own_falls_back_to_a_dash():
    """Outside a run there is nothing to stamp, and '-' says so."""
    record = logging.LogRecord("spacr.test", logging.INFO, __file__, 1,
                               "x", (), None)
    RunIdFilter().filter(record)
    assert record.run_id == (current_run_id() or "-")


def test_uninstalling_the_record_factory_only_undoes_our_own():
    """Something else replacing the factory keeps its own installation.

    Ripping ours out of the middle of a chain would discard theirs too.
    """
    original = logging.getLogRecordFactory()
    try:
        uninstall_run_id_logging()
        before = logging.getLogRecordFactory()
        install_run_id_logging()
        ours = logging.getLogRecordFactory()
        assert ours is not before

        def someone_else(*args, **kwargs):
            return ours(*args, **kwargs)

        logging.setLogRecordFactory(someone_else)
        uninstall_run_id_logging()
        assert logging.getLogRecordFactory() is someone_else

        # and a second uninstall is a no-op rather than an error
        uninstall_run_id_logging()
        assert logging.getLogRecordFactory() is someone_else
    finally:
        logging.setLogRecordFactory(original)
        runctx._OUR_RECORD_FACTORY = None
        runctx._BASE_RECORD_FACTORY = None


def test_uninstalling_restores_the_factory_that_was_there_before():
    """The ordinary case: our factory goes and the previous one comes back.

    An earlier run in this process may already have installed one, so the
    starting point is established rather than assumed.
    """
    original = logging.getLogRecordFactory()
    try:
        uninstall_run_id_logging()
        before = logging.getLogRecordFactory()

        install_run_id_logging()
        ours = logging.getLogRecordFactory()
        assert ours is not before

        # asked twice, it does not chain onto itself
        install_run_id_logging()
        assert logging.getLogRecordFactory() is ours

        uninstall_run_id_logging()
        assert logging.getLogRecordFactory() is before
    finally:
        logging.setLogRecordFactory(original)
        runctx._OUR_RECORD_FACTORY = None
        runctx._BASE_RECORD_FACTORY = None


# ---------------------------------------------------------------------------
# reading the log back
# ---------------------------------------------------------------------------

def test_a_run_killed_mid_write_still_yields_its_earlier_lines(scratch_logs,
                                                               tmp_path):
    """A half-written final line is skipped; everything before it is evidence."""
    scratch_logs.mkdir(parents=True, exist_ok=True)
    path = runctx.run_log_path("run-broken")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"levelno": 20, "logger": "spacr.mask",
                                 "message": "field 1 done"}) + "\n")
        handle.write("\n")                       # a blank line
        handle.write('{"levelno": 20, "logger": "spa')   # killed mid-write

    records = read_run_log("run-broken")
    assert [r["message"] for r in records] == ["field 1 done"]


def test_a_log_can_be_filtered_by_level_by_logger_and_by_text(scratch_logs):
    """The three filters a person actually uses when reading a run back."""
    scratch_logs.mkdir(parents=True, exist_ok=True)
    path = runctx.run_log_path("run-filter")
    rows = [
        {"levelno": 10, "logger": "spacr.mask", "message": "opening plate 1"},
        {"levelno": 30, "logger": "spacr.mask", "message": "plate 1 is thin"},
        {"levelno": 40, "logger": "spacr.measure", "message": "plate 2 failed"},
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    assert len(read_run_log("run-filter")) == 3
    assert len(read_run_log("run-filter", level="WARNING")) == 2
    assert len(read_run_log("run-filter", level=logging.ERROR)) == 1
    assert [r["message"] for r in
            read_run_log("run-filter", logger="spacr.measure")] == \
        ["plate 2 failed"]
    assert [r["message"] for r in
            read_run_log("run-filter", contains="thin")] == ["plate 1 is thin"]
    # a level name nothing knows is not a filter at all
    assert len(read_run_log("run-filter", level="LOUD")) == 3


def test_a_run_with_no_log_reads_as_no_records(scratch_logs):
    """Asking about a run that wrote nothing is [] rather than an error."""
    assert read_run_log("run-never-existed") == []


# ---------------------------------------------------------------------------
# the log handler, when the disk will not take it
# ---------------------------------------------------------------------------

def test_a_traceback_is_carried_into_the_run_log(scratch_logs, tmp_path):
    """An exception logged inside a run keeps its traceback in the record."""
    with run_context("mask", {}, log=True) as run:
        try:
            raise ValueError("the plate is empty")
        except ValueError:
            run.log.exception("plate 1 failed")
        run_id = run.run_id

    records = [r for r in read_run_log(run_id) if "traceback" in r]
    assert records, "no record carried a traceback"
    assert "ValueError: the plate is empty" in records[0]["traceback"]


def test_a_record_that_cannot_be_written_does_not_take_the_run_down(
        tmp_path, monkeypatch):
    """A full disk costs the evidence, never the result.

    The run log is evidence; the analysis is the result, and losing the
    second to protect the first would be the wrong trade.
    """
    handler = runctx._RunLogHandler("run-x", str(tmp_path / "run.jsonl"))
    reported = []
    monkeypatch.setattr(handler, "handleError", reported.append)

    def full_disk():
        raise OSError("No space left on device")

    monkeypatch.setattr(handler, "_open", full_disk)
    record = logging.LogRecord("spacr.test", logging.INFO, __file__, 1,
                               "a line", (), None)
    handler.emit(record)                       # must not raise
    assert reported == [record]
    handler.close()


def test_a_record_that_cannot_be_serialised_does_not_take_the_run_down(
        tmp_path, monkeypatch):
    """A message whose formatting throws is reported, not raised."""
    handler = runctx._RunLogHandler("run-x", str(tmp_path / "run.jsonl"))
    reported = []
    monkeypatch.setattr(handler, "handleError", reported.append)

    class Unprintable:
        def __str__(self):
            raise RuntimeError("this object refuses to be text")

    record = logging.LogRecord("spacr.test", logging.INFO, __file__, 1,
                               "%s", (Unprintable(),), None)
    handler.emit(record)                       # must not raise
    assert reported == [record]
    handler.close()


def test_a_run_whose_log_cannot_be_opened_still_runs(tmp_path, monkeypatch,
                                                     caplog):
    """The id, the seed and the policy survive; only the log path is empty."""
    def refuse(_run_id):
        raise OSError("the log directory is read-only")

    monkeypatch.setattr(runctx, "run_log_path", refuse)

    with caplog.at_level(logging.WARNING):
        with run_context("mask", {"random_seed": 5}) as run:
            assert run.run_id
            assert run.log_path == ""
            assert run.seed == 5
    assert "could not open the run log" in caplog.text


# ---------------------------------------------------------------------------
# the seed
# ---------------------------------------------------------------------------

def test_the_seed_can_be_spelled_seed_as_well_as_random_seed():
    """Both keys are read, `random_seed` first."""
    assert resolve_seed({"seed": 17}) == 17
    assert resolve_seed({"random_seed": 3, "seed": 17}) == 3


def test_a_seed_that_is_not_a_number_at_all_is_no_seed():
    """An object that cannot be an int means "do not seed", not a crash."""
    assert resolve_seed({"random_seed": object()}) is None
    assert resolve_seed({"random_seed": [1, 2]}) is None
    assert resolve_seed({"random_seed": None}) is None
    assert resolve_seed({"random_seed": False}) is None


def test_a_word_is_a_usable_reproducible_seed():
    """`random_seed: "plate3-rerun"` hashes rather than crashing.

    The same word gives the same seed every run, which is the whole point.
    """
    once = resolve_seed({"random_seed": "plate3-rerun"})
    again = resolve_seed({"random_seed": "plate3-rerun"})
    assert isinstance(once, int)
    assert once == again
    assert once != resolve_seed({"random_seed": "plate4-rerun"})


def test_a_seed_report_says_in_one_line_what_it_reached():
    """The report is printed into a log, so its `str` has to be readable."""
    report = SeedReport(seed=42, seeded=("python", "numpy"),
                        unavailable=("torch",), caveats=(), deterministic=False)
    assert str(report) == "seed 42 → python, numpy (no torch)"

    complete = SeedReport(seed=1, seeded=("python",), unavailable=(),
                          caveats=(), deterministic=False)
    assert str(complete) == "seed 1 → python"

    nothing = SeedReport(seed=1, seeded=(), unavailable=(), caveats=(),
                         deterministic=False)
    assert "nothing" in str(nothing)


def test_a_build_without_torch_says_torch_is_unavailable(monkeypatch):
    """The report names what it could NOT seed rather than implying it did."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch in this build")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    report = seed_everything(11, quiet=True)
    assert "torch" in report.unavailable
    assert "python" in report.seeded
    assert "numpy" in report.seeded


def _fake_torch(monkeypatch, *, available=True, raises=None):
    """A Torch stand-in in `sys.modules`, where the module looks for one."""
    calls = {"manual_seed": [], "cuda_seed_all": []}
    cuda = types.SimpleNamespace()

    def is_available():
        if raises is not None:
            raise raises
        return available

    cuda.is_available = is_available
    cuda.manual_seed_all = lambda value: calls["cuda_seed_all"].append(value)
    module = types.ModuleType("torch")
    module.cuda = cuda
    module.manual_seed = lambda value: calls["manual_seed"].append(value)
    module.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=True))
    module.use_deterministic_algorithms = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "torch", module)
    return calls


def test_a_card_that_is_there_is_seeded_and_the_caveat_is_recorded(
        monkeypatch):
    """CUDA gets the same seed, and the report says what remains nondeterministic."""
    calls = _fake_torch(monkeypatch, available=True)
    report = seed_everything(23, quiet=True)

    assert calls["manual_seed"] == [23]
    assert calls["cuda_seed_all"] == [23]
    assert "torch.cuda" in report.seeded
    assert any("cuda" in caveat.lower() or "nondeterministic" in caveat.lower()
               for caveat in report.caveats), report.caveats


def test_a_driver_mismatch_does_not_stop_a_cpu_run_being_seeded(monkeypatch):
    """CUDA raising is recorded as unavailable; the CPU seeding still happened."""
    calls = _fake_torch(monkeypatch, raises=RuntimeError("driver mismatch"))
    report = seed_everything(23, quiet=True)

    assert calls["manual_seed"] == [23]
    assert "torch" in report.seeded
    assert "torch.cuda" in report.unavailable


def test_asking_for_deterministic_kernels_records_what_it_costs(monkeypatch):
    """Determinism has a price, and the report names it rather than hiding it."""
    _fake_torch(monkeypatch, available=False)
    report = seed_everything(23, deterministic=True, quiet=True)

    assert report.deterministic is True
    assert "torch.cudnn" in report.seeded
    assert "torch.deterministic-algorithms" in report.seeded
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    assert report.caveats


def test_seeding_says_so_unless_asked_to_be_quiet(caplog):
    """The one line goes into the log, because a run has to record its seed."""
    with caplog.at_level(logging.INFO, logger="spacr.runctx"):
        report = seed_everything(31, quiet=False)
    assert str(report) in caplog.text


def test_cellpose_is_recorded_as_seeded_through_numpy_and_torch(monkeypatch):
    """A report that omitted Cellpose would read as though it was overlooked."""
    monkeypatch.setitem(sys.modules, "cellpose", types.ModuleType("cellpose"))
    report = seed_everything(3, quiet=True)
    assert any("cellpose" in entry for entry in report.seeded)


# ---------------------------------------------------------------------------
# the streams
# ---------------------------------------------------------------------------

def test_an_explicit_seed_overrides_the_run_seed():
    """`spacr_rng(seed=...)` is reproducible on its own terms."""
    first = spacr_rng("fold0", seed=99).random(4)
    again = spacr_rng("fold0", seed=99).random(4)
    assert np.allclose(first, again)


def test_with_no_seeding_at_all_the_default_still_reproduces(monkeypatch):
    """Never seeded is still deterministic: the default seed is used.

    "Reproducible unless you asked otherwise" is only true if the unseeded
    path lands somewhere fixed.
    """
    monkeypatch.setattr(runctx, "_ROOT_SEED_SEQUENCE", None)
    monkeypatch.delenv(runctx.SEED_ENV, raising=False)

    unnamed = spacr_rng().random(4)
    monkeypatch.setattr(runctx, "_ROOT_SEED_SEQUENCE", None)
    assert np.allclose(spacr_rng().random(4), unnamed)

    monkeypatch.setattr(runctx, "_ROOT_SEED_SEQUENCE", None)
    named = spacr_rng("worker-3").random(4)
    assert not np.allclose(named, unnamed), "a named stream was not independent"


def test_a_torch_generator_needs_torch_and_says_so(monkeypatch):
    """Handing back None would fail further away, so it raises here."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(RuntimeError) as raised:
        torch_generator()
    assert "PyTorch" in str(raised.value)


def test_a_worker_without_torch_is_still_seeded(monkeypatch):
    """A DataLoader worker must not augment identically to its siblings."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    seed_worker(0)
    first = np.random.random(3)
    seed_worker(0)
    assert np.allclose(np.random.random(3), first)

    seed_worker(1)
    assert not np.allclose(np.random.random(3), first), \
        "two workers drew the same 'random' numbers"


# ---------------------------------------------------------------------------
# how the policy describes itself
# ---------------------------------------------------------------------------

def test_a_skip_is_falsy_and_prints_as_skipped():
    """`if result:` must treat a skipped unit as no result."""
    assert bool(SKIPPED) is False
    assert repr(SKIPPED) == "SKIPPED"
    assert runctx._SkippedType() is SKIPPED


def test_a_policy_describes_its_mode_and_what_it_lost():
    """The repr is what appears in a traceback or a debugger."""
    plain = ErrorPolicy()
    assert repr(plain) == "<ErrorPolicy stop skipped=0>"

    retrying = ErrorPolicy(mode=ON_ERROR_RETRY, attempts=4, backoff=0.5)
    text = repr(retrying)
    assert "retry" in text
    assert "attempts=4" in text
    assert "backoff=0.5" in text


def test_an_attempt_names_the_unit_and_which_try_it_is():
    """The repr is read in a log line and in a debugger."""
    policy = ErrorPolicy(mode=ON_ERROR_RETRY, attempts=3, sleep=lambda _s: None)
    attempts = list(policy.attempts_for("plate1", stage="plate"))
    text = repr(attempts[0])
    assert "plate1" in text
    assert "stage='plate'" in text
    assert "1/" in text


def test_a_loop_body_that_never_ran_is_not_judged():
    """`break` inside the `with` leaves nothing to succeed or fail at.

    Treating it as a success would record a unit that was never attempted.
    """
    policy = ErrorPolicy(mode=ON_ERROR_RETRY, attempts=3, sleep=lambda _s: None)
    seen = 0
    for attempt in policy.attempts_for("plate1", stage="plate"):
        attempt.__enter__()
        seen += 1
        break
    assert seen == 1
    assert policy.n_skipped == 0
    assert policy.retries == []


def test_a_long_skip_list_is_summarised_rather_than_printed_in_full():
    """Twenty skips are listed and the rest are counted."""
    policy = ErrorPolicy(mode=ON_ERROR_SKIP)
    for i in range(25):
        for attempt in policy.attempts_for(f"plate{i}", stage="plate"):
            with attempt:
                raise ValueError(f"plate {i} is empty")

    summary = policy.summary()
    assert "SKIPPED  : 25 unit(s)" in summary
    assert "NOT in the output" in summary
    assert "... and 5 more" in summary


def test_the_summary_names_the_units_that_had_to_be_retried():
    """A unit that only succeeded on a later try is worth saying out loud."""
    policy = ErrorPolicy(mode=ON_ERROR_RETRY, attempts=3, sleep=lambda _s: None)
    tries = {"n": 0}
    for attempt in policy.attempts_for("plate1", stage="plate"):
        with attempt:
            tries["n"] += 1
            if tries["n"] < 2:
                raise OSError("the share was not mounted yet")

    assert policy.retries == [("plate1", 2)]
    assert "retried  : 1 unit(s)" in policy.summary()
    assert "plate1 — 2 attempt(s)" in policy.summary()


# ---------------------------------------------------------------------------
# the context object
# ---------------------------------------------------------------------------

def test_a_run_hands_out_its_own_seeded_streams(scratch_logs):
    """`run.rng` and `run.random_state` are the run's seed, not a fresh one."""
    with run_context("mask", {"random_seed": 77}) as run:
        assert run.random_state() == 77
        assert np.allclose(run.rng("fold0").random(3),
                           spacr_rng("fold0", seed=77).random(3))


def test_an_unseeded_run_hands_back_the_caller_s_default(scratch_logs):
    """With no seed, `random_state(default)` returns the default."""
    with run_context("mask", {"random_seed": None}) as run:
        assert run.seed is None
        assert run.random_state() is None
        assert run.random_state(default=5) == 5


def test_a_run_prints_as_one_readable_line(scratch_logs):
    """The `str` goes into a console, so it names id, module, seed and policy."""
    with run_context("mask", {"random_seed": 8, "on_error": "skip"}) as run:
        text = str(run)
        assert run.run_id in text
        assert "(mask)" in text
        assert "seed=8" in text
        assert "on_error=skip" in text

    with run_context("", {}, log=False) as anonymous:
        assert "(spacr)" in str(anonymous)


# ---------------------------------------------------------------------------
# the settings seam
# ---------------------------------------------------------------------------

def test_filling_defaults_into_nothing_makes_a_complete_dict():
    """`apply_defaults(None)` returns a new dict with every run-control key."""
    filled = apply_defaults()
    assert filled["random_seed"] == DEFAULT_SEED
    assert filled["on_error"] == runctx.DEFAULT_ON_ERROR
    assert filled["on_error_attempts"] == runctx.DEFAULT_RETRIES
    assert filled["on_error_backoff"] == runctx.DEFAULT_BACKOFF


def test_filling_defaults_never_overwrites_what_the_caller_chose():
    """The caller's dict is filled in place and its own values are kept."""
    given = {"on_error": "skip"}
    same = apply_defaults(given)
    assert same is given
    assert given["on_error"] == "skip"
    assert given["random_seed"] == DEFAULT_SEED


def test_declaring_the_settings_twice_is_a_no_op(monkeypatch):
    """Import runs it once; a second call finds the keys already declared.

    A no-op means it does not register again: the registry refuses a second
    declaration of the same key with a ValueError, so a guard that re-tried
    would either raise or bury a real conflict in a debug line.
    """
    from spacr import settings as settings_module

    runctx._register_settings()          # whatever the session left behind
    assert settings_module.has_registered_defaults("runctx")

    again = []
    monkeypatch.setattr(settings_module, "register_defaults",
                        lambda *args, **kwargs: again.append(args))
    runctx._register_settings()
    assert again == []
    assert settings_module.has_registered_defaults("runctx")


def test_without_the_settings_module_the_import_still_works(monkeypatch):
    """A trimmed install has no settings registry, and that is not fatal."""
    real_import = builtins.__import__

    refused = []

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if level and name == "settings":
            refused.append(name)
            raise ImportError("no settings module here")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    runctx._register_settings()

    # The import really was the thing that failed -- a guard that returned
    # before reaching for the registry would also raise nothing -- and it is
    # asked for once, then given up on rather than retried.
    assert refused == ["settings"]


def test_another_module_owning_a_key_is_logged_rather_than_fatal(monkeypatch,
                                                                 caplog):
    """A clash says so once instead of taking `import spacr.runctx` down."""
    from spacr import settings as settings_module

    monkeypatch.setattr(settings_module, "has_registered_defaults",
                        lambda name: False)

    def clash(*_args, **_kwargs):
        raise ValueError("'on_error' is already declared by another module")

    monkeypatch.setattr(settings_module, "register_defaults", clash)

    with caplog.at_level(logging.DEBUG, logger="spacr.runctx"):
        runctx._register_settings()      # must not raise
    assert "not registered" in caplog.text


# ---------------------------------------------------------------------------
# the rest of the policy's surface
# ---------------------------------------------------------------------------

def test_a_policy_can_be_pointed_at_a_different_ledger_after_the_fact():
    """`bind` moves the recording without moving the account of what was lost.

    `skips` stays the single answer to "what did this run not cover", which
    is why the records are not copied onto the new ledger.
    """
    from spacr.errors import RunLedger

    policy = ErrorPolicy(mode=ON_ERROR_SKIP)
    for attempt in policy.attempts_for("plate1", stage="plate"):
        with attempt:
            raise ValueError("empty")

    other = RunLedger("later-stage")
    assert policy.bind(other, record=False) is policy
    assert policy.ledger is other
    assert policy.record is False
    assert policy.skipped_units == ["plate1"]
    assert [record.unit for record in policy.skips] == ["plate1"]


def test_the_ergonomic_form_returns_the_result_or_the_skip_sentinel():
    """`policy.run` is `attempts_for` for a boundary that is already a call."""
    policy = ErrorPolicy(mode=ON_ERROR_SKIP)

    assert policy.run("plate1", lambda a, b: a + b, 2, 3, stage="plate") == 5

    def fails():
        raise OSError("the share went away")

    assert policy.run("plate2", fails, stage="plate") is SKIPPED
    assert policy.skipped_units == ["plate2"]


def test_a_retry_that_runs_out_of_attempts_behaves_like_stop():
    """The budget running out re-raises, and the unit is listed as retried."""
    policy = ErrorPolicy(mode=ON_ERROR_RETRY, attempts=2, sleep=lambda _s: None)

    with pytest.raises(OSError):
        for attempt in policy.attempts_for("plate1", stage="plate"):
            with attempt:
                raise OSError("the share is gone for good")

    assert policy.retries == [("plate1", 2)]


def test_stop_re_raises_the_original_exception():
    """The default mode aborts on the first failure, with the real error."""
    policy = ErrorPolicy()
    with pytest.raises(ValueError, match="plate 1 is empty"):
        for attempt in policy.attempts_for("plate1", stage="plate"):
            with attempt:
                raise ValueError("plate 1 is empty")


def test_a_nonsense_retry_budget_in_the_settings_falls_back_to_the_default():
    """A settings CSV can hold anything; the policy still arms."""
    from spacr.runctx import resolve_error_policy

    policy = resolve_error_policy({"on_error": "retry",
                                   "on_error_attempts": "many",
                                   "on_error_backoff": "slowly"})
    assert policy.mode == ON_ERROR_RETRY
    assert policy.attempts == runctx.DEFAULT_RETRIES
    assert policy.backoff == runctx.DEFAULT_BACKOFF


# ---------------------------------------------------------------------------
# one id on every stamp
# ---------------------------------------------------------------------------

def test_every_ledger_a_run_makes_carries_the_run_s_own_id(scratch_logs):
    """A ledger mints its own uuid; a second id would break the join.

    Log line, ledger row and artifact all have to be joinable on `run_id`,
    which is the whole point of the context.
    """
    from spacr.errors import RunLedger

    with run_context("mask", {}) as run:
        made = run.new_ledger("segmentation")
        assert made.run_id == run.run_id

        outside = RunLedger("built-elsewhere")
        assert outside.run_id != run.run_id
        adopted = run.adopt(outside)
        assert adopted is outside
        assert outside.run_id == run.run_id


def test_a_run_accounts_for_itself_as_a_dict(scratch_logs):
    """`to_dict` is what a manifest writes, so every field has to be there."""
    with run_context("mask", {"random_seed": 4, "on_error": "skip"}) as run:
        for attempt in run.policy.attempts_for("plate1", stage="plate"):
            with attempt:
                raise ValueError("empty plate")
        account = run.to_dict()
        assert run.skips and account["skipped"][0]["unit"] == "plate1"

    assert account["run_id"] == run.run_id
    assert account["module"] == "mask"
    assert account["seed"] == 4
    assert account["on_error"] == "skip"
    assert account["seed_report"]["seed"] == 4
    assert account["log_path"].endswith(".jsonl")
    json.dumps(account)


def test_a_run_that_raises_says_so_in_its_own_log(scratch_logs):
    """The failure is logged with the run id before it propagates."""
    with pytest.raises(RuntimeError):
        with run_context("mask", {}) as run:
            run_id = run.run_id
            raise RuntimeError("the plate list was empty")

    messages = [r["message"] for r in read_run_log(run_id)]
    assert any("failed after" in m and "the plate list was empty" in m
               for m in messages), messages


def test_a_run_that_skipped_something_warns_about_it_on_the_way_out(
        scratch_logs):
    """What was lost is said at the end, not only recorded."""
    with run_context("mask", {"on_error": "skip"}) as run:
        run_id = run.run_id
        for attempt in run.policy.attempts_for("plate1", stage="plate"):
            with attempt:
                raise ValueError("empty plate")

    messages = [r["message"] for r in read_run_log(run_id)]
    assert any("NOT in the output" in m for m in messages), messages
    assert any("finished in" in m and "1 skipped" in m for m in messages)


def test_the_run_id_environment_variable_is_put_back(scratch_logs,
                                                     monkeypatch):
    """A nested or sequential run must not leak its id to the next one."""
    monkeypatch.delenv(runctx.RUN_ID_ENV, raising=False)
    with run_context("mask", {}) as run:
        assert os.environ[runctx.RUN_ID_ENV] == run.run_id
    assert runctx.RUN_ID_ENV not in os.environ

    monkeypatch.setenv(runctx.RUN_ID_ENV, "an-outer-run")
    with run_context("mask", {}):
        assert os.environ[runctx.RUN_ID_ENV] != "an-outer-run"
    assert os.environ[runctx.RUN_ID_ENV] == "an-outer-run"


def test_the_seed_can_come_from_the_environment(monkeypatch):
    """`SPACR_SEED` is read when the settings name no seed."""
    monkeypatch.setenv(runctx.SEED_ENV, "1234")
    assert resolve_seed({}) == 1234
    assert resolve_seed(None) == 1234
    # and a settings value still wins over it
    assert resolve_seed({"random_seed": 7}) == 7


def test_the_word_off_means_do_not_seed():
    """Someone deliberately turning reproducibility off is a real request."""
    for text in ("off", "none", "null", "false", "", "  "):
        assert resolve_seed({"random_seed": text}) is None


def test_a_torch_generator_is_seeded_from_the_run_seed():
    """A DataLoader's shuffling is part of the run, so it takes the run seed."""
    torch = pytest.importorskip("torch")

    first = torch_generator("cpu", stream="loader")
    again = torch_generator("cpu", stream="loader")
    assert first.initial_seed() == again.initial_seed()
    assert torch_generator("cpu").initial_seed() != first.initial_seed()


def test_a_worker_with_torch_derives_its_stream_from_torch(monkeypatch):
    """Torch has already varied its seed per worker and per epoch."""
    pytest.importorskip("torch")
    seed_worker(0)
    first = np.random.random(3)
    seed_worker(2)
    assert not np.allclose(np.random.random(3), first)
