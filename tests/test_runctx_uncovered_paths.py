"""The parts of the run context a pipeline reaches only on its second pass.

Three seams here are only exercised by a caller that is doing something more
than the happy path, and each of them decides whether a number the user reads
is true:

* ``_Attempt.last`` is what a retrying body asks before it decides to give a
  failing unit one more chance of its own; getting it wrong either wastes the
  budget or spends it a try early.
* ``ErrorPolicy.bind`` is how Measure points one run's policy at a second
  ledger without losing the run-wide skip list. Binding one of the two
  arguments must not silently reset the other.
* ``RunContext.register_outputs`` is the join between a run's log lines and
  its outputs: every artifact it registers has to carry the run id, and a
  registry that cannot be written must cost a printed line rather than the
  run.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr import artifacts                                        # noqa: E402
from spacr.errors import RunLedger                                 # noqa: E402
from spacr.runctx import (ErrorPolicy, ON_ERROR_RETRY,             # noqa: E402
                          ON_ERROR_SKIP, ON_ERROR_STOP, run_context)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """No test may inherit a shared-registry override from the environment."""
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


def _measured_plate(root: Path) -> str:
    """A plate folder shaped the way a finished measure run leaves one."""
    (root / "merged").mkdir(parents=True, exist_ok=True)
    np.save(root / "merged" / "plate1_A01_0.npy",
            np.zeros((6, 6, 3), dtype=np.uint16))
    (root / "measurements").mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "measurements" / "measurements.db")
    for table in ("png_list", "cell"):
        connection.execute(f'CREATE TABLE "{table}" (value INTEGER)')
        connection.execute(f'INSERT INTO "{table}" VALUES (1)')
    connection.commit()
    connection.close()
    return str(root)


# ---------------------------------------------------------------------------
# _Attempt.last
# ---------------------------------------------------------------------------

def test_an_attempt_says_whether_the_retry_budget_ends_with_it():
    """``last`` is False until the final try of a retry budget, then True."""
    policy = ErrorPolicy(mode=ON_ERROR_RETRY, attempts=3,
                         sleep=lambda _seconds: None)
    seen = []

    with pytest.raises(OSError):
        for attempt in policy.attempts_for("plate1", stage="plate"):
            seen.append(attempt.last)
            with attempt:
                raise OSError("the share was not mounted")

    assert seen == [False, False, True]


def test_the_only_attempt_of_a_stop_run_is_already_the_last_one():
    """Without a retry budget the first try is also the final one."""
    policy = ErrorPolicy(mode=ON_ERROR_STOP)
    seen = []

    for attempt in policy.attempts_for("plate1", stage="plate"):
        seen.append((attempt.number, attempt.of, attempt.last))
        with attempt:
            pass

    assert seen == [(1, 1, True)]


# ---------------------------------------------------------------------------
# ErrorPolicy.bind
# ---------------------------------------------------------------------------

def test_binding_a_new_ledger_leaves_recording_switched_on():
    """A second ledger takes the failures; the first one keeps its own."""
    first = RunLedger("plate-a")
    second = RunLedger("plate-b")
    policy = ErrorPolicy(mode=ON_ERROR_SKIP, ledger=first)

    assert policy.bind(ledger=second) is policy
    assert policy.ledger is second
    assert policy.record is True

    for attempt in policy.attempts_for("plate1", stage="plate"):
        with attempt:
            raise ValueError("no fields in this plate")

    assert second.n_failed == 1
    assert first.n_failed == 0
    # The skip list stays with the run, not with either ledger.
    assert [record.unit for record in policy.skips] == ["plate1"]


def test_binding_only_the_record_flag_keeps_the_ledger_it_already_had():
    """Switching recording off must not reset the policy to a fresh ledger."""
    ledger = RunLedger("plate-a")
    policy = ErrorPolicy(mode=ON_ERROR_SKIP, ledger=ledger)

    assert policy.bind(record=False) is policy
    assert policy.ledger is ledger
    assert policy.record is False

    for attempt in policy.attempts_for("plate1", stage="plate"):
        with attempt:
            raise ValueError("no fields in this plate")

    assert ledger.n_failed == 0
    # Not recorded on the ledger, but never a silent drop either.
    assert [record.unit for record in policy.skips] == ["plate1"]


def test_a_policy_that_does_not_record_leaves_a_success_off_the_ledger():
    """``record=False`` keeps a success off the ledger it is bound to."""
    ledger = RunLedger("plate")
    quiet = ErrorPolicy(mode=ON_ERROR_SKIP, ledger=ledger, record=False)

    for attempt in quiet.attempts_for("plate1", stage="plate"):
        with attempt:
            pass

    assert ledger.n_succeeded == 0
    assert ledger.n_attempted == 0

    loud = ErrorPolicy(mode=ON_ERROR_SKIP, ledger=ledger, record=True)
    for attempt in loud.attempts_for("plate2", stage="plate"):
        with attempt:
            pass

    assert ledger.n_succeeded == 1


# ---------------------------------------------------------------------------
# RunContext.register_outputs
# ---------------------------------------------------------------------------

def test_register_outputs_stamps_the_run_id_on_every_artifact(tmp_path):
    """The id on the run is the id on the artifact, which is the whole join."""
    root = _measured_plate(tmp_path / "plate1")

    with run_context("measure", {"src": root}, log=False, seed=None) as run:
        registered = run.register_outputs(roots=[root])
        run_id = run.run_id

    assert registered, "a finished measure run declares at least one output"
    assert {a.run_id for a in registered} == {run_id}
    assert {a.module for a in registered} == {"measure"}

    # And the same id comes back out of the registry on disk.
    stored = artifacts.by_project(project=root)
    assert {a.run_id for a in stored} == {run_id}


def test_register_outputs_costs_a_printed_line_not_the_run(tmp_path, capsys):
    """A registration that cannot be made is reported and returns nothing."""
    root = _measured_plate(tmp_path / "plate1")

    with run_context("measure", {"src": root}, log=False, seed=None) as run:
        registered = run.register_outputs(module="not_a_real_module",
                                          roots=[root])

    assert registered == ()
    printed = capsys.readouterr().out
    assert "not_a_real_module" in printed
    assert "could not record" in printed


def test_a_caller_that_asks_for_strict_registration_still_gets_it(tmp_path):
    """``strict`` is a default, not an override: passing True must raise."""
    root = _measured_plate(tmp_path / "plate1")

    with run_context("measure", {"src": root}, log=False, seed=None) as run:
        with pytest.raises(Exception, match="no ports declared"):
            run.register_outputs(module="not_a_real_module", roots=[root],
                                 strict=True)


def test_register_outputs_defaults_to_the_run_module_and_settings(tmp_path):
    """Called bare, it registers the module and settings the run opened with."""
    root = _measured_plate(tmp_path / "plate1")
    settings = {"src": root, "cell_diameter": 30}

    with run_context("measure", settings, log=False, seed=None) as run:
        registered = run.register_outputs(roots=[root])

    assert [a.module for a in registered] == ["measure"]
    # The settings the run carried are the settings hashed into the artifact.
    expected = artifacts.settings_hash(settings)
    assert {a.settings_hash for a in registered} == {expected}
