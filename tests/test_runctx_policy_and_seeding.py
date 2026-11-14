"""How ``spacr.runctx`` judges a failed unit and how it seeds a run.

Two of the run context's three guarantees, walked as case tables:

* the error policy -- every way a unit of work can end under ``stop``,
  ``skip`` and ``retry`` (succeeded, failed, failed until the budget ran out,
  a body that never ran), and ``_give_up`` with and without an exception to
  give up over;
* the seed -- where :func:`resolve_seed` reads it from and how it spells
  "do not seed", what :func:`seed_everything` reports it reached with and
  without Torch and CUDA, and the streams :func:`spacr_rng` hands out.

Torch's CUDA is reached through a stand-in in ``sys.modules``, which is where
the module looks first. CPU-only and offline.
"""
from __future__ import annotations

import builtins
import hashlib
import logging
import os
import sys
import types

import numpy as np
import pytest

from spacr import runctx
from spacr.runctx import (DEFAULT_SEED, ON_ERROR_RETRY, ON_ERROR_SKIP,
                          ON_ERROR_STOP, SEED_CAVEATS, ErrorPolicy,
                          resolve_seed, seed_everything, spacr_rng)


@pytest.fixture(autouse=True)
def seed_state_restored(monkeypatch):
    """Seeding replaces module globals and env vars; put them all back."""
    monkeypatch.setattr(runctx, "_ROOT_SEED_SEQUENCE",
                        runctx._ROOT_SEED_SEQUENCE)
    monkeypatch.delenv(runctx.SEED_ENV, raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)


# ---------------------------------------------------------------------------
# the error policy
# ---------------------------------------------------------------------------

def _drive(policy, unit, outcomes, seen):
    """Run ``attempts_for`` with one body per attempt from ``outcomes``.

    Each outcome is ``"ok"``, an exception instance to raise inside the
    ``with``, ``"break"`` (leave the loop before the ``with``), ``"pass"``
    (go on to the next item without entering the attempt) or ``"enter"``
    (enter the attempt by hand, never exit it, and break). The repr of every
    attempt yielded is appended to ``seen``.
    """
    plan = iter(outcomes)
    for attempt in policy.attempts_for(unit, stage="plate"):
        seen.append(repr(attempt))
        outcome = next(plan)
        if outcome == "break":
            break
        if outcome == "pass":
            continue
        if outcome == "enter":
            attempt.__enter__()
            break
        with attempt:
            if isinstance(outcome, BaseException):
                raise outcome


def _no_sleep(_seconds):
    return None


def test_error_policy_judges_every_way_a_unit_can_end(subtests):
    """attempts_for: success, skip, re-raise, retry, and a body that never ran."""
    empty = ValueError("plate 1 is empty")
    gone = OSError("the share is gone for good")
    late = OSError("the share was not mounted yet")
    cases = [
        # (id, mode, attempts, outcomes, raised, attempt reprs, skipped units,
        #  retries, ledger failures, ledger successes)
        ("stop re-raises the original exception", ON_ERROR_STOP, 3,
         [empty], empty,
         ["<attempt 1/1 on 'plate1' stage='plate'>"], [], [], 1, 0),
        ("skip records the unit and carries on", ON_ERROR_SKIP, 3,
         [empty], None,
         ["<attempt 1/1 on 'plate1' stage='plate'>"], ["plate1"], [], 1, 0),
        ("retry that runs out of attempts behaves like stop", ON_ERROR_RETRY,
         2, [gone, gone], gone,
         ["<attempt 1/2 on 'plate1' stage='plate'>",
          "<attempt 2/2 on 'plate1' stage='plate'>"],
         [], [("plate1", 2)], 1, 0),
        ("retry that succeeds on the second try is listed as retried",
         ON_ERROR_RETRY, 3, [late, "ok"], None,
         ["<attempt 1/3 on 'plate1' stage='plate'>",
          "<attempt 2/3 on 'plate1' stage='plate'>"],
         [], [("plate1", 2)], 0, 1),
        ("a retry budget of one runs the body once", ON_ERROR_RETRY, 1,
         [ZeroDivisionError("division by zero")], ZeroDivisionError,
         ["<attempt 1/1 on 'plate1' stage='plate'>"],
         [], [("plate1", 1)], 1, 0),
        ("a loop left before the with is not judged", ON_ERROR_STOP, 3,
         ["break"], None,
         ["<attempt 1/1 on 'plate1' stage='plate'>"], [], [], 0, 0),
        ("an attempt entered but never exited is not judged", ON_ERROR_RETRY,
         3, ["enter"], None,
         ["<attempt 1/3 on 'plate1' stage='plate'>"], [], [], 0, 0),
        ("a retry loop whose body never ran is not retried", ON_ERROR_RETRY,
         2, ["break"], None,
         ["<attempt 1/2 on 'plate1' stage='plate'>"], [], [], 0, 0),
        ("an attempt the loop walks past ends the unit unjudged",
         ON_ERROR_RETRY, 3, ["pass"], None,
         ["<attempt 1/3 on 'plate1' stage='plate'>"], [], [], 0, 0),
    ]
    for (case_id, mode, attempts, outcomes, raised, reprs, skipped, retries,
         n_failures, n_successes) in cases:
        with subtests.test(msg=case_id):
            policy = ErrorPolicy(mode, attempts=attempts, sleep=_no_sleep)
            seen = []
            if raised is None:
                _drive(policy, "plate1", outcomes, seen)
            else:
                kind = raised if isinstance(raised, type) else type(raised)
                with pytest.raises(kind) as caught:
                    _drive(policy, "plate1", outcomes, seen)
                if not isinstance(raised, type):
                    assert caught.value is raised
            assert seen == reprs
            assert policy.skipped_units == skipped
            assert policy.n_skipped == len(skipped)
            assert policy.retries == retries
            assert len(policy.ledger.failures) == n_failures
            assert policy.ledger.n_succeeded == n_successes

    with subtests.test(msg="a retry budget below one is refused"):
        for budget in (0, -3):
            with pytest.raises(ValueError, match="at least 1"):
                ErrorPolicy(ON_ERROR_RETRY, attempts=budget)
        assert ErrorPolicy(ON_ERROR_RETRY, attempts=1).attempts == 1


def test_giving_up_records_the_failure_then_skips_or_reraises(subtests):
    """_give_up: the ledger always hears; only a real exception is raised or skipped."""
    boom = RuntimeError("the plate could not be read")
    cases = [
        # (id, mode, exc, raises, skip record (unit, stage, exc_type, reason,
        #  message, attempts) or None, ledger failure (exc_type, message))
        ("stop with nothing to raise returns", ON_ERROR_STOP, None, False,
         None, ("Failure", "unspecified failure")),
        ("skip with nothing to raise adds no skip", ON_ERROR_SKIP, None, False,
         None, ("Failure", "unspecified failure")),
        ("retry with nothing to raise returns", ON_ERROR_RETRY, None, False,
         None, ("Failure", "unspecified failure")),
        ("stop re-raises a real exception", ON_ERROR_STOP, boom, True, None,
         ("RuntimeError", "the plate could not be read")),
        ("skip records a real exception and continues", ON_ERROR_SKIP, boom,
         False,
         ("plate1", "plate", "RuntimeError",
          "RuntimeError: the plate could not be read",
          "the plate could not be read", 1),
         ("RuntimeError", "the plate could not be read")),
        ("skip of a message-less exception names only its type",
         ON_ERROR_SKIP, KeyError(), False,
         ("plate1", "plate", "KeyError", "KeyError", "", 1),
         ("KeyError", "KeyError")),
    ]
    for case_id, mode, exc, raises, skip, failure in cases:
        with subtests.test(msg=case_id):
            policy = ErrorPolicy(mode, run_id="run-7")
            if raises:
                with pytest.raises(type(exc)) as caught:
                    policy._give_up("plate1", "plate", exc, 1)
                assert caught.value is exc
            else:
                assert policy._give_up("plate1", "plate", exc, 1) is None
            recorded = policy.ledger.failures
            assert [(f.item, f.stage, f.exc_type, f.message)
                    for f in recorded] == [("plate1", "plate", *failure)]
            if skip is None:
                assert policy.skips == []
            else:
                [record] = policy.skips
                assert (record.unit, record.stage, record.exc_type,
                        record.reason, record.message,
                        record.attempts) == skip
                assert record.run_id == "run-7"
            assert policy.retries == (
                [("plate1", 1)] if mode == ON_ERROR_RETRY and raises else [])

    with subtests.test(msg="record=False keeps the ledger empty"):
        quiet = ErrorPolicy(ON_ERROR_SKIP, record=False)
        quiet._give_up("plate1", "plate", boom, 2)
        assert quiet.ledger.failures == []
        assert [r.attempts for r in quiet.skips] == [2]


# ---------------------------------------------------------------------------
# where the seed comes from
# ---------------------------------------------------------------------------

def test_resolve_seed_reads_settings_then_environment_then_default(
        subtests, monkeypatch):
    """resolve_seed: key order, the environment, words, and every spelling of off."""
    word = 971716533           # sha256("plate3-rerun")[:8] as an int
    cases = [
        # (id, settings, environment SPACR_SEED or None, default, expected)
        ("random_seed is read", {"random_seed": 3}, None, DEFAULT_SEED, 3),
        ("seed is read when random_seed is absent", {"seed": 17}, None,
         DEFAULT_SEED, 17),
        ("random_seed beats seed", {"random_seed": 3, "seed": 17}, None,
         DEFAULT_SEED, 3),
        ("nothing named gives the default", {}, None, DEFAULT_SEED, 42),
        ("no settings at all gives the default", None, None, DEFAULT_SEED, 42),
        ("a None default leaves the RNGs alone", {}, None, None, None),
        ("the environment is read when settings name no seed", {}, "1234",
         DEFAULT_SEED, 1234),
        ("the environment is read with no settings", None, "1234",
         DEFAULT_SEED, 1234),
        ("a settings value beats the environment", {"random_seed": 7}, "1234",
         DEFAULT_SEED, 7),
        ("a blank environment is not a seed", {}, "   ", DEFAULT_SEED, 42),
        ("a numeric string is an int", {"random_seed": " 25 "}, None,
         DEFAULT_SEED, 25),
        ("a hex string is an int", {"random_seed": "0x10"}, None,
         DEFAULT_SEED, 16),
        ("a float is truncated", {"random_seed": 3.7}, None, DEFAULT_SEED, 3),
        ("a word hashes to a fixed seed", {"random_seed": "plate3-rerun"},
         None, DEFAULT_SEED, word),
        ("another word hashes elsewhere", {"random_seed": "plate4-rerun"},
         None, DEFAULT_SEED, 1262850409),
        ("an object is no seed", {"random_seed": object()}, None,
         DEFAULT_SEED, None),
        ("a list is no seed", {"random_seed": [1, 2]}, None, DEFAULT_SEED,
         None),
        ("an explicit None is no seed", {"random_seed": None}, None,
         DEFAULT_SEED, None),
        ("False is no seed", {"random_seed": False}, None, DEFAULT_SEED, None),
    ]
    cases += [(f"the word {text!r} means do not seed", {"random_seed": text},
               None, DEFAULT_SEED, None)
              for text in ("off", "OFF", "none", "null", "false", "", "  ")]
    for case_id, settings, env, default, expected in cases:
        with subtests.test(msg=case_id):
            if env is None:
                monkeypatch.delenv(runctx.SEED_ENV, raising=False)
            else:
                monkeypatch.setenv(runctx.SEED_ENV, env)
            assert resolve_seed(settings, default=default) == expected
    monkeypatch.delenv(runctx.SEED_ENV, raising=False)


# ---------------------------------------------------------------------------
# what seeding reaches
# ---------------------------------------------------------------------------

_BASE_CAVEATS = (SEED_CAVEATS["hashseed"], SEED_CAVEATS["sklearn"],
                 SEED_CAVEATS["cellpose"], SEED_CAVEATS["workers"],
                 SEED_CAVEATS["threads"])
_CPU = ("python", "numpy", "numpy.Generator")


def _fake_torch(*, available=True, raises=None):
    """A Torch stand-in that records what it was seeded with."""
    calls = {"manual_seed": [], "cuda_seed_all": [], "deterministic": []}

    def is_available():
        if raises is not None:
            raise raises
        return available

    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(
        is_available=is_available,
        manual_seed_all=lambda value: calls["cuda_seed_all"].append(value))
    module.manual_seed = lambda value: calls["manual_seed"].append(value)
    module.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=True))
    module.use_deterministic_algorithms = (
        lambda *a, **k: calls["deterministic"].append((a, k)))
    return module, calls


def _block_torch_import(monkeypatch):
    """Make ``import torch`` fail, as on a build without it."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch in this build")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)


def test_seed_everything_reports_exactly_what_it_reached(subtests,
                                                         monkeypatch, caplog):
    """seed_everything: seeded / unavailable / caveats for each library state."""
    cases = [
        # (id, torch state, deterministic, cellpose loaded, seeded,
        #  unavailable, extra caveats, manual_seed calls, cuda seed calls)
        ("a build without torch says torch is unavailable", "missing", False,
         False, _CPU, ("torch",), (), None, None),
        ("a card that is there is seeded and its caveat recorded",
         "available", False, False, _CPU + ("torch", "torch.cuda"), (),
         (SEED_CAVEATS["cuda"],), [23], [23]),
        ("a driver mismatch does not stop the CPU seeding", "raises", False,
         False, _CPU + ("torch",), ("torch.cuda",), (), [23], []),
        ("deterministic kernels are switched on and their cost named",
         "absent-cuda", True, False,
         _CPU + ("torch", "torch.cudnn", "torch.deterministic-algorithms"),
         (), (SEED_CAVEATS["cublas"],), [23], []),
        ("deterministic without torch switches nothing", "missing", True,
         False, _CPU, ("torch",), (), None, None),
        ("cellpose is recorded as seeded through numpy and torch",
         "absent-cuda", False, True,
         _CPU + ("torch", "cellpose(via numpy+torch)"), (), (), [23], []),
    ]
    for (case_id, state, deterministic, cellpose, seeded, unavailable,
         extra, manual, cuda) in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            patch.delenv("PYTHONHASHSEED", raising=False)
            patch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
            if cellpose:
                patch.setitem(sys.modules, "cellpose",
                              types.ModuleType("cellpose"))
            else:
                patch.delitem(sys.modules, "cellpose", raising=False)
            calls = None
            if state == "missing":
                _block_torch_import(patch)
            else:
                module, calls = _fake_torch(
                    available=(state == "available"),
                    raises=(RuntimeError("driver mismatch")
                            if state == "raises" else None))
                patch.setitem(sys.modules, "torch", module)

            report = seed_everything(23, deterministic=deterministic,
                                     quiet=True)

            assert report.seed == 23
            assert report.seeded == seeded
            assert report.unavailable == unavailable
            assert report.caveats == _BASE_CAVEATS + extra
            assert report.deterministic is deterministic
            assert os.environ["PYTHONHASHSEED"] == "23"
            assert runctx._ROOT_SEED_SEQUENCE.entropy == 23
            if calls is not None:
                assert calls["manual_seed"] == manual
                assert calls["cuda_seed_all"] == cuda
                switched = state == "absent-cuda" and deterministic
                assert module.backends.cudnn.deterministic is switched
                assert module.backends.cudnn.benchmark is (not switched)
                assert calls["deterministic"] == (
                    [((True,), {"warn_only": True})] if switched else [])
            assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == (
                ":4096:8" if "torch.cudnn" in seeded else None)
            assert report.to_dict() == {
                "seed": 23, "seeded": list(seeded),
                "unavailable": list(unavailable),
                "caveats": list(_BASE_CAVEATS + extra),
                "deterministic": deterministic}

    with subtests.test(msg="None seeds the default and a huge seed is narrowed"):
        with monkeypatch.context() as patch:
            patch.delenv("PYTHONHASHSEED", raising=False)
            assert seed_everything(None, quiet=True).seed == DEFAULT_SEED
            assert seed_everything(2 ** 32 + 5, quiet=True).seed == 2 ** 32 + 5
            assert os.environ["PYTHONHASHSEED"] == "5"

    with subtests.test(msg="seeding says so unless asked to be quiet"):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="spacr.runctx"):
            seed_everything(31, quiet=True)
            assert caplog.records == []
            loud = seed_everything(31, quiet=False)
        assert [r.getMessage() for r in caplog.records] == [str(loud)]
        assert str(loud).startswith("seed 31 → python, numpy, numpy.Generator")


# ---------------------------------------------------------------------------
# the streams
# ---------------------------------------------------------------------------

def _draw(generator):
    return generator.random(4).tolist()


def test_spacr_rng_streams_are_reproducible_and_independent(subtests,
                                                             monkeypatch):
    """spacr_rng: explicit seed, run seed, and the never-seeded default."""
    def fixed(seed, stream=""):
        root = np.random.SeedSequence(seed)
        if not stream:
            return _draw(np.random.default_rng(root))
        key = int(hashlib.sha256(stream.encode()).hexdigest()[:8], 16)
        return _draw(np.random.default_rng(
            np.random.SeedSequence(entropy=seed, spawn_key=(key,))))

    with subtests.test(msg="an explicit seed overrides the run seed"):
        seed_everything(5, quiet=True)
        assert _draw(spacr_rng("fold0", seed=99)) == fixed(99, "fold0")
        assert _draw(spacr_rng("fold0", seed=99)) == fixed(99, "fold0")
        assert _draw(spacr_rng(seed=99)) == fixed(99)

    with subtests.test(msg="a seeded run's root is used without an override"):
        seed_everything(5, quiet=True)
        assert _draw(spacr_rng("fold0")) == fixed(5, "fold0")
        assert _draw(spacr_rng()) == fixed(5)

    with subtests.test(msg="never seeded falls back to the default seed"):
        monkeypatch.setattr(runctx, "_ROOT_SEED_SEQUENCE", None)
        monkeypatch.delenv(runctx.SEED_ENV, raising=False)
        unnamed = _draw(spacr_rng())
        assert unnamed == fixed(DEFAULT_SEED)
        assert _draw(spacr_rng()) == unnamed
        named = _draw(spacr_rng("worker-3"))
        assert named == fixed(DEFAULT_SEED, "worker-3")
        assert named != unnamed

    with subtests.test(msg="never seeded follows SPACR_SEED when it is set"):
        monkeypatch.setattr(runctx, "_ROOT_SEED_SEQUENCE", None)
        monkeypatch.setenv(runctx.SEED_ENV, "8")
        assert _draw(spacr_rng()) == fixed(8)
        monkeypatch.delenv(runctx.SEED_ENV)
