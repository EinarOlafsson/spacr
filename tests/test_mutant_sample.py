"""The mutant sampler plants fixed, reproducible bugs without editing files.

``tools/mutant_sample.py`` is how the 2026-11-12 test-suite reduction proves
that a smaller test set still catches at least as many planted bugs as the
tests it replaces. These tests run it on a two-function toy package in a
temporary directory, never on spaCR, so they do not depend on how well the
real suite covers anything.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "mutant_sample", ROOT / "tools" / "mutant_sample.py")
assert _SPEC is not None and _SPEC.loader is not None
mutant_sample = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mutant_sample
_SPEC.loader.exec_module(mutant_sample)

CALC = textwrap.dedent('''
    """A toy module: one function the tests use, one they never call."""


    def clamp(value, low=0, high=10):
        """Keep value between low and high."""
        if value < low:
            return low
        if value > high:
            return high
        return value


    def unused(flag=True):
        """Nobody calls this, so no test can kill a mutant in it."""
        return 2 if flag else 3
''')

STRONG = textwrap.dedent('''
    from toy.calc import clamp


    def test_clamp_holds_both_ends_and_the_middle():
        assert clamp(-5) == 0
        assert clamp(50) == 10
        assert clamp(4) == 4
        assert clamp(0) == 0
        assert clamp(10) == 10
        assert clamp(3, low=2, high=8) == 3
''')

WEAK = textwrap.dedent('''
    from toy.calc import clamp


    def test_clamp_returns_something():
        assert clamp(4) is not None
''')


@pytest.fixture
def toy(tmp_path):
    """A package ``toy`` with ``calc.py`` and two test files beside it."""
    (tmp_path / "toy").mkdir()
    (tmp_path / "toy" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "toy" / "calc.py").write_text(CALC, encoding="utf-8")
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "test_strong.py").write_text(STRONG, encoding="utf-8")
    (tmp_path / "checks" / "test_weak.py").write_text(WEAK, encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return tmp_path


def test_every_kind_is_found_and_docstrings_are_left_alone():
    mutants = mutant_sample.enumerate_mutants(CALC)
    kinds = {mutant.kind for mutant in mutants}

    assert kinds == {"compare", "constant", "return", "negate"}
    assert all(mutant.line > 3 for mutant in mutants)
    assert len({mutant.id for mutant in mutants}) == len(mutants)
    compare = [mutant for mutant in mutants if mutant.kind == "compare"]
    assert [mutant.description for mutant in compare] == ["Lt -> GtE", "Gt -> LtE"]


@pytest.mark.parametrize(
    ("kind", "before", "after"),
    [
        ("compare", "if value < low:", "if value >= low:"),
        ("constant", "def clamp(value, low=0, high=10):",
         "def clamp(value, low=1, high=10):"),
        ("return", "return low", "return None"),
        ("negate", "if value < low:", "if not value < low:"),
    ],
)
def test_one_mutant_changes_exactly_one_thing(kind, before, after):
    mutant = next(m for m in mutant_sample.enumerate_mutants(CALC) if m.kind == kind)
    clean = [line.strip() for line in
             mutant_sample.ast.unparse(mutant_sample.ast.parse(CALC)).splitlines()]
    mutated = [line.strip() for line in
               mutant_sample.apply_mutant(CALC, mutant).splitlines()]

    changed = [(a, b) for a, b in zip(clean, mutated) if a != b]
    assert changed == [(before, after)]


def test_a_mutant_that_is_not_in_the_source_is_refused():
    ghost = mutant_sample.Mutant("compare:999:0:0", "compare", 999, 0, "")

    with pytest.raises(ValueError, match="does not exist"):
        mutant_sample.apply_mutant(CALC, ghost)


def test_the_same_seed_plants_the_same_sample_and_another_seed_may_not():
    mutants = mutant_sample.enumerate_mutants(CALC)

    first = mutant_sample.sample_mutants(mutants, seed=3, count=4)
    again = mutant_sample.sample_mutants(mutants, seed=3, count=4)
    others = {tuple(m.id for m in mutant_sample.sample_mutants(mutants, seed=s, count=4))
              for s in range(10)}

    assert first == again
    assert len(first) == 4
    assert len(others) > 1
    assert mutant_sample.sample_mutants(mutants, seed=0, count=999) == mutants
    only = mutant_sample.sample_mutants(mutants, seed=0, count=99, kinds=["return"])
    assert {m.kind for m in only} == {"return"}


def test_lines_limit_the_sites():
    mutants = mutant_sample.enumerate_mutants(CALC, lines=[7])

    assert {(mutant.line, mutant.kind) for mutant in mutants} == {
        (7, "compare"), (7, "negate")}


def test_old_and_new_test_sets_are_compared_on_the_same_mutants(toy):
    calc = toy / "toy" / "calc.py"
    before = hashlib.sha256(calc.read_bytes()).hexdigest()
    lines = [5, 7, 8, 9, 10, 11]

    report = mutant_sample.run_sample(
        module=calc, root=toy, seed=1, count=6, lines=lines, timeout=120,
        test_sets={"strong": ["checks/test_strong.py"],
                   "weak": ["checks/test_weak.py"]},
    )

    assert hashlib.sha256(calc.read_bytes()).hexdigest() == before
    assert report["status"] == "ok"
    assert report["import_name"] == "toy.calc"
    assert {run["outcome"] for run in report["baseline"].values()} == {"survived"}
    strong, weak = report["summary"]["strong"], report["summary"]["weak"]
    assert strong["total"] == weak["total"] == 6
    assert strong["killed"] == 6
    assert weak["killed"] < strong["killed"]
    for row in report["mutants"]:
        assert set(row["results"]) == {"strong", "weak"}
    text = mutant_sample.render_text(report)
    assert "strong: killed 6 of 6" in text


def test_a_mutant_in_code_no_test_runs_survives(toy):
    calc = toy / "toy" / "calc.py"

    report = mutant_sample.run_sample(
        module=calc, root=toy, seed=0, count=1, lines=[16], kinds=["constant"],
        timeout=120, test_sets={"strong": ["checks/test_strong.py"]},
    )

    assert [row["results"]["strong"]["outcome"] for row in report["mutants"]] == [
        "survived"]


def test_tests_that_never_import_the_module_are_named_not_imported(toy):
    (toy / "checks" / "test_unrelated.py").write_text(
        "def test_arithmetic():\n    assert 1 + 1 == 2\n", encoding="utf-8")

    report = mutant_sample.run_sample(
        module=toy / "toy" / "calc.py", root=toy, seed=0, count=1,
        timeout=120, test_sets={"unrelated": ["checks/test_unrelated.py"]},
    )

    assert report["status"] == "baseline_failed"
    assert report["baseline"]["unrelated"]["outcome"] == "not_imported"
    assert "nothing was compared" in mutant_sample.render_text(report)


def test_a_test_set_failing_on_the_clean_module_stops_the_comparison(toy):
    (toy / "checks" / "test_broken.py").write_text(
        "from toy.calc import clamp\n\n\ndef test_wrong():\n"
        "    assert clamp(50) == 50\n", encoding="utf-8")

    report = mutant_sample.run_sample(
        module=toy / "toy" / "calc.py", root=toy, seed=0, count=3,
        timeout=120, test_sets={"broken": ["checks/test_broken.py"]},
    )

    assert report["status"] == "baseline_failed"
    assert report["baseline"]["broken"]["outcome"] == "killed"
    assert report["mutants"] == []
