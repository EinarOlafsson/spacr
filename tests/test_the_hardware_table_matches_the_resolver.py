"""The README's hardware table agrees with the resolver that produced it.

Instruction 323. The table says what each hardware configuration
accelerates, and it is GENERATED from
``spacr.accelerator.capabilities()`` with each backend's probe faked --
the same trick the resolver's own tests use to exercise 19 backends on a
machine that has one.

Generated rather than typed because a typed table is wrong within a day.
Instruction 321 has the precedent: the module grid carried a hand-written
copy of Home's layout and went stale the first time Home changed.

The table also has to be RIGHT in a way a regeneration cannot check --
that its three colours mean what the legend says. Those are the
assertions below with reasons attached.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.rst"
GENERATED = ROOT / "docs" / "source" / "_generated" / "hardware_table.rst"

GREEN, PURPLE, RED = "\U0001F7E2", "\U0001F7E3", "\U0001F534"


@pytest.fixture(scope="module")
def generator():
    """The generator module, loaded from packaging/ by path."""
    spec = importlib.util.spec_from_file_location(
        "_readme_visuals", ROOT / "packaging" / "generate_readme_visuals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def table(generator):
    return generator._hardware_table()


def test_the_readme_holds_what_the_generator_produces(table):
    """Regenerating changes nothing. If it does, somebody typed.

    This is the check that keeps the table honest with no effort: it
    fails the moment the resolver's answers change and the generator has
    not been re-run.
    """
    assert table.strip() in README.read_text(encoding="utf-8"), (
        "README.rst disagrees with the generated hardware table; run "
        "python packaging/generate_readme_visuals.py")


def test_the_generated_file_matches_too(table):
    assert GENERATED.read_text(encoding="utf-8").strip() == table.strip()


def _row(table_text, label):
    """The cells of one hardware row, in column order."""
    body = table_text[table_text.index(f"* - {label}"):]
    body = body[:body.index("* - ", 4)] if "* - " in body[4:] else body
    return re.findall(r"^     - (.+)$", body, re.M)


def test_cuda_is_green_across_the_board(table):
    """The only configuration with years behind it."""
    assert _row(table, "NVIDIA (CUDA)") == [f"{GREEN} GPU"] * 3


def test_the_new_backends_are_purple_not_green(table):
    """Implemented is not the same claim as exercised.

    Metal on an Intel Mac was measured -- 444.5 s to 3.2 s -- but on one
    machine, one day old. ROCm, XPU and Apple Silicon Metal were measured
    on nothing, because no such hardware was available. Painting them
    green because the code exists is what the beta tier prevents.
    """
    for label in ("AMD on Linux (ROCm)", "AMD in an Intel Mac (Metal)",
                  "Apple Silicon (Metal)", "Intel Arc/Xe (XPU)"):
        cells = _row(table, label)
        assert cells[0] == f"{PURPLE} GPU", label
        assert cells[1] == f"{PURPLE} GPU", label


def test_umap_is_red_on_every_gpu_that_is_not_cuda(table):
    """cuML ships for CUDA only, so there is nothing to wait for.

    RED, not "CPU in green": on a machine that HAS a GPU, a task that
    cannot use it is unsupported on that hardware. This is the cell the
    table exists for.
    """
    for label in ("AMD on Linux (ROCm)", "AMD in an Intel Mac (Metal)",
                  "Apple Silicon (Metal)", "Intel Arc/Xe (XPU)"):
        assert _row(table, label)[2] == f"{RED} CPU", label


def test_the_no_gpu_row_is_green_everywhere(table):
    """THE MOST DANGEROUS CELL IN THE TABLE, if it were red.

    Its cells say CPU because that is what they use. Red would say spaCR
    does not support running without a GPU -- false, and the most
    damaging claim this table could make. Every task runs and every
    result is identical; only the clock changes.
    """
    assert _row(table, "No GPU") == [f"{GREEN} CPU"] * 3


def test_the_intel_mac_row_stands_on_its_own(table):
    """It must not be merged into either neighbour.

    That grouping is the mistake instruction 319's own backend table
    made -- Metal filed under Apple Silicon, AMD under ROCm -- and since
    ROCm has no macOS build, the configuration that actually works was
    named nowhere. It hid a 139x speedup until somebody measured it.
    """
    text = README.read_text(encoding="utf-8")
    assert "AMD in an Intel Mac (Metal)" in text
    assert "Apple Silicon (Metal)" in text


def test_the_legend_names_all_three_marks(table):
    """A legend describing a mark the table never uses, or a mark with no
    legend entry, is worse than no legend."""
    used = set(re.findall(f"[{GREEN}{PURPLE}{RED}]", table))
    legend = table[table.index("supported (stable)") - 5:]
    for mark in used:
        assert mark in legend, f"{mark} is used but not in the legend"
    for mark in (GREEN, PURPLE, RED):
        assert mark in used, f"{mark} is in the legend but never used"


def test_a_column_that_lost_its_capability_row_fails_loudly(generator,
                                                            monkeypatch):
    """Renaming a task in `capabilities()` must not empty a column.

    Silently blank cells are the failure mode of deriving from another
    module's strings, so the generator refuses instead.
    """
    monkeypatch.setattr(generator, "HARDWARE_TASKS",
                        (("Nonsense", "NoSuchTaskPrefix"),))
    with pytest.raises(ValueError, match="diverged"):
        generator._hardware_table()
