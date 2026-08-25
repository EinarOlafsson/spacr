"""``DesignSpec.validate`` names every parameter that stops the sweep.

The sweep is minutes of simulation. A design that cannot run has to say so
on the form, in words naming the field, before any of that is spent -- and
it has to report *all* the problems at once, because fixing them one refusal
at a time is how a user gives up on a power calculation.

Each test here builds a design that is wrong in exactly one way and asserts
the matching message appears while the default design stays clean.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.power_design import DesignSpec  # noqa: E402


def _problems(**over):
    return DesignSpec(**over).validate()


def _joined(**over):
    return " | ".join(_problems(**over))


def test_the_default_design_has_nothing_to_complain_about():
    """The baseline every other test measures against must itself be legal."""
    assert DesignSpec().validate() == []


def test_a_library_of_one_gene_is_refused():
    """With one library unit there is no ranking, so AUROC means nothing."""
    assert "at least 2 genes" in _joined(n_genes=1)


def test_zero_guides_per_gene_is_refused():
    """``score_per='guide'`` would build a library of no constructs at all."""
    assert "at least 1 guide per gene" in _joined(n_grnas_per_gene=0)


def test_an_unknown_imaging_split_names_the_two_that_exist():
    """The split decides how cells are apportioned; there is no third rule."""
    message = _joined(imaging_split="even")
    assert "imaging_split" in message
    assert "'even'" in message
    assert "abundance" in message and "uniform" in message


def test_zero_cells_imaged_per_well_is_refused():
    """No imaged cells means no positive-rate observation to fit."""
    assert "Cells imaged per well" in _joined(cells_per_well=0.0)


def test_a_negative_cell_count_is_refused_too():
    """The bound is strictly positive, not merely non-negative."""
    assert "Cells imaged per well" in _joined(cells_per_well=-5.0)


def test_zero_constructs_per_well_is_refused():
    """A well with no library units carries no genotype to score."""
    assert "Constructs per well" in _joined(constructs_per_well=0.0)


@pytest.mark.parametrize("hit_rate", [-0.1, 1.5])
def test_a_hit_prevalence_outside_zero_to_one_is_refused(hit_rate):
    """Prevalence is a fraction of the library; outside [0, 1] it is not one."""
    assert "fraction between 0 and 1" in _joined(hit_rate=hit_rate)


def test_a_positive_rate_above_one_is_refused_before_its_variance_is_judged():
    """A rate outside [0, 1] makes the Bernoulli variance bound meaningless.

    The bound is ``mean * (1 - mean)``, which goes negative for a mean above
    1, so every variance would look "too spread out". The range check has to
    fire first and skip the variance message, or the user is told to lower a
    variance that is not the problem.
    """
    problems = _problems(background_positive_rate=1.5)
    background = [p for p in problems if "background" in p.lower()]
    assert background == ["The background positive rate has to be in [0, 1]."]


def test_a_negative_rate_is_reported_for_both_the_background_and_the_hit():
    """The hit rate is derived from the background, so one typo breaks both."""
    problems = _problems(background_positive_rate=-0.2)
    ranged = [p for p in problems if "has to be in [0, 1]" in p]
    assert len(ranged) == 2
    assert any("background" in p.lower() for p in ranged)
    assert any("hit-cell" in p.lower() for p in ranged)


def test_negative_reads_per_well_is_refused():
    """Sequencing depth is a count; a negative one has no simulation."""
    assert "Reads per well cannot be negative" in _joined(reads_per_well=-1.0)


def test_every_problem_is_reported_in_one_pass():
    """Refusing one field at a time is how a power calculation gets abandoned."""
    problems = _problems(n_genes=1, n_grnas_per_gene=0,
                         cells_per_well=0.0, constructs_per_well=0.0,
                         reads_per_well=-1.0)
    assert len(problems) >= 5
