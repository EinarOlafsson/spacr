"""Table-driven test of ``spacr.training_basis.resolve_basis``, the dataset-mode dispatch of training dataset generation.

The retired ``measurement`` basis migrates to ``annotation`` before the
dispatch, so every accepted spelling resolves to one of the two bases the
dispatch handles, and anything else is refused with a message naming the
value typed and both legal bases.
"""
from __future__ import annotations

import pytest

from spacr.training_basis import (RETIRED_BASES, TRAINING_BASES,
                                  TrainingBasisError, resolve_basis)


def test_resolve_basis_answers_one_of_the_two_bases_or_refuses_by_name(
        subtests):
    """Explicit modes win, the retired one migrates, an annotation column implies annotation, else metadata."""
    assert TRAINING_BASES == ("metadata", "annotation")
    assert RETIRED_BASES == {"measurement": "annotation"}
    accepted = [
        ("metadata", {"dataset_mode": "metadata"}, "metadata"),
        ("annotation", {"dataset_mode": "annotation"}, "annotation"),
        ("retired-measurement", {"dataset_mode": "measurement"}, "annotation"),
        ("case-and-space", {"dataset_mode": "  Metadata "}, "metadata"),
        ("mode-beats-annotation-column",
         {"dataset_mode": "metadata", "annotation_column": "x"}, "metadata"),
        ("implicit-annotation", {"annotation_column": "x"}, "annotation"),
        ("empty-mode-falls-through", {"dataset_mode": "",
                                      "annotation_column": "x"},
         "annotation"),
        ("nothing-set", {}, "metadata"),
        ("empty-annotation-column", {"annotation_column": ""}, "metadata"),
    ]
    for case_id, settings, basis in accepted:
        with subtests.test(msg=case_id):
            assert resolve_basis(settings) == basis

    for typed in ("not-real", "Measurements"):
        with subtests.test(msg=f"refused-{typed}"):
            with pytest.raises(TrainingBasisError) as raised:
                resolve_basis({"dataset_mode": typed})
            assert str(raised.value) == (
                f"dataset_mode={typed!r} is not one of "
                "['metadata', 'annotation']. A run cannot guess which labels "
                "were meant.")
            assert isinstance(raised.value, ValueError)
