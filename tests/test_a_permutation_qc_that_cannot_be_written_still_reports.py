"""Item 288: two refusals made in words rather than silence.

* The regression's exchangeability report is a courtesy. When its figure
  and JSON cannot be written to the results folder, the report is still
  returned and the reason is printed -- the run's results are not lost to a
  QC figure.
* A mask whose labels are not numbers at all (strings, objects) is refused
  before it is written as uint16, rather than cast to garbage.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import mask_io, ml
from tests.test_cov_12_ml_reporting import long_guide_frame


def test_a_qc_folder_that_cannot_be_written_is_reported_not_raised(
        tmp_path, capsys):
    blocked = tmp_path / "results"
    blocked.write_text("a file where the results folder should be")
    report = ml._report_exchangeability(
        long_guide_frame(row_effect=0.0, seed=0), "score",
        {"guide_permutation_block": "plateID",
         "guide_nuisance_columns": ["rowID", "columnID"]}, str(blocked))
    printed = capsys.readouterr().out
    assert report is not None and report["n"] == 18
    assert "qc" not in report
    assert "Permutation QC could not be written:" in printed
    assert "Exchangeability: nothing found" in printed


@pytest.mark.parametrize("labels", [np.array([["a", "b"]]),
                                    np.array([[None, 1]], dtype=object)])
def test_labels_that_are_not_numbers_are_refused(labels):
    with pytest.raises(ValueError) as excinfo:
        mask_io._as_uint16_mask(labels)
    assert "real nonnegative integers" in str(excinfo.value)
