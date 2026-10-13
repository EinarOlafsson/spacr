"""Item 288: the Compare panel's top hits from awkward result tables.

``spacr.well_scope._top_hits`` picks the guides the regression Compare panel
opens on. Three inputs the ordinary tests do not give it:

* a results file that cannot be read gives no hits, rather than an error
  in the panel;
* a results table with no p-value column gives no hits -- ranking by
  anything else would pre-select guides that are not significant;
* a guide-level term naming a guide the object table spells with a library
  prefix (``lib1_TGGT1_2_1`` for ``TGGT1_2_1``) selects that guide, and only
  that one of its gene's guides.
"""
from __future__ import annotations

import pandas as pd

from spacr.well_scope import _top_hits


def _objects():
    return pd.DataFrame({"prc": ["a", "b", "c"],
                         "grna": ["lib1_TGGT1_2_1", "lib1_TGGT1_2_2",
                                  "lib1_TGGT1_3_1"]})


def test_an_unreadable_results_file_gives_no_hits(tmp_path):
    broken = tmp_path / "results.parquet"
    broken.write_text("this is not a parquet file")
    assert _top_hits(_objects(), str(broken)) == []


def test_a_table_without_p_values_gives_no_hits():
    table = pd.DataFrame({"feature": ["grna[T.TGGT1_2_1]"],
                          "coefficient": [2.0]})
    assert _top_hits(_objects(), table) == []


def test_a_prefixed_guide_is_found_by_its_bare_name():
    table = pd.DataFrame({"feature": ["grna[T.TGGT1_2_1]"],
                          "p_value": [0.001]})
    assert _top_hits(_objects(), table) == ["lib1_TGGT1_2_1"]
