"""Promoting hit calls onto a png_list that has no object key is refused.

The store/promote/revert round trip lives in
``test_hit_attribution_store_and_crossfit.py``.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.hit_attribution import HitAttributionError, promote_calls


# --------------------------------------------------------------------------- #
#  promote_calls -- and the audit that makes it undoable
# --------------------------------------------------------------------------- #

def test_promoting_refuses_a_png_list_with_no_object_key(tmp_path):
    """Without prcfo there is nothing to promote ONTO."""
    path = tmp_path / "bad.db"
    with sqlite3.connect(str(path)) as db:
        pd.DataFrame({"png_path": ["/x.png"]}).to_sql("png_list", db,
                                                      index=False)
    with pytest.raises(HitAttributionError) as caught:
        promote_calls(str(path), "run-1", "annotation")
    assert "prcfo" in str(caught.value)
