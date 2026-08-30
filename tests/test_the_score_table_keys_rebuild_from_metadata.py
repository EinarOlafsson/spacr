"""Building the join key for a classifier's score table, and the last fallback.

The comment above the fallback records the bug it fixed: an XGBoost score file
carries plate/row/column/field and an object id under plainer spellings, and
without this branch the two sides of the join were asymmetric -- the database
could rebuild the key and the results frame could not, so the file matched zero
rows and read as "no per-object score".

That is the failure this file protects: not a crash, but a join that quietly
matches nothing.
"""
from __future__ import annotations

import pandas as pd
import pytest


def test_an_existing_prcfo_column_is_used_directly():
    """The first route, which costs nothing."""
    from spacr.predictions import _result_keys

    results = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"], "pred": [0.9]})

    keys = _result_keys("prcfo", results, timelapse=False)

    assert keys is not None
    assert len(keys) == 1


def test_a_score_table_with_only_metadata_columns_still_gets_a_key():
    """The fallback the comment is about: plainer spellings, canonicalised.

    An ml_analysis score CSV has no prcfo and no path column, only the
    metadata under names like plate/row/column/field. Rebuilding the key from
    those is what makes the join symmetric.
    """
    from spacr.predictions import _result_keys

    results = pd.DataFrame({
        "plateID": ["plate1", "plate1"],
        "rowID": ["r1", "r1"],
        "columnID": ["c1", "c2"],
        "fieldID": ["1", "1"],
        "objectID": ["o1", "o1"],
        "pred": [0.9, 0.2],
    })

    keys = _result_keys("prcfo", results, timelapse=False)

    assert keys is not None
    assert len(keys) == 2
    assert keys.nunique() == 2, "two different wells must give two keys"


def test_a_frame_that_cannot_be_canonicalised_yields_no_key(monkeypatch):
    """Lines 501-502: None rather than an exception.

    ``canonicalise_columns`` renames columns and can refuse -- a
    case-insensitive collision, for instance. Returning None means "this
    frame has no key", which the caller reports as no per-object score. An
    exception here would take down a classify run over a score file it could
    have skipped.
    """
    from spacr import predictions
    from spacr import schema

    def refuse(_frame):
        raise ValueError("two columns would collide when canonicalised")

    monkeypatch.setattr(schema, "canonicalise_columns", refuse)

    results = pd.DataFrame({"plateID": ["plate1"], "pred": [0.9]})

    assert predictions._result_keys("prcfo", results, timelapse=False) is None


def test_a_frame_with_no_usable_metadata_yields_no_key():
    """The same None by the ordinary route: there is simply nothing to build from."""
    from spacr.predictions import _result_keys

    results = pd.DataFrame({"pred": [0.9], "something_else": [1]})

    assert _result_keys("prcfo", results, timelapse=False) is None


@pytest.mark.parametrize("id_column", [
    "cell_id", "nucleus_id", "pathogen_id", "cytoplasm_id", "object",
    "organelle_id", "organelleb_id", "organellec_id", "organelled_id",
    "objectID",
])
def test_every_object_id_spelling_rebuilds_the_key(id_column):
    """Every spelling spaCR writes must rebuild, including the organelle roles.

    The list was a hand-written copy of spacr.utils.PNG_OBJECT_ID_COLUMNS and
    had drifted: the four organelle roles were absent, so an organelle-mode
    score table rebuilt no key and its join matched zero rows -- the exact
    failure the comment in _result_keys says was fixed for the plainer
    spellings. The canonical schema.OBJECT_KEY was missing too.

    Parametrised over the whole vocabulary on purpose: a list that can drift
    needs a test that walks it, not one that samples it.
    """
    from spacr.predictions import _prcfo_from_metadata

    frame = pd.DataFrame({
        "plateID": ["plate1"], "rowID": ["r1"], "columnID": ["c1"],
        "fieldID": ["1"], id_column: ["o1"],
    })

    keys = _prcfo_from_metadata(frame)

    assert keys is not None, f"{id_column} did not rebuild a key"
    assert keys.iloc[0] == "plate1_r1_c1_1_o1"


def test_the_object_id_vocabulary_covers_every_crop_mode():
    """The two lists must not drift apart again.

    spacr.utils.PNG_OBJECT_ID_COLUMNS is what filepaths_to_database WRITES;
    _OBJECT_ID_COLUMNS is what this module READS. Anything written and not
    read is a score file that silently matches nothing.
    """
    from spacr.predictions import _OBJECT_ID_COLUMNS
    from spacr.utils import PNG_OBJECT_ID_COLUMNS

    written = set(PNG_OBJECT_ID_COLUMNS.values())
    read = set(_OBJECT_ID_COLUMNS)

    assert written <= read, f"written but never read: {sorted(written - read)}"
