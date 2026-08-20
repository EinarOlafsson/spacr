"""A setting that names a column is canonicalised too, not only the frame.

FROM ~/.spacr/logs/spacr.log, on the default regression path:

    KeyError: 'ColumnID'
      ml.perform_regression -> _graph_sequencing_stats
      -> sequencing.graph_sequencing_stats
      -> df = df[df[settings['filter_column']] != c]

`graph_sequencing_stats` renames the frame's headers to spaCR's vocabulary on
the line above that one, so the frame holds `columnID`. The SETTING still held
the user's own spelling, `ColumnID`, and indexed a column that no longer existed
under that name -- four frames deep, after the counts had been read.

Instruction 145's rule is one vocabulary. Applying it to the data and not to the
setting that indexes the data is half a rule.
"""

import spacr


import pandas as pd
import pytest

from spacr.sequencing import _resolve_column


def _counts():
    return pd.DataFrame({"plateID": ["p1"], "rowID": ["r1"],
                         "columnID": ["c1"], "grna": ["g1"], "count": [10]})


def test_the_frames_own_spelling_passes_through():
    assert _resolve_column(_counts(), "columnID") == "columnID"


def test_the_users_spelling_is_resolved():
    """THE REPORTED CRASH. A settings CSV written against the original headers
    names a column that renaming has since moved."""
    assert _resolve_column(_counts(), "ColumnID") == "columnID"
    assert _resolve_column(_counts(), "COLUMNID") == "columnID"
    assert _resolve_column(_counts(), " columnID ") == "columnID"


def test_a_column_that_is_genuinely_absent_names_what_is_there():
    """A KeyError four frames down says only the name that was ASKED for,
    which is the one piece of information the user already had."""
    with pytest.raises(ValueError) as caught:
        _resolve_column(_counts(), "wellID")
    message = str(caught.value)
    assert "wellID" in message
    assert "columnID" in message, "the refusal must name what the frame HAS"
    assert "renames headers" in message, "and why the name may have moved"


def test_no_filter_column_is_its_own_message():
    with pytest.raises(ValueError) as caught:
        _resolve_column(_counts(), "")
    assert "No filter_column" in str(caught.value)


def test_the_call_site_goes_through_the_resolver():
    """A second call site indexing the frame directly is this bug again."""
    import inspect

    from spacr import sequencing

    source = inspect.getsource(sequencing.graph_sequencing_stats)
    assert "_resolve_column(df, settings.get('filter_column'))" in source
    assert "df[settings['filter_column']]" not in source
