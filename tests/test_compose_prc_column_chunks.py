"""compose_prc_column builds its keys in slices and gives the same keys.

Object-dtype concatenation over a whole frame is one C call holding the
interpreter lock; on lesson 13's regression reads it starved the window for
0.46 s. The keys must stay byte-identical to the one-pass build.
"""

import numpy as np
import pandas as pd
import pytest

from spacr import schema


def _one_pass(df, columns=None):
    plate, row, column = tuple(columns or schema.WELL_KEY_COLUMNS)
    return (df[plate].astype(str).map(schema.escape_filename_component)
            + schema.KEY_SEPARATOR + df[row].astype(str)
            + schema.KEY_SEPARATOR + df[column].astype(str))


def _frame(n, seed=0):
    rng = np.random.default_rng(seed)
    plate = rng.choice(np.array(['plate1', 'plate_2', 'p%3', 'plate 4', None, np.nan],
                                dtype=object), n)
    row = rng.choice(np.array(['r1', 2, 3.0, np.nan, 'r_5', True], dtype=object), n)
    column = rng.integers(1, 25, n).astype(float)
    column[rng.random(n) < 0.05] = np.nan
    index = pd.Index(rng.permutation(np.arange(10, 10 + 3 * n, 3)))
    return pd.DataFrame({'plateID': plate, 'rowID': row, 'columnID': column},
                        index=index)


def _same(result, expected):
    pd.testing.assert_series_equal(result, expected, check_exact=True)
    assert result.dtype == expected.dtype
    assert result.index.equals(expected.index)
    assert result.tolist() == expected.tolist()


@pytest.mark.parametrize('n', [0, 1, 7, 49_999, 50_000, 50_001, 123_457])
def test_sliced_keys_equal_the_one_pass_keys(n):
    df = _frame(n, seed=n)
    _same(schema.compose_prc_column(df), _one_pass(df))


@pytest.mark.parametrize('dtypes', [
    ('object', 'object', 'object'),
    ('int64', 'int64', 'int64'),
    ('float64', 'object', 'int64'),
    ('category', 'category', 'Int64'),
])
def test_every_kind_of_key_column_composes_identically(dtypes):
    n = 120_001
    rng = np.random.default_rng(3)
    data = {
        'p': rng.integers(1, 4, n), 'r': rng.integers(1, 17, n),
        'c': rng.integers(1, 25, n)}
    df = pd.DataFrame(data, index=pd.RangeIndex(5, 5 + n))
    for name, dtype in zip(('p', 'r', 'c'), dtypes):
        if dtype == 'object':
            df[name] = df[name].map(lambda value: f'x{value}' if value % 2 else value).astype(object)
        else:
            df[name] = df[name].astype(dtype)
    if dtypes[2] == 'Int64':
        df.loc[df.index[::97], 'c'] = pd.NA
    _same(schema.compose_prc_column(df, columns=('p', 'r', 'c')),
          _one_pass(df, columns=('p', 'r', 'c')))


def test_an_empty_frame_gives_an_empty_object_series():
    df = pd.DataFrame({'plateID': pd.Series([], dtype=object),
                       'rowID': pd.Series([], dtype=float),
                       'columnID': pd.Series([], dtype='int64')})
    _same(schema.compose_prc_column(df), _one_pass(df))


def test_no_string_step_spans_more_than_one_slice(monkeypatch):
    sizes = []
    real = schema._compose_prc_chunk

    def recorded(plate, row, column):
        assert len(plate) == len(row) == len(column)
        sizes.append(len(plate))
        return real(plate, row, column)

    monkeypatch.setattr(schema, '_compose_prc_chunk', recorded)
    df = _frame(260_000, seed=9)
    schema.compose_prc_column(df)
    assert sum(sizes) == len(df)
    assert max(sizes) <= schema._PRC_CHUNK_ROWS
    assert len(sizes) == -(-len(df) // schema._PRC_CHUNK_ROWS)


def test_a_missing_key_column_still_raises():
    with pytest.raises(KeyError, match='rowID'):
        schema.compose_prc_column(pd.DataFrame({'plateID': [1], 'columnID': [2]}))
