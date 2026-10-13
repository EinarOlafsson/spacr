"""The image filter could only say what a path MUST contain (issue #7).

It was one substring matched with ``LIKE %x%``, so there was no way to ask
for the complement -- "the cells with no pathogen crop" -- which is half of
most comparisons. The request was for NOT, and for AND/OR alongside it.

The grammar is deliberately close to what someone would type::

    pathogen                  contains "pathogen"
    !pathogen                 does NOT contain it
    NOT pathogen              the same, spelled out
    cell AND nucleus          both
    cell OR nucleus           either
    cell AND NOT pathogen     mixed

EVERY TERM IS A BOUND PARAMETER. This filter goes straight into a WHERE
clause, so a path fragment containing a quote has to be a path fragment.
"""

import sqlite3

import pytest

pytest.importorskip("PySide6")

from spacr.qt.annotate_engine import count_rows, parse_image_type


# ---------------------------------------------------------------------------
# the grammar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("expression", "expect_params"), [
    ("pathogen", ["%pathogen%"]),
    ("!pathogen", ["%pathogen%"]),
    ("NOT pathogen", ["%pathogen%"]),
    ("cell AND nucleus", ["%cell%", "%nucleus%"]),
    ("cell OR nucleus", ["%cell%", "%nucleus%"]),
    ("cell AND NOT pathogen", ["%cell%", "%pathogen%"]),
])
def test_every_term_is_a_bound_parameter(expression, expect_params):
    sql, params = parse_image_type(expression)
    assert params == expect_params
    for term in expect_params:
        assert term.strip("%") not in sql, (
            f"{term!r} was interpolated into the SQL instead of bound")


def test_an_empty_filter_selects_everything():
    assert parse_image_type("") == ("", [])
    assert parse_image_type(None) == ("", [])
    assert parse_image_type("   ") == ("", [])


def test_and_binds_tighter_than_or():
    sql, _ = parse_image_type("a OR b AND c")
    assert sql == "(png_path LIKE ? OR (png_path LIKE ? AND png_path LIKE ?))"


def test_parentheses_override_precedence():
    sql, params = parse_image_type("(a OR b) AND NOT c")
    assert params == ["%a%", "%b%", "%c%"]
    assert sql.index("OR") < sql.index("AND")


@pytest.mark.parametrize("bad", ["NOT", "a AND", "a OR", "(a", "AND b"])
def test_an_unreadable_filter_says_what_was_wrong(bad):
    """Silently selecting everything would be the worst outcome: the user
    would annotate the wrong population and never know."""
    with pytest.raises(ValueError):
        parse_image_type(bad)


# ---------------------------------------------------------------------------
# against a real database
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    path = tmp_path / "measurements.db"
    conn = sqlite3.connect(str(path))
    conn.execute('CREATE TABLE "png_list" (png_path TEXT, annotate INTEGER)')
    conn.executemany('INSERT INTO "png_list" VALUES (?, ?)', [
        ("/p/cell_png/a_cell.png", None),
        ("/p/cell_png/b_cell.png", None),
        ("/p/pathogen_png/c_pathogen.png", None),
        ("/p/nucleus_png/d_nucleus.png", None),
    ])
    conn.commit()
    conn.close()
    return str(path)


@pytest.mark.parametrize(("expression", "expected"), [
    (None, 4),
    ("cell", 2),
    ("!cell", 2),
    ("NOT cell", 2),
    ("pathogen", 1),
    ("!pathogen", 3),
    ("cell OR pathogen", 3),
    ("cell AND png", 2),
    ("png AND NOT cell", 2),
])
def test_the_filter_selects_the_rows_it_says(db, expression, expected):
    assert count_rows(db, expression) == expected


def test_a_quote_in_the_filter_is_a_path_fragment_not_an_injection(db):
    """The parameterisation, demonstrated rather than asserted.

    A single token reaches SQL as a bound parameter and simply matches
    nothing. A multi-token payload does not even get that far -- the parser
    refuses to read it -- which is the stronger outcome of the two.
    """
    # One token: bound, matched against, matches nothing.
    assert count_rows(db, "';DROP") == 0

    # Several tokens: refused before any SQL is built.
    with pytest.raises(ValueError):
        count_rows(db, "'; DROP TABLE png_list; --")

    # Either way the table is untouched.
    assert count_rows(db, None) == 4


def test_a_path_fragment_that_looks_like_sql_still_matches_its_path(db, tmp_path):
    """The other half: a quote in a filename must still be findable."""
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute('INSERT INTO "png_list" VALUES (?, ?)',
                 ("/p/o'brien_png/x_cell.png", None))
    conn.commit()
    conn.close()

    assert count_rows(db, "o'brien") == 1


@pytest.mark.parametrize('expression', [
    None, 'cell', '!cell', 'NOT pathogen', 'cell OR nucleus',
    '(cell OR nucleus) AND NOT a_cell', 'CELL', "o'brien", '[cell]',
    'cell%png',
])
def test_measurement_thresholds_preserve_the_browsing_filter(db, monkeypatch, expression):
    import pandas as pd
    from spacr import io
    from spacr.qt import annotate_engine as engine

    with sqlite3.connect(db) as connection:
        connection.execute('INSERT INTO png_list VALUES (?, ?)',
                           ("/p/o'brien_png/e_cell.png", 2))
        connection.execute('INSERT INTO png_list VALUES (?, ?)',
                           ('/p/[cell]/literal.png', 1))
        frame = pd.read_sql_query('SELECT * FROM png_list', connection)
    frame['cell_area'] = [10, 20, 30, 40, 50, 60]
    monkeypatch.setattr(io, '_read_and_join_tables', lambda _db: frame.copy())
    normal = engine.fetch_page(db, 'annotate', 0, 100, expression)
    large = set(frame.loc[frame.cell_area > 15, 'png_path'])
    actual = engine.fetch_filtered_paths(db, 'annotate', ['cell_area'], [15],
                                         ['higher'], expression)
    expected = [row for row in normal if row[0] in large]
    assert [row[0] for row in actual] == [row[0] for row in expected]
    for (_, annotation), (_, original) in zip(actual, expected):
        assert pd.isna(annotation) if original is None else annotation == original


@pytest.mark.parametrize('expression', ['NOT', 'cell AND', '(cell OR nucleus'])
def test_invalid_image_expression_is_refused_before_measurement_join(db, monkeypatch, expression):
    from spacr import io
    from spacr.qt import annotate_engine as engine

    def unexpected(_db):
        pytest.fail('Invalid filters should be refused before expensive measurement reads')

    monkeypatch.setattr(io, '_read_and_join_tables', unexpected)
    with pytest.raises(ValueError):
        engine.fetch_filtered_paths(db, 'annotate', ['cell_area'], [0],
                                    ['higher'], expression)
