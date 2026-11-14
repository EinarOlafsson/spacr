"""Which crops the annotator opens: thresholds, gates and the image filter.

``fetch_filtered_paths`` and ``gate_paths`` join ``png_list`` to the
measurement tables on ``prcfo``, and that join does not always bring the
crop path with it.  When it does not, the path is merged back on one to one
(a repeated key would show one cell under several crops); when it cannot
be, the answer is no rows rather than a grid of objects nobody can open.
An annotation column already on the join is returned, never blanked.

``parse_image_type`` turns the image-filter box into bound SQL; an empty box
is no filter at all, and every other readable text yields a clause.

The two ``spacr.io`` readers are stubbed: the shape of the joined frame --
``prcfo`` as index or column, with or without ``png_path`` or the annotation
column -- is the subject, and no on-disk schema can express all of them.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt import annotate_engine as AE  # noqa: E402

pytestmark = pytest.mark.qt

_CROPS = ["/crops/o1.png", "/crops/o2.png", "/crops/o3.png"]


def _indexed_measurements():
    """A joined frame indexed by prcfo, with no crop path on it."""
    return pd.DataFrame({"prcfo": ["o1", "o2", "o3"],
                         "cell_area": [50.0, 500.0, 900.0]}).set_index("prcfo")


def _png_list(index=True, paths=None):
    frame = pd.DataFrame({"prcfo": ["o1", "o2", "o3"],
                          "png_path": list(paths or _CROPS)})
    return frame.set_index("prcfo") if index else frame


def _stub(monkeypatch, joined, png_list=None):
    import spacr.io as io

    monkeypatch.setattr(io, "_read_and_join_tables",
                        lambda *_a, **_k: joined.copy())
    monkeypatch.setattr(
        io, "_read_db",
        lambda *_a, **_k: [(png_list if png_list is not None
                            else pd.DataFrame()).copy()])


def _database(tmp_path, name, png_paths=None):
    """An existing database file, with a ``png_list`` table when given paths."""
    path = tmp_path / name
    path.write_bytes(b"")
    if png_paths is not None:
        with sqlite3.connect(path) as connection:
            pd.DataFrame({"png_path": png_paths}).to_sql(
                "png_list", connection, index=False)
    return str(path)


def test_fetch_filtered_paths_reaches_the_crops_whatever_shape_the_join_has(
        tmp_path, monkeypatch, subtests):
    """Every join shape gives the exact (path, annotation) rows past the cuts."""
    unjoinable = pd.DataFrame({"png_path": ["/crops/o1.png"]})
    with_annotations = pd.DataFrame({"png_path": ["/c/o1.png", "/c/o2.png"],
                                     "cell_area": [500.0, 900.0],
                                     "annotate": [1, 2]})
    chained = pd.DataFrame({
        "png_path": ["/c/o1.png", "/c/o2.png", None, "/c/keep_o4.png"],
        "cell_area": [50.0, 500.0, 900.0, 900.0]})
    two_measures = pd.DataFrame({"png_path": _CROPS,
                                 "cell_area": [50.0, 500.0, 900.0],
                                 "nucleus_area": [5.0, 80.0, 10.0]})
    # case, joined, png_list, db png_list rows, args, kwargs, expected rows
    cases = [
        ("prcfo index: the crop path is merged back on",
         _indexed_measurements(), _png_list(), None,
         (["cell_area"], [100.0], ["higher"]), {},
         [("/crops/o2.png", None), ("/crops/o3.png", None)]),
        ("prcfo column on the join is not reset",
         _indexed_measurements().reset_index(), _png_list(), None,
         (["cell_area"], [100.0], ["higher"]), {},
         [("/crops/o2.png", None), ("/crops/o3.png", None)]),
        ("prcfo column on png_list is merged as is",
         _indexed_measurements(), _png_list(index=False), None,
         (["cell_area"], [100.0], ["lower"]), {},
         [("/crops/o1.png", None)]),
        ("a png_list with no key cannot be joined",
         _indexed_measurements(), unjoinable, None,
         (["cell_area"], [10.0], ["higher"]), {}, []),
        ("an annotation already on the join is returned",
         with_annotations, None, None,
         (["cell_area"], [10.0], ["higher"]), {},
         [("/c/o1.png", 1), ("/c/o2.png", 2)]),
        ("a join without the annotation column gets it, empty",
         with_annotations.drop(columns=["annotate"]), None, None,
         (["cell_area"], [10.0], ["higher"]), {},
         [("/c/o1.png", None), ("/c/o2.png", None)]),
        ("an image type narrows the crops after the merge",
         _indexed_measurements(), _png_list(), _CROPS,
         (["cell_area"], [10.0], ["higher"]), {"image_type": "o3"},
         [("/crops/o3.png", None)]),
        ("threshold, missing path and image type all apply in turn",
         chained, None, ["/c/o1.png", "/c/o2.png", None, "/c/keep_o4.png"],
         (["cell_area"], [100.0], ["higher"]), {"image_type": "keep"},
         [("/c/keep_o4.png", None)]),
        ("a threshold past every object empties the list",
         chained, None, None, (["cell_area"], [10000.0], ["higher"]), {}, []),
        ("one threshold and one direction apply to every measurement",
         two_measures, None, None,
         (["cell_area", "nucleus_area"], [20.0], "higher"), {},
         [("/crops/o2.png", None)]),
        ("an unknown direction and an unknown column cut nothing",
         two_measures, None, None,
         (["cell_area", "no_such_column"], [100.0, 1.0],
          ["sideways", "higher"]), {},
         [(path, None) for path in _CROPS]),
    ]
    for index, (case, joined, png_list, db_rows, args, kwargs,
                expected) in enumerate(cases):
        with subtests.test(msg=case):
            db = _database(tmp_path, f"m{index}.db", db_rows)
            _stub(monkeypatch, joined, png_list)
            rows = AE.fetch_filtered_paths(db, "annotate", *args, **kwargs)
            assert [tuple(row) for row in rows] == expected

    with subtests.test(msg="a repeated crop key is refused, not duplicated"):
        db = _database(tmp_path, "dup.db")
        doubled = pd.concat([_png_list(index=False),
                             _png_list(index=False).iloc[:1]])
        _stub(monkeypatch, _indexed_measurements(), doubled)
        with pytest.raises(pd.errors.MergeError):
            AE.fetch_filtered_paths(db, "annotate", ["cell_area"], [1.0],
                                    ["higher"])

    with subtests.test(msg="mismatched lengths name every count"):
        db = _database(tmp_path, "lengths.db")
        _stub(monkeypatch, two_measures)
        with pytest.raises(ValueError) as raised:
            AE.fetch_filtered_paths(db, "annotate",
                                    ["cell_area", "nucleus_area"],
                                    [1.0, 2.0, 3.0], ["higher"])
        assert str(raised.value).startswith(
            "2 measurement column(s) but 3 threshold(s) and 2 direction(s)")

    def _never(*_a, **_k):
        raise AssertionError("the tables were read for a request that needs none")

    import spacr.io as io
    monkeypatch.setattr(io, "_read_and_join_tables", _never)
    for case, db, args in [
            ("a database that is not there", str(tmp_path / "absent.db"),
             (["cell_area"], [1.0], ["higher"])),
            ("a directory is not a database", str(tmp_path), ([], [], [])),
            ("no measurements", _database(tmp_path, "none.db"),
             ([], [1.0], ["higher"])),
            ("no thresholds", _database(tmp_path, "nothr.db"),
             (["cell_area"], [], ["higher"]))]:
        with subtests.test(msg=case):
            assert AE.fetch_filtered_paths(db, "annotate", *args) == []


def _gate(low):
    from spacr.qt.widgets.gate_spec import ThresholdGate

    return ThresholdGate(name="big cells", column="cell_area", low=low)


def test_gate_paths_reaches_a_gated_population_through_the_same_merge(
        tmp_path, monkeypatch, subtests):
    """A gate chain selects the exact crop paths, or none when unreachable."""
    with_paths = pd.DataFrame({"png_path": ["/c/a.png", None, "/c/c.png"],
                               "cell_area": [500.0, 900.0, 5.0]})
    # case, joined, png_list, gates, expected paths
    cases = [
        ("prcfo index: merged back on", _indexed_measurements(), _png_list(),
         [_gate(100.0)], ["/crops/o2.png", "/crops/o3.png"]),
        ("prcfo columns on both sides",
         _indexed_measurements().reset_index(), _png_list(index=False),
         [_gate(600.0)], ["/crops/o3.png"]),
        ("a png_list with no key reaches no crops", _indexed_measurements(),
         pd.DataFrame({"png_path": ["/crops/o1.png"]}), [_gate(10.0)], []),
        ("a join that carries the path keeps it, missing paths dropped",
         with_paths, None, [_gate(100.0)], ["/c/a.png"]),
        ("a chain of gates narrows in turn", _indexed_measurements(),
         _png_list(), [_gate(100.0), _gate(600.0)], ["/crops/o3.png"]),
    ]
    db = _database(tmp_path, "measurements.db")
    for case, joined, png_list, gates, expected in cases:
        with subtests.test(msg=case):
            _stub(monkeypatch, joined, png_list)
            assert AE.gate_paths(db, gates) == expected

    with subtests.test(msg="no gates select nothing, without reading"):
        import spacr.io as io

        monkeypatch.setattr(io, "_read_and_join_tables",
                            lambda *_a, **_k: pytest.fail("tables read"))
        assert AE.gate_paths(db, []) == []


def test_parse_image_type_reads_every_filter_box(subtests):
    """Each expression gives its exact SQL and bound parameters, or a named refusal."""
    like = "png_path LIKE ?"
    cases = [
        ("empty", "", ("", [])),
        ("only whitespace", "   ", ("", [])),
        ("None", None, ("", [])),
        ("a term", "nucleus", (like, ["%nucleus%"])),
        ("punctuation is a literal", "   ,,,   ", (like, ["%,,,%"])),
        ("bang negation", "!a", (f"(NOT {like})", ["%a%"])),
        ("a lone bang", "! a", (f"(NOT {like})", ["%a%"])),
        ("NOT spelled out, any case", "not a", (f"(NOT {like})", ["%a%"])),
        ("AND binds tighter than OR", "a OR b AND c",
         (f"({like} OR ({like} AND {like}))", ["%a%", "%b%", "%c%"])),
        ("AND with NOT", "cell AND NOT pathogen",
         (f"({like} AND (NOT {like}))", ["%cell%", "%pathogen%"])),
        ("parentheses group", "(a or b) and c",
         (f"((({like} OR {like})) AND {like})", ["%a%", "%b%", "%c%"])),
        ("a quote is bound, not interpolated", "it's",
         (like, ["%it's%"])),
    ]
    for case, expression, expected in cases:
        with subtests.test(msg=case):
            assert AE.parse_image_type(expression) == expected

    refusals = [
        ("ends after an operator", "a AND", "ends after an operator"),
        ("a bare NOT", "NOT", "ends after an operator"),
        ("unclosed parenthesis", "(a", "unclosed '('"),
        ("an operator where a term belongs", "OR a",
         "has 'OR' where a path fragment was expected"),
        ("a closing parenthesis first", ")",
         "has ')' where a path fragment was expected"),
        ("two terms without an operator", "a b c d",
         "could not read the image filter after 'b c d'"),
    ]
    for case, expression, message in refusals:
        with subtests.test(msg=f"refused: {case}"):
            with pytest.raises(ValueError) as raised:
                AE.parse_image_type(expression)
            assert message in str(raised.value)

    # The tokeniser turns every non-blank text into at least one token, which
    # is why parse_image_type needs no empty-token guard.
    tokens = [
        ("a", ["a"]), ("!", ["NOT"]), ("!a", ["NOT", "a"]), ("(", ["("]),
        ("()", ["(", ")"]), ("(a)", ["(", "a", ")"]),
        ("a AND NOT b", ["a", "AND", "NOT", "b"]), ("((", ["(", "("]),
        ("-", ["-"]), ("\U0001f600", ["\U0001f600"]),
        ("x\ty", ["x", "y"]),
    ]
    for text, expected in tokens:
        with subtests.test(msg=f"tokens of {text!r}"):
            assert AE._tokenise_image_type(text) == expected
