"""The join's three routes, and the path parse that happens only once.

``join`` tries the ID columns, then the same columns recovered from the image
path, then the well from that path translated to row and column. Two of the
three read from the parsed path, and the parse is cached between them -- the
comment beside the cache is about correctness as much as cost: the object side
keeps its own columns when it has them, so the two path routes must see the
same parse rather than re-deriving it.

Reaching the third route means the first two failed, which is exactly what an
unjoinable pair of tables produces -- and the error it finally raises names
every route it tried, which is the only thing the user can act on.
"""
from __future__ import annotations

import pandas as pd
import pytest


def test_a_join_on_the_id_columns_takes_the_first_route():
    """The route that costs no parsing at all."""
    from spacr.dependent_join import join

    objects = pd.DataFrame({"plateID": ["plate1", "plate1"],
                            "rowID": ["r1", "r1"],
                            "columnID": ["c1", "c2"],
                            "fieldID": ["1", "1"],
                            "objectID": ["1", "1"],
                            "area": [10.0, 20.0]})
    dependent = objects[["plateID", "rowID", "columnID", "fieldID",
                         "objectID"]].copy()
    dependent["score"] = [0.5, 0.9]

    out, report = join(objects, dependent)

    assert report["route"] == "the ID columns"
    assert out["score"].tolist() == [0.5, 0.9]


def test_a_join_by_path_parses_the_path_and_says_so():
    """The second route, which is where the parse is first done."""
    from spacr.dependent_join import join

    objects = pd.DataFrame({"png_path": ["plate1_A01_1_1.npy",
                                         "plate1_A02_1_1.npy"],
                            "area": [10.0, 20.0]})
    dependent = pd.DataFrame({"png_path": ["plate1_A01_1_1.npy",
                                           "plate1_A02_1_1.npy"],
                              "score": [0.5, 0.9]})

    out, report = join(objects, dependent)

    assert "path" in report["route"]
    assert out["score"].tolist() == [0.5, 0.9]


def test_every_route_is_tried_and_each_failure_is_named():
    """Arc 166 -> 169: the third route reuses the parse the second made.

    Both path routes ask for ``theirs_source['path']``, and only the first of
    them computes it. Reaching the second with the cache already filled is the
    arc, and getting there means routes one and two both failed -- which is
    what two tables with no shared identity produce.

    The error names every route it tried. A bare "could not join" would leave
    the user guessing which of three identity schemes their tables were
    missing.
    """
    from spacr.dependent_join import ROUTES, join

    objects = pd.DataFrame({"png_path": ["plate1_A01_1_1.npy"],
                            "area": [10.0]})
    dependent = pd.DataFrame({"png_path": ["plate9_H12_9_9.npy"],
                              "score": [0.5]})

    with pytest.raises(ValueError) as excinfo:
        join(objects, dependent)

    message = str(excinfo.value)
    assert "could not be joined by any route" in message
    for name, _columns, _source in ROUTES:
        assert name in message, f"route {name!r} was not reported as tried"


def test_an_empty_dependent_table_is_refused_before_any_route():
    """The guard above the loop, so the routes above are reached deliberately."""
    from spacr.dependent_join import join

    with pytest.raises(ValueError, match="nothing to join"):
        join(pd.DataFrame({"area": [1.0]}), pd.DataFrame())
