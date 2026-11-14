"""How ``spacr.omero`` reads and renders the values an OMERO server hands it.

An OMERO object is a remote proxy: a pixel size may be a unit-carrying
object, a bare float or absent; an annotation value may be a Decimal, a NaN
that claims to be present, or something with no number in it at all.  Each
reader has a documented fallback, and these tables pin the exact result of
every fallback rather than the happy path (covered end to end by
``tests/test_omero.py``).
"""
from __future__ import annotations

from decimal import Decimal

from spacr import omero


class _Quantity:
    """A number-like value that compares by identity and converts to float.

    Unlike ``float('nan')`` it reports itself as PRESENT (``x != x`` is
    False), so only the NaN check after conversion can catch it.
    """

    def __init__(self, number):
        self.number = float(number)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)

    def __float__(self):
        return self.number


class _Opaque:
    """A value with no numeric conversion at all."""

    def __repr__(self):
        return "an opaque proxy"


class _Different(_Opaque):
    """Another opaque value, rendering differently."""

    def __repr__(self):
        return "a different proxy"


def test_pixel_size_from_reads_what_the_server_gave_or_nothing(subtests):
    """Floats are micrometres; booleans and None are no calibration; a length
    object's getters or attributes give value and unit, and a getter that
    raises, an unbound method or an unconvertible value leaves that half
    unknown rather than wrong."""
    class _Length:
        def getValue(self):
            return "0.325"

        def getSymbol(self):
            return "µm"

    class _UnitOnly:
        def getValue(self):
            return 0.5

        def getSymbol(self):
            return None

        def getUnit(self):
            return "MICROMETER"

    class _Broken:
        def getValue(self):
            raise RuntimeError("the server connection dropped")

        def getSymbol(self):
            raise RuntimeError("the server connection dropped")

    class _Unbound:
        def value(self):
            return 0.325

        def unit(self):
            return "MICROMETER"

    class _Wordy:
        value = "about a third of a micron"
        unit = "MICROMETER"

    class _Attributes:
        value = 2
        unit = "nm"

    cases = [
        ("None", None, None, None),
        ("True", True, None, None),
        ("False", False, None, None),
        ("float", 0.325, 0.325, "MICROMETER"),
        ("int", 2, 2.0, "MICROMETER"),
        ("getters", _Length(), 0.325, "µm"),
        ("getUnit after an empty symbol", _UnitOnly(), 0.5, "MICROMETER"),
        ("getters raise", _Broken(), None, None),
        ("unbound methods", _Unbound(), None, None),
        ("value not a number", _Wordy(), None, "MICROMETER"),
        ("plain attributes", _Attributes(), 2.0, "nm"),
    ]
    for case_id, length, value, unit in cases:
        with subtests.test(msg=case_id):
            assert omero.pixel_size_from(length) == omero.PixelSize(value, unit)


def test_format_map_value_renders_every_value_as_annotation_text(subtests):
    """Decimals render like floats (infinities included), a present NaN is
    the missing marker, and a value with no number in it is its collapsed,
    truncated text."""
    class _Verbose:
        def __repr__(self):
            return "x" * 400

    cases = [
        ("decimal", Decimal("1.5"), {}, "1.5"),
        ("decimal infinity", Decimal("Infinity"), {}, "inf"),
        ("decimal -infinity", Decimal("-Infinity"), {}, "-inf"),
        ("float", 3.0, {}, "3"),
        ("present NaN", _Quantity(float("nan")), {}, omero.MISSING_TEXT),
        ("plain NaN", float("nan"), {}, "NaN"),
        ("opaque", _Opaque(), {}, "an opaque proxy"),
        ("long text truncated", _Verbose(), {"max_chars": 10},
         "x" * 9 + "…"),
    ]
    for case_id, value, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert omero.format_map_value(value, **kwargs) == expected


def test_summarise_rows_means_numbers_and_carries_agreed_text(subtests):
    """Each column is the mean of the rows that have a usable number there
    (absent keys and NaNs, present or not, are excluded, never zero-filled);
    text is carried only when every row agrees; a mixed or empty column is
    None; an explicit column list is what gets summarised."""
    opaque = _Opaque()
    cases = [
        ("explicit columns", [{"area": 1.0, "junk": 5.0},
                              {"area": 3.0, "junk": 7.0}], ["area"],
         {"n_objects": 2, "area": 2.0}),
        ("absent key does not dilute", [{"area": 2.0}, {"other": 9.0},
                                        {"area": 4.0}], None,
         {"n_objects": 3, "area": 3.0, "other": 9.0}),
        ("present NaN excluded", [{"ratio": 2.0},
                                  {"ratio": _Quantity(float("nan"))},
                                  {"ratio": 4.0}], None,
         {"n_objects": 3, "ratio": 3.0}),
        ("mixed text and numbers", [{"score": 1.0}, {"score": "n/a"}], None,
         {"n_objects": 2, "score": None}),
        ("agreed text", [{"plateID": "p1"}, {"plateID": "p1"}], None,
         {"n_objects": 2, "plateID": "p1"}),
        ("disagreeing text", [{"plateID": "p1"}, {"plateID": "p2"}], None,
         {"n_objects": 2, "plateID": None}),
        ("nothing usable", [{"area": None}, {"area": float("nan")}], None,
         {"n_objects": 2, "area": None}),
        ("opaque values that agree", [{"tag": opaque}, {"tag": _Opaque()}],
         None, {"n_objects": 2, "tag": opaque}),
        ("opaque values that disagree", [{"tag": _Opaque()},
                                         {"tag": _Different()}], None,
         {"n_objects": 2, "tag": None}),
        ("booleans are text", [{"flag": True}, {"flag": True}], None,
         {"n_objects": 2, "flag": True}),
        ("no rows", [], None, {"n_objects": 0}),
    ]
    for case_id, rows, columns, expected in cases:
        with subtests.test(msg=case_id):
            assert omero.summarise_rows(rows, columns=columns) == expected
