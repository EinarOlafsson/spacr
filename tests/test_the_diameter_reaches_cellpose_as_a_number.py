"""A diameter typed into the GUI must not fail the run inside Cellpose.

Mask generation died after 154 seconds with::

    TypeError: '>' not supported between instances of 'str' and 'int'

Cellpose's eval tests ``diameter > 0``. Every route into that setting which is
not a Python literal hands it over as a STRING -- a number typed into the GUI,
and any settings CSV -- and the failure lands inside the segmentation call,
after the plates have been loaded and normalised.

THE BLANK CASE IS THE ONE TO BE CAREFUL WITH. ``_get_object_settings`` fills
``object_settings['diameter']`` with a magnification-derived default, so the
obvious fix -- read the already-coerced value -- would turn "blank means native
scale" into "rescale by 30/diameter" for every run that left the field empty.
That is a different segmentation, silently, and it is why the coercion below
preserves ``None``.
"""
from __future__ import annotations

import pytest

from spacr.object import _eval_diameter


@pytest.mark.parametrize("raw, expected", [
    ("30", 30.0),
    ("30.0", 30.0),
    ("  30.5  ", 30.5),        # a CSV field with whitespace
    (30, 30.0),
    (17.5, 17.5),
])
def test_a_written_number_arrives_as_a_float(raw, expected):
    result = _eval_diameter(raw, "cell")
    assert result == expected
    assert isinstance(result, float)
    assert result > 0, "the comparison Cellpose makes must not raise"


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_blank_stays_blank_rather_than_becoming_a_default(raw):
    """None means native scale. Substituting a default would rescale every
    image in every run that left the field empty."""
    assert _eval_diameter(raw, "cell") is None


def test_an_unparseable_value_is_reported_and_treated_as_blank(capsys):
    """Raising would fail the run at the same point it fails today, and
    `_get_object_settings` has always warned-and-continued for this field."""
    assert _eval_diameter("thirty", "nucleus") is None
    assert "nucleus_diameter" in capsys.readouterr().out


#: Every function in spacr/object.py that reads the user's diameter setting
#: for a Cellpose model, and how many times. The 2-D and z-stack eval calls
#: in the Cellpose-SAM generator, the Cellpose-DINO route (item 525) and the
#: Cellpose 3 route (item 503), whose eval keywords are built in
#: ``_cellpose3_eval_settings``. A new route fails here until it is listed,
#: and a listed one fails if any of its reads skips the coercion.
_DIAMETER_ROUTES = {
    "generate_cellpose_masks_sam": 2,
    "_cellpose_dino_masks": 1,
    "_cellpose3_eval_settings": 1,
}


def _diameter_reads(function):
    """``(coerced, total)`` reads of ``settings.get(f'{object_type}_diameter')``."""
    import ast

    parents = {child: parent for parent in ast.walk(function)
               for child in ast.iter_child_nodes(parent)}
    coerced = total = 0
    for node in ast.walk(function):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "settings"
                and node.args
                and ast.unparse(node.args[0]) == "f'{object_type}_diameter'"):
            continue
        total += 1
        parent = parents.get(node)
        if (isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Name)
                and parent.func.id == "_eval_diameter"
                and parent.args and parent.args[0] is node):
            coerced += 1
    return coerced, total


def test_every_segmentation_route_coerces():
    """A source check: the eval paths must not diverge, which is exactly how
    one of them came to be fixed and the other not -- and there are four
    of them now, not the two this test was written for."""
    import ast
    from pathlib import Path

    import spacr.object as object_module

    source = Path(object_module.__file__).read_text(encoding="utf-8")
    assert "diameter=settings.get(f'{object_type}_diameter')" not in source
    reads = {node.name: _diameter_reads(node)
             for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef)}
    readers = {name for name, (_coerced, total) in reads.items() if total}
    assert readers == set(_DIAMETER_ROUTES), (
        "the functions reading the diameter setting are "
        f"{sorted(readers)}, not {sorted(_DIAMETER_ROUTES)}")
    for name, expected in _DIAMETER_ROUTES.items():
        assert reads[name] == (expected, expected), (
            f"{name}: {reads[name][0]} of {reads[name][1]} reads coerced, "
            f"expected {expected} of {expected}")


def test_the_v2_route_coerces_too():
    """core.py casts every other numeric on that call and left this one raw."""
    from pathlib import Path

    import spacr.core as core_module

    source = Path(core_module.__file__).read_text(encoding="utf-8")
    assert "diameter=settings.get('cell_diameter')" not in source
    assert "_eval_diameter(" in source
