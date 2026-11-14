"""What a dropped folder means to a module.

Instruction 60. ``chaining.resolve_drop`` is ~120 statements and had NO test
at all, which is how ``chaining.py`` came to be the worst genuine non-Qt gap
in the package at 65%.

It is worth more than its line count. It is the function that decides what
happens when a user drags a folder onto a module screen, and its own docstring
states the property that makes it correct:

    "The whole point of the function is that it is *the same* resolution
     auto-chaining performs ... so a drop and an auto-chain fill the field
     with the same string."

The tests below are built on real directory layouts rather than mocks,
because what the function reads IS the directory layout -- a mocked
filesystem would test the mock. The ambiguity paths matter most: this
function is allowed to say "I don't know, you choose", and the failure that
costs a user real time is it guessing instead.
"""
from __future__ import annotations

from spacr import chaining


# --------------------------------------------------------------------------- #
#  The helpers the resolver leans on
# --------------------------------------------------------------------------- #


def test_ports_for_kinds_maps_the_vocabulary(tmp_path):
    found = chaining.ports_for_kinds(("merged-arrays",))
    assert found
    assert any(port.kind == "merged-arrays" for port in found)


def test_layout_directories_are_the_ones_spacr_writes():
    """A guard on the vocabulary itself: these names are what every layout
    walk matches on, so a rename here silently breaks every drop."""
    directories = chaining.layout_directories()
    for expected in ("merged", "measurements", "masks", "settings"):
        assert expected in directories
