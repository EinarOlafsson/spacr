"""Every GPU pattern declares the pointer uniforms and uses them.

Reported 2026-08-28: "mouse does nothing in gpu orbital mode or in gpu fold
inversion mode or in gpu space mode". The widget was feeding the values, but
no shader declared the names -- and the uniform update drops names the
loaded shader does not have, so they went nowhere.
"""
from __future__ import annotations

import re

import pytest

from spacr.qt.widgets import fractal_cascade, fractal_space, fractal_travel

SHADERS = {
    "orbit": fractal_travel.FRAGMENT_SHADER,
    "cascade": fractal_cascade.FRAGMENT_SHADER,
    "space": fractal_space.FRAGMENT_SHADER,
}

POINTER_UNIFORMS = ("u_pointer_x", "u_pointer_y", "u_pull", "u_push")


@pytest.mark.parametrize("pattern", sorted(SHADERS))
def test_the_shader_declares_the_pointer(pattern):
    source = SHADERS[pattern]
    declared = set(re.findall(r"uniform\s+\w+\s+(u_\w+)\s*;", source))
    for name in POINTER_UNIFORMS:
        assert name in declared, f"{pattern} does not declare {name}"


@pytest.mark.parametrize("pattern", sorted(SHADERS))
def test_the_shader_actually_uses_it(pattern):
    """Declaring a uniform and ignoring it is the same as not having it --
    and a compiler that optimises it away brings the warning back."""
    source = SHADERS[pattern]
    assert "toward_pointer" in source
    # Called, not merely defined.
    assert source.count("toward_pointer") >= 2, (
        f"{pattern} defines toward_pointer without calling it")


@pytest.mark.parametrize("pattern", sorted(SHADERS))
def test_the_origin_moves_rather_than_a_second_warp(pattern):
    """The pointer is the point the pattern flows to, so it shifts the
    coordinate ORIGIN; a second warp term would read as a smear."""
    source = SHADERS[pattern]
    body = source[source.index("vec2 toward_pointer"):]
    body = body[:body.index("}")]
    assert "u_pull" in body and "u_push" in body
    assert "uv -" in body or "uv +" in body


def test_every_pointer_uniform_survives_the_declared_filter():
    """The update sets only what the shader declares; that filter is what
    silenced GPU space's u_dream flood, and it must not silence these."""
    for pattern, source in SHADERS.items():
        declared = frozenset(
            re.findall(r"uniform\s+\w+\s+(u_\w+)\s*;", source))
        for name in POINTER_UNIFORMS:
            assert name in declared, f"{pattern} would drop {name}"


def test_space_still_has_no_dream_term():
    """The filter exists because the patterns do not share a uniform list."""
    declared = set(re.findall(r"uniform\s+\w+\s+(u_\w+)\s*;",
                              fractal_space.FRAGMENT_SHADER))
    assert "u_dream" not in declared, (
        "space gained a dream term; a star field has nothing to warp")
    assert "u_dream" in set(re.findall(
        r"uniform\s+\w+\s+(u_\w+)\s*;", fractal_travel.FRAGMENT_SHADER))
