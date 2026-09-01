"""Instruction 327 (5): the orbit fold, ported to the GPU.

The CPU orbit fold is the one renderer the maintainer singled out as
right -- "the orbit fold cpu effect is like a magnigying glass, which
looks cool" -- and the one the pointer complaint excluded. This is that
same map on the GPU.

A SECOND ENTRY, NOT A BACKEND SWITCH, and that is the part worth
testing. The CPU path jitters four samples across four FRAMES, averaging
four different animation times; this takes four samples of ONE instant.
They are not the same picture, so one setting drawing either would mean
the setting no longer says what appears.

The map is transliterated rather than reinvented, so the constants are
checked against the CPU function directly: two renderers that drift
apart are two patterns wearing one name.
"""
from __future__ import annotations

import inspect
import re

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import fractal_travel as FT
from spacr.qt.widgets import fractal_orbit_gpu as GPU


# ---------------------------------------------------------------------------
# It is offered, and it is its own entry
# ---------------------------------------------------------------------------

def test_it_is_a_pattern_the_user_can_choose():
    assert GPU.PATTERN_KEY in FT.PATTERNS
    assert FT.PATTERN_LABELS[GPU.PATTERN_KEY] == GPU.PATTERN_LABEL


def test_the_label_says_what_differs_rather_than_a_version_number():
    """"Orbit fold 2" was the suggestion; something descriptive is
    better, because the difference is the sampling and not an edition."""
    label = FT.PATTERN_LABELS[GPU.PATTERN_KEY]
    assert "2x2" in label or "sharp" in label.lower()
    assert not re.search(r"\b2\b(?!x)", label), (
        f"the label reads as a version number: {label!r}")


def test_the_cpu_entry_is_still_there_and_still_separate():
    """Replacing the CPU one would take away the picture the maintainer
    actually asked to keep."""
    assert "orbit" in FT.PATTERNS
    assert FT.PATTERN_LABELS["orbit"] != FT.PATTERN_LABELS[GPU.PATTERN_KEY]


# ---------------------------------------------------------------------------
# It cannot be drawn without a GPU, and says so by falling back
# ---------------------------------------------------------------------------

def test_it_is_declared_gpu_only():
    assert GPU.PATTERN_KEY in FT.GPU_ONLY_PATTERNS


def test_a_machine_with_no_gpu_gets_the_cpu_orbit_fold(monkeypatch):
    """THE TRAP THIS AVOIDS. The Mandelbrot was once the DEFAULT with no
    CPU renderer, so on a machine without a GPU the stated pattern could
    not be drawn at all."""
    assert FT.pattern_for_this_machine(GPU.PATTERN_KEY, "cpu") == "orbit"

    monkeypatch.setattr(FT, "platform_can_do_opengl", lambda: False)
    assert FT.pattern_for_this_machine(GPU.PATTERN_KEY) == "orbit"


def test_a_machine_with_a_gpu_keeps_it(monkeypatch):
    monkeypatch.setattr(FT, "platform_can_do_opengl", lambda: True)
    monkeypatch.setattr(FT, "gpu_is_available", lambda: True)
    assert FT.pattern_for_this_machine(GPU.PATTERN_KEY) == GPU.PATTERN_KEY


def test_the_patterns_that_do_have_cpu_renderers_are_left_alone():
    """So the fallback is about this pattern, not about every pattern."""
    for pattern in ("orbit", "cascade", "space"):
        assert FT.pattern_for_this_machine(pattern, "cpu") == pattern


# ---------------------------------------------------------------------------
# The map is the SAME map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("constant", [
    "0.45",          # the fold
    "0.055",         # the inversion floor
    "0.42",          # trap A's ring
    "12.0", "9.0", "18.0",          # the three trap widths
    "5.2", "3.7", "2.3",            # the palette weights
    "0.15", "2.25", "4.35",         # the palette phases
])
def test_every_constant_matches_the_cpu_renderer(constant):
    """Transliterated, not reinvented. A constant that drifts makes two
    patterns wearing one name."""
    cpu = inspect.getsource(FT)
    assert constant in GPU.FRAGMENT_SHADER, f"{constant} missing from the shader"
    assert constant in cpu, f"{constant} is no longer in the CPU renderer"


def test_the_three_orbit_traps_are_all_there():
    for trap in ("orbit_a", "orbit_b", "orbit_c"):
        assert trap in GPU.FRAGMENT_SHADER


def test_it_samples_one_instant_four_times():
    """THE DIFFERENCE from the CPU path, and the reason for a second
    entry. Four offsets inside one main(), not a jitter across frames."""
    shader = GPU.FRAGMENT_SHADER
    assert shader.count("orbit_sample(base") == 4
    for offset in ("0.25, 0.25", "0.75, 0.25", "0.25, 0.75", "0.75, 0.75"):
        assert offset in shader, f"missing sample offset {offset}"
    assert "* 0.25" in shader, "the four samples are not averaged"


def test_the_iteration_count_is_substituted_not_left_as_a_name():
    """GLSL has no preprocessor here, so an unsubstituted ITERATIONS is
    a shader that will not compile -- and the failure would be at
    runtime, on the machine that chose the pattern."""
    assert "ITERATIONS" not in GPU.FRAGMENT_SHADER
    assert f"iteration < {GPU.ITERATIONS}" in GPU.FRAGMENT_SHADER


# ---------------------------------------------------------------------------
# It bends the picture the same way everything else does
# ---------------------------------------------------------------------------

def test_it_uses_the_shared_pointer_warp():
    """Instruction 327 (4) asked for the spiral UNDER the pointer, which
    is the same warp every other pattern got."""
    shader = GPU.FRAGMENT_SHADER
    assert "0.55 * u_pull - 0.95 * u_push" in shader
    assert "clamp(strength, -1.4, 0.9)" in shader
    assert "dot(to_pointer, to_pointer) + 0.05" in shader


def test_it_does_not_translate_the_whole_plane():
    assert "uv - target * (u_pull" not in GPU.FRAGMENT_SHADER


def test_the_shader_declares_every_uniform_it_reads():
    """vispy warns once per frame for a value handed to a name it cannot
    find, and once per frame is sixty times a second into the console."""
    shader = GPU.FRAGMENT_SHADER
    declared = set(re.findall(r"uniform\s+\w+\s+(u_\w+)\s*;", shader))
    used = set(re.findall(r"\b(u_\w+)\b", shader))
    assert used - declared == set(), f"undeclared: {sorted(used - declared)}"
