"""The OpenGL 3D backend is offered only when PyOpenGL imports too.

VisPy alone cannot make 3D textures: every frame failed with "PyOpenGL is
required for 3D texture support" and the view fell back to the CPU, though
the Render backend list still offered OpenGL.
"""
from __future__ import annotations

import builtins
import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import make_masks

ROOT = Path(__file__).resolve().parents[2]


def _block_pyopengl(monkeypatch):
    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "OpenGL" or name.startswith("OpenGL."):
            raise ImportError("No module named 'OpenGL'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    monkeypatch.setattr(make_masks, "find_spec",
                        lambda name, *a: None if name == "OpenGL" else object())


def test_opengl_is_not_offered_without_pyopengl(monkeypatch):
    monkeypatch.setattr(make_masks, "volume_gpu_refusal", lambda: "")
    monkeypatch.setattr("spacr.layers.wgpu_adapters", lambda *a, **k: [])
    _block_pyopengl(monkeypatch)
    assert "opengl" not in make_masks.volume_render_backends()
    assert make_masks.volume_backend_chain("auto") == ()


def test_the_opengl_context_refuses_cleanly_without_pyopengl(monkeypatch):
    _block_pyopengl(monkeypatch)
    with pytest.raises(RuntimeError, match="PyOpenGL"):
        make_masks._GLVolumeContext(allow_software=True)


def test_pyopengl_is_a_core_requirement_and_in_every_dependency_list():
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert re.search(r"'PyOpenGL>=[0-9.]+,<4'", setup)
    recipe = (ROOT / "conda-forge" / "recipe" / "recipe.yaml").read_text(
        encoding="utf-8")
    assert "pyopengl" in recipe
    floors = (ROOT / ".github" / "constraints" / "minimum-py39.txt").read_text(
        encoding="utf-8")
    assert "PyOpenGL==" in floors
    spec = (ROOT / "packaging" / "spacr.spec").read_text(encoding="utf-8")
    assert '"OpenGL.GL"' in spec
