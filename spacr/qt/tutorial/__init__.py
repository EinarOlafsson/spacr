"""Scripted narrated tutorials for the spaCR Qt GUI.

Renders a self-contained MP4 (+ SRT sidecar) for each pipeline app.
Uses Piper for narration and QWidget.grab() for frame capture so the
recorder works on any display (real or Xvfb).

Public entry points:
    from spacr.qt.tutorial import render_tutorial
    render_tutorial("mask", out_dir="/tmp/spacr-tutorials")

Or via the CLI:
    spacr-tutorial mask
    spacr-tutorial all
"""
from __future__ import annotations

from .engine import Director, Narrator, Recorder, Step, render_tutorial
from .scripts import AVAILABLE_TUTORIALS

__all__ = [
    "AVAILABLE_TUTORIALS",
    "Director",
    "Narrator",
    "Recorder",
    "Step",
    "render_tutorial",
]
