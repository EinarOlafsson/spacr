"""Contracts for the canonical spaCR logo and its published copies."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "packaging" / "generate_spacr_logo.py"
README = ROOT / "README.rst"


def _generator():
    spec = importlib.util.spec_from_file_location("spacr_logo_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_logo_is_reproducible_and_lighter_than_its_master():
    generator = _generator()
    source = Image.open(generator.SOURCE).convert("RGBA")
    committed = Image.open(generator.OUTPUTS[0]).convert("RGBA")
    rendered = generator.build_logo()

    assert committed.size == source.size
    assert ImageChops.difference(committed, rendered).getbbox() is None
    assert np.asarray(committed.getchannel("A")).sum() < np.asarray(
        source.getchannel("A")
    ).sum()


def test_all_published_logo_copies_are_identical():
    generator = _generator()
    canonical = generator.OUTPUTS[0].read_bytes()
    assert all(path.read_bytes() == canonical for path in generator.OUTPUTS[1:])


def test_readme_uses_the_logo_from_the_current_branch():
    text = README.read_text(encoding="utf-8")
    assert ".. image:: spacr/resources/icons/logo_spacr.png" in text
    assert "raw.githubusercontent.com/EinarOlafsson/spacr/main" not in text
