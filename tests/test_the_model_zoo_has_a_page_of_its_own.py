"""The model zoo reaches a reader, table and per-model detail both.

Instruction 370 asks for "a API section for each model in the model zoo,
which should link to Huggingface". Measured 2026-09-03: the generated
`model_zoo_table.rst` was INCLUDED BY NOTHING. It had been produced on every
run, committed, and read by nobody -- so the API had no model zoo page at
all, and neither the table nor any per-model detail reached a reader.

That is the failure this file exists to stop coming back, and it is a quiet
one: a generated file that nothing includes still regenerates, still passes
its own content tests, and still shows up in `git diff` looking healthy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "docs" / "source" / "_generated"
PAGE = ROOT / "docs" / "source" / "model_zoo.rst"


def _catalogue():
    from spacr.model_zoo import BUNDLED_REMOTE_MODELS

    return BUNDLED_REMOTE_MODELS


def test_the_page_exists_and_is_in_the_toctree():
    assert PAGE.is_file(), "there is no model zoo page"
    index = (ROOT / "docs" / "source" / "index.rst").read_text(encoding="utf-8")
    assert "\n   model_zoo\n" in index, (
        "the page is not in the toctree, so Sphinx builds it and nothing "
        "links to it -- which is the state this file was written about")


@pytest.mark.parametrize("generated", ["model_zoo_table.rst",
                                       "model_zoo_sections.rst"])
def test_every_generated_block_is_included_by_the_page(generated):
    assert (GENERATED / generated).is_file(), f"{generated} was not generated"
    page = PAGE.read_text(encoding="utf-8")
    assert f"_generated/{generated}" in page, (
        f"{generated} is generated and included by nothing")


def test_every_published_model_has_a_section():
    text = (GENERATED / "model_zoo_sections.rst").read_text(encoding="utf-8")
    for entry in _catalogue():
        title = str(entry.get("display_name") or entry.get("key"))
        assert title in text, f"{title} has no section of its own"


def test_every_section_links_to_hugging_face():
    """The request is explicit that the API "should link to Huggingface"."""
    text = (GENERATED / "model_zoo_sections.rst").read_text(encoding="utf-8")
    for entry in _catalogue():
        repo = str(entry.get("repo_id") or "")
        if not repo:
            continue
        assert f"https://huggingface.co/{repo}" in text, (
            f"{repo} is not linked, so a reader cannot check the model")


def test_every_section_publishes_the_checksum():
    """An entry with no digest is refused rather than installed, so the
    digest is part of what makes a model usable rather than a footnote."""
    text = (GENERATED / "model_zoo_sections.rst").read_text(encoding="utf-8")
    for entry in _catalogue():
        digest = str(entry.get("sha256") or "")
        if digest:
            assert digest in text, f"{entry.get('key')} publishes no checksum"


def test_the_sections_carry_the_prose_the_table_deliberately_drops():
    """The table is three short columns on purpose (the maintainer called the
    long one "way to much information"). This is where the full account
    goes, which is the only reason the table is allowed to be short."""
    text = (GENERATED / "model_zoo_sections.rst").read_text(encoding="utf-8")
    for entry in _catalogue():
        trained_on = str(entry.get("trained_on") or "").strip()
        if trained_on:
            assert trained_on in text, (
                f"{entry.get('key')}'s full training description is nowhere "
                f"a reader can reach it")
