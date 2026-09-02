"""``apply_import`` turns any corpus layout into a plate spaCR already reads.

The measure that matters: after importing, does spaCR's OWN parser --
``utils._get_regex('cellvoyager')``, the one the core modules use -- read the
result? If it does, every downstream module works unchanged and the import is
genuinely finished rather than finished-shaped.

And the measure that matters second: it must not cost the plate. ``consolidate``
copies every image to rearrange its name, so a 300 GB plate costs 600 GB to
import. Nothing about renaming requires duplicating bytes.
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_corpus import BUILDERS, build_all  # noqa: E402

from spacr.image_import import (apply_import, canonical_name, load_plan,  # noqa: E402
                                plan_import)

ANSWERS = {"per_channel_folder": {0: {"DAPI": 1, "GFP": 2}}}
#: Tiles need an explicit decision -- see the test that pins it.
NEEDS_TILE_DECISION = ("tiled",)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return {t.name: t for t in build_all(tmp_path_factory.mktemp("write"))}


@pytest.fixture(scope="module")
def spacr_regex():
    with contextlib.redirect_stdout(io.StringIO()):     # the helper prints
        from spacr.utils import _get_regex
        return re.compile(_get_regex("cellvoyager", "tif"))


@pytest.mark.parametrize("name", [n for n, _ in BUILDERS])
def test_the_result_is_read_by_spacrs_own_parser(name, corpus, spacr_regex,
                                                 tmp_path):
    """Ten layouts in, one convention out, and the shipped parser reads it."""
    plan = plan_import(corpus[name].root, mapping=ANSWERS.get(name))
    result = apply_import(plan, tmp_path / name,
                          tiles_as_fields=name in NEEDS_TILE_DECISION)
    written = sorted(p.name for p in result.destination.iterdir())
    assert written, f"{name}: nothing was written"
    unreadable = [n for n in written if not spacr_regex.match(n)]
    assert not unreadable, f"{name}: spaCR cannot parse {unreadable[:3]}"
    assert result.written == len(written)


@pytest.mark.parametrize("name", [n for n, _ in BUILDERS])
def test_importing_does_not_duplicate_the_plate(name, corpus, tmp_path):
    """Links, not copies.

    `consolidate` doubles disk use to rearrange names; this must not. The
    saved bytes are reported so a caller can say what was avoided.
    """
    plan = plan_import(corpus[name].root, mapping=ANSWERS.get(name))
    result = apply_import(plan, tmp_path / name,
                          tiles_as_fields=name in NEEDS_TILE_DECISION)
    assert result.linked == result.written, (
        f"{name}: {result.written - result.linked} images were copied")
    assert result.bytes_saved > 0
    for path in result.destination.iterdir():
        assert path.is_symlink(), f"{path.name} is a real file"


def test_a_plan_with_problems_is_refused(corpus, tmp_path):
    """The irreversible half must not run past a stated problem.

    Every problem the plan lists is a way for the result to be quietly wrong.
    An unnamed axis means images that cannot be told apart, and writing them
    anyway produces a plate that looks complete.
    """
    plan = plan_import(corpus["per_channel_folder"].root)   # no dye mapping
    assert plan.problems()
    with pytest.raises(ValueError) as excinfo:
        apply_import(plan, tmp_path / "refused")
    assert "DAPI" in str(excinfo.value)
    assert not (tmp_path / "refused").exists() or \
        not list((tmp_path / "refused").iterdir())


def test_tiles_cannot_be_written_without_being_asked_about(corpus, tmp_path):
    """spaCR's filename has NO TILE SLOT, and that must not be silent.

    A field split into four tiles produces four images with one canonical
    name. Overwriting three of them is exactly the failure this module exists
    to prevent, so by default they are skipped WITH A REASON, and turning
    tiles into fields is something the caller says explicitly -- because it
    discards the fact that they are one field, which anything stitching or
    measuring per field then gets wrong.
    """
    plan = plan_import(corpus["tiled"].root)
    assert plan.counts().get("tile") == 4, "the tile axis was not even found"

    refused = apply_import(plan, tmp_path / "refused")
    assert refused.written == 8
    assert len(refused.skipped) == 24
    assert all("axis is missing" in why for why in refused.skipped.values())

    asked = apply_import(plan, tmp_path / "asked", tiles_as_fields=True)
    assert asked.written == 32, "an image was lost with tiles_as_fields on"
    assert not asked.skipped


def test_a_saved_plan_reproduces_the_import(corpus, tmp_path):
    """The second import of the week is one press, and it is scriptable.

    A lab images the same way every week; re-answering the same questions
    every time is how a tool stops being used. The saved file is the whole
    answer, so a cluster job needs no GUI.
    """
    import json

    from spacr.image_import import _plan_as_json

    tree = corpus["per_channel_folder"]
    plan = plan_import(tree.root, mapping=ANSWERS["per_channel_folder"])
    saved = tmp_path / "plan.json"
    saved.write_text(json.dumps(_plan_as_json(plan)), encoding="utf-8")

    reloaded = load_plan(saved)
    assert not reloaded.problems()
    assert reloaded.counts() == plan.counts()
    assert {k: dict(v) for k, v in reloaded.mapping.items()} == plan.mapping

    first = apply_import(plan, tmp_path / "first")
    again = apply_import(reloaded, tmp_path / "again")
    assert sorted(p.name for p in first.destination.iterdir()) == \
        sorted(p.name for p in again.destination.iterdir())


def test_a_missing_axis_defaults_to_one_not_zero():
    """spaCR's convention is one-based.

    A plate whose only timepoint is ``T0000`` reads as a broken acquisition
    rather than as "there is no time axis here".
    """
    name = canonical_name({"well": "A01", "field": 2, "channel": 1})
    assert "T0001" in name and "Z01" in name, name
    assert "T0000" not in name and "Z00" not in name
