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
        assert path.is_symlink() or path.stat().st_nlink > 1, f"{path.name} is a real file"


def test_without_symlinks_the_plate_is_hard_linked_not_copied(corpus, tmp_path, monkeypatch):
    """Windows without the symbolic-link privilege (found natively 2026-10-19) copied every image."""
    import os

    def refused(*args, **kwargs):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(os, "symlink", refused)
    name = next(n for n, _ in BUILDERS if n not in NEEDS_TILE_DECISION)
    plan = plan_import(corpus[name].root, mapping=ANSWERS.get(name))
    result = apply_import(plan, tmp_path / name)
    assert result.linked == result.written and result.bytes_saved > 0
    for path in result.destination.iterdir():
        assert not path.is_symlink() and path.stat().st_nlink > 1


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


def test_tiles_are_stitched_back_into_the_field_they_came_from(corpus,
                                                               tmp_path):
    """The maintainer's decision, 2026-09-02: "tiles be stitched at import
    with the option to not stitch but stitch by default."

    It is also what makes spaCR's filename sufficient. The convention has no
    tile slot, so four tiles of one field share one canonical name -- and
    stitching removes the question instead of answering it, because a
    stitched field IS one image with one name.
    """
    plan = plan_import(corpus["tiled"].root)
    assert plan.counts().get("tile") == 4, "the tile axis was not even found"

    stitched = apply_import(plan, tmp_path / "stitched")
    #: 2 wells x 2 fields x 2 channels, each from its four tiles.
    assert stitched.written == 8
    assert stitched.stitched == 8
    assert not stitched.skipped
    written = sorted(p.name for p in stitched.destination.iterdir())
    assert len(written) == 8

    # THE CORPUS TILES ARE BLANK, so there is nothing for the placement to
    # correlate and the mosaic says so rather than claiming a seam it did
    # not measure. That report is the point: a butt-joined field a user was
    # told about is a different thing from one they were not.
    assert len(stitched.unverified) == 8
    assert all("unverified" in why for why in stitched.unverified.values())
    assert "unverified" in stitched.summary()


def test_the_tiles_themselves_can_still_be_had(corpus, tmp_path):
    """The opt-out, and the older answer under it.

    ``tiles_as_fields`` is the more specific request and takes precedence:
    a caller who says each tile is a field has said what they want. With
    both off, tiled images are skipped WITH A REASON -- the only honest
    answer before stitching existed, and still better than writing three of
    four images over each other.
    """
    plan = plan_import(corpus["tiled"].root)

    asked = apply_import(plan, tmp_path / "asked", tiles_as_fields=True)
    assert asked.written == 32, "an image was lost with tiles_as_fields on"
    assert asked.stitched == 0
    assert not asked.skipped

    refused = apply_import(plan, tmp_path / "refused", stitch_tiles=False)
    assert refused.written == 8
    assert len(refused.skipped) == 24
    assert all("axis is missing" in why for why in refused.skipped.values())


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



@pytest.mark.parametrize("names,well,label", [
    (("live_A1_BF_000002.tiff", "live_A1_BF_000003.tiff"), "A01", "BF"),
    (("live_D8_Blue_000001.ome.tiff", "live_D8_Blue_000002.ome.tiff"), "D08", "Blue"),
])
def test_etaluma_live_collision_ordinals_import_as_independent_files(tmp_path, spacr_regex, names, well, label):
    import json
    import numpy as np
    import tifffile

    source = tmp_path / "snapshots"
    source.mkdir()
    for index, name in enumerate(names):
        tifffile.imwrite(source / name, np.full((5, 7), index + 1, dtype=np.uint16))
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    plan = plan_import(source)
    assert plan.problems() == []
    assert set(plan.files) == set(names)
    entries = plan.files
    assert {entry['well'] for entry in entries.values()} == {well}
    assert {entry['channel_name'] for entry in entries.values()} == {label}
    assert {entry['identity'] for entry in entries.values()} == {'independent_file'}
    assert all(not {'field', 't', 'z'} & entry.keys() for entry in entries.values())
    assert len({entry['import_id'] for entry in entries.values()}) == 2
    assert {'import_id', 'identity', 'channel_name'} <= set(plan.columns())
    result = apply_import(plan, tmp_path / "imported", link=False)
    assert result.written == 2 and result.skipped == {} and result.stitched == 0
    for rel, entry in entries.items():
        destination = result.destination / canonical_name(entry)
        assert destination.read_bytes() == before[rel]
        groups = spacr_regex.fullmatch(destination.name).groupdict()
        assert groups['wellID'] == well
        assert int(groups['fieldID']) == entry['import_id']
        assert int(groups['chanID']) == entry['channel']
    from spacr.image_import import save_plan
    saved = save_plan(plan, tmp_path / 'plan.json')
    assert json.loads(saved.read_text(encoding='utf-8'))['files'] == entries
    restored = load_plan(saved)
    assert restored.files == entries
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def test_etaluma_import_identity_never_pairs_channels_by_collision_ordinal(tmp_path):
    import numpy as np
    import tifffile

    source = tmp_path / 'snapshots'
    source.mkdir()
    names = ['live_A1_BF_000001.tiff', 'live_A1_Blue_000001.ome.tiff']
    for name in reversed(names):
        tifffile.imwrite(source / name, np.ones((3, 4), dtype=np.uint16))
    plan = plan_import(source)
    assert not plan.problems()
    before = plan.files
    assert {entry['well'] for entry in before.values()} == {'A01'}
    assert {entry['channel_name'] for entry in before.values()} == {'BF', 'Blue'}
    assert len({entry['import_id'] for entry in before.values()}) == 2
    assert len({canonical_name(entry) for entry in before.values()}) == 2
    tifffile.imwrite(source / 'live_A1_BF_000002.tiff', np.ones((3, 4), dtype=np.uint16))
    after = plan_import(source).files
    assert all(after[name] == before[name] for name in names)
    assert all(not {'field', 't', 'z'} & entry.keys() for entry in after.values())



def test_etaluma_identical_basenames_in_different_folders_remain_independent(tmp_path):
    import numpy as np
    import tifffile

    source = tmp_path / 'snapshots'
    for index, folder in enumerate(('sessionA', 'sessionB')):
        path = source / folder / 'live_A1_BF_000002.tiff'
        path.parent.mkdir(parents=True)
        tifffile.imwrite(path, np.full((3, 4), index + 1, dtype=np.uint16))
    plan = plan_import(source)
    assert not plan.problems()
    assert len(plan.files) == len({entry['import_id'] for entry in plan.files.values()}) == 2
    result = apply_import(plan, tmp_path / 'imported', link=False)
    assert result.written == 2 and not result.skipped
    for relative, entry in plan.files.items():
        assert (result.destination / canonical_name(entry)).read_bytes() == (source / relative).read_bytes()


def test_etaluma_import_identity_collision_is_refused_before_output(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    import numpy as np
    import tifffile

    source = tmp_path / 'snapshots'
    source.mkdir()
    for name in ('live_A1_BF_000002.tiff', 'live_A1_BF_000003.tiff'):
        tifffile.imwrite(source / name, np.ones((3, 4), dtype=np.uint16))
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    with monkeypatch.context() as local:
        local.setattr(hashlib, 'sha256', lambda value: SimpleNamespace(hexdigest=lambda: '0' * 64))
        plan = plan_import(source)
    assert len(plan.problems()) == 1 and 'identities collide' in plan.problems()[0]
    destination = tmp_path / 'refused'
    with pytest.raises(ValueError, match='identities collide'):
        apply_import(plan, destination)
    assert not destination.exists()
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before


def test_import_ids_do_not_override_regular_acquisition_fields():
    assert canonical_name({'well': 'A01', 'field': 7, 'channel': 2, 'import_id': 99}) == (
        'plate1_A01_T0001F007L01A01Z01C02.tif')
