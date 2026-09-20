"""What spaCR can import today, measured against ten real layouts.

Instruction 363 asks for an import module that accepts images "irrespective
of format or naming structure". This file is what turns that into a number.
``tests/import_corpus.py`` builds ten synthetic acquisitions from real
microscope conventions; these tests report, per tree, whether the CURRENT
parser can recover plate, well, field and channel from it.

THE FAILURES ARE THE SPECIFICATION, so they are recorded rather than raised.
A test that failed for each unsupported layout would be red on ten counts
from the day it was written and would say nothing new on any run. Instead the
supported set is PINNED: it may grow, and a shrink is a regression.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_corpus import BUILDERS, build_all  # noqa: E402


#: The layouts the CURRENT parser recovers well/field/channel from. Ten
#: trees, two supported when this was measured on 2026-09-02; FOUR since
#: instruction 441 gave `metadata_type` a record per vendor convention
#: instead of an if/elif chain with two Yokogawas in it.
#:
#: THIS TUPLE IS THE RATCHET. Adding a layout to the importer adds a name
#: here; a name disappearing means an import that used to work stopped, which
#: is the only thing in this file that should ever fail a build.
#:
#: `harmony` and `imagexpress` joined on 2026-09-19 because their trees now
#: carry a `metadata_type` -- `opera_phenix` and `imagexpress` -- and
#: `_parse` below uses it. The remaining six are not regex problems at all:
#: flat_ome, z_stack_in_file and time_in_file hide their axes INSIDE the
#: file, per_well_folder and per_channel_folder hide them in the FOLDER
#: NAME, and tiled needs stitching. None of those can be read by a filename
#: pattern, which is what `metadata_type` is; they are the Import module's.
CURRENTLY_PARSED = ("cellvoyager", "cq1", "harmony", "imagexpress")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return {t.name: t for t in build_all(tmp_path_factory.mktemp("corpus"))}


def _parse(tree):
    """What the current parser recovers from ``tree``, as a dict of sets.

    Uses the real entry points -- ``_get_regex`` and
    ``_extract_filename_metadata`` -- rather than reimplementing them, so this
    measures the shipped behaviour and not a model of it.
    """
    from spacr.utils import _extract_filename_metadata, _get_regex

    names = [p.name for p in sorted(tree.root.rglob("*")) if p.is_file()]
    if not tree.metadata_type:
        # No built-in convention claims this tree. Try spaCR's default, which
        # is what a user who does not change the setting actually gets.
        metadata_type = "cellvoyager"
    else:
        metadata_type = tree.metadata_type
    # THE EXTENSION IS READ OFF THE TREE, not assumed to be "tif". The
    # harmony tree writes .tiff and `preprocess_img_data` derives the format
    # from the folder the same way (io.py's `regex_format`), so hard-coding
    # "tif" here measured something no run does: it reported that the Opera
    # Phenix convention parsed nothing when it parses everything.
    img_format = Counter(
        name.rsplit(".", 1)[-1].lower() for name in names
    ).most_common(1)[0][0]
    regex = re.compile(_get_regex(metadata_type, img_format))
    grouped = _extract_filename_metadata(names, str(tree.root), regex,
                                         metadata_type=metadata_type)
    return grouped


@pytest.mark.parametrize("name", [n for n, _ in BUILDERS])
def test_every_corpus_tree_is_well_formed(name, corpus):
    """The corpus must be a usable specification before it can measure anything.

    Every tree writes real files, every file has a truth record, and the two
    agree -- otherwise a later "spaCR cannot parse this" would be a statement
    about the corpus rather than about spaCR.
    """
    tree = corpus[name]
    on_disk = {str(p.relative_to(tree.root))
               for p in tree.root.rglob("*") if p.is_file()}
    assert on_disk == set(tree.truth), (
        f"{name}: files on disk and truth records disagree")
    assert on_disk, f"{name} wrote no files"
    assert len(tree.wells) == 2, f"{name} must exercise more than one well"
    assert tree.note, f"{name} does not say what makes it hard"


def test_the_two_built_in_conventions_still_parse(corpus):
    """cellvoyager and cq1 are what spaCR claims to handle. They must work.

    This is the half of the file that can fail on an ordinary day, and it is
    the half worth failing: these two are the only import path most users
    have.
    """
    for name in CURRENTLY_PARSED:
        tree = corpus[name]
        grouped = _parse(tree)
        assert grouped, f"{name}: the parser recovered nothing at all"
        wells = {key[1] for key in grouped}
        fields = {key[2] for key in grouped}
        assert len(wells) == 2, (
            f"{name}: expected 2 wells, parser found {sorted(wells)}")
        assert len(fields) == 2, (
            f"{name}: expected 2 fields, parser found {sorted(fields)}")


@pytest.mark.parametrize("name", [n for n, _ in BUILDERS
                                  if n not in CURRENTLY_PARSED])
def test_an_unsupported_layout_fails_visibly_rather_than_silently(name, corpus):
    """An unparseable tree must yield NOTHING, not a partial wrong answer.

    This is the property that matters most before the importer is rewritten.
    A parser that recovers half a plate and proceeds produces a run with
    missing fields and no error -- the same class of defect as the
    ``consolidate`` bugs, where images disappeared instead of failing. A
    parser that recovers nothing at least cannot be mistaken for success.

    Recorded, not enforced as a support claim: these ten names are the
    specification for instruction 363, and the day one of them parses it
    belongs in CURRENTLY_PARSED.
    """
    tree = corpus[name]
    grouped = _parse(tree)
    if not grouped:
        return                      # the honest outcome for an unknown layout
    # It parsed SOMETHING. Then it must not have invented a plate structure:
    # a partial parse that looks plausible is the dangerous case.
    wells = {key[1] for key in grouped}
    assert wells <= tree.wells | {"", None}, (
        f"{name}: the parser invented wells {sorted(wells - tree.wells)} that "
        f"are not in the tree -- a wrong answer is worse than no answer")


def test_the_supported_set_has_not_shrunk(corpus):
    """The ratchet. Growing this is progress; shrinking it is a regression."""
    parsed = []
    for name, _builder in BUILDERS:
        try:
            if _parse(corpus[name]):
                parsed.append(name)
        except Exception:           # an unknown layout may raise; that counts
            pass                    # as not parsed, not as a test error
    assert set(CURRENTLY_PARSED) <= set(parsed), (
        f"a layout that used to import no longer does: "
        f"{sorted(set(CURRENTLY_PARSED) - set(parsed))}")
