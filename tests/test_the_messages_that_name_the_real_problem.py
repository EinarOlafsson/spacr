"""Three refusals whose value is entirely in what the message says.

Each of these could have been a bare exception and each was made specific
because the bare one sent someone looking in the wrong place. The tildeone is
GitHub issue #108: a macOS user whose settings carried ``~`` was told
``FileNotFoundError: ~/x/measurements.db``, which reads as "your database is
missing" when the database was fine and the path was never resolved.

Testing a message feels like testing prose. It is not: the message IS the
feature here, and a refactor that keeps the raise and loses the sentence would
put the user back where issue #108 found them.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# database_schema.migrate_database — the tilde nobody expanded
# ---------------------------------------------------------------------------

def test_an_unexpanded_tilde_says_the_path_was_never_resolved(tmp_path,
                                                              monkeypatch):
    """Issue #108's message.

    The contract stays strict -- this function does not expand ``~``, and the
    docstring says expansion belongs in ensure_database_schema -- but it names
    the real problem instead of blaming the database.
    """
    from spacr.database_schema import migrate_database

    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        migrate_database("~/screens/plate1/measurements.db")

    message = str(excinfo.value)
    assert "was never expanded" in message
    assert "The database itself may be fine" in message


def test_an_ordinary_missing_database_gets_the_plain_message(tmp_path):
    """The other side: no tilde, so no lecture about tildes.

    A path that really is missing must not be explained away as a resolution
    problem -- that would send the user looking for a file that is not there.
    """
    from spacr.database_schema import migrate_database

    missing = tmp_path / "no_such" / "measurements.db"

    with pytest.raises(FileNotFoundError) as excinfo:
        migrate_database(str(missing))

    assert "was never expanded" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# mask_io.save_mask — an object id that will not survive the write
# ---------------------------------------------------------------------------

def test_a_mask_whose_ids_exceed_uint16_is_refused_before_writing(tmp_path):
    """The refusal, and why saving anyway would be worse than failing.

    Masks are stored as uint16. An id above 65535 wraps on write, so object
    65536 silently becomes object 0 -- background -- and every measurement for
    it disappears into the field. The message names the offending id and tells
    the user to relabel, which is the actual fix.
    """
    from spacr.mask_io import save_mask

    mask = np.zeros((8, 8), dtype=np.int32)
    mask[1, 1] = 70000

    with pytest.raises(ValueError) as excinfo:
        save_mask(tmp_path / "mask.tif", mask)

    message = str(excinfo.value)
    assert "70000" in message
    assert "relabel" in message
    assert not (tmp_path / "mask.tif").exists(), "refused before writing"


def test_a_mask_at_the_uint16_ceiling_is_accepted(tmp_path):
    """The boundary: 65535 fits, so it must not be refused.

    An off-by-one here would reject a legitimately full field, which a dense
    plate really can produce.
    """
    from spacr.mask_io import save_mask

    mask = np.zeros((8, 8), dtype=np.int32)
    mask[1, 1] = np.iinfo(np.uint16).max

    written = save_mask(tmp_path / "mask.tif", mask)

    assert os.path.isfile(written)


def test_an_empty_mask_is_written_without_inspecting_a_maximum(tmp_path):
    """``np.size(mask)`` guarding the max: an empty array has none.

    A field where segmentation found nothing is common at a plate edge, and
    ``np.max`` of an empty array raises.
    """
    from spacr.mask_io import save_mask

    written = save_mask(tmp_path / "empty.tif",
                        np.zeros((0, 0), dtype=np.uint16))

    assert os.path.isfile(written)


# ---------------------------------------------------------------------------
# hits.guide_of — a missing feature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("feature", [None, float("nan"), np.nan])
def test_a_missing_feature_names_no_guide(feature):
    """The NaN guard, which pandas makes necessary.

    A coefficient table read from CSV carries NaN for a blank feature, and
    ``str(nan)`` is ``'nan'`` -- which would go on to be parsed as a term and
    could match a shape rule. None is the only honest answer.
    """
    from spacr.hits import guide_of

    assert guide_of(feature) is None


def test_a_gene_term_names_no_guide():
    """The contrast: a real term that is not a guide."""
    from spacr.hits import guide_of

    assert guide_of("gene_fraction:gene[TGGT1_231640]") is None


def test_a_guide_term_names_its_guide():
    """And one that is, so the Nones above are visibly decisions."""
    from spacr.hits import guide_of

    assert guide_of("grna[TGGT1_231640_3]") == "TGGT1_231640_3"
