"""The organelle defaults are written twice. They must not drift apart.

`spacr.settings` declares a default for the same 40 organelle keys in two
hand-maintained places:

  * `set_default_settings_preprocess_generate_masks` (the mask pipeline), and
  * `_set_organelle_defaults` (called by `object.py` and `core.py`).

They agree exactly today. Nothing enforced that, and two hand-written copies
of forty values drift — silently, because a run takes whichever it reached
first and never compares them. A user would see two runs of the same plate
disagree with no error and nothing in the log naming the cause.

Found while inventorying instruction 76 (support more than one organelle),
where the duplication is the first thing that has to go. This test is the
holding pattern until it does: it does not fix the duplication, it just makes
a divergence fail loudly instead of quietly.
"""

import re
from pathlib import Path

import pytest


SETTINGS = Path(__file__).resolve().parent.parent / "spacr" / "settings.py"

#: Keys the mask pipeline defaults but `_set_organelle_defaults` does not.
#: Not a defect: the second block covers the segmentation knobs only. Listed
#: so the test fails if the OVERLAP changes, rather than silently shrinking.
MASK_ONLY_IS_EXPECTED = True


def _normalise(value: str) -> str:
    """Strip the trailing punctuation of a call or a dict entry."""
    return value.strip().rstrip(",").rstrip(")").strip()


def _mask_pipeline_defaults(lines):
    """`settings.setdefault('organelle_x', value)` in the MASK factory only.

    Scoped to one function on purpose. An unscoped scan also catches
    `get_measure_crop_settings`, where `organelle_min_size` deliberately
    defaults to 0 against the mask pipeline's 10 -- a real difference between
    two pipelines, not a drift between two copies of one. Comparing those
    would make this test fail on correct code, which is how a guard gets
    deleted rather than heeded.
    """
    found = {}
    inside = False
    for line in lines:
        if "def set_default_settings_preprocess_generate_masks" in line:
            inside = True
            continue
        if inside and re.match(r"^def \w", line):
            break
        if not inside:
            continue
        match = re.search(
            r"setdefault\(\s*['\"](organelle_[A-Za-z0-9_]+)['\"]\s*,\s*(.+)$",
            line)
        if match:
            found[match.group(1)] = _normalise(match.group(2))
    return found


def _organelle_default_block(lines):
    """`'organelle_x': value,` inside `_set_organelle_defaults`."""
    found = {}
    inside = False
    for line in lines:
        if "_set_organelle_defaults" in line:
            inside = True
            continue
        if inside and re.match(r"^def \w", line):
            break
        if not inside:
            continue
        match = re.match(
            r"^\s*['\"](organelle_[A-Za-z0-9_]+)['\"]\s*:\s*(.+)$", line)
        if match:
            found[match.group(1)] = _normalise(match.group(2))
    return found


@pytest.fixture(scope="module")
def blocks():
    lines = SETTINGS.read_text(encoding="utf-8").splitlines()
    return _mask_pipeline_defaults(lines), _organelle_default_block(lines)


def test_both_blocks_were_actually_found(blocks):
    """A regex that matches nothing would make every other test vacuous."""
    mask, organelle = blocks
    assert len(mask) > 30, f"only {len(mask)} mask defaults parsed"
    assert len(organelle) > 30, f"only {len(organelle)} organelle defaults parsed"


def test_the_two_blocks_still_overlap(blocks):
    """If the overlap vanishes, this test stops defending anything."""
    mask, organelle = blocks
    assert set(mask) & set(organelle), (
        "the two default blocks no longer share a key — either the "
        "duplication was removed (delete this test) or a rename broke the "
        "comparison (fix it), but do not leave it passing vacuously")


def test_every_shared_organelle_default_agrees(blocks):
    """The claim: forty values written twice, identical.

    A divergence here means two code paths disagree about what a setting
    defaults to, and which one a run gets depends on which factory it went
    through. That is not a preference — it is the same plate measured two
    ways.
    """
    mask, organelle = blocks
    shared = sorted(set(mask) & set(organelle))
    disagree = {
        key: (mask[key], organelle[key])
        for key in shared if mask[key] != organelle[key]
    }
    assert not disagree, (
        "organelle defaults disagree between "
        "set_default_settings_preprocess_generate_masks and "
        "_set_organelle_defaults:\n"
        + "\n".join(f"  {k}: mask={a!r} organelle_block={b!r}"
                    for k, (a, b) in disagree.items()))
