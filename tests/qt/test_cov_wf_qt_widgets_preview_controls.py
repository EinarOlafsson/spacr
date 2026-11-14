"""The preview selectors on the days nothing lines up.

:mod:`spacr.qt.widgets.preview_controls` builds the two dropdowns and the cap
box that every live preview (Mask, Measure, Timelapse, Motility) wears, and
the enumeration behind them. The happy paths -- a Yokogawa plate, a numbered
channel entry, a field that is in the sample -- are covered elsewhere. What is
asserted here is the set of *mismatches* a real session produces, each of which
must degrade to something the user can still work with rather than to an empty
panel or a wrong pixel:

* a channel entry that carries a stain name instead of ``Ch 3``;
* a field of view that is no longer in the sample the dropdown is showing;
* a naming dialect ``spacr.utils._get_regex`` does not know;
* a ``spacr/utils.py`` that no longer defines ``_get_regex`` at all, which is
  what the previews' import-free "lift the function out of the source" trick
  would hit the day somebody renames it;
* a panel that ships the sets dropdown without the cap box beside it -- Mask
  puts its cap next to the ``Choose image…`` button, not next to a combo.

Nothing here opens an image: the enumeration reads file *names*, so the plates
below are empty files with real acquisition names.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QComboBox

from spacr.qt.widgets import preview_controls as pc

pytestmark = pytest.mark.qt

#: Wells and fields of the synthetic plate. Two channels per field, so a
#: grouped enumeration must report half as many entries as there are files.
WELLS = ("A01", "A02", "B03")
FIELDS = (1, 2)
CHANNELS = (1, 2)


@pytest.fixture(autouse=True)
def _qapp(qapp):
    """Qt widgets abort the process when no QApplication exists."""
    return qapp


@pytest.fixture(autouse=True)
def _fresh_regex_caches():
    """Both regex helpers are ``lru_cache``d process-wide.

    A test that makes ``_get_regex`` unreachable would otherwise leave a
    cached ``None`` behind and every later test in the session would enumerate
    an ungrouped plate for no visible reason.
    """
    pc._get_regex_callable.cache_clear()
    pc._acquisition_regex.cache_clear()
    yield
    pc._get_regex_callable.cache_clear()
    pc._acquisition_regex.cache_clear()


def _plate(root: Path) -> list:
    """Write an empty file per (well, field, channel) with Yokogawa names."""
    root.mkdir(parents=True, exist_ok=True)
    names = []
    for well in WELLS:
        for field in FIELDS:
            for chan in CHANNELS:
                name = (f"plate1_{well}_T0001F{field:03d}"
                        f"L01A01Z01C{chan:02d}.tif")
                (root / name).write_bytes(b"")
                names.append(name)
    return sorted(names)


def _entries(combo: QComboBox) -> list:
    return [combo.itemText(i) for i in range(combo.count())]


# ---------------------------------------------------------------------------
# A sets dropdown with no cap box beside it
# ---------------------------------------------------------------------------


def test_a_loaded_field_the_draw_missed_is_kept_and_called_out(tmp_path):
    """The file the user opened must stay in the panel's own dropdown.

    A user who drops one specific field on a preview and then finds it absent
    from the list has lost the thing they came to look at. The sampler pins it,
    which makes the list one longer than the cap, and
    :meth:`~spacr.qt.widgets.preview_controls.ImageSetSampler.describe` names
    the extra entry rather than quietly reporting a larger sample than was
    drawn.
    """
    root = tmp_path / "plate"
    _plate(root)
    sampler = pc.ImageSetSampler(max_sets=1)
    sampler.enumerate(root, [".tif"])
    drawn = pc.sample_image_sets(sampler.sets, 1, sampler.seed)
    missed = next(s for s in sampler.sets if s not in drawn)
    combo = QComboBox()

    note = pc.apply_sample_to_combo(combo, None, sampler,
                                    missed.path())

    assert _entries(combo) == sorted(
        [drawn[0].label, missed.label],
        key=[s.label for s in sampler.sets].index)
    assert note.startswith("showing a random sample of 1 of 6 image sets, "
                           "plus the field you loaded")
    assert combo.currentData() == str(missed.path())
