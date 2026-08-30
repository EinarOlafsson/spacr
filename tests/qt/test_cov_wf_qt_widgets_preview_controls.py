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

import sys
from pathlib import Path
from types import SimpleNamespace

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
# A channel entry that is not "Ch <n>"
# ---------------------------------------------------------------------------

def test_a_channel_entry_named_after_a_stain_selects_no_index(qtbot):
    """A dropdown filled with stain names must not be read as an index.

    :func:`~spacr.qt.widgets.preview_controls.selected_channel` feeds
    :func:`~spacr.qt.widgets.preview_controls.channel_view`, which slices the
    loaded stack on whatever comes back. Only ``Ch <n>`` names a plane; a panel
    (or a saved setting) that put ``DAPI`` in the combo must yield ``None`` --
    "show the source exactly as stored" -- because guessing a number out of a
    stain name would show the user a different channel than the one the entry
    names, with nothing on screen saying so.
    """
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(["DAPI", "Ch 2"])

    assert combo.currentText() == "DAPI"
    assert pc.selected_channel(combo) is None

    combo.setCurrentIndex(1)
    assert pc.selected_channel(combo) == 2


# ---------------------------------------------------------------------------
# A field of view the list no longer holds
# ---------------------------------------------------------------------------

def test_a_field_outside_the_sample_leaves_the_dropdown_on_its_first_entry(
        qtbot, tmp_path):
    """Re-drawing the sample must not blank the field-of-view dropdown.

    The panels rebuild their selectors on every load and hand
    :func:`~spacr.qt.widgets.preview_controls.populate_fov_combo` whatever
    path is loaded. After a reshuffle or a change of cap that path can be
    absent from the new sample; the combo must then simply stay on its first
    entry, still carrying every path as item data, so the next step through
    the plate works. Selecting an index of -1 would leave the panel with no
    source at all and no way to pick one but the file dialog.
    """
    sources = [tmp_path / "A01 f001.tif", tmp_path / "A02 f001.tif"]
    labels = ["A01 f001", "A02 f001"]
    dropped = tmp_path / "B12 f009.tif"
    combo = QComboBox()
    qtbot.addWidget(combo)

    pc.populate_fov_combo(combo, sources, current=dropped, labels=labels)

    assert _entries(combo) == labels
    assert [combo.itemData(i) for i in range(combo.count())] == \
        [str(p) for p in sources]
    assert combo.currentIndex() == 0
    assert combo.currentData() == str(sources[0])

    # The same call with a path the list does hold moves the selection, so the
    # index above is the missing entry's doing and not a dead code path.
    pc.populate_fov_combo(combo, sources, current=sources[1], labels=labels)
    assert combo.currentIndex() == 1
    assert combo.currentData() == str(sources[1])


# ---------------------------------------------------------------------------
# A naming dialect the project does not know
# ---------------------------------------------------------------------------

def test_a_naming_dialect_spacr_does_not_know_still_lists_every_file(tmp_path):
    """An unrecognised ``metadata_type`` must cost grouping, not the panel.

    ``_get_regex`` raises for a dialect outside its vocabulary, and the
    preview swallows that into "no pattern for this extension". The user has
    then typed something wrong in the settings, and what they must get is the
    plain per-file listing the previews always had -- every name in the
    dropdown -- rather than an empty source selector that gives them nothing to
    look at while they work out what the setting should say.
    """
    root = tmp_path / "plate"
    names = _plate(root)

    grouped, channels = pc.enumerate_image_sets(root, [".tif"], "cellvoyager")
    assert [s.label for s in grouped] == [
        "A01 f001 (2ch)", "A01 f002 (2ch)", "A02 f001 (2ch)",
        "A02 f002 (2ch)", "B03 f001 (2ch)", "B03 f002 (2ch)"]
    assert channels == ["01", "02"]

    flat, no_channels = pc.enumerate_image_sets(root, [".tif"], "klingon")
    assert pc._acquisition_regex("klingon", "tif") is None
    assert [s.key for s in flat] == [("", "", n) for n in names]
    assert [s.label for s in flat] == names
    assert len(flat) == len(names) == 2 * len(grouped)
    assert no_channels == []


def test_an_unparsable_custom_regex_falls_back_to_one_set_per_file(tmp_path):
    """A half-typed custom pattern must not take the preview down.

    ``metadata_type='custom'`` hands the user's own text to ``re.compile``, and
    a pattern is unbalanced for as long as it takes to type it. Every keystroke
    would otherwise raise out of the enumeration and leave the panel dead;
    instead the folder lists one entry per file until the pattern closes.
    """
    root = tmp_path / "plate"
    names = _plate(root)

    broken, no_channels = pc.enumerate_image_sets(
        root, [".tif"], "custom", "(unclosed")
    assert pc._acquisition_regex("custom", "tif", "(unclosed") is None
    assert [s.key for s in broken] == [("", "", n) for n in names]
    assert no_channels == []

    # Close the group and the very same folder groups by well and field, so
    # the flat listing above is the pattern's doing.
    good, channels = pc.enumerate_image_sets(
        root, [".tif"], "custom",
        r".*_(?P<wellID>[A-Z]\d+)_T\d+F(?P<fieldID>\d+)"
        r"L\d+A\d+Z\d+C(?P<chanID>\d+)")
    assert [s.label for s in good] == [
        "A01 f001 (2ch)", "A01 f002 (2ch)", "A02 f001 (2ch)",
        "A02 f002 (2ch)", "B03 f001 (2ch)", "B03 f002 (2ch)"]
    assert channels == ["01", "02"]


# ---------------------------------------------------------------------------
# A spacr/utils.py that no longer defines _get_regex
# ---------------------------------------------------------------------------

def test_a_utils_source_without_get_regex_degrades_instead_of_raising(
        tmp_path, monkeypatch):
    """The import-free "lift the function out of the source" trick must fail soft.

    The Qt layer refuses to import ``spacr.utils`` (3.2 s and ~900 MB of RSS,
    for a dropdown), so when the module is not already loaded it parses
    ``utils.py`` and compiles ``_get_regex`` alone out of the AST. That trick
    is one rename away from finding nothing: the loop over the module body then
    falls through with no function to return. What must happen next is the
    documented degradation -- no pattern, one set per file -- and not a
    ``NameError`` or a ``KeyError`` escaping into whichever preview happened to
    be building its selector.
    """
    stub = tmp_path / "utils_without_the_helper.py"
    stub.write_text("def _something_else():\n    return 'not the regex'\n",
                    encoding="utf8")
    root = tmp_path / "plate"
    names = _plate(root)

    monkeypatch.delitem(sys.modules, "spacr.utils", raising=False)
    monkeypatch.setattr(pc, "importlib", SimpleNamespace(
        util=SimpleNamespace(find_spec=lambda name: SimpleNamespace(
            origin=str(stub)))))
    pc._get_regex_callable.cache_clear()
    pc._acquisition_regex.cache_clear()

    assert pc._get_regex_callable() is None
    assert pc._acquisition_regex("cellvoyager", "tif") is None
    orphaned, no_channels = pc.enumerate_image_sets(
        root, [".tif"], "cellvoyager")
    assert [s.key for s in orphaned] == [("", "", n) for n in names]
    assert no_channels == []

    # Put the real module source back and the identical folder groups again,
    # which is what makes the listing above the missing helper's doing.
    monkeypatch.undo()
    pc._get_regex_callable.cache_clear()
    pc._acquisition_regex.cache_clear()
    grouped, channels = pc.enumerate_image_sets(root, [".tif"], "cellvoyager")
    assert callable(pc._get_regex_callable())
    assert [s.label for s in grouped] == [
        "A01 f001 (2ch)", "A01 f002 (2ch)", "A02 f001 (2ch)",
        "A02 f002 (2ch)", "B03 f001 (2ch)", "B03 f002 (2ch)"]
    assert channels == ["01", "02"]


def test_a_utils_source_that_cannot_be_located_is_logged_not_raised(
        tmp_path, monkeypatch, caplog):
    """A broken ``find_spec`` must leave a trace instead of a traceback.

    ``find_spec`` returning ``None`` -- a zipped install, a stripped wheel --
    makes ``spec.origin`` an ``AttributeError`` deep inside a dropdown build.
    The helper swallows it and says so at debug level, because the alternative
    is a preview panel that refuses to open and a stack trace naming
    ``importlib`` rather than the packaging problem it really is.
    """
    monkeypatch.delitem(sys.modules, "spacr.utils", raising=False)
    monkeypatch.setattr(pc, "importlib", SimpleNamespace(
        util=SimpleNamespace(find_spec=lambda name: None)))
    pc._get_regex_callable.cache_clear()

    with caplog.at_level("DEBUG", logger="spacr.qt.preview_controls"):
        assert pc._get_regex_callable() is None
    assert any("Could not lift _get_regex" in r.message
               for r in caplog.records)


# ---------------------------------------------------------------------------
# A sets dropdown with no cap box beside it
# ---------------------------------------------------------------------------

def test_a_sets_dropdown_without_a_cap_box_still_says_it_is_a_sample(tmp_path):
    """Mask keeps its cap box elsewhere; the sentence must still be written.

    :func:`~spacr.qt.widgets.preview_controls.apply_sample_to_combo` takes the
    cap box as optional because Mask's cap sits beside the ``Choose image…``
    button rather than beside a combo. Passing ``None`` must configure nothing
    and still fill the dropdown and return the "showing N of M" sentence: a
    sampled preview that does not say it is a sample is a preview the user
    reads as the whole plate.
    """
    root = tmp_path / "plate"
    _plate(root)
    sampler = pc.ImageSetSampler(max_sets=2)
    sampler.enumerate(root, [".tif"])
    assert sampler.total == 6
    combo = QComboBox()

    note = pc.apply_sample_to_combo(combo, None, sampler, None,
                                    tooltip="Image set")

    assert note == (f"showing a random sample of 2 of 6 image sets "
                    f"(seed {sampler.seed:016x})")
    assert combo.toolTip() == f"Image set — {note}."
    assert len(_entries(combo)) == 2
    # The sample is drawn across the plate and then restored to plate order.
    every = [s.label for s in sampler.sets]
    assert set(_entries(combo)) <= set(every)
    assert _entries(combo) == sorted(_entries(combo), key=every.index)
    # Nothing was clamped or disabled, because there was no box to touch.
    assert sampler.max_sets == 2


def test_a_cap_box_beside_the_dropdown_is_clamped_to_the_folder(tmp_path):
    """The cap box and the dropdown must never disagree about the sample.

    With a box present the same call points it at the freshly enumerated
    folder: the suffix carries the total so the control reads ``3 of 6 sets``,
    the maximum is clamped to what exists so it can never offer to show 50 of
    6, and the clamped value is fed back to the sampler. Skipping that
    feedback is what once left the box reading 6 while the dropdown listed 50
    entries' worth of nothing.
    """
    root = tmp_path / "plate"
    _plate(root)
    sampler = pc.ImageSetSampler(max_sets=2)
    sampler.enumerate(root, [".tif"])
    box = pc.FlatSpinBox(value=50)
    combo = QComboBox()

    note = pc.apply_sample_to_combo(combo, box, sampler, None)

    assert box.suffix() == " of 6 sets"
    assert box.maximum() == 6
    assert box.value() == 6
    assert box.isEnabled() is True
    assert sampler.max_sets == 6
    assert note == "showing all 6 image sets"
    assert _entries(combo) == [s.label for s in sampler.sets]


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
