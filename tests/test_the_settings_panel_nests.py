"""The settings panel groups by what a setting DOES, three levels deep.

Instruction 73, 2026-08-10: "move settings like remove background, intensity
merge split, min distance, etc. but have all the similar settings here
organized into sub and sub-sub sections with sub section headers like object
filtration, image preprocessing with sub-sub sections for each object (cell,
nuclei, pathogen, organelle)."

The panel could not express that. `SettingsWidgets.build_sections` returned
one header and its rows -- no third element, no recursion -- so a
sub-sub-section did not exist. Widening that return type is a contract change
for every module in the tool, which is why what it returns now is a TUPLE
SUBCLASS: it still IS the pair it always was, and the tree hangs off
attributes beside it.

The two failures worth excluding are opposite:

  * a caller that unpacks the old pair breaking on the new type, and
  * a control reaching no heading at all, which is a control the user cannot
    reach -- the same failure as hiding one, arrived at by omission.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import spacr.settings as S
from spacr.qt.screens.settings_model import (
    OBJECT_SUBHEADING_TOOLTIPS,
    SettingsSection,
    SettingsWidgets,
    _shared_category_parents,
    category_tooltip,
    section_tooltip,
    section_tooltip_is_curated,
)


@pytest.fixture(scope="module")
def mask_model(qapp):
    """Keep the full-slot Mask model alive for its section-tree checks."""
    from spacr.qt.screens.settings_model import PANEL_ORGANELLE_SLOTS

    return SettingsWidgets("mask", current={
        "number_of_organelles": PANEL_ORGANELLE_SLOTS,
        "nucleus_channel": 1,
        "pathogen_channel": 2,
    })


@pytest.fixture(scope="module")
def mask_tree(mask_model):
    """The mask panel's sections. Mask is the module with the most of them."""
    return mask_model.build_sections()


@pytest.fixture(scope="module")
def measure_tree(qapp):
    """Keep the current Measure model and its native row owners alive."""
    model = SettingsWidgets("measure")
    tree = model.build_sections()
    yield tree


def _walk(sections):
    for section in sections:
        yield from section.walk()


def _named(sections, title):
    for section in _walk(sections):
        if section.title == title:
            return section
    raise AssertionError(
        f"no {title!r} section among {[s.title for s in _walk(sections)]}")


# ---------------------------------------------------------------------------
# The old contract, which every module still depends on
# ---------------------------------------------------------------------------

def test_a_section_is_still_the_pair_it_always_was(mask_tree):
    """Eleven call sites unpack this result; none of them may have to change.

    `for title, rows in build_sections()` and `dict(build_sections())` are
    both in the tree today, in the screens and in the tests, so the nesting
    had to be additive rather than a new shape.
    """
    for section in mask_tree:
        title, rows = section
        assert title == section.title
        assert rows == section.rows
        assert section == (title, rows)
        assert len(section) == 2
    assert dict(mask_tree)


def test_a_flat_reader_sees_every_row_exactly_once(mask_tree, mask_model):
    """A panel that cannot draw a tree must still draw every control.

    The outer `rows` therefore holds the WHOLE subtree, not just the rows the
    umbrella owns itself -- an umbrella whose own rows were empty and whose
    children were invisible to a flat reader would silently drop 128 controls
    out of the mask panel while leaving every one of them in the settings
    dict, which is how a run gets values nobody can see.
    """
    flat = [widget for _title, rows in mask_tree for _label, widget in rows]
    assert len(flat) == len({id(w) for w in flat}), "a row rendered twice"

    assert len(flat) == len(mask_model._widgets)


def test_a_category_with_no_parent_is_still_one_flat_section(mask_tree):
    """The nesting is opt-in per category, not a reshuffle of the panel."""
    # 2026-09-29 (item 592): Runtime & Reliability was merged into Mask's
    # "Quality Control", which now nests Segmentation Robustness α; a heading
    # that still has no parent and no children is Visualization & Diagnostics.
    plain = _named(mask_tree, "Visualization & Diagnostics")
    assert plain.children == ()
    assert plain.own_rows == plain.rows
    assert plain.path == ("Visualization & Diagnostics",)


# ---------------------------------------------------------------------------
# The three levels the request asked for
# ---------------------------------------------------------------------------

def test_the_advanced_families_nest_under_one_umbrella(measure_tree):
    """The current correction families share one populated umbrella."""
    umbrella = _named(measure_tree, "Image Preprocessing")
    assert [child.title for child in umbrella.children] == [
        "Bleach Correction α", "Spectral Unmixing α",
        "Image Deconvolution (PSF)", "Illumination Correction",
        "Intensity Calibration α", "Plate Barcode Linkage α",
    ]
    assert umbrella.own_rows == [], "the umbrella holds headings, not rows"
    assert umbrella.rows, "and it reports the rows underneath it"


def test_the_umbrella_takes_the_place_of_its_first_family(measure_tree):
    """The current correction umbrella retains its workflow position."""
    titles = [section.title for section in measure_tree]
    umbrella = titles.index("Image Preprocessing")
    assert titles[umbrella - 1] == "Mask & Channel Mapping"
    assert titles[umbrella + 1] == "Features"
    assert "Bleach Correction α" not in titles


def test_each_family_splits_into_a_sub_section_per_object(measure_tree):
    """Current nested families own their declared rows exactly once."""
    from spacr.qt.screens.settings_model import categories_for_app, resolve_default_settings

    umbrella = _named(measure_tree, "Image Preprocessing")
    categories = categories_for_app("measure", S.categories)
    defaults = resolve_default_settings("measure")
    assert umbrella.children
    seen = set()
    for child in umbrella.children:
        expected = [key for key in categories[child.title] if key in defaults]
        actual = [widget.property("settingKey") for _label, widget in child.own_rows]
        assert actual == expected and actual, f"{child.title} lost its own rows"
        assert child.children == ()
        assert not set(actual) & seen
        seen.update(actual)
    assert len(umbrella.rows) == len(seen)


def test_a_sub_section_records_the_whole_path_down_to_it(measure_tree):
    umbrella = _named(measure_tree, "Image Preprocessing")
    assert umbrella.path == ("Image Preprocessing",)
    child = umbrella.children[0]
    assert child.title == "Bleach Correction α"
    assert child.path == ("Image Preprocessing", "Bleach Correction α")


def test_an_app_layout_that_renames_a_family_keeps_its_place():
    """Mask spells "Object filtration" its own way and still nests.

    Derived from the group the layout entry references rather than restated
    in a second table: this project has already shipped three defects from a
    module being registered in one table and not its partner.
    """
    parents = _shared_category_parents()
    assert parents["Object filtration"] == "Advanced settings"
    assert parents["Object Filtration (all objects)"] == "Advanced settings"
    assert parents["Image Preprocessing (per object)"] == "Advanced settings"
    assert "Quality Control" not in parents


# ---------------------------------------------------------------------------
# Image preprocessing, and the gap it makes visible
# ---------------------------------------------------------------------------

def test_the_preprocessing_family_shows_each_object_what_it_actually_has(measure_tree):
    """Schema object differences survive while Measure renders its corrections."""
    from spacr.settings import advanced_object_of

    keys = S.categories[S.PER_OBJECT_PREPROCESSING]
    sizes = {role: sum(advanced_object_of(key) == role for key in keys)
             for role in ("cell", "nucleus", "pathogen", "organelle")}
    assert sizes["cell"] == sizes["nucleus"] == sizes["pathogen"] == 3
    assert sizes["organelle"] == 7
    family = _named(measure_tree, "Image Preprocessing")
    assert family.children and family.rows
    assert all(child.own_rows for child in family.children)
    assert [widget for child in family.children for _label, widget in child.rows] == [
        widget for _label, widget in family.rows]


def test_the_per_object_family_is_not_the_whole_image_one():
    """Two headings spelled alike would share one blurb.

    The category-help table is keyed on the heading's exact text. Mask
    already draws "Image Preprocessing" for the whole-image steps --
    normalize, upscale, denoise -- so a per-object group spelled the same way
    would silently serve that group's help, which is the half of the
    "Computer Vision --" prefix precedent that broke every lookup for its
    groups.
    """
    assert S.PER_OBJECT_PREPROCESSING == "Image preprocessing (per object)"
    assert (category_tooltip("mask", "Image Preprocessing (per object)")
            != category_tooltip("mask", "Image Preprocessing"))


def test_the_background_floor_left_the_per_object_segmentation_headings():
    """It is now filed with the same decision taken for the other objects."""
    preprocessing = S.categories[S.PER_OBJECT_PREPROCESSING]
    for key in ("cell_background", "nucleus_background", "pathogen_background",
                "remove_background_cell", "remove_background_nucleus",
                "remove_background_pathogen"):
        assert key in preprocessing, key
    for heading in ("Cell", "Nucleus", "Pathogen"):
        assert not (set(S.categories[heading]) & set(preprocessing)), heading


def test_the_families_still_only_hold_keys_that_already_existed():
    """The regroup MOVES settings. A settings CSV names keys, not headings,
    so a file written before it loads and means exactly what it meant."""
    known = set(S.expected_types)
    for heading in S.CATEGORY_PARENTS:
        for key in S.categories.get(heading, ()):
            assert key in known, (heading, key)


# ---------------------------------------------------------------------------
# Help, which is keyed on the heading text and therefore collides
# ---------------------------------------------------------------------------

def test_every_heading_in_the_tree_has_written_help(mask_tree):
    """A new heading with no blurb is a header the reader has to guess at."""
    bare = [section.path for section in _walk(mask_tree)
            if not section_tooltip_is_curated("mask", section)]
    assert not bare, f"headings with no written help: {bare}"


def test_a_sub_heading_is_not_given_the_categorys_help(measure_tree):
    """A populated nested child receives its own guidance, not its parent's."""
    parent = _named(measure_tree, "Input & Experiment")
    child = _named(parent.children, "Cloud α")
    assert parent.own_rows and child.own_rows
    assert child.path == ("Input & Experiment", "Cloud α")
    guidance = section_tooltip("measure", child)
    assert guidance and guidance == category_tooltip("measure", "Cloud α")
    assert guidance != category_tooltip("measure", parent.title)
    assert "Cellpose" not in guidance


def test_a_top_level_heading_keeps_its_own_help(mask_tree):
    """The path-aware lookup may not change what an unnested heading says."""
    quality = _named(mask_tree, "Quality Control")
    assert section_tooltip("mask", quality) == \
        category_tooltip("mask", "Quality Control")


def test_a_plain_pair_resolves_the_way_it_always_did():
    """`section_tooltip` takes anything shaped like a section, so a caller
    holding an old-style pair is not a special case at the call site."""
    assert section_tooltip("mask", ("Quality Control", [])) == \
        category_tooltip("mask", "Quality Control")


def test_every_object_sub_heading_spaCR_can_draw_has_help():
    """Including the slots a screen with four organelle channels reaches."""
    from spacr.object_roles import organelle_label
    from spacr.schema import ORGANELLE_ROLES

    expected = {"CELL", "NUCLEUS", "PATHOGEN", "CYTOPLASM"}
    expected |= {organelle_label(r).upper() for r in ORGANELLE_ROLES}
    assert expected <= set(OBJECT_SUBHEADING_TOOLTIPS)


# ---------------------------------------------------------------------------
# The section type itself
# ---------------------------------------------------------------------------

def test_a_row_whose_key_names_no_object_stays_with_its_family():
    """It may not be dropped on the way to a sub-heading.

    A control that reaches no heading is a control the user cannot reach,
    and the split is driven by a NAME -- so the day a family gains a member
    spelled some third way, the failure has to be a visible row under the
    family rather than a missing one.
    """
    from spacr.qt.screens.settings_model import _split_rows_by_object

    rows = [("Cell min size", object()), ("Something shared", object())]
    own, children = _split_rows_by_object(rows, ["cell_min_size", "shared"])
    assert own == [rows[1]]
    assert [c.title for c in children] == ["Cell"]
    assert children[0].own_rows == [rows[0]]


def test_walk_reaches_every_heading():
    leaf = SettingsSection("Leaf", [("a", object())])
    middle = SettingsSection("Middle", (), [leaf])
    root = SettingsSection("Root", (), [middle])
    assert [s.title for s in root.walk()] == ["Root", "Middle", "Leaf"]
    assert leaf.path == ("Root", "Middle", "Leaf")
    assert root.rows == leaf.rows
