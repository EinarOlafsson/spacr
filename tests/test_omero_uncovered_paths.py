"""What the OMERO bridge does when the server hands back less than it usually does.

An OMERO wrapper is a remote proxy, and the calls this module makes on one can
fail, be absent, or return nothing at all. None of that may become a traceback
in the middle of an import, and none of it may become a wrong number in an
OMERO panel. Each test here drives one such shortfall and pins the answer.
"""
from __future__ import annotations

import pytest

from spacr import omero
from tests.test_omero import (FakeDataset, FakeImage, FakePlate,
                              build_gateway, build_plate)


class _SampleLessWell:
    """A well that reports a sample count but cannot hand a sample over.

    ``WellWrapper`` exposes several ways to reach its fields depending on the
    omero-py version, and a proxy that has none of them still answers
    ``countWellSample()``. Counting three and then producing none is the shape
    the reader has to survive.
    """

    def __init__(self, row, column, count):
        self.row = row
        self.column = column
        self.count = count

    def getRow(self):
        return self.row

    def getColumn(self):
        return self.column

    def countWellSample(self):
        return self.count


class _NamelessImage(FakeImage):
    """An image whose ``getName()`` fails the way a dropped session does."""

    def getName(self):
        raise RuntimeError("Ice.ConnectionLostException")


# --- the id guard, when the caller does not say what it expects -------------

def test_a_typed_reference_is_accepted_when_no_type_is_expected():
    """With no ``expect``, the id comes back and the type is not questioned."""
    assert omero.parse_object_id("Plate:4711") == 4711
    assert omero.parse_object_id("Dataset:12") == 12


def test_a_bare_id_is_returned_unchanged_when_no_type_is_expected():
    """A bare integer names no type, and none is demanded of it."""
    assert omero.parse_object_id(" 88 ") == 88


def test_the_type_guard_still_bites_when_a_type_is_expected():
    """The contrast: naming the wrong type is refused, and says which."""
    with pytest.raises(omero.OmeroIdError, match="is a Plate, but a Dataset"):
        omero.parse_object_id("Plate:4711", expect="Dataset")


# --- reading a well that will not list its fields ---------------------------

def test_a_well_that_counts_fields_it_cannot_hand_over_contributes_no_images():
    """The well is still placed on the plate; it just brings no images.

    Skipping the well entirely would lose ``A01`` from the plate's well list,
    which is the geometry every later measurement is keyed on.
    """
    plate = FakePlate(4711, "Assay plate 3", [_SampleLessWell(0, 0, 3)])
    gateway = build_gateway(plate=plate)

    listing = omero.inspect_container(gateway, "Plate:4711")

    assert listing.wells == ("A01",)
    assert listing.images == ()
    assert listing.n_planes == 0
    assert listing.unplaced_wells == 0


def test_a_well_that_can_hand_its_fields_over_one_by_one_is_read():
    """The contrast: the same count, with the getter present, yields images."""

    class _IndexedWell(_SampleLessWell):
        def __init__(self, row, column, images):
            super().__init__(row, column, len(images))
            self.images = list(images)

        def getWellSample(self, index):
            return self.images[index]

    plate = FakePlate(4711, "Assay plate 3",
                      [_IndexedWell(0, 0, [FakeImage(11, "A01 site 1")])])
    listing = omero.inspect_container(build_gateway(plate=plate), "Plate:4711")

    assert listing.wells == ("A01",)
    assert [image.image_id for image in listing.images] == [11]


# --- a call that raises on the server ---------------------------------------

def test_an_image_whose_name_call_fails_is_still_listed_without_a_name():
    """One failed metadata call costs the name, not the whole inspection."""
    dataset = FakeDataset(7, "dropped session", [_NamelessImage(21, "unused")])
    gateway = build_gateway(dataset=dataset)

    listing = omero.inspect_container(gateway, "Dataset:7")

    assert listing.n_images == 1
    assert listing.images[0].name == ""
    assert listing.images[0].image_id == 21
    assert listing.images[0].size_x == 4


# --- describing a container that has neither wells nor images ---------------

def test_an_empty_dataset_describes_itself_without_wells_or_dimensions():
    """No images means no size line and no channel line to invent.

    The count and the import estimate are still printed, because "0" is the
    answer the user asked for.
    """
    gateway = build_gateway(dataset=FakeDataset(7, "archive", []))

    listing = omero.inspect_container(gateway, "Dataset:7")
    text = listing.describe()

    assert "images  : 0" in text
    assert "would write 0 TIFF(s)" in text
    assert "wells" not in text
    assert "size" not in text
    assert "channels" not in text


# --- describing an export that found every target ---------------------------

def test_an_export_that_found_every_well_says_nothing_about_missing_ones():
    """No 'not found' clause when nothing was missing."""
    plate = build_plate()
    gateway = build_gateway(plate=plate)

    result = omero.export_plate_summaries(
        gateway, "Plate:4711",
        {"A01": [{"cell_area": 10.0}], "B06": [{"cell_area": 5.0}]},
        annotation_factory=gateway.annotation_factory)

    assert result.missing == ()
    text = result.describe()
    assert "not found" not in text
    assert "2 create" in text
    assert text.startswith(omero.NS_WELL_SUMMARY)
