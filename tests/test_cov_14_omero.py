"""The OMERO bridge answers from whatever the server actually hands it.

An OMERO object is a remote proxy, and the attributes this module reads are
optional on it: a pixel size may be a unit-carrying object, a bare float, or
absent; a well may list its samples or make you count them; an annotation
value may be a numpy scalar, a Decimal, or something that only claims to be a
number. None of that may reach a user as a traceback in the middle of a plate
import, and none of it may reach an OMERO panel as a wrong number.

So each reader here has a documented fallback, and the tests below drive the
fallbacks rather than the happy path -- the happy path is covered by the
end-to-end import in ``tests/test_omero.py``.
"""
from __future__ import annotations

import sys
import types

import pytest

from spacr import omero, schema
from tests.test_omero import (FakeGateway, FakeImage, FakePlate, FakeWell,
                              build_gateway, build_plate)


# -- the extra ---------------------------------------------------------------

def test_a_gateway_that_resolves_to_spacrs_own_module_is_refused(monkeypatch):
    """A self-import is named as a broken ``sys.path``, not as a missing extra.

    ``spacr/omero.py`` shadowing ``omero.gateway`` gives an AttributeError on
    ``BlitzGateway`` several calls later, which reads as an OMERO version
    problem and is not one.
    """
    impostor = types.ModuleType(omero.OMERO_GATEWAY_MODULE)
    impostor.__file__ = omero.__file__
    monkeypatch.setitem(sys.modules, omero.OMERO_GATEWAY_MODULE, impostor)

    with pytest.raises(omero.OmeroExtraMissing, match="sys.path"):
        omero.require_omero()


def test_the_default_gateway_factory_builds_from_the_settings(monkeypatch):
    """The one call that needs omero-py passes host, port and credential on.

    A session key is used INSTEAD of a password, never alongside it: sending
    both is how a stale password ends up in a login attempt.
    """
    built = []

    gateway_module = types.ModuleType(omero.OMERO_GATEWAY_MODULE)
    gateway_module.__file__ = "/site-packages/omero/gateway.py"

    def _blitz(user, credential, host=None, port=None, secure=None):
        built.append((user, credential, host, port, secure))
        return "gateway"

    gateway_module.BlitzGateway = _blitz
    monkeypatch.setitem(sys.modules, omero.OMERO_GATEWAY_MODULE,
                        gateway_module)

    with_password = omero.OmeroConnection(
        host="omero.example.org", port=4064, username="jdoe",
        password="secret", secure=True)
    with_key = omero.OmeroConnection(
        host="omero.example.org", port=4064, username="jdoe",
        session_key="abc123", secure=True)

    assert omero._default_gateway_factory(with_password) == "gateway"
    assert omero._default_gateway_factory(with_key) == "gateway"

    assert built[0] == ("jdoe", "secret", "omero.example.org", 4064, True)
    assert built[1] == ("jdoe", "abc123", "omero.example.org", 4064, True)


def test_an_annotation_wrapper_comes_from_omero_py_when_no_factory_is_given(
        monkeypatch):
    """Without an injected factory the wrapper is looked up on omero-py."""
    gateway_module = types.ModuleType(omero.OMERO_GATEWAY_MODULE)
    gateway_module.__file__ = "/site-packages/omero/gateway.py"
    gateway_module.MapAnnotationWrapper = lambda gateway: ("wrapper", gateway)
    monkeypatch.setitem(sys.modules, omero.OMERO_GATEWAY_MODULE,
                        gateway_module)

    made = omero._make_annotation("gw", "MapAnnotationWrapper", None)

    assert made == ("wrapper", "gw")


# -- connection settings ------------------------------------------------------

def test_a_group_is_named_in_the_connection_description():
    """The group reaches the log line, because it changes what is visible."""
    settings = omero.OmeroConnection(
        host="omero.example.org", port=4064, username="jdoe",
        password="secret", secure=True, group="Screening")

    described = settings.describe()

    assert "group=Screening" in described
    assert "secret" not in described


def test_a_boolean_setting_is_taken_as_it_is():
    """A real bool needs no parsing, and a yes/no word is understood."""
    assert omero._parse_bool(True, source="secure") is True
    assert omero._parse_bool(False, source="secure") is False
    assert omero._parse_bool("yes", source="secure") is True
    assert omero._parse_bool("off", source="secure") is False


def test_a_connection_switches_to_the_named_group():
    """A configured group is selected on the session once it is open.

    A user in several groups sees a different set of plates in each; without
    this the import silently reads whichever group happens to be default.
    """
    class _Gateway:
        def __init__(self):
            self.group = None

        def connect(self):
            return True

        def setGroupNameForSession(self, name):
            self.group = name

    gateway = _Gateway()
    settings = omero.OmeroConnection(
        host="omero.example.org", port=4064, username="jdoe",
        password="secret", secure=True, group="Screening")

    returned = omero.connect(settings, gateway_factory=lambda _s: gateway)

    assert returned is gateway
    assert gateway.group == "Screening"


# -- well names ---------------------------------------------------------------

def test_a_well_the_schema_cannot_place_is_refused(monkeypatch):
    """An id the schema declines to index is refused, not turned into ``None-1``.

    ``omero_indices`` subtracts one from each index; a ``None`` there is a
    TypeError several frames inside the export loop instead of a named refusal
    about the well that caused it.
    """
    monkeypatch.setattr(schema, "row_index", lambda value: None)

    with pytest.raises(omero.OmeroWellError, match="A01"):
        omero.omero_indices("A01")


def test_a_name_token_that_is_not_a_well_is_skipped(monkeypatch):
    """A token the parser rejects lets the search move on to the next one.

    An image name often carries several delimited tokens; stopping at the
    first unparseable one would lose the well that is in the name.
    """
    real = schema.parse_well

    def _picky(value, *, strict=False):
        if str(value).upper() == "AB12":
            raise schema.WellParseError("not a well here")
        return real(value, strict=strict)

    monkeypatch.setattr(schema, "parse_well", _picky)

    assert omero.well_from_image_name("img_AB12_C03_1.tif") == "C03"


# -- pixel sizes --------------------------------------------------------------

def test_an_object_with_none_of_the_named_accessors_gives_the_default():
    """``_call`` falls back rather than raising on a proxy without the method."""
    assert omero._call(object(), "getWellSamples", default="fallback") == \
        "fallback"


# -- walking a plate ----------------------------------------------------------

def test_a_well_that_only_counts_its_samples_is_still_walked():
    """A well with no ``listChildren`` is walked through its sample count.

    Older OMERO well proxies expose ``countWellSample`` /
    ``getWellSample(i)`` and nothing else; treating that as an empty well
    would import a plate with no images and report success.
    """
    class _CountingWell:
        def __init__(self, images):
            self.images = list(images)

        def countWellSample(self):
            return len(self.images)

        def getWellSample(self, index):
            return self.images[index]

    class _Sample:
        def __init__(self, image):
            self.image = image

        def getImage(self):
            return self.image

    first, second = FakeImage(1, "A01 site 1"), FakeImage(2, "A01 site 2")
    well = _CountingWell([_Sample(first), None, _Sample(second)])

    walked = list(omero._iter_well_images(well))

    assert walked == [first, second]


def test_a_well_sample_with_no_image_at_all_is_skipped(monkeypatch, tmp_path):
    """A field that resolves to nothing is not counted as a field.

    Counting it would shift every later field's number, and the field number
    is part of the TIFF filename spaCR then reads back.
    """
    plate = build_plate()
    gateway = build_gateway(plate=plate)
    real = omero._iter_well_images

    def _with_a_hole(well):
        yield None
        for image in real(well):
            yield image

    monkeypatch.setattr(omero, "_iter_well_images", _with_a_hole)

    listing = omero.inspect_container(gateway, f"Plate:{plate.plate_id}")

    assert [info.field_id for info in listing.images if info.well == "A01"] \
        == [2, 3]

    result = omero.import_plate(gateway, f"Plate:{plate.plate_id}",
                                tmp_path / "out", dry_run=True)

    assert result.n_images == 3
    assert all("_f1_" not in plan.filename for plan in result.planned)


def test_an_import_stops_at_its_limit(tmp_path):
    """``limit`` stops the walk and the result says it was truncated.

    Reporting a limited import as complete is how a five-image trial run gets
    mistaken for the whole plate.
    """
    from tests.test_omero import FakeDataset

    dataset = FakeDataset(77, "trial", [FakeImage(21, "A01_1"),
                                        FakeImage(22, "A02_1"),
                                        FakeImage(23, "A03_1")])
    gateway = build_gateway(dataset=dataset)

    result = omero.import_dataset(gateway, "Dataset:77", tmp_path / "out",
                                  limit=1, dry_run=True)

    assert result.n_images == 1
    assert result.limited is True


# -- rendering values ---------------------------------------------------------

def test_a_key_longer_than_the_limit_is_truncated():
    """An over-long key is shortened with an ellipsis, never silently cut."""
    key = "a" * (omero.MAX_KEY_CHARS + 20)

    cleaned = omero._clean_key(key)

    assert len(cleaned) == omero.MAX_KEY_CHARS
    assert cleaned.endswith("…")


# -- exporting per-well summaries ---------------------------------------------

def test_a_well_with_no_position_is_skipped_on_export():
    """A well whose row/column OMERO cannot give is passed over.

    An unplaced well cannot be matched to a spaCR summary, and guessing one
    would write another well's numbers onto it.
    """
    unplaced = FakeWell(None, None, [FakeImage(31, "A01 site 1")])
    placed = FakeWell(0, 0, [FakeImage(32, "A01 site 1")])
    plate = FakePlate(99, "plate", [unplaced, placed])
    gateway = build_gateway(plate=plate)

    result = omero.export_plate_summaries(
        gateway, "Plate:99", {"A01": [{"area": 2.0}]},
        annotation_factory=gateway.annotation_factory)

    assert result.targets == ("A01",)
    assert result.missing == ()
    assert len(result.results) == 1
