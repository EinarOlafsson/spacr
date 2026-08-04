"""`spacr.omero` without an OMERO server, and without ``omero-py`` installed.

Everything worth testing about an OMERO bridge is reachable from a machine
that has neither, and that is the point of the way :mod:`spacr.omero` is
split: the arithmetic and the string formatting — the well mapping, the
filenames, the id parsing, the float rendering, the replace-or-append
decision — are pure functions over plain data, and the ``BlitzGateway`` calls
are a thin layer that this file replaces with about a hundred lines of fake.

So these tests cover:

* the missing-extra guard, driven with ``omero`` genuinely unimportable
  (a ``sys.meta_path`` finder that refuses it), mirroring
  ``tests/qt/test_qt_launch_without_pyside6.py``;
* a full Plate import against :class:`FakeGateway`, asserting the exact TIFF
  filenames — hand-written here, not derived from the code under test — and
  the pixel content that lands in each one;
* the OMERO ``(row, column)`` -> spaCR well mapping in both directions,
  including row 25 -> ``Z`` and row 26 -> ``AA``;
* the measurement -> key/value projection, including what a 400-column table
  does;
* replace-vs-append, including that a foreign annotation is untouched and
  that nothing is ever deleted;
* that no password reaches a repr, a log record or a sidecar;
* every refusal path.

``omero-py`` is not installed in this environment and must not become
installed for these tests to pass.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.abc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import convert, omero, schema

REPO_ROOT = Path(__file__).resolve().parents[1]


# ===========================================================================
# The fake gateway
# ===========================================================================

def plane_array(image_id: int, z: int, c: int, t: int) -> np.ndarray:
    """Deterministic pixel content for one plane, shared by fake and assertions."""
    base = (image_id % 40) * 1000 + z * 100 + c * 10 + t
    return np.full((3, 4), base, dtype=np.uint16)


class FakeLength:
    """A stand-in for omero's ``LengthI``: a magnitude plus a unit symbol."""

    def __init__(self, value, symbol):
        self._value = value
        self._symbol = symbol

    def getValue(self):
        return self._value

    def getSymbol(self):
        return self._symbol


class FakeChannel:
    """A stand-in for ``ChannelWrapper``."""

    def __init__(self, label):
        self._label = label

    def getLabel(self):
        return self._label


class FakeAnnotation:
    """A Map/Tag annotation wrapper: namespace, value, and an id on save."""

    def __init__(self, gateway, kind, ns=None, value=None):
        self._gateway = gateway
        self.kind = kind
        self._ns = ns
        self._value = value
        self._id = None
        self.saves = 0

    def getId(self):
        return self._id

    def getNs(self):
        return self._ns

    def setNs(self, ns):
        self._ns = ns

    def getValue(self):
        return self._value

    def setValue(self, value):
        self._value = value

    def save(self):
        self.saves += 1
        if self._id is None:
            self._id = self._gateway.next_annotation_id()
        return self


class Annotatable:
    """The annotation half of every OMERO object wrapper."""

    def __init__(self):
        self.annotations = []
        self.link_calls = 0

    def listAnnotations(self, ns=None):
        # Deliberately ignores `ns`: spacr.omero filters client-side on
        # purpose, and a fake that honoured the filter would hide a bug where
        # it did not.
        return list(self.annotations)

    def linkAnnotation(self, annotation):
        self.link_calls += 1
        self.annotations.append(annotation)
        return annotation


class FakePixels:
    """``getPrimaryPixels()``. Every ``getPlane`` call is recorded."""

    def __init__(self, image):
        self.image = image

    def getPlane(self, z=0, c=0, t=0):
        self.image.plane_calls.append((z, c, t))
        return plane_array(self.image.image_id, z, c, t)


class FakeImage(Annotatable):
    """An ``ImageWrapper``."""

    def __init__(self, image_id, name, size_c=2, size_z=1, size_t=1,
                 size_x=4, size_y=3, channels=None,
                 pixel_size=(0.325, "MICROMETER"), has_pixels=True):
        super().__init__()
        self.image_id = image_id
        self.name = name
        self.size_c = size_c
        self.size_z = size_z
        self.size_t = size_t
        self.size_x = size_x
        self.size_y = size_y
        self.channels = channels or [f"ch{i + 1}" for i in range(size_c)]
        self.pixel_size = pixel_size
        self.has_pixels = has_pixels
        self.plane_calls = []

    def getId(self):
        return self.image_id

    def getName(self):
        return self.name

    def getSizeX(self):
        return self.size_x

    def getSizeY(self):
        return self.size_y

    def getSizeZ(self):
        return self.size_z

    def getSizeC(self):
        return self.size_c

    def getSizeT(self):
        return self.size_t

    def getChannels(self):
        return [FakeChannel(label) for label in self.channels]

    def _length(self):
        if self.pixel_size is None:
            return None
        return FakeLength(*self.pixel_size)

    getPixelSizeX = getPixelSizeY = getPixelSizeZ = _length

    def getPrimaryPixels(self):
        return FakePixels(self) if self.has_pixels else None


class FakeWellSample:
    """A ``WellSampleWrapper`` — one imaging site inside a well."""

    def __init__(self, image):
        self.image = image

    def getImage(self):
        return self.image


class FakeWell(Annotatable):
    """A ``WellWrapper``. ``row``/``column`` are 0-based, as OMERO reports."""

    def __init__(self, row, column, images):
        super().__init__()
        self.row = row
        self.column = column
        self.images = list(images)

    def getRow(self):
        return self.row

    def getColumn(self):
        return self.column

    def listChildren(self):
        return [FakeWellSample(image) for image in self.images]


class FakePlate(Annotatable):
    """A ``PlateWrapper``."""

    def __init__(self, plate_id, name, wells):
        super().__init__()
        self.plate_id = plate_id
        self.name = name
        self.wells = list(wells)

    def getId(self):
        return self.plate_id

    def getName(self):
        return self.name

    def getWells(self):
        return list(self.wells)

    listChildren = getWells


class FakeDataset(Annotatable):
    """A ``DatasetWrapper``."""

    def __init__(self, dataset_id, name, images):
        super().__init__()
        self.dataset_id = dataset_id
        self.name = name
        self.images = list(images)

    def getId(self):
        return self.dataset_id

    def getName(self):
        return self.name

    def listChildren(self):
        return list(self.images)


class FakeGateway:
    """The handful of ``BlitzGateway`` methods :mod:`spacr.omero` actually uses.

    ``deleteObjects`` is present and raises: spaCR promises never to delete
    anything, and the cheapest way to keep that promise honest is to make a
    call fail the test suite loudly.
    """

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.get_object_calls = []
        self.file_annotations = []
        self._annotation_id = 100
        self.connect_result = True
        self.closed = False

    # --- object lookup ---------------------------------------------------
    def getObject(self, kind, object_id):
        self.get_object_calls.append((kind, object_id))
        return self.objects.get((kind, object_id))

    # --- session ---------------------------------------------------------
    def connect(self):
        return self.connect_result

    def close(self):
        self.closed = True

    # --- annotations -----------------------------------------------------
    def next_annotation_id(self):
        self._annotation_id += 1
        return self._annotation_id

    def annotation_factory(self, wrapper_name, gateway):
        assert gateway is self
        return FakeAnnotation(self, wrapper_name)

    def createFileAnnfromLocalFile(self, path, mimetype=None, ns=None, desc=None):
        annotation = FakeAnnotation(self, "FileAnnotationWrapper", ns=ns,
                                    value=str(path))
        annotation.mimetype = mimetype
        annotation.description = desc
        annotation.save()
        self.file_annotations.append(annotation)
        return annotation

    # --- the thing that must never happen --------------------------------
    def deleteObjects(self, *args, **kwargs):        # pragma: no cover - guard
        raise AssertionError(
            "spacr.omero called deleteObjects; it promises never to delete "
            f"anything (args={args!r}, kwargs={kwargs!r})")


def build_plate(plate_id=4711, name="Assay plate 3"):
    """A 2-well plate: A01 with two fields, B06 with one. Two channels each."""
    a01 = FakeWell(0, 0, [FakeImage(11, "A01 site 1"), FakeImage(12, "A01 site 2")])
    b06 = FakeWell(1, 5, [FakeImage(13, "B06 site 1")])
    return FakePlate(plate_id, name, [a01, b06])


def build_gateway(plate=None, dataset=None):
    """A gateway holding one plate and/or one dataset."""
    objects = {}
    if plate is not None:
        objects[("Plate", plate.plate_id)] = plate
    if dataset is not None:
        objects[("Dataset", dataset.dataset_id)] = dataset
    return FakeGateway(objects)


# ===========================================================================
# 1. The missing optional dependency
# ===========================================================================

class _RaiseOnImport(importlib.abc.MetaPathFinder):
    """Make one module name fail to import with a chosen exception."""

    def __init__(self, target, exc):
        self.target = target
        self.exc = exc

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.target or fullname.startswith(self.target + "."):
            raise self.exc
        return None


@pytest.fixture
def block_omero(monkeypatch):
    """Return a callable making ``import omero`` raise ``exc``."""

    def _block(exc=None):
        exc = exc or ModuleNotFoundError("No module named 'omero'", name="omero")
        for name in list(sys.modules):
            if name == "omero" or name.startswith("omero."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(
            sys, "meta_path", [_RaiseOnImport("omero", exc)] + sys.meta_path)
        importlib.invalidate_caches()

    return _block


def test_require_omero_without_the_extra_raises_the_install_message(block_omero):
    block_omero()

    with pytest.raises(omero.OmeroExtraMissing) as excinfo:
        omero.require_omero()

    message = str(excinfo.value)
    assert 'pip install "spacr[omero]"' in message
    assert "omero" in message
    # It must be an ImportError (so `except ImportError` keeps working) but
    # NOT the bare ModuleNotFoundError the user would otherwise see.
    assert isinstance(excinfo.value, ImportError)
    assert not isinstance(excinfo.value, ModuleNotFoundError)
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_connect_without_the_extra_raises_the_same_message(block_omero):
    """The entry point a user actually calls, not just the helper."""
    block_omero()
    settings = omero.OmeroConnection(host="omero.example.org", username="u",
                                     password="s3cret")

    with pytest.raises(omero.OmeroExtraMissing) as excinfo:
        omero.connect(settings)

    assert 'pip install "spacr[omero]"' in str(excinfo.value)
    assert "s3cret" not in str(excinfo.value)


def test_a_missing_ice_is_reported_as_the_omero_extra(block_omero):
    """A half-built zeroc-ice is the most likely field failure, and says 'Ice'."""
    block_omero(ModuleNotFoundError("No module named 'Ice'", name="Ice"))

    with pytest.raises(omero.OmeroExtraMissing) as excinfo:
        omero.require_omero()

    message = str(excinfo.value)
    assert 'pip install "spacr[omero]"' in message
    assert "zeroc-ice" in message
    assert "Ice" in message


def test_an_unrelated_import_error_keeps_its_traceback(block_omero):
    """A genuine bug inside an installed module must not be mislabelled."""
    boom = ModuleNotFoundError("No module named 'nonexistent_dep'",
                               name="nonexistent_dep")
    block_omero(boom)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        omero.require_omero()

    assert excinfo.value is boom
    assert not isinstance(excinfo.value, omero.OmeroExtraMissing)


def test_have_omero_reports_false_without_raising(block_omero):
    block_omero()
    assert omero.have_omero() is False


@pytest.mark.parametrize(
    "exc, expected",
    [
        (ModuleNotFoundError("x", name="omero"), "omero"),
        (ModuleNotFoundError("x", name="omero.gateway"), "omero"),
        (ModuleNotFoundError("x", name="Ice"), "Ice"),
        (ModuleNotFoundError("x", name="IcePy"), "IcePy"),
        (ModuleNotFoundError("x", name="Glacier2"), "Glacier2"),
        (ModuleNotFoundError("x", name="omero_version"), "omero_version"),
        (ImportError("libIce.so.3.6: cannot open shared object file"), None),
        (ImportError("cannot import name 'BlitzGateway' from 'omero'"), "omero"),
        (ImportError("something else entirely"), None),
        (ImportError("boom"), None),
    ],
)
def test_missing_omero_extra_classifies_the_failure(exc, expected):
    """Mirrors tests/qt/test_qt_launch_without_pyside6.py for the same helper."""
    assert omero._missing_omero_extra(exc) == expected


def test_missing_omero_extra_ignores_a_none_name():
    exc = ImportError("nothing recognisable")
    assert exc.name is None
    assert omero._missing_omero_extra(exc) is None


def test_the_message_names_the_module_and_the_command():
    text = omero.missing_omero_message("omero")
    assert "missing module: omero" in text
    assert 'python -m pip install "spacr[omero]"' in text
    assert "spacr[all]" in text          # says why it is not in `all`
    # The Ice paragraph is for Ice failures only; an ordinary missing omero-py
    # must not be diagnosed as a broken C++ build.
    assert "Ice development headers" not in text
    ice_text = omero.missing_omero_message("Ice")
    assert "Ice development headers" in ice_text
    assert "comes from zeroc-ice, not from omero-py" in ice_text


# --- the module name, which is the trap ------------------------------------

def test_spacr_omero_does_not_shadow_the_third_party_package(tmp_path, monkeypatch):
    """`spacr/omero.py` importing "omero" must get omero-py, not itself.

    Absolute imports have been the default since Python 3, so this cannot
    happen through normal resolution — but a self-import here would be silent
    and would look exactly like a broken OMERO install, so it is checked with
    a decoy package rather than argued about.
    """
    decoy = tmp_path / "omero"
    decoy.mkdir()
    (decoy / "__init__.py").write_text("MARKER = 'third-party'\n", encoding="utf-8")
    (decoy / "gateway.py").write_text(
        "from omero import MARKER\n\n\nclass BlitzGateway:\n    pass\n",
        encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    saved = {name: sys.modules.pop(name) for name in list(sys.modules)
             if name == "omero" or name.startswith("omero.")}
    importlib.invalidate_caches()
    try:
        gateway = omero.require_omero()
        assert omero.have_omero() is True
        assert gateway.MARKER == "third-party"
        assert Path(gateway.__file__).resolve() == (decoy / "gateway.py").resolve()
        assert Path(gateway.__file__).resolve() != Path(omero.__file__).resolve()
        assert Path(sys.modules["omero"].__file__).resolve() != \
            Path(omero.__file__).resolve()
    finally:
        for name in list(sys.modules):
            if name == "omero" or name.startswith("omero."):
                del sys.modules[name]
        sys.modules.update(saved)
        importlib.invalidate_caches()


def test_omero_py_is_reached_through_a_string_literal_and_must_not_be_removed():
    """``omero-py`` has no import *statement*, and that is deliberate.

    ``tests/test_declared_dependencies_match_imports.py`` reads import
    statements out of the AST, so it cannot see this dependency — the same
    blind spot ``umap`` sits in. This pins the fact from the other side: if a
    future census reads "omero-py: unused", the answer is here rather than in
    a deleted extra.

    If a literal ``import omero`` is ever added to ``spacr/omero.py``, delete
    this test *and* add ``"omero": "omero-py"`` to ``IMPORT_TO_DIST`` in that
    census — without it, the census reports an undeclared ``omero``
    distribution that does not exist on PyPI under that name.
    """
    source = Path(omero.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    literal = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            literal += [a.name for a in node.names if a.name.split(".")[0] == "omero"]
        elif isinstance(node, ast.ImportFrom):
            if not node.level and node.module and node.module.split(".")[0] == "omero":
                literal.append(node.module)
    assert not literal, (
        f"spacr/omero.py now has literal omero import(s) {literal}. That is "
        f"allowed, but the dependency census needs "
        f'\'"omero": "omero-py"\' in IMPORT_TO_DIST '
        f"(tests/test_declared_dependencies_match_imports.py) or it will "
        f"report an undeclared 'omero' distribution.")

    assert omero.OMERO_GATEWAY_MODULE == "omero.gateway"
    assert 'OMERO_GATEWAY_MODULE = "omero.gateway"' in source

    setup_py = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    assert "'omero': ['omero-py" in setup_py, (
        "the `omero` extra left setup.py while spacr/omero.py still loads "
        "omero-py by name. Do not leave it ambiguous.")


# ===========================================================================
# 2. The well mapping, in both directions
# ===========================================================================

@pytest.mark.parametrize(
    "row, column, well, row_id, column_id",
    [
        (0, 0, "A01", "r1", "c1"),
        (0, 11, "A12", "r1", "c12"),
        (1, 5, "B06", "r2", "c6"),
        (7, 11, "H12", "r8", "c12"),          # the last well of a 96 plate
        (15, 23, "P24", "r16", "c24"),        # the last well of a 384 plate
        (25, 0, "Z01", "r26", "c1"),          # row 25 is Z
        (26, 0, "AA01", "r27", "c1"),         # row 26 is AA, not '[' and not IndexError
        (31, 47, "AF48", "r32", "c48"),       # the last well of a 1536 plate
    ],
)
def test_omero_row_column_becomes_the_spacr_well(row, column, well, row_id, column_id):
    position = omero.well_position(row, column)
    assert position.well == well
    assert position.row_id == row_id
    assert position.column_id == column_id
    assert position.row_index == row + 1
    assert position.column_index == column + 1


def test_row_26_is_AA_which_is_what_a_1536_plate_needs():
    """Stated rather than left to whatever ``chr()`` happens to do.

    ``chr(65 + 26)`` is ``'['`` and ``string.ascii_uppercase[26]`` is an
    IndexError; spaCR's answer is bijective base 26, so the 27th row is AA and
    a 1536-well plate's rows AA..AF come out right.
    """
    assert omero.well_position(26, 0).well == "AA01"
    assert omero.well_position(25, 0).well == "Z01"
    assert chr(65 + 26) == "["                     # what it must NOT be
    assert omero.well_position(31, 47).well == "AF48"
    assert omero.well_position(31, 47).plate_format == 1536


@pytest.mark.parametrize("well", ["A01", "H12", "P24", "Z01", "AA01", "AF48"])
def test_the_well_mapping_round_trips(well):
    row, column = omero.omero_indices(well)
    assert omero.well_position(row, column).well == well


def test_the_round_trip_agrees_with_schema():
    """The mapping is spaCR's own, not a second opinion about well names."""
    for row in range(0, 33):
        for column in (0, 11, 23, 47):
            position = omero.well_position(row, column)
            assert position.well == schema.well_id(row + 1, column + 1)
            assert (position.row_id, position.column_id) == \
                schema.parse_well(position.well)


@pytest.mark.parametrize("row, column", [(-1, 0), (0, -1), (None, 0), (0, None),
                                         ("A", 0), (1.5, 0), (True, 0)])
def test_an_unusable_well_position_is_refused(row, column):
    """An unplaced well must not silently become A01 on top of a real one."""
    with pytest.raises(omero.OmeroWellError):
        omero.well_position(row, column)


def test_omero_indices_refuses_a_non_well():
    with pytest.raises(omero.OmeroWellError):
        omero.omero_indices("not a well")


def test_prc_is_the_key_spacr_groups_by():
    assert omero.well_position(1, 5).prc("plate1") == "plate1_r2_c6"


# ===========================================================================
# 3. Filenames
# ===========================================================================

def test_the_filename_is_the_one_spacr_already_parses():
    name = omero.plane_filename("plate1", "A01", 1, 1)
    assert name == "plate1_A01_T0001F001L01A01Z01C01.tif"
    assert name == convert.target_name("plate1", "A01", 1, 1)


@pytest.mark.parametrize(
    "well, field_id, channel, z, t",
    [("A01", 1, 1, 1, 1), ("H12", 9, 2, 1, 1), ("P24", 1, 1, 1, 1),
     ("AF48", 12, 4, 9, 3)],
)
def test_the_filename_is_parsed_by_spacrs_own_ingestion_regex(
        well, field_id, channel, z, t):
    """The regex Mask/Measure actually applies to the folder it is given."""
    import re

    from spacr.utils import _get_regex

    pattern = re.compile(_get_regex("cellvoyager", "tif"))
    name = omero.plane_filename("plate1", well, field_id, channel, z=z, t=t)

    match = pattern.match(name)
    assert match, name
    assert match.group("plateID") == "plate1"
    assert match.group("wellID") == well
    assert int(match.group("fieldID")) == field_id
    assert int(match.group("chanID")) == channel
    assert int(match.group("sliceID")) == z
    assert int(match.group("timeID")) == t


def test_the_filename_also_satisfies_the_strict_gui_regex():
    """The stricter detector in the Qt GUI, for the plate formats it covers.

    NOTE: ``spacr/qt/regex_detect.py::YOKOGAWA`` (duplicated at
    ``spacr/pipeline_v2.py``) matches the well as ``[A-Z]\\d{2}`` — one letter
    — so it cannot recognise a 1536-well plate's ``AA01``..``AF48``, which
    ``spacr.convert`` and ``spacr.schema`` both support and
    ``tests/test_convert_1536.py`` writes. That is a limitation of the
    detector, not of the names produced here, so this test covers the formats
    the detector claims and the test above covers the ingestion regex that
    decides whether the import is readable at all.
    """
    import re

    from spacr.qt import regex_detect

    pattern = re.compile(regex_detect.YOKOGAWA)
    for well in ("A01", "H12", "P24"):
        name = omero.plane_filename("plate1", well, 1, 1)
        match = pattern.match(name)
        assert match, name
        assert match.group("wellID") == well


def test_plate_token_agrees_with_the_converter():
    """One rule for plate tokens in spaCR, not two that drift apart."""
    for name in ["Assay plate 3", "my_run", "plate1", "  ", "-", "%%%", "µ-plate"]:
        assert omero.plate_token(name) == convert._sanitise(name)
    assert "_" not in omero.plate_token("my_run")
    assert omero.plate_token("") == "plate"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("A01 site 1", "A01"),
        ("exp_B06_field2.tif", "B06"),
        ("plate1_AA01_T0001", "AA01"),
        ("no well here", None),
        ("", None),
        (None, None),
        ("ZZ99", None),                # row 702: past any real plate
    ],
)
def test_a_well_can_be_recovered_from_an_image_name(name, expected):
    assert omero.well_from_image_name(name) == expected


# ===========================================================================
# 4. Pixel size carries its unit
# ===========================================================================

def test_a_length_object_keeps_its_unit():
    size = omero.pixel_size_from(FakeLength(120.0, "NANOMETER"))
    assert size.value == 120.0
    assert size.unit == "NANOMETER"
    assert "NANOMETER" in size.describe()


def test_a_bare_float_records_the_unit_omero_py_assumed():
    size = omero.pixel_size_from(0.325)
    assert size.value == 0.325
    assert size.unit == "MICROMETER"


def test_an_uncalibrated_image_has_no_pixel_size():
    size = omero.pixel_size_from(None)
    assert size.value is None
    assert size.unit is None
    assert not size
    assert size.describe() == "unknown"


def test_a_length_like_object_with_plain_attributes_still_reads():
    class Plain:
        value = 2.0
        unit = "MILLIMETER"

    size = omero.pixel_size_from(Plain())
    assert (size.value, size.unit) == (2.0, "MILLIMETER")


# ===========================================================================
# 5. Ids
# ===========================================================================

@pytest.mark.parametrize(
    "value, kind, object_id",
    [
        (123, None, 123),
        ("123", None, 123),
        ("  123 ", None, 123),
        ("Dataset:12", "Dataset", 12),
        ("dataset-12", "Dataset", 12),
        ("PLATE:7", "Plate", 7),
        ("https://omero.example.org/webclient/?show=plate-42", "Plate", 42),
        ("https://omero.example.org/webclient/?show=dataset-9", "Dataset", 9),
    ],
)
def test_object_references_parse(value, kind, object_id):
    ref = omero.parse_object_ref(value)
    assert ref.kind == kind
    assert ref.object_id == object_id


@pytest.mark.parametrize("value", [0, -1, "-5", "Dataset:0", "Dataset:-3"])
def test_a_non_positive_id_is_refused(value):
    with pytest.raises(omero.OmeroIdError, match="positive"):
        omero.parse_object_ref(value)


@pytest.mark.parametrize("value", [None, "", "   ", "abc", 1.5, True, "Widget:3",
                                   "Dataset:", ":12"])
def test_an_unusable_id_is_refused(value):
    with pytest.raises(omero.OmeroIdError):
        omero.parse_object_ref(value)


def test_a_plate_reference_is_refused_by_the_dataset_importer():
    with pytest.raises(omero.OmeroIdError, match="Plate.*Dataset"):
        omero.parse_object_id("Plate:7", expect="Dataset")


def test_a_bare_id_is_taken_at_the_callers_word():
    assert omero.parse_object_id(7, expect="Dataset") == 7


def test_an_omero_ref_passes_through_unchanged():
    ref = omero.parse_object_ref("Plate:7")
    assert omero.parse_object_ref(ref) is ref


# ===========================================================================
# 6. Connection settings and the password
# ===========================================================================

def test_settings_come_from_the_environment_when_not_passed():
    env = {"OMERO_HOST": "omero.example.org", "OMERO_PORT": "14064",
           "OMERO_USER": "jdoe", "OMERO_PASSWORD": "hunter2",
           "OMERO_GROUP": "lab", "OMERO_SECURE": "false"}
    settings = omero.connection_settings(env=env)
    assert settings.host == "omero.example.org"
    assert settings.port == 14064
    assert settings.username == "jdoe"
    assert settings.password == "hunter2"
    assert settings.group == "lab"
    assert settings.secure is False
    assert settings.auth_mode == "password"


def test_arguments_beat_the_environment():
    env = {"OMERO_HOST": "from-env", "OMERO_PASSWORD": "p"}
    settings = omero.connection_settings("from-arg", username="u", env=env)
    assert settings.host == "from-arg"


def test_omero_pass_is_accepted_as_well_as_omero_password():
    env = {"OMERO_HOST": "h", "OMERO_USER": "u", "OMERO_PASS": "cli-spelling"}
    assert omero.connection_settings(env=env).password == "cli-spelling"


def test_a_session_key_is_a_credential_without_a_user_name():
    settings = omero.connection_settings(
        env={"OMERO_HOST": "h", "OMERO_SESSION_KEY": "uuid-1234"})
    assert settings.session_key == "uuid-1234"
    assert settings.username is None
    assert settings.auth_mode == "session"


def test_a_missing_host_is_refused_and_names_the_variable():
    with pytest.raises(omero.OmeroConnectionError, match="OMERO_HOST"):
        omero.connection_settings(env={"OMERO_PASSWORD": "p"})


def test_a_missing_credential_is_refused_and_names_the_variable():
    with pytest.raises(omero.OmeroConnectionError, match="OMERO_PASSWORD"):
        omero.connection_settings(env={"OMERO_HOST": "h"})


def test_a_password_without_a_user_name_is_refused():
    with pytest.raises(omero.OmeroConnectionError, match="user name"):
        omero.connection_settings(env={"OMERO_HOST": "h", "OMERO_PASSWORD": "p"})


@pytest.mark.parametrize("port", ["0", "70000", "-1", "not-a-port"])
def test_an_impossible_port_is_refused(port):
    with pytest.raises(omero.OmeroConnectionError):
        omero.connection_settings(
            env={"OMERO_HOST": "h", "OMERO_USER": "u", "OMERO_PASSWORD": "p",
                 "OMERO_PORT": port})


def test_an_unreadable_secure_flag_is_refused():
    with pytest.raises(omero.OmeroConnectionError, match="yes/no"):
        omero.connection_settings(
            env={"OMERO_HOST": "h", "OMERO_USER": "u", "OMERO_PASSWORD": "p",
                 "OMERO_SECURE": "maybe"})


PASSWORD = "correct-horse-battery-staple"
SESSION = "session-uuid-do-not-print"


def _settings_with_secrets():
    return omero.connection_settings(
        env={"OMERO_HOST": "omero.example.org", "OMERO_USER": "jdoe",
             "OMERO_PASSWORD": PASSWORD, "OMERO_SESSION_KEY": SESSION})


def test_the_password_never_appears_in_a_repr_or_a_str():
    settings = _settings_with_secrets()
    for text in (repr(settings), str(settings), f"{settings}", format(settings),
                 settings.describe(), json.dumps(settings.redacted())):
        assert PASSWORD not in text
        assert SESSION not in text
    # ...and the reader can still see that a credential is set.
    assert omero.SECRET_PLACEHOLDER in repr(settings)
    assert settings.redacted()["password"] == omero.SECRET_PLACEHOLDER


def test_the_password_never_appears_in_a_log_record(caplog):
    caplog.set_level(logging.DEBUG, logger="spacr.omero")
    settings = _settings_with_secrets()
    gateway = FakeGateway()

    assert omero.connect(settings, gateway_factory=lambda s: gateway) is gateway

    assert caplog.records, "connect() should say what it connected to"
    for record in caplog.records:
        blob = " ".join([record.getMessage(), str(record.args), str(record.msg)])
        assert PASSWORD not in blob
        assert SESSION not in blob
    assert "omero.example.org" in caplog.text          # the useful half is there


def test_a_refused_login_raises_without_naming_the_credential():
    settings = _settings_with_secrets()
    gateway = FakeGateway()
    gateway.connect_result = False

    with pytest.raises(omero.OmeroConnectionError) as excinfo:
        omero.connect(settings, gateway_factory=lambda s: gateway)

    assert PASSWORD not in str(excinfo.value)
    assert "omero.example.org" in str(excinfo.value)


def test_connect_builds_settings_from_keywords_when_none_are_given():
    gateway = FakeGateway()
    result = omero.connect(gateway_factory=lambda s: gateway, host="h",
                           username="u", password="p",
                           env={"OMERO_HOST": "ignored"})
    assert result is gateway


# ===========================================================================
# 7. Inspecting a container without downloading it
# ===========================================================================

def test_inspecting_a_plate_never_fetches_a_pixel():
    plate = build_plate()
    gateway = build_gateway(plate=plate)

    listing = omero.inspect_container(gateway, "Plate:4711")

    assert listing.kind == "Plate"
    assert listing.name == "Assay plate 3"
    assert listing.n_images == 3
    assert listing.wells == ("A01", "B06")
    assert listing.n_planes == 6                 # 3 images x 2 channels
    assert listing.channels == ("ch1", "ch2")
    for well in plate.wells:
        for image in well.images:
            assert image.plane_calls == [], "inspect_container fetched pixels"
    assert "would write 6 TIFF(s)" in listing.describe()


def test_inspecting_a_dataset_lists_its_images():
    dataset = FakeDataset(77, "bag of images",
                          [FakeImage(21, "one"), FakeImage(22, "two")])
    gateway = build_gateway(dataset=dataset)

    listing = omero.inspect_container(gateway, 77, kind="Dataset")

    assert listing.n_images == 2
    assert listing.wells == ()
    assert listing.images[0].pixel_size_x.unit == "MICROMETER"


def test_an_unplaced_well_is_counted_not_imported():
    plate = FakePlate(1, "p", [FakeWell(None, None, [FakeImage(31, "x")]),
                               FakeWell(0, 0, [FakeImage(32, "y")])])
    listing = omero.inspect_container(build_gateway(plate=plate), "Plate:1")
    assert listing.unplaced_wells == 1
    assert listing.wells == ("A01",)
    assert "unplaced" in listing.describe()


def test_a_bare_id_with_no_kind_is_refused():
    with pytest.raises(omero.OmeroIdError, match="bare id"):
        omero.inspect_container(FakeGateway(), 4711)


def test_a_contradicted_kind_is_refused():
    with pytest.raises(omero.OmeroIdError, match="Say it once"):
        omero.inspect_container(FakeGateway(), "Plate:1", kind="Dataset")


def test_an_unsupported_kind_is_refused():
    with pytest.raises(omero.OmeroIdError, match="neither"):
        omero.inspect_container(FakeGateway(), "Image:1")


def test_an_id_that_resolves_to_nothing_is_refused():
    with pytest.raises(omero.OmeroContainerError, match="does not exist"):
        omero.inspect_container(FakeGateway(), "Plate:999")


# ===========================================================================
# 8. The import, end to end, against the fake gateway
# ===========================================================================

#: Hand-written, not derived from the code under test. Well A01 is OMERO
#: (row 0, column 0) with two fields; well B06 is (row 1, column 5) with one.
EXPECTED_PLATE_FILES = [
    "plate1_A01_T0001F001L01A01Z01C01.tif",
    "plate1_A01_T0001F001L01A01Z01C02.tif",
    "plate1_A01_T0001F002L01A01Z01C01.tif",
    "plate1_A01_T0001F002L01A01Z01C02.tif",
    "plate1_B06_T0001F001L01A01Z01C01.tif",
    "plate1_B06_T0001F001L01A01Z01C02.tif",
]


def test_importing_a_plate_writes_the_expected_tiffs(tmp_path):
    plate = build_plate()
    gateway = build_gateway(plate=plate)

    result = omero.import_plate(gateway, "Plate:4711", tmp_path, plate="plate1")

    on_disk = sorted(p.name for p in tmp_path.glob("*.tif"))
    assert on_disk == EXPECTED_PLATE_FILES
    assert sorted(result.written) == EXPECTED_PLATE_FILES
    assert result.n_images == 3
    assert result.dry_run is False
    assert result.limited is False
    assert "wrote" in result.describe()


def test_the_pixels_land_in_the_right_file(tmp_path):
    """The well/field/channel mapping is checked through the actual bytes."""
    plate = build_plate()
    omero.import_plate(build_gateway(plate=plate), "Plate:4711", tmp_path,
                       plate="plate1")

    # image 11 is A01 field 1, image 12 is A01 field 2, image 13 is B06 field 1
    for filename, image_id, channel in [
        ("plate1_A01_T0001F001L01A01Z01C01.tif", 11, 1),
        ("plate1_A01_T0001F001L01A01Z01C02.tif", 11, 2),
        ("plate1_A01_T0001F002L01A01Z01C01.tif", 12, 1),
        ("plate1_B06_T0001F001L01A01Z01C02.tif", 13, 2),
    ]:
        written = tifffile.imread(str(tmp_path / filename))
        # getPlane is 0-based in z/c/t; the filename is 1-based.
        expected = plane_array(image_id, 0, channel - 1, 0)
        assert np.array_equal(written, expected), filename


def test_a_multi_z_multi_t_image_writes_every_plane(tmp_path):
    image = FakeImage(41, "stack", size_c=2, size_z=3, size_t=2)
    plate = FakePlate(5, "p", [FakeWell(0, 0, [image])])

    result = omero.import_plate(build_gateway(plate=plate), "Plate:5", tmp_path,
                               plate="plate1")

    assert len(result.written) == 12                     # 2c x 3z x 2t
    assert "plate1_A01_T0002F001L01A01Z03C02.tif" in result.written
    written = tifffile.imread(
        str(tmp_path / "plate1_A01_T0002F001L01A01Z03C02.tif"))
    assert np.array_equal(written, plane_array(41, 2, 1, 1))


def test_the_sidecars_record_the_keys_the_units_and_no_secret(tmp_path):
    plate = build_plate()
    settings = _settings_with_secrets()

    omero.import_plate(build_gateway(plate=plate), "Plate:4711", tmp_path,
                       plate="plate1", settings=settings)

    csv_text = (tmp_path / omero.SIDECAR_CSV).read_text(encoding="utf-8")
    assert "plate1_B06_T0001F001L01A01Z01C02.tif" in csv_text
    assert "plate1_r2_c6" in csv_text                    # the prc key
    assert "MICROMETER" in csv_text
    header = csv_text.splitlines()[0].split(",")
    assert header == list(omero.MAP_CSV_COLUMNS)

    payload = json.loads((tmp_path / omero.SIDECAR_JSON).read_text(encoding="utf-8"))
    assert payload["container"] == {"kind": "Plate", "id": 4711,
                                    "name": "Assay plate 3"}
    assert payload["plate"] == "plate1"
    assert payload["n_planes"] == 6
    assert payload["pixel_size"]["unit"] == "MICROMETER"
    assert payload["server"]["password"] == omero.SECRET_PLACEHOLDER

    for text in (csv_text, json.dumps(payload)):
        assert PASSWORD not in text
        assert SESSION not in text


def test_a_dry_run_plans_everything_and_fetches_nothing(tmp_path):
    plate = build_plate()
    gateway = build_gateway(plate=plate)

    result = omero.import_plate(gateway, "Plate:4711", tmp_path, plate="plate1",
                               dry_run=True)

    assert [p.filename for p in result.planned] == EXPECTED_PLATE_FILES
    assert result.written == ()
    assert list(tmp_path.glob("*.tif")) == []
    for well in plate.wells:
        for image in well.images:
            assert image.plane_calls == []
    assert (tmp_path / omero.SIDECAR_CSV).exists()       # the plan is still written
    assert "would write" in result.describe()


def test_limit_stops_the_walk_early(tmp_path):
    plate = build_plate()
    result = omero.import_plate(build_gateway(plate=plate), "Plate:4711",
                                tmp_path, plate="plate1", limit=1)
    assert result.n_images == 1
    assert result.limited is True
    assert sorted(result.written) == EXPECTED_PLATE_FILES[:2]


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_a_limit_that_imports_nothing_is_refused(tmp_path, limit):
    """limit=0 would produce an empty folder and a successful-looking return."""
    with pytest.raises(omero.OmeroError, match="limit"):
        omero.import_plate(build_gateway(plate=build_plate()), "Plate:4711",
                           tmp_path, limit=limit)

    dataset = FakeDataset(77, "bag", [FakeImage(21, "A01 s1", size_c=1)])
    with pytest.raises(omero.OmeroError, match="limit"):
        omero.import_dataset(build_gateway(dataset=dataset), "Dataset:77",
                             tmp_path, limit=limit)


def test_a_rerun_resumes_instead_of_redownloading(tmp_path):
    plate = build_plate()
    gateway = build_gateway(plate=plate)
    omero.import_plate(gateway, "Plate:4711", tmp_path, plate="plate1")
    calls_after_first = sum(len(i.plane_calls) for w in plate.wells for i in w.images)

    again = omero.import_plate(gateway, "Plate:4711", tmp_path, plate="plate1")

    assert again.written == ()
    assert sorted(again.skipped) == EXPECTED_PLATE_FILES
    assert sum(len(i.plane_calls) for w in plate.wells for i in w.images) == \
        calls_after_first
    assert "skipped" in again.describe()

    forced = omero.import_plate(gateway, "Plate:4711", tmp_path, plate="plate1",
                                overwrite=True)
    assert sorted(forced.written) == EXPECTED_PLATE_FILES


def test_the_plate_token_defaults_to_the_omero_plate_name(tmp_path):
    result = omero.import_plate(build_gateway(plate=build_plate()), "Plate:4711",
                                tmp_path, dry_run=True)
    assert result.plate == "Assay-plate-3"
    assert result.planned[0].filename.startswith("Assay-plate-3_A01_")


def test_importing_a_dataset_assigns_wells_and_says_how(tmp_path):
    dataset = FakeDataset(
        77, "bag",
        [FakeImage(21, "A01 site 1", size_c=1), FakeImage(22, "mystery", size_c=1)])

    result = omero.import_dataset(build_gateway(dataset=dataset), "Dataset:77",
                                  tmp_path, plate="plate1")

    sources = {p.filename: p.well_source for p in result.planned}
    assert sources == {"plate1_A01_T0001F001L01A01Z01C01.tif": "name",
                       "plate1_A01_T0001F002L01A01Z01C01.tif": "sequence"}
    assert sorted(p.name for p in tmp_path.glob("*.tif")) == sorted(sources)


def test_well_from_name_can_be_turned_off(tmp_path):
    dataset = FakeDataset(77, "bag", [FakeImage(21, "A01 site 1", size_c=1),
                                      FakeImage(22, "B02 site 1", size_c=1)])
    result = omero.import_dataset(build_gateway(dataset=dataset), 77, tmp_path,
                                  plate="plate1", well_from_name=False,
                                  dry_run=True)
    assert [p.well for p in result.planned] == ["A01", "A02"]
    assert {p.well_source for p in result.planned} == {"sequence"}


def test_a_plate_id_handed_to_the_dataset_importer_is_refused(tmp_path):
    """Both are integers on the same server; this is an easy and silent mistake."""
    with pytest.raises(omero.OmeroIdError, match="Plate"):
        omero.import_dataset(build_gateway(plate=build_plate()), "Plate:4711",
                             tmp_path)


def test_a_dataset_id_handed_to_the_plate_importer_is_refused(tmp_path):
    with pytest.raises(omero.OmeroIdError, match="Dataset"):
        omero.import_plate(FakeGateway(), "Dataset:77", tmp_path)


def test_importing_an_id_that_resolves_to_nothing_is_refused(tmp_path):
    with pytest.raises(omero.OmeroContainerError, match="does not exist"):
        omero.import_plate(FakeGateway(), "Plate:999", tmp_path)


def test_an_empty_container_is_refused_rather_than_silently_succeeding(tmp_path):
    empty = FakePlate(3, "empty", [FakeWell(None, None, [FakeImage(1, "x")])])
    with pytest.raises(omero.OmeroContainerError, match="nothing to import"):
        omero.import_plate(build_gateway(plate=empty), "Plate:3", tmp_path)

    dataset = FakeDataset(78, "empty", [])
    with pytest.raises(omero.OmeroContainerError, match="no images"):
        omero.import_dataset(build_gateway(dataset=dataset), "Dataset:78", tmp_path)


def test_an_image_still_importing_on_the_server_is_reported(tmp_path):
    plate = FakePlate(9, "p", [FakeWell(0, 0, [FakeImage(51, "x", has_pixels=False)])])
    with pytest.raises(omero.OmeroContainerError, match="no pixels"):
        omero.import_plate(build_gateway(plate=plate), "Plate:9", tmp_path)


def test_import_container_dispatches_on_the_reference(tmp_path):
    plate = build_plate()
    dataset = FakeDataset(77, "bag", [FakeImage(21, "A01 s1", size_c=1)])
    gateway = build_gateway(plate=plate, dataset=dataset)

    assert omero.import_container(gateway, "Plate:4711", tmp_path / "a",
                                  dry_run=True).kind == "Plate"
    assert omero.import_container(gateway, 77, tmp_path / "b", kind="Dataset",
                                  dry_run=True).kind == "Dataset"
    with pytest.raises(omero.OmeroIdError, match="bare id"):
        omero.import_container(gateway, 77, tmp_path / "c")
    with pytest.raises(omero.OmeroIdError, match="not supported"):
        omero.import_container(gateway, "Image:1", tmp_path / "d")
    with pytest.raises(omero.OmeroIdError, match="Say it once"):
        omero.import_container(gateway, "Plate:4711", tmp_path / "e", kind="Dataset")


# ===========================================================================
# 9. Measurements -> key/value pairs
# ===========================================================================

def test_every_value_is_a_string_because_omero_has_no_numeric_map_value():
    pairs = omero.measurement_pairs(
        {"plateID": "plate1", "cell_area": 1234.5678, "object_label": 7,
         "is_hit": True})
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in pairs)
    assert dict(pairs)["cell_area"] == "1234.57"          # six significant digits
    assert dict(pairs)["object_label"] == "7"
    assert dict(pairs)["is_hit"] == "True"


@pytest.mark.parametrize(
    "value, expected",
    [
        (3.0, "3"),
        (1.23456789e-5, "1.23457e-05"),
        (1234567.0, "1.23457e+06"),
        (0.1 + 0.2, "0.3"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
        (float("nan"), "NaN"),
        (None, "NaN"),
        (12345678901234, "12345678901234"),               # ints never go through %g
        (True, "True"),
        (False, "False"),
        ("  text\nwith  breaks ", "text with breaks"),
    ],
)
def test_the_float_and_missing_formatting_is_what_it_says_it_is(value, expected):
    assert omero.format_map_value(value) == expected


def test_numpy_scalars_render_like_their_builtin_equivalents():
    assert omero.format_map_value(np.float64(3.0)) == "3"
    assert omero.format_map_value(np.int64(7)) == "7"
    assert omero.format_map_value(np.float32("nan")) == "NaN"
    assert omero.format_map_value(np.bool_(True)) == "True"


def test_a_very_long_value_is_truncated():
    text = omero.format_map_value("x" * 1000)
    assert len(text) == omero.MAX_VALUE_CHARS
    assert text.endswith("…")


def test_nan_is_kept_by_default_and_dropped_on_request():
    row = {"a": 1.0, "b": float("nan"), "c": None}
    assert dict(omero.measurement_pairs(row))["b"] == omero.MISSING_TEXT
    assert dict(omero.measurement_pairs(row))["c"] == omero.MISSING_TEXT
    dropped = dict(omero.measurement_pairs(row, nan_policy=omero.NAN_DROP))
    assert dropped == {"a": "1"}


def test_a_400_column_table_does_not_become_a_400_entry_annotation():
    """The panel is 300 px tall; a 400-row table in it is not an annotation."""
    row = {f"feature_{i}": float(i) for i in range(400)}
    row["plateID"] = "plate1"

    pairs = omero.measurement_pairs(row)

    assert len(pairs) == omero.MAX_MAP_PAIRS == 50
    assert pairs[0] == ("plateID", "plate1")              # identity first
    key, notice = pairs[-1]
    assert key == omero.TRUNCATION_KEY
    assert "of 401" in notice
    assert omero.NS_FILE in notice, "the notice must point at the full table"


def test_the_cap_is_configurable_and_still_leaves_room_for_the_notice():
    row = {f"f{i}": i for i in range(10)}
    pairs = omero.measurement_pairs(row, max_pairs=4)
    assert len(pairs) == 4
    assert pairs[-1][0] == omero.TRUNCATION_KEY

    exact = omero.measurement_pairs({f"f{i}": i for i in range(4)}, max_pairs=4)
    assert len(exact) == 4
    assert exact[-1][0] != omero.TRUNCATION_KEY           # nothing was dropped


def test_the_identity_columns_are_promoted_to_the_top():
    row = {"zzz": 1, "cell_area": 2, "columnID": "c1", "plateID": "p", "rowID": "r1"}
    keys = [k for k, _ in omero.measurement_pairs(row)]
    assert keys[:3] == ["plateID", "rowID", "columnID"]


def test_columns_and_extra_are_honoured():
    row = {"a": 1, "b": 2, "c": 3}
    pairs = omero.measurement_pairs(row, columns=["c", "a"],
                                    extra={"spacr_version": "1.3.6"})
    assert dict(pairs) == {"c": "3", "a": "1"}
    assert "spacr_version" not in dict(pairs)             # not in `columns`
    assert dict(omero.measurement_pairs(row, extra={"a": 99}))["a"] == "99"


@pytest.mark.parametrize("kwargs", [{"nan_policy": "invent"}, {"max_pairs": 1}])
def test_a_nonsense_projection_is_refused(kwargs):
    with pytest.raises(omero.OmeroError):
        omero.measurement_pairs({"a": 1}, **kwargs)


def test_a_pandas_row_is_just_a_mapping():
    """The documented way in: ``df.iloc[i].to_dict()``."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"plateID": ["plate1"], "cell_area": [12.5],
                          "pathogen_area": [float("nan")]})
    pairs = dict(omero.measurement_pairs(frame.iloc[0].to_dict()))
    assert pairs == {"plateID": "plate1", "cell_area": "12.5",
                     "pathogen_area": "NaN"}


def test_a_well_summary_averages_the_objects_and_counts_them():
    rows = [{"plateID": "plate1", "cell_area": 10.0, "pathogen_area": float("nan")},
            {"plateID": "plate1", "cell_area": 20.0, "pathogen_area": 4.0}]

    summary = omero.summarise_rows(rows)

    assert summary["n_objects"] == 2
    assert summary["cell_area"] == 15.0
    # the NaN row is excluded from its own column's mean, not zero-filled
    assert summary["pathogen_area"] == 4.0
    assert summary["plateID"] == "plate1"


def test_a_column_the_objects_disagree_about_is_not_guessed():
    rows = [{"plateID": "plate1"}, {"plateID": "plate2"}]
    assert omero.summarise_rows(rows)["plateID"] is None


def test_a_column_with_nothing_usable_in_it_is_reported_as_missing():
    rows = [{"x": float("nan")}, {"x": None}]
    summary = omero.summarise_rows(rows)
    assert summary["x"] is None
    assert dict(omero.measurement_pairs(summary))["x"] == "NaN"


def test_well_summary_pairs_ties_the_two_together():
    rows = [{"cell_area": 10.0}, {"cell_area": 20.0}]
    assert dict(omero.well_summary_pairs(rows)) == {"n_objects": "2",
                                                    "cell_area": "15"}


def test_is_missing_covers_the_sentinels():
    assert omero.is_missing(None)
    assert omero.is_missing(float("nan"))
    assert omero.is_missing(np.float64("nan"))
    assert not omero.is_missing(0)
    assert not omero.is_missing("")
    pd = pytest.importorskip("pandas")
    assert omero.is_missing(pd.NA)
    assert omero.is_missing(pd.NaT)


# ===========================================================================
# 10. Replace vs append
# ===========================================================================

FOREIGN_NS = "openmicroscopy.org/omero/bulk_annotations"


def test_the_namespace_is_spacrs_own_and_matched_exactly():
    assert omero.NAMESPACE_ROOT == "github.com/EinarOlafsson/spacr"
    assert omero.NS_MEASUREMENTS.startswith(omero.NAMESPACE_ROOT + "/")
    assert omero.is_spacr_namespace(omero.NS_MEASUREMENTS)
    assert not omero.is_spacr_namespace(FOREIGN_NS)
    assert not omero.is_spacr_namespace(None)
    # a prefix match would claim a fork's namespace as spaCR's own
    assert not omero.is_spacr_namespace(omero.NS_MEASUREMENTS + "/extra")
    assert not omero.is_spacr_namespace(
        "github.com/EinarOlafsson/spacr-fork/1/measurements")


def test_plan_annotation_updates_its_own_and_ignores_everything_else():
    plan = omero.plan_annotation(
        [(5, FOREIGN_NS), (7, omero.NS_MEASUREMENTS), (9, None)],
        omero.NS_MEASUREMENTS)
    assert plan.action == omero.ACTION_UPDATE
    assert plan.annotation_id == 7
    assert plan.duplicates == ()


def test_plan_annotation_creates_when_there_is_nothing_of_its_own():
    plan = omero.plan_annotation([(5, FOREIGN_NS)], omero.NS_MEASUREMENTS)
    assert plan.action == omero.ACTION_CREATE
    assert plan.annotation_id is None


def test_plan_annotation_updates_the_oldest_and_reports_the_rest():
    plan = omero.plan_annotation(
        [(9, omero.NS_MEASUREMENTS), (4, omero.NS_MEASUREMENTS)],
        omero.NS_MEASUREMENTS)
    assert plan.annotation_id == 4
    assert plan.duplicates == (9,)
    assert "does not delete" in plan.reason


def test_append_never_looks_at_what_is_there():
    plan = omero.plan_annotation([(7, omero.NS_MEASUREMENTS)],
                                 omero.NS_MEASUREMENTS, mode=omero.APPEND)
    assert plan.action == omero.ACTION_CREATE


def test_an_unknown_mode_is_refused():
    with pytest.raises(omero.OmeroError, match="mode"):
        omero.plan_annotation([], omero.NS_MEASUREMENTS, mode="clobber")
    with pytest.raises(omero.OmeroError, match="mode"):
        omero.plan_tag([], omero.NS_TAG, "hit", mode="clobber")


def test_planning_outside_spacrs_own_namespace_is_refused():
    """"Replace" is only safe because it can only ever replace spaCR's own."""
    with pytest.raises(omero.OmeroError, match="not a spaCR namespace"):
        omero.plan_annotation([], FOREIGN_NS)
    with pytest.raises(omero.OmeroError, match="not a spaCR namespace"):
        omero.plan_tag([], FOREIGN_NS, "hit")


def test_exporting_twice_leaves_exactly_one_spacr_annotation():
    """The whole point of owning a namespace."""
    gateway = FakeGateway()
    image = FakeImage(11, "img")
    foreign = FakeAnnotation(gateway, "MapAnnotationWrapper", ns=FOREIGN_NS,
                             value=[("theirs", "do not touch")])
    foreign.save()
    image.linkAnnotation(foreign)

    first = omero.export_map_annotation(
        gateway, image, [("cell_area", "12.5")],
        annotation_factory=gateway.annotation_factory)
    second = omero.export_map_annotation(
        gateway, image, [("cell_area", "99.0")],
        annotation_factory=gateway.annotation_factory)

    assert first.action == omero.ACTION_CREATE
    assert second.action == omero.ACTION_UPDATE

    mine = [a for a in image.annotations if a.getNs() == omero.NS_MEASUREMENTS]
    assert len(mine) == 1
    assert mine[0].getValue() == [("cell_area", "99.0")]

    # ...and the neighbour's annotation is exactly as it was.
    assert foreign in image.annotations
    assert foreign.getValue() == [("theirs", "do not touch")]
    assert foreign.saves == 1                      # never re-saved by spaCR


def test_append_piles_up_on_purpose():
    gateway = FakeGateway()
    image = FakeImage(11, "img")
    for value in ("1", "2"):
        omero.export_map_annotation(
            gateway, image, [("run", value)], mode=omero.APPEND,
            annotation_factory=gateway.annotation_factory)
    mine = [a for a in image.annotations if a.getNs() == omero.NS_MEASUREMENTS]
    assert [a.getValue() for a in mine] == [[("run", "1")], [("run", "2")]]


def test_export_casts_values_to_strings_as_a_last_defence():
    gateway = FakeGateway()
    image = FakeImage(11, "img")
    omero.export_map_annotation(gateway, image, [("n", 3)],
                                annotation_factory=gateway.annotation_factory)
    assert image.annotations[0].getValue() == [("n", "3")]


def test_list_spacr_annotations_reports_only_spacrs_own():
    gateway = FakeGateway()
    image = FakeImage(11, "img")
    for ns in (omero.NS_MEASUREMENTS, FOREIGN_NS, None, omero.NS_TAG):
        annotation = FakeAnnotation(gateway, "MapAnnotationWrapper", ns=ns)
        annotation.save()
        image.linkAnnotation(annotation)

    found = omero.list_spacr_annotations(image)

    assert [ns for _, ns in found] == [omero.NS_MEASUREMENTS, omero.NS_TAG]


def test_a_foreign_namespace_cannot_be_written_to():
    with pytest.raises(omero.OmeroError, match="not a spaCR namespace"):
        omero.export_map_annotation(FakeGateway(), FakeImage(11, "i"),
                                    [("a", "b")], namespace=FOREIGN_NS)


# --- tags -------------------------------------------------------------------

def test_a_repeated_verdict_is_idempotent():
    gateway = FakeGateway()
    image = FakeImage(11, "img")

    first = omero.export_tag_annotation(
        gateway, image, "hit", annotation_factory=gateway.annotation_factory)
    second = omero.export_tag_annotation(
        gateway, image, "hit", annotation_factory=gateway.annotation_factory)

    assert first.action == omero.ACTION_CREATE
    assert second.action == omero.ACTION_UNCHANGED
    assert len([a for a in image.annotations if a.getNs() == omero.NS_TAG]) == 1


def test_a_changed_verdict_never_renames_the_shared_tag():
    """Renaming a tag would relabel every other object carrying it."""
    gateway = FakeGateway()
    image = FakeImage(11, "img")
    omero.export_tag_annotation(gateway, image, "hit",
                                annotation_factory=gateway.annotation_factory)
    old = image.annotations[0]

    result = omero.export_tag_annotation(
        gateway, image, "not hit", annotation_factory=gateway.annotation_factory)

    assert result.action == omero.ACTION_CREATE
    assert old.getValue() == "hit"                 # untouched
    assert old.saves == 1
    assert result.duplicates == (old.getId(),)
    assert "shared object" in result.reason


def test_an_empty_verdict_is_refused():
    with pytest.raises(omero.OmeroError, match="empty verdict"):
        omero.export_tag_annotation(FakeGateway(), FakeImage(11, "i"), "   ")


def test_plan_tag_appends_unconditionally_in_append_mode():
    plan = omero.plan_tag([(3, omero.NS_TAG, "hit")], omero.NS_TAG, "hit",
                          mode=omero.APPEND)
    assert plan.action == omero.ACTION_CREATE


# --- files ------------------------------------------------------------------

def test_a_results_file_is_attached_with_the_spacr_namespace(tmp_path):
    gateway = FakeGateway()
    dataset = FakeDataset(77, "bag", [])
    results = tmp_path / "results.csv"
    results.write_text("a,b\n1,2\n", encoding="utf-8")

    result = omero.export_file_annotation(gateway, dataset, results)

    assert result.action == omero.ACTION_CREATE
    assert result.namespace == omero.NS_FILE
    attached = gateway.file_annotations[0]
    assert attached.mimetype == "text/csv"
    assert attached.getNs() == omero.NS_FILE
    assert "spaCR" in attached.description
    assert dataset.annotations == [attached]


def test_file_annotations_append_and_say_so(tmp_path):
    """Replacing a file would mean deleting the previous run's evidence."""
    gateway = FakeGateway()
    dataset = FakeDataset(77, "bag", [])
    results = tmp_path / "results.csv"
    results.write_text("a\n1\n", encoding="utf-8")

    omero.export_file_annotation(gateway, dataset, results)
    second = omero.export_file_annotation(gateway, dataset, results)

    assert len(gateway.file_annotations) == 2
    assert second.duplicates == (gateway.file_annotations[0].getId(),)
    assert "does not delete" in second.reason


def test_attaching_something_that_is_not_a_file_is_refused(tmp_path):
    with pytest.raises(omero.OmeroError, match="not a file"):
        omero.export_file_annotation(FakeGateway(), FakeDataset(1, "d", []),
                                     tmp_path / "nope.csv")


# --- the loop: per-well summaries back onto the plate ------------------------

def test_per_well_summaries_land_on_the_wells_they_came_from():
    plate = build_plate()
    gateway = build_gateway(plate=plate)
    summaries = {
        "A01": [{"cell_area": 10.0}, {"cell_area": 20.0}],
        "B06": [{"cell_area": 5.0}],
        "H12": [{"cell_area": 1.0}],                 # not on this plate
    }

    result = omero.export_plate_summaries(
        gateway, "Plate:4711", summaries,
        annotation_factory=gateway.annotation_factory)

    assert result.targets == ("A01", "B06")
    assert result.missing == ("H12",)
    a01, b06 = plate.wells
    assert dict(a01.annotations[0].getValue())["cell_area"] == "15"
    assert dict(b06.annotations[0].getValue())["cell_area"] == "5"
    assert a01.annotations[0].getNs() == omero.NS_WELL_SUMMARY
    assert "2 create" in result.describe()


def test_rerunning_the_well_export_updates_rather_than_duplicates():
    plate = build_plate()
    gateway = build_gateway(plate=plate)
    for area in (10.0, 40.0):
        omero.export_plate_summaries(
            gateway, "Plate:4711", {"A01": [{"cell_area": area}]},
            annotation_factory=gateway.annotation_factory)

    a01 = plate.wells[0]
    mine = [a for a in a01.annotations if a.getNs() == omero.NS_WELL_SUMMARY]
    assert len(mine) == 1
    assert dict(mine[0].getValue())["cell_area"] == "40"


def test_exporting_summaries_to_a_non_plate_is_refused():
    with pytest.raises(omero.OmeroIdError, match="Dataset"):
        omero.export_plate_summaries(FakeGateway(), "Dataset:77", {})


# ===========================================================================
# 11. Nothing is ever deleted
# ===========================================================================

def test_the_module_contains_no_delete_call():
    """A promise this cheap to check should be checked.

    ``FakeGateway.deleteObjects`` raises, so every test above would fail if a
    delete ever happened on a path they cover; this catches the ones they do
    not.
    """
    source = Path(omero.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    for forbidden in ("deleteObjects", "removeAnnotations", "unlinkAnnotations"):
        assert forbidden not in calls, (
            f"spacr.omero calls {forbidden}; the module docstring promises it "
            f"never deletes anything the user did not name.")


# ===========================================================================
# 12. Shape of the module
# ===========================================================================

def test_every_exported_name_exists():
    missing = [name for name in omero.__all__ if not hasattr(omero, name)]
    assert not missing


@pytest.mark.parametrize(
    "cls, kwargs, attribute",
    [
        (omero.OmeroConnection, {"host": "h"}, "host"),
        (omero.PixelSize, {}, "value"),
        (omero.OmeroRef, {"kind": None, "object_id": 1, "text": "1"}, "object_id"),
        (omero.AnnotationPlan,
         {"action": "create", "namespace": omero.NS_TAG}, "namespace"),
        (omero.WellPosition,
         {"row_index": 1, "column_index": 1, "row_id": "r1", "column_id": "c1",
          "well": "A01"}, "well"),
        (omero.AnnotationResult,
         {"action": "create", "namespace": omero.NS_TAG}, "action"),
    ],
)
def test_the_dataclasses_are_frozen(cls, kwargs, attribute):
    """A settings object that changed halfway through a run would be a bug."""
    instance = cls(**kwargs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, attribute, "mutated")


def test_the_n_planes_arithmetic_is_the_product():
    info = omero.ImageInfo(image_id=1, name="i", size_z=3, size_c=2, size_t=4)
    assert info.n_planes == 24


def test_a_missing_dimension_defaults_to_one_rather_than_zero():
    """A server that returns None for sizeZ must not import zero planes."""
    class Sparse(FakeImage):
        def getSizeZ(self):
            return None

        def getSizeT(self):
            return 0

    info = omero._image_info(Sparse(61, "sparse", size_c=2))
    assert (info.size_z, info.size_t, info.size_c) == (1, 1, 2)
    assert info.n_planes == 2
