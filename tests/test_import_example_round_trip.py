"""Import reads back every format and naming its test data was written in.

Ledger item 462. The published set (einarolafsson/spacr-example-import) writes
the same twelve planes once per filename convention Mask's ``metadata_type``
offers and once per container format, and ships a manifest stating what each
file truly is. These tests build the same set from synthetic pixels with the
SAME builder (``tools/build_import_example.py``), import every variant with
the Import module's own planner, and compare each output plane's well, field,
channel, z and t with the manifest.

A PASS HERE IS ONLY WORTH SOMETHING IF A WRONG READ FAILS, so the controls at
the end read the same folders the wrong way -- by folders instead of by the
convention, and with the zero-based table emptied -- and require the check to
object.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_import_example as builder  # noqa: E402

sys.path.remove(str(ROOT / "tools"))

from spacr import convert as cv  # noqa: E402
from spacr import foreign as fgn  # noqa: E402
from spacr import import_examples as ix  # noqa: E402
from spacr import regex_infer  # noqa: E402
from spacr.errors import ConfigurationError  # noqa: E402

GENERATED = [v.key for v in ix.IMPORT_VARIANTS if v.route == "import"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """The whole generated set, from synthetic pixels, built once."""
    out = tmp_path_factory.mktemp("import_example") / "import_example"
    return builder.build(builder.synthetic_fields(), out)


@pytest.fixture(scope="module")
def results(dataset):
    """Every variant's problems."""
    return builder.verify(dataset)


def test_every_generated_variant_is_in_the_manifest(dataset):
    listed = {row["variant"] for row in ix.manifest_rows(dataset)}
    assert listed == set(GENERATED)


def test_every_metadata_type_mask_offers_has_a_variant():
    """"represent each of those": every convention in Mask's dropdown."""
    offered = set(regex_infer._metadata_convention_keys())
    covered = {v.metadata_type for v in ix.IMPORT_VARIANTS}
    assert offered <= covered, sorted(offered - covered)


@pytest.mark.parametrize("key", GENERATED)
def test_the_variant_imports_as_what_it_truly_is(results, key):
    assert results[key] == [], "\n".join(results[key])


@pytest.mark.parametrize("key", GENERATED)
def test_the_manifest_lists_every_file_on_disk(dataset, key):
    """The check compares against the manifest, so the manifest must be whole."""
    listed = {row["path"] for row in ix.manifest_rows(dataset)
              if row["variant"] == key}
    on_disk = {p.relative_to(dataset).as_posix()
               for p in (dataset / "variants" / key).rglob("*")
               if p.is_file()}
    assert on_disk == listed


def test_the_zeiss_variant_is_real_czi_carrying_its_channel_name(dataset):
    czifile = pytest.importorskip("czifile")
    first = sorted((dataset / "variants" / "zeiss_czi" / "plate1").rglob(
        "*.czi"))[0]
    with czifile.CziFile(str(first)) as handle:
        assert "DAPI" in handle.metadata()


def test_field_numbers_the_names_state_are_kept(dataset):
    """F009 stays field 9: renumbering 1..N is what the manifest forbids."""
    inputs = ix.variant_inputs(dataset, "cellvoyager")
    plan = fgn.plan_import(inputs["images"], inputs["masks"],
                           inputs["measurements"],
                           metadata_type="cellvoyager")
    assert sorted(plan.stems) == ["plate1_E01_10", "plate1_E01_9",
                                  "plate1_E02_10", "plate1_E02_9"]


def _read_by(dataset, key, metadata_type):
    """Plan one variant with a chosen convention and report its problems."""
    rows = [r for r in ix.manifest_rows(dataset) if r["variant"] == key]
    inputs = ix.variant_inputs(dataset, key)
    original = ix.variant_inputs

    def _override(root, which):
        found = original(root, which)
        found["metadata_type"] = metadata_type
        return found

    ix.variant_inputs = _override
    try:
        return builder._check_import(dataset, key, rows)
    finally:
        ix.variant_inputs = original


@pytest.mark.parametrize("key", ["opera_phenix", "arrayscan", "cq1",
                                 "incell", "zeiss_czi"])
def test_reading_by_folders_instead_does_not_pass(dataset, key):
    """The negative control: the check objects when the naming is ignored."""
    assert _read_by(dataset, key, "auto"), (
        f"{key} passed when read by folders: the check is not checking")


def test_reading_by_the_wrong_convention_does_not_pass(dataset):
    assert _read_by(dataset, "arrayscan", "cellvoyager")


def test_forgetting_that_arrayscan_counts_from_zero_does_not_pass(
        dataset, monkeypatch):
    monkeypatch.setitem(regex_infer._METADATA_ZERO_BASED, "arrayscan", ())
    problems = _read_by(dataset, "arrayscan", "arrayscan")
    assert any("truly" in p for p in problems), problems


@pytest.mark.parametrize("key,token,well", [
    ("cq1", "0097", "E01"),
    ("cq1", "1", "A01"),
    ("opera_phenix", "r05c01", "E01"),
    ("leica_matrix_screener", "U00--V04", "E01"),
    ("leica_matrix_screener", "U02--V00", "A03"),
    ("incell", "E - 01", "E01"),
    ("scanr", "E1", "E01"),
    ("zeiss_zen_split_tiles", "00001", None),
    ("micromanager_mda", "img", None),
])
def test_each_vendor_spelling_of_a_well_is_read(key, token, well):
    assert cv._convention_well(key, token) == well


def test_a_file_that_breaks_the_convention_is_reported_not_guessed(tmp_path):
    import tifffile

    tifffile.imwrite(str(tmp_path / "plate1_A01_T0001F001L01A01Z01C01.tif"),
                     np.zeros((4, 4), np.uint16))
    tifffile.imwrite(str(tmp_path / "notes_thumbnail.tif"),
                     np.zeros((4, 4), np.uint16))
    sources = cv.scan(str(tmp_path), metadata_type="cellvoyager")
    bad = [s for s in sources if not s.readable]
    assert [Path(s.path).name for s in bad] == ["notes_thumbnail.tif"]
    assert "cellvoyager" in bad[0].error


def test_custom_without_a_pattern_is_refused(tmp_path):
    with pytest.raises(ConfigurationError, match="custom_regex"):
        cv.scan(str(tmp_path), metadata_type="custom")


def test_an_unknown_convention_is_refused(tmp_path):
    with pytest.raises(ConfigurationError, match="metadata_type"):
        cv.scan(str(tmp_path), metadata_type="zeiss")


def test_repeated_field_numbers_fall_back_to_counting(tmp_path):
    """Two fields claiming one number keep 1..N rather than collide."""
    sources = [cv.SourceImage(path=f"/x/{i}.tif", plate="p", well="A01",
                              field=name, meta={"field_number": 3})
               for i, name in enumerate(("a", "b"))]
    assert cv._named_field_numbers(sources, ["a", "b"]) == {}


def test_a_lif_is_described_series_by_series(tmp_path, monkeypatch):
    """One LIF mixing a 2-channel snapshot with a 1-channel z-stack."""
    import types
    from collections import namedtuple

    dims = namedtuple("Dims", "x y z t m")

    class Image:
        def __init__(self, channels, z):
            self.dims = dims(x=4, y=4, z=z, t=1, m=1)
            self.channels = channels
            self.name = "s"

        def get_frame(self, z=0, t=0, c=0, m=0):
            return np.full((4, 4), c, np.uint8)

    class LifFile:
        def __init__(self, path):
            self.images = [Image(2, 1), Image(1, 3)]

        def get_iter_image(self, img_n=0):
            return iter(self.images)

    package = types.ModuleType("readlif")
    reader = types.ModuleType("readlif.reader")
    reader.LifFile = LifFile
    package.reader = reader
    monkeypatch.setitem(sys.modules, "readlif", package)
    monkeypatch.setitem(sys.modules, "readlif.reader", reader)
    (tmp_path / "mixed.lif").write_bytes(b"lif")
    sources = cv.scan(str(tmp_path))
    assert [(s.n_channels, s.z) for s in sources] == [(2, 1), (1, 3)]
    assert cv._read_source(sources[1]).shape == (1, 3, 1, 4, 4)
