"""The branches the 2026-10-11 batch added and no test reached.

CI run 38188593854 measured them as newly uncovered: the version fallbacks
N676 gave the run journal and the archive report, the field-size guards it
gave the saved masks, the offline read cache's refusals, a pasted ``src``
that names no folder, and the DigitalMicrograph reader's two early stops.
Each test drives the real function into the branch and checks what it says.
"""
from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest

from spacr import cli, core, report, run_journal, sra
from spacr import io as sio
from spacr import object as spacr_object
from tests.tem_files import write_dm


def _no_version():
    raise RuntimeError("version metadata unreadable")


def test_the_archive_text_says_unknown_when_the_version_cannot_be_read(
        monkeypatch):
    import spacr.version as version

    monkeypatch.setattr(version, "get_version", _no_version)
    text = report._archive_analysis_text({})
    assert text.startswith("Images were analysed with spaCR unknown.")


def test_the_run_journal_falls_back_to_the_installed_metadata(monkeypatch):
    import spacr.version as version

    monkeypatch.setattr(version, "get_version", _no_version)
    monkeypatch.setattr(run_journal, "_pkg_version",
                        lambda name: f"dist-info of {name}")
    assert run_journal._spacr_version() == "dist-info of spacr"


def test_a_blank_src_is_not_read_as_a_list_of_plates():
    types = {"src": (str, list)}
    assert cli.coerce_value("src", "", "path", types) == ""
    assert cli.coerce_value("src", "   ", "path", types) == "   "


@pytest.mark.parametrize("bad", (123, 4.5, {"folder": "x"}))
def test_a_watch_source_that_is_not_a_path_fills_in_nothing(bad):
    settings = {"z_stack": True, "z_segmentation_mode": None,
                "z_axis": None, "src": bad}
    result, filled = core._watch_vendor_recipe(settings)
    assert result is settings
    assert filled == {}
    assert settings["z_segmentation_mode"] is None


def test_only_a_channel_last_batch_names_a_field_shape():
    assert spacr_object._field_shape_of(None) is None
    assert spacr_object._field_shape_of([[1, 2]]) is None
    assert spacr_object._field_shape_of(np.zeros((2, 9, 7))) is None
    assert spacr_object._field_shape_of(np.zeros((2, 9, 7, 3))) == (9, 7)
    assert spacr_object._field_shape_of(np.zeros((2, 4, 9, 7, 3))) == (9, 7)


def test_a_mask_whose_header_cannot_be_read_again_is_generated_again(
        tmp_path, monkeypatch, capsys):
    import spacr.resume as resume

    np.save(tmp_path / "a.npy", np.zeros((6, 5), np.uint16))
    batch = np.zeros((1, 6, 5, 3), np.float32)
    kept, names = sio._check_masks(batch, ["a.npy"], str(tmp_path),
                                   resume=True)
    assert names == [] and len(kept) == 0

    real = resume.read_npy_header
    reads = []

    def vanishes_after_validation(path):
        reads.append(path)
        if len(reads) > 1:
            raise OSError(f"{path} was replaced while it was read")
        return real(path)

    monkeypatch.setattr(resume, "read_npy_header", vanishes_after_validation)
    kept, names = sio._check_masks(batch, ["a.npy"], str(tmp_path),
                                   resume=True)
    assert names == ["a.npy"]
    assert len(kept) == 1
    assert len(reads) == 2


def _dm4_reader(body: bytes) -> sio._DMReader:
    header = struct.pack(">i", 4) + struct.pack(">Q", len(body) + 16)
    header += struct.pack(">i", 1)
    return sio._DMReader(io.BytesIO(header + body))


def test_a_dm_value_cut_short_is_refused():
    reader = _dm4_reader(b"\x01\x00\x02")
    with pytest.raises(ValueError, match="ends inside a value"):
        reader._values(np.dtype("<u2"), 2)


@pytest.mark.parametrize("tail", (b"\x00", b""))
def test_a_dm_group_stops_at_padding_or_the_end_of_the_file(tail):
    body = b"\x00\x01" + struct.pack(">Q", 3) + tail
    assert _dm4_reader(body).group() == {}


def test_a_dm_root_that_announces_one_tag_too_many_still_reads(tmp_path):
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    path = tmp_path / "field.dm4"
    write_dm(path, image, thumbnail=False, scale=2.0)
    raw = bytearray(path.read_bytes())
    count = struct.unpack(">Q", bytes(raw[18:26]))[0]
    raw[18:26] = struct.pack(">Q", count + 1)
    padded = tmp_path / "padded.dm4"
    padded.write_bytes(bytes(raw))
    read, meta = sio.read_dm(padded)
    assert np.array_equal(read, image)
    assert meta["pixel_size_nm"] == pytest.approx(2.0)


def _manifest(folder, manifest, listing=True):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / sra.CACHED_MANIFEST).write_text(json.dumps(manifest))
    if listing:
        (folder / sra.CACHED_LISTING).write_text("run_accession\n")
    return folder


def _run_file():
    return sra.RunFile(run="SRR1", library="lib", mate=1,
                       url="https://example.org/a/SRR1_1.fastq.gz")


def test_a_cache_with_an_unreadable_read_count_holds_none(tmp_path):
    folder = _manifest(tmp_path / "reads", {"reads_per_file": "lots"})
    assert sra._cached_reads_per_file(folder) == 0
    other = _manifest(tmp_path / "list", {"reads_per_file": [5]})
    assert sra._cached_reads_per_file(other) == 0
    assert sra._cached_reads_per_file(tmp_path / "absent") == 0


def test_no_cached_copy_without_a_set_or_with_an_unreadable_count(tmp_path):
    one = _run_file()
    assert sra._cached_copy(one, 10, tmp_path / "absent") is None
    unlisted = _manifest(tmp_path / "unlisted", {"reads_per_file": 50},
                         listing=False)
    (unlisted / one.filename).write_bytes(b"")
    assert sra._cached_copy(one, 10, unlisted) is None
    folder = _manifest(tmp_path / "reads", {
        "reads_per_file": 50,
        "files": {one.filename: {"reads": "many"}}})
    (folder / one.filename).write_bytes(b"")
    assert sra._cached_copy(one, 10, folder) is None
    good = _manifest(tmp_path / "good", {
        "reads_per_file": 50, "files": {one.filename: {"reads": 50}}})
    (good / one.filename).write_bytes(b"")
    assert sra._cached_copy(one, 10, good) == good / one.filename
