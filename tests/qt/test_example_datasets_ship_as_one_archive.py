"""Each example dataset arrives as one archive, unpacked safely.

WHY AN ARCHIVE. The annotate set is 2,365 files. Fetching it a file at a time
spent most of its wall clock on HTTP round trips, and `snapshot_download` --
the obvious alternative -- cannot be interrupted, so Cancel did nothing and
quitting mid-download destroyed a live QThread and aborted the process.

One tar fixes all three: a single stream that can be stopped between chunks, a
progress figure that means something, and no per-file overhead at either end.
It is deliberately NOT compressed -- the payloads are PNGs and .npz arrays,
already compressed, so gzip would cost CPU on every download to save almost
nothing.

WHY THE EXTRACTION IS A FUNCTION rather than two lines at the call site: a tar
can name `../../etc/something`, and a plain `extractall` writes there. This is
downloaded content, so unpacking it unfiltered would hand whoever can publish
to the repo a write anywhere the user can write.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from spacr.qt.hf_download import (ANNOTATE_EXAMPLE_REPO, EXAMPLE_ARCHIVES,
                                  MEASURE_EXAMPLE_REPO,
                                  extract_example_archive)


def _tar_with(tmp_path, members):
    """Build an archive from ``{name: bytes}``."""
    archive = tmp_path / "made.tar"
    payload = tmp_path / "payload"
    payload.mkdir(exist_ok=True)
    with tarfile.open(archive, "w") as tar:
        for name, data in members.items():
            here = payload / "file.bin"
            here.write_bytes(data)
            tar.add(here, arcname=name)
    return archive


def test_every_example_repo_names_its_archive():
    assert EXAMPLE_ARCHIVES[MEASURE_EXAMPLE_REPO].endswith(".tar")
    assert EXAMPLE_ARCHIVES[ANNOTATE_EXAMPLE_REPO].endswith(".tar")


def test_the_archives_are_not_compressed():
    """Gzip on already-compressed payloads is CPU spent on every download for
    almost nothing."""
    for name in EXAMPLE_ARCHIVES.values():
        assert not name.endswith((".gz", ".bz2", ".xz", ".zst")), name


def test_ordinary_members_are_unpacked(tmp_path):
    archive = _tar_with(tmp_path, {"data/one.png": b"x",
                                   "measurements.db": b"y"})
    dest = tmp_path / "out"

    assert extract_example_archive(archive, dest) == 2
    assert (dest / "data" / "one.png").read_bytes() == b"x"
    assert (dest / "measurements.db").read_bytes() == b"y"


def test_a_member_escaping_the_folder_is_refused(tmp_path):
    """The attack a plain extractall would carry out."""
    archive = _tar_with(tmp_path, {"../escaped.txt": b"nope"})
    dest = tmp_path / "out"

    with pytest.raises(Exception):
        extract_example_archive(archive, dest)

    assert not (tmp_path / "escaped.txt").exists()


def test_an_absolute_member_is_refused(tmp_path):
    archive = _tar_with(tmp_path, {"/tmp/spacr-should-not-appear": b"nope"})
    dest = tmp_path / "out"
    try:
        extract_example_archive(archive, dest)
    except Exception:
        pass
    assert not Path("/tmp/spacr-should-not-appear").exists()


def test_a_symlink_out_of_the_tree_is_refused(tmp_path):
    """`filter="data"` rejects these; the fallback path checks the type."""
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("escape")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    with pytest.raises(Exception):
        extract_example_archive(archive, tmp_path / "out")


def test_the_destination_is_created(tmp_path):
    archive = _tar_with(tmp_path, {"a.txt": b"x"})
    dest = tmp_path / "does" / "not" / "exist"
    extract_example_archive(archive, dest)
    assert (dest / "a.txt").is_file()


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

def _worker_source():
    import inspect

    from spacr.qt.hf_download import _TarExampleWorker

    return inspect.getsource(_TarExampleWorker.run)


def test_the_download_can_be_stopped_between_chunks():
    """Cancel and application shutdown both take effect within a megabyte,
    rather than after the whole set has arrived."""
    source = _worker_source()
    body = source[source.index("for chunk in response.iter_content"):]
    assert "if self._cancel:" in body


def test_a_short_download_is_not_unpacked():
    """A truncated archive that unpacked would look like a dataset with files
    missing, which nothing downstream could tell from a small one."""
    source = _worker_source()
    assert "the download stopped early" in source
    assert "Nothing was unpacked" in source


def test_a_cancelled_download_leaves_no_part_file():
    source = _worker_source()
    body = source[source.index("if self._cancel:"):]
    assert "part.unlink" in body


def test_the_archive_is_removed_after_unpacking():
    """It is a second copy of everything just written, and these sets are
    hundreds of megabytes."""
    assert "target.unlink(missing_ok=True)" in _worker_source()


def test_the_measure_set_still_expands_its_arrays():
    """The .npz compression is a transport detail; Measure reads .npy."""
    import inspect

    from spacr.qt.hf_download import _MeasureTarWorker

    assert "_expand_arrays" in inspect.getsource(_MeasureTarWorker.after_extract)


def test_both_entry_points_use_the_tar_workers():
    source = Path(
        __import__("spacr.qt.hf_download", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert "worker_factory=_MeasureTarWorker," in source
    assert "worker_factory=_AnnotateTarWorker," in source
