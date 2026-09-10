"""Offline checks for the current :mod:`spacr.example_archives` residuals."""

from __future__ import annotations

import io
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import pytest

import spacr.example_archives as module


def test_example_sets_are_found_by_key_and_bad_keys_list_the_choices():
    """A late catalogue hit succeeds and a typo cannot look like success."""
    assert module.example_set("annotate").key == "annotate"

    with pytest.raises(KeyError) as caught:
        module.example_set("not-a-set")

    message = str(caught.value)
    assert "not-a-set" in message
    assert all(item.key in message for item in module.EXAMPLE_SETS)


def test_archive_download_reports_each_nonempty_chunk(monkeypatch, tmp_path):
    """Progress observes written bytes and the server's declared total."""
    calls = []

    class Response:
        headers = {"Content-Length": "3"}

        def raise_for_status(self):
            return None

        def iter_content(self, *, chunk_size):
            calls.append(("chunk_size", chunk_size))
            return iter((b"a", b"", b"bc"))

    def get(url, *, stream, timeout):
        calls.append(("request", url, stream, timeout))
        return Response()

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=get))
    progress = []

    path = module.download_archive(
        "owner/dataset",
        "nested/example.tar",
        tmp_path,
        progress=lambda written, expected: progress.append((written, expected)),
        chunk_size=2,
    )

    assert path == tmp_path / "example.tar"
    assert path.read_bytes() == b"abc"
    assert progress == [(1, 3), (3, 3)]
    assert calls == [
        (
            "request",
            "https://huggingface.co/datasets/owner/dataset/resolve/main/"
            "nested/example.tar?download=true",
            True,
            30,
        ),
        ("chunk_size", 2),
    ]


def _write_tar(path: Path, member: tarfile.TarInfo, payload: bytes = b"") -> None:
    """Write one member to a local tar fixture."""
    with tarfile.open(path, "w") as archive:
        if member.isfile():
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        else:
            archive.addfile(member)


def test_legacy_tar_fallback_extracts_plain_members_and_rejects_escape_shapes(
    monkeypatch, tmp_path
):
    """Pre-3.12 extraction keeps the same path and member-type boundary."""
    monkeypatch.delattr(tarfile, "data_filter")

    safe = tmp_path / "safe.tar"
    with tarfile.open(safe, "w") as archive:
        folder = tarfile.TarInfo("folder")
        folder.type = tarfile.DIRTYPE
        folder.mode = 0o755
        archive.addfile(folder)
        plain = tarfile.TarInfo("folder/data.txt")
        plain.mode = 0o644
        plain.size = 4
        archive.addfile(plain, io.BytesIO(b"safe"))
    destination = tmp_path / "safe-out"
    assert module.extract_example_archive(safe, destination) == 2
    assert (destination / "folder" / "data.txt").read_bytes() == b"safe"

    traversal = tmp_path / "traversal.tar"
    _write_tar(traversal, tarfile.TarInfo("../escape.txt"), b"bad")
    with pytest.raises(ValueError, match="escapes the destination"):
        module.extract_example_archive(traversal, tmp_path / "traversal-out")

    link_archive = tmp_path / "link.tar"
    link = tarfile.TarInfo("link")
    link.type = tarfile.SYMTYPE
    link.linkname = "outside"
    _write_tar(link_archive, link)
    with pytest.raises(ValueError, match="not a plain file or directory"):
        module.extract_example_archive(link_archive, tmp_path / "link-out")


def test_measure_expansion_removes_redundant_archives_and_keeps_bad_ones(
    tmp_path, caplog
):
    """An existing target wins, while a corrupt source remains diagnosable."""
    merged = tmp_path / "merged"
    merged.mkdir()
    redundant = merged / "field.npz"
    np.savez_compressed(redundant, image=np.ones((2, 2)))
    np.save(merged / "field.npy", np.zeros((2, 2)))
    corrupt = merged / "later.npz"
    corrupt.write_bytes(b"not a numpy archive")

    with caplog.at_level("WARNING"):
        module.expand_measure_arrays(merged)

    assert not redundant.exists()
    assert np.array_equal(np.load(merged / "field.npy"), np.zeros((2, 2)))
    assert corrupt.read_bytes() == b"not a numpy archive"
    assert "could not unpack" in caplog.text


def test_an_unreadable_settings_file_does_not_block_other_path_repairs(
    monkeypatch, tmp_path
):
    """One inaccessible settings CSV is skipped without aborting the pass."""
    settings = tmp_path / "settings"
    settings.mkdir()
    blocked = settings / "blocked.csv"
    blocked.write_text(module.DATASET_PLACEHOLDER, encoding="utf-8")
    real_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == blocked:
            raise OSError("permission denied")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert module.make_the_example_paths_absolute(tmp_path) == 0
    assert blocked.is_file()
