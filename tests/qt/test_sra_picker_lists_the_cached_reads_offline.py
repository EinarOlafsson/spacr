"""The Load test data picker opens on the cached read set without ENA.

Tutorial capture runs with the network unshared. With the ``reads`` example
set unpacked the picker must list the runs from its copy of ENA's listing and
fetch a limited set of reads from it; ENA is asked only when there is no
cache. The network is mocked throughout.
"""
from __future__ import annotations

import gzip
import json
import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


from spacr import sra  # noqa: E402
from spacr.qt.widgets import sra_picker  # noqa: E402

pytestmark = pytest.mark.qt

LISTING = (
    "run_accession\tlibrary_name\tfastq_ftp\tfastq_bytes\tread_count\n"
    "SRR33531217\thilib_p1\t"
    "ftp.sra.ebi.ac.uk/a/SRR33531217_1.fastq.gz;"
    "ftp.sra.ebi.ac.uk/a/SRR33531217_2.fastq.gz\t"
    "2833522805;3222897932\t73698595\n"
)


@pytest.fixture
def cached_home(tmp_path, monkeypatch):
    """A HOME whose example data holds a cached read set of 20 reads a file."""
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = sra._cached_read_folder()
    folder.mkdir(parents=True)
    (folder / sra.CACHED_LISTING).write_text(LISTING, encoding="utf-8")
    files = {}
    for mate in (1, 2):
        name = f"SRR33531217_{mate}.fastq.gz"
        with gzip.open(folder / name, "wt") as handle:
            for n in range(20):
                handle.write(f"@r{mate}_{n}\nACGT\n+\nIIII\n")
        files[name] = {"reads": 20}
    (folder / sra.CACHED_MANIFEST).write_text(json.dumps(
        {"reads_per_file": 20, "files": files}), encoding="utf-8")
    return folder


def _no_ena(*_args, **_kwargs):
    raise AssertionError("ENA was asked")


def test_the_picker_lists_and_fetches_from_the_cache(qtbot, tmp_path,
                                                     cached_home, monkeypatch):
    monkeypatch.setattr(sra_picker, "runs_for", _no_ena)
    monkeypatch.setattr("urllib.request.urlopen", _no_ena)
    dialog = sra_picker.SraPicker(tmp_path / "out")
    qtbot.addWidget(dialog)
    assert [one.filename for one in dialog._files] == [
        "SRR33531217_1.fastq.gz", "SRR33531217_2.fastq.gz"]
    assert dialog._download.isEnabled()
    dialog._reads.setMinimum(1)
    dialog._reads.setValue(5)
    dialog._start()
    qtbot.waitUntil(lambda: dialog._worker is None, timeout=10000)
    assert len(dialog.written) == 2
    with gzip.open(dialog.written[0], "rt") as handle:
        assert sum(1 for _ in handle) == 20


def test_without_a_cache_the_picker_still_asks_ena(qtbot, tmp_path,
                                                   monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    asked = []

    def _listing(*_args, **_kwargs):
        asked.append(True)
        return ()

    monkeypatch.setattr(sra_picker, "runs_for", _listing)
    dialog = sra_picker.SraPicker(tmp_path / "out")
    qtbot.addWidget(dialog)
    assert asked == [True]
    assert dialog.chosen_files() == []
