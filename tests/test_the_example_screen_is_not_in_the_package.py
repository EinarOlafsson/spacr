"""The example screen must never enter a pip or conda release.

The maintainer's condition, 2026-08-20: "as long as the files dont get
included in pip releases or conda releases."

WHY A TEST AND NOT A PROMISE. `MANIFEST.in` carries
``recursive-include spacr/resources/data *.csv``, so the moment anyone drops
one of these CSVs into that folder -- which is exactly what the first version
of this feature was asked to do -- it lands in the sdist, in the wheel, and
in every conda build made from them. Nothing would fail; every install would
just be 33 MB heavier and would carry an unpublished screen.

They live on a GitHub release instead, fetched on demand by
`spacr.example_data` and cached in the user's cache directory. That is also
what makes them removable: a release asset can be deleted, and a file
committed to a public repository cannot.
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: How the eight files are named. Matched as patterns because the point is to
#: catch a file LIKE these, not only these exact eight.
PATTERNS = ("*_dv.csv", "*unique_combinations*.csv")

#: A ceiling on everything `MANIFEST.in` ships out of resources/data.
#:
#: A PER-FILE THRESHOLD DOES NOT WORK HERE. The folder already holds
#: `toxoplasma_metadata.csv` at 2.9 MB -- deliberately, it is the bundled
#: annotation -- and the smallest example CSV is 3.5 MB, so any per-file line
#: that admits the first and rejects the second is a number chosen to sit in
#: a 0.6 MB gap and would break on the next legitimate addition.
#:
#: The total is the honest measure: the folder is 4.8 MB today and every one
#: of those bytes is in every install. 8 MB leaves room to grow and still
#: trips on the FIRST example file dropped in, which is the mistake this
#: guards -- the original ask was literally "add my cound and dependent
#: variable csvs to the datafolder in spacr".
DATA_FOLDER_CEILING = 8 * 1024 * 1024


def _tracked() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=False)
    return [line for line in out.stdout.splitlines() if line.strip()]


class TestTheScreenIsNotShipped:

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_no_example_csv_is_tracked(self, pattern):
        import fnmatch

        hits = [p for p in _tracked() if fnmatch.fnmatch(p.split("/")[-1],
                                                         pattern)]

        assert not hits, (
            f"{pattern} is in the repository, so it reaches the sdist and "
            f"the wheel: {hits}. The example screen is a release asset -- "
            f"see spacr/example_data.py.")

    def test_the_shipped_data_folder_stays_small(self):
        """`MANIFEST.in` ships every CSV under resources/data, so anything
        dropped there is in everybody's install and nothing fails to warn."""
        folder = ROOT / "spacr" / "resources" / "data"
        if not folder.is_dir():
            pytest.skip("no bundled data folder in this checkout")
        sizes = {p.name: p.stat().st_size for p in folder.iterdir()
                 if p.is_file()}
        total = sum(sizes.values())

        assert total <= DATA_FOLDER_CEILING, (
            f"resources/data is {total / 1024 / 1024:.1f} MB and every byte "
            f"ships in every pip and conda install. Largest: "
            f"{sorted(sizes.items(), key=lambda kv: -kv[1])[:3]}. If this is "
            f"a large dataset, put it on a release and fetch it like "
            f"spacr/example_data.py does.")

    def test_the_downloader_itself_is_small(self):
        """The code that fetches 33 MB should not itself be 33 MB."""
        for name in ("example_data.py", "example_data_manifest.py"):
            size = (ROOT / "spacr" / name).stat().st_size
            assert size < 64 * 1024, f"{name} is {size} bytes"


class TestTheManifestStillDescribesTheFiles:
    """The manifest is what makes a truncated download an error rather than
    a parse failure, so it has to keep describing all eight."""

    def test_it_names_eight_files_of_two_kinds(self):
        from spacr.example_data_manifest import FILES

        assert len(FILES) == 8
        assert {f["kind"] for f in FILES} == {"counts", "scores"}
        assert len({f["name"] for f in FILES}) == 8

    def test_every_entry_carries_a_size_and_a_digest(self):
        from spacr.example_data_manifest import FILES

        for entry in FILES:
            assert entry["bytes"] > 0
            assert len(entry["sha256"]) == 64

    def test_the_release_tag_is_not_a_version(self):
        """A version tag would orphan the assets on the next release."""
        from spacr.example_data import RELEASE_TAG

        assert not RELEASE_TAG[:1].isdigit()
        assert not RELEASE_TAG.lower().startswith("v")
