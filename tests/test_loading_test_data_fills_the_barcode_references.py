"""Loading test data supplies the barcode tables it does not already have.

Asked for on 2026-09-15, verbatim: "loading test data in map barecodes should
loade these tables if if they are not already loaded, they should be available
to downloade from the package github." Filed as requirement B of item 403.

The three CSVs ship inside the wheel, so the ordinary path touches no network.
These tests cover the ordinary path, the never-overwrite contract that makes
it safe to call, and the fetch that exists for an install whose package data
was stripped -- with the fetch injected, because a test that reaches GitHub
is a test that fails when GitHub is slow.
"""

import hashlib
import os

import pytest

from spacr import settings as S


def test_the_pinned_barcode_hashes_are_the_bundled_files():
    """The pin is recomputed from the files, so it cannot go stale silently.

    A hand-written hash table that nothing re-checks drifts the moment
    someone edits a CSV, and the drift does not show up here -- it shows up
    as a rejected download on the machine of a user whose package data was
    stripped, months later, with an integrity message that blames the
    network. `tools/build_instruction_index.py` had exactly this shape and
    kept items blocked for weeks after they were unblocked.
    """
    for kind, expected in S.BUNDLED_BARCODE_SHA256.items():
        path = S.bundled_barcode_path(kind)
        actual = hashlib.sha256(
            open(path, "rb").read()).hexdigest()
        assert actual == expected, (
            f"the bundled {kind} barcode CSV changed; update "
            f"BUNDLED_BARCODE_SHA256[{kind!r}] to {actual!r} in the same "
            f"change that edited the file")


def test_every_bundled_reference_has_a_hash_and_a_settings_key():
    """The three tables are described by three tables, which must agree.

    Adding a fourth reference and forgetting one of these maps is a reference
    that is never filled in, or one that is downloaded and never checked.
    """
    kinds = set(S._BUNDLED_BARCODE_FILES)

    assert set(S.BUNDLED_BARCODE_SHA256) == kinds
    assert set(S.BUNDLED_BARCODE_SETTING) == kinds


def test_an_empty_reference_is_filled_from_the_package():
    """The common case: nothing set, three tables present, no network."""
    settings = {}

    filled = S._fill_missing_barcode_references(settings)

    assert sorted(filled) == ["column_csv", "grna_csv", "row_csv"]
    for key in filled:
        assert os.path.exists(settings[key])


def test_a_reference_the_user_chose_is_never_overwritten():
    """Loading test data must not replace a table the user already picked.

    This is the failure that would not look like one: the run completes and
    reports numbers, mapped against the wrong guides.
    """
    settings = {"grna_csv": "/somewhere/my_own_guides.csv"}

    filled = S._fill_missing_barcode_references(settings)

    assert settings["grna_csv"] == "/somewhere/my_own_guides.csv"
    assert "grna_csv" not in filled
    assert sorted(filled) == ["column_csv", "row_csv"]


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_reference_counts_as_empty(blank):
    """A key present but blank is a reference the user has not chosen.

    The screen writes "" into a path setting the user cleared, so treating
    only a MISSING key as empty would leave a cleared field unfilled.
    """
    settings = {"row_csv": blank}

    filled = S._fill_missing_barcode_references(settings)

    assert "row_csv" in filled
    assert os.path.exists(settings["row_csv"])


def test_the_url_is_pinned_to_the_installed_release_tag():
    """A branch URL would return today's file, which is not this release's."""
    from spacr import __version__

    url = S._bundled_barcode_url("grna")

    assert url.endswith("/spacr/resources/data/barcodes_grna.csv")
    assert f"/v{__version__}/" in url
    assert "nightly" not in url and "/main/" not in url


def test_an_unknown_reference_is_refused_by_both_entry_points():
    """Neither the URL nor the fetch invents a table name."""
    with pytest.raises(ValueError, match="Unknown barcode reference"):
        S._bundled_barcode_url("protein")

    with pytest.raises(ValueError, match="Unknown barcode reference"):
        S._ensure_bundled_barcode("protein")


def test_a_missing_table_is_fetched_and_written(tmp_path, monkeypatch):
    """The fetch path runs when, and only when, the local copy is gone."""
    real = S.bundled_barcode_path("row")
    payload = open(real, "rb").read()
    moved = tmp_path / "barcodes_row.csv"
    os.replace(real, moved)
    asked = []

    def fetch(url):
        asked.append(url)
        return payload

    try:
        got = S._ensure_bundled_barcode("row", fetch=fetch)
        assert got == real
        assert open(got, "rb").read() == payload
        assert len(asked) == 1 and asked[0] == S._bundled_barcode_url("row")
        # And the second call is served locally, so a screen that fills the
        # references twice does not fetch twice.
        S._ensure_bundled_barcode("row", fetch=fetch)
        assert len(asked) == 1
    finally:
        os.replace(moved, real) if moved.exists() else None
        if not os.path.exists(real):
            open(real, "wb").write(payload)


def test_a_fetched_table_that_is_not_the_right_one_is_not_written(tmp_path):
    """A wrong table maps reads to the wrong wells and still finishes.

    So the bytes are checked before they are written, and a mismatch leaves
    no file behind -- a half-right CSV on disk would be served from the local
    path forever after, and the check would never run again.
    """
    real = S.bundled_barcode_path("column")
    payload = open(real, "rb").read()
    moved = tmp_path / "barcodes_column.csv"
    os.replace(real, moved)

    try:
        with pytest.raises(ValueError, match="not the one this release ships"):
            S._ensure_bundled_barcode(
                "column", fetch=lambda url: b"name,sequence\nfake,ACGT\n")
        assert not os.path.exists(real)
        assert not os.path.exists(f"{real}.partial")
    finally:
        os.replace(moved, real)


def test_a_reference_that_cannot_be_produced_leaves_the_others_filled(tmp_path):
    """One unavailable table must not cost the user the other two.

    "Load test data" that raises is worse than one that fills what it can:
    an empty field is a state the screen already knows how to show.
    """
    real = S.bundled_barcode_path("grna")
    payload = open(real, "rb").read()
    moved = tmp_path / "barcodes_grna.csv"
    os.replace(real, moved)
    settings = {}

    try:
        def fetch(url):
            raise OSError("no route to host")

        filled = S._fill_missing_barcode_references(settings, fetch=fetch)

        assert sorted(filled) == ["column_csv", "row_csv"]
        assert not settings.get("grna_csv")
    finally:
        os.replace(moved, real)
