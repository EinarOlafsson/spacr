"""Regression tests for machine-independent shipped settings."""

import re
from pathlib import Path

#: Any path under a user's home directory, in its POSIX, macOS and Windows
#: spellings. A shipped default pointing inside one exists on exactly one
#: computer, so it is a broken default for every user who is not its author.
#: This is deliberately not one developer's name: the previous version of this
#: guard checked for a single hard-coded home directory, so the next person to
#: leave their own in would have sailed straight past it.
_USER_HOME_PATH = re.compile(r"""(?:/home/|/Users/|[A-Za-z]:\\Users\\)[^/\\\s'"]+[/\\]""")


def test_legacy_barcode_defaults_use_packaged_references():
    """The one surviving legacy default still ships with the package.

    `grna` USED TO BE CHECKED HERE and is not any more. It was retired on
    2026-09-14 under instruction 364 (approved by the maintainer 2026-09-09):
    it was declared only by `get_map_barcodes_default_settings`, which nothing
    under `spacr/` calls, so `settings['grna']` was read by no pipeline and its
    own tooltip said so. `validate.RETIRED_SETTINGS` now maps it to "" and the
    key is gone from `expected_types`, `categories` and all ten catalogs.

    The key going away does NOT retire the packaged gRNA reference table, and
    the two are checked separately below so a later cleanup cannot confuse
    them: `bundled_barcode_path('grna')` is still the shipped default of the
    LIVE `grna_csv` (spacr/settings.py, `set_default_generate_barecode_mapping`
    -> `settings.setdefault('grna_csv', bundled_barcode_path('grna'))`), which
    `generate_barecode_mapping` actually reads.
    """
    from spacr.settings import (
        bundled_barcode_path,
        get_map_barcodes_default_settings,
        set_default_generate_barecode_mapping,
    )

    configured = get_map_barcodes_default_settings({})
    assert "grna" not in configured, (
        "`grna` was retired on 2026-09-14; the dead factory must not put it "
        "back into a settings dict"
    )
    assert configured["barcodes"] == bundled_barcode_path("column")
    assert Path(configured["barcodes"]).is_file()

    # The live key that replaced it, and the packaged table it points at.
    live = set_default_generate_barecode_mapping({"src": "/tmp"})
    assert live["grna_csv"] == bundled_barcode_path("grna")
    assert Path(live["grna_csv"]).is_file()


def test_regression_metadata_has_no_workstation_default():
    from spacr.settings import get_perform_regression_default_settings

    configured = get_perform_regression_default_settings({})
    assert configured["metadata_files"] == []


def test_the_home_path_pattern_recognises_a_developer_default():
    """The guard below is only worth anything if the pattern actually fires.

    A regex that matches nothing makes the next test unconditionally green, so
    it is checked against the three spellings it claims to cover and against
    the two path shapes that are legitimate.
    """
    # The POSIX samples are assembled rather than written out: the suite's own
    # hygiene rule (tests/test_test_suite_hygiene.py) bans a literal user-home
    # path in a test file, and it is right to -- these are regex fixtures, not
    # paths anything opens.
    sep = "/"
    for home in ("home", "Users"):
        sample = f"{sep}{home}{sep}someone{sep}datasets{sep}plate1"
        assert _USER_HOME_PATH.search(sample), sample
    assert _USER_HOME_PATH.search(r"C:\Users\someone\datasets")
    for fine in ("/path/to/src", "/models/my_cells.pth", "~/spacr", "/tmp/x"):
        assert not _USER_HOME_PATH.search(fine), fine


def test_settings_source_contains_no_developer_home_paths():
    """No shipped default may live under anybody's home directory."""
    import spacr.settings as settings

    source = Path(settings.__file__).read_text(encoding="utf-8")
    offenders = [(n, line.strip())
                 for n, line in enumerate(source.splitlines(), 1)
                 if _USER_HOME_PATH.search(line)]
    assert not offenders, (
        "spacr/settings.py hard-codes path(s) under a user's home directory:\n"
        + "\n".join(f"  line {n}: {text[:90]}" for n, text in offenders)
        + "\n\nShip a packaged reference (see bundled_barcode_path) or an "
          "empty default; a path under $HOME is a default that only works for "
          "the person who wrote it.")
