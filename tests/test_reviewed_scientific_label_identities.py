"""Shared scientific vocabulary must not disable another locale's prose gate."""
import importlib
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def builder():
    import sys

    directory = str(Path(__file__).resolve().parents[1] / "tools")
    sys.path.insert(0, directory)
    try:
        yield importlib.import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(directory)


@pytest.mark.parametrize("source,language", [
    ("Median", "sv"), ("Median", "de"),
    ("Minimum", "sv"), ("Minimum", "de"), ("Minimum", "fr"),
    ("Diameter", "sv"), ("Plaque", "de"), ("Plaque", "fr"),
    ("Detector", "es"), ("Detector", "pt"),
    ("Voxels", "pt"), ("Voxels", "fr"), ("Triangle", "fr"),
])
def test_shared_word_is_accepted_only_in_its_reviewed_locale(builder, source, language):
    assert not builder._translation_rejection_reasons(
        source, source, language, force=True)
    assert builder._translation_rejection_reasons(
        source, source, "zh_CN", force=True)
    assert builder._translation_rejection_reasons(
        source + " missing translation", source + " missing translation",
        language, force=True)
