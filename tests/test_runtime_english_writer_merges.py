"""Generated English sources retain values and merge independent caption additions."""
import hashlib
from importlib import import_module
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def generate_english(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    builder = import_module("build_i18n_catalogs")

    def generate(name, captions):
        monkeypatch.setattr(builder, "CATALOG_DIR", tmp_path / name)
        sources = {
            "setting_labels": {"example": "Example"},
            "setting_tooltips": {"example": "An example setting."},
            "module_summaries": {"example": "Example module"},
            "categories": ("Example category",),
            "ui": tuple(sorted(captions)),
        }
        return builder.write_english(sources)

    return generate


def read_generated(path):
    namespace = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("captions", [
    frozenset(),
    frozenset({"Single caption"}),
    frozenset({"Quotes: ' and \"", "Line one\nLine two", "Íslenska 中文", "{path}"}),
])
def test_normal_writer_preserves_source_values_and_hashes(generate_english, captions):
    result = read_generated(generate_english("roundtrip", captions))
    assert result["UI_SOURCES"] == captions
    assert result["CATEGORY_SOURCES"] == frozenset({"Example category"})
    assert result["SETTING_LABELS"] == {"example": "Example"}
    expected = {
        ("SETTING_LABELS", "example"): "Example",
        ("SETTING_TOOLTIPS", "example"): "An example setting.",
        ("MODULE_SUMMARIES", "example"): "Example module",
        ("CATEGORY_HELP", "Example category"): "Example category",
        **{("UI", text): text for text in captions},
    }
    assert result["SOURCE_HASHES"] == {
        key: hashlib.sha256(value.encode()).hexdigest() for key, value in expected.items()
    }


def test_disjoint_generated_caption_additions_merge_without_conflicts(generate_english, tmp_path):
    captions = frozenset(f"Caption {number:03d}" for number in range(60))
    left_added, right_added = "Caption 005a", "Caption 050a"
    base = generate_english("base", captions)
    left = generate_english("left", captions | {left_added})
    right = generate_english("right", captions | {right_added})
    merged = subprocess.run(
        ["git", "merge-file", "-p", str(left), str(base), str(right)],
        capture_output=True, text=True, timeout=10,
    )
    assert merged.returncode == 0, "Independent generated caption additions conflicted"
    output = tmp_path / "merged.py"
    output.write_text(merged.stdout)
    result = read_generated(output)
    expected = read_generated(generate_english("expected", captions | {left_added, right_added}))
    for assignment in (
        "UI_SOURCES", "CATEGORY_SOURCES", "SETTING_LABELS", "SETTING_TOOLTIPS",
        "MODULE_SUMMARIES", "SOURCE_HASHES",
    ):
        assert result[assignment] == expected[assignment], assignment
