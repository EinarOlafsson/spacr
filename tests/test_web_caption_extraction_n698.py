"""N698 owns explicit web captions without importing its inference service."""
from importlib import import_module
from pathlib import Path

import pytest


WEB_FILES = (
    "app.py", "detector.py", "examples.py", "inputs.py", "masks.py",
    "plaque.py", "registry.py", "results.py", "run.py", "service.py",
)


@pytest.fixture
def web_sources(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    builder = import_module("build_i18n_catalogs")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    directory = tmp_path / "tools" / "web_mask_generator"
    directory.mkdir(parents=True)

    def write(filename, source):
        (directory / filename).write_text(source, encoding="utf-8")

    return builder, write, tmp_path


@pytest.mark.parametrize("filename", WEB_FILES)
def test_each_declared_web_owner_extracts_literals_without_execution(web_sources, filename):
    builder, write, _root = web_sources
    caption = f"Visible caption owned by {filename}: {{count}}"
    write(filename, f'''
raise RuntimeError("This web file must never be imported by extraction.")
import nonexistent_web_model_dependency
def nested_owner():
    def report():
        return tr({caption!r}, count=3)
    return report
''')
    assert builder._web_mask_generator_ui_sources() == {caption}


def test_web_extractor_excludes_unowned_files_and_noncaption_data(web_sources):
    builder, write, root = web_sources
    write("not_an_owner.py", 'tr("An unowned sibling caption.")')
    (root / "tools" / "other.py").write_text('tr("An unrelated tool caption.")', encoding="utf-8")
    write("service.py", '''
"""An internal module description is not a web caption."""
METADATA = {"key": "A model fact is not a translated caption."}
FAMILY_LABELS = {"model_key": "A same-named table outside app.py."}
tr(METADATA["key"])
tr(f"A dynamically formatted identifier {model_id}")
some_object.tr("An unrelated translation method.")
raise ValueError("An unwrapped internal diagnostic.")
''')
    assert builder._web_mask_generator_ui_sources() == set()


@pytest.mark.parametrize("declaration", ["FAMILY_LABELS =", "FAMILY_LABELS: dict ="])
def test_only_module_family_display_values_are_dynamic_caption_sources(web_sources, declaration):
    builder, write, _root = web_sources
    write("app.py", declaration + ''' {
    "internal_storage_key": "Family display caption",
    "other_key": "Another family display caption",
    "nonliteral_key": name_from_elsewhere,
}
OTHER_METADATA = {"key": "Do not inventory model provenance."}
def nested():
    FAMILY_LABELS = {"key": "A nested same-name table is not the owner."}
tr(FAMILY_LABELS.get(current_family, current_family))
''')
    assert builder._web_mask_generator_ui_sources() == {
        "Family display caption", "Another family display caption",
    }


def test_web_markdown_heading_keeps_its_exact_source_and_global_filter(web_sources):
    builder, write, _root = web_sources
    write("app.py", '''
tr("# spaCR mask generator")
tr("# ")
tr("# 123")
tr("# /private/file")
tr("/private/file")
tr("https://example.invalid/model")
tr("QWidget { background-color: red; }")
tr("Adjacent " "literal {value}", value=3)
''')
    write("run.py", 'tr("# A heading outside the web page owner")')
    assert builder._web_mask_generator_ui_sources() == {
        "# spaCR mask generator", "Adjacent literal {value}",
    }
    assert builder._looks_translatable("# spaCR mask generator") is False


def test_web_sources_join_production_inventory_before_compact_filtering(web_sources, monkeypatch):
    builder, write, root = web_sources
    source_root = root / "spacr"
    (source_root / "qt").mkdir(parents=True)
    for filename in (
        "model_compare.py", "embeddings.py", "plot.py", "model_zoo.py", "convert.py",
        "install_cleanup.py", "data_manager.py", "doctor.py", "updater.py",
        "settings.py", "object.py", "remote_execution.py",
        "attribution.py", "measure.py", "channel_sorting.py", "utils.py",
        "object_classifier.py", "core.py", "pipeline_v2.py",
        "psf_pipeline.py", "image_import.py",
        "_lineage_trees.py", "run_journal.py", "ml.py",
        "regression_layout.py", "regression_failure.py", "regression_backends.py", "power_model.py",
    ):
        (source_root / filename).write_text("", encoding="utf-8")
    monkeypatch.setattr(builder, "_indirect_runtime_ui_sources", set)
    from spacr.qt import i18n
    monkeypatch.setitem(i18n._ROWS, "Compact web caption", ())
    monkeypatch.setitem(i18n._TERM_ROWS, "Term-owned web caption", ())
    write("app.py", '''
FAMILY_LABELS = {"storage_key": "Family label through production inventory"}
tr("Compact web caption")
tr("Term-owned web caption")
tr("Web caption through production inventory")
''')
    (source_root / "psf_pipeline.py").write_text('''
raise RuntimeError("Spectral caption collection must not import processing code")
def _prepare_unmixing():
    return tr("The spectral control settings are invalid.")
def _prepare_measure_unmixing():
    return tr("Calibrated measurement is not supported.")
def _load_unmix_calibration():
    return tr("The measured spectral calibration is invalid.")
class _UnmixCalibration:
    def apply(self):
        return tr("The calibrated image has no valid overlap.")
class _UnmixPlan:
    def apply(self):
        return tr("An explicit calibrated region is required.")
def unrelated_legacy_helper():
    return tr("This unrelated legacy caption must remain excluded.")
''', encoding="utf-8")
    assert set(builder.extract_static_ui_sources()) == {
        "Family label through production inventory",
        "Web caption through production inventory",
        "The spectral control settings are invalid.",
        "Calibrated measurement is not supported.",
        "The measured spectral calibration is invalid.",
        "The calibrated image has no valid overlap.",
        "An explicit calibrated region is required.",
    }


def test_no_web_source_directory_is_safe_for_small_source_fixtures(web_sources):
    builder, _write, root = web_sources
    (root / "tools" / "web_mask_generator").rmdir()
    assert builder._web_mask_generator_ui_sources() == set()


def test_web_sources_use_existing_canonical_whitespace_normalization(web_sources):
    builder, write, _root = web_sources
    write("results.py", 'tr("Result package: {name}\\n", name=name)')
    assert builder._web_mask_generator_ui_sources() == {"Result package: {name}"}
