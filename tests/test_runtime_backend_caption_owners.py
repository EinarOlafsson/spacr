"""Backend runtime captions enter catalogs only through their declared owners."""
from importlib import import_module
from pathlib import Path

import pytest


@pytest.fixture
def extract_backend_sources(tmp_path, monkeypatch):
    """Run the production extractor over a small source tree, without catalogs."""
    tools_dir = str(Path(__file__).resolve().parents[1] / "tools")
    monkeypatch.syspath_prepend(tools_dir)
    builder = import_module("build_i18n_catalogs")
    source_root = tmp_path / "spacr"
    (source_root / "qt").mkdir(parents=True)
    for filename in (
        "model_compare.py", "embeddings.py", "plot.py", "model_zoo.py",
        "install_cleanup.py", "settings.py", "object.py", "remote_execution.py",
        "attribution.py", "measure.py", "object_classifier.py", "core.py",
        "psf_pipeline.py",
        "_lineage_trees.py",
    ):
        (source_root / filename).write_text("", encoding="utf-8")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "_indirect_runtime_ui_sources", set)

    def extract(filename, source):
        (source_root / filename).write_text(source, encoding="utf-8")
        return set(builder.extract_static_ui_sources())

    return extract


def test_model_download_caption_is_extracted_from_shared_acquisition_owner(
    extract_backend_sources,
):
    sources = extract_backend_sources("model_zoo.py", '''
def _ensure_catalogue_model_file(entry, *, download=True):
    """Internal acquisition documentation is not a runtime caption."""
    raise ModelUnreadable(tr(
        "Automatic model downloads are disabled by SPACR_MODEL_DOWNLOAD. "
        "Copy verified model {model} (sha256 {sha256}) to {path}.",
        model=entry.key, sha256=entry.sha256, path=str(cached)))
''')
    assert sources == {
        "Automatic model downloads are disabled by SPACR_MODEL_DOWNLOAD. "
        "Copy verified model {model} (sha256 {sha256}) to {path}."
    }


def test_model_download_scope_excludes_other_functions_and_metadata(
    extract_backend_sources,
):
    sources = extract_backend_sources("model_zoo.py", '''
METADATA = tr("This metadata is outside the runtime owner.")
def _ensure_model_file(requested, *, download=True):
    return tr("The former wrapper must not become an extra runtime owner.")
def unrelated_model_backend():
    return tr("Other backend diagnostics remain outside this extraction scope.")
class DifferentOwner:
    def _ensure_catalogue_model_file(self):
        return tr("An unrelated method with the same name must not be scanned.")
def _ensure_catalogue_model_file(entry, *, download=True):
    raise RuntimeError("An untranslated backend detail must not be inventoried.")
''')
    assert sources == set()


def test_install_cleanup_cli_literals_enter_runtime_sources(
    extract_backend_sources,
):
    captions = (
        "remove only this recognized installation, preserving its data",
        "require an interpreter outside the selected root",
        "restore one unchanged retained private POSIX runtime",
        "previous desktop process that must have exited",
        "copy the private managed Python outside this installation",
        "Uninstall preparation stopped: {error}",
        "Runtime recovery stopped: {error}",
        "Uninstall requires a helper and interpreter outside the selected installation.",
        "Targeted removal requires one recognized canonical installation and cannot purge user data.",
    )
    source = 'def _main():\n'
    source += '    parser.add_argument("--private-action", dest="private_action")\n'
    source += ''.join(f'    tr({caption!r})\n' for caption in captions)
    source += '    raise RuntimeError("Internal cleanup detail is not translated.")\n'
    assert extract_backend_sources("install_cleanup.py", source) == set(captions)


@pytest.mark.parametrize('owner, indices', (
    ('_cf_preserved_channels', (0, 1)),
    ('_cf_preserve_reference_channels', (2, 3)),
    ('_counterfactual_sequences', (4, 5)),
    ('_counterfactual_report', (4, 5)),
))
def test_counterfactual_reference_diagnostics_reach_the_runtime_inventory(
    extract_backend_sources, owner, indices,
):
    captions = (
        "Preserved channels must be a list of valid zero-based whole channel indices.",
        "Select each preserved channel only once.",
        "Reference and edited crops must have matching floating NCHW shapes, dtype and device.",
        "Reference and edited crops must contain finite pixels.",
        "Reference preservation requires NCHW crop tensors.",
        "Reference crops must contain finite pixels.",
    )
    selected = {captions[index] for index in indices}
    source = f'def {owner}():\n' + ''.join(
        f'    tr({caption!r})\n' for caption in sorted(selected))
    assert extract_backend_sources('attribution.py', source) == selected


def test_counterfactual_scope_excludes_unrelated_and_untranslated_diagnostics(
    extract_backend_sources,
):
    sources = extract_backend_sources('attribution.py', '''
tr("A module-level diagnostic outside the reference-channel owners.")
def unrelated_attribution_method():
    return tr("An unrelated backend diagnostic must not enter the catalog.")
class OtherOwner:
    def _counterfactual_report(self):
        return tr("A method sharing an owner name must not enter the catalog.")
def _cf_preserved_channels():
    """Internal reference-channel documentation is not a runtime caption."""
    raise ValueError("An old non-translated diagnostic remains outside the inventory.")
''')
    assert sources == set()


def test_well_label_reader_captions_enter_only_through_its_owner(extract_backend_sources):
    captions = {
        "Cannot recover acquisition metadata from object key {key}.",
        "The well-label table assigns conflicting labels to {well}.",
        "Well label {well} is ambiguous across plates; use the full plate_row_column identifier.",
    }
    source = 'def _mil_frame_from_db():\n' + ''.join(
        f'    tr({caption!r})\n' for caption in sorted(captions))
    assert extract_backend_sources('embeddings.py', source) == captions


def test_existing_encoder_entry_captions_remain_inventoried(extract_backend_sources):
    caption = "No scorecard. Measure retrieval against labelled phenotype controls and attach the results."
    assert extract_backend_sources('embeddings.py',
        f'def encoder_entry():\n    return tr({caption!r})\n') == {caption}


def test_mil_bag_identity_caption_enters_only_through_its_owner(extract_backend_sources):
    caption = "Well identifier {well} spans different acquisition wells; use complete, unique well identifiers."
    assert extract_backend_sources('embeddings.py',
        f'def _mil_bags():\n    raise ValueError(tr({caption!r}, well="plate_1_A_1"))\n') == {caption}


def test_unmixing_selector_caption_enters_only_through_its_owner(extract_backend_sources):
    caption = "Spectral unmixing requires distinct nonnegative integer channel indices."
    assert extract_backend_sources('psf_pipeline.py',
        f'def _prepare_unmixing():\n    raise ValueError(tr({caption!r}))\n') == {caption}


def test_lineage_parent_identity_caption_enters_only_through_its_owner(extract_backend_sources):
    caption = "Track {track} has conflicting known parent IDs in the lineage table."
    assert extract_backend_sources('_lineage_trees.py',
        f'def _lineage_explicit_parents():\n    raise ValueError(tr({caption!r}, track="t1"))\n') == {caption}


def test_manual_viability_caption_enters_only_through_its_owner(extract_backend_sources):
    caption = "Manual viability thresholds must be finite numbers or automatic."
    assert extract_backend_sources('measure.py',
        f'def _viability_manual():\n    raise ValueError(tr({caption!r}))\n') == {caption}


@pytest.mark.parametrize('owner', ['_phases_by_xgboost', '_phases_by_torch'])
def test_tabular_phase_holdout_caption_is_extracted(extract_backend_sources, owner):
    caption = "The requested acquisition groups cannot place every class in both training and held-out data."
    assert extract_backend_sources('measure.py',
        f'def {owner}():\n    return tr({caption!r})\n') == {caption}


@pytest.mark.parametrize('filename,owner', [
    ('embeddings.py', '_mil_frame_from_db'),
    ('embeddings.py', '_mil_bags'),
    ('measure.py', '_phases_by_xgboost'),
    ('measure.py', '_phases_by_torch'),
    ('psf_pipeline.py', '_prepare_unmixing'),
    ('_lineage_trees.py', '_lineage_explicit_parents'),
    ('measure.py', '_viability_manual'),
    ('measure.py', '_measurement_channel_index'),
    ('measure.py', '_wound_calibration'),
])
def test_new_backend_owners_do_not_admit_unrelated_diagnostics(
        extract_backend_sources, filename, owner):
    source = f"""
tr("An unrelated module caption.")
def unrelated():
    return tr("An unrelated function caption.")
class Unrelated:
    def {owner}(self):
        return tr("An unrelated class method caption.")
def {owner}():
    raise ValueError("An untranslated internal diagnostic.")
"""
    assert extract_backend_sources(filename, source) == set()


@pytest.mark.parametrize('filename,owner,captions', [
    ('object_classifier.py', '_annotated_real_crops', (
        'Annotated crop {path} no longer matches its sample-creation fingerprint.',
        'Annotated crop {path} has conflicting labels or acquisition identities.',
    )),
    ('measure.py', '_wound_plane', (
        'Wound-plane index {index} must be a finite, nonnegative whole number smaller than {planes}.',
    )),
])
def test_annotated_crop_and_wound_reader_diagnostics_enter_declared_scopes(
        extract_backend_sources, filename, owner, captions):
    source = f'def {owner}():\n' + ''.join(
        f'    tr({caption!r})\n' for caption in captions)
    assert extract_backend_sources(filename, source) == set(captions)


@pytest.mark.parametrize('filename,owner', [
    ('object_classifier.py', '_annotated_real_crops'),
    ('measure.py', '_wound_plane'),
])
def test_reader_diagnostic_scope_excludes_unrelated_owners(
        extract_backend_sources, filename, owner):
    source = f"""
tr("Unrelated module caption.")
def unrelated():
    return tr("Unrelated function caption.")
class Unrelated:
    def {owner}(self):
        return tr("Unrelated method caption.")
def {owner}():
    return "Internal non-translated detail."
"""
    assert extract_backend_sources(filename, source) == set()


def test_feedback_position_validation_captions_reach_the_runtime_inventory(extract_backend_sources):
    captions = {
        'microscope_positions {path}: field {field} has a non-finite or invalid {axis} coordinate.',
        'microscope_positions {path} gives conflicting stage positions for field {field}.',
    }
    source = 'def _microscope_positions():\n' + ''.join(
        f'    tr({caption!r})\n' for caption in sorted(captions))
    assert extract_backend_sources('core.py', source) == captions


def test_feedback_position_scope_excludes_unrelated_core_diagnostics(extract_backend_sources):
    assert extract_backend_sources('core.py', """
tr('An unrelated module caption.')
def unrelated_watch_function():
    return tr('An unrelated watch caption.')
class Other:
    def _microscope_positions(self):
        return tr('An unrelated method caption.')
def _microscope_positions():
    raise ValueError('An existing untranslated internal diagnostic.')
""") == set()


def test_measurement_channel_identity_caption_enters_only_through_its_owner(extract_backend_sources):
    caption = "Measurement channels must be nonnegative integer indices."
    assert extract_backend_sources('measure.py',
        f'def _measurement_channel_index():\n    raise ValueError(tr({caption!r}))\n') == {caption}


def test_wound_calibration_caption_enters_only_through_its_owner(extract_backend_sources):
    caption = "Wound calibration {name} must be a finite number greater than zero, or blank."
    assert extract_backend_sources('measure.py',
        f'def _wound_calibration():\n    raise ValueError(tr({caption!r}, name="pixel_size"))\n') == {caption}
