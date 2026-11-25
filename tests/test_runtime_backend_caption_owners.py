"""Backend runtime captions enter catalogs only through their declared owners."""
import ast
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
    (source_root / "figures").mkdir()
    for filename in (
        "model_compare.py", "embeddings.py", "plot.py", "model_zoo.py",
        "install_cleanup.py", "data_manager.py", "doctor.py", "updater.py",
        "settings.py", "object.py", "_mask_workers.py", "_mask_array_batches.py",
        "remote_execution.py",
        "attribution.py", "measure.py", "object_classifier.py", "core.py",
        "psf_pipeline.py", "pipeline_v2.py", "convert.py", "ome_zarr_convert.py",
        "_lineage_trees.py",
        "utils.py", "channel_sorting.py", "image_import.py", "run_journal.py", "ml.py",
        "regression_layout.py", "regression_failure.py", "regression_backends.py", "power_model.py", "io.py",
        "gene_measurement_compare.py", "ops_prefix.py", "ops_decode.py",
        "uniprot.py", "validate.py", "torch_artifacts.py",
        "classification_pixels.py", "crop_loader.py", "crop_store.py",
        "crops.py", "ops_engine.py", "ome_zarr.py", "image_store.py",
        "ome_zarr_registration.py",
        "ops_cell_cycle.py", "ops_mitotic.py", "ops_flatfield.py", "cpg_measure.py", "cli.py",
        "ops_sbs.py", "ops_run_infra.py",
        "figures/style.py", "figures/bundle.py",
    ):
        (source_root / filename).write_text("", encoding="utf-8")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "_indirect_runtime_ui_sources", set)

    def extract(filename, source):
        (source_root / filename).parent.mkdir(parents=True, exist_ok=True)
        (source_root / filename).write_text(source, encoding="utf-8")
        return set(builder.extract_static_ui_sources())

    return extract


def test_independent_import_identity_refusal_enters_its_declared_owner(
        extract_backend_sources):
    caption = "Independent-file import identities collide; no images can be imported."
    source = f'class ImportPlan:\n    def problems(self):\n        return [tr({caption!r})]\n'
    assert extract_backend_sources('image_import.py', source) == {caption}


def test_explicit_cellpose_selection_refusal_enters_its_declared_owner(
        extract_backend_sources):
    caption = (
        "Cellpose model {name} is not an installed stock name, an existing checkpoint "
        "or an exact model-zoo entry. Choose a listed model or provide its checkpoint; "
        "stock weights were not substituted.")
    assert extract_backend_sources('utils.py',
        f'def _resolve_cellpose_pretrained(name):\n    raise ValueError(tr({caption!r}, name=name))\n') == {caption}


def test_feature_exclusion_diagnostics_have_one_exact_runtime_owner(extract_backend_sources):
    path = Path(__file__).resolve().parents[1] / "spacr" / "utils.py"
    source = path.read_text(encoding="utf-8")
    owner = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "filter_dataframe_features")
    isolated = ast.get_source_segment(source, owner) + '''
tr("Unrelated utility metadata.")
def unrelated():
    def filter_dataframe_features():
        return tr("Nested same-name helper.")
class Unrelated:
    def filter_dataframe_features(self):
        return tr("Same-name method.")
'''
    assert extract_backend_sources("utils.py", isolated) == {
        "Invalid feature-exclusion regular expression {pattern!r}: {error}",
        "Feature exclusion {pattern!r}: {count} matching column(s).",
        "Excluded columns for {pattern!r}: {columns}",
        "Feature exclusion {pattern!r} matched no columns.",
    }


def test_cellpose_selection_owner_excludes_other_utility_scopes(
        extract_backend_sources):
    source = '''
METADATA = tr("Unrelated utility metadata.")
def unrelated():
    def _resolve_cellpose_pretrained():
        return tr("A nested same-name helper is outside the runtime owner.")
    return tr("An unrelated utility caption.")
class Unrelated:
    def _resolve_cellpose_pretrained(self):
        return tr("An unrelated same-name method.")
def _unrelated_choose_model():
    return tr("A different utility helper must remain outside this scope.")
def _resolve_cellpose_pretrained(name):
    """Unrelated owner documentation is not a caption."""
    return "An untranslated implementation detail."
'''
    assert extract_backend_sources('utils.py', source) == set()


def test_offline_cellpose_stock_refusal_enters_constructor_owner(extract_backend_sources):
    caption = (
        "Automatic model downloads are disabled by SPACR_MODEL_DOWNLOAD. "
        "Cellpose stock model {model} is not cached at {path}. Provide its existing "
        "checkpoint or download it before running offline.")
    assert extract_backend_sources('utils.py',
        f'def _choose_model():\n    raise ValueError(tr({caption!r}))\n') == {caption}


def test_offline_constructor_owner_excludes_same_named_nested_and_class_helpers(
        extract_backend_sources):
    assert extract_backend_sources('utils.py', '''
tr("Unrelated utility metadata.")
def unrelated():
    def _choose_model():
        return tr("Nested same-name utility helper.")
class Unrelated:
    def _choose_model(self):
        return tr("Same-name method outside constructor owner.")
def _choose_model():
    """Constructor documentation is not a runtime caption."""
    return "Untranslated internal constructor detail."
''') == set()


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


@pytest.mark.parametrize("owner,captions", [
    ("_curated_scorecard_rows", {
        "F1 @ IoU 0.5", "Boundary F1 (2 px)", "Precision", "Recall", "AJI", "Dice",
    }),
    ("_curated_scorecard_lines", {
        "Evaluation: {dataset}; {fields} fields, {objects} objects; "
        "boundary tolerance {pixels} px; stock {stock}.",
        "{metric}: {value:.4f}",
        "{metric}: {value:.4f} (stock {stock:.4f}, difference {delta:+.4f})",
    }),
])
def test_curated_scorecard_captions_enter_only_through_declared_owners(
        extract_backend_sources, owner, captions):
    """Keep prose and formats, while the existing identifier gate preserves AJI."""
    source = f"def {owner}():\n" + "".join(
        f"    tr({caption!r})\n" for caption in sorted(captions))
    assert extract_backend_sources("model_zoo.py", source) == captions - {"AJI"}


@pytest.mark.parametrize("owner", ["_curated_scorecard_rows", "_curated_scorecard_lines"])
def test_curated_scorecard_scope_excludes_unrelated_and_nested_owner_names(
        extract_backend_sources, owner):
    source = f'''
tr("Unrelated model metadata.")
def unrelated():
    def {owner}():
        return tr("A nested same-name function must remain outside this scope.")
    return tr("An unrelated backend diagnostic.")
class Unrelated:
    def {owner}(self):
        return tr("An unrelated same-name method.")
def {owner}():
    """A scorecard helper docstring is not a runtime caption."""
    return "An untranslated implementation detail."
'''
    assert extract_backend_sources("model_zoo.py", source) == set()


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



@pytest.mark.parametrize('owner,caption', [
    ('_cf_structure_weight', 'The counterfactual structure weight must be finite and nonnegative.'),
    ('_cf_diffusion_strength', 'The counterfactual diffusion strength must be between zero and one.'),
    ('_cf_structure_loss', 'Structure preservation requires matching finite floating-point NCHW tensors.'),
])
def test_diffusion_structure_diagnostics_enter_exact_owners(
        extract_backend_sources, owner, caption):
    assert extract_backend_sources('attribution.py',
        f'def {owner}():\n    raise ValueError(tr({caption!r}))\n') == {caption}


@pytest.mark.parametrize('owner', [
    '_cf_structure_weight', '_cf_diffusion_strength', '_cf_structure_loss',
])
def test_diffusion_structure_owner_excludes_unrelated_and_nested_scopes(
        extract_backend_sources, owner):
    source = f'''
tr("An unrelated module diagnostic.")
def unrelated():
    def {owner}():
        return tr("An unrelated nested same-name function.")
    return tr("An unrelated backend diagnostic.")
class Unrelated:
    def {owner}(self):
        return tr("An unrelated same-name method.")
def {owner}():
    """Internal diffusion documentation is not a runtime caption."""
    return "An untranslated implementation detail."
'''
    assert extract_backend_sources('attribution.py', source) == set()


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


def test_current_embedding_identity_refusals_enter_only_through_their_owner(extract_backend_sources):
    captions = {
        "The current embedding has no unique source object keys.",
        "Some current embedding objects no longer have crop metadata.",
    }
    source = 'def _mil_frame_from_result():\n' + ''.join(
        f'    raise ValueError(tr({caption!r}))\n' for caption in sorted(captions))
    assert extract_backend_sources('embeddings.py', source) == captions


def test_current_embedding_owner_does_not_admit_unrelated_or_nested_captions(extract_backend_sources):
    source = '\n'.join([
        'tr("Unrelated module prose remains outside the catalog.")',
        'def unrelated():',
        '    return tr("Unrelated callable prose remains outside the catalog.")',
        'class Foreign:',
        '    def _mil_frame_from_result(self):',
        '        return tr("An owner method in another class remains outside the catalog.")',
        'def outer():',
        '    def _mil_frame_from_result():',
        '        return tr("An unrelated nested owner remains outside the catalog.")',
    ])
    assert extract_backend_sources('embeddings.py', source) == set()


def test_existing_encoder_entry_captions_remain_inventoried(extract_backend_sources):
    caption = "No scorecard. Measure retrieval against labelled phenotype controls and attach the results."
    assert extract_backend_sources('embeddings.py',
        f'def encoder_entry():\n    return tr({caption!r})\n') == {caption}


def test_foundation_comparison_input_caption_enters_its_declared_owner(
        extract_backend_sources):
    caption = "Provide a four-dimensional crop stack and one label per crop."
    assert extract_backend_sources('embeddings.py',
        f'def _backbone_scorecards():\n    raise EmbeddingError(tr({caption!r}))\n') == {caption}


def test_foundation_comparison_owner_does_not_admit_other_scopes(
        extract_backend_sources):
    source = '''
tr("An unrelated module caption.")
def unrelated():
    def _backbone_scorecards():
        return tr("An unrelated nested same-name function.")
    return tr("An unrelated function caption.")
class Unrelated:
    def _backbone_scorecards(self):
        return tr("An unrelated class method caption.")
def _backbone_scorecards():
    """Internal comparison documentation is not a runtime caption."""
    return "An untranslated implementation detail."
'''
    assert extract_backend_sources('embeddings.py', source) == set()


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


def test_shared_stock_preflight_literal_caption_is_owned(extract_backend_sources):
    caption = "Cellpose stock model {model} is not cached at {path}."
    assert extract_backend_sources("utils.py",
        f"def _cellpose_stock_preflight():\n    raise ValueError(tr({caption!r}))\n") == {caption}


def test_shared_stock_preflight_excludes_nonliteral_nested_and_class_helpers(
        extract_backend_sources):
    assert extract_backend_sources("utils.py", '''
tr("Unrelated utility metadata.")
def unrelated():
    def _cellpose_stock_preflight():
        return tr("Nested same-name helper.")
class Unrelated:
    def _cellpose_stock_preflight(self):
        return tr("Same-name method.")
def _cellpose_stock_preflight():
    """Internal documentation is not a caption."""
    return "Internal untranslated detail."
''') == set()


@pytest.mark.parametrize("owner", ["_dino_classifier_from_config", "_dino_classifier_from_checkpoint", "_dino_finetune_classifier", "_dino_classifier_apply"])
def test_own_classifier_diagnostics_have_exact_top_level_owners(extract_backend_sources, owner):
    caption = "Own-DINO classifier settings, checkpoint or input are incompatible."
    assert extract_backend_sources("embeddings.py",
        f"def {owner}():\n    def refuse():\n        raise ValueError(tr({caption!r}))\n") == {caption}


def test_classifier_diagnostic_ownership_excludes_unrelated_nested_and_methods(extract_backend_sources):
    assert extract_backend_sources("embeddings.py", '''
def unrelated():
    def _dino_classifier_apply():
        return tr("Unrelated nested diagnostic.")
class Unrelated:
    def _dino_classifier_apply(self):
        return tr("Unrelated method diagnostic.")
def _dino_classifier_apply():
    """Internal documentation is not a caption."""
    return "Unwrapped internal detail."
''') == set()


def test_zoo_source_tooltip_owner_extracts_values_and_literal_heading(extract_backend_sources):
    assert extract_backend_sources("qt/widgets/model_zoo_picker.py", '''
_SOURCE_TOOLTIPS = {
    "Untranslated private source key": "Models trained by the spaCR maintainer and users.",
    "Other private source key": "Cellpose models published on bioimage.io, read through its artifact API.",
}
_IGNORED = {"hidden": "Unrelated internal module metadata."}
def build():
    set_translatable_text(widget, "spaCR community")
''') == {"Models trained by the spaCR maintainer and users.",
           "Cellpose models published on bioimage.io, read through its artifact API.",
           "spaCR community"}


def test_zoo_source_tooltip_owner_excludes_other_modules_and_local_shadow(extract_backend_sources):
    assert extract_backend_sources("qt/widgets/unrelated_picker.py", '''
_SOURCE_TOOLTIPS = {"key": "Unrelated source tooltip metadata."}
''') == set()
    assert extract_backend_sources("qt/widgets/model_zoo_picker.py", '''
def unrelated():
    _SOURCE_TOOLTIPS = {"key": "Local source tooltip metadata."}
_SOURCE_TOOLTIPS = {"private": {"nested": "Nested internal metadata."}}
''') == set()


def test_field_table_progress_has_exact_top_level_owner(extract_backend_sources):
    caption = "Measuring field {field} of {total}."
    assert extract_backend_sources("measure.py",
        f"def measure_from_field_table():\n    report(tr({caption!r}))\n") == {caption}


def test_field_table_progress_excludes_same_named_methods_and_nested_helpers(extract_backend_sources):
    assert extract_backend_sources("measure.py", '''
def unrelated():
    def measure_from_field_table():
        return tr("Unrelated nested progress.")
class Unrelated:
    def measure_from_field_table(self):
        return tr("Unrelated method progress.")
def measure_from_field_table():
    """Documentation is not progress output."""
    return "Unwrapped internal detail."
''') == set()


def test_pinned_diffusion_refusal_has_exact_backend_caption_owner(extract_backend_sources):
    caption = "Choose a compatible local diffusion checkpoint and its SHA-256 digest."
    assert extract_backend_sources("attribution.py",
        f"def _cf_checkpoint_error():\n    raise ValueError(tr({caption!r}))\n") == {caption}


def test_pinned_diffusion_refusal_excludes_nested_and_method_names(extract_backend_sources):
    assert extract_backend_sources("attribution.py", '''
def unrelated():
    def _cf_checkpoint_error():
        return tr("Unrelated nested checkpoint error.")
class Unrelated:
    def _cf_checkpoint_error(self):
        return tr("Unrelated checkpoint method.")
def _cf_checkpoint_error():
    """Internal prose is not a caption."""
    return "Unwrapped internal error."
''') == set()


def test_model_resolution_context_has_its_exact_runtime_owner(extract_backend_sources):
    caption = (
        "Model setting {setting}: no existing checkpoint or exact model-zoo identity "
        "matches {requested}. Available compatible zoo identities: {candidates}.")
    source = f'''def _model_resolution_context(requested, *, setting, kinds=()):
    return tr({caption!r}, setting=setting, requested=repr(requested), candidates="[]")
'''
    assert extract_backend_sources('model_zoo.py', source) == {caption}


def test_resolution_context_owner_does_not_expand_to_other_backend_text(extract_backend_sources):
    source = '''
METADATA = tr("Unrelated model metadata is not a UI diagnostic.")
def another_resolver():
    def _model_resolution_context():
        return tr("A same-named nested helper is not the declared owner.")
class OtherResolver:
    def _model_resolution_context(self):
        return tr("A same-named method is not the declared owner.")
def _model_resolution_context(requested, *, setting):
    """Runtime context owner documentation is not a UI caption."""
    return "An unwrapped internal detail is not a UI caption."
'''
    assert extract_backend_sources('model_zoo.py', source) == set()


@pytest.mark.parametrize("source,caption", [
    ('def build_plan():\n    return tr("No complete image sets are available.")\n',
     "No complete image sets are available."),
    ('def _few():\n    return tr("{shown} and {more} more")\n',
     "{shown} and {more} more"),
    ('class SetReport:\n    def summary(self):\n        return tr("{sets} complete set(s).")\n',
     "{sets} complete set(s)."),
])
def test_organizer_report_captions_have_exact_backend_owners(
        extract_backend_sources, source, caption):
    assert extract_backend_sources("channel_sorting.py", source) == {caption}


def test_organizer_report_owners_exclude_metadata_other_methods_and_nested_shadows(
        extract_backend_sources):
    assert extract_backend_sources("channel_sorting.py", '''
METADATA = tr("Unrelated channel metadata.")
def unrelated():
    def build_plan():
        return tr("A nested same-name plan is not an organizer diagnostic.")
    class SetReport:
        def summary(self):
            return tr("A nested same-name report is not displayed.")
class OtherReport:
    def summary(self):
        return tr("An unrelated report is not displayed.")
class SetReport:
    label = tr("A report class constant is not a status message.")
    def unrelated(self):
        return tr("An unrelated report method is not displayed.")
    def summary(self):
        """Documentation is not a status caption."""
        return "Unwrapped internal detail."
''') == set()


@pytest.mark.parametrize("filename,owner,caption", [
    ('regression_layout.py', 'normalise_count_table_layout', '{error}\nCheck the Regression input column settings: count_grna_column={guide_column!r} must name the guide identifier, and count_value_column={count_column!r} must name the numeric read-count column for each guide in each well, not a guide name or a row, column or plate identifier. Available columns after reading (up to 20): {available}. Select the actual read-count header for long input; do not switch a table with guide identifiers to wide layout to bypass this check.'),
    ('regression_failure.py', '_remedy', "In Regression, check count_grna_column and count_value_column against the count table's available columns. The guide column contains guide identifiers; the value column must contain numeric read counts, not guide names or well, row, column or plate identifiers. For long input, choose the actual read-count header and keep the long layout; do not use wide layout to hide a missing long-table column."),
])
def test_regression_count_guidance_enters_only_registered_backend_files(
        extract_backend_sources, filename, owner, caption):
    source = f"def {owner}():\n    return tr({caption!r})\n"
    assert extract_backend_sources(filename, source) == {caption}
    assert extract_backend_sources(filename, "") == set()
    assert extract_backend_sources("unregistered_backend.py", source) == set()


def test_reset_shortcut_action_has_exact_top_level_owner(extract_backend_sources):
    caption = "Restore both interface scales from the reset shortcut."
    source = f'''
SHORTCUTS: List[ShortcutSpec] = [
    ShortcutSpec("Ctrl+Alt+0", {caption!r}, "Unrelated category caption."),
    ShortcutSpec("Ctrl+0", "Another shortcut action stays outside this owner."),
]
'''
    assert extract_backend_sources("qt/shortcuts.py", source) == {caption}


def test_reset_shortcut_owner_excludes_nested_and_nonliteral_declarations(
        extract_backend_sources):
    source = '''
ACTION = "An indirect reset label is not this literal owner."
OTHER: List[ShortcutSpec] = [ShortcutSpec("Ctrl+Alt+0", "A different table.")]
def unrelated():
    SHORTCUTS: List[ShortcutSpec] = [ShortcutSpec("Ctrl+Alt+0", "A nested table.")]
class Unrelated:
    SHORTCUTS: List[ShortcutSpec] = [ShortcutSpec("Ctrl+Alt+0", "A class table.")]
SHORTCUTS: List[ShortcutSpec] = [
    ShortcutSpec("Ctrl+0", "A different key."),
    ShortcutSpec("Ctrl+Alt+0", ACTION),
    ShortcutSpec("Ctrl+Alt+0", f"A dynamic label {ACTION}."),
    OtherShortcutSpec("Ctrl+Alt+0", "A different constructor."),
    other.ShortcutSpec("Ctrl+Alt+0", "An attribute constructor."),
]
SHORTCUTS = [ShortcutSpec("Ctrl+Alt+0", "An unannotated table.")]
'''
    assert extract_backend_sources("qt/shortcuts.py", source) == set()


def test_reset_shortcut_owner_excludes_other_files(extract_backend_sources):
    source = '''
SHORTCUTS: List[ShortcutSpec] = [
    ShortcutSpec("Ctrl+Alt+0", "A same-name table in an unrelated module."),
]
'''
    assert extract_backend_sources("qt/unrelated_shortcuts.py", source) == set()
    assert extract_backend_sources("qt/widgets/shortcuts.py", source) == set()


def test_pruning_failure_caption_has_exact_backend_owner(extract_backend_sources):
    caption = "Eligibility checks could not be completed; keep this artifact."
    source = f'''
def is_prunable(path, registry):
    try:
        return _regenerable_reason(path, registry)
    except Exception:
        from .qt.i18n import tr
        return tr({caption!r})
'''
    assert extract_backend_sources("data_manager.py", source) == {caption}
    assert extract_backend_sources("data_manager.py", "") == set()
    assert extract_backend_sources("unregistered_backend.py", source) == set()


def test_pruning_owner_excludes_unrelated_and_nested_scopes(extract_backend_sources):
    source = '''
METADATA = tr("A data registry constant is not this pruning status.")
def unrelated():
    def is_prunable():
        return tr("A nested same-name helper is not this status.")
    return tr("An unrelated data-management caption.")
class Other:
    def is_prunable(self):
        return tr("A same-name method is not this status.")
def is_prunable(path, registry):
    """Owner documentation does not become a caption."""
    return "An internal unwrapped reason."
'''
    assert extract_backend_sources("data_manager.py", source) == set()


def test_channel_provenance_refusals_have_only_two_declared_io_owners(extract_backend_sources):
    source = '\n'.join((
        'METADATA = tr("Unrelated I/O metadata.")',
        'def _validate_normalized_source_channels(path, expected):',
        '    raise ValueError(tr("The normalized channel order cannot be verified."))',
        'def _resume_normalized_archives(src, settings=None):',
        '    raise ValueError(tr("Restore the original image stacks."))',
        'def unrelated():',
        '    def _validate_normalized_source_channels():',
        '        return tr("A nested same-name helper is not an owner.")',
        '    return tr("An unrelated I/O caption.")',
        'class Unrelated:',
        '    def _resume_normalized_archives(self):',
        '        return tr("A same-name method is not an owner.")',
    ))
    assert extract_backend_sources('io.py', source) == {
        'The normalized channel order cannot be verified.',
        'Restore the original image stacks.',
    }


def test_busy_uniprot_notice_enters_only_its_annotation_owner(extract_backend_sources):
    caption = ("UniProt is busy or unreachable, try again later. "
               "Results for {name} are not annotated.")
    source = (f"def annotation_for(text):\n    return None, tr({caption!r}, name=text)\n"
              "def _fetch(url):\n    return tr('An internal request detail.')\n")
    assert extract_backend_sources('uniprot.py', source) == {caption}
    real = Path(__file__).resolve().parents[1] / "spacr" / "uniprot.py"
    assert caption in extract_backend_sources('uniprot.py', real.read_text(encoding="utf-8"))


def test_regression_preflight_findings_enter_only_their_validate_owners(extract_backend_sources):
    """The two regression preflight checks own their captions; other checks do not."""
    caption = "Regression needs at least one score table and one count table."
    source = (f"def _check_regression_inputs(settings):\n    return tr({caption!r})\n"
              "def _check_other(settings):\n    return tr('Not a regression finding')\n")
    assert extract_backend_sources('validate.py', source) == {caption}
    real = Path(__file__).resolve().parents[1] / "spacr" / "validate.py"
    assert caption in extract_backend_sources('validate.py', real.read_text(encoding="utf-8"))


def test_old_custom_classifier_refusals_enter_their_owners(extract_backend_sources):
    """The old-head refusals of torch_artifacts and the channel refusal of the classifier are catalogued."""
    root = Path(__file__).resolve().parents[1] / "spacr"
    artifacts = extract_backend_sources(
        'torch_artifacts.py', (root / "torch_artifacts.py").read_text(encoding="utf-8"))
    assert any(text.startswith("This model was trained with the old spaCR custom classifier")
               for text in artifacts)
    assert any(text.startswith("This custom classifier checkpoint records") for text in artifacts)
    utils = extract_backend_sources('utils.py', (root / "utils.py").read_text(encoding="utf-8"))
    assert any(text.startswith("This classifier was trained on {expected}-channel") for text in utils)


def test_ops_channel_refusal_enters_only_the_shared_input_owner(extract_backend_sources):
    """Keep the moved GUI refusal admitted without scanning other OPS scopes."""
    path = Path(__file__).resolve().parents[1] / "spacr" / "ops_engine.py"
    text = path.read_text(encoding="utf-8")
    owner = next(node for node in ast.parse(text).body
                 if isinstance(node, ast.FunctionDef) and node.name == "_ph_segmentation_inputs")
    source = ast.get_source_segment(text, owner) + """
tr("Unrelated OPS module metadata.")
def unrelated():
    def _ph_segmentation_inputs():
        return tr("A nested same-name helper is outside the owner.")
    return tr("An unrelated OPS caption.")
class Other:
    def _ph_segmentation_inputs(self):
        return tr("A same-name method is outside the owner.")
"""
    assert extract_backend_sources("ops_engine.py", source) == {
        "ops_ph_channels {channels} must name {nuclear} and the cell channel {cell}"
    }


def test_cpg_painting_frame_messages_resolve_module_constants(extract_backend_sources):
    """All eight real painting-frame messages reach the normal static extractor."""
    path = Path(__file__).resolve().parents[1] / "spacr" / "cpg_measure.py"
    source = path.read_text(encoding="utf-8")
    names = {"_NO_RECORD", "_GUARD", "_PAINTING_MASKS", "_UNKNOWN_MODE",
             "_UNKNOWN_FRAME", "_NO_MASKS", "_SHAPES", "_NO_DESTINATION"}
    captions = {node.value.value for node in ast.parse(source).body
                if isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in names}
    assert len(captions) == 8
    assert extract_backend_sources("cpg_measure.py", source) == captions


def test_cpg_painting_frame_messages_keep_exact_owners(extract_backend_sources):
    """Other scopes, constants and presentation calls cannot add captions."""
    source = '''
_NO_RECORD = "The assigned missing shift message."
_GUARD = "A constant from another owner."
_OTHER = "Unrelated module metadata."
tr(_OTHER)
def site_painting_shift():
    tr(_NO_RECORD)
    tr(_GUARD)
    tr(_OTHER)
    tr("An unrelated literal.")
    ui.tr(_OTHER)
    def nested():
        tr(_OTHER)
    return lambda: tr(_OTHER)
def unrelated():
    tr(_GUARD)
    def painting_frame_mode():
        tr(_GUARD)
class Unrelated:
    def painting_frame_mode(self):
        tr(_GUARD)
'''
    assert extract_backend_sources("cpg_measure.py", source) == {
        "The assigned missing shift message."}
    nested = '''
_NO_RECORD = "The nested missing shift message."
def site_painting_shift():
    ui.tr(_NO_RECORD)
    def nested():
        tr(_NO_RECORD)
    class Nested:
        def message(self):
            tr(_NO_RECORD)
    return lambda: tr(_NO_RECORD)
'''
    assert extract_backend_sources("cpg_measure.py", nested) == set()
    assert extract_backend_sources("cpg_measure.py", "") == set()
    assert extract_backend_sources("unregistered.py", source) == set()


@pytest.mark.parametrize("binding", [
    "_NO_RECORD = 'A local replacement.'",
    "from elsewhere import _NO_RECORD",
    "def _NO_RECORD():\n        return 'A local function.'",
])
def test_cpg_painting_frame_constants_cannot_be_shadowed(
        extract_backend_sources, binding):
    """A local binding cannot be mistaken for the module's source constant."""
    source = ("_NO_RECORD = 'The module missing shift message.'\n"
              "def site_painting_shift():\n"
              f"    {binding}\n"
              "    tr(_NO_RECORD)\n")
    assert extract_backend_sources("cpg_measure.py", source) == set()
    assert extract_backend_sources("cpg_measure.py",
        "_NO_RECORD = 'The module missing shift message.'\n"
        "def site_painting_shift(_NO_RECORD):\n"
        "    tr(_NO_RECORD)\n") == set()


def test_cell_cycle_engine_refusals_have_exact_scopes(extract_backend_sources):
    """Only assigned literal messages in the two engine owners enter the catalog."""
    captions = (
        "ops_mitotic_source cell_cycle needs the ops_cell_cycle table; "
        "set ops_cell_cycle or run the ph_cell_cycle phase first",
        "ops_mitotic_source must be rule, classifier or cell_cycle; got {value}",
    )
    source = "\n".join((
        f"METADATA = tr({captions[0]!r})",
        "def _ph_aggregate():",
        "    if True:",
        f"        raise ValueError(tr({captions[0]!r}))",
        "    tr('Unrelated aggregate refusal.')",
        "    tr('ops_mitotic_source must be excluded from this owner')",
        "    def nested():",
        "        return tr('ops_mitotic_source cell_cycle needs nested exclusion')",
        "    async def nested_async():",
        "        return tr('ops_mitotic_source cell_cycle needs async exclusion')",
        "    local = lambda: tr('ops_mitotic_source cell_cycle needs lambda exclusion')",
        "    class Local:",
        "        note = tr('ops_mitotic_source cell_cycle needs class exclusion')",
        "def _ph_profile_options():",
        f"    raise ValueError(tr({captions[1]!r}, value='other'))",
        "    tr('Unrelated profile refusal.')",
        "    tr('ops_mitotic_source cell_cycle needs the other owner')",
        "    tr(dynamic_caption)",
        "    tr('ops_mitotic_source must be ' + dynamic_value)",
        "    other.tr('ops_mitotic_source must be excluded from another receiver')",
        "def unrelated():",
        "    tr('ops_mitotic_source cell_cycle needs an excluded owner')",
        "    def _ph_aggregate():",
        "        return tr('ops_mitotic_source cell_cycle needs nested-owner exclusion')",
        "class Other:",
        "    def _ph_profile_options(self):",
        "        return tr('ops_mitotic_source must be excluded from a method')",
    ))
    assert extract_backend_sources('ops_engine.py', source) == set(captions)


def test_unknown_settings_notices_have_exact_cli_owners(extract_backend_sources):
    """Admit all three actual notices without claiming the reused Bad settings caption."""
    path = Path(__file__).resolve().parents[1] / "spacr" / "cli.py"
    source = path.read_text(encoding="utf-8")
    owners = {"_report_unknown_keys", "_unknown_setting_fix"}
    selected = [node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name in owners]
    assert {node.name for node in selected} == owners
    isolated = "\n".join(ast.get_source_segment(source, node) for node in selected)
    assert extract_backend_sources("cli.py", isolated) == {
        "Unknown setting '{key}' ignored; nearest known setting: '{nearest}'.",
        "Unknown setting '{key}' ignored.",
        "Remove it from the settings file, or check the spelling.",
    }


def test_unknown_settings_owner_excludes_other_cli_scopes(extract_backend_sources):
    """Keep help, nested scopes, other owners and nonliteral calls outside this inventory."""
    source = '''
tr("Unknown setting '{key}' ignored.")
def unrelated():
    def _report_unknown_keys():
        return tr("Unknown setting '{key}' ignored.")
class Other:
    def _unknown_setting_fix(self):
        return tr("Remove it from the settings file, or check the spelling.")
def _report_unknown_keys():
    """Owner documentation is not a runtime notice."""
    tr("Bad settings")
    tr("An unrelated CLI diagnostic.")
    parser.add_argument("--strict-settings", help="Unrelated argparse help.")
    def inner():
        return tr("Unknown setting '{key}' ignored.")
    nested = lambda: tr("Unknown setting '{key}' ignored.")
    return tr(f"Unknown setting '{key}' ignored.")
def _unknown_setting_fix():
    """The ordinary return and indirect calls are not literal tr notices."""
    other.tr("Remove it from the settings file, or check the spelling.")
    tr(caption)
    return "Remove it from the settings file, or check the spelling."
'''
    assert extract_backend_sources("cli.py", source) == set()


@pytest.mark.parametrize('filename,owner', [
    ('ops_sbs.py', 'never_barcodes'), ('ops_decode.py', 'never_scale'),
    ('ops_decode.py', 'calibrate_fdr_scale'), ('ops_decode.py', '_field_candidates'),
    ('ops_decode.py', '_write_calibration_report'), ('ops_decode.py', '_never_settings')])
def test_never_calibration_captions_have_exact_top_level_owners(extract_backend_sources, filename, owner):
    """Admit the shared controls without exposing unrelated, nested or same-named methods."""
    caption = 'A never-control run needs valid settings.'
    source = f"""raise RuntimeError('Backend source must not be imported')
tr('Unrelated module metadata')
def {owner}():
    return tr({caption!r})
def unrelated():
    def {owner}():
        return tr('Unrelated nested caption')
class Other:
    def {owner}(self):
        return tr('Unrelated method caption')
"""
    assert extract_backend_sources(filename, source) == {caption}



def test_phenotype_stager_queue_error_has_an_exact_method_owner(extract_backend_sources):
    """Collect the queue error while excluding other classes, functions, nested scopes and metadata."""
    caption = 'Phenotype field {site} is not in the staging queue; request a field scheduled for this run.'
    source = f"""raise RuntimeError('Backend caption extraction must not import the source')
tr('Unrelated module metadata')
def path():
    return tr('An unrelated top-level function')
class Other:
    def path(self):
        return tr('An unrelated same-named method')
class FieldStager:
    def path(self, site):
        def nested():
            return tr('An unrelated nested caption')
        raise ValueError(tr({caption!r}, site=site))
    def load(self, site):
        return tr('An unrelated staging method')
"""
    assert extract_backend_sources('ops_run_infra.py', source) == {caption}
    assert extract_backend_sources('ops_run_infra.py', source.replace('class FieldStager:', 'class Renamed:')) == set()


def test_actual_phenotype_stager_queue_template_enters_the_runtime_inventory(extract_backend_sources):
    """The real adjacent-literal template must reach the declared caption route without importing OPS."""
    source = (Path(__file__).resolve().parents[1] / 'spacr/ops_run_infra.py').read_text(encoding='utf-8')
    assert extract_backend_sources('ops_run_infra.py', source) == {
        'Phenotype field {site} is not in the staging queue; request a field scheduled for this run.'}
