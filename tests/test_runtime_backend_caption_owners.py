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
        "install_cleanup.py", "doctor.py", "updater.py",
        "settings.py", "object.py", "remote_execution.py",
        "attribution.py", "measure.py", "object_classifier.py", "core.py",
        "psf_pipeline.py",
        "_lineage_trees.py",
        "utils.py", "channel_sorting.py", "run_journal.py", "ml.py",
        "regression_layout.py", "regression_failure.py",
    ):
        (source_root / filename).write_text("", encoding="utf-8")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "_indirect_runtime_ui_sources", set)

    def extract(filename, source):
        (source_root / filename).parent.mkdir(parents=True, exist_ok=True)
        (source_root / filename).write_text(source, encoding="utf-8")
        return set(builder.extract_static_ui_sources())

    return extract


def test_explicit_cellpose_selection_refusal_enters_its_declared_owner(
        extract_backend_sources):
    caption = (
        "Cellpose model {name} is not an installed stock name, an existing checkpoint "
        "or an exact model-zoo entry. Choose a listed model or provide its checkpoint; "
        "stock weights were not substituted.")
    assert extract_backend_sources('utils.py',
        f'def _resolve_cellpose_pretrained(name):\n    raise ValueError(tr({caption!r}, name=name))\n') == {caption}


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
