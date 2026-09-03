"""Keep completed Instruction 368 modules fully documented.

The broader documentation checks cover public callables and nested helpers,
but not private module-level functions.  This zero-debt registry closes that
gap one finished module at a time without pretending the whole package is
already complete.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FULLY_DOCUMENTED = (
    "spacr/__main__.py",
    "spacr/accelerator.py",
    "spacr/active_learning.py",
    "spacr/agreement.py",
    "spacr/align.py",
    "spacr/api.py",
    "spacr/artifacts.py",
    "spacr/attribution_columns.py",
    "spacr/baseline.py",
    "spacr/batch.py",
    "spacr/benchmark.py",
    "spacr/annotation.py",
    "spacr/annotation_validation.py",
    "spacr/cancellation.py",
    "spacr/cell_montage.py",
    "spacr/chaining.py",
    "spacr/checkpoint.py",
    "spacr/classify_classes.py",
    "spacr/classifier_quality.py",
    "spacr/cli.py",
    "spacr/cli_database.py",
    "spacr/cli_leakage.py",
    "spacr/cli_download.py",
    "spacr/cli_plugins.py",
    "spacr/cli_repro.py",
    "spacr/cli_workspace.py",
    "spacr/column_groups.py",
    "spacr/columns.py",
    "spacr/confusion.py",
    "spacr/control_names.py",
    "spacr/convert.py",
    "spacr/counting.py",
    "spacr/crop_source.py",
    "spacr/curation.py",
    "spacr/custom_features.py",
    "spacr/database_concurrency.py",
    "spacr/database_schema.py",
    "spacr/diameter.py",
    "spacr/errors.py",
    "spacr/example_data.py",
    "spacr/external_masks.py",
    "spacr/feature_dict.py",
    "spacr/figures/distributions.py",
    "spacr/figures/fast_render.py",
    "spacr/figures/panels.py",
    "spacr/figures/scene.py",
    "spacr/figures/sheet.py",
    "spacr/figures/summary.py",
    "spacr/foreign.py",
    "spacr/frame_handoff.py",
    "spacr/flowview/_classify_stages.py",
    "spacr/flowview/classify_blueprint.py",
    "spacr/flowview/collector.py",
    "spacr/flowview/events.py",
    "spacr/flowview/feeder.py",
    "spacr/flowview/export.py",
    "spacr/flowview/layout.py",
    "spacr/flowview/model.py",
    "spacr/flowview/trace.py",
    "spacr/flowview/theme.py",
    "spacr/flowview/thumbs.py",
    "spacr/intensity_rescale.py",
    "spacr/gpu_reduce.py",
    "spacr/gate_library.py",
    "spacr/guide_concordance.py",
    "spacr/guide_permutation.py",
    "spacr/gene_measurement_compare.py",
    "spacr/gene_measurement_sweep.py",
    "spacr/hit_investigation.py",
    "spacr/image_colors.py",
    "spacr/gene_tile.py",
    "spacr/lineage.py",
    "spacr/logger.py",
    "spacr/logging_util.py",
    "spacr/localisation.py",
    "spacr/macro.py",
    "spacr/mask_io.py",
    "spacr/measure_hooks.py",
    "spacr/measurement_scan.py",
    "spacr/metadata_resolution.py",
    "spacr/merge_tables.py",
    "spacr/model_check.py",
    "spacr/mixed_gpu.py",
    "spacr/multi_database.py",
    "spacr/nonparametric_fits.py",
    "spacr/normalization.py",
    "spacr/notebook_export.py",
    "spacr/organelle_types.py",
    "spacr/openmp_guard.py",
    "spacr/outlier_filter.py",
    "spacr/omero.py",
    "spacr/parameter_sweep.py",
    "spacr/permutation_qc.py",
    "spacr/picture_settings.py",
    "spacr/plate_measurements.py",
    "spacr/plate_qc.py",
    "spacr/plot.py",
    "spacr/plugins.py",
    "spacr/png_list.py",
    "spacr/portable_paths.py",
    "spacr/predictions.py",
    "spacr/profiler.py",
    "spacr/projects.py",
    "spacr/qc_quarantine.py",
    "spacr/regression_failure.py",
    "spacr/regression_families.py",
    "spacr/regression_layout.py",
    "spacr/regression_panels.py",
    "spacr/regression_qc.py",
    "spacr/report.py",
    "spacr/resources/home/versions/_generators/common.py",
    "spacr/resources/home/versions/_generators/parts.py",
    "spacr/resources/home/versions/_generators/render.py",
    "spacr/resources/home/versions/_generators/variants.py",
    "spacr/resources/icons/backup_icons/_generators/group_trellis_gate_feature_napari.py",
    "spacr/restart_state.py",
    "spacr/row_exclusions.py",
    "spacr/roi.py",
    "spacr/rra.py",
    "spacr/run_recommendations.py",
    "spacr/run_journal.py",
    "spacr/runctx.py",
    "spacr/schema.py",
    "spacr/screen_data.py",
    "spacr/selection.py",
    "spacr/sequencing.py",
    "spacr/sequencing_qc.py",
    "spacr/settings_advisor.py",
    "spacr/style_base.py",
    "spacr/sweep_child.py",
    "spacr/sra.py",
    "spacr/surrogate.py",
    "spacr/thresholds.py",
    "spacr/tiff_io.py",
    "spacr/timelapse.py",
    "spacr/trial_metrics.py",
    "spacr/uniprot.py",
    "spacr/umap_annotations.py",
    "spacr/umap_search.py",
    "spacr/updater.py",
    "spacr/volcano_style.py",
    "spacr/well_scope.py",
    "spacr/well_spec.py",
    "spacr/workspace.py",
)


def _undocumented_functions(path: Path) -> list[str]:
    """Return every function or method in ``path`` without a docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        f"{node.name} (line {node.lineno})"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not ast.get_docstring(node)
    ]


@pytest.mark.parametrize("relative", FULLY_DOCUMENTED)
def test_every_function_in_a_finished_module_stays_documented(relative):
    """A finished phase-4 module may not regain an undocumented helper."""
    missing = _undocumented_functions(ROOT / relative)
    assert not missing, (
        f"{relative} was fully documented and gained undocumented functions: "
        f"{missing}"
    )


def test_the_scanner_detects_a_removed_docstring(tmp_path):
    """The guard observes both the documented and regressed source shapes."""
    source = tmp_path / "documented.py"
    source.write_text(
        'def helper(value):\n    """Return the supplied value."""\n'
        "    return value\n",
        encoding="utf-8",
    )
    assert _undocumented_functions(source) == []

    source.write_text(
        "def helper(value):\n    return value\n",
        encoding="utf-8",
    )
    assert _undocumented_functions(source) == ["helper (line 1)"]
