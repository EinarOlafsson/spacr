"""Focused contracts for the source-only AutoAPI localization extractor."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

builder = importlib.import_module("build_documentation_i18n")


# Review-set fingerprints make the coverage contract independent of the
# investigator's /tmp report while still proving that all exact audited ids
# remain represented.  A source/API change requires regenerating and reviewing
# that report before deliberately updating either digest.
_NEW_VISIBLE_DIGEST = (
    "0aa19275ff594eb3441be42e42ad6a1277c09c16c3a9514dc392d635b47bfb6f"
)
_ALIASES_DIGEST = (
    "5167459a662cc68d3de274d216297020ba159155bad4e9e8af8e751e69cdba66"
)


def _sha256_lines(lines) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def _source_nodes():
    """Yield public source nodes using the same package-name convention."""
    for path in sorted((ROOT / "spacr").rglob("*.py")):
        if any(
            part in {"tests", "__pycache__", "backup_icons", "i18n_catalogs"}
            for part in path.parts
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        module = builder._module_name(path)
        yield path, module, tree


def _visible_special_members() -> set[str]:
    keys: set[str] = set()
    for path, module, tree in _source_nodes():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    builder._is_visible_function_name(
                        node.name, module_is_package=path.name == "__init__.py",
                    )
                    and builder._clean_doc(node)
                    and node.name.startswith("__")
                ):
                    keys.add(f"{module}.{node.name}")
                continue
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            for child in node.body:
                if not isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    continue
                if (
                    child.name.startswith("__")
                    and builder._is_visible_function_name(
                        child.name, module_is_package=False,
                    )
                    and builder._clean_doc(child)
                ):
                    keys.add(f"{module}.{node.name}.{child.name}")
    return keys


def _visible_assignment_docs() -> set[str]:
    keys: set[str] = set()
    for _path, module, tree in _source_nodes():
        keys.update(builder._additional_assignment_docs(tree.body, module))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                owner = f"{module}.{node.name}"
                keys.update(builder._additional_assignment_docs(node.body, owner))
    return keys


def test_documented_parameter_names_match_the_source_signatures():
    """Every Sphinx ``:param:`` field names a real Python parameter."""
    field = re.compile(r":param\s+(?:[^:\s]+\s+)?([*\w]+)\s*:")
    checked = 0
    mismatches = []
    for path, _module, tree in _source_nodes():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = ast.get_docstring(node, clean=False) or ""
            parameters = {
                item.arg
                for item in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
            }
            if node.args.vararg is not None:
                parameters.add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                parameters.add(node.args.kwarg.arg)
            for match in field.finditer(docstring):
                checked += 1
                name = match.group(1).lstrip("*")
                if name not in parameters:
                    mismatches.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} "
                        f"{node.name} documents {name!r} but accepts "
                        f"{sorted(parameters)!r}"
                    )
    assert checked >= 6_000
    assert not mismatches, "\n".join(mismatches)


def test_public_docstrings_matches_reviewed_visible_coverage():
    docs = builder.public_docstrings()
    dunders = _visible_special_members()
    assignments = _visible_assignment_docs()

    # The 80 omissions in the audited pages are precisely the documented
    # special members and PEP-258/value attributes discovered from source.
    # 65 since 2026-08-18: `GenePanel.__del__` is documented because it is
    # the guard that stops Qt aborting the process when a panel is collected
    # with its warm-up thread still running -- a reader who deletes it needs
    # to know that, so it carries a docstring and therefore surface.
    assert len(dunders) == 65
    assert len(assignments) == 18
    assert _sha256_lines(
        [*(f"new_dunder\0{key}" for key in dunders),
         *(f"new_constant_attribute\0{key}" for key in assignments)]
    ) == _NEW_VISIBLE_DIGEST
    assert dunders | assignments <= docs.keys()

    # Freeze the complete public API surface together with its exact aliases.
    # Any intentional public docstring addition must update this count and
    # regenerate all target catalogs in the same change; otherwise localized
    # API pages would silently omit the new contract.
    # +40: the regression workbench — spacr.multiple_testing (every offered
    # correction), spacr.regression_diagnostics (design/residual/inference
    # panels), spacr.volcano_style (the shared renderer and its style object)
    # and the two Qt widgets plus the screen that host them. Their localized
    # catalogs are regenerated by tools/build_documentation_i18n.py.
    # +14: the shared metadata resolver and the two desktop picker surfaces.
    # +5: explicit regression input pairing, generated setting-applicability
    # rules, and the paired-file table/pairing proposal helpers.
    # +12/-5: the regression settings sweep moved from the non-visual
    # spacr.regression_search helper into the complete spacr.parameter_sweep
    # engine and its desktop screen. The admitted public entries are the
    # module, SweepSpace, be_polite, build_trials, memory_is_low,
    # recommended_workers, run_sweep, run_sweep_parallel, summarise_sweep,
    # the Qt screen module, and its register function. The retired entries are
    # regression_search.SearchSpace, build_trials, run_search and
    # summarise_search; the regression_search module itself remains as a
    # compatibility explanation. Enumerated rather than merely counted: this
    # ratchet is only useful if a bump names what it admitted and retired.
    # +73/-0 on the 2026-08-16 merge of the two concurrent sessions. Both
    # sides added public surface and neither retired any, so this is a pure
    # admission. Enumerated by module, because a ratchet moved without saying
    # what it admitted is a rubber stamp:
    #   47  spacr.qt.widgets     -- the figure/live-preview widgets, plus
    #                               spacr.qt.widgets.flash (the shared 650 ms
    #                               mark the console copy glyph and the new
    #                               clear-figures control both use)
    #    8  spacr.multi_database -- describe_merge / read_merged and their
    #                               MergePlan, SourceSummary and MergeRefused
    #    5  spacr.guide_concordance
    #    4  spacr.qt.preferences
    #    4  spacr.qt.screens
    #    3  spacr.parameter_sweep
    #    2  spacr.qt.ai          -- get_console_aware / set_console_aware
    #
    # +47/-0 on 2026-08-16, the regression figure/sweep surface. Enumerated
    # against ab503cc6, which is the commit the 6728 above was measured on --
    # not guessed from the diff, computed by running public_docstrings() in a
    # worktree at that commit and subtracting:
    #   22  spacr.qt.widgets     -- figure_grid_view (the whole module, its
    #                               FigureGridView and the cell_span /
    #                               cells_across layout helpers), the
    #                               fast_plots key/restyle API, three
    #                               FigureQueue accessors,
    #                               figure_settings.save_figure_as,
    #                               sweep_runs, and file_list.side_for_header
    #   10  spacr.trial_metrics  -- the per-trial scalar diagnostics
    #    5  spacr.qt.preferences -- get/set figure style, general and
    #                               per-graph, plus apply_figure_style
    #    4  spacr.figure_style   -- the module and its apply / rc_params /
    #                               resolve
    #    3  spacr.parameter_sweep
    #    1  spacr.hits
    #    1  spacr.qt.dnd_handlers -- SweepInputsDropHandler
    #    1  spacr.sweep_child
    #
    # +1 more while this very bump was being written: the count went 6656 ->
    # 6657 between measuring the delta above and running the test, because
    # several sessions are landing public surface concurrently. Worth knowing
    # when reading a failure here -- a one- or two-symbol drift is far more
    # likely to be a sibling session than a mistake, and the fix is to
    # re-measure and name it, never to relax the assertion into an inequality.
    #
    # THE CATALOGS FOR THESE 48 HAVE NOT BEEN REGENERATED. That is the debt
    # this bump takes on, recorded here rather than left implicit: the
    # reviewed API batches under docs/i18n/reviewed/api/<lang>/ are produced
    # by tools/build_documentation_i18n.py behind a translation model, which
    # is a separate job from this test. Until it runs, the localized API
    # pages omit these contracts.
    # +182/-0 on 2026-08-17, the finished regression module (instruction 124)
    # and the coverage push around it. Enumerated against 613218ee -- the
    # commit the 6657 above was measured on -- by running public_docstrings()
    # in a worktree at that commit and taking the set difference, not by
    # reading the diff. The before count came back at exactly 6657, so the
    # two ends of this bump are measured on the same definition.
    #
    #   20  spacr.gene_tile          -- the UniProt/ToxoDB link surface
    #   17  spacr.figures.panels     -- the seven house-style panels and the
    #                                   Panel record they return
    #   17  spacr.qt.widgets.fast_plots -- the key join, the restyle menu,
    #                                   offer_refit / offer_baselines /
    #                                   offer_compartments, build_style_menu
    #   13  spacr.figures.style      -- figure_style, Palette, ROLES, and the
    #                                   annotate / legend / reference-line
    #                                   helpers every panel draws through
    #   12  spacr.figures.distributions
    #   10  spacr.figures.plates     -- the small-multiple layout
    #   10  spacr.measurement_scan
    #   10  spacr.qt.widgets.regression_results
    #    9  spacr.figures.stats      -- automatic test selection
    #    8  spacr.qt.widgets.gene_tile
    #    7  spacr.qt.widgets.measurement_scan_panel
    #    6  spacr.baseline           -- what an effect is measured FROM
    #    6  spacr.figures.sheet
    #    6  spacr.refit              -- re-fitting from the plot
    #    5  spacr.localisation       -- one LOPIT compartment against grey
    #    5  spacr.qt.widgets.refit_dialog
    #    4  spacr.multi_database
    #    2  spacr.figures.summary, spacr.qt.widgets.figure_queue,
    #       spacr.qt.widgets.file_list, spacr.schema (2 each)
    #    1  spacr.figures, spacr.hits, spacr.ml, spacr.parameter_sweep,
    #       spacr.qt.widgets.figure_grid_view, .figure_settings, .gate_spec,
    #       spacr.regression_spec, spacr.sweep_child (1 each)
    #
    # NOTHING RETIRED. `regression_spec` is a re-export of tables that were
    # already public on `spacr.ml`, so ml keeps every one of its names and
    # this is an addition rather than a move -- checked, the removed set is
    # empty.
    #
    # THE TRANSLATIONS FOR THESE 182 ARE NOT REGENERATED, and that debt is
    # recorded here rather than left implicit, exactly as the previous bump
    # recorded its 48. The English catalog IS regenerated and audits clean
    # (`tools/build_documentation_i18n.py --audit` reports no `en/` entry);
    # the nine target languages need the translation model, which is a
    # separate job. Until it runs the localized API pages omit these
    # contracts -- the English pages do not.
    # +1 on 2026-08-17: `spacr.io.migrate_unescaped_plate_names`, renamed
    # from `_migrate_unescaped_plate_names`. It moves `stack/`,
    # `norm_channel_stack/`, `merged/` and `masks/` for a plate folder whose
    # name holds an underscore, so a plate that could not be measured becomes
    # measurable WITHOUT re-segmenting -- hours to days of work. The person
    # who needs it is a user with an `exp_1` folder full of masks, and a
    # recovery tool they have to reach past a leading underscore to call is
    # one most people will not find.
    # +5 on 2026-08-17: the collapsible figure sections and the Runs tab
    # (instruction 125 C). Named PUBLIC rather than left private, which is
    # the fix for a real smell -- they were written as `_toggle_section`,
    # `_record_run` and so on ONLY to avoid moving this counter, and
    # app_screen was reaching through two wrappers into another widget's
    # private surface as a result. A ratchet that makes people hide API is a
    # ratchet being gamed, and the answer is to move it, not to route around
    # it.
    #
    #   3  spacr.qt.widgets.figure_grid_view -- toggle_section,
    #      is_section_collapsed, set_section_collapsed
    #   2  spacr.qt.widgets.sweep_runs       -- record_run, update_run
    # +25 on 2026-08-17: instruction 128 D/E/F -- the live graph as a grid
    # tile, the residual diagnostics as tabs, and the mark type on the
    # right-click menu. Enumerated by the agent that wrote them and checked
    # against a HEAD worktree, where the ratchet passed, so all 25 are from
    # that one commit (16f32ce9).
    #
    #   spacr.qt.widgets.fast_plots       GroupedPlot, ScaleLocationPlot,
    #                                     InfluencePlot, mark_advice,
    #                                     context_from_model, FastPlot.snapshot,
    #                                     .offer_marks, .add_group_mark,
    #                                     GroupedPlot.mark / set_mark / redraw /
    #                                     group_sizes / mark_note and the two
    #                                     subclass overrides of each
    #   spacr.qt.widgets.regression_results
    #                                     results_frame, diagnostic_plots,
    #                                     set_diagnostics, clear_diagnostics
    #   plus the documented constants beside them.
    #
    # NOTHING WAS MADE PRIVATE TO DODGE THIS COUNTER, which is the failure
    # mode from the bump before last. `results_frame()` in fact REMOVES a
    # private reach: `_show_publication_sheet` was doing
    # `getattr(panel, "_frame")` into another widget's internals.
    # +59 on 2026-08-17, from four concurrent workstreams, set ONCE at the
    # end as the standing rule requires -- bumping it mid-flight was wrong
    # twice today because another agent was still landing surface.
    #
    #   instruction 130, the measurements database per plate:
    #     spacr.qt.widgets.file_list        is_database_path,
    #                                       attach_database, missing_databases
    #     spacr.qt.dnd_handlers             MeasurementsDropHandler.database_file
    #     spacr.qt.widgets.measurement_scan_panel
    #                                       DatabaseMergePanel and its surface,
    #                                       set_database_provider,
    #                                       refresh_databases, attached_databases
    #     spacr.plate_measurements          the headless merge composition
    #
    #   instruction 127's cleanups and 126's backdrop work, in the modules the
    #   other workflow owned.
    #
    #   spacr.gene_tile.uniprot_accessions  the bundled ToxoDB -> UniProt map.
    #
    # NOTHING WAS MADE PRIVATE TO DODGE THIS COUNTER. One agent reported
    # dropping a fifth candidate (`attached_databases` on the table widget)
    # not to avoid the ratchet but because the panel already read its rows
    # directly -- one vocabulary rather than two, which is the right reason.
    # +2 on 2026-08-17: the statsmodels summary tab --
    # `regression_results.summary_text` and
    # `RegressionResultsPanel.set_summary`. Both public because a caller
    # outside the widget renders one: `summary_text` is pure and is what the
    # tests drive, so it is not a widget method pretending to be one.
    # +7 on 2026-08-17: the effect-size cut moved onto the plot.
    #   spacr.thresholds        the module, METHODS, canonical, describe,
    #                           coefficient_threshold -- seven ways of
    #                           measuring the control spread, in ONE place so
    #                           the run and the right-click menu cannot offer
    #                           different ones. It was two methods inline in
    #                           ml.py.
    #   spacr.qt.widgets.fast_plots.FastPlot.offer_thresholds
    #   spacr.qt.widgets.regression_results  set_threshold_method,
    #                                        set_threshold_multiplier
    # +38 on 2026-08-17, the last regression-module sweep: the full restyle
    # menu (axis limits, aspect ratio, dimensions, font colour, line colour
    # and width, cmap-by-column, shape-by-column), the gene/guide filter
    # reaching every tab, the homogeneity verdict, the permutation run's
    # effect-size cut, Runs-before-Results and the run-follows-selection
    # binding, and the figure-queue caption fix. Set once, after all four
    # slices committed and with the tree clean.
    # +44 on 2026-08-17, the regression endgame: the two missing pyqtgraph
    # panels and their tabs (EffectRankPlot, BinnedPlot), the grid's live-tile
    # set (set_live_tiles, live_tile_keys, is_live_section_collapsed,
    # live_tiles_from_panels), spacr.cell_montage for instruction 131's
    # headless half, and the third test-selection engine in plot.py being
    # routed through spacr.figures.stats.
    #
    # Set once at the end, with every agent stopped and the tree clean --
    # which is the third time today that rule has been the difference between
    # a correct number and a stale one.
    #
    # +62/-0 on 2026-08-18, the night that finished instructions 131, 133 and
    # 135. A pure admission; nothing was retired, because the settings that
    # went away (score_column, log_x, log_y, x_lim, y_lims, split_axis_lims,
    # guide_permutation_plot, volcano, toxo) are dict keys and tooltip
    # strings, not public docstrings. Enumerated by module, because a ratchet
    # moved without saying what it admitted is a rubber stamp:
    #   25  spacr.qt.widgets.cell_montage_view -- instruction 131's Cells
    #                          tab: which objects a dot on the volcano is
    #                          most consistent with, loaded off disk or out
    #                          of the merged .npy stacks
    #    8  spacr.columns    -- a column that is not there offers the ones
    #                          that are: headers/available/missing/resolve/
    #                          describe/suggest and ColumnNotFound
    #    6  spacr.annotation -- the bundled Toxoplasma join: annotate,
    #                          supplementary, columns, gene_number,
    #                          clear_cache and SOURCES
    #    6  spacr.group_lasso -- fit, gene_effects, stability_selection,
    #                          max_lambda, describe and the module
    #    3  spacr.rra        -- rank_aggregate, describe and the module
    #   14  spread across spacr.ml, spacr.settings, spacr.parameter_sweep and
    #                          spacr.qt.screens: _toxoplasma_is_on's public
    #                          neighbours, the two new sweep `qc` parameters,
    #                          Section.add_prose, and the section-explainer
    #                          registry and its two helpers
    #
    # +35/-0 on 2026-08-18, the Gene tab and the figure restyle. Enumerated
    # by module, for the reason above:
    #   16  spacr.gene_facts -- everything known about ONE gene, gathered
    #                          from the bundled annotation and the screen's
    #                          own table: GeneFacts, Segment, facts,
    #                          facts_for, available, unavailable_reason,
    #                          warm, clear_cache
    #   19  spacr.qt.widgets.gene_panel -- the panel that shows it, plus
    #                          warm_annotation and the accessors the tests
    #                          drive it through (warm_now, show_feature,
    #                          save_topology, to_pixmap, is_warm)
    #
    # `spacr.plot`'s restyle added none: 45 figures moved inside the house
    # style and one private helper (_montage_type_size) was added, which is
    # private and therefore not surface.
    # +68 on 2026-08-18, measured rather than counted off a diff: 7117 ->
    # 7185. NINETEEN of them are the interactive plot widget's, from
    # instructions 147, 148 and 108 -- enumerated below, each measured by
    # taking the set difference against `a3944e21`, the commit that batch
    # started from:
    #
    #   spacr.qt.widgets.fast_plots -- 19
    #     the log transform being OURS rather than pyqtgraph's (148 A):
    #       log_axes, set_log_axes, log_reason
    #     grid and the canvas shape, off the checkbox strip and onto the
    #     right-click menu (148 C, 147 B):
    #       grid_shown, set_grid, canvas_shape, canvas_ratio,
    #       set_canvas_shape, resizeEvent
    #     reading a menu by what a user can REACH, which is what let the
    #     categories land without rewriting sixty assertions (147 C):
    #       menu_entries, menu_groups, menu_reading_order
    #     the level a plot is drawing, said on the plot (147 A):
    #       level_note
    #     a restyle menu built from a style dataclass's own fields (108.3):
    #       offer_style, add_style_entries, style_field_kind,
    #       style_field_choices, style_field_group, style_field_label
    #
    # THE OTHER 49 ARE NOT THIS BATCH'S and are not enumerated here: several
    # sessions land public surface concurrently, this count was already 7166
    # with the plot widget at a3944e21, and attributing another session's
    # symbols by guesswork would be a worse record than saying so. Whoever
    # lands the next bump should name theirs the same way -- by set
    # difference, not by reading a diff.
    #
    # +478/-1 through origin/nightly on 2026-08-20, measured against
    # dad9195d with this extractor rather than inferred from diffs. The
    # largest additions are 205 spacr.qt.widgets symbols, 26 in updater,
    # 20 in gene_measurement_sweep, 15 each in regression_summary and
    # qt.preferences, and the documented gene-measurement and lightweight
    # png_list APIs. ``spacr.io.crop_rows_from_png_list`` is the one retired
    # canonical body; the documented implementation now lives at
    # ``spacr.png_list.crop_rows_from_png_list`` and remains re-exported for
    # compatibility. The localized catalogs are regenerated with this bump.
    # +96/-0 through the final nightly documentation sweep. Measured by
    # comparing extractor output with the reviewed 7,662-symbol snapshot:
    # 25 workspace, 15 regex inference, 9 restart state, 7 AppScreen,
    # 5 each GIL priority and preferences, 4 each cell montage and fast
    # plots, 3 ml, 2 each CLI workspace/database set/figure grid/regression
    # results/sweep runs, and 9 single-symbol modules. The two documented
    # workspace constants account for the assignment set growing 16 -> 18.
    # Nothing retired, and all target catalogs are regenerated with this bump.
    # +6 for the transform/link-family preflight: guide_attribution.Preflight
    # (class, two documented attributes, and preflight), plus
    # ml.fit_quality_note and ml.resolve_glm_transform_conflict. Nothing
    # retired.
    # +7/-0 for the control-name resolver: the module, ControlSpec, note,
    # common_prefix, matches, resolve_control, and resolve_controls. Measured
    # against 1e36a6f9 with this extractor; no existing symbol retired.
    # +8/-0 for the measurement-comparison follow-up, measured against the
    # 7,890-entry snapshot: effects_grid_from_results, write_effects_grid,
    # ControlNotFound, rows_for, clear_picking_override,
    # multivariate_shortfall, nothing_to_compare_against, and the public
    # MeasurementScanPanel.sections accessor.
    # +12/-0 for optional example-screen data and figure model identity:
    # the example_data module, its error/result types, files/note,
    # cache_folder, fetch, is_whole, missing and total_bytes; the
    # example_data_manifest module; and regression_summary.model_identity_line.
    # Target catalogs are regenerated with this admission.
    # +2/-0 for the example-screen follow-up: AppScreen.load_the_example_screen
    # is the public entry point that populates the GUI example, and
    # MeasurementScanPanel.section_is_shown lets callers inspect the visible
    # result sections without reaching into widget internals. Nothing retired.
    # +6/-0 for the shared RGB channel picker: the channel_picker module,
    # ChannelPicker, its value/set_value methods, and parse/to_text. The same
    # typed channel vocabulary is now available to every image dialog.
    # +23/-0 for the plate-map and opt-in attribution follow-ups: 9 symbols in
    # well_spec, 7 in plate_map_picker, 5 in attribution_columns, the public
    # AppScreen.pick_wells_for entry point, and annotate_engine's cache reset.
    # These are callable headless contracts as well as GUI implementation.
    # +4/-0 for sweep diagnostics and screen-state inspection:
    # trial_metrics.qc_verdicts plus AppScreen.showing_the_figure_grid,
    # showing_the_live_graph, and showing_the_results. Nothing retired.
    # +13/-0 for measurement comparison and styled export. Seven headless
    # helpers resolve wells, identities, contrasts, and measurement-table
    # joins; five widget methods expose comparison rows, well selection, and
    # the join action; SaveFigureDialog.preview exposes the detached preview.
    # +59/-0 for the import preview, settings advisor, and shared figure
    # style follow-up, measured against the 7,839-canonical snapshot:
    #   17  spacr.settings_advisor
    #   10  spacr.import_plan
    #   11  spacr.qt.widgets.settings_advisor_dialog
    #    9  spacr.qt.widgets.import_workbench
    #    7  spacr.style_base
    #    2  spacr.qt.screens.app_screen
    #    2  spacr.gene_measurement_compare
    #    1  spacr.figures.style
    # The five advisor accessors documented during review are included in
    # these counts. Nothing retired; all additions are translated with the
    # catalog regeneration accompanying this ratchet.
    # +80/-0 for annotation validation and the user-facing analysis helpers:
    #    6  spacr.annotation_power
    #    6  spacr.annotation_umap_qc
    #   15  spacr.annotation_validation
    #   17  spacr.classifier_quality
    #   16  Qt dialog, plate-map, shortcut, and live-panel helpers
    #    8  spacr.read_background
    #    2  spacr.settings_advisor preflight helpers
    #   10  spacr.sudoku
    # Each public contract is present in the regenerated API catalogs.
    # +2/-0 for ``requirements_for_unit`` and the shared control-block reader
    # in the final advisor/design follow-up.
    # +6/-0 for the shared bar-spread module, its three public helpers, its
    # choice table, and ``GraphSpec.spread``.
    # +26/-0 for permutation quality control and its desktop surfaces:
    #    5  spacr.permutation_qc
    #    8  spacr.qt.setup_screen
    #    5  spacr.qt.widgets.annotation_umap_tab
    #    4  spacr.qt.widgets.setup_card
    #    4  spacr.run_recommendations
    # Nothing retired; the localized API catalogs are regenerated with this
    # admission.
    # +10/-0 for coordinated multiselection across the volcano, results table,
    # and cell montage:
    #    3  spacr.qt.widgets.cell_montage_view
    #    6  spacr.qt.widgets.fast_plots
    #    1  spacr.qt.widgets.regression_results
    # Nothing retired; these contracts are included in the same catalog pass.
    # +1/-0 for ``ClassEditorWidget.attach_sql_picker``, which exposes the
    # database-backed class-column chooser to settings panels.
    # +10/-0 for persistent foldable-panel state and recent-run inspection:
    #    7  spacr.qt.preferences and spacr.qt.widgets.foldable
    #    2  spacr.settings_advisor completed-run readers
    #    1  RegressionResultsPanel.colour_channels
    # Nothing retired; these contracts enter the regenerated API catalogs.
    # +14/-0 for response-transformation diagnostics and reproducible figure
    # bundles:
    #    6  spacr.response_distribution
    #    4  FastPlot bundle and comparison accessors
    #    4  spacr.figures.bundle
    # Nothing retired; these contracts enter the regenerated API catalogs.
    # +6/-0 for dependent-variable joins:
    #    5  spacr.dependent_join
    #    1  MeasurementComparePanel.set_dependent_frame
    # Nothing retired; these contracts enter the regenerated API catalogs.
    # +8/-0 for first-run configuration and provider persistence:
    #    6  spacr.qt.widgets.setup_dialog
    #    2  spacr.qt.preferences provider accessors
    # Nothing retired; these contracts enter the regenerated API catalogs.
    # +13/-0 for well-scoped plots and pre-annotation outlier filtering:
    #    5  spacr.outlier_filter
    #    4  MeasurementComparePanel scope accessors
    #    4  spacr.well_scope
    # Nothing retired; these contracts enter the regenerated API catalogs.
    # +2/-0 for geometry-derived montage paging:
    # ``fits_on_a_page`` and ``WellTab.per_page``. Nothing retired.
    # +12/-0 for reproducible streamed datasets and figure geometry:
    #   10  spacr.stream_dataset
    #    2  spacr.figure_style chrome/page helpers
    # Nothing retired; these contracts enter the regenerated API catalogs.
    # +8/-0 for data-aware graph choices and queued coefficient montages:
    #    7  spacr.graph_types
    #    1  CellMontageView.build_every_selected
    # Nothing retired; these contracts enter the regenerated API catalogs.
    # +9/-0 for class-derived compatibility values and incremental training
    # metrics:
    #    3  spacr.classify_classes
    #    6  spacr.qt.widgets.training_monitor
    # Nothing retired; these contracts enter the regenerated API catalogs.
    # +2/-0 for reaching the end of a console (instruction 232):
    # ``ConsolePanel.jump_to_the_end`` and ``ConsolePanel.at_the_end``, which
    # expose the tail-following control without requiring widget internals.
    # Nothing retired; these contracts enter the regenerated API catalogs.
    expected = 8107
    actual = len(docs) - len(builder.API_DOC_ALIASES)
    assert actual == expected, (
        f"the public API surface is {actual}, reviewed at {expected} "
        f"({actual - expected:+d}). A public docstring addition must bump this "
        "count, name what it admitted in the comment above, and regenerate "
        "the target catalogs with tools/build_documentation_i18n.py -- "
        "otherwise the localized API pages silently omit the new contract."
    )
    # The same surface WITH the aliases, which is what the catalog actually
    # carries. It tracks `expected` by a constant 119 -- the alias count --
    # and moving one without the other means the aliases changed, which is a
    # different event from the API growing and is worth failing separately.
    # It was a bare number with no sentence beside it, which is how it came
    # to be the second thing to update and the first thing forgotten.
    assert len(docs) == expected + len(builder.API_DOC_ALIASES) == 8226
    assert set(builder.API_DOC_ALIASES) <= docs.keys()

    # These are the only substantive audit bodies intentionally unresolved:
    # one external stdlib inheritance and two source-less Sphinx markers.
    assert "spacr.logging_util.LevelSetFilter.filter" not in docs
    assert "spacr.qt.widgets.gate_spec.Gate.columns" not in docs
    assert "spacr.qt.widgets.gate_spec.Gate.kind" not in docs


def test_documented_dunders_exclude_init_private_and_package_forwarders():
    docs = builder.public_docstrings()

    assert "spacr.version.__getattr__" in docs
    assert "spacr.qt.theme.__getattr__" in docs
    assert "spacr.active_learning.StoppingVerdict.__bool__" in docs
    assert not any(key.endswith(".__init__") for key in docs)
    assert "spacr.illumination._source_folders" not in docs
    # Package-level lazy forwarding hooks are not emitted in AutoAPI pages.
    assert "spacr.__getattr__" not in docs
    assert "spacr.qt.widgets.__getattr__" not in docs


def test_assignment_docs_are_ast_source_text_without_show_value_artifact():
    docs = builder.public_docstrings()
    assignment_keys = _visible_assignment_docs()

    assert len(assignment_keys) == 18
    assert assignment_keys <= docs.keys()
    assert all("Show Value" not in docs[key] for key in assignment_keys)
    assert docs["spacr.batch_correction.METHODS"] == (
        "Supported correction methods."
    )
    assert docs["spacr.anndata_export.ANNDATA_MISSING_MESSAGE"].startswith(
        "Exporting to AnnData (.h5ad) needs the optional `anndata` extra"
    )
    assert docs[
        "spacr.qt.widgets.graph_builder.GraphCanvas.RESCALE_ON_FILTER"
    ].startswith("The chart itself: a spec in, a faceted figure out")
    assert docs["spacr.workspace.MODES"].startswith(
        "Supported workspace persistence modes"
    )
    assert docs["spacr.workspace.SCHEMA_VERSION"].startswith(
        "Schema version written to"
    )


def test_exact_alias_map_and_manifest_records_are_identical():
    docs = builder.public_docstrings()
    aliases = builder.API_DOC_ALIASES

    assert len(aliases) == 119
    assert _sha256_lines(
        f"{alias}\0{canonical}" for alias, canonical in aliases.items()
    ) == _ALIASES_DIGEST
    assert not (set(aliases) & set(aliases.values()))
    assert all(docs[alias] == docs[canonical]
               for alias, canonical in aliases.items())
    assert "spacr.logging_util.LevelSetFilter.filter" not in aliases

    english = builder._english_manifest(docs)["symbols"]
    for alias, canonical in aliases.items():
        assert english[alias]["alias_of"] == canonical
        assert {
            key: value for key, value in english[alias].items()
            if key != "alias_of"
        } == english[canonical]


def test_localized_manifest_materializes_alias_translation(tmp_path, monkeypatch):
    docs = builder.public_docstrings()
    canonical = "spacr.layers.Layer.ndim"
    alias = "spacr.layers.ImageLayer.ndim"
    translations = {key: f"localized:{index}" for index, key in enumerate(docs)}
    # Alias model output must never win: its record references the one
    # canonical translation and all identical freshness hashes.
    translations[canonical] = "localized canonical body"
    translations[alias] = "incorrect duplicate decode"
    monkeypatch.setattr(builder, "API_DIR", tmp_path)

    builder.write_language(docs, "de", translations)
    payload = json.loads((tmp_path / "de.json").read_text(encoding="utf-8"))
    symbols = payload["symbols"]
    assert symbols[alias]["alias_of"] == canonical
    assert symbols[alias]["text"] == "localized canonical body"
    assert {
        key: value for key, value in symbols[alias].items()
        if key != "alias_of"
    } == symbols[canonical]
