#!/usr/bin/env python3
"""Build the canonical English catalog for every spaCR tutorial lesson.

The compact source table below is deliberately the authority for narration,
portal metadata, capture routing, and production order.  Generated JSON lives
under ``catalog/`` and is consumed by the localization, audio, video, and web
publishing tools.
"""
from __future__ import annotations

import json
from pathlib import Path

from pronunciation import spoken_form

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "catalog"


# Lessons for capabilities that remain available from an icon in a current
# host module.  ``app_key`` remains the stable lesson/capability identity;
# ``host_app_key`` is the only module route presented to users and capture
# tools.  Keeping this explicit prevents a retired Home tile from returning
# silently when the catalog is rebuilt.
FOLDED_LESSON_HOSTS = {
    "activation": "classify_merged",
    "agreement": "annotate",
    "anndata_export": "measure",
    "barcode_qc": "map_barcodes",
    "cellpose_masks": "make_masks",
    "classifier_evaluation": "classify_merged",
    "classify": "classify_merged",
    "curate": "make_masks",
    "explain_cv": "classify_merged",
    "hit_list": "regression",
    "illumination": "measure",
    "image_scatter": "umap",
    "methods_export": "regression",
    "ml_analyze": "classify_merged",
    "model_compare": "make_masks",
    "model_zoo": "make_masks",
    "motility": "measure",
    "napari_bridge": "make_masks",
    "parameter_sweep": "regression",
    "pca": "umap",
    "timelapse": "mask",
    "train_cellpose": "make_masks",
    "volcano_explorer": "regression",
}


def lesson(
    number: int,
    slug: str,
    title: str,
    description: str,
    *,
    series: int,
    app_key: str | None = None,
    section: str = "spaCR",
    inputs: str,
    settings: str,
    run: str,
    outputs: str,
    prerequisite: str = "Follow the earlier workflow lessons when this module needs upstream results.",
    objectives: list[str] | None = None,
) -> dict:
    lesson_id = f"{number:02d}_{slug}"
    input_text = inputs[:1].upper() + inputs[1:]
    run_text = run[:1].upper() + run[1:]
    objectives = objectives or [
        f"Understand when to use {title}.",
        "Configure the important inputs and settings.",
        "Run the workflow and locate its outputs.",
    ]
    scenes = [
        {
            "narration": f"This tutorial covers {title}. {description}",
            "visual": "overview",
            "hold_after": 0.45,
        },
        {
            # The module is already open in its captured keyframe. Keeping
            # navigation and input setup in one sentence made the emphasis
            # box point at the sidebar while the voice described controls in
            # the module. Start directly at the actual input region instead.
            "narration": input_text,
            "visual": "input",
            "hold_after": 0.45,
        },
        {
            "narration": f"In the main configuration area, {settings}",
            "visual": "settings",
            "hold_after": 0.45,
        },
        {
            "narration": run_text,
            "visual": "run",
            "hold_after": 0.45,
        },
        {
            "narration": f"When the workflow finishes, {outputs}",
            "visual": "output",
            "hold_after": 0.65,
        },
    ]
    return {
        "id": lesson_id,
        "number": number,
        "slug": slug,
        "title": title,
        "series": series,
        "app_key": app_key,
        "section": section,
        "description": description,
        "objectives": objectives,
        "prerequisite": prerequisite,
        "scenes": scenes,
    }


def scripted_lesson(
    number: int,
    slug: str,
    title: str,
    description: str,
    scenes: list[tuple[str, str]],
    *,
    prerequisite: str,
    objectives: list[str],
) -> dict:
    """Build a non-application lesson whose visuals have explicit scenes."""
    return {
        "id": f"{number:02d}_{slug}",
        "number": number,
        "slug": slug,
        "title": title,
        "series": 1,
        "app_key": None,
        "section": "spaCR",
        "description": description,
        "objectives": objectives,
        "prerequisite": prerequisite,
        "scenes": [
            {"narration": narration, "visual": visual, "hold_after": 0.7}
            for visual, narration in scenes
        ],
    }


def captured_lesson(
    number: int,
    slug: str,
    title: str,
    description: str,
    scenes: list[tuple[str, str | None, str]],
    *,
    series: int,
    app_key: str,
    section: str,
    prerequisite: str,
    objectives: list[str],
) -> dict:
    """Build a lesson routed to named frames from a real application capture."""
    lesson_scenes = []
    for visual, focus_key, narration in scenes:
        scene = {"narration": narration, "visual": visual, "hold_after": 0.7}
        if focus_key:
            scene["focus_key"] = focus_key
        lesson_scenes.append(scene)
    return {
        "id": f"{number:02d}_{slug}",
        "number": number,
        "slug": slug,
        "title": title,
        "series": series,
        "app_key": app_key,
        "section": section,
        "description": description,
        "objectives": objectives,
        "prerequisite": prerequisite,
        "scenes": lesson_scenes,
    }


LESSONS = [
    scripted_lesson(
        1, "pypi_github", "PyPI, GitHub, and conda-forge",
        "Choose the official source for a Python package, conda-forge package, desktop installer, or development branch.",
        [
            ("sources", "Before installing spaCR, choose an official distribution route. PyPI and conda-forge provide released packages, while GitHub provides source code, desktop installers, and release records."),
            ("pypi_release", "On PyPI, the release header shows the current stable version and the pip install spacr command."),
            ("pypi_links", "The Project links section opens the documentation, homepage, issue tracker, and source repository."),
            ("github_repo", "On GitHub, use the branch selector to choose the source line. Main is the released line; nightly contains newer work still being tested."),
            ("github_release", "In a GitHub Release, expand Assets to download the Windows, macOS, or Linux installer, the Python package files, or the checksum list."),
            ("conda_status", "The official conda-forge package installs spaCR and its desktop dependencies into the active Conda environment. Use the exact command conda install conda-forge::spacr."),
            ("decision", "Use conda-forge or PyPI for a managed Python environment, GitHub Releases for a desktop installer, and nightly only when you need unreleased work. Next, choose the installation tutorial for your route."),
        ],
        prerequisite="No installation is required.",
        objectives=[
            "Identify the official PyPI and conda-forge packages.",
            "Distinguish stable releases from the nightly branch.",
            "Select the appropriate installation route for the intended environment.",
        ],
    ),
    scripted_lesson(
        2, "conda_install", "Installation with Conda",
        "Create an isolated Python 3.12 environment and install the complete spaCR desktop package directly from conda-forge.",
        [
            ("overview", "Install spaCR from conda-forge in a dedicated Conda environment. The package includes the desktop interface and declares spaCR's dependencies."),
            ("create", "Open a Conda-enabled terminal and create an environment named spacr with Python three point twelve."),
            ("activate", "Activate the environment. Confirm that Python belongs to the spacr environment before installing anything."),
            ("install", "Install the official package with conda install conda-forge::spacr. The channel-qualified command selects the conda-forge build explicitly."),
            ("verify", "Confirm that Conda reports spaCR in the environment, then import it and print the installed version."),
            ("doctor", "Run spacr-doctor before the first launch. It reports the active installation, Qt, PyTorch, optional components, and hardware support with a suggested fix for every failed check."),
            ("finish", "After spacr-doctor passes, launch spaCR with the spacr command. Keep this environment activated whenever you run spaCR."),
        ],
        prerequisite="Install Miniforge, Miniconda, Anaconda, or another Conda-compatible manager.",
        objectives=[
            "Create and activate an isolated spaCR environment.",
            "Install spaCR directly from the official conda-forge channel.",
            "Verify the package and diagnostics before launching spaCR.",
        ],
    ),
    scripted_lesson(
        3, "pip_install", "Installation with pip",
        "Install spaCR into a standard virtual environment without using Conda.",
        [
            ("overview", "The pip route works on Windows, macOS, and Linux. Always use an isolated virtual environment rather than modifying the operating system's Python installation."),
            ("venv_posix", "On macOS or Linux, create the environment with Python three point twelve, then activate it from its bin directory."),
            ("venv_windows", "On Windows, use the Python launcher to create the environment and activate it from the Scripts directory."),
            ("upgrade", "Upgrade pip in the active environment. The command python -m pip keeps installation tied to the Python you just activated."),
            ("install", "Install the plain spacr package for headless use, or spacr with the qt extra for the desktop interface."),
            ("verify", "Verify the imported version and run spacr-doctor. Resolve any failed checks before opening experimental data."),
            ("finish", "The environment is now ready. Keep it activated whenever you use spaCR, and use python -m pip install --upgrade spacr when you intentionally update it."),
        ],
        prerequisite="Install a supported Python version; Python 3.12 is recommended.",
        objectives=[
            "Create a virtual environment on each supported operating system.",
            "Install the correct spaCR package variant.",
            "Verify the environment before launching the application.",
        ],
    ),
    scripted_lesson(
        4, "platform_installers", "Platform installers",
        "Install spaCR on Windows, macOS, or Linux without an existing Python or Conda installation.",
        [
            ("overview", "The lightweight native installers create a private Python three point twelve runtime, then download Qt, PyTorch, spaCR, and its dependencies."),
            ("release", "Download installers only from the latest spaCR GitHub Release. Match the file to your operating system and use the published SHA-256 file when you need to verify the download."),
            ("windows", "On Windows ten or eleven, open the online setup executable. Automatic hardware acceleration is selected by default and installs CUDA support on compatible NVIDIA hardware. Clear that component only when you require a smaller CPU-only installation."),
            ("macos", "On macOS eleven or newer, open the universal package for Intel and Apple silicon. If Gatekeeper blocks the unnotarized package, use Privacy and Security, Open Anyway, then run it again."),
            ("linux", "On sixty four bit Linux, make the downloaded .run file executable, then start it from a terminal. Automatic backend selection is the default and installs CUDA support when compatible NVIDIA hardware is available."),
            ("linux_gpu", "To require a smaller CPU-only installation on Linux, pass --torch-backend cpu. Otherwise, keep the default automatic selection so CUDA acceleration is used when supported."),
            ("safety", "Each installer validates the replacement before switching versions. An interrupted update leaves the previous environment intact, and install.log remains in the private spaCR installation directory."),
            ("performance", "Preferences has one Performance level selector with Laptop, Extra Performance, Performance, Balanced, and Workstation in increasing resource order. Choose the profile for this computer; it changes caching, worker limits, cleanup, and animation detail, but not scientific settings or results."),
            ("finish", "After installation, use the operating system launcher. No separate Python or Conda setup is required."),
        ],
        prerequisite="Download the installer for your operating system from the official GitHub release.",
        objectives=[
            "Select the correct installer for Windows, macOS, or Linux.",
            "Handle platform security and acceleration choices.",
            "Locate diagnostics if installation fails.",
        ],
    ),
    scripted_lesson(
        5, "home", "Home screen and navigation",
        "Find modules, inspect application status, and move into a workflow from the spaCR Home screen.",
        [
            ("overview", "When all maturity levels are visible, Home displays all forty four registered spaCR modules. The Home tab groups them under Core, Data, Results & QC, Explore, Assays, and Design; each category tab filters the same registry."),
            ("navigation", "The left sidebar keeps Home and the visible Core modules directly accessible. With all maturity levels enabled, this is the six-step Core sequence. Select a category heading to expand its remaining visible modules."),
            ("modules", "The Core band follows the workflow order: Mask, Measure, Annotate, Classify, Map Barcodes, and Regression. Hover over a tile to display its module description and its Alpha, Beta, or Stable maturity color. Preferences can hide Alpha or Beta modules from Home and navigation; Stable modules remain visible."),
            ("performance", "Preferences has one Performance level selector with Laptop, Extra Performance, Performance, Balanced, and Workstation in increasing resource order. Choose the profile for this computer; it changes caching, worker limits, cleanup, and animation detail, but not scientific settings or results."),
            ("open_module", "Select a module tile to open its settings screen. The module name and description at the top confirm which workflow is active."),
        ],
        prerequisite="Install and launch the spaCR desktop application.",
        objectives=[
            "Find modules from Home, category tabs, or the sidebar.",
            "Read system and module-state information.",
            "Open a workflow and confirm the active module.",
        ],
    ),
    scripted_lesson(
        6, "api", "Python API and headless workflows",
        "Run reproducible spaCR pipelines from Python or a shell without the desktop interface.",
        [
            ("import", "Import spaCR and print its version to confirm that the active Python environment can use the package."),
            ("list", "Run spacr-run with the list option to see the available pipeline entry points, including mask, measure, annotate, classify, and report."),
            ("settings", "Start a module with spacr-run, the module name, and an exported settings file. spaCR validates paths, channels, workers, and hardware options before work begins."),
            ("dry_run", "Add --dry-run to resolve settings, print the execution plan, and run pre-flight validation without processing data. Fix every reported problem before submitting unattended work."),
            ("run", "Remove --dry-run only after the plan and pre-flight checks pass. Headless jobs save the same outputs, settings, version record, and logs as the desktop workflow."),
        ],
        prerequisite="Complete one installation tutorial first.",
        objectives=[
            "Confirm that spaCR is available from Python.",
            "Choose and configure a spacr-run pipeline.",
            "Validate the resolved plan before a full headless run.",
        ],
    ),
    captured_lesson(
        7, "mask", "Mask",
        "Configure segmentation, validate it with Live Preview, and create masks for downstream measurement.",
        [
            ("01_console", None, "Mask places settings on the left and the runtime console on the right. The console reports validation, progress, warnings, and saved outputs."),
            ("05_metadata", "metadata_dialog", "Drop the source image folder into Mask. Confirm the detected filename pattern and metadata, then map each acquisition channel to its biological signal. In this example, the four channels contain nuclei, endoplasmic reticulum, lipid droplets, and parasites."),
            ("08_all_settings", "settings_controls", "Essentials shows the controls needed for a typical run. Use All only when you need advanced options, Modified to review changed values, and Recipes to save a configuration you will reuse."),
            ("12_category_01_input_metadata", "category", "In Input and Metadata, verify the source folder, channel assignments, magnification, and filename metadata before configuring segmentation."),
            ("12_category_02_workflow_test_run", "category", "Workflow and Test Run controls which masks are created, whether existing work resumes, and how many representative fields are used for a test."),
            ("12_category_04_cell_segmentation", "category", "The cell, nucleus, pathogen, and organelle sections each group the model, channel, core thresholds, and object filters for that compartment. Change only settings needed for the current images."),
            ("25_cell_ready", "canvases", "Enable Live Preview, load a representative field, and choose the object you are tuning. This example previews the cell signal."),
            ("26_live_settings", "segmentation", "In Live settings, confirm the object and channel first. Adjust diameter, flow threshold, or cell probability only when the preview gives you a reason."),
            ("26_live_settings", "propagate", "Turn on Propagate settings so the values you validate here become the settings used by the full run."),
            ("27_cell_outlines", "action_and_canvases", "Run the preview and inspect Overlay beside the source. Boundaries should follow the visible cells without joining neighbors or missing clear objects."),
            ("28_cell_flows", "action_and_canvases", "Switch to Flows to inspect the direction field used by Cellpose. Irregular flow patterns can help explain masks that split or merge incorrectly."),
            ("29_cell_objects", "action_and_canvases", "The Masks view displays the final labels against a black background. Use it to assess object coverage and separation without the source image."),
            ("30_cell_size_filter", "min_area", "Object filters update the cached preview immediately. Raise minimum area only enough to remove obvious fragments; the correct cutoff depends on image scale."),
            ("31_cell_filtered", "canvases", "Compare the filtered overlay with the original preview. Genuine cells should remain; if they disappear, lower the minimum area."),
            ("12_category_12_output_storage", "category", "Choose the masks and intermediate files you need in Output and Storage. Run a small test first, then process the full source after the preview and console checks pass."),
        ],
        series=1, app_key="mask", section="Core",
        prerequisite="Use a microscopy source whose channel identities and filename metadata are known.",
        objectives=[
            "Verify source metadata and channel mapping.",
            "Configure only the segmentation controls required by the images.",
            "Use Live Preview to validate masks before a full run.",
        ],
    ),
    captured_lesson(
        8, "measure", "Measure",
        "Extract object-level features from images and masks, then inspect the resulting crop gallery and database output.",
        [
            ("01_console", "console", "Measure places settings on the left and the runtime console on the right. The console reports source validation, progress, warnings, and saved measurement output."),
            ("05_folder", "src", "Choose the merged project folder produced by Mask. Measure reads the stored image, mask, and project metadata directly."),
            ("08_all_settings", "settings_controls", "Start with Essentials. Use All for advanced controls, Modified to audit changed values, and Recipes when the same measurement setup will be reused."),
            ("12_category_02_mask_channel_mapping", "category", "In Mask and Channel Mapping, verify the image channels and the label planes for cells, nuclei, pathogens, organelles, and any derived compartments."),
            ("12_category_03_measurement_features", "category", "Measurement Features selects the intensity, morphology, texture, colocalization, and spatial summaries needed for the analysis. Leave unused feature families off."),
            ("12_category_05_crop_output", "category", "Object Filtering removes known artifacts, while Crop Output controls whether representative object images are saved for annotation or classification. Choose filters and crop scale from the experiment rather than copying fixed values."),
            ("12_category_06_preview_diagnostics", "category", "Preview and Diagnostics defines a small test sample. Use it before measuring the full experiment."),
            ("20_cell_crops", "live_and_grid", "Enable Live Preview and load a completed merged array. The gallery groups real object crops so you can check mask assignments, channel content, and crop framing."),
            ("24_er_channel", "grid", "Switch gallery channels to inspect each biological signal. This display choice helps quality control and does not limit which configured channels are measured."),
            ("25_outputs", "status", "After the test succeeds, run the full source. Measure saves traceable object rows in measurements dot db and any requested crops beside the project outputs."),
        ],
        series=1, app_key="measure", section="Core",
        prerequisite="Complete Mask and confirm that the merged project contains usable image and label planes.",
        objectives=[
            "Verify image channels and mask-plane mapping.",
            "Choose only the feature families, filters, and crops the analysis needs.",
            "Validate a test in Live Preview before writing full measurement outputs.",
        ],
    ),
    captured_lesson(
        9, "annotate", "Annotate",
        "Review object crops, assign consistent labels, audit coverage, and hand saved labels to a classifier.",
        [
            ("01_console", "console_and_chat", "Annotate prioritizes the crop grid. Open the console only when you need database messages, save status, warnings, or local help, then hide it to return space to the images."),
            ("05_folder", "grid", "Open a measured project, choose or create an annotation column, and confirm that the expected object crops load. Define what each class means before labeling begins."),
            ("06_settings", "dialog", "Settings controls the displayed channels, crop appearance, object filters, and optional uncertainty queue. Keep the view consistent for everyone contributing labels."),
            ("08_keyboard_reference", "legend", "Open the keyboard reference for the small set of actions used repeatedly: assign a class, move focus, skip, undo, and save or advance."),
            ("09_keyboard_class_one", "grid", "Assign labels with number keys or the mouse. The class-colored ring confirms the saved label, and focus advances so many objects can be reviewed efficiently."),
            ("12_undo", "focused_crop", "Use navigation and undo to correct a recent choice without clearing other saved work. Save before moving to another page or ending the session."),
            ("16_coverage", "report", "Coverage shows whether labels span classes, plates, wells, sources, and active-learning rounds. Add examples from missing groups before training."),
            ("21_active_learning", "active_controls", "When enough representative labels exist, Retrain can score the remaining crops and prioritize uncertain examples. Treat this as an optional review aid, not a replacement for coverage."),
            ("22_training_handoffs", "training", "Train CV sends image crops and this annotation column to the vision classifier. Train XG sends the same labels to the feature-based classifier; both open a configured module without starting training automatically."),
            ("24_saved_result", "status", "Annotations remain in measurements dot db with object identifiers and provenance intact. Continue until coverage is adequate for the intended model or quality-control decision."),
        ],
        series=1, app_key="annotate", section="Core",
        prerequisite="Complete Measure with object crops and decide on a documented class scheme.",
        objectives=[
            "Configure a consistent crop view and annotation column.",
            "Label, navigate, undo, and save efficiently.",
            "Audit coverage before handing labels to a classifier.",
        ],
    ),
    captured_lesson(
        10, "classify_cv", "Classify, computer vision",
        "Train and evaluate an image classifier from annotated object crops, with grouped validation and optional hyperparameter search.",
        [
            ("01_console", "console", "Classify CV trains image models on annotated object crops. The console reports dataset checks, training progress, held-out metrics, inference, warnings, and output paths."),
            ("12_category_01_plate_sources_workflow", "category", "Plate Sources and Workflow selects the measured projects and the stages to run: build a split, train, evaluate, or apply an existing checkpoint."),
            ("08_all_settings", "settings_controls", "Start with Essentials, use Modified to audit deliberate changes, and open All only for controls required by the experiment. Save a Recipe when the setup will be repeated."),
            ("12_category_02_labels_classes", "category", "Labels and Classes selects the annotation or metadata field used as ground truth and maps its saved values to documented class names. Verify that mapping before training."),
            ("12_category_03_images_cropping", "category", "Images and Cropping chooses the object images, channels, balancing, and held-out fraction. Group related images by plate or well so near-duplicates cannot leak between training and evaluation."),
            ("12_category_04_model_regularization", "category", "Model and Regularization selects the network, input channels and size, pretrained weights, normalization, and regularization expected by the images. Begin with one reasonable baseline."),
            ("12_category_05_training_loss", "category", "Training and Loss contains learning rate, augmentation, batches, epochs, and early stopping. Change a small number of controls at a time and judge them on validation behavior."),
            ("12_category_06_evaluation_results", "category", "Evaluation and Results adds grouped evaluation, calibration, and strict identity and content-leakage checks. Keep final held-out data separate from model selection."),
            ("12_category_06_evaluation_results", "category", "The same Evaluation and Results section controls full-dataset inference, saved probabilities and class calls, and representative examples. Review those outputs before using predictions biologically."),
            ("23_hyperparameter_search", "results", "Enable Hyperparameter Search when several plausible configurations need comparison. The mini workbench runs grouped trials and displays each score, fold variation, parameters, and status."),
            ("24_search_settings", "dialog", "Search Settings defines a focused candidate space, ranking metric, grouped folds, and trial budget. Avoid broad searches that spend compute on choices the experiment cannot distinguish."),
            ("25_run_pipeline", "actions", "Run first validates labels and splits, then trains and evaluates the configured model. Review held-out metrics and examples before applying it broadly; spaCR stores the checkpoint, predictions, settings, and run record."),
        ],
        series=1, app_key="classify", section="Core",
        prerequisite="Complete Annotate with documented classes and representative coverage across the intended plates or wells.",
        objectives=[
            "Map saved labels and object crops into a leakage-safe dataset.",
            "Configure a baseline model and grouped evaluation.",
            "Use Hyperparameter Search only for a focused comparison.",
        ],
    ),
    captured_lesson(
        11, "classify_ml", "Classify, machine learning",
        "Train and evaluate a feature-based classifier from measured objects, with plate-aware correction and optional hyperparameter search.",
        [
            ("01_console", "console", "Classify ML trains a classical model on per-object measurements. The console reports database loading, feature preparation, validation, importance, predictions, and errors."),
            ("12_category_01_labels_classes", "category", "Labels and Classes selects the measurement project, label field, positive and negative values, and experimental grouping. Verify the biological meaning of that mapping before fitting a model."),
            ("08_all_settings", "settings_controls", "Start with Essentials and audit changes in Modified. Open All only for controls the analysis needs, and save a Recipe for repeated runs."),
            ("12_category_02_feature_preparation", "category", "Feature Preparation selects the channel of interest and removes excluded, invalid, low-variance, highly correlated, or poorly supported measurements before training."),
            ("12_category_03_plate_batch_correction", "category", "Plate and Batch Correction can normalize each plate against its controls. Use a correction that preserves the biological contrast and can be applied consistently to future data."),
            ("12_category_04_classifier_validation", "category", "Classifier and Validation selects the estimator, complexity, held-out split, and cross-validation behavior. Group related objects by plate or well to avoid leakage."),
            ("12_category_05_feature_selection_importance", "category", "Feature Selection and Importance prunes the final feature set and estimates which measurements support the classifier. Interpret importance only after held-out performance is acceptable."),
            ("12_category_06_output_database", "category", "Output and Database controls whether probabilities, class calls, model files, and selected features are saved or written back to the measurement project."),
            ("23_hyperparameter_search", "results", "Hyperparameter Search is a mini workbench for comparing a focused set of model choices with grouped folds. The results table ranks completed trials and shows their variation."),
            ("24_search_settings", "dialog", "Search Settings defines candidate values, trial budget, folds, seed, and ranking metric. Keep the search space small enough to explain and reproduce."),
            ("25_run_pipeline", "actions", "Run validates controls, prepares features, applies any plate correction, evaluates the classifier, calculates importance, and saves enabled outputs. Review held-out and per-plate behavior before using predictions biologically."),
        ],
        series=1, app_key="ml_analyze", section="Core",
        prerequisite="Complete Measure and provide a reliable label or control mapping with leakage-safe grouping columns.",
        objectives=[
            "Prepare valid measurement features and plate-aware labels.",
            "Configure grouped validation and interpretable outputs.",
            "Use Hyperparameter Search for a focused, reproducible comparison.",
        ],
    ),
    captured_lesson(
        12, "map_barcodes", "Map Barcodes",
        "Decode sequencing reads into plate positions and guide identities, with explicit references and quality-control output.",
        [
            ("01_console", "console", "Map Barcodes decodes sequencing reads into plate positions and guide identities. The console reports sample discovery, parser validation, read progress, output paths, and failures."),
            ("13_category_01_sequencing_input", "category", "Sequencing Input selects the FASTQ folder and whether paired reads are combined or one read direction is processed alone. Confirm the naming pattern before a large run."),
            ("09_all_settings", "settings_controls", "Essentials contains the inputs and parsing controls needed for a first run. Use Modified to review library-specific changes, All for advanced controls, and Recipes for a reusable layout."),
            ("13_category_02_barcode_references", "category", "Barcode References points to lookup tables for guides, plate rows, and plate columns. Each table must pair a unique name with the expected sequence."),
            ("13_category_03_read_parsing", "category", "Read Parsing locates a known anchor, extracts the barcode window, and assigns named row, column, and guide groups. Derive this pattern from the library design and verify it on example reads."),
            ("13_category_05_runtime_reliability", "category", "Runtime and Reliability sets chunk size, workers, and test mode. Start with a small test chunk; reduce the chunk or worker count when memory is constrained."),
            ("20_run", "actions", "Run validates paths and references, discovers samples, counts reads, and starts chunked mapping. Stop requests an orderly interruption after the current safe point."),
            ("21_running", "console", "Watch the active sample, batch count, matched fraction, and warnings in the progress display and console. Investigate poor mapping before accepting the complete run."),
            ("22_output", "console", "A successful run writes unique barcode combinations with row, column, guide, and count, plus a quality-control table, settings, and a manifest that make the mapping reproducible."),
        ],
        series=1, app_key="map_barcodes", section="Core",
        prerequisite="Provide FASTQ reads and verified lookup tables that match the sequencing-library design.",
        objectives=[
            "Configure read direction, barcode references, and parsing from the library design.",
            "Validate mapping on a small test chunk.",
            "Interpret mapping progress and quality-control outputs.",
        ],
    ),
    captured_lesson(
        13, "regression", "Regression",
        "Relate pooled-screen guide abundance to measured phenotypes with "
        "blocked permutation tests or simultaneous regression, then inspect "
        "support, diagnostics, and effect estimates.",
        [
            ("01_console", "console", "Regression relates phenotype scores to sequencing-derived guide abundance. The console reports table alignment, filtering, the selected inference procedure, diagnostics, results, and reproducibility records."),
            ("12_category_01_input_tables", "category", "Input Tables selects phenotype scores, sequencing counts, optional gene metadata, and the result destination. Leave the destination blank to write beside the first count table; confirm matching plate and well identifiers before analysis."),
            ("08_all_settings", "settings_controls", "Essentials contains the inputs and inference controls required for a first analysis. Use Modified to audit deliberate changes, All to inspect advanced controls, and Recipes to save a reusable design."),
            ("12_category_02_controls_filters", "category", "Controls and Filters identifies positive, negative, and neutral controls; excludes specified guides, wells, or table rows; and sets the evidence and normalization rules that determine which observations enter the analysis."),
            ("12_category_03_plate_batch_correction", "category", "Plate and Batch Correction can remove technical offsets by centering, robust scaling, reference controls, or batch models. Apply correction only when the design supports it."),
            ("12_category_04_response", "category", "Response selects the phenotype column, analysis unit, aggregation, optional inversion, and transform. These choices define the quantity associated with guide abundance."),
            ("12_category_05_model_inference", "category", "Model and Inference chooses nonparametric guide testing, parametric simultaneous regression, or automatic selection. It also defines the reported level, model family when applicable, multiple-testing correction, and significance rule."),
            ("12_category_06_estimator_tuning", "category", "Estimator Tuning contains covariance, regularization, robust-fit, quantile, bootstrap, and rank-aggregation controls. Only settings used by the selected estimator affect the analysis."),
            ("12_category_07_permutation_test", "category", "Permutation Test configures the guide statistic, minimum well support, permutation count and seed, exchangeability block, nuisance variables, presence threshold, and memory-bounded batch size."),
            ("21_run", "actions", "Run validates schemas and well keys, applies exclusions and correction, aggregates the response, and executes the selected inference procedure. The default nonparametric procedure uses plate-blocked Freedman–Lane permutations and does not fit a regression model."),
            ("23_output", "console", "The completed project contains aligned analysis data, guide- and gene-level effect estimates, empirical or model-based significance tables, diagnostic figures, saved settings, and a rerunnable manifest."),
        ],
        series=1, app_key="regression", section="Core",
        prerequisite="Complete barcode mapping and prepare phenotype and guide-count tables with matching plate and well identifiers.",
        objectives=[
            "Align phenotype, guide-count, and control information.",
            "Choose a justified inference procedure, correction, and evidence rules.",
            "Interpret effects together with controls and diagnostics.",
        ],
    ),
    captured_lesson(
        14, "make_masks", "Make Masks",
        "Create and correct segmentation labels interactively for quality control or Cellpose training pairs.",
        [
            ("02_loaded", "canvas", "Open a folder of representative microscopy images. Matching labels load from its masks folder as colored overlays, while images without a companion mask begin empty."),
            ("03_tools", "tools", "The Tools card groups painting, erasing, intensity-aware wand, and zoom modes. Choose the smallest operation that corrects the observed mask error."),
            ("04_brush", "brush", "Brush adds foreground with an adjustable radius. Use short strokes to repair missing boundaries or objects without painting across neighboring structures."),
            ("05_erase", "button", "Erase removes only the painted pixels, while Erase object removes one complete label. Undo immediately if the selected region was valid."),
            ("07_wand_add", "wand", "Wand plus and Wand minus follow connected image intensities. Start with a conservative tolerance and pixel limit, then inspect the selected region before continuing."),
            ("09_zoom", "button", "Zoom into difficult boundaries and adjust display normalization only when it helps visibility. Contrast controls do not change the image data or mask geometry, so choose them from the current field rather than fixed tutorial percentiles."),
            ("11_history", "history", "Undo and Redo step through mask edits independently of zoom and contrast, making it safe to compare a correction with the prior label."),
            ("12_object_operations", "operations", "Object operations can fill holes, relabel connected foreground, invert labels, or clear the mask. Use area filtering only with a threshold justified by image scale and expected object size."),
            ("14_save", "save", "Save mask relabels connected components and writes the label image under the source stem. Move through the folder systematically so every training field is reviewed."),
            ("16_blank_mask", "brush", "For an image without labels, create objects with Brush or Wand plus, clean the boundaries, and save the new image-mask pair for model training."),
        ],
        series=1, app_key="make_masks", section="Segmentation models",
        prerequisite="Prepare representative microscopy images and, when available, matching masks in an isolated working copy.",
        objectives=[
            "Select an editing tool appropriate to each mask error.",
            "Use contrast and object filters without copying dataset-specific values.",
            "Save consistently reviewed image-mask pairs.",
        ],
    ),
    captured_lesson(
        15, "image_umap", "Image UMAP",
        "Embed measured objects in two dimensions and inspect phenotype neighborhoods using their real microscopy crops.",
        [
            ("01_console", "console", "Image UMAP reduces high-dimensional object measurements to two coordinates and places microscopy crops on the map. The console records joins, preprocessing, embedding, clustering, and outputs."),
            ("12_category_01_paths", "category", "Paths selects the measured project. Image UMAP locates measurements dot db and uses existing crop images or builds crops from merged arrays when supported."),
            ("12_category_02_measurements", "category", "Measurements selects the joined object tables and feature families, removes explicit exclusions, and can drop redundant correlated features. These choices define the geometry of the map."),
            ("12_category_03_plate_layout_controls", "category", "Plate Layout and Controls excludes failed wells, controls, or metadata groups that should not shape this exploratory embedding."),
            ("12_category_04_computer_vision_data_source", "category", "Computer Vision Data Source chooses whether image-derived features supplement the measured features and where their crops come from. Use them only when image appearance is part of the question."),
            ("12_category_05_embedding_clustering", "category", "Embedding and Clustering selects UMAP or t-SNE, neighborhood behavior, distance metric, compartment, and optional clustering. Compare reproducible settings and do not treat one layout as ground truth."),
            ("12_category_09_umap_display", "category", "UMAP Display controls point styling, crop glyph count and zoom, background, and saved figures. Keep crop glyphs only slightly larger than the points so the embedding remains visible; these display choices do not change fitted coordinates."),
            ("19_hyperparameter_search", "search_card", "Before the full run, use Hyperparameter Search to compare a focused set of UMAP choices. Its small-multiple results expose layouts that are unstable or hide useful structure."),
            ("20_hyperparameter_apply", "dialog", "Select a reviewed search result, then choose Propagate settings. This copies its neighborhood size, minimum distance, and metric into the main Image UMAP form before the full run."),
            ("21_run", "actions", "Run validates the database and crop source, joins and filters numeric features, samples rows, fits the reducer and optional clusters, then writes aligned result records."),
            ("23_output", "figures", "The completed map keeps the UMAP point cloud visible and overlays small transparent microscopy crops at representative positions. Saved outputs include coordinates, cluster labels, the figure, settings, and a reproducibility manifest."),
        ],
        series=1, app_key="umap", section="Results and quality control",
        prerequisite="Complete Measure with valid object features and accessible crop images or merged arrays.",
        objectives=[
            "Choose biologically relevant features and exclusions.",
            "Search a focused parameter range and propagate a reviewed result.",
            "Interpret a readable crop-overlay map and saved outputs.",
        ],
    ),
    captured_lesson(
        16, "activation", "Activation maps",
        "Inspect which image regions influence a trained vision classifier and test whether those explanations are credible.",
        [
            ("01_console", "console", "Activation Maps asks which image regions influence a vision model's prediction. The console records artifact loading, attribution batches, progress, outputs, and reproducibility information."),
            ("12_category_01_model_data", "category", "Model and Data selects the crop archive, trained artifact, architecture, input size, object type, and channel order. These must match training or every attribution is invalid."),
            ("08_all_settings", "settings_controls", "Essentials contains the controls needed for a first run. Use Modified to audit changes, All for validation and quantification controls, and Recipes for repeatable comparisons."),
            ("12_category_02_attribution_method", "category", "Attribution Method chooses Grad-CAM, saliency, integrated gradients, occlusion, or another registered method plus any compatible layer or sampling controls."),
            ("12_category_03_attribution_validation", "category", "Attribution Validation provides deletion, insertion, baseline, and model-randomization checks. A visually pleasing map is not useful if it remains unchanged after model information is destroyed."),
            ("12_category_04_map_display", "category", "Map Display applies training-compatible input preprocessing and controls overlays and grids. Display contrast changes presentation, not evidence that the model learned valid biology."),
            ("12_category_05_map_quantification", "category", "Map Quantification can compare input channels with attribution intensity and save those summaries. Enable it only when the statistic answers a defined question."),
            ("19_hyperparameter_search", "search_card", "Hyperparameter Search compares a small, explicit set of attribution methods and validation settings. Rank the completed trials with deletion, insertion, pointing-game, and sanity evidence together; no single score proves that an explanation is biologically correct."),
            ("21_run", "actions", "Run validates the model and crops, recreates training preprocessing, computes predictions and maps, renders requested grids, and records the settings."),
            ("23_output", "console", "Review maps across correct, incorrect, high-confidence, and low-confidence examples, together with validation checks. Draw biological conclusions only from a validated trained classifier."),
        ],
        series=1, app_key="activation", section="Results and quality control",
        prerequisite="Train and validate a vision classifier and retain its exact crop, channel, architecture, and preprocessing metadata.",
        objectives=[
            "Match the attribution run to the trained model and inputs.",
            "Choose an attribution method with appropriate validation controls.",
            "Interpret maps across representative successes and failures.",
        ],
    ),
    captured_lesson(
        17, "timelapse", "Timelapse",
        "Segment time-resolved images, link selected objects into tracks, and validate the resulting movies and trajectories.",
        [
            ("01_console", "console", "Timelapse segments each frame and links selected objects into stable tracks. The console records preprocessing, masks, tracking attempts, quality control, movies, and reproducibility files."),
            ("12_category_01_input_metadata", "category", "Input and Metadata selects the image folder, assigns acquired channels to object types, and defines how plate, well, field, time, depth, and channel identifiers are parsed."),
            ("08_all_settings", "settings_controls", "Essentials contains the controls needed for a first sequence. Use Modified to audit changes, All for optional axes and backends, and Recipes for a repeated acquisition design."),
            ("12_category_03_image_preprocessing", "category", "Image Preprocessing applies normalization, background handling, denoising, and resizing. Use one consistent transform across all frames so preprocessing does not create artificial motion."),
            ("12_category_04_cell_segmentation", "category", "Configure the segmentation section for each object that must be masked or tracked. Validate model, channel, scale, and core filters on representative early, middle, and late frames."),
            ("12_category_08_quality_control", "category", "Quality Control checks counts, sizes, borders, foreground coverage, splits, and acceptable field failures. Begin in report mode so suspicious frames remain visible."),
            ("12_category_09_tracking_setup", "category", "Tracking Setup chooses the object types, frame range, lifetime filters, and movie rate. Keep short tracks during validation so failures are not hidden."),
            ("12_category_10_tracking_backends", "category", "Tracking Backends offers several linkers with different displacement, gap, division, and worker controls. Choose the simplest backend that matches the motion and verify its identities visually."),
            ("12_category_11_visualization_diagnostics", "category", "Visualization and Diagnostics creates examples for checking segmentation and track continuity, while Output and Storage selects masks, merged arrays, movies, and retained intermediates."),
            ("19_preview_result", "preview", "Open Track Preview on one representative sequence before the full run. Inspect early, middle, and late frames together, then check masks, identities, gaps, divisions, and implausible jumps before propagating the tuned settings."),
            ("21_run", "actions", "Run validates filenames and channels, segments requested objects, links masks, calculates quality control, and writes movies and arrays. Inspect completed tracks for swaps, gaps, and implausible jumps before analysis."),
            ("23_output", "console", "A successful project contains time-indexed masks, stable track identities, quality-control records, movies, merged arrays, saved settings, and a rerunnable manifest."),
        ],
        series=2, app_key="timelapse", section="Core",
        prerequisite="Prepare a time series with reliable time, field, depth, and channel identifiers and known frame interval.",
        objectives=[
            "Map acquisition axes and segment representative frames consistently.",
            "Choose and visually validate an appropriate tracking backend.",
            "Run a small test before producing full tracks and movies.",
        ],
    ),
    captured_lesson(
        18, "motility", "Motility assay",
        "Measure calibrated motion from tracked images and compare trajectories by infection or experimental group.",
        [
            ("01_console", "console", "Motility Assay rebuilds measurements from tracked arrays and calculates displacement, velocity, path length, and straightness. The console records derived tables, quality checks, and figures."),
            ("12_category_01_objects_channels", "category", "Objects and Channels selects the tracked source, moving object, and channels carrying cell, nucleus, and pathogen signals. Incorrect mapping invalidates intensity and infection results."),
            ("08_all_settings", "settings_controls", "Essentials contains the controls needed for a first assay. Use Modified to audit changes, All for optional classifiers and embedding, and Recipes for repeated acquisitions."),
            ("12_category_02_spatial_temporal_calibration", "category", "Spatial and Temporal Calibration supplies the frame interval and pixel scale. Verify both from acquisition metadata because errors scale every physical displacement and velocity."),
            ("12_category_03_motion_filtering", "category", "Motion Filtering removes implausible jumps, broken tracks, and optional feature or velocity outliers. Inspect excluded trajectories so real fast motion is not discarded."),
            ("12_category_04_infection_classification", "category", "Infection Classification can start from pathogen-mask overlap and optionally refine uncertain tracks with intensity evidence. Use the simplest rule supported by controls."),
            ("12_category_05_xgboost_infection_model", "category", "The optional XGBoost, clustering, and embedding sections provide alternative infection classifiers when direct masks are insufficient. Validate any learned rule on held-out labeled tracks."),
            ("19_preview_result", "preview", "Open Track Preview before the full assay. The bounded sample overlays real trajectories and reports displacement, velocity, path length, straightness, gaps, and calibration so a broken tracker or scale is visible before summary plots are written."),
            ("21_run", "actions", "Run measures frames, smooths and filters tracks, assigns infection status, computes calibrated summaries and correlations, and saves tables and figures."),
            ("23_output", "console", "Review per-frame and per-track tables together with trajectory and quality-control figures. Confirm track continuity, calibration units, and group labels before comparing conditions."),
        ],
        series=2, app_key="motility", section="Core",
        prerequisite="Complete Timelapse with reviewed tracks and know the frame interval, pixel calibration, channels, and infection controls.",
        objectives=[
            "Map tracked objects and calibration correctly.",
            "Filter implausible motion without hiding real trajectories.",
            "Validate infection labels and interpret calibrated motility outputs.",
        ],
    ),
    captured_lesson(
        19, "train_cellpose", "Train Cellpose",
        "Fine-tune a Cellpose or CPSAM-compatible model from reviewed image and instance-mask pairs.",
        [
            ("01_console", "console", "Train Cellpose fits a segmentation model from paired microscopy images and instance-label masks. The console reports data validation, progress, checkpoint path, and reproducibility records."),
            ("05_inputs", "console", "Choose the dataset root containing matching training images and masks. Review label quality and reserve representative fields that will not be used to fit the model."),
            ("08_all_settings", "settings_controls", "Essentials contains the controls needed for a first calibration. Use Modified to audit changes, All for preprocessing and runtime options, and Recipes for a repeatable training design."),
            ("12_category_01_starting_point", "category", "Starting Point chooses the model family, pretrained checkpoint, fine-tuning or training from scratch, and output name. Fine-tuning usually needs fewer reviewed images than a random start."),
            ("12_category_02_training_schedule", "category", "Training Schedule sets learning rate, regularization, batches, augmentation, and epochs. Increase training only after image-mask pairs and held-out evaluation are correct."),
            ("12_category_03_image_geometry", "category", "Image Geometry records source and target size, expected object scale, and resizing. Images and masks must receive the same spatial transform while preserving label identities."),
            ("12_category_04_background_denoising", "category", "Background and Denoising applies optional intensity cleanup before training. Use it only when the same preprocessing can be reproduced during model application."),
            ("21_run", "actions", "Run validates matching filenames, previews paired fields, and starts training. Inspect the image above its mask in the figures panel before trusting a long fit."),
            ("23_output", "console", "After training, test candidate checkpoints on held-out fields and inspect failures, not only loss. spaCR saves the checkpoint, settings, run manifest, and paired-field previews together."),
        ],
        series=2, app_key="train_cellpose", section="Segmentation models",
        prerequisite="Create diverse, independently reviewed image-mask pairs and reserve held-out fields for evaluation.",
        objectives=[
            "Validate paired training data and choose a sensible starting checkpoint.",
            "Configure reproducible geometry, preprocessing, and schedule controls.",
            "Evaluate saved checkpoints on held-out images.",
        ],
    ),
    captured_lesson(
        20, "cellpose_masks", "Cellpose Masks",
        "Apply a packaged or custom Cellpose model, inspect masks and flows, and save one instance-label mask per field.",
        [
            ("01_console", "console", "Cellpose Masks applies a packaged model or custom checkpoint to microscopy images. The console records model resolution, preprocessing, progress, diagnostics, and output paths."),
            ("12_category_01_input_channels", "category", "Input and Channels selects the image folder, planes, grayscale or inversion behavior, and optional normalization. Verify the channel first; use preprocessing only when the representative images require it."),
            ("08_all_settings", "settings_controls", "Essentials contains the controls needed for a first application. Use Modified to audit the current model and preprocessing, All for geometry and runtime, and Recipes for repeatable inference."),
            ("12_category_02_model", "category", "Model selects packaged weights or a custom checkpoint and records expected object scale. The checkpoint and channel arrangement should match the images used during training."),
            ("12_category_03_detection_thresholds", "category", "Detection Thresholds controls probability, flow agreement, rescaling, resampling, and hole filling. Adjust them only after confirming the image, channel, model, and scale."),
            ("12_category_04_image_geometry", "category", "Image Geometry can resize fields for inference and restore masks to original dimensions. Preserve aspect ratio and keep pixel scale consistent with training."),
            ("12_category_06_output_runtime", "category", "Output and Runtime controls saved masks, batch size, and diagnostic figures. Reduce the batch when accelerator memory is limited."),
            ("21_run", "actions", "Run loads the checkpoint, applies the configured preprocessing, predicts masks and flows, performs selected cleanup, and saves each label image."),
            ("23_output", "figures", "Inspect representative originals, mask outlines, and flow fields in the figure panel before downstream measurement. Choose normalization and thresholds from those previews rather than copying tutorial values."),
        ],
        series=2, app_key="cellpose_masks", section="Segmentation models",
        prerequisite="Prepare representative microscopy images and choose a model trained for their object type, channels, and scale.",
        objectives=[
            "Match images, channels, object scale, and checkpoint.",
            "Tune preprocessing and thresholds from diagnostic previews.",
            "Save reviewed instance masks for downstream workflows.",
        ],
    ),
    captured_lesson(
        21, "model_compare", "Model Compare",
        "Run two Cellpose configurations on the same representative fields and compare their masks without assuming either is ground truth.",
        [
            ("01_overview", "screen", "Model Compare combines object counts, foreground agreement, matching, fragmentation, and side-by-side overlays so two segmentation candidates can be evaluated on the same fields."),
            ("02_fields_loaded", "source_controls", "Choose a small but representative image folder and limit the field count for interactive comparison. Include easy and difficult examples rather than only clean fields."),
            ("03_models", "model_panels", "Configure Model A and Model B. Keep channels, scale, preprocessing, and shared thresholds matched unless that setting is the intended difference."),
            ("04_compare", "model_panels", "Compare segments every selected field with both resolved configurations. Advanced arguments are recorded, and unsupported parameters are reported instead of silently treated as effective."),
            ("06_parameters", "parameters", "Parameters that reached each model separates honored values from deprecated or ignored controls. Confirm that the comparison actually differs in the intended way."),
            ("07_metrics", "metrics", "The field table reports counts, foreground agreement, matched overlap, splits, merges, unmatched objects, and independent mask-quality checks. No single metric captures every failure mode."),
            ("08_field_one", "previews", "Select a row to inspect both previews. Look for new detections, misses, boundary changes, splits, and merges, then repeat across representative fields."),
            ("09_field_two", "previews", "Choose a model only when the visual differences and metrics improve the errors that matter for the downstream analysis. Equal results are informative and do not demonstrate improvement."),
        ],
        series=2, app_key="model_compare", section="Segmentation models",
        prerequisite="Choose two compatible models or configurations and a representative bounded image set.",
        objectives=[
            "Construct a controlled two-model comparison.",
            "Verify which parameters actually reached each model.",
            "Interpret agreement metrics together with side-by-side masks.",
        ],
    ),
    captured_lesson(
        22, "model_zoo", "Model Zoo",
        "Browse, verify, download, and benchmark reusable segmentation and classifier models.",
        [
            ("01_overview", "catalogue", "Model Zoo lists reusable models available from bundled, local, or declared catalogue sources. Start by matching the task, channels, object scale, and expected preprocessing."),
            ("02_scanned", "scan_controls", "Choose a model folder and scan it. The catalogue keeps local paths and remote metadata together while avoiding duplicate entries."),
            ("03_provenance", "detail", "Select a model to review its provenance, license, checksum, training context, and expected geometry. A model that loads successfully is not automatically suitable for your images."),
            ("04_fields", "field_controls", "Choose a small set of representative fields for a bounded benchmark, including difficult examples."),
            ("05_test", "test", "Test one installed model. The benchmark reports speed, object counts, quality flags, and previews without treating those checks as ground-truth accuracy."),
            ("07_benchmark", "benchmark", "Review the table and mask preview together. Investigate warnings in the image context before accepting or rejecting a model."),
            ("09_two_selected", "compare", "Select two compatible local models when you need a controlled side-by-side evaluation."),
            ("10_compare_handoff", "comparison", "Model Compare receives both checkpoints and the same fields, so differences in masks and quality metrics can be judged on identical inputs."),
        ],
        series=2, app_key="model_zoo", section="Segmentation models",
        prerequisite="Prepare a few representative fields and know their channels, object type, and approximate scale.",
        objectives=["Review model compatibility and provenance.", "Benchmark a model on representative fields.", "Hand two candidates to Model Compare."],
    ),
    captured_lesson(
        23, "agreement", "Annotator Agreement",
        "Measure agreement between annotation columns and inspect the objects responsible for disagreement.",
        [
            ("01_overview", "screen", "Annotator Agreement compares categorical label columns without changing the database. Use it for reviewers, annotation passes, or reviewer-versus-model checks."),
            ("02_database", "source_controls", "Open a measurements database or run folder. spaCR discovers annotation and prediction columns that refer to the same objects."),
            ("03_columns", "columns", "Select at least two columns. Additional columns enable pairwise comparisons and a multi-rater summary."),
            ("04_compute", "compute", "Compute Agreement aligns labels by object. Missing labels are reported as abstentions rather than silently counted as disagreements."),
            ("06_pairwise", "pairwise", "Read raw agreement, class balance, abstentions, and kappa together. Kappa can be undefined when a column contains only one class; that is a prevalence warning, not zero agreement."),
            ("07_confusion", "confusion", "Use the confusion matrix to see which classes are confused and whether imbalance dominates the summary."),
            ("08_disagreements", "review", "The disagreement table identifies the exact objects that need review."),
            ("09_crop_one", "crop", "Inspect representative crops before changing labels or class definitions. Aggregate statistics cannot explain why reviewers disagreed."),
        ],
        series=3, app_key="agreement", section="Results and quality control",
        prerequisite="Create two or more categorical columns for the same objects.",
        objectives=["Configure a fair label comparison.", "Interpret agreement and imbalance together.", "Review the objects behind disagreements."],
    ),
    captured_lesson(
        24, "plaque", "Plaque Assay",
        "Segment or reuse plaque masks, validate them, and summarize plaque formation by condition.",
        [
            ("01_console", "console", "Plaque Assay measures cleared regions in plate images. The console records resolved settings, progress, warnings, and output paths."),
            ("05_inputs", "inputs", "Choose the plaque-image folder and decide whether to segment source images or reuse correctly named label masks."),
            ("06_essentials", "settings_controls", "Essentials contains the controls needed for a first run. Modified audits deliberate changes, while All exposes advanced troubleshooting controls."),
            ("12_category_02_model", "category", "Match the model and expected plaque scale to representative images rather than copying a tutorial value."),
            ("12_category_03_detection_thresholds", "category", "Detection settings affect candidate regions, shape consistency, resampling, and cleanup. Tune them from several control and treatment wells."),
            ("12_category_05_background_denoising", "category", "Use background correction or denoising only when previews show that the acquisition requires it."),
            ("21_run", "actions", "Run after checking source, model, scale, and output behavior. Start with a small validation set when segmentation is being recalibrated."),
            ("23_output", "figures", "Inspect plaque overlays and distributions before comparing conditions. The database, summaries, statistics, and quality-control figures remain linked to the run."),
        ],
        series=3, app_key="analyze_plaques", section="Toxoplasma assays",
        prerequisite="Prepare plaque images, controls, plate metadata, and pixel calibration.",
        objectives=["Choose between segmentation and existing masks.", "Tune plaque detection from previews.", "Validate and locate assay outputs."],
    ),
    captured_lesson(
        25, "recruitment", "Recruitment",
        "Quantify enrichment of a marker at parasite-associated regions relative to host-cell background.",
        [
            ("01_console", "console", "Recruitment reads existing measurements and calculates marker enrichment at parasite-associated regions. It does not segment images again."),
            ("05_inputs", "inputs", "Choose the experiment root containing the measurements database and merged fields."),
            ("06_essentials", "settings_controls", "Essentials contains source, channel mapping, filters, plate labels, and diagnostic controls. Modified provides a compact pre-run audit."),
            ("12_category_02_mask_channel_mapping", "category", "Map each intensity and mask plane carefully, then choose the channel of interest. A valid but incorrect channel index can still produce plausible ratios."),
            ("12_category_03_object_filtering", "category", "Object filters remove implausible cells, nuclei, and parasites. Choose gates from distributions and overlays rather than fixed tutorial values."),
            ("12_category_04_plate_layout_controls", "category", "Map conditions and controls to the correct plate identifiers before interpreting group differences."),
            ("21_run", "actions", "Run joins measurements, applies the reviewed gates, calculates recruitment, groups wells, and produces diagnostics."),
            ("23_output", "figures", "Inspect segmentation overlays and control distributions before trusting condition summaries. Object- and well-level outputs keep their source identifiers."),
        ],
        series=3, app_key="recruitment", section="Toxoplasma assays",
        prerequisite="Measure the relevant compartments and prepare plate metadata with appropriate controls.",
        objectives=["Map channels and masks correctly.", "Choose defensible object filters and controls.", "Validate recruitment with images and distributions."],
    ),
    captured_lesson(
        26, "invasion", "Invasion Assay",
        "Use an outside-versus-total stain to distinguish attached from invaded parasites.",
        [
            ("01_console", "console", "Invasion Assay scores measured parasites from a two-colour staining design. The console records settings, quality checks, and outputs."),
            ("05_inputs", "inputs", "Choose an experiment containing parasite measurements, stain channels, controls, and plate metadata."),
            ("06_essentials", "settings_controls", "Essentials shows the inputs, channel mapping, thresholding, controls, filters, and outputs needed for a first analysis."),
            ("12_category_02_channels_intensity", "category", "Map the outside stain and total parasite stain in the correct order. Swapping them reverses the biological interpretation without necessarily causing an error."),
            ("12_category_03_thresholding", "category", "Derive classification thresholds from appropriate controls and inspect sensitivity around the chosen boundary."),
            ("12_category_04_controls_minimum_counts", "category", "Define staining controls and minimum evidence for fields and wells so weak groups are reported rather than overinterpreted."),
            ("21_run", "actions", "Run after verifying stain mapping, controls, object filters, and condition metadata."),
            ("23_output", "figures", "Review attached, invaded, and uncertain examples together with well-level summaries. Exported results retain the assay settings and source identifiers."),
        ],
        series=3, app_key="invasion", section="Toxoplasma assays",
        prerequisite="Prepare a validated outside-versus-total staining assay with controls and parasite measurements.",
        objectives=["Map stain channels correctly.", "Calibrate the classification from controls.", "Review examples and well-level invasion results."],
    ),
    captured_lesson(
        27, "replication", "Replication Assay",
        "Assign parasites to vacuoles and summarize intracellular replication by condition.",
        [
            ("01_console", "console", "Replication Assay scores parasite counts per vacuole from existing relationships and measurements. The console records settings, filtering, and outputs."),
            ("05_inputs", "inputs", "Choose the experiment root and confirm that host cells, vacuoles, and parasites have stable identifiers."),
            ("06_essentials", "settings_controls", "Essentials presents assignment, condition, filtering, scoring, and output controls. Use Modified to audit deliberate changes."),
            ("12_category_02_vacuole_assignment", "category", "Choose how parasites are assigned to vacuoles and how ambiguous or distant relationships are handled."),
            ("12_category_04_object_filtering", "category", "Filter segmentation failures from reviewed distributions and images, without erasing uncommon but valid vacuoles."),
            ("12_category_05_replication_scoring", "category", "Define the parasite-count groups and summaries that answer the biological question before running statistics."),
            ("21_run", "actions", "Run after checking assignments, conditions, filters, and scoring definitions."),
            ("23_output", "figures", "Inspect vacuoles across the count range and review merged or fragmented masks. The saved distributions and statistics remain linked to object-level evidence."),
        ],
        series=3, app_key="replication", section="Toxoplasma assays",
        prerequisite="Measure host cells, vacuoles, and parasites with reviewed relationships and condition metadata.",
        objectives=["Configure vacuole assignment.", "Define defensible filters and score groups.", "Validate replication summaries against images."],
    ),
    captured_lesson(
        28, "training_runs", "Training Runs",
        "Compare model-training runs by overlaying curves and diffing settings.",
        [
            ("01_overview", "screen", "Training Runs compares several fits without reducing model choice to one loss value."),
            ("02_source", "source", "Choose a folder or registry containing compatible training runs and scan it."),
            ("03_discovered", "runs", "Review discovered runs, available metrics, incomplete records, and warnings before selecting candidates."),
            ("06_controls", "metrics", "Choose the metric, smoothing, fold display, and the settings fields that should be compared."),
            ("08_accuracy", "plot", "Overlay validation behavior to compare convergence, stability, and overfitting on the same axes."),
            ("10_best_last", "picked", "Compare the best checkpoint with the final checkpoint when late training may have degraded generalization."),
            ("14_fold_mean", "plot", "Fold summaries reveal variation hidden by a single aggregate curve."),
            ("16_per_fold", "diff", "Use per-fold curves and the settings diff with held-out performance to select a reproducible run."),
        ],
        series=4, app_key="train_compare", section="Results and quality control",
        prerequisite="Complete at least two compatible training runs with recorded metrics and settings.",
        objectives=["Load and validate comparable runs.", "Interpret curves across checkpoints and folds.", "Relate performance differences to setting changes."],
    ),
    captured_lesson(
        29, "report", "Report",
        "Create a shareable HTML or PDF summary of quality control, figures, statistics, settings, and versions.",
        [
            ("01_overview", "screen", "Report assembles a traceable project summary rather than asking you to copy figures and settings by hand."),
            ("02_source", "source", "Choose a completed project or run folder and scan its available evidence."),
            ("03_verdict", "verdict", "Review the detected quality verdict and missing inputs before choosing report sections."),
            ("04_sections", "sections", "Include only the quality checks, figures, statistics, settings, and provenance needed by the audience."),
            ("05_options", "options", "Set title, authorship, output format, figure limits, and any privacy-sensitive exclusions."),
            ("07_generate", "generate", "Generate after resolving missing files and failed quality checks that would make the document misleading."),
            ("11_report_top", "report", "Preview the finished report and verify conclusions, versions, and links before sharing it."),
        ],
        series=4, app_key="report", section="Results and quality control",
        prerequisite="Complete and review the runs that should appear in the report.",
        objectives=["Select relevant report evidence.", "Resolve quality and provenance gaps.", "Generate and review a portable report."],
    ),
    captured_lesson(
        30, "plate_queue", "Plate Queue",
        "Run multiple plates through the same validated pipeline with visible, resumable status.",
        [
            ("02_open_queue", "screen", "Plate Queue applies one validated workflow across multiple plates while keeping each plate's state separate."),
            ("03_add_current", "add", "Add the current configured module or a prepared plate source to the queue."),
            ("05_folder_drop", "table", "Add further plates and review their source, destination, settings, and order in the table."),
            ("06_persistence", "run_controls", "Choose worker limits, stop behavior, and resume policy. Queue state is persisted so an interruption does not erase the plan."),
            ("07_run", "run", "Test the workflow on a bounded plate or field before starting the remaining queue."),
            ("08_running", "status", "During execution, follow per-plate progress and open logs when one item warns or fails."),
            ("12_finished", "table", "Finished plates retain isolated outputs and run records; stopped or failed items remain identifiable and resumable."),
        ],
        series=4, app_key="queue", section="Data and batch runs",
        prerequisite="Validate the module sequence and settings on representative data first.",
        objectives=["Build and review a multi-plate queue.", "Run and monitor it safely.", "Resume or diagnose individual plates."],
    ),
    captured_lesson(
        31, "external_masks", "External Masks",
        "Import third-party images and label masks as a measured, traceable spaCR project.",
        [
            ("01_overview", "screen", "External Masks imports source images and externally generated labels, then prepares the project for normal spaCR measurement and annotation."),
            ("02_detected", "input_table", "Add images and mask sources, then verify that spaCR detected the intended roles."),
            ("03_assignment", "input_table", "Map every input to its object class. Check filenames, dimensions, and field pairing instead of relying on an inferred assignment."),
            ("04_category_input_mapping", "category", "Input Mapping controls channel and mask-plane ordering, naming, and destination layout."),
            ("08_category_measurements", "category", "Measurement settings define which objects and features should be produced after import."),
            ("13_preview_settings", "preview_only", "Keep Preview enabled for the first run and choose a new destination."),
            ("15_preview_result", "console", "Review pairing, channel counts, object classes, warnings, and blocking errors before writing anything."),
            ("18_finished", "console", "Run the reviewed import. The resulting images, masks, measurements, and provenance are ready for downstream annotation."),
        ],
        series=4, app_key="external_masks", section="Data and batch runs",
        prerequisite="Prepare source images, integer label masks, naming rules, and object-class definitions.",
        objectives=["Map external inputs correctly.", "Validate the import in Preview.", "Create a measured project with provenance."],
    ),
    captured_lesson(
        32, "align_stitch", "Align and Stitch",
        "Register image tiles and write a stitched mosaic with bounded memory.",
        [
            ("01_overview", "screen", "Align and Stitch separates registration planning from writing so placement and resource estimates can be reviewed first."),
            ("02_source", "source", "Choose the tile folder and destination, keeping source acquisitions unchanged."),
            ("03_layout_settings", "layout_settings", "Describe grid order, overlap, stage information, and a reference channel that contains useful structure."),
            ("04_quality_settings", "quality", "Configure match confidence, neighboring constraints, blending, and the memory budget for output bands."),
            ("05_plan_run", "plan", "Plan estimates pairwise offsets and solves accepted constraints without creating the mosaic."),
            ("06_plan_result", "report", "Inspect registered and fallback tiles, residuals, confidence, canvas size, and write estimate."),
            ("07_tile_detail", "tile_detail", "Select weak or isolated tiles to understand their solved position before committing the output."),
            ("10_finished", "report", "Write only after the plan is acceptable. spaCR saves the mosaic, transforms, coordinates, and registration report."),
        ],
        series=4, app_key="align", section="Data and batch runs",
        prerequisite="Prepare tiled acquisitions with known order, overlap or stage positions, and a useful registration channel.",
        objectives=["Describe the tile layout.", "Audit a registration plan.", "Write a traceable bounded-memory mosaic."],
    ),
    captured_lesson(
        33, "plate_viewer", "Plate Viewer",
        "Explore measurements as plate heatmaps and detect spatial artifacts or edge effects.",
        [
            ("01_overview", "screen", "Plate Viewer turns object measurements into well-level heatmaps with spatial quality checks beside them."),
            ("02_database", "source", "Open a measurements database or run folder. The source is queried read-only."),
            ("03_measurement", "selection", "Choose the object table, numeric feature, and plate to display."),
            ("04_options", "options", "Select aggregation, color scaling, and the minimum evidence required for a well. Count is useful for spotting segmentation or seeding differences."),
            ("05_render", "render", "Render the view, then compare the heatmap with row, column, ring, and edge diagnostics."),
            ("07_well_detail", "well_detail", "Select a well to inspect its identifier, object count, displayed value, and position."),
            ("11_filtered_result", "result", "Wells below the evidence threshold become explicit blanks rather than misleading zeros."),
            ("13_export", "export", "Export the current tidy well grid with counts and spatial annotations for audit or replotting."),
        ],
        series=4, app_key="plate_view", section="Results and quality control",
        prerequisite="Measure a plate with stable plate and well identifiers.",
        objectives=["Build an appropriate plate heatmap.", "Recognize spatial and low-count artifacts.", "Inspect and export traceable well summaries."],
    ),
    captured_lesson(
        34, "database", "Database Browser",
        "Browse, filter, and export spaCR measurement databases without a command line.",
        [
            ("01_overview", "screen", "Database Browser inspects large spaCR databases in a bounded, read-only view."),
            ("02_open", "source", "Choose a measurements database or its run folder and open it."),
            ("03_tables", "tables", "Select a table after reviewing its purpose and read-only status."),
            ("04_cell_table", "preview", "The preview loads a bounded page while reporting the complete row and column counts separately."),
            ("05_column_search", "column_search", "Search columns to make wide feature tables readable without changing the underlying rows."),
            ("08_filter", "filter", "Build a structured filter. Column names and operators are validated, and values are bound safely."),
            ("09_filtered", "status", "The exact filtered count describes the full query even though only one page is drawn."),
            ("10_export", "export", "Export the complete current filter and visible-column selection. The source database remains unchanged."),
        ],
        series=4, app_key="db_browser", section="Data",
        prerequisite="Have a spaCR SQLite database or run folder available.",
        objectives=["Open and page through a database safely.", "Filter a complete table without confusing it with the preview.", "Export a traceable subset."],
    ),
    captured_lesson(
        35, "converter", "Format Converter",
        "Convert microscopy formats into spaCR-compatible TIFF names with a reviewed provenance map.",
        [
            ("01_overview", "screen", "Format Converter uses a preview-first workflow so naming and dimensional mapping are visible before files are written."),
            ("02_source", "source", "Choose source files and a separate destination."),
            ("03_options", "options", "Describe folder layout, series, time, depth, plate naming, and resume behavior. Projection choices discard information, so select them deliberately."),
            ("04_preview", "preview", "Preview scans headers and builds the conversion plan without writing images."),
            ("05_plan", "mapping", "Review each source beside its target plate, well, field, channel, time, and depth identifiers."),
            ("06_convert", "convert", "Convert only when the mapping is correct and collision checks pass. Sources are never renamed or replaced."),
            ("07_result", "summary", "The result includes standardized TIFFs and a provenance map back to every original file and identifier."),
            ("08_resume", "resume", "Resume validates completed field checkpoints before skipping them, so missing or damaged outputs are rebuilt."),
        ],
        series=4, app_key="convert", section="Data",
        prerequisite="Prepare supported microscopy files and decide how acquisition dimensions map to spaCR identifiers.",
        objectives=["Configure a loss-aware mapping.", "Validate the complete preview plan.", "Convert or resume without altering sources."],
    ),
    captured_lesson(
        36, "import", "Import Project",
        "Map another laboratory's images, masks, and measurements into a traceable spaCR project.",
        [
            ("01_overview", "screen", "Import Project separates inferred mappings from reviewed decisions before creating a spaCR project."),
            ("02_images", "images", "Choose external images and a new destination, preserving original files."),
            ("03_masks", "mask_list", "Add each mask class and verify that every field has the expected matching labels."),
            ("04_measurements", "measurements", "Add measurements, calibration, identifiers, and a conflict policy for foreign column names."),
            ("05_preview", "preview", "Preview checks headers, field pairing, labels, joins, units, and name conflicts without writing project data."),
            ("06_plan", "mapping", "Review every proposed target column, transform, unit, and join warning."),
            ("07_edit_mapping", "target_cell", "Correct inferred mappings directly and rerun validation until the plan reflects the source experiment."),
            ("11_import", "import", "Import only the reviewed plan. spaCR writes canonical images, masks, measurements, and provenance while retaining external identifiers."),
        ],
        series=5, app_key="foreign", section="Data",
        prerequisite="Prepare source images, optional masks, measurements, identifiers, and physical calibration.",
        objectives=["Map foreign data explicitly.", "Resolve joins, units, and naming conflicts in Preview.", "Create a standardized project with provenance."],
    ),
    lesson(37, "batch", "Batch Runner", "Queue modules, projects, and settings for reproducible unattended execution.", series=5, app_key="batch", section="Data and batch runs", inputs="Add projects and assemble the ordered module sequence each project should run.", settings="choose presets, dependencies, resource limits, resume behavior, logging, and failure policy.", run="validate the batch plan, test one job, and then run the remaining queue overnight or unattended.", outputs="every job has its own status, settings snapshot, logs, outputs, and resumable run record."),
    lesson(38, "distributed_jobs", "Distributed Jobs", "Submit and monitor spaCR work on SSH workstations, Slurm clusters, or cloud and HPC commands.", series=5, app_key="distributed_jobs", section="Data and batch runs", inputs="Configure a remote profile, connection method, working directory, environment, and data paths.", settings="define scheduler resources, command templates, file synchronization, credentials, and status polling.", run="submit a small validation job before scaling to expensive analyses.", outputs="local records track remote job identifiers, logs, state, transferred artifacts, and failures."),
    lesson(39, "classifier_evaluation", "Classifier Evaluation", "Audit held-out predictions, nested cross-validation, calibration, leakage, and per-plate performance.", series=5, app_key="classifier_evaluation", section="Results and quality control", inputs="Choose predictions with ground truth, split metadata, plate identifiers, probabilities, and model-run records.", settings="configure class order, thresholds, calibration bins, grouping, bootstrap uncertainty, and leakage checks.", run="evaluate held-out and per-plate behavior before using a classifier for biological conclusions.", outputs="spaCR produces discrimination, calibration, confusion, leakage, and robustness reports."),
    lesson(40, "run_history", "Run History", "Search every job's settings, files, warnings, failures, and performance from one place.", series=5, app_key="run_history", section="Results and quality control", inputs="Open the run registry for the current workspace or selected project roots.", settings="filter by module, status, date, project, version, warning, duration, or resource use.", run="open a run to inspect its immutable settings, logs, outputs, and failure context.", outputs="you can reproduce, compare, resume, or report prior work without guessing how it was run."),
    lesson(41, "classify", "Classify", "Choose one classification workflow for image crops or measured features and evaluate it without leaking related objects across splits.", series=5, app_key="classify_merged", section="Core", inputs="Choose the project and labels before selecting the representation the model will use.", settings="use Classifier family as the top-level choice: Computer Vision trains on object crops, while Machine Learning trains on measured features.", run="run a small grouped validation before committing to expensive training; the dedicated Classify CV and Classify ML lessons demonstrate their bounded Hyperparameter Search workbenches.", outputs="the console and run record link the model, predictions, split records, metrics, and settings."),
    lesson(42, "curate", "Curate", "Correct segmentation masks and tracks by hand while preserving an auditable record of every edit.", series=5, app_key="curate", section="Core", inputs="Open a project and select the fields, mask class, or tracks that need review.", settings="choose the editing tool, label behavior, navigation, and the review scope before changing data.", run="make a small correction, inspect neighboring frames or objects, and save only when the edit is correct.", outputs="corrected labels and track changes are stored with provenance so downstream measurements can be refreshed."),
    lesson(43, "illumination", "Illumination", "Estimate and validate flat-field correction before intensity features are measured.", series=5, app_key="illumination", section="Data", inputs="Choose representative merged fields and the channels that need correction.", settings="review sampling, smoothing, exclusion, and quality-control options for each channel.", run="preview the estimated illumination fields and reject corrections that follow biological structure.", outputs="approved correction fields and diagnostics are saved for reproducible preprocessing."),
    lesson(44, "data_manager", "Data Manager", "Inspect project storage and reclaim derived files without touching original acquisitions.", series=5, app_key="data_manager", section="Data", inputs="Select a project or workspace and let spaCR inventory its artifacts.", settings="review artifact type, size, provenance, rebuildability, and dependency warnings before selecting anything.", run="preview the cleanup plan and remove only derived artifacts you are prepared to rebuild.", outputs="the refreshed inventory and cleanup record show what remains and what can be regenerated."),
    lesson(45, "project_browser", "Project Browser", "Find spaCR projects and see their stage, size, latest run, and stale outputs in one table.", series=5, app_key="project_browser", section="Data", inputs="Choose the roots that contain your projects and scan them.", settings="filter and sort by stage, size, age, module status, or stale artifacts.", run="open a selected project or its latest run after checking the status summary.", outputs="the browser links you to the relevant files and modules without altering project data."),
    lesson(46, "napari_bridge", "Napari Bridge", "Review a mask in napari and return corrected labels to spaCR with provenance.", series=5, app_key="napari_bridge", section="Segmentation models", inputs="Choose a project, field, and mask plane to inspect.", settings="confirm label class, source image layers, destination, and overwrite safeguards.", run="edit labels in napari, return them to spaCR, and review the change before acceptance.", outputs="corrected masks and the curation record remain connected to the original field."),
    lesson(47, "barcode_qc", "Barcode QC", "Check mapping depth, coverage, collisions, positional effects, and an evidence-based abundance threshold.", series=5, app_key="barcode_qc", section="Results and quality control", inputs="Open a completed barcode-mapping run with plate and library metadata.", settings="review intended guide representation, controls, grouping, and threshold sweep options.", run="generate the quality-control views and inspect weak wells, unmapped reads, and library coverage together.", outputs="the report records the selected abundance threshold, supporting diagnostics, and affected wells."),
    lesson(48, "hit_list", "Hit List", "Rank and review screen hits using effect size, uncertainty, false-discovery rate, and guide agreement.", series=5, app_key="hit_list", section="Results and quality control", inputs="Choose a completed analysis and the result columns that define the screen outcome.", settings="configure controls, direction, significance, effect-size, annotation, and agreement filters.", run="preview the ranked table and inspect individual candidates before exporting a final subset.", outputs="the filtered hit list retains scores, evidence, annotations, and source identifiers."),
    lesson(49, "methods_results", "Methods & Results", "Draft traceable methods and results text from recorded runs rather than reconstructing settings from memory.", series=5, app_key="methods_export", section="Results and quality control", inputs="Select the projects and completed runs that should support the document.", settings="choose sections, comparisons, figures, statistics, citation style, and privacy-sensitive fields.", run="generate a draft and follow each statement back to its setting, file, or computed result.", outputs="editable methods and results documents are exported with a provenance summary."),
    lesson(50, "run_compare", "Run Compare", "Place two runs side by side to understand changes in settings, counts, outputs, and hits.", series=5, app_key="run_compare", section="Results and quality control", inputs="Select two compatible run records.", settings="choose which settings, artifacts, counts, and result tables should be compared.", run="compute the comparison and inspect material differences before deciding that one run improved the analysis.", outputs="the comparison report links every difference back to both source runs."),
    lesson(51, "control_charts", "Control Charts", "Track controls across plates and detect drift against a stated baseline.", series=5, app_key="control_chart", section="Results and quality control", inputs="Choose repeated control measurements with plate order and baseline metadata.", settings="define the baseline, grouping, center line, limits, and alert policy.", run="render the chart and investigate shifts, trends, or out-of-control plates in context.", outputs="the chart, alerts, and baseline definition can be exported for campaign monitoring."),
    lesson(52, "pipeline_graph", "Pipeline Graph", "See which artifacts produced each result and which downstream outputs are stale or missing.", series=5, app_key="pipeline_graph", section="Explore", inputs="Open a spaCR project or registered run collection.", settings="filter the graph by module, artifact type, status, or workflow branch.", run="select nodes and edges to inspect their settings, files, dependencies, and stale reasons.", outputs="the graph identifies the exact module that should be rerun to restore consistency."),
    lesson(53, "prediction_profiler", "Prediction Profiler", "Vary one model input and observe how a fitted prediction responds.", series=5, app_key="profiler", section="Explore", inputs="Choose a fitted model, a representative row, and the feature to vary.", settings="set a defensible feature range while holding or sampling other inputs deliberately.", run="generate the profile and compare it across relevant classes or reference objects.", outputs="the response curve describes model behavior, not causal biological effect, and can be exported with its settings."),
    lesson(54, "qc_dashboard", "QC Dashboard", "Combine segmentation, unit, leakage, plate-effect, and annotation checks into one review surface.", series=5, app_key="qc_dashboard", section="Explore", inputs="Choose the project and runs whose quality evidence should be summarized.", settings="review which checks are available, their scope, and the criteria behind each verdict.", run="refresh the dashboard and open warning cards to inspect the underlying evidence.", outputs="the combined verdict remains traceable to individual reports rather than replacing them."),
    lesson(55, "image_scatter", "Image Scatter", "Plot measurements and inspect the image behind any point.", series=5, app_key="image_scatter", section="Explore", inputs="Choose a measurement table, x and y features, and the crop or field image column.", settings="configure colour, grouping, filters, sampling, and axis transforms without hiding excluded rows.", run="render the scatter, hover for image context, and click representative or unusual points.", outputs="selected objects and the plot can be exported with stable source identifiers."),
    lesson(56, "lineage", "Lineage", "Inspect containment relationships such as cell, nucleus, pathogen, and organelle assignments.", series=5, app_key="lineage", section="Explore", inputs="Open measured objects with relationship identifiers.", settings="choose parent and child classes, filters, fields, and the relationship attributes to display.", run="inspect unexpected missing, duplicate, or cross-boundary relationships on representative images.", outputs="the reviewed relationship table helps distinguish biological structure from assignment errors."),
    lesson(57, "layer_viewer", "Layer Viewer", "Inspect images, masks, points, and regions of interest as aligned layers.", series=5, app_key="layer_viewer", section="Explore", inputs="Open a project or compatible image and annotation layers.", settings="choose layer visibility, channels, colors, opacity, contrast, and navigation.", run="toggle layers and inspect boundaries and coordinates before drawing conclusions from an overlay.", outputs="the view and selected regions can be exported without modifying sources unless an edit is explicitly saved."),
    lesson(58, "graph_builder", "Graph Builder", "Build exploratory charts by assigning table columns to visual roles.", series=5, app_key="graph_builder", section="Explore", inputs="Choose a measurement table and drag features onto x, y, colour, size, or facet roles.", settings="select geometry, aggregation, filtering, transforms, and grouping appropriate to the question.", run="render the chart and inspect counts, missing values, and source rows behind patterns.", outputs="the chart specification and data selection can be reused or exported."),
    lesson(59, "anndata_export", "AnnData Export", "Write spaCR measurements as an AnnData file for scanpy or scvi-tools.", series=5, app_key="anndata_export", section="Explore", inputs="Choose the measurement table, object type, features, observations, and metadata columns.", settings="review joins, missing values, feature scaling, sparse storage, and stable identifiers.", run="preview dimensions and metadata before writing the file.", outputs="the h5ad file preserves the selected matrix, annotations, and provenance for downstream analysis."),
    lesson(60, "pca", "PCA", "Summarize correlated measurements with principal components and inspect the features driving them.", series=5, app_key="pca", section="Explore", inputs="Choose a measurement table, numeric features, and optional labels or groups.", settings="configure filtering, missing-value handling, scaling, components, and sampling.", run="fit PCA and inspect scores, explained variance, loadings, and image-linked outliers together.", outputs="scores, loadings, settings, and plots can be exported for reproducible exploration."),
    lesson(61, "tabulate", "Tabulate", "Pivot measurements into grouped summaries while keeping the contributing count visible.", series=5, app_key="tabulate", section="Explore", inputs="Choose a table and assign grouping fields, value columns, and aggregations.", settings="review filters, missing values, category order, totals, and the count behind each cell.", run="build the table and inspect sparse or unexpectedly empty groups before export.", outputs="the pivoted table and its specification remain linked to the source selection."),
    lesson(62, "feature_dictionary", "Feature Dictionary", "Find what a measured feature means by name or concept.", series=5, app_key="feature_dict", section="Explore", inputs="Search for a feature name, measurement family, object type, or biological idea.", settings="filter by compartment, channel, unit, computation, or availability in the current project.", run="open an entry to review its definition, expected units, dependencies, and caveats.", outputs="the selected definitions can be copied or exported alongside an analysis."),
    lesson(63, "small_multiples", "Small Multiples", "Compare the same chart across groups on a grid with honest shared axes.", series=5, app_key="trellis", section="Explore", inputs="Choose a table, chart variables, and the field that defines each panel.", settings="configure layout, shared or independent scales, aggregation, filtering, and panel order.", run="render the grid and check sample sizes and scale choices before comparing panels.", outputs="the trellis specification and figure can be exported with the selected data."),
    captured_lesson(
        64, "gate_editor", "Gate Editor",
        "Draw, inspect, nest, and reuse population gates in 2D, 3D, or a reduced high-dimensional view.",
        [
            ("01_overview", None, "Gate Editor turns regions of a measurement table into named, reusable population gates. The workspace keeps the feature axes, drawing tools, full data view, and gate hierarchy together."),
            ("02_two_d_gate", "two_d_workspace", "In 2D, choose X and Y for a scatter, or leave Y empty for a histogram. Select a threshold or shape tool and draw around the intended population. The full cloud remains visible while included objects are highlighted."),
            ("03_three_d_volume", "three_d_workspace", "Choose 3D and a Z measurement to inspect a rotatable volume. The axis buttons control the spin direction, and Box gate records bounds on all three displayed measurements."),
            ("04_xd_projection", "xd_workspace", "Choose xD to project the eligible measurements into principal components. The source label reports explained variance, and PC1, PC2, and PC3 become ordinary axes for the same gating tools."),
            ("05_gate_hierarchy", "gate_tree", "The gate hierarchy reports the object count, percentage of its parent, and percentage of all rows. Toggle a gate's check box to show or hide it, and select a gate before drawing when the next population should be nested inside it."),
            ("06_save_reuse", "save_controls", "Give each population a meaningful name, then save the gate strategy. Load it on a compatible table to reuse the same feature names, boundaries, and hierarchy, and verify the new counts before continuing."),
            ("07_cluster_walk", "walk_result", "Cluster Walk searches a bounded sequence of DBSCAN radii on the displayed measurements and reports the selected radius, population count, and excluded fraction. Inspect the resulting gates, then annotate or export only after the populations match the visible data."),
        ],
        series=5, app_key="gate_editor", section="Explore",
        prerequisite="Prepare a measurement table with numeric features and stable object identifiers.",
        objectives=[
            "Draw and inspect a population gate in 2D.",
            "Compare the real 3D volume and xD projection modes.",
            "Read, nest, save, and safely reuse a named gate hierarchy.",
            "Verify a bounded cluster Walk before annotation or export.",
        ],
    ),
    lesson(65, "feature_explorer", "Feature Explorer", "Rank features by how well they separate selected classes and inspect the evidence behind the ranking.", series=5, app_key="feature_explorer", section="Explore", inputs="Choose a table, class column, compared groups, and candidate numeric features.", settings="review scoring, validation, filtering, sampling, and multiple-comparison controls.", run="compute the ranking and inspect distributions and image-linked examples for leading features.", outputs="the ranked table and plots describe association with the classes, not causal importance."),
    lesson(66, "outliers", "Outliers", "Flag unusual objects or wells with robust rules while preserving every row.", series=5, app_key="outliers", section="Explore", inputs="Choose the table, analysis level, grouping, and features to assess.", settings="select a robust method, direction, threshold, missing-value handling, and reference groups.", run="preview flagged rows and inspect their measurements and images before accepting the rule.", outputs="spaCR writes an outlier flag and diagnostics rather than silently deleting observations."),
    lesson(67, "experiment_design", "Experiment Design", "Lay out conditions, controls, and replicates before acquisition.", series=5, app_key="experiment_design", section="Design", inputs="Choose the plate format, conditions, controls, replicate plan, and constraints.", settings="review randomization, blocking, edge handling, balance, reserved wells, and metadata labels.", run="generate and inspect the proposed layout, then revise conflicts or imbalances.", outputs="the validated plate map and metadata can be exported for acquisition and later spaCR analysis."),
    lesson(68, "power_design", "Power / Design", "Estimate cells and wells needed to detect an effect under stated assumptions.", series=5, app_key="power", section="Design", inputs="Enter the outcome type, expected effect, variability, grouping, and planned analysis.", settings="review significance, target power, clustering, attrition, and uncertainty ranges.", run="calculate the design and use sensitivity views to see which assumptions drive sample size.", outputs="the report records assumptions and recommended sampling rather than presenting one number without context."),
    lesson(69, "dose_response", "Dose–Response", "Fit dose-response curves while reporting uncertainty and refusing unsupported EC50 estimates.", series=5, app_key="dose_response", section="Design", inputs="Choose concentration, response, replicate, condition, and control columns.", settings="review normalization, direction, bounds, weighting, grouping, and minimum evidence requirements.", run="fit and inspect the curve, residuals, confidence interval, and any refusal or one-sided bound.", outputs="curve parameters, uncertainty, diagnostics, and source data can be exported together."),
    lesson(70, "explain_cv", "Explain CV Model", "Reproduce computer-vision predictions from measured features and audit whether the resulting explanations are faithful and leakage-free.", series=5, app_key="explain_cv", section="Results and quality control", inputs="Choose the per-object prediction file, its measurements database, the predicted class or score, and the microplate-well or plate grouping that must remain intact.", settings="select the surrogate-model backend, reserved validation groups, feature exclusions, permutation repeats, and the bounded SHAP budget.", run="validate the leakage exclusions and grouped split, then fit the surrogate model and compare its validation fidelity with the majority baseline before interpreting importance.", outputs="the run records fidelity, validation-set permutation importance, SHAP summaries, excluded columns, grouping, backend, and provenance; weak fidelity is reported rather than hidden."),
    lesson(71, "investigate_hit", "Investigate Hit", "Return one selected regression finding to candidate cells and quantitative microplate-well evidence without overwriting reviewed annotations.", series=5, app_key="investigate_hit", section="Results and quality control", inputs="Choose the regression run, selected gene or guide-level finding, phenotype direction, false-discovery threshold, guide support, and matching measurements database.", settings="review the object table, score column, microplate-well grouping, evidence thresholds, covariates, and the optional cross-fitted hierarchical mixture.", run="preview the score-ranked candidate cells and microplate-well summaries, then run the cross-fitted model only when the selected finding and evidence contract are correct.", outputs="spaCR saves ranked candidate cells, independent-well comparisons, optional probabilities that each cell matches the selected phenotype, uncertainty, and a versioned call that never replaces hand annotations."),
    lesson(72, "volcano_explorer", "Volcano Explorer", "Open a completed regression, interrogate every point, restyle the plot, and export a publication-sized vector figure without rerunning the analysis.", series=5, app_key="volcano_explorer", section="Results and quality control", inputs="Choose a completed regression folder or results CSV and confirm the outcome, guide-level table, support family, effect column, and adjusted P-value column.", settings="review significance and effect thresholds, axis transforms, labels, colours, shapes, annotations, broken-axis choices, fonts, and output dimensions.", run="click representative points to inspect their complete rows, merge optional annotations, and verify that styling changes preserve the underlying values and thresholds.", outputs="PDF and PNG exports are re-rendered from the recorded data and style at the requested size while the original regression results remain unchanged."),
    lesson(73, "parameter_sweep", "Parameter Sweep", "Run a bounded regression settings sweep and compare which conclusions survive defensible analysis choices.", series=5, app_key="parameter_sweep", section="Results and quality control", inputs="Pair the per-object score and guide-count CSV files for each plate, choose the response column, and set a separate output folder.", settings="choose which axes vary, pin all other values, select grid or random sampling, cap the trials, record the seed, and accept the memory-bounded worker limit.", run="use Estimate first to inspect the legal search size and cost, then start a small bounded sweep and monitor successful, rejected, and failed trials.", outputs="the resumable results table records every trial's settings, status, discovery count, positive-control rank, duration, output folder, and failure reason."),
]


SERIES = [
    {"number": 1, "title": "Getting started and core analysis"},
    {"number": 2, "title": "Time-resolved analysis and segmentation models"},
    {"number": 3, "title": "Annotation quality control and biological assays"},
    {"number": 4, "title": "Operations, reporting, and data utilities"},
    {"number": 5, "title": "Additional registered modules"},
]


def apply_speech_overrides(catalog: dict) -> None:
    """Keep technical English pronunciations stable in Kokoro."""
    for item in catalog["lessons"]:
        for scene in item["scenes"]:
            # Deep-capture overrides may carry reviewed, sentence-specific
            # phonetics that are deliberately more precise than the generic
            # normalizer.  Generate speech text only when the lesson did not
            # already author it.
            scene.setdefault(
                "speech_text", spoken_form(scene["narration"], "en"))


def apply_release_overrides(catalog: dict) -> None:
    """Apply the one maintained deep-capture lesson after the source table.

    Batch has twelve real queue states rather than the five-state generic
    module template.  Keeping that reviewed override in the normal builder
    makes rebuilding the English catalog idempotent instead of silently
    collapsing the accepted release back to a placeholder walkthrough.
    """
    for filename in ("37_batch_override.json",):
        override = json.loads((CATALOG_DIR / filename).read_text())
        for index, current in enumerate(catalog["lessons"]):
            if current["id"] == override["id"]:
                catalog["lessons"][index] = override
                break
        else:  # pragma: no cover - release source corruption
            raise KeyError(f"release override {override['id']!r} is unknown")


def apply_host_routes(catalog: dict) -> None:
    """Attach the current module route to every consolidated capability."""
    lessons = {
        item.get("app_key"): item
        for item in catalog["lessons"] if item.get("app_key")
    }
    missing = sorted(set(FOLDED_LESSON_HOSTS).difference(lessons))
    if missing:
        raise KeyError(f"folded tutorial routes name unknown lessons: {missing}")
    for app_key, host_app_key in FOLDED_LESSON_HOSTS.items():
        lessons[app_key]["host_app_key"] = host_app_key


def main() -> int:
    numbers = [item["number"] for item in LESSONS]
    ids = [item["id"] for item in LESSONS]
    if numbers != list(range(1, 74)):
        raise RuntimeError(f"lesson numbering is not contiguous: {numbers}")
    if len(ids) != len(set(ids)):
        raise RuntimeError("lesson ids must be unique")

    catalog = {
        "schema": 1,
        "title": "spaCR complete tutorial library",
        "series": SERIES,
        "lessons": LESSONS,
    }
    apply_release_overrides(catalog)
    apply_host_routes(catalog)
    apply_speech_overrides(catalog)
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    target = CATALOG_DIR / "lessons_en.json"
    target.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    print(target)
    print(f"lessons={len(LESSONS)} scenes={sum(len(x['scenes']) for x in LESSONS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
