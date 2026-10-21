# spaCR notebooks

Start with [the workflow browser](00_workflow_browser.ipynb) to choose any registered headless workflow, inspect its current defaults, or reuse settings saved by the desktop application.

## Guided workflows

| Notebook | Purpose |
|---|---|
| [Generate masks](01_generate_masks.ipynb) | Generate per-object masks for cells, nuclei, pathogens, and organelles from microscopy images. |
| [Generate and track timelapse masks](01b_generate_timelapse_masks.ipynb) | Generate masks from time-series microscopy images and link selected objects across consecutive frames. |
| [Measure objects and crop images](02_measure_and_crop.ipynb) | Compute per-object morphology, intensity, and spatial measurements from merged image-mask arrays, with optional object-centred crops. |
| [Classify images with a neural network](03_classify_computer_vision.ipynb) | Train a convolutional image classifier from labelled object crops and apply it to the experimental dataset. |
| [Classify objects from measured features](04_classify_machine_learning.ipynb) | Fit a supervised tabular classifier to per-object measurements and apply it to the full dataset. |
| [Map sequencing reads to microplate wells](05_map_barcodes.ipynb) | Process amplicon sequencing reads and quantify guide-RNA assignments for each microplate well. |
| [Estimate genotype-phenotype associations](06_regression.ipynb) | Estimate guide- or gene-level effects by modelling phenotype scores as a function of guide abundance. |
| [Embed object images](07_image_umap.ipynb) | Compute a two-dimensional embedding of single-object images or features and display representative objects in embedding space. |
| [Train a Cellpose model](08_train_cellpose.ipynb) | Fine-tune a Cellpose segmentation model using annotated microscopy images and corresponding reference masks. |
| [Apply a Cellpose model](09_apply_cellpose.ipynb) | Apply a pretrained or custom Cellpose model to a microscopy image collection. |
| [Evaluate a Cellpose model](10_test_cellpose.ipynb) | Compare predicted masks with reference masks using per-image segmentation metrics. |
| [Generate activation maps](11_activation_maps.ipynb) | Estimate the image regions contributing to a trained classifier's predictions using saliency or activation-mapping methods. |
| [Quantify molecular recruitment](12_recruitment.ipynb) | Quantify fluorescent-marker enrichment at a pathogen or vacuole relative to the surrounding cellular compartment. |
| [Analyze a plaque assay](13_plaque_assay.ipynb) | Segment plaques and quantify plaque number and area for each image or well. |
| [Analyze object motility](14_motility_assay.ipynb) | Generate and track time-series masks, then quantify trajectory, velocity, displacement, and infection-related quality control. |
| [Quantify intracellular replication](15_replication.ipynb) | Estimate intracellular parasite number per vacuole and compare count distributions across experimental groups. |
| [Quantify invasion](16_invasion.ipynb) | Classify pathogens as intracellular or extracellular using differential-staining measurements and summarize invasion efficiency. |
| [Quantify endodyogeny](17_endodyogeny.ipynb) | Classify intracellular replication states from pathogen size and summarize their frequencies across experimental groups. |
| [Estimate percent positive](18_percent_positive.ipynb) | Classify objects relative to an intensity threshold and estimate the positive fraction for each experimental unit. |
| [Count annotated phenotypes](19_count_phenotypes.ipynb) | Count objects assigned to each annotation or predicted phenotype class within experimental groups. |
| [Analyze activation maps](20_activation_analysis.ipynb) | Aggregate image-attribution maps and compare attribution summaries across model classes. |
| [Transfer knowledge between models](21_model_knowledge_transfer.ipynb) | Train a student image classifier using predictions from one or more teacher models. |
| [Fuse compatible models](22_model_fusion.ipynb) | Combine parameters from compatible trained image-classifier checkpoints using a specified aggregation method. |
| [Compare sequencing reads with phenotype scores](23_compare_reads_to_scores.ipynb) | Join per-well guide abundance with imaging-derived phenotype scores and quantify their association. |
| [Interpret a vision-model score](24_interpret_vision_model.ipynb) | Model vision-classifier outputs using measured morphology and intensity features to identify associated quantitative attributes. |
| [Generate score heatmaps](25_score_heatmap.ipynb) | Arrange per-well phenotype scores in microplate coordinates and compare score tables when multiple inputs are supplied. |
| [Evaluate sequencing quality](26_sequencing_stats.ipynb) | Summarize sequencing depth, read quality, consensus, and guide-count distributions. |
| [Analyze regression results](27_post_regression_analysis.ipynb) | Derive downstream guide correlations, ranked findings, and figures from a completed regression analysis. |
| [Analyze gene-ontology enrichment](28_go_term_enrichment.ipynb) | Test whether a selected gene set is enriched for annotated gene ontology categories. |
| [Plot gene phenotypes](29_gene_phenotype_plots.ipynb) | Display selected gene-level phenotype estimates relative to the screen-wide distribution. |
| [Generate a volcano plot](30_volcano_plot.ipynb) | Display effect estimates against negative log10-transformed p-values and annotate selected findings. |

## All registered workflows

The browser offers every entry below, including workflows without a dedicated guided notebook. Its dropdown and settings come from the installed version at execution time.

| Workflow | Purpose | Public API |
|---|---|---|
| `activation` | Generate class-activation maps for a trained classifier. | [`spacr.deep_spacr.generate_activation_map`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.generate_activation_map) |
| `align` | Register and stitch an arbitrary number of tiles into one canvas. | [`spacr.align.align_folder`](https://einarolafsson.github.io/spacr/api/spacr/align/index.html#spacr.align.align_folder) |
| `analyze_plaques` | Segment and quantify plaques in a plaque assay. | [`spacr.submodules.analyze_plaques`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques) |
| `anndata_export` | Export the measurement tables as AnnData (.h5ad) for scanpy and scvi-tools. | [`spacr.anndata_export.run_anndata_export`](https://einarolafsson.github.io/spacr/api/spacr/anndata_export/index.html#spacr.anndata_export.run_anndata_export) |
| `barcode_qc` | QC a barcode-mapping run and derive its abundance threshold from a gRNAs-per-well target. | [`spacr.sequencing_qc.barcode_qc`](https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html#spacr.sequencing_qc.barcode_qc) |
| `cellpose_masks` | Run one Cellpose model over a folder and save the masks. | [`spacr.spacr_cellpose.identify_masks_finetune`](https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html#spacr.spacr_cellpose.identify_masks_finetune) |
| `classify` | Full DL pipeline: build dataset, train, apply the model, merge predictions. | [`spacr.deep_spacr.deep_spacr`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.deep_spacr) |
| `classify_merged` | Train a classifier on single objects: Torch on crops, or gradient boosting on measured features. | [`spacr.classify.classify`](https://einarolafsson.github.io/spacr/api/spacr/classify/index.html#spacr.classify.classify) |
| `convert` | Convert vendor images into mapped, collision-safe Yokogawa TIFFs. | [`spacr.convert.convert_folder`](https://einarolafsson.github.io/spacr/api/spacr/convert/index.html#spacr.convert.convert_folder) |
| `endodyogeny` | Legacy size proxy: bin pathogen area-derived volume by doublings. | [`spacr.submodules.analyze_endodyogeny`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_endodyogeny) |
| `explain_cv` | Explain existing CV predictions with measured object features. | [`spacr.surrogate.run_explain_cv`](https://einarolafsson.github.io/spacr/api/spacr/surrogate/index.html#spacr.surrogate.run_explain_cv) |
| `external_masks` | Measure images using label masks generated outside spaCR. | [`spacr.external_masks.prepare_external_masks`](https://einarolafsson.github.io/spacr/api/spacr/external_masks/index.html#spacr.external_masks.prepare_external_masks) |
| `foreign` | Import external images, masks, and a measurement table as a spaCR project. | [`spacr.foreign.import_project`](https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html#spacr.foreign.import_project) |
| `host_pathogen` | Combine vacuole recruitment, explicit parasite counts and host infection denominators. | [`spacr.host_pathogen.analyze_host_pathogen`](https://einarolafsson.github.io/spacr/api/spacr/host_pathogen/index.html#spacr.host_pathogen.analyze_host_pathogen) |
| `illumination` | Estimate the plate's illumination field and install the correction every measure worker applies. | [`spacr.illumination.prepare_illumination_correction`](https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html#spacr.illumination.prepare_illumination_correction) |
| `invasion` | Red/green invasion assay: score every parasite attached or invaded, per well. | [`spacr.submodules.analyze_invasion`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_invasion) |
| `investigate_hit` | Resolve one exact regression hit to cross-fitted candidate cells. | [`spacr.hit_investigation.investigate_hit`](https://einarolafsson.github.io/spacr/api/spacr/hit_investigation/index.html#spacr.hit_investigation.investigate_hit) |
| `map_barcodes` | Map row / column / gRNA barcodes out of sequencing reads onto wells. | [`spacr.sequencing.generate_barecode_mapping`](https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html#spacr.sequencing.generate_barecode_mapping) |
| `mask` | Segment cells / nuclei / pathogens with Cellpose and write merged stacks. | [`spacr.core.preprocess_generate_masks`](https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks) |
| `measure` | Measure per-object morphology / intensity and crop single-object PNGs. | [`spacr.measure.measure_crop`](https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop) |
| `ml_analyze` | Classical ML (XGBoost / RF / logistic) on per-object screen features. | [`spacr.ml.generate_ml_scores`](https://einarolafsson.github.io/spacr/api/spacr/ml/index.html#spacr.ml.generate_ml_scores) |
| `motility` | Automated motility assay: per-track velocity plus infection QC. | [`spacr.timelapse.automated_motility_assay`](https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html#spacr.timelapse.automated_motility_assay) |
| `ops` | Stitch, segment and decode the wells of an optical pooled screen's sequencing acquisition. | [`spacr.ops_engine.run_ops`](https://einarolafsson.github.io/spacr/api/spacr/ops_engine/index.html#spacr.ops_engine.run_ops) |
| `recruitment` | Analyze recruitment of a channel of interest to pathogen compartments. | [`spacr.submodules.analyze_recruitment`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment) |
| `regression` | Regress per-well scores against sgRNA counts to call screen hits. | [`spacr.ml.perform_regression`](https://einarolafsson.github.io/spacr/api/spacr/ml/index.html#spacr.ml.perform_regression) |
| `replication` | Count parasites per vacuole and compare replication distributions. | [`spacr.submodules.analyze_replication`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_replication) |
| `simulation` | Sweep the pooled-screen simulator across a grid of parameters. | [`spacr.sim.run_multiple_simulations`](https://einarolafsson.github.io/spacr/api/spacr/sim/index.html#spacr.sim.run_multiple_simulations) |
| `spatial_map` | Count every Xenium transcript or Visium HD bin into the cells Spatial Mask Generation segmented. | [`spacr.ops_engine.run_spatial_map`](https://einarolafsson.github.io/spacr/api/spacr/ops_engine/index.html#spacr.ops_engine.run_spatial_map) |
| `spatial_masks` | Segment a whole Xenium morphology or Visium HD H&E slide in overlapping tiles into Measure-ready fields. | [`spacr.ops_engine.run_spatial_masks`](https://einarolafsson.github.io/spacr/api/spacr/ops_engine/index.html#spacr.ops_engine.run_spatial_masks) |
| `timelapse` | Mask pipeline with object tracking across the frames of a time series. | [`spacr.core.preprocess_generate_masks_timelapse`](https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks_timelapse) |
| `train_cellpose` | Train or fine-tune a Cellpose model on your own labelled images. | [`spacr.submodules.train_cellpose`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.train_cellpose) |
| `train_only` | The training stage alone: train / evaluate on an existing dataset folder. | [`spacr.deep_spacr.train_test_model`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.train_test_model) |
| `umap` | Embed single-object images with UMAP and plot them as image glyphs. | [`spacr.core.generate_image_umap`](https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap) |

## Edit, inspect and run

1. Select the Python environment in which spaCR is installed; run the environment check first.
2. Edit essential inputs. Expand a parameter reference group for descriptions and required/conditional status.
3. Advanced and organelle settings are ordinary Python cells with Jupyter's `source_hidden` metadata. JupyterLab and Notebook 7 show an ellipsis that expands them. Other viewers may show their source instead. Every setting remains editable.
4. Run all configuration cells in order, including collapsed cells, then run the analysis cell. Restart the kernel and run all cells when changing workflows or restoring defaults.

HTML parameter groups are clickable in compatible notebook viewers. Plain text and exported formats can still read their contents. No notebook extension is required.

The browser optionally uses ipywidgets for a workflow picker and an advanced-settings editor. Without widgets, edit the Python variables shown in the same notebook. Analysis is opt-in, so running its setup cells does not launch a pipeline.

## Desktop tools and additional interfaces

Interactive mask drawing/lineage curation, annotation, cluster profiles and other desktop tools use the [graphical workflows](https://einarolafsson.github.io/spacr/workflows.html). The [complete API](https://einarolafsson.github.io/spacr/api/) covers additional public functions; `spacr-run --help` lists management commands such as archive export and validation.

- [Python guide](https://einarolafsson.github.io/spacr/python_api.html)
- [Tutorials](https://einarolafsson.github.io/spacr/tutorials/)
- [Jupyter cell metadata](https://nbformat.readthedocs.io/en/latest/format_description.html#cell-metadata)
- [Widget containers](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#Container/Layout-widgets)

Notebooks under `legacy/` retain their original URLs but are also regenerated with current settings and APIs. Combined ML/CV examples retain their regression stage; the Cellpose example retains optional image/mask dataset preparation.
