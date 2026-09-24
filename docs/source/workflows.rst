Choose a workflow after installation
====================================

Start on Home with the first module for your experiment. Use that module's example-data control when available, inspect the inputs and preview, then run a bounded example before your own data.

These routes and the API handoffs share the bundled module workflow map. A walkthrough explains the steps; it does not run an experiment automatically.

Open **Pipeline overviews** to choose a pathway. Its graph includes external input nodes that explain the files you supply; these are not runnable modules. Persistent explanation cards describe modules and connections below the graph, with API links at the end of module cards. Select a graph element to outline its card, and drag the blue divider to adjust the space between graph and explanations. **Start example** opens the pathway's first real module; **Walkthrough** keeps the route available afterward. Alternative input and annotation branches do not require you to execute every listed step.

.. _workflow-pooled_screen:

A pooled spaCR screen
---------------------

Combine image-derived phenotypes with guide abundance to rank perturbations. Segment and measure images, create reviewed labels with Annotate, Gate Editor or Image UMAP, and train and validate a classifier. Map the matching sequencing reads on an independent branch. Join compatible phenotype scores and guide counts by experimental identity for Regression. Annotation and segmentation routes are alternatives, not requirements to run every branch.

**Choose an input route:**

* **Microscopy images**: Supply the original microscopy images with channel identities and field naming. For a time series, preserve frame order and timing. Choose segmentation settings appropriate for the objects in this experiment. Continue with :ref:`Mask <workflow-module-mask>`.
* **Existing masks and matching images**: Import source images together with matching integer object-label masks. Preserve object classes and field identities. This is an alternative to generating new masks; inspect the imported project before measuring. Continue with :ref:`Import <workflow-module-foreign>`.
* **FASTQ: R1/R2, R1 only or R2 only**: Supply gzipped FASTQ files and the barcode reference CSVs. Use paired mode for R1/R2. For single-end reads, choose single mode and set single_direction to R1 or R2 to match the files and extraction orientation. Continue with :ref:`Map Barcodes <workflow-module-map_barcodes>`.

**Module steps and alternatives:**

#. :ref:`Mask <workflow-module-mask>`: Start on Home and open Mask. Load the example data, check channels and segmentation, then run.
#. :ref:`Map Barcodes <workflow-module-map_barcodes>`: Return Home and open Map Barcodes for the matching sequencing reads and references.
#. :ref:`Import <workflow-module-foreign>`: If segmentation already exists, import the matching images and external masks into a spaCR project and inspect the resulting object identities. Skip segmentation and proceed to Measure when its required arrays are present.
#. :ref:`Measure <workflow-module-measure>`: Open Measure with the Mask project and create measurements and object crops.
#. :ref:`Annotate <workflow-module-annotate>`: Open Annotate, inspect representative objects and save phenotype labels.
#. :ref:`Gate Editor <workflow-module-gate_editor>`: Define threshold or polygon gates on actual feature/coordinate columns, then apply the saved gate to compatible objects. Use Annotate to write the displayed gates to an annotation column; review the selected objects and choose binary or multiclass labels.
#. :ref:`Image UMAP <workflow-module-umap>`: Project measured features or supplied encoder features and inspect representative crops. A cluster is a candidate grouping, not a validated phenotype. Use the lasso and annotation controls to write reviewed selections to an annotation column in the matching measurement database. A geometric selection alone does not establish a biological phenotype.
#. :ref:`Classify <workflow-module-classify_merged>`: Open Classify, choose images or measured features, and inspect held-out predictions.
#. :ref:`Regression <workflow-module-regression>`: Open Regression with compatible phenotype scores and per-well guide counts; inspect hits and diagnostics.

.. _workflow-high_content:

High-content image analysis
---------------------------

Quantify an imaging experiment without sequencing. Start with microscopy images or import existing masks with their source images, measure the objects, then explore embeddings, define gated populations or build figures. These downstream branches answer different questions and can be used independently.

**Choose an input route:**

* **Microscopy images**: Supply the original microscopy images with channel identities and field naming. For a time series, preserve frame order and timing. Choose segmentation settings appropriate for the objects in this experiment. Continue with :ref:`Mask <workflow-module-mask>`.
* **Existing masks and matching images**: Import source images together with matching integer object-label masks. Preserve object classes and field identities. This is an alternative to generating new masks; inspect the imported project before measuring. Continue with :ref:`Import <workflow-module-foreign>`.

**Module steps and alternatives:**

#. :ref:`Mask <workflow-module-mask>`: Start on Home with Mask and inspect segmentation on representative fields.
#. :ref:`Import <workflow-module-foreign>`: If segmentation already exists, import the matching images and external masks into a spaCR project and inspect the resulting object identities. Skip segmentation and proceed to Measure when its required arrays are present.
#. :ref:`Measure <workflow-module-measure>`: Open Measure on the resulting project and quantify objects.
#. :ref:`Image UMAP <workflow-module-umap>`: Open Image UMAP to explore measured phenotypes and inspect crops.
#. :ref:`Gate Editor <workflow-module-gate_editor>`: Open Gate Editor to define an explicit population filter.
#. :ref:`Graph Builder <workflow-module-graph_builder>`: Open Graph Builder to compare the chosen measurements and groups.

Other routes for the appropriate question: :ref:`Classify <workflow-module-classify_merged>`, :ref:`Recruitment <workflow-module-recruitment>`, :ref:`Invasion Assay <workflow-module-invasion>`, :ref:`Replication Assay <workflow-module-replication>`.

.. _workflow-segmentation_training:

Train a segmentation model
--------------------------

Build a segmentation model from curated image and integer-mask pairs. Correct representative images in Make Masks, train in Cellpose Workbench, and select the resulting checkpoint in Mask. Evaluate independent fields before processing the full experiment; training images are not a held-out accuracy test.

**Choose an input route:**

* **Microscopy images and optional starting masks**: Open representative images and optionally import existing integer masks for correction. Save matching curated pairs; keep separate validation images outside the training set. Continue with :ref:`Make Masks <workflow-module-make_masks>`.
* **Independent microscopy images**: Apply the trained checkpoint to independent images with compatible channels. Check segmentation quality before processing the full experiment. Continue with :ref:`Mask <workflow-module-mask>`.

**Module steps and alternatives:**

#. :ref:`Make Masks <workflow-module-make_masks>`: Start on Home with Make Masks and curate matching images and integer label masks.
#. :ref:`Cellpose Workbench <workflow-module-train_cellpose>`: Inside Make Masks open Cellpose Workbench, choose Train and use the curated pairs.
#. :ref:`Mask <workflow-module-mask>`: Return Home, open Mask and select the trained checkpoint; verify separate fields before a full run.

.. _workflow-parasite_assay:

Toxoplasma and parasite imaging assays
--------------------------------------

Choose the assay that matches the experiment. Recruitment, invasion and replication use measured objects with their required compartment identities and staining channels. Recruitment compares compartment intensities; invasion distinguishes internal and external parasites using differential staining; replication counts parasites within the appropriate vacuoles. Plaque Assay is an independent branch using plaque images or masks, with calibration for physical areas. The diagram groups these assays without implying that one assay must run before another.

**Choose an input route:**

* **Microscopy images**: Supply the original microscopy images with channel identities and field naming. For a time series, preserve frame order and timing. Choose segmentation settings appropriate for the objects in this experiment. Continue with :ref:`Mask <workflow-module-mask>`.
* **Existing masks and matching images**: Import source images together with matching integer object-label masks. Preserve object classes and field identities. This is an alternative to generating new masks; inspect the imported project before measuring. Continue with :ref:`Import <workflow-module-foreign>`.
* **Plaque images or existing plaque masks**: Choose plaque images for detection or existing plaque masks for analysis. Supply image calibration when physical plaque areas are required. This branch does not require the cell-compartment Measure workflow. Continue with :ref:`Plaque Assay <workflow-module-analyze_plaques>`.

**Module steps and alternatives:**

#. :ref:`Mask <workflow-module-mask>`: Start on Home with Mask and configure the host, parasite and required compartment masks.
#. :ref:`Import <workflow-module-foreign>`: If segmentation already exists, import the matching images and external masks into a spaCR project and inspect the resulting object identities. Skip segmentation and proceed to Measure when its required arrays are present.
#. :ref:`Measure <workflow-module-measure>`: Open Measure and collect the compartment or staining measurements needed by your assay.
#. :ref:`Recruitment <workflow-module-recruitment>`: For a recruitment question, open Recruitment and inspect compartment intensity ratios.
#. :ref:`Host–Pathogen Analysis <workflow-module-host_pathogen>`: Measure whole vacuoles and host reference compartments, keeping uninfected host cells for the infection denominator. Select marker channels and control-calibrated ratio thresholds. Optionally count individually segmented parasites from explicit vacuole links or a measured count column. Review vacuole, host and well tables, joint marker states, replication distributions and unmatched parasite links; unknown measurements remain unknown.
#. :ref:`Invasion Assay <workflow-module-invasion>`: Use the required two-colour differential-staining measurements and stain-baseline controls to distinguish attachment from invasion.
#. :ref:`Replication Assay <workflow-module-replication>`: Count parasites using explicit vacuole identity and compare condition distributions; host identity alone does not define a vacuole.
#. :ref:`Plaque Assay <workflow-module-analyze_plaques>`: For a plaque experiment, open Plaque Assay from Toxoplasma and supply plaque images or existing plaque masks. Inspect detection and calibration before comparing plaque counts and areas.

Invasion and Replication read their required measured compartments; Plaque Assay instead takes plaque images or masks. Open these assays from Home → Toxoplasma.

.. _workflow-optical_screen:

An optical pooled screen
------------------------

Decode an optical pooled screen from the original sequencing cycles and corresponding phenotype images. Enter OPS through Mask, inspect cycle/site alignment and barcode decoding, then join decoded object identities with phenotype responses and aggregate compatible inputs for Regression. Align & Stitch is optional; a stitched image alone cannot replace sequencing cycles.

**Choose an input route:**

* **Sequencing cycles and phenotype images**: Supply original cycle/site images, channel assignments and the corresponding phenotype alignment. Preserve cycle and site identities; mosaic output alone is insufficient for decoding. Continue with :ref:`OPS <workflow-module-ops>`.
* **Compatible phenotype responses**: Join decoded object identities to phenotype responses and aggregate them to the analysis unit required by Regression. Verify object/well identity and guide assignment; OPS tables are not a direct FASTQ count CSV handoff. Continue with :ref:`Regression <workflow-module-regression>`.

**Module steps and alternatives:**

#. :ref:`OPS <workflow-module-ops>`: Start on Home: open Mask, then OPS; supply original cycle/site images and phenotype alignment.
#. :ref:`Regression <workflow-module-regression>`: Join and aggregate decoded object identities with compatible phenotype responses before opening Regression.

Align & Stitch is optional when your images need tile-geometry inspection or coordinate mapping; it is not a prerequisite for the OPS example. A stitched image alone cannot replace the sequencing cycles. OPS tables require an explicit identity join/aggregation before regression; the walkthrough does not perform that conversion automatically.

.. _workflow-import_images:

Import existing data
--------------------

Bring an existing experiment into spaCR using the route matching its contents. Image-only data need segmentation. Matching images and integer masks can be formatted into a project for measurement. Existing feature tables require explicit column and object mappings and may already be ready for exploration. Check identities, channels and crop paths before following a downstream branch.

**Choose an input route:**

* **Microscopy images**: Supply the original microscopy images with channel identities and field naming. For a time series, preserve frame order and timing. Choose segmentation settings appropriate for the objects in this experiment. Continue with :ref:`Import <workflow-module-foreign>`.
* **Existing masks and matching images**: Import source images together with matching integer object-label masks. Preserve object classes and field identities. This is an alternative to generating new masks; inspect the imported project before measuring. Continue with :ref:`Import <workflow-module-foreign>`.
* **External measurements and object mappings**: Import existing feature tables with explicit object, field and column mappings. Provide matching crop paths when image exploration is needed. Avoid remeasuring rows already represented by imported features. Continue with :ref:`Import <workflow-module-foreign>`.

**Module steps and alternatives:**

#. :ref:`Import <workflow-module-foreign>`: Start on Home with Import and select the route matching images, external masks or external measurements.
#. :ref:`Mask <workflow-module-mask>`: For image-only imports, open Mask to add segmentation; skip this when valid masks already exist.
#. :ref:`Measure <workflow-module-measure>`: Use Measure when features have not already been imported or computed by External Masks.
#. :ref:`Image UMAP <workflow-module-umap>`: Open Image UMAP to inspect the resulting measured objects and matching crops.

.. _workflow-phenotype_discovery:

Discover phenotypes without labels
----------------------------------

Explore a measured, segmented experiment without initial phenotype labels. Compare measured features or encoded image features in Image UMAP, inspect representative crops and create reproducible selections in Gate Editor. Review candidate labels in Annotate before supervised training. A cluster or gate is a hypothesis about a phenotype, not its validation.

**Choose an input route:**

* **Segmented image project**: Use merged image arrays and matching object-label planes from Mask or Import. Check channel roles, object identities and crop settings before measuring. Continue with :ref:`Measure <workflow-module-measure>`.

**Module steps and alternatives:**

#. :ref:`Measure <workflow-module-measure>`: Start on Home with Measure using an already segmented project, or use its example data.
#. :ref:`Embeddings <workflow-module-embeddings>`: Open Embeddings to encode object images; retain the object IDs and encoder/channel settings.
#. :ref:`Image UMAP <workflow-module-umap>`: Open Image UMAP with compatible measured or encoder feature columns and inspect candidate groups.
#. :ref:`Gate Editor <workflow-module-gate_editor>`: Define a reproducible selection in Gate Editor using the matching feature or projection columns.
#. :ref:`Annotate <workflow-module-annotate>`: Open Annotate and review candidate labels before using them as supervised training data.

.. _workflow-screen_planning:

Plan a screen
-------------

Design conditions, controls, replicates and plate assignments before acquisition. Use Power / Design to explore sampling needs under explicit effect-size and variability assumptions; pilot measurements can inform these assumptions. The design handoff is a planning dependency, not an automatically executed experiment.

**Choose an input route:**

* **Conditions, controls and design assumptions**: Enter experimental conditions, controls, replicate requirements and plate constraints. These inputs may be entered directly; a pre-existing data file is not required. Continue with :ref:`Experiment Design <workflow-module-experiment_design>`.
* **Effect sizes, variability and optional pilot data**: Supply explicit assumptions for effect size and variability; optionally use compatible pilot measurements. Review how changing assumptions affects the required sampling effort. Continue with :ref:`Power / Design <workflow-module-power>`.

**Module steps and alternatives:**

#. :ref:`Experiment Design <workflow-module-experiment_design>`: Start on Home with Experiment Design; assign conditions, controls, replicates and wells.
#. :ref:`Power / Design <workflow-module-power>`: Open Power / Design and estimate sampling needs from explicit effect-size and variability assumptions.

.. _workflow-live_cell_tracking:

Track live cells
----------------

Follow segmented objects through an ordered image sequence, then quantify their measurements and motion. Check frame order, frame interval, object identity and pixel calibration. Import existing masks with their source images when available, and ensure that the tracked project has valid identities before interpreting motility.

**Choose an input route:**

* **Microscopy images**: Supply the original microscopy images with channel identities and field naming. For a time series, preserve frame order and timing. Choose segmentation settings appropriate for the objects in this experiment. Continue with :ref:`Mask <workflow-module-mask>`.
* **Existing masks and matching images**: Import source images together with matching integer object-label masks. Preserve object classes and field identities. This is an alternative to generating new masks; inspect the imported project before measuring. Continue with :ref:`Import <workflow-module-foreign>`.

**Module steps and alternatives:**

#. :ref:`Mask <workflow-module-mask>`: Start on Home with Mask and point it at an ordered time-series project.
#. :ref:`Timelapse <workflow-module-timelapse>`: Open the Timelapse action in Mask, configure timing and inspect linked objects.
#. :ref:`Import <workflow-module-foreign>`: If segmentation already exists, import the matching images and external masks into a spaCR project and inspect the resulting object identities. Skip segmentation and proceed to Measure when its required arrays are present.
#. :ref:`Measure <workflow-module-measure>`: Open Measure on the tracked project to quantify the objects.
#. :ref:`Motility Assay <workflow-module-motility>`: Inside Measure open Motility Assay and check frame interval, pixel calibration and track filters.

Module inputs, outputs and next steps
-------------------------------------

.. _workflow-module-mask:

Mask
~~~~

Select channels and models, inspect a preview, then run Mask. Measure consumes the merged arrays; the counts database is not yet a feature table.

**Open:** Home → Mask.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Segmentation checkpoint** — Saved Cellpose-compatible checkpoint or a compatible installed backend selected with its own configuration.

**Outputs**

* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.
* **Object counts** — measurements/measurements.db; Mask counts alone are not per-object feature measurements.

**Before this module**

* :ref:`Cellpose Workbench <workflow-module-train_cellpose>`: Select the saved compatible checkpoint in Mask.
* :ref:`Import Images <workflow-module-import_images>`: Use imported image planes and identities; image-only imports still need segmentation.
* :ref:`Format Converter <workflow-module-convert>`: Use the converted layout and preserve source identity mappings.
* :ref:`Import <workflow-module-foreign>`: For image-only imports, use Import Images or Format Converter and point Mask at the formatted image project. External measurements alone are not segmentation input.

**After this module**

* :ref:`Measure <workflow-module-measure>`: Use the same project and the correct image/mask channel indices.
* :ref:`Timelapse <workflow-module-timelapse>`: Enable the nested time-series route before generating linked labels.

:doc:`API reference </api/spacr/core/index>`.

`Module tutorial <tutorials/#lesson=07_mask>`__.

.. _workflow-module-measure:

Measure
~~~~~~~

Measure reads images and label planes together. Enable crop saving if you need PNG files; keep the database and source project together for streamed crops.

**Open:** Home → Measure.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**Outputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Before this module**

* :ref:`Mask <workflow-module-mask>`: Use the same project and the correct image/mask channel indices.
* :ref:`Make Masks <workflow-module-make_masks>`: Use FEATURES to pair images and masks and write a measured project; standalone masks are not merged arrays.
* :ref:`External Masks <workflow-module-external_masks>`: Re-measure only when needed; External Masks can already perform measurement.
* :ref:`Timelapse <workflow-module-timelapse>`: Use the time-series project with stable frame/object identities.
* :ref:`Import <workflow-module-foreign>`: Import matching images and external integer masks to build merged project arrays, then open Measure on that project. Skip this step when compatible measurements have already been imported or computed. Do not append duplicate measurements to an existing imported table.

**After this module**

* :ref:`Annotate <workflow-module-annotate>`: Save or stream crops with stable object identities.
* :ref:`Classify <workflow-module-classify_merged>`: Choose the image or tabular family to match your input.
* :ref:`Image UMAP <workflow-module-umap>`: Choose feature columns and inspect representative crops.
* :ref:`Embeddings <workflow-module-embeddings>`: Retain encoder and channel-policy provenance.
* :ref:`Gate Editor <workflow-module-gate_editor>`: Use the actual measured feature definitions and units.
* :ref:`Graph Builder <workflow-module-graph_builder>`: Choose explicit variables, groups and filters.
* :ref:`QC <workflow-module-qc_dashboard>`: Review available checks and missing evidence.
* :ref:`Recruitment <workflow-module-recruitment>`: Require the intended compartment intensities and identities.
* :ref:`Invasion Assay <workflow-module-invasion>`: Require two-colour stain measurements and appropriate baseline controls.
* :ref:`Replication Assay <workflow-module-replication>`: Require explicit parasite-to-vacuole identities.
* :ref:`Motility Assay <workflow-module-motility>`: Combine measured objects with matching tracks.
* :ref:`Feature Explorer <workflow-module-feature_explorer>`: Define the class comparison and inspect filtering.
* :ref:`Database Browser <workflow-module-db_browser>`: Inspect actual tables before exporting.
* :ref:`AnnData Export <workflow-module-anndata_export>`: Export compatible feature and metadata columns.
* :ref:`Dose–Response <workflow-module-dose_response>`: Join the measured response to explicit doses and controls.
* :ref:`Endodyogeny size proxy <workflow-module-endodyogeny>`: Supply the measured project roots and required object/png_list tables. Verify host-cell aggregation and area units before interpreting size bins; the Mask counts database alone is insufficient.
* :ref:`Host–Pathogen Analysis <workflow-module-host_pathogen>`: Keep uninfected cells in Measure. Supply whole-vacuole masks, host reference intensities and optional explicit parasite-to-vacuole links; host identity alone does not define a vacuole.

:doc:`API reference </api/spacr/measure/index>`.

`Module tutorial <tutorials/#lesson=08_measure>`__.

.. _workflow-module-annotate:

Annotate
~~~~~~~~

Label representative objects in png_list before supervised training. Use separate annotations for different questions and keep training and evaluation groups independent.

**Open:** Home → Annotate.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.
* **Reusable gates** — Saved threshold/polygon gate definitions or a selected object set; apply a gate to the same feature definitions.

**Outputs**

* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Save or stream crops with stable object identities.
* :ref:`External Masks <workflow-module-external_masks>`: Keep the newly measured project and crop index together.
* :ref:`Import <workflow-module-foreign>`: Imported tables require explicit column and object-identity mappings.
* :ref:`Gate Editor <workflow-module-gate_editor>`: Apply compatible gates, then review candidate labels.

**After this module**

* :ref:`Classify <workflow-module-classify_merged>`: Keep labelled training and evaluation groups separate.
* :ref:`Annotator Agreement <workflow-module-agreement>`: Choose independent annotation columns for the same objects.

:doc:`API reference </api/spacr/qt/screens/annotate/index>`.

`Module tutorial <tutorials/#lesson=09_annotate>`__.

.. _workflow-module-classify_merged:

Classify
~~~~~~~~

Choose Computer Vision for crops or streamed object images, or Tabular Machine Learning for measured feature columns. The two families consume different inputs.

**Open:** Home → Classify.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.
* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.

**Outputs**

* **Object classification scores** — Saved score CSVs and, when merged, measurements/measurements.db, table png_list.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``pred``, ``cv_predictions``, ``ml_pred``, ``predictions``.
* **Trained image classifier** — Saved classifier checkpoint, model settings and matching channel/preprocessing configuration.
* **Fitted feature classifier** — The fitted tabular classifier and its recorded feature list, training settings and validation results.
* **Classifier evaluation bundle** — Held-out predictions, labels, split metadata and calibration/leakage metrics for a saved classifier run.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Choose the image or tabular family to match your input.
* :ref:`Annotate <workflow-module-annotate>`: Keep labelled training and evaluation groups separate.
* :ref:`Gate Editor <workflow-module-gate_editor>`: Write reviewed gate selections to an annotation column, then select that same column and object population in Classify. Keep validation objects separate from training labels.
* :ref:`Image UMAP <workflow-module-umap>`: Write reviewed lasso selections to an annotation column in the matching object database, then select that column in Classify. Inspect crops and validate labels; embedding clusters are not ground truth.

**After this module**

* :ref:`Regression <workflow-module-regression>`: Select the intended CV or ML score column and preserve plate/well identity.
* :ref:`Classifier Evaluation <workflow-module-classifier_evaluation>`: Use the matching held-out predictions and split metadata.
* :ref:`Activation <workflow-module-activation>`: Also provide the matching crops and channel preprocessing.
* :ref:`Explain CV Model <workflow-module-explain_cv>`: Select CV predictions and the matching measured objects.

:doc:`API reference </api/spacr/classify/index>`.

`Module tutorial <tutorials/#lesson=41_classify>`__.

.. _workflow-module-map_barcodes:

Map Barcodes
~~~~~~~~~~~~

Supply the references that encode your experiment. Inspect mapped counts and barcode QC before pairing well-level guide counts with phenotype scores.

**Open:** Home → Map Barcodes.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Sequencing reads** — Single-end or paired FASTQ files, with the matching barcode reference tables.

**Outputs**

* **Guide counts per well** — Map Barcodes run folder: unique_combinations.csv and annotated_reads.h5. Well identity requires the corresponding barcode references.
  Relevant columns, depending on the route: ``count``.

**After this module**

* :ref:`Regression <workflow-module-regression>`: Pair guide counts with phenotype scores using consistent plate/well keys.
* :ref:`Barcode QC <workflow-module-barcode_qc>`: Check mapping and coverage before guide filtering.

:doc:`API reference </api/spacr/sequencing/index>`.

`Module tutorial <tutorials/#lesson=12_map_barcodes>`__.

.. _workflow-module-regression:

Regression
~~~~~~~~~~

Match plate and well identifiers across phenotype and guide-count inputs, select the response and controls, and inspect diagnostics before interpreting hits. Direct measured responses are also supported.

**Open:** Home → Regression.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Object classification scores** — Saved score CSVs and, when merged, measurements/measurements.db, table png_list.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``pred``, ``cv_predictions``, ``ml_pred``, ``predictions``.
* **Guide counts per well** — Map Barcodes run folder: unique_combinations.csv and annotated_reads.h5. Well identity requires the corresponding barcode references.
  Relevant columns, depending on the route: ``count``.
* **Optical barcode assignments** — OPS destination measurements.db: per-well geometry, phenotype alignment, nuclei and barcode tables; optional per-cycle reads.
  Relevant tables, depending on the route: ``ops_geometry``, ``ops_phenotype``, ``ops_objects``, ``ops_barcodes``, ``ops_reads``.
* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Regression results and hits** — Selected run results folder: coefficient/result CSVs, hit tables, settings and diagnostic figures.

**Before this module**

* :ref:`Classify <workflow-module-classify_merged>`: Select the intended CV or ML score column and preserve plate/well identity.
* :ref:`Map Barcodes <workflow-module-map_barcodes>`: Pair guide counts with phenotype scores using consistent plate/well keys.
* :ref:`OPS <workflow-module-ops>`: Join/aggregate decoded objects to phenotype and guide inputs explicitly before Regression; this is not a direct CSV handoff.

**After this module**

* :ref:`Run Compare <workflow-module-run_compare>`: Compare compatible saved result sets and their settings.
* :ref:`Hit List <workflow-module-hit_list>`: Inspect ranked hits and guide agreement.
* :ref:`Volcano Explorer <workflow-module-volcano_explorer>`: Inspect saved effects and adjusted significance.
* :ref:`Diagnostics <workflow-module-regression_diagnostics>`: Open the diagnostics actually written by this run.
* :ref:`Methods & Results <workflow-module-methods_export>`: Review exported prose and every traced result.
* :ref:`Investigate Hit <workflow-module-investigate_hit>`: Join compatible phenotype and object data for the chosen hit.

:doc:`API reference </api/spacr/ml/index>`.

`Module tutorial <tutorials/#lesson=13_regression>`__.

.. _workflow-module-feature_explorer:

Feature Explorer
~~~~~~~~~~~~~~~~

Rank measured features for a chosen class comparison; review filtering and class definitions before interpreting the ranking.

**Open:** Classify → Feature Explorer.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Define the class comparison and inspect filtering.

:doc:`API reference </api/spacr/qt/screens/feature_explorer/index>`.

`Module tutorial <tutorials/#lesson=65_feature_explorer>`__.

.. _workflow-module-convert:

Format Converter
~~~~~~~~~~~~~~~~

Convert supported microscope formats to a configured TIFF layout while preserving source mappings. Conversion does not generate object measurements.

**Open:** Import → Format Converter.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

**Outputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

**After this module**

* :ref:`Mask <workflow-module-mask>`: Use the converted layout and preserve source identity mappings.

:doc:`API reference </api/spacr/convert/index>`.

`Module tutorial <tutorials/#lesson=35_converter>`__.

.. _workflow-module-foreign:

Import
~~~~~~

Import external measurements with explicit object/column mappings, or choose Import Images, Format Converter or External Masks. Imported measurements are not a fresh Measure run. With matching source images and external integer masks, Import builds the merged project arrays used by Measure. Choose the route matching your files and inspect object identities before measuring; imported feature tables can be used directly when they already contain the required measurements.

**Open:** Home → Import.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.
* **External measurements** — Third-party measurement CSV/database tables and their image, mask and object-identity mappings.

**Outputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.
* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**After this module**

* :ref:`Annotate <workflow-module-annotate>`: Imported tables require explicit column and object-identity mappings.
* :ref:`Measure <workflow-module-measure>`: Import matching images and external integer masks to build merged project arrays, then open Measure on that project. Skip this step when compatible measurements have already been imported or computed. Do not append duplicate measurements to an existing imported table.
* :ref:`Mask <workflow-module-mask>`: For image-only imports, use Import Images or Format Converter and point Mask at the formatted image project. External measurements alone are not segmentation input.

:doc:`API reference </api/spacr/foreign/index>`.

`Module tutorial <tutorials/#lesson=36_import>`__.

.. _workflow-module-external_masks:

External Masks
~~~~~~~~~~~~~~

Assign corresponding images and existing integer label masks, then create and measure a spaCR project without rerunning segmentation.

**Open:** Import → External Masks.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**Outputs**

* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Before this module**

* :ref:`Direct Cellpose mask generation <workflow-module-cellpose_masks>`: Provide the saved label TIFFs and their original images to External Masks, assign object roles and create the merged project before Measure.

**After this module**

* :ref:`Measure <workflow-module-measure>`: Re-measure only when needed; External Masks can already perform measurement.
* :ref:`Annotate <workflow-module-annotate>`: Keep the newly measured project and crop index together.

:doc:`API reference </api/spacr/external_masks/index>`.

`Module tutorial <tutorials/#lesson=31_external_masks>`__.

.. _workflow-module-queue:

Plate Queue
~~~~~~~~~~~

Run the chosen pipeline across plates. Outputs are those of each queued module; retain each plate and its settings separately.

**Open:** the application's Help/tools menus.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Run queue** — Saved module/plate/settings job definitions and dependency order.

**Outputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.

:doc:`API reference </api/spacr/qt/plate_queue/index>`.

`Module tutorial <tutorials/#lesson=30_plate_queue>`__.

.. _workflow-module-batch:

Batch Runner
~~~~~~~~~~~~

Validate and execute module jobs in dependency order. Each job writes its own module outputs and log.

**Open:** the application's Help/tools menus.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Run queue** — Saved module/plate/settings job definitions and dependency order.

**Outputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.

:doc:`API reference </api/spacr/batch/index>`.

`Module tutorial <tutorials/#lesson=37_batch>`__.

.. _workflow-module-distributed_jobs:

Distributed Jobs
~~~~~~~~~~~~~~~~

Submit a configured run to a remote execution target and inspect its status and logs. Retrieve the actual module artifacts before downstream analysis.

**Open:** the application's Help/tools menus.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Run queue** — Saved module/plate/settings job definitions and dependency order.

**Outputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.

:doc:`API reference </api/spacr/remote_execution/index>`.

`Module tutorial <tutorials/#lesson=38_distributed_jobs>`__.

.. _workflow-module-db_browser:

Database Browser
~~~~~~~~~~~~~~~~

Browse selected tables and export rows; table inspection does not rerun image analysis.

**Open:** the application's Help/tools menus.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Inspect actual tables before exporting.

:doc:`API reference </api/spacr/qt/screens/db_browser/index>`.

`Module tutorial <tutorials/#lesson=34_database>`__.

.. _workflow-module-run_history:

Run History
~~~~~~~~~~~

Inspect saved run status, settings, outputs and logs; a completed run does not establish biological validity.

**Open:** the application's Help/tools menus.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.

**Outputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.

:doc:`API reference </api/spacr/run_journal/index>`.

`Module tutorial <tutorials/#lesson=40_run_history>`__.

.. _workflow-module-report:

Report
~~~~~~

Package recorded results, settings and QC into a shareable report without silently rerunning the analysis.

**Open:** the application's Help/tools menus.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.
* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Outputs**

* **Shareable reports** — Exported HTML/PDF or methods/results documents derived from recorded analysis outputs.

:doc:`API reference </api/spacr/report/index>`.

`Module tutorial <tutorials/#lesson=29_report>`__.

.. _workflow-module-data_manager:

Data Manager
~~~~~~~~~~~~

Inspect project files and disk usage. Removing derived artifacts invalidates downstream uses of those files; original images must remain recoverable.

**Open:** the application's Help/tools menus.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.

**Outputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.

:doc:`API reference </api/spacr/data_manager/index>`.

`Module tutorial <tutorials/#lesson=44_data_manager>`__.

.. _workflow-module-pipeline_graph:

Pipeline Graph
~~~~~~~~~~~~~~

Read recorded artifact dependencies and stale/missing status. The graph describes provenance and does not regenerate missing artifacts.

**Open:** the application's Help/tools menus.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/pipeline_graph/index>`.

`Module tutorial <tutorials/#lesson=52_pipeline_graph>`__.

.. _workflow-module-qc_dashboard:

QC
~~

Inspect stored segmentation, unit, leakage and plate-effect checks. Open its nested viewers for detailed inspection.

**Open:** Home → QC.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Review available checks and missing evidence.

:doc:`API reference </api/spacr/qt/screens/qc_dashboard/index>`.

`Module tutorial <tutorials/#lesson=54_qc_dashboard>`__.

.. _workflow-module-lineage:

Lineage
~~~~~~~

Inspect recorded cell/nucleus/pathogen/organelle containment links; this is object containment, not time-series lineage inference.

**Open:** Help search → Database Browser → Lineage.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/lineage/index>`.

`Module tutorial <tutorials/#lesson=56_lineage>`__.

.. _workflow-module-experiment_design:

Experiment Design
~~~~~~~~~~~~~~~~~

Enter conditions, controls, replicates and plate constraints, then export the experimental layout before acquisition.

**Open:** Home → Experiment Design.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Screen design and power estimates** — Saved planning tables or figures based on explicitly chosen effect sizes, variability and sampling assumptions.

**Outputs**

* **Experimental layout** — Exported plate/condition/control/replicate map. Keep its plate and well identifiers consistent with the acquired data.

**After this module**

* :ref:`Power / Design <workflow-module-power>`: Use the experimental layout to define sampling assumptions, then revise the design.
* :ref:`Dose–Response <workflow-module-dose_response>`: Acquire and measure the experiment first, preserving dose/condition identities.

:doc:`API reference </api/spacr/qt/screens/experiment_design/index>`.

`Module tutorial <tutorials/#lesson=67_experiment_design>`__.

.. _workflow-module-layer_viewer:

Layer Viewer
~~~~~~~~~~~~

Inspect aligned image, label and point/ROI layers for the selected field.

**Open:** QC → Layer Viewer.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/layer_viewer/index>`.

`Module tutorial <tutorials/#lesson=57_layer_viewer>`__.

.. _workflow-module-power:

Power / Design
~~~~~~~~~~~~~~

Estimate sampling needs from chosen effect sizes and variability; pilot measurements are optional evidence for assumptions.

**Open:** Home → Power / Design.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Screen design and power estimates** — Saved planning tables or figures based on explicitly chosen effect sizes, variability and sampling assumptions.
* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Experimental layout** — Exported plate/condition/control/replicate map. Keep its plate and well identifiers consistent with the acquired data.

**Outputs**

* **Screen design and power estimates** — Saved planning tables or figures based on explicitly chosen effect sizes, variability and sampling assumptions.

**Before this module**

* :ref:`Experiment Design <workflow-module-experiment_design>`: Use the experimental layout to define sampling assumptions, then revise the design.

:doc:`API reference </api/spacr/qt/screens/power/index>`.

`Module tutorial <tutorials/#lesson=68_power_design>`__.

.. _workflow-module-run_compare:

Run Compare
~~~~~~~~~~~

Compare compatible saved settings, counts and hit lists without treating changed input data as a controlled model comparison.

**Open:** Home → Run Compare.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.
* **Regression results and hits** — Selected run results folder: coefficient/result CSVs, hit tables, settings and diagnostic figures.

**Outputs**

* **Run/model comparison** — Comparison tables and figures from compatible saved runs or masks; agreement is not ground-truth accuracy.

**Before this module**

* :ref:`Regression <workflow-module-regression>`: Compare compatible saved result sets and their settings.

:doc:`API reference </api/spacr/qt/screens/run_compare/index>`.

`Module tutorial <tutorials/#lesson=50_run_compare>`__.

.. _workflow-module-tabulate:

Tabulate
~~~~~~~~

Choose grouping variables and measurement columns, then inspect pivot tables and groupwise sample sizes.

**Open:** Help search → Database Browser → Tabulate.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/tabulate/index>`.

`Module tutorial <tutorials/#lesson=61_tabulate>`__.

.. _workflow-module-dose_response:

Dose–Response
~~~~~~~~~~~~~

Supply dose and response columns, controls and units before curve fitting; this route does not require sequencing.

**Open:** Home → Dose–Response.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Experimental layout** — Exported plate/condition/control/replicate map. Keep its plate and well identifiers consistent with the acquired data.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Experiment Design <workflow-module-experiment_design>`: Acquire and measure the experiment first, preserving dose/condition identities.
* :ref:`Measure <workflow-module-measure>`: Join the measured response to explicit doses and controls.

:doc:`API reference </api/spacr/qt/screens/dose_response/index>`.

`Module tutorial <tutorials/#lesson=69_dose_response>`__.

.. _workflow-module-project_browser:

Project Browser
~~~~~~~~~~~~~~~

Locate projects and inspect their stage, last run, disk use and stale outputs.

**Open:** the application's Help/tools menus.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.

**Outputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.

:doc:`API reference </api/spacr/qt/screens/project_browser/index>`.

`Module tutorial <tutorials/#lesson=45_project_browser>`__.

.. _workflow-module-outliers:

Outliers
~~~~~~~~

Review robust object/well outlier scores and the selected exclusion rule before carrying filtered data into analysis.

**Open:** QC → Outliers.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/outliers/index>`.

`Module tutorial <tutorials/#lesson=66_outliers>`__.

.. _workflow-module-embeddings:

Embeddings
~~~~~~~~~~

Encode object images with a chosen model and channel policy. Retain object identities and encoder provenance when supplying the features to downstream exploration.

**Open:** Home → Embeddings.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Outputs**

* **Image embeddings** — Object-indexed encoder features, with channel policy and encoder provenance. The encoding API returns features; persistence is caller-dependent.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Retain encoder and channel-policy provenance.

**After this module**

* :ref:`Image UMAP <workflow-module-umap>`: Supply the encoder feature table with matching object IDs; do not assume every GUI route automatically persists it.

:doc:`API reference </api/spacr/qt/screens/embeddings/index>`.

`Module tutorial <tutorials/#lesson=77_embeddings>`__.

.. _workflow-module-control_chart:

Control Charts
~~~~~~~~~~~~~~

Compare designated controls across plates and inspect drift; the definition of a control comes from the experiment.

**Open:** QC → Control Charts.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Experimental layout** — Exported plate/condition/control/replicate map. Keep its plate and well identifiers consistent with the acquired data.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/control_chart/index>`.

`Module tutorial <tutorials/#lesson=51_control_charts>`__.

.. _workflow-module-train_compare:

Training Runs
~~~~~~~~~~~~~

Compare saved training histories, metrics and settings from compatible model runs.

**Open:** Classify → Training Runs.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.

**Outputs**

* **Run/model comparison** — Comparison tables and figures from compatible saved runs or masks; agreement is not ground-truth accuracy.

:doc:`API reference </api/spacr/train_compare/index>`.

`Module tutorial <tutorials/#lesson=28_training_runs>`__.

.. _workflow-module-align:

Align & Stitch
~~~~~~~~~~~~~~

Inspect tile geometry, overlap and channel mapping, then save the mosaic with coordinate records. OPS also needs the original cycle/site identities.

**Open:** Home → Align & Stitch.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

**Outputs**

* **Aligned mosaic and coordinates** — Align & Stitch destination: composed image and the tile-coordinate/layout records needed to interpret it.

**After this module**

* :ref:`OPS <workflow-module-ops>`: Carry tile coordinates and original sequencing cycles into OPS; a flattened mosaic alone is insufficient.

:doc:`API reference </api/spacr/align/index>`.

`Module tutorial <tutorials/#lesson=32_align_stitch>`__.

.. _workflow-module-make_masks:

Make Masks
~~~~~~~~~~

Curate image/mask pairs for segmentation training, or use FEATURES to assign images and masks and invoke measurement. Saving a mask does not train a classifier.

**Open:** Home → Make Masks.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**Outputs**

* **Curated training fields** — Separate image and integer-mask files with matching field identities; preserve original images and labels.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**After this module**

* :ref:`Cellpose Workbench <workflow-module-train_cellpose>`: Use independently checked image/mask pairs.
* :ref:`Measure <workflow-module-measure>`: Use FEATURES to pair images and masks and write a measured project; standalone masks are not merged arrays.
* :ref:`Plaque Assay <workflow-module-analyze_plaques>`: Use plaque masks with matching source images; cell masks are not automatically plaque labels.

:doc:`API reference </api/spacr/qt/screens/make_masks/index>`.

`Module tutorial <tutorials/#lesson=14_make_masks>`__.

.. _workflow-module-plate_view:

Plate Viewer
~~~~~~~~~~~~

Aggregate the selected measurement by well and inspect plate patterns and controls.

**Open:** Graph Builder → Plate Viewer.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Experimental layout** — Exported plate/condition/control/replicate map. Keep its plate and well identifiers consistent with the acquired data.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/plate_qc/index>`.

.. _workflow-module-umap:

Image UMAP
~~~~~~~~~~

Project measured features or supplied encoder features and inspect representative crops. A cluster is a candidate grouping, not a validated phenotype. Use the lasso and annotation controls to write reviewed selections to an annotation column in the matching measurement database. A geometric selection alone does not establish a biological phenotype.

**Open:** Home → Image UMAP.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.
* **Image embeddings** — Object-indexed encoder features, with channel policy and encoder provenance. The encoding API returns features; persistence is caller-dependent.

**Outputs**

* **Projection and clusters** — Image UMAP/PCA coordinate tables, selected clusters and figures for the loaded measurement data.
* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Choose feature columns and inspect representative crops.
* :ref:`Embeddings <workflow-module-embeddings>`: Supply the encoder feature table with matching object IDs; do not assume every GUI route automatically persists it.

**After this module**

* :ref:`Gate Editor <workflow-module-gate_editor>`: Supply the matching coordinate/feature columns when defining a selection.
* :ref:`Classify <workflow-module-classify_merged>`: Write reviewed lasso selections to an annotation column in the matching object database, then select that column in Classify. Inspect crops and validate labels; embedding clusters are not ground truth.

:doc:`API reference </api/spacr/core/index>`.

`Module tutorial <tutorials/#lesson=15_image_umap>`__.

.. _workflow-module-profiler:

Prediction Profiler
~~~~~~~~~~~~~~~~~~~

Inspect how a fitted model responds as an input feature changes; predictions do not establish causal effects.

**Open:** Regression → Prediction Profiler.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Fitted feature classifier** — The fitted tabular classifier and its recorded feature list, training settings and validation results.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/profiler/index>`.

`Module tutorial <tutorials/#lesson=53_prediction_profiler>`__.

.. _workflow-module-graph_builder:

Graph Builder
~~~~~~~~~~~~~

Choose variables, groups and plotting settings from the loaded table, then export the figure with its analysis context.

**Open:** Home → Graph Builder.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Choose explicit variables, groups and filters.

:doc:`API reference </api/spacr/qt/screens/graph_builder/index>`.

`Module tutorial <tutorials/#lesson=58_graph_builder>`__.

.. _workflow-module-investigate_hit:

Investigate Hit
~~~~~~~~~~~~~~~

Link a regression hit to candidate objects and quantitative well-level evidence. This does not independently validate the hit.

**Open:** Regression → Investigate Hit.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Regression results and hits** — Selected run results folder: coefficient/result CSVs, hit tables, settings and diagnostic figures.
* **Object classification scores** — Saved score CSVs and, when merged, measurements/measurements.db, table png_list.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``pred``, ``cv_predictions``, ``ml_pred``, ``predictions``.
* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Regression <workflow-module-regression>`: Join compatible phenotype and object data for the chosen hit.

:doc:`API reference </api/spacr/hit_investigation/index>`.

.. _workflow-module-gate_editor:

Gate Editor
~~~~~~~~~~~

Define threshold or polygon gates on actual feature/coordinate columns, then apply the saved gate to compatible objects. Use Annotate to write the displayed gates to an annotation column; review the selected objects and choose binary or multiclass labels.

**Open:** Home → Gate Editor.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Projection and clusters** — Image UMAP/PCA coordinate tables, selected clusters and figures for the loaded measurement data.

**Outputs**

* **Reusable gates** — Saved threshold/polygon gate definitions or a selected object set; apply a gate to the same feature definitions.
* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.

**Before this module**

* :ref:`Image UMAP <workflow-module-umap>`: Supply the matching coordinate/feature columns when defining a selection.
* :ref:`Measure <workflow-module-measure>`: Use the actual measured feature definitions and units.

**After this module**

* :ref:`Annotate <workflow-module-annotate>`: Apply compatible gates, then review candidate labels.
* :ref:`Classify <workflow-module-classify_merged>`: Write reviewed gate selections to an annotation column, then select that same column and object population in Classify. Keep validation objects separate from training labels.

:doc:`API reference </api/spacr/qt/screens/gate_editor/index>`.

`Module tutorial <tutorials/#lesson=64_gate_editor>`__.

.. _workflow-module-feature_dict:

Feature Dictionary
~~~~~~~~~~~~~~~~~~

Look up feature definitions and units before choosing measurement columns.

**Open:** the application's Help/tools menus.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Feature definitions** — The installed feature dictionary; definitions describe existing measurements and do not compute them.

**Outputs**

* **Feature definitions** — The installed feature dictionary; definitions describe existing measurements and do not compute them.

:doc:`API reference </api/spacr/feature_dict/index>`.

`Module tutorial <tutorials/#lesson=62_feature_dictionary>`__.

.. _workflow-module-trellis:

Small Multiples
~~~~~~~~~~~~~~~

Compare groups in small-multiple plots with explicit shared or independent axes.

**Open:** Graph Builder → Small Multiples.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/trellis/index>`.

`Module tutorial <tutorials/#lesson=63_small_multiples>`__.

.. _workflow-module-analyze_plaques:

Plaque Assay
~~~~~~~~~~~~

Analyse plaque images or existing masks with the configured plaque model. This route need not pass through Measure; use the dedicated plaque example.

**Open:** Toxoplasma → Plaque Assay.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**Before this module**

* :ref:`Make Masks <workflow-module-make_masks>`: Use plaque masks with matching source images; cell masks are not automatically plaque labels.

:doc:`API reference </api/spacr/submodules/index>`.

`Module tutorial <tutorials/#lesson=24_plaque>`__.

.. _workflow-module-recruitment:

Recruitment
~~~~~~~~~~~

Use compartment intensity measurements and matching host/pathogen identities to compute recruitment ratios.

**Open:** Toxoplasma → Recruitment.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Require the intended compartment intensities and identities.

:doc:`API reference </api/spacr/submodules/index>`.

`Module tutorial <tutorials/#lesson=25_recruitment>`__.

.. _workflow-module-invasion:

Invasion Assay
~~~~~~~~~~~~~~

Use the required two-colour differential-staining measurements and stain-baseline controls to distinguish attachment from invasion.

**Open:** Toxoplasma → Invasion Assay.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Require two-colour stain measurements and appropriate baseline controls.

:doc:`API reference </api/spacr/submodules/index>`.

`Module tutorial <tutorials/#lesson=26_invasion>`__.

.. _workflow-module-replication:

Replication Assay
~~~~~~~~~~~~~~~~~

Count parasites using explicit vacuole identity and compare condition distributions; host identity alone does not define a vacuole.

**Open:** Toxoplasma → Replication Assay.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Require explicit parasite-to-vacuole identities.

:doc:`API reference </api/spacr/submodules/index>`.

`Module tutorial <tutorials/#lesson=27_replication>`__.

.. _workflow-module-train_cellpose:

Cellpose Workbench
~~~~~~~~~~~~~~~~~~

Open Cellpose Workbench inside Make Masks and train from verified image/mask pairs. Evaluate on separate fields before selecting the checkpoint in Mask.

**Open:** Make Masks → Cellpose Workbench.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Curated training fields** — Separate image and integer-mask files with matching field identities; preserve original images and labels.

**Outputs**

* **Segmentation checkpoint** — Saved Cellpose-compatible checkpoint or a compatible installed backend selected with its own configuration.

**Before this module**

* :ref:`Make Masks <workflow-module-make_masks>`: Use independently checked image/mask pairs.

**After this module**

* :ref:`Mask <workflow-module-mask>`: Select the saved compatible checkpoint in Mask.
* :ref:`Direct Cellpose mask generation <workflow-module-cellpose_masks>`: Pass the trained checkpoint as custom_model with the matching image channels and preprocessing.

:doc:`API reference </api/spacr/submodules/index>`.

`Module tutorial <tutorials/#lesson=19_train_cellpose>`__.

.. _workflow-module-cellpose_all:

Mask the whole folder
~~~~~~~~~~~~~~~~~~~~~

Apply the selected segmentation model to the open image folder through Make Masks; inspect saved labels before measuring them.

**Open:** Make Masks → Mask the whole folder.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Segmentation checkpoint** — Saved Cellpose-compatible checkpoint or a compatible installed backend selected with its own configuration.

**Outputs**

* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

:doc:`API reference </api/spacr/spacr_cellpose/index>`.

`Module tutorial <tutorials/#lesson=20_cellpose_masks>`__.

.. _workflow-module-model_compare:

Model Compare
~~~~~~~~~~~~~

Compare model masks on identical fields. Without independent reference labels, agreement does not measure segmentation accuracy.

**Open:** Make Masks → Model Compare.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.
* **Segmentation checkpoint** — Saved Cellpose-compatible checkpoint or a compatible installed backend selected with its own configuration.

**Outputs**

* **Run/model comparison** — Comparison tables and figures from compatible saved runs or masks; agreement is not ground-truth accuracy.

:doc:`API reference </api/spacr/model_compare/index>`.

`Module tutorial <tutorials/#lesson=21_model_compare>`__.

.. _workflow-module-model_zoo:

Model Zoo
~~~~~~~~~

Inspect model provenance and download/install compatible checkpoints or backends. A listed backend is not itself a checkpoint file.

**Open:** Make Masks → Model Zoo.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Segmentation checkpoint** — Saved Cellpose-compatible checkpoint or a compatible installed backend selected with its own configuration.

**Outputs**

* **Segmentation checkpoint** — Saved Cellpose-compatible checkpoint or a compatible installed backend selected with its own configuration.

:doc:`API reference </api/spacr/model_zoo/index>`.

`Module tutorial <tutorials/#lesson=22_model_zoo>`__.

.. _workflow-module-curate:

Curate
~~~~~~

Correct labels and track assignments while retaining a record of the edits.

**Open:** Make Masks → Curate.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.
* **Linked time-series objects** — Tracked labels and frame/object associations from the time-series project, with frame interval and units.

**Outputs**

* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.
* **Linked time-series objects** — Tracked labels and frame/object associations from the time-series project, with frame interval and units.

:doc:`API reference </api/spacr/qt/screens/curate/index>`.

`Module tutorial <tutorials/#lesson=42_curate>`__.

.. _workflow-module-napari_bridge:

Napari Bridge
~~~~~~~~~~~~~

Send the matching image/labels to napari and import the revised labels back with field identity preserved.

**Open:** Make Masks → Napari Bridge.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**Outputs**

* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

:doc:`API reference </api/spacr/napari_bridge/index>`.

`Module tutorial <tutorials/#lesson=46_napari_bridge>`__.

.. _workflow-module-import_images:

Import Images
~~~~~~~~~~~~~

Review field identities, image channels and optional external masks before writing a separate project. Image-only imports still need segmentation before measurement.

**Open:** Import → Import Images.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

**Outputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**After this module**

* :ref:`Mask <workflow-module-mask>`: Use imported image planes and identities; image-only imports still need segmentation.

:doc:`API reference </api/spacr/image_import/index>`.

`Module tutorial <tutorials/#lesson=74_import_images>`__.

.. _workflow-module-barcode_qc:

Barcode QC
~~~~~~~~~~

Inspect mapping depth, coverage, collisions and positional effects before guide-count filtering.

**Open:** Map Barcodes → Barcode QC.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Guide counts per well** — Map Barcodes run folder: unique_combinations.csv and annotated_reads.h5. Well identity requires the corresponding barcode references.
  Relevant columns, depending on the route: ``count``.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Map Barcodes <workflow-module-map_barcodes>`: Check mapping and coverage before guide filtering.

:doc:`API reference </api/spacr/sequencing_qc/index>`.

`Module tutorial <tutorials/#lesson=47_barcode_qc>`__.

.. _workflow-module-image_scatter:

Image Scatter
~~~~~~~~~~~~~

Plot selected measurement columns with object-crop inspection.

**Open:** Image UMAP → Image Scatter.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Outputs**

* **Projection and clusters** — Image UMAP/PCA coordinate tables, selected clusters and figures for the loaded measurement data.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/image_scatter/index>`.

`Module tutorial <tutorials/#lesson=55_image_scatter>`__.

.. _workflow-module-pca:

PCA
~~~

Inspect principal components and feature loadings for the selected measured objects.

**Open:** Image UMAP → PCA.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Projection and clusters** — Image UMAP/PCA coordinate tables, selected clusters and figures for the loaded measurement data.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/pca/index>`.

`Module tutorial <tutorials/#lesson=60_pca>`__.

.. _workflow-module-volcano_explorer:

Volcano Explorer
~~~~~~~~~~~~~~~~

Open a saved regression result and inspect effect sizes, adjusted significance and individual hit records.

**Open:** Regression → Volcano Explorer.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Regression results and hits** — Selected run results folder: coefficient/result CSVs, hit tables, settings and diagnostic figures.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Regression <workflow-module-regression>`: Inspect saved effects and adjusted significance.

:doc:`API reference </api/spacr/volcano_style/index>`.

`Module tutorial <tutorials/#lesson=72_volcano_explorer>`__.

.. _workflow-module-hit_list:

Hit List
~~~~~~~~

Filter and rank saved hits with their effects, FDR and guide agreement.

**Open:** Regression → Hit List.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Regression results and hits** — Selected run results folder: coefficient/result CSVs, hit tables, settings and diagnostic figures.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Regression <workflow-module-regression>`: Inspect ranked hits and guide agreement.

:doc:`API reference </api/spacr/qt/screens/hit_list/index>`.

`Module tutorial <tutorials/#lesson=48_hit_list>`__.

.. _workflow-module-methods_export:

Methods & Results
~~~~~~~~~~~~~~~~~

Draft methods/results text from recorded settings and results; review every claim before publication.

**Open:** Regression → Methods & Results.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Run history and artifacts** — Project run records, settings, output paths, artifact provenance, status and logs.
* **Regression results and hits** — Selected run results folder: coefficient/result CSVs, hit tables, settings and diagnostic figures.

**Outputs**

* **Shareable reports** — Exported HTML/PDF or methods/results documents derived from recorded analysis outputs.

**Before this module**

* :ref:`Regression <workflow-module-regression>`: Review exported prose and every traced result.

:doc:`API reference </api/spacr/methods_export/index>`.

`Module tutorial <tutorials/#lesson=49_methods_results>`__.

.. _workflow-module-regression_diagnostics:

Diagnostics
~~~~~~~~~~~

Inspect the diagnostic panels saved by the chosen regression run; distinguish missing diagnostics from passed assumptions.

**Open:** Regression → Diagnostics.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Regression results and hits** — Selected run results folder: coefficient/result CSVs, hit tables, settings and diagnostic figures.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Regression <workflow-module-regression>`: Open the diagnostics actually written by this run.

:doc:`API reference </api/spacr/qt/screens/regression/index>`.

`Module tutorial <tutorials/#lesson=75_regression_diagnostics>`__.

.. _workflow-module-illumination:

Illumination
~~~~~~~~~~~~

Estimate and apply flat-field correction at configured stages. Keep corrected display values distinct from raw measurements and record the applied policy.

**Open:** Measure → Illumination.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

**Outputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

:doc:`API reference </api/spacr/illumination/index>`.

`Module tutorial <tutorials/#lesson=43_illumination>`__.

.. _workflow-module-anndata_export:

AnnData Export
~~~~~~~~~~~~~~

Export an AnnData .h5ad file with object metadata and measured features for downstream single-cell analysis.

**Open:** Measure → AnnData Export.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **AnnData export** — An exported .h5ad file containing measured features and object metadata.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Export compatible feature and metadata columns.

:doc:`API reference </api/spacr/anndata_export/index>`.

`Module tutorial <tutorials/#lesson=59_anndata_export>`__.

.. _workflow-module-motility:

Motility Assay
~~~~~~~~~~~~~~

Inspect track speed and infection-related measurements with an explicit frame interval and pixel calibration.

**Open:** Measure → Motility Assay.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Linked time-series objects** — Tracked labels and frame/object associations from the time-series project, with frame interval and units.
* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Timelapse <workflow-module-timelapse>`: Supply frame interval and pixel calibration.
* :ref:`Measure <workflow-module-measure>`: Combine measured objects with matching tracks.

:doc:`API reference </api/spacr/timelapse/index>`.

`Module tutorial <tutorials/#lesson=18_motility>`__.

.. _workflow-module-timelapse:

Timelapse
~~~~~~~~~

Open Timelapse within Mask to segment and link objects across ordered frames; inspect links before downstream motility analysis.

**Open:** Mask → Timelapse.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

**Outputs**

* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.
* **Linked time-series objects** — Tracked labels and frame/object associations from the time-series project, with frame interval and units.

**Before this module**

* :ref:`Mask <workflow-module-mask>`: Enable the nested time-series route before generating linked labels.

**After this module**

* :ref:`Measure <workflow-module-measure>`: Use the time-series project with stable frame/object identities.
* :ref:`Motility Assay <workflow-module-motility>`: Supply frame interval and pixel calibration.

:doc:`API reference </api/spacr/core/index>`.

`Module tutorial <tutorials/#lesson=17_timelapse>`__.

.. _workflow-module-ops:

OPS
~~~

Use sequencing-cycle images and phenotype alignment to decode barcodes per nucleus. Aggregate/join the decoded identities to compatible phenotype inputs before Regression; the OPS database is not a drop-in FASTQ count CSV.

**Open:** Mask → OPS.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Aligned mosaic and coordinates** — Align & Stitch destination: composed image and the tile-coordinate/layout records needed to interpret it.

**Outputs**

* **Optical barcode assignments** — OPS destination measurements.db: per-well geometry, phenotype alignment, nuclei and barcode tables; optional per-cycle reads.
  Relevant tables, depending on the route: ``ops_geometry``, ``ops_phenotype``, ``ops_objects``, ``ops_barcodes``, ``ops_reads``.

**Before this module**

* :ref:`Align & Stitch <workflow-module-align>`: Carry tile coordinates and original sequencing cycles into OPS; a flattened mosaic alone is insufficient.

**After this module**

* :ref:`Regression <workflow-module-regression>`: Join/aggregate decoded objects to phenotype and guide inputs explicitly before Regression; this is not a direct CSV handoff.

:doc:`API reference </api/spacr/ops_engine/index>`.

`Module tutorial <tutorials/#lesson=76_ops>`__.

.. _workflow-module-classifier_evaluation:

Classifier Evaluation
~~~~~~~~~~~~~~~~~~~~~

Inspect held-out predictions, calibration and split/leakage evidence for the matching classifier.

**Open:** Classify → Classifier Evaluation.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Classifier evaluation bundle** — Held-out predictions, labels, split metadata and calibration/leakage metrics for a saved classifier run.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Classify <workflow-module-classify_merged>`: Use the matching held-out predictions and split metadata.

:doc:`API reference </api/spacr/classifier_evaluation/index>`.

`Module tutorial <tutorials/#lesson=39_classifier_evaluation>`__.

.. _workflow-module-explain_cv:

Explain CV Model
~~~~~~~~~~~~~~~~

Fit a measured-feature surrogate to existing computer-vision scores, preserving identity joins. A surrogate explanation is not causality.

**Open:** Classify → Explain CV Model.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object classification scores** — Saved score CSVs and, when merged, measurements/measurements.db, table png_list.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``pred``, ``cv_predictions``, ``ml_pred``, ``predictions``.

**Outputs**

* **Fitted feature classifier** — The fitted tabular classifier and its recorded feature list, training settings and validation results.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Classify <workflow-module-classify_merged>`: Select CV predictions and the matching measured objects.

:doc:`API reference </api/spacr/surrogate/index>`.

`Module tutorial <tutorials/#lesson=70_explain_cv>`__.

.. _workflow-module-activation:

Activation
~~~~~~~~~~

Apply the matching image classifier and preprocessing to inspect saliency/model-response maps.

**Open:** Classify → Activation.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Trained image classifier** — Saved classifier checkpoint, model settings and matching channel/preprocessing configuration.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Outputs**

* **Image attribution maps** — Saved model-response/saliency images for the matching classifier and input preprocessing.

**Before this module**

* :ref:`Classify <workflow-module-classify_merged>`: Also provide the matching crops and channel preprocessing.

:doc:`API reference </api/spacr/deep_spacr/index>`.

`Module tutorial <tutorials/#lesson=16_activation>`__.

.. _workflow-module-agreement:

Annotator Agreement
~~~~~~~~~~~~~~~~~~~

Compare independent annotation columns and inspect discordant objects; agreement is separate from classification accuracy.

**Open:** Annotate → Annotator Agreement.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Annotate <workflow-module-annotate>`: Choose independent annotation columns for the same objects.

:doc:`API reference </api/spacr/agreement/index>`.

`Module tutorial <tutorials/#lesson=23_agreement>`__.

.. _workflow-module-classify:

Computer Vision
~~~~~~~~~~~~~~~

Train an image classifier from labelled object crops, archives or supported streamed crops. Match channels and preprocessing when applying its checkpoint.

**Open:** Classify → Computer Vision.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.
* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.

**Outputs**

* **Object classification scores** — Saved score CSVs and, when merged, measurements/measurements.db, table png_list.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``pred``, ``cv_predictions``, ``ml_pred``, ``predictions``.
* **Trained image classifier** — Saved classifier checkpoint, model settings and matching channel/preprocessing configuration.
* **Classifier evaluation bundle** — Held-out predictions, labels, split metadata and calibration/leakage metrics for a saved classifier run.

:doc:`API reference </api/spacr/deep_spacr/index>`.

`Module tutorial <tutorials/#lesson=10_classify_cv>`__.

.. _workflow-module-ml_analyze:

Tabular Machine Learning
~~~~~~~~~~~~~~~~~~~~~~~~

Train a feature-based classifier from measured objects and labels. Inspect missing-feature exclusions and grouped held-out performance before using scores.

**Open:** Classify → Tabular Machine Learning.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.

**Outputs**

* **Object classification scores** — Saved score CSVs and, when merged, measurements/measurements.db, table png_list.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``pred``, ``cv_predictions``, ``ml_pred``, ``predictions``.
* **Fitted feature classifier** — The fitted tabular classifier and its recorded feature list, training settings and validation results.
* **Classifier evaluation bundle** — Held-out predictions, labels, split metadata and calibration/leakage metrics for a saved classifier run.

:doc:`API reference </api/spacr/ml/index>`.

`Module tutorial <tutorials/#lesson=11_classify_ml>`__.

.. _workflow-module-parameter_sweep:

Parameter Sweep
~~~~~~~~~~~~~~~

Run explicitly selected regression settings and compare their diagnostics. Keep the input data and evaluation question fixed.

**Open:** Regression → Parameter Sweep.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Object classification scores** — Saved score CSVs and, when merged, measurements/measurements.db, table png_list.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``pred``, ``cv_predictions``, ``ml_pred``, ``predictions``.
* **Guide counts per well** — Map Barcodes run folder: unique_combinations.csv and annotated_reads.h5. Well identity requires the corresponding barcode references.
  Relevant columns, depending on the route: ``count``.

**Outputs**

* **Run/model comparison** — Comparison tables and figures from compatible saved runs or masks; agreement is not ground-truth accuracy.
* **Regression results and hits** — Selected run results folder: coefficient/result CSVs, hit tables, settings and diagnostic figures.

:doc:`API reference </api/spacr/parameter_sweep/index>`.

`Module tutorial <tutorials/#lesson=73_parameter_sweep>`__.

.. _workflow-module-toxoplasma:

Toxoplasma
~~~~~~~~~~

Open the organism guide from Home, inspect the compartment diagram, then select Plaque Assay, Recruitment, Invasion Assay or Replication Assay. The four Coming soon tiles describe proposals and cannot run. The guide itself does not analyse a project or produce a measurement table.

**Open:** Home → Toxoplasma.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Bundled organism reference** — Installed organism descriptions and SwissBioPics cell diagrams; no project input is required.

**Outputs**

* **Organism guide and assay selection** — GUI-only compartment highlights and navigation to available assays; no measurements or files are produced.

:doc:`API reference </api/spacr/qt/screens/organism_screen/index>`.

`Module tutorial <tutorials/#lesson=82_toxoplasma>`__.

.. _workflow-module-plasmodium:

Plasmodium spp.
~~~~~~~~~~~~~~~

Open the organism guide from Home to read the planned malaria imaging workflows and inspect the shared apicomplexan diagram. All eight assay tiles are Coming soon; none starts an analysis or produces a result.

**Open:** Home → Plasmodium spp..

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Bundled organism reference** — Installed organism descriptions and SwissBioPics cell diagrams; no project input is required.

**Outputs**

* **Organism guide and assay selection** — GUI-only compartment highlights and navigation to available assays; no measurements or files are produced.

:doc:`API reference </api/spacr/qt/screens/organism_screen/index>`.

`Module tutorial <tutorials/#lesson=83_plasmodium>`__.

.. _workflow-module-candida:

Candida spp.
~~~~~~~~~~~~

Open the organism guide from Home to read the planned fungal imaging workflows and inspect the generic budding-yeast diagram. All eight assay tiles are Coming soon; none starts an analysis or produces a result.

**Open:** Home → Candida spp..

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Bundled organism reference** — Installed organism descriptions and SwissBioPics cell diagrams; no project input is required.

**Outputs**

* **Organism guide and assay selection** — GUI-only compartment highlights and navigation to available assays; no measurements or files are produced.

:doc:`API reference </api/spacr/qt/screens/organism_screen/index>`.

`Module tutorial <tutorials/#lesson=84_candida>`__.

.. _workflow-module-endodyogeny:

Endodyogeny size proxy
~~~~~~~~~~~~~~~~~~~~~~

Read measured compartment areas, annotate conditions and bin area ** 1.5 into log2 size doublings. Defaults aggregate pathogen area per host cell, not per vacuole: multiple vacuoles in one cell are combined. This is an area-derived size proxy, not measured volume or a parasite count. Use Replication Assay with explicit vacuole identity for parasites-per-vacuole counts. Configure compartment, area filters, calibration, conditions and grouping before calling the API; saving is optional.

**Use from Python:** :func:`spacr.submodules.analyze_endodyogeny`. This API-only workflow has no Home tile or menu entry.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Host-cell-aggregated compartment areas** — Each src project root/measurements/measurements.db; tables defaults to cell, nucleus, pathogen and cytoplasm, with png_list added for merging. The compartment setting selects the area column; default pathogen_area is summed per host cell.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``, ``png_list``.
  Relevant columns, depending on the route: ``cell_id``, ``pathogen_area``.

**Outputs**

* **Area-derived size-proxy results** — Returned data and chi_squared DataFrames; save=True also writes data.csv, chi_squared_results.csv, chi_squared_pairwise_results.csv and a figure under the first project root/results/analyze_endodyogeny/.
  Relevant columns, depending on the route: ``pathogen_area``, ``pathogen_volume``, ``pathogen_volume_bin``, ``bin_index``.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Supply the measured project roots and required object/png_list tables. Verify host-cell aggregation and area units before interpreting size bins; the Mask counts database alone is insufficient.

:doc:`API reference </api/spacr/submodules/index>`.

.. _workflow-module-cellpose_masks:

Direct Cellpose mask generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run a stock or custom Cellpose model on TIFF fields through the Python API. Despite its historical identify_masks_finetune name, this function performs inference, not training. Review normalization, channels, resizing and model parameters; masks are written only when save is enabled. Import compatible image/mask pairs through External Masks before Measure, or curate the pairs before training. This call does not build a Measure-ready merged project.

**Use from Python:** :func:`spacr.spacr_cellpose.identify_masks_finetune`. This API-only workflow has no Home tile or menu entry.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **TIFF fields for direct Cellpose inference** — Top-level, lowercase .tif files in src; existing same-name files in src/masks are skipped. Supply the configured channels and a compatible stock model_name or custom_model checkpoint.
* **Segmentation checkpoint** — Saved Cellpose-compatible checkpoint or a compatible installed backend selected with its own configuration.

**Outputs**

* **Direct Cellpose TIFF masks** — When save=True, src/masks/<image-name>.tif contains integer labels. The call returns None and does not produce merged arrays or measurements.db.

**Before this module**

* :ref:`Cellpose Workbench <workflow-module-train_cellpose>`: Pass the trained checkpoint as custom_model with the matching image channels and preprocessing.

**After this module**

* :ref:`External Masks <workflow-module-external_masks>`: Provide the saved label TIFFs and their original images to External Masks, assign object roles and create the merged project before Measure.

:doc:`API reference </api/spacr/spacr_cellpose/index>`.

.. _workflow-module-simulation:

Pooled-screen simulation sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the Python sweep API to explore stated screen-design assumptions. It expands combinations, runs simulations in a process pool and writes synthetic summary statistics; it returns None. Set a small explicit max_workers value and a bounded parameter grid before running. The output does not replace measurements or barcode counts for experimental Regression. Use the findings as planning evidence, with their assumptions recorded. Review the synthetic summaries when choosing planning assumptions; transfer those assumptions manually into Power / Design or Experiment Design. Neither tool imports this simulation database, and neither exports the simulation settings grid.

**Use from Python:** :func:`spacr.sim.run_multiple_simulations`. This API-only workflow has no Home tile or menu entry.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Pooled-screen simulation assumptions** — Python settings dictionary: iterable sweep values for screen size, occupancy, classifier accuracy and sequencing assumptions, plus replicates, src, name, variable, plot and max_workers. generate_parameters expands their Cartesian product; begin with a small sweep and an explicit worker bound.

**Outputs**

* **Synthetic screen simulation summaries** — src/<YYMMDD>/<name>/simulations.db; the sweep appends summary rows to simulations and optionally writes plots. These are synthetic performance estimates, not measured experimental hits.
  Relevant tables, depending on the route: ``simulations``.

:doc:`API reference </api/spacr/sim/index>`.

.. _workflow-module-host_pathogen:

Host–Pathogen Analysis
~~~~~~~~~~~~~~~~~~~~~~

Measure whole vacuoles and host reference compartments, keeping uninfected host cells for the infection denominator. Select marker channels and control-calibrated ratio thresholds. Optionally count individually segmented parasites from explicit vacuole links or a measured count column. Review vacuole, host and well tables, joint marker states, replication distributions and unmatched parasite links; unknown measurements remain unknown.

**Open:** Toxoplasma → Host–Pathogen Analysis.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Keep uninfected cells in Measure. Supply whole-vacuole masks, host reference intensities and optional explicit parasite-to-vacuole links; host identity alone does not define a vacuole.

:doc:`API reference </api/spacr/host_pathogen/index>`.

`Module tutorial <tutorials/#lesson=85_host_pathogen>`__.

