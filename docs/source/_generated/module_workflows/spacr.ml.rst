Workflow inputs and outputs
---------------------------

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

`Module tutorial <../../../tutorials/#lesson=13_regression>`__.

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

`Module tutorial <../../../tutorials/#lesson=11_classify_ml>`__.

