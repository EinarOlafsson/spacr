Explain CV models and investigate hits
======================================

spaCR separates three questions that are easy to blur together:

* activation or occlusion maps show **where** a vision model attended;
* the **Explain CV Model** module shows which measured morphology can
  reproduce the model's decisions; and
* **Investigate Hit** asks which cells in wells supporting a regression hit
  have morphology consistent with that hit.

The last result is weakly supervised evidence, not a cell-resolved barcode.
The UI and exports therefore use ``EAF1-hit-like`` and
``hit_like_probability`` rather than calling a cell ``EAF1-KO``.

Explain CV Model
----------------

Choose the exact ``measurements.db`` and an existing per-object CV prediction
CSV. The module never reruns or silently substitutes the original classifier.
It joins predictions one-to-one to measured objects and excludes identifiers,
prediction/class columns, scores, annotations, and user exclusions from the
surrogate feature matrix.

The surrogate family is a fixed choice: Random Forest, scikit-learn histogram
gradient boosting, or XGBoost when that optional dependency is installed.
The exact backend and package version are saved. Wells are held intact by
default; plate-held-out validation is available when the experiment has enough
plates.

Read the output in this order:

1. held-out fidelity, the majority-class baseline, and improvement over it;
2. class-wise precision/recall/F1 and the confusion matrix; then
3. gain, held-out permutation importance, and mean absolute SHAP.

If fidelity does not clear the configured improvement over the majority
baseline, spaCR withholds the importance table. This means the surrogate did
not reproduce the CV model well enough for its feature ranking to explain that
model. Per-cell signed SHAP values, correlated-feature pairs, exact feature and
leakage-exclusion lists, split units, random seed, model parameters, and sampled
object IDs are retained in the artifacts.

Investigate Hit
---------------

Open **Hit List**, select the exact gene result, and choose
**Investigate selected**. The hand-off carries the result folder, gene, effect
direction, FDR, phenotype, and agreeing guides; it does not search for the
newest results file. Investigate Hit hashes the selected regression CSV/JSON
bytes into its provenance.

Supply the measurements database, the original per-object prediction CSV, and
the per-well guide-fraction CSV. Joins use explicit object and plate/row/column
keys. Duplicate object keys or duplicate well/guide fractions stop the run.

The first output is an honest review queue: cells in target-containing wells
ranked in the regression's effect direction, with the well-level target-guide
fraction shown beside them. The optional attribution model then:

* treats wells as bags and cells as instances;
* learns background and target-like morphology across many wells;
* uses guide fraction as a noisy learned prior, never as a forced cell
  prevalence;
* scores every cell with a model that excluded its well, or its plate when the
  design supports plate cross-fitting; and
* keeps independent guides visible before reporting gene-level consensus.

The original phenotype score is excluded from the morphology model by default.
Including it is an explicit, warned setting because selecting, embedding, and
validating with the same score would be circular.

Quantitative evidence
---------------------

The report uses wells, not cells, as the independent experimental unit. It
reports target-versus-control candidate prevalence, a well-bootstrap interval,
a plate-blocked permutation test, guide-dose response, guide-specific effects,
and two stricter nulls. The stricter guide-fraction and well-label nulls repeat
the cross-fitting pipeline inside every permutation rather than shuffling only
the final summary labels.

The two-dimensional comparison is a PCA fit only on target-free control-cell
morphology and then used to transform all cells. Guide identity, target-well
status, and the original classifier output are excluded. The exported table
contains exact object keys and sample counts so a visual cluster can be traced
back to its wells and images; the quantitative enrichment remains the primary
claim.

A stratified gallery samples high, borderline, low, and false-looking control
objects. The reviewer CSV is shuffled and reveals only a random review ID and
image path. Its separate analyst-only key retains the stratum and probability.
After one or more blinded reviewers return binary calls,
``evaluate_blinded_reviews`` reports calibration error, precision, recall,
ROC AUC when defined, and pairwise Cohen's kappa.

Database annotations are versioned in dedicated provenance tables. Existing
hand annotations are unchanged. **Promote calls to annotation** is a separate
action that accepts only a fresh column name, and **Undo promotion** clears
exactly the values written by that promotion while preserving its audit row.

Evidence ladder
---------------

A coherent UMAP/PCA population is useful exploratory evidence but is not
ground truth. Confidence increases with negative and positive controls,
independent guides, replicate plates, well/plate cross-fitting, refitted nulls,
bootstrap stability, blinded review, and agreement between measured-feature
SHAP and pixel-level attribution. The strongest validation is an orthogonal
experiment that assigns perturbation identity per cell, such as an arrayed
perturbation/rescue or direct in-situ guide-barcode readout.

Headless entry points
---------------------

.. code-block:: python

   from spacr.surrogate import run_explain_cv
   from spacr.hit_investigation import investigate_hit

   explanation = run_explain_cv(explain_settings)
   investigation = investigate_hit(hit_settings)

The statistical APIs are :mod:`spacr.surrogate`,
:mod:`spacr.hit_attribution`, and :mod:`spacr.hit_investigation`.
