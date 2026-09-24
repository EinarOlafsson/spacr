Workflow inputs and outputs
---------------------------

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

`Module tutorial <../../../tutorials/#lesson=73_parameter_sweep>`__.

