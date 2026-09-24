Workflow inputs and outputs
---------------------------

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

`Module tutorial <../../../tutorials/#lesson=21_model_compare>`__.

