Workflow inputs and outputs
---------------------------

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

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=14_make_masks>`__.

