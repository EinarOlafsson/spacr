Workflow inputs and outputs
---------------------------

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

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=20_cellpose_masks>`__.

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

