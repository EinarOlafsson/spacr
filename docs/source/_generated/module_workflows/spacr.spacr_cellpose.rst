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

