Workflow inputs and outputs
---------------------------

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

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=46_napari_bridge>`__.

