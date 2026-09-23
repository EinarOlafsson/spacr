Workflow inputs and outputs
---------------------------

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

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=32_align_stitch>`__.

