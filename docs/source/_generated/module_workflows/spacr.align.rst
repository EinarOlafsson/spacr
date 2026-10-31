Workflow inputs and outputs
---------------------------

Align & Stitch
~~~~~~~~~~~~~~

Press Plan to compute the tile layout and list tiles that fell back to their stage positions; Plan writes nothing. Write stack then writes the stitched stack to the Write to folder or .npy file and records tile coordinates in measurements.db when the Database field is set. Every channel uses the registration measured on Ref. channel. OPS also needs the original cycle/site identities.

**Open:** Home → Align & Stitch.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

**Outputs**

* **Aligned mosaic and coordinates** — The Write to folder or .npy file: the stitched stack. Tile coordinates are saved to the measurements.db named in Database only when that field is set.

**After this module**

* :ref:`OPS <workflow-module-ops>`: Carry tile coordinates and original sequencing cycles into OPS; a flattened mosaic alone is insufficient.

:doc:`API reference </api/spacr/align/index>`.

`Module tutorial <../../../tutorials/#lesson=32_align_stitch>`__.

