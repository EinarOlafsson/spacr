Workflow inputs and outputs
---------------------------

OPS
~~~

Use sequencing-cycle images and phenotype alignment to decode barcodes per nucleus. Aggregate/join the decoded identities to compatible phenotype inputs before Regression; the OPS database is not a drop-in FASTQ count CSV.

**Open:** Mask → OPS.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Aligned mosaic and coordinates** — Align & Stitch destination: composed image and the tile-coordinate/layout records needed to interpret it.

**Outputs**

* **Optical barcode assignments** — OPS destination measurements.db: per-well geometry, phenotype alignment, nuclei and barcode tables; optional per-cycle reads.
  Relevant tables, depending on the route: ``ops_geometry``, ``ops_phenotype``, ``ops_objects``, ``ops_barcodes``, ``ops_reads``.

**Before this module**

* :ref:`Align & Stitch <workflow-module-align>`: Carry tile coordinates and original sequencing cycles into OPS; a flattened mosaic alone is insufficient.

**After this module**

* :ref:`Regression <workflow-module-regression>`: Join/aggregate decoded objects to phenotype and guide inputs explicitly before Regression; this is not a direct CSV handoff.

:doc:`API reference </api/spacr/ops_engine/index>`.

`Module tutorial <../../../tutorials/#lesson=76_ops>`__.

