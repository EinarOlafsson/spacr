Workflow inputs and outputs
---------------------------

Barcode QC
~~~~~~~~~~

Inspect mapping depth, coverage, collisions and positional effects before guide-count filtering.

**Open:** Map Barcodes → Barcode QC.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Guide counts per well** — Map Barcodes run folder: unique_combinations.csv and annotated_reads.h5. Well identity requires the corresponding barcode references.
  Relevant columns, depending on the route: ``count``.

**Outputs**

* **Quality-control results** — Stored project checks and QC reports; a missing check is not a passing result.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Map Barcodes <workflow-module-map_barcodes>`: Check mapping and coverage before guide filtering.

:doc:`API reference </api/spacr/sequencing_qc/index>`.

`Module tutorial <../../../tutorials/#lesson=47_barcode_qc>`__.

