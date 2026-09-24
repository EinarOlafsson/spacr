Workflow inputs and outputs
---------------------------

Map Barcodes
~~~~~~~~~~~~

Supply the references that encode your experiment. Inspect mapped counts and barcode QC before pairing well-level guide counts with phenotype scores.

**Open:** Home → Map Barcodes.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Sequencing reads** — Single-end or paired FASTQ files, with the matching barcode reference tables.

**Outputs**

* **Guide counts per well** — Map Barcodes run folder: unique_combinations.csv and annotated_reads.h5. Well identity requires the corresponding barcode references.
  Relevant columns, depending on the route: ``count``.

**After this module**

* :ref:`Regression <workflow-module-regression>`: Pair guide counts with phenotype scores using consistent plate/well keys.
* :ref:`Barcode QC <workflow-module-barcode_qc>`: Check mapping and coverage before guide filtering.

:doc:`API reference </api/spacr/sequencing/index>`.

`Module tutorial <../../../tutorials/#lesson=12_map_barcodes>`__.

