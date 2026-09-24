Workflow inputs and outputs
---------------------------

AnnData Export
~~~~~~~~~~~~~~

Export an AnnData .h5ad file with object metadata and measured features for downstream single-cell analysis.

**Open:** Measure → AnnData Export.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **AnnData export** — An exported .h5ad file containing measured features and object metadata.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Export compatible feature and metadata columns.

:doc:`API reference </api/spacr/anndata_export/index>`.

`Module tutorial <../../../tutorials/#lesson=59_anndata_export>`__.

