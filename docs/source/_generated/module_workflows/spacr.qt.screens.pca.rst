Workflow inputs and outputs
---------------------------

PCA
~~~

Inspect principal components and feature loadings for the selected measured objects.

**Open:** Image UMAP → PCA.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Projection and clusters** — Image UMAP/PCA coordinate tables, selected clusters and figures for the loaded measurement data.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/pca/index>`.

`Module tutorial <../../../../../tutorials/#lesson=60_pca>`__.

