Workflow inputs and outputs
---------------------------

Image Scatter
~~~~~~~~~~~~~

Plot selected measurement columns with object-crop inspection.

**Open:** Image UMAP → Image Scatter.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Outputs**

* **Projection and clusters** — Image UMAP/PCA coordinate tables, selected clusters and figures for the loaded measurement data.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/image_scatter/index>`.

`Module tutorial <../../../../../tutorials/#lesson=55_image_scatter>`__.

