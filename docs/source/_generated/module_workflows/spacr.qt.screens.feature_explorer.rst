Workflow inputs and outputs
---------------------------

Feature Explorer
~~~~~~~~~~~~~~~~

Rank measured features for a chosen class comparison; review filtering and class definitions before interpreting the ranking.

**Open:** Classify → Feature Explorer.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Define the class comparison and inspect filtering.

:doc:`API reference </api/spacr/qt/screens/feature_explorer/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=65_feature_explorer>`__.

