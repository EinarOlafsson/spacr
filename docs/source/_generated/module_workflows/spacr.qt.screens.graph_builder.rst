Workflow inputs and outputs
---------------------------

Graph Builder
~~~~~~~~~~~~~

Choose variables, groups and plotting settings from the loaded table, then export the figure with its analysis context.

**Open:** Home → Graph Builder.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Choose explicit variables, groups and filters.

:doc:`API reference </api/spacr/qt/screens/graph_builder/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=58_graph_builder>`__.

