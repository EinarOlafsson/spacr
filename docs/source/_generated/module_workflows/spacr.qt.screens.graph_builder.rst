Workflow inputs and outputs
---------------------------

Graph Builder
~~~~~~~~~~~~~

Load a table or use Merge tables to combine compatible measurements. For converted images, use Merge original filenames to recover names from a conversion map. Use Annotate conditions to assign a named column with metadata regex rules or dragged rows; save an annotated SQLite table or export CSV with its rules. Choose variables, groups and plotting settings, then export the figure with its analysis context.

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

`Module tutorial <../../../../../tutorials/#lesson=58_graph_builder>`__.

