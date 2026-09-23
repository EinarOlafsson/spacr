Workflow inputs and outputs
---------------------------

Small Multiples
~~~~~~~~~~~~~~~

Compare groups in small-multiple plots with explicit shared or independent axes.

**Open:** Graph Builder → Small Multiples.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

:doc:`API reference </api/spacr/qt/screens/trellis/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=63_small_multiples>`__.

