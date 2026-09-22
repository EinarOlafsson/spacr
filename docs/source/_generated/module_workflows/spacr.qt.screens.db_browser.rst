Workflow inputs and outputs
---------------------------

Database Browser
~~~~~~~~~~~~~~~~

Browse selected tables and export rows; table inspection does not rerun image analysis.

**Open:** the application's Help/tools menus.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Inspect actual tables before exporting.

:doc:`API reference </api/spacr/qt/screens/db_browser/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=34_database>`__.

