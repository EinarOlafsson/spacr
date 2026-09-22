Workflow inputs and outputs
---------------------------

Motility Assay
~~~~~~~~~~~~~~

Inspect track speed and infection-related measurements with an explicit frame interval and pixel calibration.

**Open:** Measure → Motility Assay.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Linked time-series objects** — Tracked labels and frame/object associations from the time-series project, with frame interval and units.
* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.
* **Figures and table exports** — The output location chosen by the tool; exports describe the selected data and filters.

**Before this module**

* :ref:`Timelapse <workflow-module-timelapse>`: Supply frame interval and pixel calibration.
* :ref:`Measure <workflow-module-measure>`: Combine measured objects with matching tracks.

:doc:`API reference </api/spacr/timelapse/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=18_motility>`__.

