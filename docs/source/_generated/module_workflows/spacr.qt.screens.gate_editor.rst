Workflow inputs and outputs
---------------------------

Gate Editor
~~~~~~~~~~~

Define threshold or polygon gates on actual feature/coordinate columns, then apply the saved gate to compatible objects.

**Open:** Home → Gate Editor.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Projection and clusters** — Image UMAP/PCA coordinate tables, selected clusters and figures for the loaded measurement data.

**Outputs**

* **Reusable gates** — Saved threshold/polygon gate definitions or a selected object set; apply a gate to the same feature definitions.

**Before this module**

* :ref:`Image UMAP <workflow-module-umap>`: Supply the matching coordinate/feature columns when defining a selection.
* :ref:`Measure <workflow-module-measure>`: Use the actual measured feature definitions and units.

**After this module**

* :ref:`Annotate <workflow-module-annotate>`: Apply compatible gates, then review candidate labels.

:doc:`API reference </api/spacr/qt/screens/gate_editor/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=64_gate_editor>`__.

