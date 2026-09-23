Workflow inputs and outputs
---------------------------

Power / Design
~~~~~~~~~~~~~~

Estimate sampling needs from chosen effect sizes and variability; pilot measurements are optional evidence for assumptions.

**Open:** Home → Power / Design.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Screen design and power estimates** — Saved planning tables or figures based on explicitly chosen effect sizes, variability and sampling assumptions.
* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Experimental layout** — Exported plate/condition/control/replicate map. Keep its plate and well identifiers consistent with the acquired data.

**Outputs**

* **Screen design and power estimates** — Saved planning tables or figures based on explicitly chosen effect sizes, variability and sampling assumptions.

**Before this module**

* :ref:`Experiment Design <workflow-module-experiment_design>`: Use the experimental layout to define sampling assumptions, then revise the design.

**After this module**

* :ref:`Pooled-screen simulation sweep <workflow-module-simulation>`: Translate planning assumptions into the simulation settings dictionary manually; Power / Design does not export a ready-to-run simulation grid.

:doc:`API reference </api/spacr/qt/screens/power/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=68_power_design>`__.

