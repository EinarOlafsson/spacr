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

**Outputs**

* **Screen design and power estimates** — Saved planning tables or figures based on explicitly chosen effect sizes, variability and sampling assumptions.

:doc:`API reference </api/spacr/qt/screens/power/index>`.

