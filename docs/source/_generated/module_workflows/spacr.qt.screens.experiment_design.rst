Workflow inputs and outputs
---------------------------

Experiment Design
~~~~~~~~~~~~~~~~~

Enter conditions, controls, replicates and plate constraints, then export the experimental layout before acquisition.

**Open:** Home → Experiment Design.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Screen design and power estimates** — Saved planning tables or figures based on explicitly chosen effect sizes, variability and sampling assumptions.

**Outputs**

* **Experimental layout** — Exported plate/condition/control/replicate map. Keep its plate and well identifiers consistent with the acquired data.

**Before this module**

* :ref:`Pooled-screen simulation sweep <workflow-module-simulation>`: Use simulated performance to reconsider sampling and plate constraints manually; the simulation database is not an importable plate layout.

**After this module**

* :ref:`Power / Design <workflow-module-power>`: Use the experimental layout to define sampling assumptions, then revise the design.
* :ref:`Dose–Response <workflow-module-dose_response>`: Acquire and measure the experiment first, preserving dose/condition identities.

:doc:`API reference </api/spacr/qt/screens/experiment_design/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=67_experiment_design>`__.

