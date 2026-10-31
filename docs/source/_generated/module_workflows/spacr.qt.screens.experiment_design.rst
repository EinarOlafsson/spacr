Workflow inputs and outputs
---------------------------

Experiment Design
~~~~~~~~~~~~~~~~~

Enter conditions, controls, replicates and plate constraints, then export the experimental layout before acquisition.

**Open:** Home → Experiment Design.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Plate design entries** — Entered in Experiment Design: plate identifier, plate format (6 to 1536 wells), conditions with replicate counts and roles (treatment, positive control, negative control, blank), layout, edge use and seed. Load template… fills in a bundled design; no input file is required.

**Outputs**

* **Experimental layout** — Export plate map... writes plate_map.csv (one row per well keyed by plateID, rowID and columnID), plate_map.json and plate_map_settings.json to the chosen folder. Keep its plate and well identifiers consistent with the acquired data.

**After this module**

* :ref:`Dose–Response <workflow-module-dose_response>`: Acquire and measure the experiment first, preserving dose/condition identities.

:doc:`API reference </api/spacr/qt/screens/experiment_design/index>`.

`Module tutorial <../../../../../tutorials/#lesson=67_experiment_design>`__.

