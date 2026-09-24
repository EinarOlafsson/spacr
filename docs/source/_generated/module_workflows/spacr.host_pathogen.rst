Workflow inputs and outputs
---------------------------

Host–Pathogen Analysis
~~~~~~~~~~~~~~~~~~~~~~

Measure whole vacuoles and host reference compartments, keeping uninfected host cells for the infection denominator. Select marker channels and control-calibrated ratio thresholds. Optionally count individually segmented parasites from explicit vacuole links or a measured count column. Review vacuole, host and well tables, joint marker states, replication distributions and unmatched parasite links; unknown measurements remain unknown.

**Open:** Toxoplasma → Host–Pathogen Analysis.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Keep uninfected cells in Measure. Supply whole-vacuole masks, host reference intensities and optional explicit parasite-to-vacuole links; host identity alone does not define a vacuole.

:doc:`API reference </api/spacr/host_pathogen/index>`.

`Module tutorial <../../../tutorials/#lesson=85_host_pathogen>`__.

