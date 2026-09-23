Workflow inputs and outputs
---------------------------

Plaque Assay
~~~~~~~~~~~~

Analyse plaque images or existing masks with the configured plaque model. This route need not pass through Measure; use the dedicated plaque example.

**Open:** Toxoplasma → Plaque Assay.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**Before this module**

* :ref:`Make Masks <workflow-module-make_masks>`: Use plaque masks with matching source images; cell masks are not automatically plaque labels.

:doc:`API reference </api/spacr/submodules/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=24_plaque>`__.

Recruitment
~~~~~~~~~~~

Use compartment intensity measurements and matching host/pathogen identities to compute recruitment ratios.

**Open:** Toxoplasma → Recruitment.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Require the intended compartment intensities and identities.

:doc:`API reference </api/spacr/submodules/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=25_recruitment>`__.

Invasion Assay
~~~~~~~~~~~~~~

Use the required two-colour differential-staining measurements and stain-baseline controls to distinguish attachment from invasion.

**Open:** Toxoplasma → Invasion Assay.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Require two-colour stain measurements and appropriate baseline controls.

:doc:`API reference </api/spacr/submodules/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=26_invasion>`__.

Replication Assay
~~~~~~~~~~~~~~~~~

Count parasites using explicit vacuole identity and compare condition distributions; host identity alone does not define a vacuole.

**Open:** Toxoplasma → Replication Assay.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.

**Outputs**

* **Assay results** — Assay-specific result tables and figures in the configured destination, preserving well and condition identities.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Require explicit parasite-to-vacuole identities.

:doc:`API reference </api/spacr/submodules/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=27_replication>`__.

Cellpose Workbench
~~~~~~~~~~~~~~~~~~

Open Cellpose Workbench inside Make Masks and train from verified image/mask pairs. Evaluate on separate fields before selecting the checkpoint in Mask.

**Open:** Make Masks → Cellpose Workbench.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Curated training fields** — Separate image and integer-mask files with matching field identities; preserve original images and labels.

**Outputs**

* **Segmentation checkpoint** — Saved Cellpose-compatible checkpoint or a compatible installed backend selected with its own configuration.

**Before this module**

* :ref:`Make Masks <workflow-module-make_masks>`: Use independently checked image/mask pairs.

**After this module**

* :ref:`Mask <workflow-module-mask>`: Select the saved compatible checkpoint in Mask.

:doc:`API reference </api/spacr/submodules/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=19_train_cellpose>`__.

