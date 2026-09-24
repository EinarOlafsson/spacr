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

`Module tutorial <../../../tutorials/#lesson=24_plaque>`__.

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

`Module tutorial <../../../tutorials/#lesson=25_recruitment>`__.

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

`Module tutorial <../../../tutorials/#lesson=26_invasion>`__.

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

`Module tutorial <../../../tutorials/#lesson=27_replication>`__.

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
* :ref:`Direct Cellpose mask generation <workflow-module-cellpose_masks>`: Pass the trained checkpoint as custom_model with the matching image channels and preprocessing.

:doc:`API reference </api/spacr/submodules/index>`.

`Module tutorial <../../../tutorials/#lesson=19_train_cellpose>`__.

Endodyogeny size proxy
~~~~~~~~~~~~~~~~~~~~~~

Read measured compartment areas, annotate conditions and bin area ** 1.5 into log2 size doublings. Defaults aggregate pathogen area per host cell, not per vacuole: multiple vacuoles in one cell are combined. This is an area-derived size proxy, not measured volume or a parasite count. Use Replication Assay with explicit vacuole identity for parasites-per-vacuole counts. Configure compartment, area filters, calibration, conditions and grouping before calling the API; saving is optional.

**Use from Python:** :func:`spacr.submodules.analyze_endodyogeny`. This API-only workflow has no Home tile or menu entry.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Host-cell-aggregated compartment areas** — Each src project root/measurements/measurements.db; tables defaults to cell, nucleus, pathogen and cytoplasm, with png_list added for merging. The compartment setting selects the area column; default pathogen_area is summed per host cell.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``, ``png_list``.
  Relevant columns, depending on the route: ``cell_id``, ``pathogen_area``.

**Outputs**

* **Area-derived size-proxy results** — Returned data and chi_squared DataFrames; save=True also writes data.csv, chi_squared_results.csv, chi_squared_pairwise_results.csv and a figure under the first project root/results/analyze_endodyogeny/.
  Relevant columns, depending on the route: ``pathogen_area``, ``pathogen_volume``, ``pathogen_volume_bin``, ``bin_index``.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Supply the measured project roots and required object/png_list tables. Verify host-cell aggregation and area units before interpreting size bins; the Mask counts database alone is insufficient.

:doc:`API reference </api/spacr/submodules/index>`.

