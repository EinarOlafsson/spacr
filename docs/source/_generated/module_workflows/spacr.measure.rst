Workflow inputs and outputs
---------------------------

Measure
~~~~~~~

Measure reads images and label planes together. Enable crop saving if you need PNG files; keep the database and source project together for streamed crops.

**Open:** Home → Measure.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**Outputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Before this module**

* :ref:`Mask <workflow-module-mask>`: Use the same project and the correct image/mask channel indices.
* :ref:`Make Masks <workflow-module-make_masks>`: Use FEATURES to pair images and masks and write a measured project; standalone masks are not merged arrays.
* :ref:`External Masks <workflow-module-external_masks>`: Re-measure only when needed; External Masks can already perform measurement.
* :ref:`Timelapse <workflow-module-timelapse>`: Use the time-series project with stable frame/object identities.
* :ref:`Import <workflow-module-foreign>`: Import matching images and external integer masks to build merged project arrays, then open Measure on that project. Skip this step when compatible measurements have already been imported or computed. Do not append duplicate measurements to an existing imported table.

**After this module**

* :ref:`Annotate <workflow-module-annotate>`: Save or stream crops with stable object identities.
* :ref:`Classify <workflow-module-classify_merged>`: Choose the image or tabular family to match your input.
* :ref:`Image UMAP <workflow-module-umap>`: Choose feature columns and inspect representative crops.
* :ref:`Embeddings <workflow-module-embeddings>`: Retain encoder and channel-policy provenance.
* :ref:`Gate Editor <workflow-module-gate_editor>`: Use the actual measured feature definitions and units.
* :ref:`Graph Builder <workflow-module-graph_builder>`: Choose explicit variables, groups and filters.
* :ref:`QC <workflow-module-qc_dashboard>`: Review available checks and missing evidence.
* :ref:`Recruitment <workflow-module-recruitment>`: Require the intended compartment intensities and identities.
* :ref:`Invasion Assay <workflow-module-invasion>`: Require two-colour stain measurements and appropriate baseline controls.
* :ref:`Replication Assay <workflow-module-replication>`: Require explicit parasite-to-vacuole identities.
* :ref:`Motility Assay <workflow-module-motility>`: Combine measured objects with matching tracks.
* :ref:`Feature Explorer <workflow-module-feature_explorer>`: Define the class comparison and inspect filtering.
* :ref:`Database Browser <workflow-module-db_browser>`: Inspect actual tables before exporting.
* :ref:`AnnData Export <workflow-module-anndata_export>`: Export compatible feature and metadata columns.
* :ref:`Dose–Response <workflow-module-dose_response>`: Join the measured response to explicit doses and controls.
* :ref:`Endodyogeny size proxy <workflow-module-endodyogeny>`: Supply the measured project roots and required object/png_list tables. Verify host-cell aggregation and area units before interpreting size bins; the Mask counts database alone is insufficient.

:doc:`API reference </api/spacr/measure/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=08_measure>`__.

