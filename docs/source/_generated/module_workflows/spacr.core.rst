Workflow inputs and outputs
---------------------------

Mask
~~~~

Select channels and models, inspect a preview, then run Mask. Measure consumes the merged arrays; the counts database is not yet a feature table.

**Open:** Home → Mask.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Segmentation checkpoint** — Saved Cellpose-compatible checkpoint or a compatible installed backend selected with its own configuration.

**Outputs**

* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.
* **Object counts** — measurements/measurements.db; Mask counts alone are not per-object feature measurements.

**Before this module**

* :ref:`Cellpose Workbench <workflow-module-train_cellpose>`: Select the saved compatible checkpoint in Mask.
* :ref:`Import Images <workflow-module-import_images>`: Use imported image planes and identities; image-only imports still need segmentation.
* :ref:`Format Converter <workflow-module-convert>`: Use the converted layout and preserve source identity mappings.
* :ref:`Import <workflow-module-foreign>`: For image-only imports, use Import Images or Format Converter and point Mask at the formatted image project. External measurements alone are not segmentation input.

**After this module**

* :ref:`Measure <workflow-module-measure>`: Use the same project and the correct image/mask channel indices.
* :ref:`Timelapse <workflow-module-timelapse>`: Enable the nested time-series route before generating linked labels.

:doc:`API reference </api/spacr/core/index>`.

`Module tutorial <../../../tutorials/#lesson=07_mask>`__.

Image UMAP
~~~~~~~~~~

Project measured features or supplied encoder features and inspect representative crops. A cluster is a candidate grouping, not a validated phenotype. Use the lasso and annotation controls to write reviewed selections to an annotation column in the matching measurement database. A geometric selection alone does not establish a biological phenotype.

**Open:** Home → Image UMAP.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.
* **Image embeddings** — Object-indexed encoder features, with channel policy and encoder provenance. The encoding API returns features; persistence is caller-dependent.

**Outputs**

* **Projection and clusters** — Image UMAP/PCA coordinate tables, selected clusters and figures for the loaded measurement data.
* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Choose feature columns and inspect representative crops.
* :ref:`Embeddings <workflow-module-embeddings>`: Supply the encoder feature table with matching object IDs; do not assume every GUI route automatically persists it.

**After this module**

* :ref:`Gate Editor <workflow-module-gate_editor>`: Supply the matching coordinate/feature columns when defining a selection.
* :ref:`Classify <workflow-module-classify_merged>`: Write reviewed lasso selections to an annotation column in the matching object database, then select that column in Classify. Inspect crops and validate labels; embedding clusters are not ground truth.

:doc:`API reference </api/spacr/core/index>`.

`Module tutorial <../../../tutorials/#lesson=15_image_umap>`__.

Timelapse
~~~~~~~~~

Open Timelapse within Mask to segment and link objects across ordered frames; inspect links before downstream motility analysis.

**Open:** Mask → Timelapse.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

**Outputs**

* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.
* **Linked time-series objects** — Tracked labels and frame/object associations from the time-series project, with frame interval and units.

**Before this module**

* :ref:`Mask <workflow-module-mask>`: Enable the nested time-series route before generating linked labels.

**After this module**

* :ref:`Measure <workflow-module-measure>`: Use the time-series project with stable frame/object identities.
* :ref:`Motility Assay <workflow-module-motility>`: Supply frame interval and pixel calibration.

:doc:`API reference </api/spacr/core/index>`.

`Module tutorial <../../../tutorials/#lesson=17_timelapse>`__.

