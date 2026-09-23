Workflow inputs and outputs
---------------------------

Annotate
~~~~~~~~

Label representative objects in png_list before supervised training. Use separate annotations for different questions and keep training and evaluation groups independent.

**Open:** Home → Annotate.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.
* **Reusable gates** — Saved threshold/polygon gate definitions or a selected object set; apply a gate to the same feature definitions.

**Outputs**

* **Training annotations** — A chosen annotation column in measurements/measurements.db, table png_list; labels belong to object identities.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``prcfo``.

**Before this module**

* :ref:`Measure <workflow-module-measure>`: Save or stream crops with stable object identities.
* :ref:`External Masks <workflow-module-external_masks>`: Keep the newly measured project and crop index together.
* :ref:`Import <workflow-module-foreign>`: Imported tables require explicit column and object-identity mappings.
* :ref:`Gate Editor <workflow-module-gate_editor>`: Apply compatible gates, then review candidate labels.

**After this module**

* :ref:`Classify <workflow-module-classify_merged>`: Keep labelled training and evaluation groups separate.
* :ref:`Annotator Agreement <workflow-module-agreement>`: Choose independent annotation columns for the same objects.

:doc:`API reference </api/spacr/qt/screens/annotate/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=09_annotate>`__.

