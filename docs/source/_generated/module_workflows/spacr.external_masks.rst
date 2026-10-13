Workflow inputs and outputs
---------------------------

External Masks
~~~~~~~~~~~~~~

Assign corresponding images and existing integer label masks, then create and measure a spaCR project without rerunning segmentation.

**Open:** Import → External Masks.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**Outputs**

* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.

**Before this module**

* :ref:`Direct Cellpose mask generation <workflow-module-cellpose_masks>`: Provide the saved label TIFFs and their original images to External Masks, assign object roles and create the merged project before Measure.

**After this module**

* :ref:`Measure <workflow-module-measure>`: Re-measure only when needed; External Masks can already perform measurement.
* :ref:`Annotate <workflow-module-annotate>`: Keep the newly measured project and crop index together.

:doc:`API reference </api/spacr/external_masks/index>`.

`Module tutorial <../../../tutorials/#lesson=31_external_masks>`__.

