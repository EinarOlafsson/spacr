Workflow inputs and outputs
---------------------------

Import
~~~~~~

Import external measurements with explicit object/column mappings, or choose Import Images, Format Converter or External Masks. Imported measurements are not a fresh Measure run. With matching source images and external integer masks, Import builds the merged project arrays used by Measure. Choose the route matching your files and inspect object identities before measuring; imported feature tables can be used directly when they already contain the required measurements.

**Open:** Home → Import.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.
* **External measurements** — Third-party measurement CSV/database tables and their image, mask and object-identity mappings.

**Outputs**

* **Measured objects** — measurements/measurements.db; object tables depend on the enabled cell, nucleus, pathogen and organelle masks.
  Relevant tables, depending on the route: ``cell``, ``nucleus``, ``pathogen``, ``cytoplasm``.
  Relevant columns, depending on the route: ``plateID``, ``rowID``, ``columnID``, ``fieldID``.
* **Object crops** — data/\*\*/\*_png when save_png is enabled; png_list indexes saved crops. Supported workflows can instead stream crops from merged arrays and masks.
  Relevant tables, depending on the route: ``png_list``.
  Relevant columns, depending on the route: ``png_path``, ``prcfo``.
* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**After this module**

* :ref:`Annotate <workflow-module-annotate>`: Imported tables require explicit column and object-identity mappings.
* :ref:`Measure <workflow-module-measure>`: Import matching images and external integer masks to build merged project arrays, then open Measure on that project. Skip this step when compatible measurements have already been imported or computed. Do not append duplicate measurements to an existing imported table.
* :ref:`Mask <workflow-module-mask>`: For image-only imports, use Import Images or Format Converter and point Mask at the formatted image project. External measurements alone are not segmentation input.

:doc:`API reference </api/spacr/foreign/index>`.

`Module tutorial <../../../tutorials/#lesson=36_import>`__.

