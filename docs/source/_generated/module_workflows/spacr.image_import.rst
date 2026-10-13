Workflow inputs and outputs
---------------------------

Import Images
~~~~~~~~~~~~~

Review field identities, image channels and optional external masks before writing a separate project. Image-only imports still need segmentation before measurement.

**Open:** Import → Import Images.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

**Outputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.
* **Images and label masks** — merged/\*.npy in the project; channels and integer label planes share each field array.
* **Label masks** — masks/ when retained, or explicitly saved image/mask pairs. Intermediate masks may be removed by cleanup.

**After this module**

* :ref:`Mask <workflow-module-mask>`: Use imported image planes and identities; image-only imports still need segmentation.

:doc:`API reference </api/spacr/image_import/index>`.

`Module tutorial <../../../tutorials/#lesson=74_import_images>`__.

