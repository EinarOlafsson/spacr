Workflow inputs and outputs
---------------------------

Format Converter
~~~~~~~~~~~~~~~~

Convert supported microscope formats to a configured TIFF layout while preserving source mappings. Conversion does not generate object measurements.

**Open:** Import → Format Converter.

Inputs and outputs below include conditional alternatives. The guidance and handoff notes say which route applies.

**Inputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

**Outputs**

* **Microscope images** — Source image folder; original files, supported vendor files or imported TIFFs.

**After this module**

* :ref:`Mask <workflow-module-mask>`: Use the converted layout and preserve source identity mappings.

:doc:`API reference </api/spacr/convert/index>`.

`Module tutorial <https://einarolafsson.github.io/spacr/tutorials/#lesson=35_converter>`__.

