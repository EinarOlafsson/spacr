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

* **Converted TIFFs and conversion map** — Format Converter destination folder: Yokogawa-named TIFFs (for example plate1_A01_T0001F001L01A01Z01C01.tif) and conversion_map.csv, which maps each output name back to its source file.

**After this module**

* :ref:`Mask <workflow-module-mask>`: Use the converted layout and preserve source identity mappings.

:doc:`API reference </api/spacr/convert/index>`.

`Module tutorial <../../../tutorials/#lesson=35_converter>`__.

