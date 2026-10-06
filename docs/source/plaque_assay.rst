Plaque Assay: fields, figures and reviewed conditions
=======================================================

Open **Home → Assays → Toxoplasma → Plaque Assay**. Choose **Plaque** for
plaque fields or **Figure** for published figures containing wells, panels
and surrounding text. These inputs do not require a preceding Measure run.
See the :ref:`module map <workflow-module-analyze_plaques>`, the
`Plaque Assay tutorial <tutorials/#lesson=24_plaque>`_ and
:func:`spacr.submodules.analyze_plaques` for the surrounding workflow.

Preview a plaque field
-----------------------

#. Choose the source folder and select an image in the preview picker.
#. Select **Plaque** mode. In **Settings… → Plaque detection**, choose the
   plaque checkpoint, diameter, flow threshold and cell-probability threshold.
   Use **Model zoo…** to inspect available models or **Browse…** for a local
   checkpoint. A model key is different from a detector backend: both the
   checkpoint and a compatible runtime must be available.
#. Run the preview and inspect the labels against the image. Counts and mean
   areas describe the proposed segmentation, so check merged plaques,
   missed plaques and non-plaque regions before interpreting them.
#. Inspect the object, probability and flow views when the chosen segmenter
   supplies those outputs. Missing flow output is not a zero-valued result.
#. Choose **Use these settings** to copy the tuned values into the form that
   the analysis run reads. A preview alone does not run the folder analysis.

The preview resolves local checkpoints without downloading them implicitly.
When a model is absent, the panel explains what is missing and offers the
appropriate download. Keep the selected model and settings with the results;
``bundled`` names the historical packaged checkpoint, not an alias for the
current Model Zoo model.

Read a published figure
------------------------

#. Select **Figure** mode. Supply a folder of figures, or use
   **From a paper…** to retrieve figures and legends from a DOI, PMID,
   PMC identifier or PDF into a new folder.
#. If the figure reader is missing, use the panel's **Install** action.
   It installs the YOLO/OCR reader in its own backend environment under
   ``~/.spacr/backends``. Inspect the installation result before previewing.
#. In **Settings… → Figure**, choose the well detector, inference sizes and
   confidence cutoff; the text-reading settings are on the **Text detection**
   tab. **Run preview** finds wells and reads the figure text; this first
   pass does not segment the plaques.
#. Click a well in the image or a table row. **Plaque preview** segments that
   well with the plaque settings. **Find plaques in all wells** processes
   the detected wells in sequence; Cancel stops after the current well.
#. Compare the proposed condition with the nearby label and the relevant
   legend passage. Correct the condition, mark reviewed entries **OK**, and
   save the annotations. With **Confirm annotations** enabled, the run
   measures only the saved approved entries.
#. Copy the tuned settings into the form before starting the analysis run.
   Retain the original figure, legend, annotation review and calibration
   information alongside the measurements.

The preview saves condition reviews in ``figure_annotations.csv`` and pasted
legends in ``legends.csv`` in the source folder, preserving other figures'
entries. The batch figure workflow uses these files and writes its database
under ``<src>/plaque_figures/plaque_figures.db`` by default. Reprocessing a
figure replaces its previous rows; duplicate image content is recorded rather
than counted as an independent image. See
:func:`spacr.plaque_papers.measure_figure_folder` for the full file contract.

Areas in pixels and calibrated areas are different quantities. Verify the
reported scale and its source before comparing physical areas between images.
A detector box or an automatically read condition is a proposal to inspect,
not evidence that the experimental identity or calibration is correct.

For a figure crop, its own labeled scale bar takes priority, followed by its
own unlabeled bar whose length is stated in the legend. A crop without its
own bar can share an agreeing calibration from similarly sized crops in the
same grid. Conflicting peer bars leave that crop in pixels with a conflict
note. Whole-well calibration is a later fallback when the plate format is
known; stated magnification alone does not calibrate a rescaled figure.

Drop images, PDFs and folders
------------------------------

Drop any number of images, PDFs or folders onto Plaque Assay. Images are read
in **Plaque** mode and PDFs in **Figure** mode; every PDF is read in turn,
with progress naming the paper being read, and a paper that cannot be read is
reported by name without stopping the others. Files that are neither images
nor PDFs are left out and counted once. When the drop does not match the
current mode, a prompt offers to switch; when it holds both images and PDFs,
choose which mode to run. If the figure reader needs installing or
reinstalling, the prompt installs it in place and reports the result.

Bio-Rad Gel Doc images (``.scn``)
---------------------------------

Image Lab ``.scn`` files from a Gel Doc or ChemiDoc imager are read directly,
without exporting them first, wherever a TIFF is accepted: Plaque Assay in
both modes, Make Masks and its curation queue, and the Format Converter. A
folder of ``.scn`` files is a source like a folder of figures. Image Lab
stores these images with zero as white, so spaCR inverts them to look like
Image Lab's own PDF and TIFF exports (dark wells on light plastic) and keeps
the 12-bit values. Plaque Assay shows them as 8-bit grey scaled linearly to
the imager's ceiling. When the file records its physical field size, that
pixel size calibrates Figure-mode plaque areas (scale source
``image metadata``) unless a pixel scale is set in the settings or on a well.
In Python, :func:`spacr.convert.read_scn` returns the image and its metadata:
pixel size, imager, acquisition date, exposure and application.

Inspect the result views
-------------------------

The view selector above the preview offers **Overlay**, **Masks**, **Flows**
and **Cell probability** in both modes. In Figure mode, each segmented well's
outputs are placed at its box on the figure. Right-click the image for the
overlay options: outlines or filled objects, random colours, **Overlay
settings…** and **Save picture…**. The overlay settings include the outline
and fill colours, fill opacity and the **Line weight** of the **Well boxes**.
**Ruler** measures a distance on the image, in µm when the pixel size is
known and in pixels otherwise; right-click clears it.

Contribute training data
-------------------------

**Contribute training data…** sends an annotated image to spaCR's community
training data on Hugging Face. In Figure mode, box every well for the well
detector; in Plaque mode, paint every plaque for the next plaque model. An
image without annotations is not sent, and you must agree to the licence
before uploading. The dialog shows the destination dataset as a link that
can be opened, selected and copied. A contribution is reviewed before it is
used for training, so draw the annotations carefully.

Changing selections while a preview runs
----------------------------------------

Changing the source, selected image or mode abandons the previous preview.
Its late result cannot replace the newly selected view. An empty source folder
clears the prior mask, object views and figure tables. Start a new preview for
the intended selection after it loads. Cancellation discards the old result;
the current model call finishes before **Run preview** and the well controls
become available again. Wait for those controls before rerunning with changed
settings.

After choosing **Save annotations**, wait for the saved confirmation in the
preview status before starting the batch analysis. Saving happens in the
background so the image remains responsive; a queued save is not yet a
completed write.

Python preview contracts
-------------------------

The same operations are available as worker-safe functions; they do not touch
widgets. Their returned data is preview output, separate from the batch
analysis database.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Result and scope
   * - :func:`spacr.qt.widgets.plaque_preview.plaque_pass`
     - Segment one plaque image and return labels, counts, areas and available
       flow outputs.
   * - :func:`spacr.qt.widgets.plaque_preview.detect_figure`
     - Find figure regions and read text without segmenting plaques.
   * - :func:`spacr.qt.widgets.plaque_preview.prepare_figure_review`
     - Read saved review information and propose ruler calibration for the
       detected figure while preserving supplied manual edits.
   * - :func:`spacr.qt.widgets.plaque_preview.segment_well`
     - Segment a selected well crop with the plaque settings.
   * - :func:`spacr.qt.widgets.plaque_preview.figure_pass`
     - Find, read and segment a figure in one function call. This differs from
       the GUI's initial detection-only preview.

With the default segmenter, ``plaque_pass``, ``segment_well`` and
``figure_pass`` use :func:`spacr.plaque.segment_plaque_image`, including the
configured ``diameter``, ``flow_threshold``, ``CP_prob`` and channel-axis
policy. A custom ``segment`` callback supplies its own segmentation behavior.
Keep those settings explicit when comparing Python results with the GUI.

For GUI integrations, ``PlaquePreviewPanel.preview_running()`` remains true
while a cancelled worker is finishing. ``set_preview_busy(False)`` therefore
keeps rerun controls disabled until that worker exits. ``save_annotations()``
returns the queued destination; observe the preview status for completion or
failure before consuming the file.
