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

Optional scale and time estimates
----------------------------------

In Figure mode, enter any known **Pixels per µm** and formation time for
each well first. Choose **Estimate scale / time (experimental)** to show
suggestions in the **Estimated pixels per µm**, **Estimated time (hours)**
and **Estimate basis** columns. Review the basis alongside each suggestion.
The option is off by default and keeps suggestions separate from entered
calibration and measured results.

The calculation uses the largest quarter of plaques and assumes linear
diameter growth. Its default reference is an RH/HFF control diameter of
about 894 µm at seven days. This reference does not establish a growth curve
across times or conditions. When both scale and time are unknown, the
reference duration is assumed; it is not a separately measured time.
Conflicting known times prevent pooling wells into one page estimate.

Use **Experimental Growth Estimates** settings to choose a reference
diameter and its corresponding duration for your experiment. Keep the
reference and assumptions with exported suggestions. See
:func:`spacr.plaque_growth.estimate_page` for the input and output fields.

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
