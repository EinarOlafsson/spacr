Make Masks: editing, detection and measurement
==============================================

Open **Home → Tools → Make Masks** to inspect an image, correct its integer
label mask and save the result. Each positive label identifies one object;
zero is background. This screen can also propose objects with a detector,
grow cell masks from existing nucleus masks, and send curated image/mask
pairs to **Features** for measurement.

For the inputs and outputs of the surrounding workflow, see
:ref:`Make Masks in the module map <workflow-module-make_masks>` and the
`Make Masks tutorial <tutorials/#lesson=14_make_masks>`_. The
:doc:`screen API <api/spacr/qt/screens/make_masks/index>` documents the
implementation and :mod:`spacr.qt.mask_engine` documents the editing routines.

Open a field and save an edit
-----------------------------

#. Choose **Open folder…** and select the image folder. The ordinary layout
   keeps corresponding masks in its ``masks/`` subfolder. A saved TIFF mask
   uses the image's filename stem; a missing mask starts empty. The loader
   also supports the sibling masks layout and Cellpose ``_seg.npy`` bundles.
#. Inspect the image and labels. Images and masks must have matching spatial
   dimensions. An ordinary colour image is converted to grayscale; prepare
   the intended channel as a separate image when channel identity matters.
#. Correct an object with a tool, or choose a detection method and inspect its
   preview before accepting objects. **Undo** and **Redo** apply to mask edits.
#. Press **Save mask** or **Ctrl+S** before moving to another field. Saving
   writes a ``uint16`` label TIFF in the ordinary layout; a Cellpose bundle
   is updated as a bundle. The source image is preserved.
#. Use **Keep** or **Discard** to record a field-level curation verdict and
   advance. These verdicts go to ``csv/keep_discard.csv``; Discard records a
   decision without deleting the image or its mask. Save edits separately.

A saved edit history is stored beside the mask as ``<mask>.curation.json``.
Opening an image without editing it does not create evidence of manual
curation. Existing history is retained when editing a previously curated
mask. See :func:`spacr.qt.mask_engine.save_mask` and
:class:`spacr.curation.CurationLog` for the file contract.

Canvas tools and navigation
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Tool
     - Action
   * - Brush
     - Paint disks along the pointer path using the active label.
   * - Erase
     - Remove pixels under the brush.
   * - Erase object
     - Remove the complete label clicked.
   * - Wand + / Wand −
     - Flood from the clicked pixel within the chosen intensity tolerance;
       add or remove the resulting region.
   * - Draw
     - Trace an outline and fill its interior as one object.
   * - Divide
     - Draw a cut through a merged object. The larger component retains its
       ID; the smaller receives a new ID.
   * - Zoom
     - Drag a rectangle to change the view without changing labels.
   * - Recrop
     - Write a cropped image/mask pair as another field, then retire the
       original when leaving the field after recropping.
   * - Ruler
     - Drag a line to measure image-pixel distance. Right-click with Ruler
       selected to clear it. Zooming and panning preserve the measurement.

**Shift or Alt + drag** pans. The wheel zooms about the cursor; **Esc** resets
zoom. **Left/Right** selects the previous/next field. **Ctrl+Z/Ctrl+Y** undoes
or redoes an edit. **Ctrl + left click** splits an object at its waist;
**Ctrl + right click** removes the object under the cursor. The shortcut
panel beside the view lists the current gestures, including magnifier
controls.

Recrop creates files and changes the field queue. Objects cut by a crop's
boundary are omitted, and retained objects are renumbered within the new
crop. Very small crops and near-duplicates are refused. Completed originals
move into ``recropped_originals/``. Use the original image and mask when
maintaining primary/secondary IDs across the full field; a renumbered crop
is a different pairing context. The precise limits and paths are documented
by :func:`spacr.qt.mask_engine.write_recrop`.

Wand tolerance and leaking regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Wand's relative tolerance is a fraction of the image intensity range;
absolute tolerance uses the processed image's intensity units. The optional
runaway detector looks for sudden widening as the flood leaves the clicked
object. **Growth ratio**, **Warm-up**, **Min baseline** and **Confirmation**
control how much widening is required, where checking begins, the minimum
object width and how long widening must persist.

**Re-flood below the escape** searches for a lower tolerance whose region
stays inside the object. **Search steps** controls that search's precision.
**Taper onto the gradient** refines a provisional cut using the intensity
boundary; smoothing, transition-band width and foreground inset determine
where the edge may move. These controls help with a bright object connected
to another region by a weak bridge; inspect the resulting edge when the
bridge and object have similar intensity.

The Wand uses the currently applied image enhancement. If enhancement is
still being calculated, wait for its completion before trying the click
again. Post-detection morphology and splitting belong to detector output,
not to the manual Wand gesture.

Display, Levels and intensity units
-----------------------------------

**Lower %** and **Upper %** set the percentiles drawn as black and white.
**Levels…** opens a full-field histogram: drag either marker or enter a
black/white intensity cutoff. Changes update the percentile controls. A
constant-intensity image has no range to stretch. **Reset levels** uses the
full range, 0–100 percentiles.

With **Detect on the normalized image** off, these levels affect display.
With it on, detectors use the stretched image before any applied enhancement.
The normalization is computed for the whole field before extracting a
magnifier crop, so moving the magnifier does not redefine the percentile
levels. The setting changes future proposals; it does not modify existing
labels or rewrite the loaded image.

**Invert image** affects both the picture and detector input. It normalizes
the field to 0–1 and takes its complement, so an absolute detection threshold
must be interpreted on that scale. The pixel value in the corner readout
follows inversion and detection normalization; object mean intensity and
the **Filter** intensity bounds use the original loaded values. Changing
contrast or inversion therefore does not change the meaning of a measured
object's original mean intensity.

**Swap object and background**, under Object operations, changes the label
mask. Use it only when that label transformation is intended; it does not
perform image inversion.

Choose a detection method
-------------------------

The method selector controls the detector used by the live preview and the
corresponding whole-image detection action. Only applicable method controls
are shown. Model methods require their model or backend; the Model Zoo
indicates installation and download state.

The **Object detection** toolbar action runs model loading, inference and
postprocessing in a worker so the window remains responsive. It captures the
current input and settings when started. If you switch fields or change the
image or mask before it finishes, the result is discarded; run detection again
on the intended field. Closing the window does not wait for that result.
The Python method :meth:`spacr.qt.screens.make_masks.MakeMasksScreen.run_cellpose`
remains synchronous and returns zero if another detection is already running.

CPU Cellpose inference uses float32 weights through the shared device policy.
Supported GPU precision depends on the backend and device. Different precision
can produce different predictions, so inspect the masks instead of assuming
identical output across devices. This inference policy does not change training
precision. See :func:`spacr.accelerator.cellpose_kwargs`.

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Method
     - What to inspect and adjust
   * - Otsu
     - A global threshold for foreground/background populations. Adjust
       correction, blur, bright/dark foreground, hole filling, touching-object
       splitting, border exclusion and minimum area. Local Otsu is an
       additional option for whole-image detection.
   * - Li, Yen, Triangle, IsoData, Mean, Minimum
     - Alternative global levels with the shared threshold cleanup controls.
       Minimum can refuse a histogram without two separable peaks. An
       algorithm name alone does not establish accuracy on a new image.
   * - Multi-Otsu
     - Choose the number of intensity classes and the foreground band.
   * - Sauvola, Niblack
     - Local-window statistics with window size and contrast weight ``k``.
       Niblack uses ``mean - k*standard_deviation``. Sauvola uses a default
       ``R=1`` on float input without rescaling its range; check the detector
       input scale when transferring settings.
   * - Maxima + propagate
     - Find bright centres after Gaussian blur, then grow watershed basins
       using intensity and a stop rule. Centre spacing and seed level govern
       proposed seeds; a seed can disappear during subsequent filtering.
   * - Secondary objects from primary masks
     - Grow from every pixel of each labelled primary object while retaining
       its ID. See the pairing walkthrough below.
   * - Adaptive threshold
     - Local Gaussian-weighted threshold, offset and morphological cleanup
       through the irregular-organelle engine.
   * - LoG blobs, DoG blobs
     - Spot detection over Gaussian scales. Inspect scale range, response
       threshold and whether spots grow by watershed rather than disk stamps.
   * - Ridge filter
     - Frangi, Sato or Meijering response for network-like structures;
       select response scales and global or adaptive response threshold.
   * - Hysteresis
     - Grow from strong response through connected weaker response. Values
       below 1 are interpreted as percentile fractions by this engine.
   * - U-Net
     - Load a compatible ``.pt``/``.pth`` checkpoint and choose its sigmoid
       probability cutoff; optional skeletonization applies to the result.
   * - Cellpose and other installed backends
     - Use the corresponding model controls. Cellpose exposes diameter,
       flow-error threshold, cell-probability threshold and normalization.
       Inspect its cell-probability and flow panes alongside accepted labels.

Ordinary Otsu retains a legacy magnifier preprocessing path. Its cropped
preview and whole-field detection are not guaranteed to be pixel-identical.
Other threshold modes run the shared engine on the requested crop; a crop
can still have a different histogram from the whole field. Tune on several
representative fields and inspect the final whole-image output.

Enhancement before and after detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Compare** previews the configured enhancement; **Apply** enables it for
detection and display. The order is percentile stretch, background
subtraction, denoise, contrast, sharpen, detection, morphology, then split.
Disabled steps leave their input unchanged. Background estimation radius
should exceed the structures you want to retain. Denoising and the separate
Maxima/secondary blur can compound, so check both settings.

Post-detection opening/closing and splitting modify label shapes. These
steps are bypassed for paired secondary objects to retain primary IDs.
Make Masks enhancement is configured separately from the batch Mask
pipeline; transfer the intended preprocessing explicitly when training or
running a model elsewhere. See :mod:`spacr.qt.detect_chain`.

Grow secondary objects from a primary mask
------------------------------------------

#. Open the image channel in which the secondary objects should grow, such
   as a cell channel, then select **Secondary objects from primary masks**.
#. Choose different **Primary object class** and **Secondary object class**
   values, for example Nucleus and Cell. Custom class names can be entered.
#. Use **File…** for a primary label mask belonging to this field, or
   **Folder…** for masks matched to successive image filename stems. An
   explicitly selected file is bound to the current field; it is not reused
   silently when moving to another image.
#. Wait for the primary-object count. The primary mask must match the image's
   height and width and contain valid nonnegative integer IDs within
   ``uint16`` capacity. Its path must differ from the editable output mask,
   including aliases. Primary data are read-only.
#. Choose **Intensity watershed** or **Distance watershed** under Growth.
   Intensity follows the negative blurred image; Distance floods a flat
   surface outward from primary pixels. Both retain the primary labels.
#. Select a stop rule and inspect the preview. A global threshold is the
   screen's initial secondary setting. It is useful when primary objects,
   such as nuclei, are dark in the secondary channel. The separate Maxima
   mode starts with a fraction-of-peak rule.
#. Use whole-image **Replace** when starting a new paired output. Adding to
   an unrelated existing mask is refused even if some numeric IDs happen to
   match. Clear the existing objects or replace the whole image first.
#. Inspect relationship diagnostics, correct errors and save. A source
   changed after loading or an output path that would overwrite a primary
   source is refused; reload the source and inspect a new preview.

The four stop rules are **Fraction of the primary object's peak**,
**A threshold algorithm's level**, **Absolute intensity** and
**Percentile of the image**. Common threshold, absolute and
percentile rules constrain four-connected growth paths. Fraction-of-peak
trims each basin after growth; it does not constrain those paths during the
flood. The fraction is a ratio of intensity, not an image quantile. Neither
Growth option implements CellProfiler's Propagation algorithm.

**Matched** reports IDs present in both masks; **Missing secondary** identifies
primaries without a secondary label; **No primary** identifies secondary IDs
without a primary; **Primary not enclosed** identifies matched secondary
labels that do not fully contain their primary pixels. **Not expanded**
identifies matched labels that have not grown beyond their own primary.
These are relationship checks, not a
claim that the cell boundary is biologically correct. Do not use consecutive
relabeling to repair a pairing: it changes the association. In this mode,
clicking accepts objects and dragging does not merge distinct primary IDs.

See :func:`spacr.qt.mask_engine.secondary_object_instances`,
:class:`spacr.qt.secondary_masks.PrimaryMaskSource` and
:class:`spacr.qt.widgets.primary_mask_selector.PrimaryMaskSelector` for
label, source and diagnostic contracts.

Live magnifier, filtering and measurement
-----------------------------------------

Toggle **Magnifier** or press **M** to inspect proposed objects around the
pointer. Its wheel changes box zoom; **Shift + wheel** changes box size.
**Ctrl+L+right click** locks or unlocks the box. Wait for an updating preview
before accepting its objects.

**Objects added** selects every object in the zoom area or only objects
under the mouse. In ordinary modes, dragging can join encountered pieces
into one object. Whole-image preview accepts objects under the pointer;
secondary mode preserves primary identities instead of joining them.

The corner readout identifies the pixel and object under the cursor. Use
object area and original mean intensity to choose **Filter** bounds. A bound
of zero is disabled. Inspect the removal report and use Undo if a filter
removes wanted objects. Object operations also offer hole filling,
dilation, shrinking, relabeling and clearing; shrinking can remove thin or
small objects entirely.

Choose **Features** to pair image channels and mask classes in the measurement
input table. It runs the Measure workflow and produces its project folders
and measurements database. A folder of standalone TIFF masks is not itself
a Measure ``merged/`` dataset: the Features handoff supplies the pairing.
See :ref:`Measure inputs and outputs <workflow-module-measure>`.

The masthead also opens Cellpose Workbench, Mask the whole folder, Model
Compare, Model Zoo, Curate and Napari Bridge. Their input/output contracts
are linked from the :ref:`module map <workflow-module-make_masks>`.

Engine parameter reference
--------------------------

The following definitions are generated from the same parameter docstrings
as the API pages. They include programmatic field names for reproducing a
configuration in code. The GUI shows only the subset used by the selected
method. Secondary growth has its own controls, independent of Maxima +
propagate, and defaults to a global threshold in the screen.

.. include:: _generated/make_masks_parameters.rst
