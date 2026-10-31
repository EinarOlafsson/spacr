Make Masks: editing, detection and measurement
==============================================

Open **Home → Tools → Make Masks** to inspect an image, correct its integer
label mask and save the result. Each positive label identifies one object;
zero is background. This screen can also propose objects with a detector,
grow cell masks from existing nucleus masks, and organize curated images
and masks for Measure with **Organize for Measure…**.

For the inputs and outputs of the surrounding workflow, see
:ref:`Make Masks in the module map <workflow-module-make_masks>` and the
`Make Masks tutorial <tutorials/#lesson=14_make_masks>`_. The
:doc:`screen API <api/spacr/qt/screens/make_masks/index>` documents the
implementation and :mod:`spacr.qt.mask_engine` documents the editing routines.

Hover a setting or its label to read the explanation and follow its API help
link. Controls with a matching animation also offer an animation in that
popup; each control remembers its own reveal state. Image-enhancement help
links to the detection-chain documentation. A control without a matching
animation still keeps its written explanation and API link.

Centre-pixel puncta inside cysts or cells
-----------------------------------------

Choose **Centre-pixel puncta within parent masks** in **Object detection**.
Open the original single-channel images and select a separate parent-mask
folder with matching image stems, or a parent-mask file for the current image.
Parent masks are read-only. Use whole-image **Replace**, then **Save mask**.

The detector finds scale-normalised LoG peaks at sigma 1.5, 2, 3, 4 and 6
pixels. It estimates each parent's pixel noise as the scaled median absolute
deviation (1.4826 × MAD) of the image minus its sigma-1 Gaussian blur, divided
by 0.87, and propagates that noise through each
LoG kernel. The candidate threshold defaults to 2.5 noise units. Peaks undergo
scale-dependent nonmaximum suppression and a three-pixel parent-edge exclusion.
Change the scale list, candidate threshold, peak spacing or edge margin when
the acquisition requires it; these are image-pixel settings, not calibration.

Each punctum is measured using the pixels nearest its intensity-weighted
subpixel centre in a nine-by-nine window. **Centre pixels per punctum**
defaults to 20 and accepts 1 to 81, the size of that window. The same-parent local-background
annulus starts at max(4, 2.5 sigma + 2) pixels and is four pixels wide. It
excludes other candidate centres when enough background remains. The default
inclusion floor is centre mean minus local background >= 3 native intensity
units. Display normalization, inversion, enhancement and Min area are bypassed.

The saved integer mask gives every retained punctum its own label. Nearby
centre windows can overlap: each integer peak is reserved for its object,
other shared pixels go to the nearest centre, and pixels outside the parent
are omitted. Consequently a mask may contain fewer than the requested N
pixels. This is a centre sampling mask, not the punctum's physical boundary.

Saving an unchanged whole-image detection also writes ``<stem>.puncta.csv``
and ``<stem>.puncta.json`` beside the mask. The table contains all candidates,
their inclusion flags, output object labels, parent IDs, coordinates, noise,
local background, exact 10/20/40-pixel centre means and the selected N-pixel
mean. Exact centre means retain overlapping samples independently of label
ownership. Use ``center_corrected`` for the selected N-pixel mean minus local
background, or ``corrected20`` for the fixed 20-pixel variant. The
``corrected`` column holds the three-by-three-window mean minus local
background.
The receipt binds the image, parent mask, output mask, CSV and detector
settings to checksums. Edited or combined labels cannot silently receive the
old centre measurements. Local previews estimate noise within their crop;
use whole-image detection for scientific results.

Use **Organize for Measure…** to place the puncta in an organelle mask slot
alongside the parent masks and original channels. Measure can report ordinary
mask-region intensities and parent relationships, but those means can differ
from exact centre means when samples overlap. Use the source-bound puncta
table for the centre/annulus analysis. This detector alone does not implement
the noise-null model, hierarchical tests or a composed publication figure.

Thumbnail display quality
-------------------------

Choose **Thumbnail quality** in **Display** or **Organize for Measure…**.
**Low** retains the original 64-pixel source sampling. **Medium** uses up to
256 pixels and **High** up to 1024 pixels on the longest side, capped by the
source resolution. These choices read the source image and mask again;
they do not enlarge a Low-quality bitmap. The separate **Size** control
sets the logical cell size.

The choice is remembered and updates open thumbnail views. Only visible
organizer cells are loaded, with a bounded cache. Changing quality affects
display only; saved images, masks and measurements are unchanged.

Open a field and save an edit
-----------------------------

#. Choose **Open File / Folder…** (or **Open folder…** in its arrow menu) and
   select the image folder. The ordinary layout
   keeps corresponding masks in its ``masks/`` subfolder. A saved TIFF mask
   uses the image's filename stem; a missing mask starts empty. The loader
   also supports the sibling masks layout and Cellpose ``_seg.npy`` bundles.
#. Inspect the image and labels. Images and masks must have matching spatial
   dimensions. An ordinary colour image is converted to grayscale; prepare
   the intended channel as a separate image when channel identity matters.
#. Correct an object with a tool, or choose a detection method and inspect its
   preview before accepting objects. **Undo** and **Redo** apply to mask edits.
#. Press **Save mask** or **Ctrl+S** before moving to another field. Saving
   writes a label TIFF in the ordinary layout. Ordinary segmentation uses
   ``uint16``; existing ``uint32`` masks and exact IDs requiring that width
   retain ``uint32`` storage. A Cellpose bundle is updated as a bundle.
   The source image is preserved.
#. Use **Keep** or **Discard** to record a field-level curation verdict and
   advance. These verdicts go to ``csv/keep_discard.csv``; Discard records a
   decision without deleting the image or its mask. Save edits separately.

**Save mask** and the previous/next-image buttons are in the navigation group
below the image. Editing and annotation controls wrap in the top bar as the
window narrows. When the wrapped top controls need more height than available,
scroll vertically within that bar to reach them.

**Save on navigation**, in the **Save mask** arrow menu, saves an edited mask before
**Keep**, **Discard**, **Next** or **Previous** moves on. It is
off by default and remembered between sessions. If the save fails, the image
stays open and no verdict is recorded. **Skip** never writes a mask.

To try the screen without your own data, **Load test data…** downloads ten
unsegmented Toxoplasma vacuole fields, with their curated masks kept apart
in ``ground_truth_masks/``, and opens the first. The arrow on the same
button offers a sample of fields from the dataset each published model was
trained on, the electron microscopy test data and the FIB-SEM volumes
described under :ref:`mm-tem` and :ref:`mm-volumes`, and three timelapse
examples (yeast, bacteria and mammalian cells) opened with their masks and
lineage.

.. _mm-volumes:

3D volumes in the Mask view
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A volume opens in the Mask view, where 2-D fields open. A TIFF stack, an
OME-TIFF or an MRC volume (``.mrc``, ``.rec`` or ``.map``) with three spatial
axes is recognised as a volume when it is in the folder you open or when you
open it with **Open file (2D or 3D)…** in the arrow menu of
**Open File / Folder…**. Time, channel and RGB stacks
are refused rather than read as depth. An MRC volume needs the optional
``mrcfile`` package. The status line reports the number of slices and the
voxel size.

* The **slice slider** at the left of the image steps through the slices;
  **Page Up** and **Page Down** step too. Every 2-D tool draws on the slice
  on screen.
* The **Z** button above the slider is the Z wheel toggle. On (blue), the
  mouse wheel changes the slice; off, the wheel zooms.
* The **View** list beside the Mask tab chooses **Default**, **Orthogonal**
  or **True 3D**. Default draws on one slice at a time. Orthogonal adds the XZ
  and YZ side views through a crosshair; the middle mouse button moves the
  crosshair. True 3D shows the volume and its objects in three dimensions.
* The row above the image holds **Object ID** (an existing object to edit,
  or an unused ID for a new one), **Interpolate slices**, **Assign 3D
  objects** and **Delete object**.

**Interpolate slices** fills the empty slices between slices where the
chosen object is drawn, blending its outline from one to the next. **Assign
3D objects** gives every separate 3D object its own ID: parts of one ID that
do not touch in 3D become new objects, so you can draw many objects with one
ID and then assign them. **Delete object** removes the chosen object from
every slice. Each of these is one undo step.

In True 3D, draw an object, then drag its anchors or select surfaces
(Ctrl-click for several) and drag them or turn the mouse wheel. **Anchor
density** sets how many anchors each edge and surface gets, and **Show
anchors** hides them. The view is a sampled point-cloud preview; the edits
and voxel counts use the full image. These keys work while True 3D is
shown, and are listed under **Change shortcuts**:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Action
   * - **Shift+E**
     - Edit: drag anchors and surfaces.
   * - **Shift+S**
     - Spin the view freely.
   * - **Shift+D**
     - Draw a new object.
   * - **Shift+X**, **Shift+Y**, **Shift+Z**
     - Choose the X, Y or Z drawing plane. Pressed within 1.5 s of
       **Shift+S**, they choose the spin axis instead.

Edits change the mask voxels themselves, keep the image's axis order and voxel
spacing, and never overwrite another object. **Save mask** or **Ctrl+S**
writes the labels to ``masks/<image stem>.tif`` and their editable boundaries
to a ``.volume.json`` file beside it; the source volume is never overwritten.

.. _mm-tem:

Electron micrographs
~~~~~~~~~~~~~~~~~~~~

Make Masks opens transmission electron micrographs as ``.dm3``, ``.dm4``,
``.mrc`` and ``.rec`` files, with TIFF, PNG and the other image formats. The
pixel size in nanometres is read from the file. MRC files need the optional
``mrcfile`` package.

**Contrast preset**, under **Lower %** and **Upper %** in the display
settings, sets the display in one step. **Electron microscopy (TEM)** windows
the image between its 0.5th and 99.5th percentiles, so holes in the section
and stain precipitate cannot take the whole range, and turns on **CLAHE**
(128 px tiles) under **Image enhancement** to even out uneven section
thickness. Membranes stay dark. **Electron microscopy, inverted** also turns
on **Invert image**, so dark membranes and organelles are drawn bright and
Otsu and the magnifier find them. **Custom** changes nothing, and every value
stays editable.

The arrow on **Load test data…** offers **Mitochondria in TEM images (test
data)**: ten transmission electron micrographs (about 32 MB, EMPIAR-10982,
CC0), with their mitochondria masks in ``ground_truth_masks/``, which the
editor does not read. **FIB-SEM volume, 3D (test data)** downloads a FIB-SEM
volume of a HeLa cell (about 25 MB, EMPIAR-10982, CC0) and opens it
unsegmented in the 3D editor. **FIB-SEM volume with masks, 3D (test data)**
opens the same volume with its mitochondria masks, copied once to
``corrected_masks/`` for correction; ``ground_truth_masks/`` stays
unchanged. All are cached after the first download.

With **Show alpha features** on in Preferences, the magnifier's **Method**
list offers **MitoNet (TEM mitochondria)**. MitoNet v1 mini finds
mitochondria in electron micrographs with dark membranes; it is not a
fluorescence model. The weights are third-party (CC-BY-4.0, Conrad and
Narayan, Cell Systems 2023), listed under Third-party in the Model Zoo, and
download from Zenodo (110 MB) the first time the method runs. **Minimum
area** sets the smallest mitochondrion kept. This method is alpha.

Draw bounding boxes for YOLO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Open the source-image folder and choose **Box** beside **Draw**, or press
   **X**. Choose a class, or use **Add class** to name another class.
#. Drag across the image to draw a box. Drag inside a box to move it, drag
   a corner to resize it, and right-click a box to delete it. Hold **Ctrl**
   while dragging to add an overlapping or contained box.
#. Select a box to change its class. **Undo** and **Redo** apply to box edits
   while Box is selected. Use **Save boxes**, or **Ctrl+S** in Box mode, to
   save editable annotations. Moving to another image or closing saves
   changed boxes first; a failed save keeps the current image open.
#. Choose **Export YOLO labels** and save the image's ``.txt`` label file.
   The export folder also receives ``.classes.json`` with the class-name
   mapping. A reviewed image without boxes exports an empty label file.

YOLO label rows contain the class ID followed by centre X, centre Y, width
and height, normalized to the full source image. Pair each label file with
its original image when preparing a training dataset. Make Masks stores
editable boxes in ``.spacr_yolo_annotations.json`` beside the source images;
these annotations retain their image dimensions and source identity.

Finish **Recrop** before adding boxes. For Cellpose ``_seg.npy`` bundles,
open or export a standalone image first. Named YOLO export becomes available
after unblinding a blinded session. Boxes have their own editing history
and leave image pixels and segmentation masks unchanged. Dataset splitting
and YOLO model training are separate steps.

Drop images and folders
~~~~~~~~~~~~~~~~~~~~~~~

You can also drop files and folders onto the screen. Make Masks works out
what the drop is:

* images are queued in the order dropped; one folder of images opens as a
  folder;
* images dropped together with their masks open with those masks. A
  ``masks`` folder is used as it is; other mask files are copied, after a
  question, to ``masks/<image stem>.tif`` beside their images, keeping any
  mask already there;
* images kept in subfolders prompt an offer to consolidate them: their
  images are copied into one folder, each renamed with its folder path so no
  name is lost, and a ``rename_manifest.csv`` records the original names;
* subfolders named like channels (DAPI, GFP, ``ch1``, ``C01``…), or several
  image folders, open **Organize for Measure** with a channel column per
  folder. Cancelling it opens the drop as an ordinary folder;
* a spaCR output folder, such as ``merged/`` or ``sorted_channels``, opens
  its images.

Anything not used is listed in the console with the reason.

A saved edit history is stored beside the mask as ``<mask>.curation.json``.
Opening an image without editing it does not create evidence of manual
curation. Existing history is retained when editing a previously curated
mask. See :func:`spacr.qt.mask_engine.save_mask` and
:class:`spacr.curation.CurationLog` for the file contract.

Ordinary segmentation stores at most 65,535 positive labels. Saving refuses a
binary mask with more connected objects; it does not wrap oversized IDs into
smaller numbers. Existing ``uint32`` masks and the exact-ID path retain that
wider label storage when needed.
A refused save preserves an existing TIFF or Cellpose bundle and its metadata.
In ordinary mode, a mask with one foreground value is interpreted as binary
and its connected components receive separate IDs. Deliberate groups created
with Divide / Merge or a magnifier drag retain their shared IDs when saved
and reopened with their matching curation history. For primary/secondary
relationships, use the exact-ID path described below. See
:func:`spacr.qt.mask_engine.canonical_labels` for these distinct conventions.

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
   * - Wand
     - Flood from the clicked pixel within the chosen intensity tolerance;
       add the resulting region by default. Hold Ctrl while clicking to
       remove that region.
   * - Draw
     - Trace an outline and fill its interior as one object.
   * - Divide / Merge
     - Left-drag a cut through an object to divide it. The larger component
       retains its ID; the smaller receives a new ID. Right-drag a line across
       objects to merge them under the first crossed ID without painting the
       background between them.
   * - Zoom
     - Drag a rectangle to change the view without changing labels.
   * - Recrop
     - Write a cropped image/mask pair as another field, then retire the
       original when leaving the field after recropping.
   * - Ruler
     - Drag a line to measure image-pixel distance. Right-click with Ruler
       selected to clear it. Zooming and panning preserve the measurement.
   * - Pick
     - Shown while mask layers are loaded: remove, restore or add objects of
       the curated overlay; see `Curate against mask layers`_.

**Alt + drag** pans; with tools other than Brush, **Shift + drag** also pans
(with Brush it keeps painting the last object across a gap). The wheel zooms about the cursor; **Esc** resets
zoom. **Left/Right** selects the previous/next field. **Ctrl+Z/Ctrl+Y** undoes
or redoes an edit. With Wand selected, **Ctrl + left click** removes the
bounded intensity region. Outside Wand, **Ctrl + left click** splits an
object at its waist; **Ctrl + right click** removes the object under the
cursor. The shortcut panel beside the view lists the current gestures,
including magnifier controls.

**Clear all objects**, under **Object operations**, asks for confirmation
before removing every segmentation object in the current image. Cancelling
keeps the masks; undo restores them. The acquired image is unchanged.

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

The **Object detection** button under **Object operations** runs model loading, inference and
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
Live Cellpose previews also use float32 when explicitly forced to CPU or when
accelerator detection falls back to CPU. A Cellpose 3 checkpoint rejected by
Cellpose 4 produces a preview-specific compatibility message; choose a
checkpoint supported by the installed preview runtime. See
:func:`spacr.qt.widgets.preview_contract.preview_cellpose_model`.

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

Ordinary Otsu uses a separate magnifier preprocessing path, so its cropped
preview and whole-field detection are not guaranteed to be pixel-identical.
Other threshold modes run the shared engine on the requested crop; a crop
can still have a different histogram from the whole field. Tune on several
representative fields and inspect the final whole-image output.

Enhancement before and after detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Compare** previews the configured enhancement; **Apply** enables it for
detection and display. The order is percentile stretch, background
subtraction, optional point-spread processing, optional restoration, denoise,
contrast, sharpen,
detection, morphology, then split.
Disabled steps leave their input unchanged. Background estimation radius
should exceed the structures you want to retain. Denoising and the separate
Maxima/secondary blur can compound, so check both settings.

Compare opens with a pending result while enhancement runs. Wait for the
right-hand image before judging the effect. Normalization uses the whole
field before extracting the displayed crop. If you change the field or
settings, open a new comparison for that selection. **Cancel** closes the
comparison; an enhancement already running finishes in the background and
its abandoned result is discarded.

Post-detection opening/closing and splitting modify label shapes. These
steps are bypassed for paired secondary objects to retain primary IDs.
Mask generation applies the same image steps through its ``enhance_*``
settings, which are set separately from the Make Masks controls; morphology
and splitting apply only in Make Masks. Set matching preprocessing when
training or running a model elsewhere. See :mod:`spacr.qt.detect_chain`.

For calibrated optical processing, choose **Convolve (blur)** or
**Deconvolve (Richardson–Lucy)** under **Point spread function**. Set the
image pixel spacing in micrometres, directly or from **Objective** and
**Camera**, then choose a measured TIFF/NPY kernel with matching spacing or a Gaussian approximation with explicit
Y/X full widths at half maximum. Use **Reload** after changing a kernel file.
Compare the result before applying it; more deconvolution iterations can
amplify noise. The original image intensities stay available for measurement.
See :doc:`point_spread` for the complete workflow and Measure settings.

Restore an image with Cellpose 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under **Image enhancement**, choose **Denoise**, **Deblur**, or
**One-click restoration** in **Deep image enhancement**. The default is
**Off**. If needed, select **Install Cellpose 3…** to install its separate
backend environment; this does not replace spaCR's own Cellpose installation.
First use may download the selected model's weights.

Open **Restoration model settings** and choose **Cells (cyto3)**,
**Cells (cyto2)**, or **Nuclei**, according to the structures in the selected
intensity channel. Set their approximate diameter in pixels. This controls
model rescaling, not microscope calibration. **Load / retry model** reloads
the current choice. Wait for the ready message before comparing or applying
it; that message identifies the device used by the isolated backend.

Loading and restoration run in the background. A whole field takes longer
than a magnifier region; timing depends on the image size, model and
hardware. A backend installed with supported GPU
acceleration can use it, but the workflow also runs on a CPU.

Use **Compare** to inspect several representative regions before choosing
**Apply**. Restoration follows background and PSF processing and precedes
classical denoising and contrast adjustments. Apply feeds the restored
floating-point intensities to detection and the Wand while keeping display
scaling separate. The image dimensions and loaded source pixels stay
unchanged; upsampling models are not offered here.

Restored values are normalized model output, not calibrated fluorescence;
they can be negative or exceed one. Measurements retain the original
intensities. A clearer-looking image does not establish recovered structure
or more accurate masks, and these weights are not validated for every
organelle or acquisition. Inspect the resulting masks as well as the image.
Keep the saved enhancement record with the masks: it identifies the model,
Cellpose version, checkpoint hash, device, diameter and processing settings.
Make Masks does not offer CARE or Noise2Void restoration.

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
before manually accepting its objects.

In the **Magnification settings** category, turn on
**Instantly accept proposed objects** to accept a current proposal without
clicking. This is off by default. Region mode follows **Objects added**;
whole-image mode accepts only the object under the pointer. Moving outside
the image or disabling the magnifier prevents automatic acceptance. Each
acceptance is undoable, and an undo does not immediately accept the same
proposal again. Use **Save mask** (or **Save on navigation**) to write
accepted masks.

**Overlap** chooses what happens when a proposal overlaps an existing mask:

* **Fuse with existing object** adds the proposal to the existing ID and
  retains the old object's pixels, including pixels outside the box. A
  proposal overlapping several objects joins them into the smallest old ID.
  A proposal on background becomes a new object.
* **Add non-overlapping pixels** preserves existing objects and adds the
  proposal's largest remaining connected piece as a new object.
* **Replace overlapping object** removes overlapping old objects in full
  before adding the proposal. Rejected small proposals remove nothing.
* **Skip overlapping objects** ignores proposals that overlap existing masks.

Secondary mode preserves primary IDs and does not allow fusing identities.
Explicit click-and-drag editing retains its single undo step.

**Objects added** selects every object in the zoom area or only objects
under the mouse. In ordinary modes, including Otsu and Cellpose, click and
drag to add encountered detections to the object under the initial click.
Starting on background creates one new object. Separate detections share
that object's ID without filling the background between them; one undo step
reverses the drag, and the shared ID survives Save, Next and reopening.
Whole-image preview accepts objects under the pointer;
secondary mode preserves primary identities instead of joining them.

The corner readout identifies the pixel and object under the cursor. The
**Filter** category starts empty: choose a property and **Add a filter** to
add a row with a minimum and a maximum. Every scalar
:func:`skimage.measure.regionprops` property is offered, such as area,
eccentricity or solidity; intensity statistics are offered only while an
intensity image is open and use the original loaded values. A blank bound is
off. The list applies as you edit it and when a field opens; the log below it
names each hidden object and the bound that hid it, and removing a row brings
back what it hid. The same list is the ``object_filters`` setting of Mask
generation. See :func:`spacr.qt.mask_engine.filter_properties`. Object
operations also offer hole filling,
dilation, shrinking, relabeling and clearing; shrinking can remove thin or
small objects entirely.

Organize images and masks for Measure
-------------------------------------

A folder of images and standalone TIFF masks is not itself a Measure
dataset. **Organize for Measure…** arranges it into the layout Measure reads.
Fill its table in any of three ways, each on its own:

* give a **Source folder** and choose a filename convention in the **Regex**
  box — the same conventions as Mask generation, with a custom regex — then
  press **Sort by regex**. **Auto regex** proposes one and **Detect sets**
  pairs images into fields and shows example sets to confirm;
* drop files or folders into the table's columns, one column per channel.
  Dropped files sort by name and are matched across columns into rows, one
  row per field;
* choose **Teach me…** and answer "Which channel is this?" for one image at
  a time. spaCR learns a regex from the answers and asks again only about a
  name it cannot yet read.

When the source folder keeps its images in subfolders, tick **Consolidate
subfolders into filenames first**. Sorting then copies the images into one
new folder, each named after its folders, and reads that copy; the originals
are not touched. If any image cannot be copied, for example because the disk
is full or a file is not readable, the popup shows **Consolidation failed**
with the number of files not copied and the path of the copy's
``rename_manifest.csv``. Sorting, **Auto regex**, **Teach me…** and
**Detect sets** stop at that point. The source folder, the consolidation
choice and the table keep their previous contents, and successful copies
remain in place. Fix the cause and run the step again; the retry makes a
complete new copy that includes each image once.

**Add channel** and **Add mask** add columns; each mask column names its
object class (cell, nucleus, pathogen or organelle) and the channel whose
images it outlines. A consolidated folder is read under its files' original
names. Rows missing an image block **Apply** and say how to fix them; RGB
images and z-stacks are offered for conversion to one grey plane, keeping
the originals. Nothing moves before **Apply**. The images are then moved
into ``sorted_channels/`` inside the source folder, with a folder per
channel, the masks and ``merged/``, and every move is recorded in
``channel_sorting_manifest.csv``. Point Measure's ``src`` at that
``sorted_channels`` folder. See :func:`spacr.channel_sorting.build_plan`
and :ref:`Measure inputs and outputs <workflow-module-measure>`.

Organize images in Mask Generation and Import Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same popup, titled **Organize images**, also opens from two other
modules. It fills its table the same ways: by regex, by dropping files and
folders into the columns, or with **Teach me…**.

* In **Mask Generation**, **Organize images…** above ``src`` arranges
  intensity images, from any folder structure or naming, into the one folder
  of Yokogawa-named images that Mask Generation reads. It has channel columns
  only. After the move, ``src`` points at that folder, ``metadata_type`` is
  set to ``cellvoyager``, ``channels`` lists the organized channels, and the
  folder is added to the module's recent sources.
* In **Import Images**, **Organize images and masks…** beside the images folder's **Choose…**
  takes intensity images and masks. **Write for** chooses the layout:
  **Mask Generation** writes one folder of Yokogawa-named images with the
  masks in its ``masks/`` folder, and **Measure** writes the channel folders,
  the masks and ``merged/`` as Organize for Measure does. The organized
  folder is then first in that module's recent sources.

A file that holds several channels is split rather than refused: choose
**Split multi-channel files**, or accept the offer when you press **Apply**.
Each plane is copied into ``split_channels/`` with one channel column per
plane, and the original files are not touched. Files move in the
background, and the manifest records every move. If a move fails, the
message names the error, and the manifest lists every move made before it.

Curate against mask layers
--------------------------

Mask layers compare any number of mask sets for the same images, such as the
curated training labels and the masks two models made, and build one curated
set from them. A layer is a folder of label masks matched to the images by
stem, like ``masks/``. Add one with **Add mask layer…** in the arrow menu of
**Open File / Folder…**, or start the session from a terminal:

.. code-block:: bash

   spacr-make-masks --folder /data/plaques --layers masks model_a model_b

A layer name that is not a folder where the command runs is looked for in
``--folder``. The **Mask layers** settings category lists the layers. Each
is drawn as outlines in its own colour; click a layer's swatch to change the
colour, or clear its check box to hide it. The choices are kept in
``curate_layers.json`` in the folder, so the next session of the folder
opens with them.

The editable **curated overlay** starts as a copy of the **Base layer**, the
first layer unless another is chosen. With **Show only flagged** off, the
**Pick** tool edits it as follows:

* Click an object of the curated overlay to remove it. A removed object of
  the base layer is drawn dashed; click it again to restore it.
* Click an object of another visible layer to add it under a new, unused
  ID. An added object fills only background: where it overlaps an object of
  the overlay, the overlay keeps its pixels. When two visible layers have an
  object under the click, the first in the list is added.
* Hidden layers are not picked from. To add another layer's object where a
  removed base object lies, hide the base layer first.

With **Show only flagged** on, picking a visible flagged object removes the
curated objects it overlaps; picking it again restores them. A flagged object
over background can instead be added. Hidden curated objects are not picked.

Brush, erase and the other tools edit the same overlay, and **Ctrl+Z** and
**Ctrl+Y** undo and redo every click.

**Save to** chooses where the overlay is written. The default is
**New mask folder**, ``masks_curated_<date>`` beside the images (an existing
``masks_curated_*`` folder is reused, so a session resumes into it); type
another name or a full path under **Folder name**, or pass ``--save-to``.
Choosing one of the layer folders overwrites its masks field by field, and
takes effect only after **Overwrite** in the confirmation shown under the
control. A field already saved to the save folder opens from there.

**Keep** saves the overlay as edited, with its object IDs unchanged, and
moves to the next field; **Next** behaves as in any queue. Edited saves write
curation history (``<stem>.tif.curation.json``). Layer decisions are recorded in
``<stem>.decisions.json`` beside the mask: the base IDs that were removed and
each added object's ID, source layer and ID in that layer. Progress is kept
in the ``curate_status.csv`` inside the save folder, so a review resumes on
its own record and offers again the fields the folder's ordinary curation
already finished. Without layers, Make Masks behaves and saves exactly as
before.

Upload data
-----------

**Upload data…**, in the **Save mask** arrow menu, sends the image on screen
with its mask,
every curated image in the folder with its saved mask, or a chosen images
folder and masks folder, to spaCR's community datasets on Hugging Face for
training future models. Every image needs a saved mask of the same name and
size. Name the dataset, add notes, and agree to the licence; the dialog shows
the destination as a link you can open, select and copy. A new dataset
appears with its first upload, and each contribution is reviewed before it
is used.

The masthead also opens Cellpose Workbench, Mask folder, Model Compare,
Model Zoo, Curate (alpha) and Napari Bridge (alpha). Their input/output contracts
are linked from the :ref:`module map <workflow-module-make_masks>`.

Large plates in Mask generation
-------------------------------

The **Mask** batch-generation module processes a plate; **Make Masks** edits
the field currently open. For a crash while processing a raw plate, check
the batch module's console and saved ``settings/gen_mask_settings.csv``
to identify the stage and settings used.

In the V1 batch pipeline, raw-file preprocessing projects and writes one
field at a time to ``stack/``. It retains filenames across the plate, not
every field's pixel data. Z planes are combined incrementally. Completed
field stacks are published atomically and can be reused when preprocessing
resumes. This stage's pixel memory therefore follows the size and channels
of a field, rather than the number of fields in the plate.

Normalization and segmentation still need working memory for a batch and
the model. Completed segmentation input, mask and flow buffers are released
before the next batch; reducing plate-wide retention does not make an
individual very large image or model cost-free. The V1 ``batch_size`` also
sets the normalization pool, so changing it can change normalized values
and subsequent masks. Retain it when reproducing a previous analysis.

If memory still rises unexpectedly, record whether the last console stage
was preprocessing filenames, normalization or model evaluation, together
with image dimensions, channel count and the saved settings. Existing
``stack/`` files help distinguish a completed ingestion stage from a failure
before the first field was written.

Calibrated spectral live preview (alpha)
----------------------------------------

The optional calibrated route supports organized two-dimensional channel-last
``stack/*.npy`` fields in Segmentation Live Preview. Enable alpha features and
spectral unmixing, supply the existing single-stain control wells, and set
``unmix_calibration`` to your measured calibration JSON. Set
``unmix_pixel_size_um`` explicitly to the source's ``[Y, X]`` sampling in
micrometers. No microscope calibration is inferred. Legacy batch segmentation,
Measure, volumes, and additional PSF or enhancement processing refuse this
route. A separate, explicitly configured calibrated V2 batch route supports
independent two-dimensional fields within its validated settings limits.
Leaving the calibration blank preserves legacy processing.

The calibration JSON has ``version: 1``, ``axes: "YX"``,
``pixel_size_um: [Y, X]``, ``reference_channel`` and a ``channels`` list.
Every physical channel appears exactly once. Each channel entry supplies
``channel``, ``shift_to_reference_px: [dy, dx]``, a measured two-dimensional
``psf`` array, ``psf_origin_px: [Y, X]`` and ``psf_pixel_size_um: [Y, X]``.
The reference channel has zero shift. A shift means
``aligned[y, x] = acquired[y - dy, x - dx]``; only integer shifts are
supported. Each finite nonnegative PSF must have positive mass, odd dimensions
and its declared origin at its center. Image and PSF sampling must agree;
no implicit resampling is performed.

Each channel is convolved with all the other channels' supplied PSFs. Their
common effective PSF is the product convolution of all supplied kernels:
resolution is broadened, not restored by deconvolution. This assumes a
shift-invariant PSF per observation channel and a common linear mixing matrix.
Only the intersection supported by real acquired pixels is used. No reflected,
wrapped or zero-padded image pixels enter the solve or segmentation.
Controls and the selected image undergo the same calibrated operation.

The valid crop retains signed float residuals, including negative values from
noise or model mismatch; these are not negative biological concentrations.
Provenance records the matrix, calibration and kernel hashes, shifts, sampling,
common support, original-canvas crop and negative fractions. Segmentation
receives a separately recorded nonnegative view. Labels are restored to the
original canvas; complete labels touching the valid crop boundary are excluded
and recorded rather than presented as complete objects. Source files and raw
measurement data remain unchanged. Real known-amount double-stain validation
and calibrated Measure integration remain open.

Calibration is bounded to 16 channels, 31-pixel individual PSF dimensions,
127-pixel combined support, integer shifts within 4096 pixels and 16,777,216
selected channel pixels per image. Oversized or incompatible inputs refuse
before calibrated channel-copy allocation. These limits do not alter the
legacy route. Across all selected control fields, at most 67,108,864 channel
pixels are retained, bounding their matched float64 arrays to 512 MiB; this is
not a total process-memory limit, as convolution and other working arrays
require additional memory. Choose fewer or smaller control fields if this
aggregate limit is exceeded. This limit never triggers additional sampling or
dropping; the existing selection of up to 24 fields per dye is unchanged.

Engine parameter reference
--------------------------

The following definitions are generated from the same parameter docstrings
as the API pages. They include programmatic field names for reproducing a
configuration in code. The GUI shows only the subset used by the selected
method. Secondary growth has its own controls, independent of Maxima +
propagate, and defaults to a global threshold in the screen.

.. include:: _generated/make_masks_parameters.rst
