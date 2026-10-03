Measure: preview checked images and verify mask planes
======================================================

Open **Measure** with a run folder or its ``merged`` folder. The **Live**
view loads a field and shows crops from its configured object mask. Use
**Crop settings…** to choose the object, mask planes, image channels, crop
size and filters before measuring the whole experiment.

Compare objects from several fields
-----------------------------------

Open **Checked images** and check the fields to include. Each checkbox is
independent. The grid combines their crops and names the source field in
each group; it does not superimpose fields. Hover a crop for its full source
path, label, area and filter result. Double-click a crop to open that exact
source, including when another field contains the same label number.

Uncheck a field to remove its crops, or choose **Uncheck all images** to
clear the grid. Changing channels, object type or filters refreshes all
checked fields. The field dropdown retains ordinary single-field navigation
when only its current field is checked. With several fields checked, it
changes the active source without replacing the checked selection.

The image list uses the existing bounded random sample. Already checked
fields remain available when the sample changes. **Maximum preview crops**
limits the combined grid, with slots divided among checked fields. Increase
it if there are more checked fields than available slots. Additional fields
are read one at a time. A load failure names the field and leaves valid
fields available for inspection.

Checking images changes the preview only. It does not select the fields
that Measure processes or change saved measurements or exports.
**Propagate settings** retains its existing purpose: copying crop and
filter settings into the Measure form.

Preview and export unmixed crops
--------------------------------

With **Show alpha features** enabled, set ``unmix=True`` and configure
``unmix_controls`` in the **Measure** form. Toggle **Unmixed display** to use
those single-stain controls in the crop preview. The default display is raw.
Invalid or missing controls are reported. This toggle changes the display
only; it leaves run settings, source arrays and batch PNG outputs unchanged.

After the preview finishes, choose **Export displayed crops…**, select a
parent folder and enter a new folder name. Only the completed preview's crops
are exported, with their displayed colours and raw or unmixed pixels.
``provenance.json`` records source paths, object identities, display settings
and unmixing information. Existing folders are refused. Cancelling or changing
the preview discards an unfinished export; source files and batch outputs
remain unchanged.

Preview reference-well calibration
----------------------------------

With **Show alpha features** enabled, turn on ``intensity_calibration`` and
configure the reference wells, camera offset and statistic in the **Measure**
form. Turn off **Test mode**, then choose **Preview** beside
**Intensity calibration wells**. Use a local run folder or its ``merged``
folder. The preview reads the current settings without running **Measure** or
writing images, measurements or settings.

Each source folder is planned separately. The first plate in name order is
the reference. The table shows each plate and channel's multiplicative gain,
reference statistic and reference-field count. Changing settings invalidates
the result. **Cancel** discards the preview while any source scan already
running finishes safely; reopening waits for that scan to stop.

Measure on a GPU
----------------

With **Show alpha features** enabled, turn on ``measure_gpu`` under **GPU
Measurement (Alpha)**. The per-object intensity statistics, GLCM homogeneity
and Zernike moments are then computed with PyTorch on a CUDA GPU, all objects
of a field at once. When cuCIM is also installed (the ``gpu`` extra), the
per-object morphology table is computed on the GPU too; without cuCIM it stays
on the CPU. Values match a CPU run within floating-point tolerance. Only 2-D
masks without voxel spacing use the GPU; anything else, or a missing PyTorch
or CUDA device, measures on the CPU as usual.

Resolve a stored plane-layout conflict
--------------------------------------

A merged folder can include ``.spacr_plane_layout.json``, which records the
image channels and mask planes written with its arrays. Measure checks
explicit mask-plane settings against this record before using them.
``uninfected=True`` keeps uninfected cells; it does not make an incorrect
pathogen mask plane valid.

If a saved form disagrees with the folder, **Run** explains the mismatch.
Choose **Use stored image layout** to copy the recorded planes into the
form, review the object settings, then press **Run** again. An absent object
uses ``None``. **Cancel** leaves the form unchanged. Sources with different
or unknown layouts must be run separately with their matching settings.

For Python calls, use the plane indices recorded for the selected merged
folder. A conflicting explicit index raises
:class:`spacr.crops.PlaneLayoutConflict`. Unspecified defaults can be
resolved through :func:`spacr.crops.reconcile_merged_mask_dims`. Do not edit
or remove the sidecar to bypass the check: that can make an intensity plane
look like an object mask.
