Measure: preview checked images and verify mask planes
======================================================

Open **Measure** with a run folder or its ``merged`` folder. If ``src`` points
at one of the plate's output subfolders, such as ``measurements`` or
``masks``, Measure reads that plate's ``merged`` folder and says so in the
console. If there is no ``merged`` folder to read, Measure does not start and
its message names the missing folder and what to set instead. The **Live**
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
