Overview: inspect plate images and export figures
=================================================

Set the image-folder source, then use the **Field-of-view overview** icon beside the source in Generate Masks, or the matching icon in Make Masks. The icon opens the metadata step before the plate browser. This workflow inspects images; it does not segment objects.

Check metadata before proceeding
--------------------------------

Start with **Automatic filename metadata** under **Microscope preset**, then press **Preview metadata**. Check each plate's parsed wells, fields and channel IDs. Use a microscope preset or **Custom filename regex** if automatic detection does not describe your filenames. A custom pattern must include a named ``wellID`` group; select only channel IDs reported by the images.

Select a plate to edit its filename pattern; **Apply to all plates** shares that pattern with the others. Confirm **Plate geometry**, time and Z choices, and pixel size when known, then press **Next**. The default source scope reads this folder only; including subfolders can also admit derived images. Removing every source disables continuation until a valid source is restored.

Navigate plates, wells and images
---------------------------------

Use **Plate**, **Row**, **Column**, **Field**, **Time** and **Z** to choose the acquisition context. Click a well to see its fields, then open a field image. **Back** returns from an image to its well, then from the well to the plate. Ctrl/Shift selection retains fields or wells for an explicit export selection.

Display and canvas
------------------

Choose display channels, colours and normalization in the settings. **Channel to edit** selects shared or channel-specific values; **Apply settings to channel** stores that channel's own values. Display normalization changes the rendering, not the stored source pixels.

Use **Export…** for **Guided export**, **Custom export** or **Contact sheet PDFs**. Guided export walks through wells, fields, placement and preview. Custom export lets you drag images onto a page grid and arrange text and images. Captured image display settings belong to each placed image.

In the canvas, **Save layout…** writes editable JSON; **Open layout…** reopens it. Keep the image files available: a layout is not an archive of their pixels. A saved layout whose captured display cannot be unpacked or lacks required channel, limit or display entries is refused before replacing the current canvas. Missing image files can leave a partially recoverable layout; restore the files and reload their source as needed to recover those images.

Preview quality and PDF resolution
----------------------------------

**Preview resolution** limits the maximum preview edge; smaller previews load faster. This thumbnail choice is separate from PDF export quality. Contact sheets offer full source resolution or a 1600/3200-pixel field limit plus DPI. In the custom canvas, **Settings…** controls export DPI; **Every source pixel** keeps all source pixels. DPI limits embedded image density; it cannot add detail absent from the source.

Image measurements and source ownership
---------------------------------------

**Export image-level measurements** writes one row per selected image to the ``fov`` table in ``measurements/measurements.db``. By default, features describe whole images. Whole-image intensity statistics use finite stored pixels without display normalization; some texture features use reduced copies. Under **Objects**, supplying existing Mask generation masks adds per-image object summaries without segmenting new objects. CSV export is on by default and writes ``image_measurements.csv`` beside the database; the output status lists written paths.

The measurements database is written under the first path in the current loaded-source list. Editing Source alone does not change ownership of retained images or measurements. Load and validate the intended source before measuring it. Adding or removing plates updates the loaded source set; check the current plate and output status before export. For individual-object rows, use the separate Measure workflow in :doc:`workflows`.
