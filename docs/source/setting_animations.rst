Setting animation gallery
=========================

spaCR includes short, deterministic GIFs for settings whose effect is
easier to understand visually. In the desktop interface a purple dot
above the teal API dot opens the corresponding animation immediately
above the setting. The midpoint between both dots remains aligned with
the setting label.

The diagrams use a shared biological grammar: white fibroblast or
motile immune-cell outlines, blue nuclei with unequal nucleoli, teal
Toxoplasma tachyzoites inside an outline-only parasitophorous vacuole,
and soft-magenta Golgi cisternae. Filled regions are translucent and
the rounded field perimeter remains white on black.

The cell, nucleus and nucleoli, two-parasite vacuole, and Golgi use
the reviewed, artist-authored SVG paths checked into ``tools/``.
Qt renders those exact Bezier paths at high resolution before each
animation frame is composited, keeping the outlines smooth at small
sizes. Source-template SHA-256 hashes are recorded in the manifest.

Animations are resolved by exact setting key through
:mod:`spacr.setting_animations`; the assets and manifest are generated
reproducibly by ``tools/generate_setting_animations.py``.

Mask filtering
--------------

.. _setting-animation-cell-remove-border-objects:

Cell — Remove border objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_remove_border_objects.gif
   :alt: Cell — Remove border objects setting animation
   :width: 300px

**Settings:** ``cell_remove_border_objects``, ``remove_border_cells``

.. _setting-animation-cell-min-area:

Cell — Minimum object area
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_min_area.gif
   :alt: Cell — Minimum object area setting animation
   :width: 300px

**Settings:** ``cell_min_area``, ``cell_min_size``

.. _setting-animation-cell-max-area:

Cell — Maximum object area
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_max_area.gif
   :alt: Cell — Maximum object area setting animation
   :width: 300px

**Settings:** ``cell_max_area``

.. _setting-animation-cell-min-intensity-percentile:

Cell — Minimum intensity percentile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_min_intensity_percentile.gif
   :alt: Cell — Minimum intensity percentile setting animation
   :width: 300px

**Settings:** ``cell_min_intensity_percentile``

.. _setting-animation-cell-max-intensity-percentile:

Cell — Maximum intensity percentile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_max_intensity_percentile.gif
   :alt: Cell — Maximum intensity percentile setting animation
   :width: 300px

**Settings:** ``cell_max_intensity_percentile``

.. _setting-animation-nucleus-remove-border-objects:

Nucleus — Remove border objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_remove_border_objects.gif
   :alt: Nucleus — Remove border objects setting animation
   :width: 300px

**Settings:** ``nucleus_remove_border_objects``, ``remove_border_nuclei``

.. _setting-animation-nucleus-min-area:

Nucleus — Minimum object area
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_min_area.gif
   :alt: Nucleus — Minimum object area setting animation
   :width: 300px

**Settings:** ``nucleus_min_area``, ``nucleus_min_size``

.. _setting-animation-nucleus-max-area:

Nucleus — Maximum object area
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_max_area.gif
   :alt: Nucleus — Maximum object area setting animation
   :width: 300px

**Settings:** ``nucleus_max_area``

.. _setting-animation-nucleus-min-intensity-percentile:

Nucleus — Minimum intensity percentile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_min_intensity_percentile.gif
   :alt: Nucleus — Minimum intensity percentile setting animation
   :width: 300px

**Settings:** ``nucleus_min_intensity_percentile``

.. _setting-animation-nucleus-max-intensity-percentile:

Nucleus — Maximum intensity percentile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_max_intensity_percentile.gif
   :alt: Nucleus — Maximum intensity percentile setting animation
   :width: 300px

**Settings:** ``nucleus_max_intensity_percentile``

.. _setting-animation-pathogen-remove-border-objects:

Pathogen — Remove border objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_remove_border_objects.gif
   :alt: Pathogen — Remove border objects setting animation
   :width: 300px

**Settings:** ``pathogen_remove_border_objects``, ``remove_border_pathogens``

.. _setting-animation-pathogen-min-area:

Pathogen — Minimum object area
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_min_area.gif
   :alt: Pathogen — Minimum object area setting animation
   :width: 300px

**Settings:** ``pathogen_min_area``, ``pathogen_min_size``

.. _setting-animation-pathogen-max-area:

Pathogen — Maximum object area
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_max_area.gif
   :alt: Pathogen — Maximum object area setting animation
   :width: 300px

**Settings:** ``pathogen_max_area``

.. _setting-animation-pathogen-min-intensity-percentile:

Pathogen — Minimum intensity percentile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_min_intensity_percentile.gif
   :alt: Pathogen — Minimum intensity percentile setting animation
   :width: 300px

**Settings:** ``pathogen_min_intensity_percentile``

.. _setting-animation-pathogen-max-intensity-percentile:

Pathogen — Maximum intensity percentile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_max_intensity_percentile.gif
   :alt: Pathogen — Maximum intensity percentile setting animation
   :width: 300px

**Settings:** ``pathogen_max_intensity_percentile``

.. _setting-animation-organelle-remove-border-objects:

Organelle — Remove border objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_remove_border_objects.gif
   :alt: Organelle — Remove border objects setting animation
   :width: 300px

**Settings:** ``organelle_remove_border_objects``, ``organelle_remove_border``, ``remove_border_organelles``

.. _setting-animation-organelle-min-area:

Organelle — Minimum object area
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_min_area.gif
   :alt: Organelle — Minimum object area setting animation
   :width: 300px

**Settings:** ``organelle_min_area``, ``organelle_min_size``

.. _setting-animation-organelle-max-area:

Organelle — Maximum object area
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_max_area.gif
   :alt: Organelle — Maximum object area setting animation
   :width: 300px

**Settings:** ``organelle_max_area``, ``organelle_max_size``

.. _setting-animation-organelle-min-intensity-percentile:

Organelle — Minimum intensity percentile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_min_intensity_percentile.gif
   :alt: Organelle — Minimum intensity percentile setting animation
   :width: 300px

**Settings:** ``organelle_min_intensity_percentile``

.. _setting-animation-organelle-max-intensity-percentile:

Organelle — Maximum intensity percentile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_max_intensity_percentile.gif
   :alt: Organelle — Maximum intensity percentile setting animation
   :width: 300px

**Settings:** ``organelle_max_intensity_percentile``

Mask repair
-----------

.. _setting-animation-merge-edge-pathogen-cells:

Merge edge-pathogen cells
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/merge_edge_pathogen_cells.gif
   :alt: Merge edge-pathogen cells setting animation
   :width: 300px

**Settings:** ``merge_edge_pathogen_cells``

.. _setting-animation-adjust-cells:

Adjust fragmented cells
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/adjust_cells.gif
   :alt: Adjust fragmented cells setting animation
   :width: 300px

**Settings:** ``adjust_cells``

.. _setting-animation-cell-perimeter-fraction:

Cell perimeter merge
~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_perimeter_fraction.gif
   :alt: Cell perimeter merge setting animation
   :width: 300px

**Settings:** ``cell_perimeter_fraction``

.. _setting-animation-cell-intensity-merge:

Cell intensity merge
~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_intensity_merge.gif
   :alt: Cell intensity merge setting animation
   :width: 300px

**Settings:** ``cell_intensity_merge``, ``cell_intensity_threshold_method``, ``cell_intensity_percentile``

.. _setting-animation-cell-intensity-split:

Cell watershed split
~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_intensity_split.gif
   :alt: Cell watershed split setting animation
   :width: 300px

**Settings:** ``cell_intensity_split``, ``cell_area_multiplier``, ``cell_min_distance``, ``cell_min_object_area``

.. _setting-animation-nucleus-perimeter-fraction:

Nucleus perimeter merge
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_perimeter_fraction.gif
   :alt: Nucleus perimeter merge setting animation
   :width: 300px

**Settings:** ``nucleus_perimeter_fraction``

.. _setting-animation-nucleus-intensity-merge:

Nucleus intensity merge
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_intensity_merge.gif
   :alt: Nucleus intensity merge setting animation
   :width: 300px

**Settings:** ``nucleus_intensity_merge``, ``nucleus_intensity_threshold_method``, ``nucleus_intensity_percentile``

.. _setting-animation-nucleus-intensity-split:

Nucleus watershed split
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_intensity_split.gif
   :alt: Nucleus watershed split setting animation
   :width: 300px

**Settings:** ``nucleus_intensity_split``, ``nucleus_area_multiplier``, ``nucleus_min_distance``, ``nucleus_min_object_area``

.. _setting-animation-pathogen-perimeter-fraction:

Pathogen perimeter merge
~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_perimeter_fraction.gif
   :alt: Pathogen perimeter merge setting animation
   :width: 300px

**Settings:** ``pathogen_perimeter_fraction``

.. _setting-animation-pathogen-intensity-merge:

Pathogen intensity merge
~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_intensity_merge.gif
   :alt: Pathogen intensity merge setting animation
   :width: 300px

**Settings:** ``pathogen_intensity_merge``, ``pathogen_intensity_threshold_method``, ``pathogen_intensity_percentile``

.. _setting-animation-pathogen-intensity-split:

Pathogen watershed split
~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_intensity_split.gif
   :alt: Pathogen watershed split setting animation
   :width: 300px

**Settings:** ``pathogen_intensity_split``, ``pathogen_area_multiplier``, ``pathogen_min_distance``, ``pathogen_min_object_area``

.. _setting-animation-organelle-perimeter-fraction:

Organelle perimeter merge
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_perimeter_fraction.gif
   :alt: Organelle perimeter merge setting animation
   :width: 300px

**Settings:** ``organelle_perimeter_fraction``

.. _setting-animation-organelle-intensity-merge:

Organelle intensity merge
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_intensity_merge.gif
   :alt: Organelle intensity merge setting animation
   :width: 300px

**Settings:** ``organelle_intensity_merge``, ``organelle_intensity_threshold_method``, ``organelle_intensity_percentile``

.. _setting-animation-organelle-intensity-split:

Organelle watershed split
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_intensity_split.gif
   :alt: Organelle watershed split setting animation
   :width: 300px

**Settings:** ``organelle_intensity_split``, ``organelle_area_multiplier``, ``organelle_min_distance``, ``organelle_min_object_area``

.. _setting-animation-fill-in:

Fill holes in masks
~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/fill_in.gif
   :alt: Fill holes in masks setting animation
   :width: 300px

**Settings:** ``fill_in``

Segmentation
------------

.. _setting-animation-cell-CP-prob:

Cell probability threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_CP_prob.gif
   :alt: Cell probability threshold setting animation
   :width: 300px

**Settings:** ``cell_CP_prob``

.. _setting-animation-cell-FT:

Cell flow threshold
~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_FT.gif
   :alt: Cell flow threshold setting animation
   :width: 300px

**Settings:** ``cell_FT``

.. _setting-animation-cell-diameter:

Cell diameter
~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_diameter.gif
   :alt: Cell diameter setting animation
   :width: 300px

**Settings:** ``cell_diameter``

.. _setting-animation-nucleus-CP-prob:

Nucleus probability threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_CP_prob.gif
   :alt: Nucleus probability threshold setting animation
   :width: 300px

**Settings:** ``nucleus_CP_prob``

.. _setting-animation-nucleus-FT:

Nucleus flow threshold
~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_FT.gif
   :alt: Nucleus flow threshold setting animation
   :width: 300px

**Settings:** ``nucleus_FT``

.. _setting-animation-nucleus-diameter:

Nucleus diameter
~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_diameter.gif
   :alt: Nucleus diameter setting animation
   :width: 300px

**Settings:** ``nucleus_diameter``

.. _setting-animation-pathogen-CP-prob:

Pathogen probability threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_CP_prob.gif
   :alt: Pathogen probability threshold setting animation
   :width: 300px

**Settings:** ``pathogen_CP_prob``

.. _setting-animation-pathogen-FT:

Pathogen flow threshold
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_FT.gif
   :alt: Pathogen flow threshold setting animation
   :width: 300px

**Settings:** ``pathogen_FT``

.. _setting-animation-pathogen-diameter:

Pathogen diameter
~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_diameter.gif
   :alt: Pathogen diameter setting animation
   :width: 300px

**Settings:** ``pathogen_diameter``

.. _setting-animation-organelle-diameter:

Organelle diameter
~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_diameter.gif
   :alt: Organelle diameter setting animation
   :width: 300px

**Settings:** ``organelle_diameter``

.. _setting-animation-organelle-CP-prob:

Organelle probability threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_CP_prob.gif
   :alt: Organelle probability threshold setting animation
   :width: 300px

**Settings:** ``organelle_CP_prob``

.. _setting-animation-organelle-FT:

Organelle flow threshold
~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_FT.gif
   :alt: Organelle flow threshold setting animation
   :width: 300px

**Settings:** ``organelle_FT``

Image preprocessing
-------------------

.. _setting-animation-remove-background-cell:

Cell background subtraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/remove_background_cell.gif
   :alt: Cell background subtraction setting animation
   :width: 300px

**Settings:** ``remove_background_cell``, ``cell_background``

.. _setting-animation-cell-Signal-to-noise:

Cell signal-to-noise
~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cell_Signal_to_noise.gif
   :alt: Cell signal-to-noise setting animation
   :width: 300px

**Settings:** ``cell_Signal_to_noise``

.. _setting-animation-remove-background-nucleus:

Nucleus background subtraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/remove_background_nucleus.gif
   :alt: Nucleus background subtraction setting animation
   :width: 300px

**Settings:** ``remove_background_nucleus``, ``nucleus_background``

.. _setting-animation-nucleus-Signal-to-noise:

Nucleus signal-to-noise
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/nucleus_Signal_to_noise.gif
   :alt: Nucleus signal-to-noise setting animation
   :width: 300px

**Settings:** ``nucleus_Signal_to_noise``

.. _setting-animation-remove-background-pathogen:

Pathogen background subtraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/remove_background_pathogen.gif
   :alt: Pathogen background subtraction setting animation
   :width: 300px

**Settings:** ``remove_background_pathogen``, ``pathogen_background``

.. _setting-animation-pathogen-Signal-to-noise:

Pathogen signal-to-noise
~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/pathogen_Signal_to_noise.gif
   :alt: Pathogen signal-to-noise setting animation
   :width: 300px

**Settings:** ``pathogen_Signal_to_noise``

.. _setting-animation-normalization-percentiles:

Image normalization percentiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/normalization_percentiles.gif
   :alt: Image normalization percentiles setting animation
   :width: 300px

**Settings:** ``normalization_percentiles``, ``normalize``, ``normalize_plots``

Organelle preprocessing
-----------------------

.. _setting-animation-organelle-fill-holes:

Fill small organelle holes
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_fill_holes.gif
   :alt: Fill small organelle holes setting animation
   :width: 300px

**Settings:** ``organelle_fill_holes``

.. _setting-animation-organelle-watershed-spots:

Split touching organelle spots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_watershed_spots.gif
   :alt: Split touching organelle spots setting animation
   :width: 300px

**Settings:** ``organelle_watershed_spots``

.. _setting-animation-organelle-skeletonize:

Skeletonize organelle networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_skeletonize.gif
   :alt: Skeletonize organelle networks setting animation
   :width: 300px

**Settings:** ``organelle_skeletonize``

.. _setting-animation-organelle-rolling-ball:

Rolling-ball background correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_rolling_ball.gif
   :alt: Rolling-ball background correction setting animation
   :width: 300px

**Settings:** ``organelle_rolling_ball``, ``organelle_rolling_ball_radius``

.. _setting-animation-organelle-clahe:

Organelle local contrast (CLAHE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_clahe.gif
   :alt: Organelle local contrast (CLAHE) setting animation
   :width: 300px

**Settings:** ``organelle_clahe``, ``organelle_clahe_clip_limit``

.. _setting-animation-organelle-mask-within-cells:

Mask organelles within cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_mask_within_cells.gif
   :alt: Mask organelles within cells setting animation
   :width: 300px

**Settings:** ``organelle_mask_within_cells``

.. _setting-animation-organelle-log-threshold:

Organelle segmentation threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/organelle_log_threshold.gif
   :alt: Organelle segmentation threshold setting animation
   :width: 300px

**Settings:** ``organelle_log_threshold``, ``organelle_unet_threshold``

Plot appearance
---------------

.. _setting-animation-outline-thickness:

Mask outline thickness
~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/outline_thickness.gif
   :alt: Mask outline thickness setting animation
   :width: 300px

**Settings:** ``outline_thickness``

Crop output
-----------

.. _setting-animation-use-bounding-box:

Keep bounding-box context
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/use_bounding_box.gif
   :alt: Keep bounding-box context setting animation
   :width: 300px

**Settings:** ``use_bounding_box``

.. _setting-animation-dialate-pngs:

Dilate crop masks
~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/dialate_pngs.gif
   :alt: Dilate crop masks setting animation
   :width: 300px

**Settings:** ``dialate_pngs``, ``dialate_png_ratios``

.. _setting-animation-crop-mode:

Choose crop target
~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/crop_mode.gif
   :alt: Choose crop target setting animation
   :width: 300px

**Settings:** ``crop_mode``

.. _setting-animation-png-size:

Crop canvas size
~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/png_size.gif
   :alt: Crop canvas size setting animation
   :width: 300px

**Settings:** ``png_size``

Measurement
-----------

.. _setting-animation-cytoplasm:

Derive cytoplasm compartment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/cytoplasm.gif
   :alt: Derive cytoplasm compartment setting animation
   :width: 300px

**Settings:** ``cytoplasm``

.. _setting-animation-radial-dist:

Radial-distance shells
~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/radial_dist.gif
   :alt: Radial-distance shells setting animation
   :width: 300px

**Settings:** ``radial_dist``, ``distance_gaussian_sigma``

.. _setting-animation-uninfected:

Keep or remove uninfected cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/uninfected.gif
   :alt: Keep or remove uninfected cells setting animation
   :width: 300px

**Settings:** ``uninfected``

Tracking & volumetric
---------------------

.. _setting-animation-timelapse-remove-transient:

Remove transient tracks
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/timelapse_remove_transient.gif
   :alt: Remove transient tracks setting animation
   :width: 300px

**Settings:** ``timelapse_remove_transient``, ``timelapse_frame_limits``

.. _setting-animation-timelapse-displacement:

Maximum linking displacement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/timelapse_displacement.gif
   :alt: Maximum linking displacement setting animation
   :width: 300px

**Settings:** ``timelapse_displacement``, ``ultrack_max_distance``, ``t_max_displacement_px``, ``t_max_displacement_um``

.. _setting-animation-timelapse-memory:

Tracking memory
~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/timelapse_memory.gif
   :alt: Tracking memory setting animation
   :width: 300px

**Settings:** ``timelapse_memory``

.. _setting-animation-t-link-threshold:

Timepoint overlap threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/t_link_threshold.gif
   :alt: Timepoint overlap threshold setting animation
   :width: 300px

**Settings:** ``t_link_threshold``

.. _setting-animation-stitch-threshold:

Z-plane stitch threshold
~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/stitch_threshold.gif
   :alt: Z-plane stitch threshold setting animation
   :width: 300px

**Settings:** ``stitch_threshold``

.. _setting-animation-z-projection:

Z projection
~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/z_projection.gif
   :alt: Z projection setting animation
   :width: 300px

**Settings:** ``z_projection``, ``all_to_mip``, ``pick_slice``

.. _setting-animation-t-project-for-tracking:

Project volumes for tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/t_project_for_tracking.gif
   :alt: Project volumes for tracking setting animation
   :width: 300px

**Settings:** ``t_project_for_tracking``

.. _setting-animation-straightness-filter:

Remove overly straight tracks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/straightness_filter.gif
   :alt: Remove overly straight tracks setting animation
   :width: 300px

**Settings:** ``straightness_filter``, ``straightness_threshold``

.. _setting-animation-zscore-thresh:

Smooth per-track outliers
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/zscore_thresh.gif
   :alt: Smooth per-track outliers setting animation
   :width: 300px

**Settings:** ``zscore_thresh``

.. _setting-animation-ultrack-division-weight:

Cell-division linking
~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/ultrack_division_weight.gif
   :alt: Cell-division linking setting animation
   :width: 300px

**Settings:** ``ultrack_division_weight``

.. _setting-animation-ultrack-contour-sigma:

Contour smoothing
~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/ultrack_contour_sigma.gif
   :alt: Contour smoothing setting animation
   :width: 300px

**Settings:** ``ultrack_contour_sigma``

Image UMAP
----------

.. _setting-animation-n-neighbors:

UMAP neighborhood size
~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/n_neighbors.gif
   :alt: UMAP neighborhood size setting animation
   :width: 300px

**Settings:** ``n_neighbors``

.. _setting-animation-min-dist:

UMAP minimum distance
~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/min_dist.gif
   :alt: UMAP minimum distance setting animation
   :width: 300px

**Settings:** ``min_dist``

.. _setting-animation-plot-images:

Show object images
~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/plot_images.gif
   :alt: Show object images setting animation
   :width: 300px

**Settings:** ``plot_images``

.. _setting-animation-remove-image-canvas:

Remove image canvas
~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/remove_image_canvas.gif
   :alt: Remove image canvas setting animation
   :width: 300px

**Settings:** ``remove_image_canvas``

.. _setting-animation-plot-outlines:

Show cluster outlines
~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/plot_outlines.gif
   :alt: Show cluster outlines setting animation
   :width: 300px

**Settings:** ``plot_outlines``

.. _setting-animation-plot-points:

Show embedding points
~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/plot_points.gif
   :alt: Show embedding points setting animation
   :width: 300px

**Settings:** ``plot_points``

.. _setting-animation-smooth-lines:

Smooth cluster outlines
~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/smooth_lines.gif
   :alt: Smooth cluster outlines setting animation
   :width: 300px

**Settings:** ``smooth_lines``

.. _setting-animation-remove-cluster-noise:

Remove cluster noise
~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/remove_cluster_noise.gif
   :alt: Remove cluster noise setting animation
   :width: 300px

**Settings:** ``remove_cluster_noise``

.. _setting-animation-plot-by-cluster:

Sample images by cluster
~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/plot_by_cluster.gif
   :alt: Sample images by cluster setting animation
   :width: 300px

**Settings:** ``plot_by_cluster``

.. _setting-animation-dot-size:

Embedding point size
~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/dot_size.gif
   :alt: Embedding point size setting animation
   :width: 300px

**Settings:** ``dot_size``

.. _setting-animation-img-zoom:

Embedding image zoom
~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/img_zoom.gif
   :alt: Embedding image zoom setting animation
   :width: 300px

**Settings:** ``img_zoom``

.. _setting-animation-eps:

Density clustering radius
~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/eps.gif
   :alt: Density clustering radius setting animation
   :width: 300px

**Settings:** ``eps``, ``min_samples``, ``clustering``

Alignment & stitching
---------------------

.. _setting-animation-overlap:

Tile overlap
~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/overlap.gif
   :alt: Tile overlap setting animation
   :width: 300px

**Settings:** ``overlap``

.. _setting-animation-blend:

Tile seam blending
~~~~~~~~~~~~~~~~~~

.. image:: ../../spacr/resources/setting_animations/gifs/blend.gif
   :alt: Tile seam blending setting animation
   :width: 300px

**Settings:** ``blend``
