Host–Pathogen Analysis
========================

From **Home → Assays → Toxoplasma**, open **Host–Pathogen Analysis**.
This alpha module combines vacuole-level marker recruitment with host and well
summaries. Existing :doc:`Recruitment <recruitment>` remains available.

Try the real microscopy test data
----------------------------------------

#. Open **Home → Assays → Toxoplasma → Host–Pathogen Analysis** and select
   **Load test data…**. The download is approximately 114 MB.
#. Enable **Live** beside the bottom action controls, then select
   **Run preview**. Choose either measured field and select vacuoles in the
   image or table to inspect their host links and marker ratios.
#. Set **Image channel** to 2 for Toxoplasma, 1 for RNF213 or 3 for CellMask.
   Leave the mask-plane selectors on **Auto**; the dataset includes the plane
   manifest. These display choices do not change the analysis channels.
#. Turn **Live** off and expand **Actions** if the run controls are collapsed.
   Select **Run** to analyze both fields. Compare the results with the supplied
   ``example_cells.csv``, ``example_vacuoles.csv`` and ``example_wells.csv``.

Follow the `Host–Pathogen video walkthrough
<tutorials/#lesson=85_host_pathogen>`_ for the complete example.

The `Host–Pathogen dataset
<https://huggingface.co/datasets/einarolafsson/spacr-example-host-pathogen/tree/83b73d7a0c4f8a9ea145304c2ba90acc16768031>`_
contains two acquired THP-1/RNF213 fields, four intensity channels and prepared
automatic masks. Its measurements cover 164 host cells, 189 nuclei,
97 whole vacuoles and 164 cytoplasm objects across the two fields. The acquired
intensity planes are unchanged; the masks underwent Measure's normal
parent/child reconciliation so the distributed outlines match measured objects.
The dataset card and ``example_manifest.json`` describe the source fields,
preparation and checks. These masks are not manually validated ground truth.

The supplied settings compare vacuole RNF213 means with their associated host
cytoplasm means in channel 1. Marker thresholds are deliberately unset: ratios
are available, while positive/negative marker states remain unknown until you
choose suitable control-calibrated thresholds. Individual-parasite counts are
not supplied, so replication remains ``not_measured``. This small sample does
not establish biological differences between its control conditions.

The command-line download is ``spacr-download host_pathogen``. The prepared
measurements let you use this example directly; your own image project needs
the Mask and Measure preparation described below.

Inspect one field before running
--------------------------------

After selecting a measured project in **Source**, enable the bottom **Live** control.
The preview uses the current form settings and the same analysis
functions as a full run. Select **Run preview** for the initial calculation.
While the panel remains visible, later settings changes refresh that preview.
Use **Refresh** to reload the field choices and **Cancel** to cancel a pending
preview. Preview reads the measurements database without writing result files.

Its host counts and infection fractions describe the selected field only.
They are not whole-well denominators. Fields containing vacuoles without
measured host cells remain available. Missing or zero reference intensities
produce unknown marker states; an unknown state must not be read as negative.

Overlay plane selection defaults to **Auto**, using the recorded
``.spacr_plane_layout.json`` metadata. Select explicit host, vacuole and
parasite plane indices when metadata is absent,
or use ``-1`` to hide an overlay. Changing the displayed intensity channel
does not change the analysis marker settings. If an image cannot be loaded,
the measured results remain available in the table.

For scripted previews, :func:`spacr.host_pathogen_preview.preview_fields`
offers at most 50 fields by default and reports whether more exist.
:func:`spacr.host_pathogen_preview.preview_field` analyzes one selected field,
refusing input tables with more than 100,000 rows for that field by default.
These bounds keep the preview limited; use the full analysis for project-wide
reports.

Prepare the counting units
--------------------------

Run Mask and Measure first. Keep uninfected host cells in Measure
(``uninfected=True``) when the intended infection denominator is all measured
hosts. Previously discarded host cells cannot be reconstructed here.

Supply a cell table, a table with one object per whole vacuole (default:
``pathogen``), and a host reference compartment table (default:
``cytoplasm``). Individual-parasite masks cannot substitute for whole-vacuole
masks. Select marker channels and per-channel ratio thresholds calibrated
with appropriate controls; the module does not learn these thresholds.

Replication counts require either a parasite table with explicit parent-vacuole
identities or an existing count column. Choose one count source. Without one,
replication remains unmeasured rather than being inferred from vacuole area.
Host identity alone is insufficient when a host contains multiple vacuoles.

When Measure links individual organelle-role objects to whole-vacuole masks,
one parent must cover strictly more than half of a child's pixels. Outside
objects and ambiguous overlaps retain a missing parent. The stored parent is
``pathogen_id``; the stored overlap feature is role-prefixed, for example
``organelle_pathogen_overlap_fraction``. Use such an object table as parasites
only when those masks actually represent individual parasites.

Interpret the reports
---------------------

The output directory ``results/host_pathogen`` contains ``vacuoles.csv``,
``cells.csv``, ``wells.csv``, ``marker_states.csv``,
``replication_distribution.csv``, ``orphan_parasites.csv`` and the exact
``settings.json``. With multiple input projects, the combined report is saved
under the first project. Database paths separate sources even when plate or
field identifiers repeat.

Unlinked vacuoles and missing, invalid or zero host-reference intensities yield
unknown marker states. Joint marker fractions include unknown vacuoles in
their denominator; replication fractions use only vacuoles with counts.
Host infection fractions describe retained measured host cells. These
denominators answer different questions and should be reported explicitly.

For scripted use, see :func:`spacr.host_pathogen.analyze_host_pathogen` and
:func:`spacr.host_pathogen.summarize_tables`. The headless module name is
``host_pathogen`` in ``spacr-run``.

Choosing a Replication Assay method
--------------------------------------

The separate **Replication Assay → Method** selector defaults to
``direct_count``, which counts individually segmented parasites per assigned
vacuole. ``size_proxy`` uses the existing endodyogeny analysis and its
**Size Proxy (Legacy)** controls. That readout aggregates pathogen area per
host cell; multiple vacuoles in one host are combined. It is an area-derived
proxy, not a parasite count or measured volume.

Both available routes return ``replication_method`` metadata. The whole-vacuole
deep-learning option is marked coming soon and cannot run before its model is
available. See :func:`spacr.submodules.analyze_replication` and
:func:`spacr.submodules.analyze_endodyogeny` for the exact settings and outputs.
