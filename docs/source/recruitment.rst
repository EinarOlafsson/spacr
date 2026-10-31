Recruitment: compartment ratios and channel identity
========================================================

From **Home** → **Organism** → **Toxoplasma**, open **Recruitment** after segmentation
and measurement. Supply the project containing
``measurements/measurements.db`` and verify that its cell, nucleus, pathogen
and cytoplasm measurements refer to the intended objects and channels.
See :ref:`Recruitment in the module map <workflow-module-recruitment>` and
:func:`spacr.submodules.analyze_recruitment` for the surrounding workflow.

For whole-vacuole marker states, explicit host infection denominators and
optional parasite counts, see :doc:`Host–Pathogen Analysis <host_pathogen>`.
The two modules remain separate analysis choices.

Select ``channel_of_interest`` for the fluorescent marker whose recruitment
you want to measure. Each row is one host cell: the mean intensities of the
pathogen objects assigned to that cell are averaged, and the primary
``recruitment`` column divides that pathogen mean by the cell's cytoplasm mean
intensity in the same channel. Configure
the condition metadata, object size/intensity filters and minimum objects per
well before comparing results. Inspect compartment identities and denominator
intensities; an absent or zero denominator does not establish recruitment.

Auxiliary ratios retain their channel
----------------------------------------

``channel_dims`` selects image-channel indices for overlays and auxiliary
recruitment measurements. The auxiliary calculation also includes
``channel_of_interest`` if it was omitted from that list. Output names follow
this pattern::

   pathogen_channel_<channel>_<compartment>_<statistic>_ratio

``compartment`` is ``cell``, ``cytoplasm`` or ``nucleus``. Each denominator is
the mean intensity of that compartment in the same channel. ``statistic``
identifies the pathogen numerator:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Statistic
     - Measurement used as numerator
   * - ``mean``
     - ``pathogen_channel_<channel>_mean_intensity``
   * - ``q75``
     - ``pathogen_channel_<channel>_percentile_75``
   * - ``outside_mean``
     - ``pathogen_channel_<channel>_outside_mean``
   * - ``outside_q75``
     - ``pathogen_channel_<channel>_outside_percentile_75``
   * - ``periphery_mean``
     - ``pathogen_channel_<channel>_periphery_mean``

For example, a channel 2 pathogen mean of 12 and cytoplasm mean of 4 produces
``pathogen_channel_2_cytoplasm_mean_ratio = 3``. If channel 3 has means of 100
and 4, its separate column is
``pathogen_channel_3_cytoplasm_mean_ratio = 25``. Calculating channel 3 retains
the channel 2 result. These are illustrative intensities, not experiment data.

The calculation produces fifteen auxiliary ratios per channel. It does not
create constant-one pathogen/nucleus slope columns; these ratios do not
estimate a spatial slope. Genuine slope columns already supplied by the
caller are preserved. When comparing historical exports, check whether their
auxiliary column names identify a channel before combining runs.

Saved results
-------------

The run writes ``results/cells.csv`` with the retained per-host-cell rows and
``results/wells.csv`` with per-well means of those rows, alongside
recruitment plots. Rows in ``cells.csv`` combine all pathogen objects assigned
to a host cell; they are not independent per-vacuole measurements. For
measurements linked to individual vacuoles, use
:doc:`Host–Pathogen Analysis <host_pathogen>`. Verify condition assignments,
filtering and the selected channel when comparing these two aggregation
levels.
