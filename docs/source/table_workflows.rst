Combine measurement tables for plots and gates
==============================================

Open **Graph Builder** or **Gate Editor** from Home to explore measurements.
Both screens offer **Merge tables** immediately beside the table picker.
The same merge definition produces the same measurements in either screen.
For the surrounding workflow, see :doc:`workflows`.

Choose a source and an observation level
----------------------------------------

**Load table** opens a SQLite measurement database or a CSV/TSV table.
A physical table is an existing table in that database. A derived table is
a named result calculated from physical tables using a saved merge definition.
Creating a derived table preserves the original measurement tables.

Merging requires one SQLite database. The button is unavailable for a single
CSV/TSV or a Gate Editor session combining several databases. Load the one
database containing the tables you want to join.

In Graph Builder, choosing a table changes the chart's source. Gate Editor
also supports a working set of compatible physical object tables, shown as
removable chips; selecting another compatible table adds its measurements.
The last remaining chip stays in place so the editor always has a source.
A named derived table is selected as a complete source. Use the merge popup
when you need a named, reproducible result or explicit custom relationships.

The **Output observation table** determines what one result row represents.
For example, a cell-level result has one row per cell even when that cell
contains several pathogens. A nucleus count, total pathogen area and cell
intensity can then be compared on the same row.

Create a standard spaCR merge
------------------------------

#. Load your measurement database and press **Merge tables**.
#. Tick the source tables and choose **Output observation table**. The base
   table is included automatically. Standard merges use compatible spaCR
   object tables and cell/cytoplasm observation levels; the list also shows
   other tables whose relationships may require customization.
#. Enter a **Result name**, such as ``Cell and pathogen measurements``.
   It must differ from every physical table name.
#. Review the **spaCR defaults active** summary and the per-column
   **Aggregation** choices. For an ordinary compatible database, no manual
   key mapping is needed.
#. Press **Validate and preview**. Inspect the input and output row counts,
   join keys, unmatched records, missing child keys, repeated child-key rows
   and the first 12 output rows. Validation examines the full input, even
   though only a few result rows are displayed.
#. Press **Create merged table** after validation succeeds. The result is
   selected in the normal table picker. Its columns are available for charts
   and gates.

Changing a table, name or rule invalidates the preview. Validate again before
creating the result. Selecting a different table or base also resets custom
relationships and overrides, so choose your sources before customizing them.

How the defaults combine measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shared spaCR rules aggregate each child table independently before
joining it to the base observations. Adding both pathogens and organelles
does not multiply their rows together. Identity includes the available plate,
well, field and time metadata together with the relevant object/parent ID;
an object number alone is not a globally unique identity.

.. list-table:: Default aggregation for child measurements
   :header-rows: 1
   :widths: 45 55

   * - Measurement
     - Rule
   * - Area, integrated/total intensity and counts
     - Sum the child values.
   * - Minimum intensity
     - Retain the minimum child value.
   * - Maximum intensity
     - Retain the maximum child value.
   * - Median measurements
     - Take the median of the child measurements.
   * - Other numeric measurements, including means
     - Take the mean of the child measurements.
   * - Text and identifiers
     - Retain the first nonmissing value in source order by default;
       identifiers cannot be averaged or summed.

For four pathogens with areas 1, 2, 4 and 8, total pathogen area is 15.
A mean intensity is the mean of the four object measurements, not an
intensity recalculated from all their pooled pixels. Inspect the rule shown
for each column when its name does not describe its measurement convention.

The default nucleus join retains only matching base observations. Pathogen
and organelle joins retain uninfected cells. Missing measurements remain
missing, including a sum for which every contributing value is missing.
An absent intensity measurement is not converted to a measured zero.
Child columns use table-qualified names where needed to distinguish them.

Available overrides depend on column values and role. Numeric measurements
offer mean, median, sum, minimum, maximum, count and other compatible choices;
text supports choices such as first/last, count and distinct count;
Boolean-valued data also support **any** and **all**. **count** counts
nonmissing values. Default one-to-one rows attach intact, so their aggregation
controls are disabled until custom merging is selected.

Customize an alternative schema
--------------------------------

Choose **Customize merging** when the database uses different table names,
identifiers or deliberately different relationships. Choose the base and
child tables in the main popup first. Mapping their actual columns does not
rename or modify the source tables.

The warning explains that incorrect relationships can link unrelated objects,
duplicate or omit observations, and change measurements used in plots and
gates. Read it and check **I understand the warning and will check the merge
preview.** before accepting the custom settings. Acknowledgment does not
bypass validation.

* **Base observation keys** identify one unique, nonmissing base row. Include
  field and time columns when IDs repeat across images or timepoints.
* Each child row in the configuration specifies **Base keys** and **Child
  keys**. Enter comma-separated column names in matching order. For example,
  base ``image_id, cell_id`` can map to child ``image_id, parent_cell_id``.
* Choose **one-to-one** when at most one child row belongs to each base key,
  or **one-to-many** when several child rows must first be aggregated.
* A **left** join keeps unmatched base observations with missing child
  measurements. An **inner** join retains only matching observations.
  Child rows with missing join keys do not match a base row.
* Use **Identifiers** to identify additional child columns that must not be
  treated as continuous measurements. Review the per-column aggregation
  controls in the main popup after accepting the mapping.

The popup now says **Custom rules active**. Validate and inspect the preview
before creating the result. Duplicate base identities, contradictory
one-to-one mappings, missing keys and incompatible operations produce an
error. Repeated child keys can be expected in a one-to-many relationship;
they require investigation when the relationship is supposed to be one-to-one.

**Cancel** leaves the previous configuration intact. **Reset to spaCR
defaults** removes custom relationships and overrides. Changing sources or
the output table also rebuilds the defaults.

Use, save and reload the result
--------------------------------

In Graph Builder, place result columns on **X**, **Y**, colour, size or facet
channels. In Gate Editor, use the columns for thresholds, polygons and other
gate shapes. The Gate Editor may display a sample according to its
``sample_fraction`` and ``max_points`` settings; gate export evaluates the
full merged table.

The named definition is stored beside its database. For ``measurements.db``,
the file is ``measurements.db.spacr-merges.json``. Refreshing or reopening
the source makes the result available again and recalculates it from the
source. This is a reproducible recipe, not a frozen copy of all result rows.
Reuse checks the source path and selected tables' schemas. A different
database or changed schema requires a newly reviewed mapping; definitions
are not silently applied to unrelated sources.

**Save chart** in Graph Builder saves chart channels, the source and the full
merge definition to JSON. **Load chart** reconstructs and validates the
table before plotting. **Save gates…** also embeds the merge definition;
load the original database before using **Load gates…**. These saved files
can reconstruct the merge even if its sidecar is missing. They still require
the original source data.

**Export gates…** writes gate membership columns to the database's
``filters`` table, preserving the base object's identity. This explicit
export is a database write; constructing and previewing a merge does not
write into the source measurement tables. **Save graph** exports the Gate
Editor figure as PNG or PDF, separately from saving its gating strategy.

A custom result without verified spaCR image/object provenance remains
usable for plotting and tabular gating. Image navigation, image annotation
and gate export to the measurement database are unavailable for that result,
with an explanation in the interface. Keep the saved chart or gating
strategy for tabular reuse; column names alone cannot establish links to
microscopy objects.

Annotate experimental conditions in Graph Builder
--------------------------------------------------

**Annotate conditions** adds a condition column to the working table in
Graph Builder. It works with a physical database table, a named derived
table or an imported CSV/TSV. These are experimental group labels for
tabular analysis; they do not require image links or write image annotations.

#. Open **Annotate conditions** and set **Output column**. The default is
   ``condition``; choose another name, such as ``treatment_group``, if the
   source already has that column. Existing source columns cannot be replaced.
#. Press **Add condition** for each group and give each box a distinct,
   nonempty **Condition name**.
#. In a box, choose the metadata **Column** to match, then enter an **Include**
   regular expression, an **Exclude** expression, or both. For manual-only
   grouping, leave Include blank and drop source rows into the box.
#. Inspect **Preview condition** beside the source columns, each box's
   matching/manual counts, and the assigned, unmatched and overlapping totals.
   **Preview assignments** reruns this check; edits also refresh the preview.
#. Resolve invalid patterns and overlapping conditions, then choose
   **Apply conditions**. The output column becomes available for chart
   channels and filtering. **Cancel** discards the dialog's draft changes.

Rules and manual selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

Patterns search the chosen column's displayed string values. They are
case-sensitive unless the expression includes a flag such as ``(?i)``.
Use anchors when the whole value must match: ``^A0[1-3]$`` selects wells
``A01``, ``A02`` and ``A03``. ``.*`` includes all nonmissing values in the
chosen column. A blank Include expression selects no rows automatically.

Manual rows are added to the rows matched by Include. Exclude then removes
matches from both sets, including rows you dropped manually. Missing
metadata values do not match a regex; they can still be assigned manually.

Select multiple rows in the source table and drag them into a condition
box. Use Shift for a range and Ctrl/Cmd for individual rows. Sorting by a
column or using **Filter source metadata or preview conditions** changes
the visible view, while assignments still refer to the original source
rows. The filter box is a case-insensitive text search, separate from the
condition regex rules. Dropping a row twice into the same box does not
duplicate its assignment.

Use **Remove selected manual rows** or **Clear manual rows** to remove
dropped memberships. An Include rule can still match those rows; change
the rule or add an Exclude expression when they must leave the condition.
**Remove condition** removes the whole box and its rules.

A row may belong to only one condition when applying. If two boxes match it,
the preview shows the competing labels and **Apply conditions** stays
disabled. Adjust the include/exclude patterns or manual assignments until
the overlap count is zero. No condition silently wins. Unmatched rows remain
blank in the output column and remain present in the table.

Keep conditions with the analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reopen **Annotate conditions** to edit the current assignments. They survive
table refreshes in the current Graph Builder session when the source is
unchanged. Use **Save chart** to preserve the rules, manual assignments,
output-column name and any merge definition with the chart. **Load chart**
reconstructs the source and checks its identity before applying those labels.

Saved conditions belong to the exact source table, including its row order,
values and schema. Changed source data or a different merge definition
requires reviewing and recreating the conditions. Sorting the dialog's view
does not change that source identity. This check prevents saved manual
selections from labelling unrelated rows after a source changes.

For a SQLite source, **Save annotated table…** saves the working values as a
new physical table in the same database. The prompt **New table name
(existing tables are preserved)** proposes the current table name followed
by ``_annotated``. Existing tables, views and saved merged-table names cannot
be overwritten; names beginning with ``sqlite_`` or ``_spacr_`` are reserved.
This action adds a table and its annotation provenance while preserving the
original measurement tables.

The saved table is selected in the picker and can be loaded in other table
screens, including Gate Editor. It contains a snapshot of the annotated
values, unlike a derived merge that is recalculated from its inputs.
Reopening it in Graph Builder restores editable condition rules when the
saved rows and schema are unchanged. Save revised values under a new table
name. Changed saved data are still readable, but old editable rules are not
silently restored onto them. Image-specific actions continue to require
appropriate object provenance.

**Export table…** writes the working table, including the condition column,
to a new CSV and records its source, merge definition and condition rules in
``<export>.csv.conditions.json``. It exports the working table rather than
just a brushed selection. The original database or imported file is
preserved, and exporting over that source path is refused. Reopening the
CSV gives you its saved labels; use the saved chart to reopen the original
source and editable condition rules.
