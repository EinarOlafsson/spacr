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

Recover original filenames before labelling conditions
-------------------------------------------------------

If images were renamed to Yokogawa format, use **Merge original filenames…**
inside **Merge tables** to recover the names recorded during conversion.
Select ``conversion_map.csv``, a legacy ``rename_log.csv``, a
``channel_sorting_manifest.csv``, or a SQLite database containing the
``conversion_map`` table. The file picker starts near a known mapping when
one is found beside the measurement database or in its parent folders.
The mapping must contain recorded source-to-output relationships; this action
does not infer treatment names from renamed images.

For filename metadata alone, select just the table you want to annotate.
This retains every row and existing column, including tables without spaCR
object IDs. For an existing merged result, open its merge popup to preserve
the reviewed relationships and aggregation while adding the filename mapping.
With several source tables selected, measurements are merged first and the
original names are attached to the resulting observations.

#. Press **Merge original filenames…** and select the mapping.
#. Set a new **Result name**, then press **Validate and preview**.
#. Inspect the matched and unmatched row counts and the preview's
   ``original_filename`` and ``original_path`` columns.
#. Press **Create merged table**, then open **Annotate conditions** and choose
   ``original_filename`` as a rule's metadata column.

Several channels or z-planes belonging to one field produce a sorted,
deduplicated list separated by a semicolon and a space in the new columns. They do not multiply
the table's observations. Plate and timepoint identities remain distinct;
ambiguous or conflicting identities cause validation to fail. Unmatched
names remain blank. Existing ``original_filename`` or ``original_path``
columns are preserved by refusing to overwrite them.

``original_path`` is the path recorded in the mapping; the original files
need not still exist at that location. Image files and source measurement
tables are never renamed or rewritten by this action. Adding filename text
does not establish missing microscopy object provenance.

The saved merge records the mapping's location and a content checksum.
Reopening a merge or chart requires that mapping to remain available and
unchanged. For an embedded ``conversion_map`` table, unrelated changes to
other database tables do not invalidate the mapping. If the mapping changes,
select it again and validate a new result before applying it. **Remove
filename mapping** removes the enrichment from the draft without deleting
the file. **Reset to spaCR defaults** also clears the mapping.

Annotate experimental conditions in Graph Builder
--------------------------------------------------

**Annotate conditions** adds annotation columns to the working table in
Graph Builder. It works with a physical database table, a named derived
table or an imported CSV/TSV. Use rules to assign labels or extract variable
metadata, then compose a final column from those values. Every generated
column is retained. These table
operations do not require image links or write image annotations.

#. Open **Annotate conditions**, name the **Output column**, and choose
   **Assign values**, **Extract text**, or **Compose column**. Existing source
   columns cannot be replaced.
#. Define the value rules, text extraction, or composition for that output.
   Use **Add column** for another output. A named-group regex can also create
   several metadata columns together.
#. Inspect all generated values beside the source columns. **Preview
   assignments** reruns the check; edits also refresh the preview.
#. Resolve invalid rules, missing dependencies and conflicting assignments,
   then choose **Apply conditions**. Every generated column becomes available
   for chart channels and filtering. **Cancel** discards the draft changes.

Extract several metadata columns from a filename
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose **Extract text** and select the source column, such as
``original_filename``. A regular expression extracts the matching text into
the output column. Select a named or numbered capture group to choose which
part of the match becomes the value.

For example, this pattern describes ``HeLa_rep2_24h_DMSO.tif``:

.. code-block:: text

   ^(?P<cell_type>[^_]+)_(?P<replicate>rep\d+)_(?P<timepoint>\d+h)_(?P<drug>.+)\.tif$

The named groups describe four output columns: ``cell_type``, ``replicate``,
``timepoint`` and ``drug``. **Create columns from named groups** creates these
outputs together. The example row produces ``HeLa``, ``rep2``, ``24h`` and
``DMSO``. Another matching filename produces its own values; they are not
fixed labels copied from the example. Adapt the pattern to your own filenames
and inspect the preview before applying it.

Extraction uses the first regex match. Nonmatching rows and missing or empty
captures remain blank. Invalid patterns or capture groups stop the preview
with an error. Existing source column names cannot be overwritten.

Assign values with readable rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose **Assign values** when matching rows should receive a value you type,
such as ``WildType`` or ``control``. Each rule selects a column, an operator
and a comparison value. Operators include contains, does not contain, equals,
does not equal, starts with, ends with and regex matching. Text operators
interpret punctuation literally; regex operators interpret it as a pattern.
Matching is case-sensitive. Contains, prefix, suffix and regex criteria need
nonempty text. Use an explicit ``.*`` regex to match all nonmissing text, or
equals with an empty value to match an actual empty string.

Choose **all rules match** when every criterion must match, or **any rule
matches** when at least one must match. Criteria may inspect different source columns or earlier generated
columns. For example, assign ``control`` when ``drug`` equals ``DMSO`` and
``cell_type`` equals ``HeLa``. Missing source values do not satisfy a negative
criterion. An actual empty string is a value and can be matched explicitly.

Several rules may assign the same value without a conflict. Different values
assigned to one row in the same output column require resolution before Apply.

Compose the final column
~~~~~~~~~~~~~~~~~~~~~~~~

Add an output column, name it ``condition`` or another name you choose, and
select **Compose column**. Drag available columns into the composition field.
Drag its tokens to change their order. Insert fixed text to add separators,
prefixes or words between the values.

For example, the tokens ``cell_type``, ``_``, ``replicate``, ``_``,
``timepoint``, ``_``, ``drug`` produce ``HeLa_rep2_24h_DMSO`` for the example
above. Reordering the column tokens changes the result without changing their
extraction rules. Fixed text is literal; it cannot execute code.

A composition can reference original source columns and earlier generated
columns. A missing or empty column value leaves the composed value blank.
The preview shows all intermediate columns and the final column together.
Applying, exporting or saving a new SQLite table retains every generated
column and the editable recipe. The original measurements remain unchanged.

Build genotype, replicate and a combined condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Name the first **Output column** ``genotype`` and keep its mode set to
**Assign values**. Use **Add condition** for its labels. Press **Add column** to
create ``replicate`` with its own boxes. In each box, choose **Exact values**
to enter lists of metadata values without writing a regular expression.
For example:

.. list-table:: Example annotation rules
   :header-rows: 1
   :widths: 18 18 32 32

   * - Output column
     - Metadata column
     - Included values
     - Assigned label
   * - ``genotype``
     - ``columnID``
     - ``c1,c2,c3``
     - ``WildType``
   * - ``genotype``
     - ``columnID``
     - ``c7,c4,c5,c6``
     - ``mutant``
   * - ``replicate``
     - ``rowID``
     - ``r1,r4,r5,r6``
     - ``replicate 1``
   * - ``replicate``
     - ``rowID``
     - ``r7,r9,r10``
     - ``replicate 1``

Repeated entries such as ``c1,c1`` or ``r7,r7`` have no extra effect.
Exact matching keeps ``c1`` separate from ``c10``. Two rules that assign
the same label may select the same row; two different labels in one output
column require a correction before applying. Each output column has its own
assignments, so a row can have both a genotype and a replicate.

Press **Add column**, name it ``condition``, and choose **Compose column**.
Drag ``genotype`` into the composition field, insert a text token containing
``_``, then drag in ``replicate``. The token order determines the joining order.
A row labelled ``WildType`` and ``replicate 1`` then receives
``WildType_replicate 1``. The label's space is retained. Column names, labels,
component order and separator are editable; the example does not prescribe
the names for your experiment.

Combination rules can use source columns or previously created annotation
columns. Keep dependencies earlier in the column order. Missing components
leave the combined value blank; they do not produce text such as ``nan``.
Preview all generated columns before applying. Saving a chart, exporting the
table or saving a new annotated SQLite table retains the complete recipe.

Rules and manual selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

Above the condition boxes, **Examples for column** offers copyable patterns
based on the selected column's values. Choose an example type, inspect or
edit its text, press **Copy regex**, then paste into a condition's **Include**
or **Exclude** field. Copying an example does not change the condition rules.
Literal punctuation in the sampled values is escaped automatically.

For original names such as ``drug_A_rep1.tif`` and ``control_rep1.tif``:

* ``drug_A|control`` matches either text.
* ``drug_A_rep.`` allows one character after ``rep``.
* ``drug_A.*rep`` allows any intervening text, including none.
* ``^drug_A_rep1\.tif$`` matches that entire value.
* ``(?i)drug_a`` ignores letter case.

Use Include to select the group and Exclude to remove exceptions, such as
Include ``drug_A`` with Exclude ``failed|outlier``. A filename column containing
several original names separated by semicolons is searched as one string, so a
contains pattern can match any recorded name; a whole-value anchor applies
to the entire combined string.

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

A row may receive only one distinct label within each annotation column.
If boxes assign different labels to it, the preview shows those competing
labels and **Apply conditions** stays
disabled. Adjust the include/exclude patterns or manual assignments until
the overlap count is zero. No condition silently wins. Unmatched rows remain
blank in the output column and remain present in the table.

Keep conditions with the analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reopen **Annotate conditions** to edit the current assignments. They survive
table refreshes in the current Graph Builder session when the source is
unchanged. Use **Save chart** to preserve the rules, manual assignments,
output-column definitions and any merge definition with the chart. **Load chart**
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

**Export table…** writes the working table, including every annotation column,
to a new CSV and records its source, merge definition and condition rules in
``<export>.csv.conditions.json``. It exports the working table rather than
just a brushed selection. The original database or imported file is
preserved, and exporting over that source path is refused. Reopening the
CSV gives you its saved labels; use the saved chart to reopen the original
source and editable condition rules.
