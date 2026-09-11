Lineage — containment, not ancestry
==================================

Open Database Browser from Help, then its Lineage submodule. Browse to the
measurement database from the downloaded Annotate example. The separately
linked Graph_Builder_real_measurements.zip contains the exact same measurement
database used for this lesson; it does NOT contain the corresponding images.
For the crop demonstration, use Annotate's complete downloaded example with
matching database identities and image files. Do not mix a database with crops
from a different measurement run merely because their numeric labels match.

This view reads stored containment relationships. It does not infer cell
division, genetic ancestry, infection state or segmentation accuracy.
The complete source has 2,341 cells, 2,682 nuclei and 2,178 objects in the table
named pathogen. All child records have existing parents. No cell is childless,
but 1,168 cells have no pathogen-table child: a nucleus also counts as a child.

Only 2,000 of 2,341 parent families appear in the tree: 6,087 displayed nodes,
341 omitted parent families. The full hierarchy has 7,201 nodes. Sorting the
header changes order, not containment or the display cap. Numeric labels repeat
between fields and object types; the full typed key is the identity.

Selecting one child is different from Select with contents, which expands the
family. Selecting a parent and one of its children does not require a duplicate
copy of the child. These actions publish object identities; they do not prove
that a matching crop exists. The key column is hidden in this recorded UI.

IMPORTANT — unresolved application defect, not a repaired workflow
-----------------------------------------------------------------
Open crops expands the selected family. With this cell-only crop dataset, the
request for cell3, nucleus5 and pathogen1 in plate1/r12/c1/f1 incorrectly shows
cell3, cell5 and cell1 crops. The latter two are unrelated cells, NOT images of
the requested nucleus and pathogen. The tutorial does not approve this fallback
or recommend using it to label children. The original hold remains preserved.

The verified counterpart is double-clicking the parent cell alone. With
Annotate open and its matching source loaded, it requests only the typed parent
key and displays exactly that cell crop. Shift-click enlarges its existing
preview without assigning a class; Escape returns to the grid. Enlarging the
preview does not create additional image detail. No labels are edited here.

What this archive contains
--------------------------
lineage_api_export.csv: the real 7,201-row output of
spacr.lineage.lineage_frame, independently checked against the source database.
review.json: recorded scope, source hashes, hierarchy counts and crop warning.

This CSV was written through the Python API, NOT a Lineage export button.
The Database Browser controls surrounding the nested view belong to the host;
they are not a claim that Lineage has its own graph or hierarchy export button.
The CSV is for inspection, not a replacement measurement database to browse.

Python uses spacr.lineage.read_object_tables, build_forest and lineage_frame
for the same containment model. A CSV preserves the fields, parent keys, types,
depths and child counts; it does not include the image pixels or validate the
biological interpretation. No new segmentation, tracking, model training or AI
request was performed, and original data and crops remain unchanged.
