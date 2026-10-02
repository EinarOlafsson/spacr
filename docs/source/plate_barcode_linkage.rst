Link plate barcodes to sample metadata
==========================================

In **Measure**, enable **Show alpha features** in Preferences and open
**Plate Barcode Linkage (Alpha)**. Set ``plate_barcode_source`` to a table
or a LIMS service URL. The linkage runs after measurement.

A local CSV, TSV, Excel or Parquet table needs a barcode column, a well
column and the sample metadata to attach. For example::

   barcode,well,strain,compound,concentration
   BC001,A01,RH,DMSO,0
   BC001,A02,RH,pyrimethamine,2

Set ``plate_barcode_column`` if your barcode column has another name.
``plate_barcodes`` maps an imaged plate name to its barcode. Otherwise,
spaCR reads ``barcode.txt`` in the plate folder or uses the plate name.
Retain meaningful concentration units in your metadata.

Link during Convert
-------------------

In **Convert**, enable **Show alpha features** and choose a local sample-records
CSV in **Plate barcode linkage (Alpha)**. The CSV needs a barcode column, a
well column and whichever sample fields you want in the plate map. Preview
validates the linkage before writing images. Convert uses the output plate
names shown in its preview; these may differ from source folder names.

Assign output plate barcodes in the form ``plate1=BC001; plate2=BC002``,
or put a UTF-8 ``barcode.txt`` at the acquisition root or in an actual source
plate folder. A file containing one plain barcode applies only when that
folder holds one source plate. For several plates, write one
``source_plate=barcode`` entry per plate, separated by lines or semicolons;
those keys are the *original source* plate names. Explicit assignments in
Convert use the *output* plate names and can fill any remaining plates. spaCR
keeps leading zeroes, rejects ambiguous or disagreeing assignments, and
requires a barcode for every output plate.

After a complete conversion, read ``plate_barcode_linkage/plate_map_lims.csv``
in the destination. Review ``plate_barcode_mismatches.csv`` there for unmatched
wells and other disagreements. The ``complete.json`` receipt is written last;
its absence means the linkage bundle is incomplete. The Convert screen shows
a bounded preview of mismatches and the paths to both CSV files. Original
images, barcode files and existing user plate maps are left unchanged. This
Convert path accepts a local CSV; use Measure for an HTTP LIMS service.

Read records from a LIMS service
------------------------------------

Use a URL such as ``https://lims.example/plates/{barcode}/wells``.
spaCR substitutes the encoded barcode. Without ``{barcode}``, it adds
a ``barcode`` query parameter. If authentication is needed, set the
token in an environment variable and put that variable's name in
``plate_barcode_token_env``. Do not put the token in the URL or settings.

The service may return a list of well records, or an object containing
that list under ``wells``, ``records``, ``results``, ``data`` or ``items``.
Plain fields outside the list supply metadata defaults for its wells::

   {
     "strain": "RH",
     "wells": [{"well": "A01", "compound": "DMSO"}],
     "next": "?page=2"
   }

spaCR follows ``next``, ``next_url``, ``@odata.nextLink`` or ``links.next``.
The link may be a URL string or an object with an ``href`` field. Relative
links resolve against the response URL. A missing link, null or empty
string ends the list. Metadata defaults carry forward to later pages;
explicit fields on a well override those defaults. Pagination fields do
not become sample metadata.

Every link and HTTP redirect must stay on the same scheme, host and port
as the original service. Conflicting links and repeated pages are errors.
Collection is limited to 16 MiB per response, 64 MiB across the lookup,
1,000 page addresses per barcode and 100,000 records across the lookup.
An error stops the metadata lookup without returning a partial plate map.
Existing measurement results remain available. Vendor-specific cursor
protocols that do not provide a next-page URL need a separate adapter.

Inspect the result
----------------------

The filled map is written to ``measurements/plate_map_lims.csv``.
It fills an empty ``profiling_metadata`` setting. When compound and
concentration columns exist, it also fills an empty ``viability_plate_map``.
An existing user-supplied map is compared, not silently replaced.

Review ``measurements/plate_barcode_mismatches.csv`` and the log for missing
barcodes, shared barcodes, unmatched wells, conflicting records and
disagreements with an existing map. Resolve those mismatches before
interpreting treatment comparisons.
