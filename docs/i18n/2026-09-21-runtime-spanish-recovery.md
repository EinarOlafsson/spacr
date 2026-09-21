# Spanish runtime recovery — 21 September 2026

The post-reboot recovery was reviewed and audited against `3d9eb57c5`, plus
the changes in this checkpoint. No model generation or GPU inference was used.

The three recovered UI review files contain 227 distinct sources: 76, 80 and
71 records. Direct comparison with the preserved machine drafts gives 161
corrected translations and 66 reviewed retentions. These records extend the
39 scientific-setting and 42 UI/category records already published. Three
OPS descriptions already reviewed as categories are also reused in the UI.

Three additional action captions now read Vectorizar, Filtrar and Probabilidad
celular. Help text follows Vectorizar and the existing inline button label
Registrar como incidencia. Protected commands, paths, model identifiers and
placeholders are retained. The nine Import captions and two synthetic Invasion
captions have separate source-bound review files. Invasion's wording explicitly
identifies the data as synthetic and distinguishes drawing from imaging and
segmentation.

Removed 42 catalog entries absent from canonical English: two labels, two
tooltips, two category descriptions and 36 UI strings. Review history was not
deleted. The English manifest now includes the two new Invasion sources.

Validation, freshly rerun under a 4-GiB hard memory cap:

```
tools/build_i18n_catalogs.py --audit --languages es
verified external runtime catalogs: languages=1 settings=985 categories=194 ui=3824 modules=68
```

All 642 distinct reviewed Spanish sources bind to current English, preserve
source hashes and pass the existing syntax, semantic and target-language
gates. This is a Spanish runtime result, not a green API audit or docs job.
The other eight runtime languages still owe these eleven new captions, and
the API translation/catalog debt remains open.
