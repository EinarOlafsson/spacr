# Proposed model identities in settings

N702 preparation is complete as a source-only proposal. Review `design-corrected.txt`, `compatibility-corrected.json`, the structural schema and `fixture-index-final.json`. The schema keeps an additive nested model_metadata value in existing CSV/JSON settings. Exact path bindings, opaque versions and hashes, GUI preservation, overrides, legacy inputs, missing weights and offline metadata are covered; no implementation is delivered.

The 19 bundled rows have no declared release version. The proposal distinguishes an explicitly declared opaque string from an unspecified/null version pinned by key, filename, kind and checksum. Home must approve that policy and the trusted index/cache authorities before implementation. A generated default or filename/cache suffix cannot establish a publisher version.

Four reviewed corrections make metadata remapping field-aware, validate loaded pins before overrides can remove them, preserve declared/unspecified legacy provenance and require complete trusted offline cache identity. Given-local matching bytes remain a distinct recorded-label route. Original design, compatibility, schema, fixtures and manifests are preserved; the schema shape is unchanged.

Independent mechanical review passes 151 checks across 32 original artifacts, 25 A744 source bindings, 51 excerpts and 18 static fixtures (14 JSON, 4 CSV). Source-only branch evidence preserves version "001" and a digit-only SHA as strings. Earlier three-CSV snapshot counts remain historical. Proposed relational failures are not observed production rejections. Semantic and final delta peers report no remaining required fixes. Root independently rechecked all current/candidate source hashes and copied artifacts.

No app/parser/schema-engine tests, model downloads, runtime/native/scientific acceptance, source/catalog changes or hold clearance occurred. Parent N702 implementation remains open.
