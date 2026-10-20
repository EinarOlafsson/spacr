# Source-bound translation backlog during the release freeze

Existing runtime catalogs have 10,455 source-current records per locale, with
no missing/stale/orphaned keys or blank values. Standalone runtime review
metadata is current. This proves metadata freshness, not translation quality.

The API inventory has 13,418 English symbols. Each of nine locale catalogs
has 20 missing and 20 stale existing symbols: 180 missing plus 180 stale
locale-symbol pairs. Sixteen context mismatches per locale are a subset of
those 20 stale symbols, not extra deficits. Three active reviewed API blocks
per locale also have source/context mismatches, for 27 total. Exact IDs,
current/historical hashes and affected files are in the bound JSON.

Four Mask Live arrivals are missing, including the existing Run-start collect
hook; no extra arrival contract is invented. Mixed-model blocks 6/8/9/10 need
current evidence. Item 701 retains its four current UI records and three API
review blocks in every locale. The English API manifest is source-current.

The Dynamic animation tooltip is source-current but semantically obsolete:
visible backdrops are capped at up to 30 fps during CPU work, while Spaceout
fractals can still pause. This is a cap, not measured frame-rate acceptance.
The AmbientWidget.stop docstring also has obsolete CPU-pause wording and is
catalog-stale. Disabling Dynamic animation does not remove every pacing limit.
Proposed precise English is advisory for Home, not applied to frozen source.

The unreleased 1.5.1.5 note has 31 English bullets and no matching localized
version document in the nine-locale inventory. Its unconditional no-freeze
wording and broad translation claim need final English truth review. No
1.5.1.5 released-news metadata was fabricated. Guide-message coverage remains
unmeasured without current gettext extraction; no Sphinx build was run.

Root independently rehashed all 4,306 static source/catalog/review bindings.
The coordination handoff is bound separately to immutable Git snapshots.
No source, catalogs, reviewed records, audio or translations were changed.
This evidence supplies the deferred queue and lightweight Home task choices;
it does not move translations ahead of accepted fresh lesson 20 clips.
