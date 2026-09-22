# Numbered organelle background switches

The background switches introduced in `ee6a608e6` now have source-bound
labels and scientific help in all nine runtime languages. The 72 reviewed
records cover the four materialized slots, including corrections to the
primary switch's previously poor wording. Each tooltip preserves its own
background-floor identifier, the False default, and the warning that clipping
can erase real signal and bias subsequent intensity measurements.

The generator recognizes `remove_background_<organelle role>` alongside its
existing prefix-style organelle keys. Slots above four reuse the reviewed
slot-two wording only when the supplied English exactly matches that numbered
template and its translation source hash is current. Slot identifiers and
numbers are substituted explicitly. Stale English, missing templates and
stale translation hashes return no translation through the normal fallback.
This keeps all 702 slots translated without storing 701 near-duplicate pairs
per language or changing the application's analytical behavior.

The integrated German runtime audit now passes with 985 settings, 194
categories, 3,796 UI strings and 68 module summaries. The new runtime checks
exercise all nine locales at slots 1–5, 26, 27 and 702. Twenty-two numbered-slot
and existing dynamic-slot checks pass; the remaining 21 selected catalog and
review contracts also passed. Swedish/French record-count changes are proved
by subtracting the exact new eight-source sets; earlier count evidence is
retained. The complete multilingual catalog repair and API work remain open.
