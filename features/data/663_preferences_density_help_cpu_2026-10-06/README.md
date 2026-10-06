2026-10-06 CPU Preferences and exact claim-census receipt

Preferences now saves density and gravity at 1%, 10% and 50%, with one-percent
keyboard steps. Its import-failure fallback uses the real 0.01 density floor.
Custom colors use the existing safe Qt color picker. Theme changes update the
Help strip and accessible description without reintroducing popup tooltips.
Obsolete default/order expectations now match the offered field-first/None-last
catalog; paintable selections, Cancel/Save and real picker behavior stay tested.

The affected-preferences log contains 114 passing Preferences, gravity,
Help-strip, safe-picker and crisp-control cases. Its sole failure is the old
claim-count arithmetic checksum; the narrow exact-default-census-test log
records that corrected check passing. No test assertion was removed.

The source-current default census is exactly 919 -> 920: one new matching
mask/watch_normalization_pool literal, with no changed or removed old pair.
All recorded default variants are identical. The exact arithmetic checksum
now names that extra pair; this is an inventory correction, not a coverage,
performance, source-size or mismatch ceiling increase. Portable fresh-module
census script and full before/after inventories are retained here.

This CPU receipt does not establish hosted green, resolve the reported Save
SIGSEGV, or close user aesthetic acceptance. Source-current derived API,
translation, Help and tutorial artifacts remain the workstation lane.
