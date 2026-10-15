# Native organism assay navigation repair

Lesson24 turn35 ran20:32:33–20:32:55EDT and stopped at its opening action:
the recorder sought a standalone Home tile for analyze_plaques. Current public
UI uses Home → Toxoplasma guide → Plaque Assay tile. No finalized clip was
accepted. App exit0, failed-turn exit1, cleanup survivors empty; no retry or
current-role recording began. Original failed output remains preserved.

Held local commit a82cc5d110710b1aba24a0e2eafe112fbd87c9ae changes only
tools/tutorials/capture_ui.py and its focused test. The live helper clicks the
actual guide, scrolls to and clicks its unique visible enabled organismModuleKey
tile, and waits for the actual assay screen. No off-camera slot call, alpha
enablement, app/input/gate change, prewarm, environment or package modification.

All15 focused CPU tests pass in1.08s, including real OrganismScreen/AppTile
signals, all5assays, unavailable tiles, alpha-off and unchanged Mask/Measure
routes. Independent source review passes. These are route tests, not native
capture acceptance. Root will record the whole lesson afresh on the clean pin.

The patch is exact git format-patch --stdout --no-signature --binary
--full-index -1 a82cc5d11 output. Source remains held; this docs/features review
artifact does not integrate it. Home owns integration. The lesson43 checkpoint
and one-line publication-arrival audit import remain requested separately.
