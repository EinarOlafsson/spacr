# Toxoplasma public route blocked after normal registration

Home source-owner classification decision requested for lessons 24 and 26. Work has not enabled alpha or changed application code to expose a hidden route.

Actual source: a82cc5d110710b1aba24a0e2eafe112fbd87c9ae. Turn 35 (1204c8090) failed opening the legacy standalone Plaque entry; the tested recorder-only a82 fix follows Home → Toxoplasma → the real organism assay tile. Turn 36 then failed opening the parent. Both turns finalized zero clips, stopped before retry/current roles and cleaned their private hosts with no survivors. Raw failures remain preserved.

The real Home constructor now confirms the cause under the normal maturity pass: Toxoplasma starts stable, becomes alpha, disappears from visible_apps and produces zero Home tiles. Home contains exactly 12 tiles: Core 5, Data 3, Tools 4, matching turn 36's actual frame/inventory. All three alpha preferences stayed off. This bounded 4 GB CPU reproduction constructed real make_home_page widgets in a private profile; no MainWindow, app launch, OpenGL, GPU or native recording. The earlier route test used a synthetic Home and therefore missed normal registration.

Exact source seams, unchanged in a82:

- spacr/qt/__init__.py SELF_REGISTERING_MODULES includes spacr.qt.maturity.
- spacr/qt/app.py registers Toxoplasma with stage=None. It has no explicit APP_STAGE entry.
- spacr/qt/maturity.py omits Toxoplasma from the assessment tables; apply() gives unassessed registered apps UNASSESSED_STAGE=alpha.
- app_is_visible()/visible_apps() and make_home_page() consequently omit the organism guide when alpha is off.
- The real Home AppTile still uses moduleAppKey inside HomeTabs; this is not a property/hierarchy selector defect. OrganismScreen's child tiles use organismModuleKey and emit the real assay key.

Please resolve the intended maturity of the navigation-only Toxoplasma guide in normal registration, with a real make_home_page regression after that pass. Preserve the chosen alpha policy; do not force a hidden route from the recorder. Review the actual assay maturity separately for 26 (Invasion currently declares alpha). A parent classification repair is not assay scientific/native acceptance. Work needs an explicit reviewed source pin before fresh whole 24/26 capture.

Receipt copies are byte-identical; their dated hypotheses and prepared-turn 37 fields are preserved historical observations, superseded by this diagnosis/current handoff. No images were copied or prepared.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| lesson24-real-home-diagnostic.json | 708 | 3905737275f9cde47180abe093b71564ac0f077b7f9812f0a1f0e29fe0a92f52 |
| turn35-failure-review.json | 3727 | 3742b30ebf43e4d98569a149c35595df93d8bcd76e9673bf3f9759c139d32586 |
| turn 36-failure-review.json | 4569 | b938e387e5f5baea26adda0974967e05ed6299f13b2275110663c809df31a1c3 |

Lesson 43's standard HF/prepared-hosted-Pages verification/record/docs-features push is complete (96d590cde). Home still needs the coherent 148-file generated 43 checkpoint and native-arrival-audit patch in features/data/676_publication_2026-10-14_43_illumination/. Public Space quota/403 resolution is separate; no live-site update is claimed. Work is preparing fresh PUBLIC20 turn 38, not queued at this handoff. Failed 24 footage is not accepted, and no alpha/app/gate/input changes were made for this packet.
