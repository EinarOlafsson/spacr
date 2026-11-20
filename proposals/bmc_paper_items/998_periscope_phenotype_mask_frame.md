# Item spec 998: PERISCOPE phenotype measurement in the SBS mask frame

- **Lifecycle:** future (alpha), behind "Show alpha features" (off by default for non-cpg0021 layouts). Registered with the 919 frame settings.
- **Number:** 998. Recheck `features/feature_list/` and the claims files before landing.
- **Rule:** a spaCR settings-file rerun gives IDENTICAL output (same shifts, same features). Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly plus the 919 fix.
- **Dependency:** item 919 frame fix, branch `claude-919fix-1119` @ `118d0d8240` (lands in train 5). It provides `spacr.cpg_cropped.shift_image`, `periscope_image_csv` and `painting_to_sbs_shift` ('periscope_logged' | 'register'). 998 reuses them and adds no second shift implementation.
- **BUILD GATE:** do not build before 919 has landed on nightly.
- **Paper-critical:** yes. Every PERISCOPE (cpg0021, CP186A/B) phenotype or feature result in the BMC paper depends on it.
- **Readiness:** "PERISCOPE phenotype/feature measurement on cpg0021 (painting frame vs SBS-frame masks)" is Under development (row in `proposals/READINESS.md`). Until 998 is Ready, every spaCR PERISCOPE phenotype or feature result on cpg0021 is INVALID unless the recorded Align shift was applied.

## Why
On cpg0021 (PERISCOPE CP186A/B) the corrected/cropped painting images (CorrDNA and the other painting channels) are in the PAINTING frame. PERISCOPE's own Cells and Nuclei masks are in the SBS frame. The two frames differ by the shift PERISCOPE's 04Align module recorded per site (`Align_Yshift_DAPI_Painting`, `Align_Xshift_DAPI_Painting` in `Image.csv`; 3-77 px on CP186A Well 1).

Any measurement of painting features with those masks must therefore first shift the painting images by the recorded Align shift (or the masks by the inverse). Measuring unshifted painting images through SBS-frame masks reads the wrong pixels and gives plausible-looking but wrong features.

Evidence (all from `reanalysis/cellcycle_transfer_1119/`, card section 12, tables `dna_check_*`):
- Phase correlation of the painting DNA image on the SBS DAPI image equals minus the logged Align shift on 75/75 sites.
- Nucleus/background DNA ratio: 1.43 unshifted, 17.0 after the shift.
- Per-nucleus DNA against our CellProfiler 4.1.3 rerun of PERISCOPE's own pipeline: r = 0.39 unshifted, 1.0000 after the shift (22,025 nuclei).
- G1/G2 peaks return on 10/10 sites after the shift.

## What to build
1. **Setting `cpg_painting_frame_shift`** (alpha group, next to the 919 frame settings):
   - `'periscope_logged'` (default for cpg0021 painting reads): shift from the site's `Image.csv` (or `cpg_align_table`), through `painting_to_sbs_shift(source='periscope_logged')`.
   - `'register'`: spaCR's own phase correlation (`source='register'`).
   - `'none'`: no shift; allowed only for masks that are in the painting frame (spaCR's own segmentation on the painting crops). Explicit, never implied.
2. **Apply:** in the Measure read of cpg0021 painting channels, every painting plane of a site is moved with `shift_image(plane, dy, dx)` (whole pixels, zero fill, no wrap) before intensity, texture, granularity and other features are computed with the SBS-frame masks. The sign convention is fixed by the identity test below (the shift that makes phase correlation return zero), not by this text.
3. **Refuse a site with no record:** under `'periscope_logged'` a site with no Align row, or non-finite values, raises with the site named. It is never given a zero shift. Under `'register'` a registration that fails the 919 acceptance raises likewise.
4. **Record the applied shift per site:** `painting_frame_shift_<plate>_<well>.csv` (plate, well, site, mode, dy, dx, source, border zero-fill width) in the run folder, and the setting is stamped into the run's settings file.
5. **Guard:** measuring cpg0021 painting images with masks declared in the SBS frame (PERISCOPE masks, or labels moved by 919) while `cpg_painting_frame_shift` is `'none'` or unset raises, naming this setting. The guard reads the layout and the 919 `ops_prefix_masks_frame` declaration; it never guesses from pixels.
6. **Border:** the zero-filled band is outside the shifted image; features of a nucleus touching it are flagged (`painting_shift_border` column) and counted in the shift table, not silently kept.

## Settings (alpha)
| Key | Default | Meaning |
|---|---|---|
| `cpg_painting_frame_shift` | `'periscope_logged'` on cpg0021 painting reads, otherwise `'none'` | how painting images are moved into the SBS mask frame |
| `cpg_align_source` | as 919 | where the logged shift comes from (reused from 919) |
| `cpg_align_table` | as 919 | explicit Align table (reused from 919) |

## Tests (CPU, `CUDA_VISIBLE_DEVICES=''`, `pytest -n 4` max)
- **T1 synthetic shift:** a synthetic site with planted nuclei, SBS-frame masks, and painting images moved by a known (dy, dx) per site. With the logged shift the features (integrated intensity, mean intensity, texture) equal the features of the unshifted truth exactly, except for nuclei touching the zero-filled border, which are flagged.
- **T2 refusal:** a site with no Align row, NaN shift, or failed registration raises with the site named; no output is written for it; `'none'` plus SBS-frame masks on cpg0021 raises.
- **T3 off-switch identity:** `'none'` with painting-frame masks equals current nightly byte for byte; non-cpg0021 layouts are unchanged.
- **T4 record:** the per-site shift table lists every site, equals the Image.csv values, and a settings-file rerun gives an identical table and identical features.
- **T5 real-data identity (3 sites of CP186A Well 1):** per-nucleus DNA integrated intensity from spaCR (SBS masks, shifted painting) against our CellProfiler 4.1.3 rerun of PERISCOPE's pipeline on the same sites: r >= 0.9999 and max relative difference reported; the unshifted control r is reported alongside (0.39 on the full 22,025-nucleus check). Evidence base: `cellcycle_transfer_1119` section 12, `dna_check_*`.
- Ratchets before handover: raw-reader, nested-function, documented-callable and hash-comment.

## Acceptance (makes it Ready)
- **A1:** T1-T4 pass on nightly after 919.
- **A2:** T5 passes on 3 sites, then the nucleus/background ratio and per-nucleus DNA check reproduces the card's values on the 75 sites.
- **A3:** the G1/G2 peaks are recovered on the same sites through spaCR (995 field-centred fit), as in the card (10/10).
- **A4:** `'none'` is byte-identical to nightly.

Until then PERISCOPE phenotype results stay INVALID in the readiness register.

## Strings for Codex
For `/mnt/firecuda2/codex/scratch-docs/codex_inbox/tr_998_task.md` at implementation (sources with SHA-256, 9 locales). This item writes no translations.
- Label: `Painting frame shift`
- Help: `How painting images are moved into the frame of the masks. "PERISCOPE logged" uses the shift recorded by PERISCOPE for each site, "Register" measures it from the images, "None" applies no shift and is only valid for masks drawn on the painting images.`
- Error (no record): `No recorded Align shift for site {site} of {plate} {well}; the painting images cannot be matched to the masks. Set cpg_align_table or use Register.`
- Error (guard): `These masks are in the sequencing frame but the painting images are not moved. Set cpg_painting_frame_shift to "periscope_logged" or "register".`
- Column help: `painting_shift_border: the nucleus touches the band filled with zeros by the shift.`

## Effort
S-M, after 919.
