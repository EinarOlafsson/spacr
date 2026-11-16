# Item 957: point-count annotation mode in Make Masks

- **Number:** 957 (assigned by master 2026-11-15). **Priority:** high: it blocks the PERISCOPE S3 nuclei verdict and the nucleus-splitting tuning decision. **Lifecycle:** alpha, with 9-locale strings and a GROUPS line (curation / annotation).
- **Maintainer:** 2026-11-15, "I should be able to use spaCR for this task" (the hand-check of 20 nuclei crops, until now planned in Fiji).

## Spec
1. `spacr-make-masks --folder <dir> --points`, and the same choice in the GUI's Make Masks. The folder opens as the existing resumable curation queue (`spacr.curation_queue`).
2. **Point mode controls:**
   - left click adds a point at the cursor;
   - right click removes the nearest point (within a few px);
   - a key toggles "unsure" on the nearest or last point;
   - undo;
   - the point count is shown;
   - zoom and pan as in the editor.
3. **Auto-save** per image to `<folder>/../annotations/<image stem>_points.csv`, where `<image stem>` is the file name without `_clean.png`; for example `crop01_site3_y1234_x567_clean.png` saves as `crop01_site3_y1234_x567_points.csv`. Columns:
   - **`X`, `Y`** (float, image pixels, origin top-left, x = column, y = row). This is the Fiji Measure convention, and the scorer reads exactly these two column names (`P[["Y", "X"]]`, rounded and clipped to 0–511);
   - `unsure` (0/1);
   - optional `note`.
   - Writing is atomic (temp file + rename), and saving again overwrites.
4. **Blind by default in point mode** (`--hide-overlays` on). An overlay folder (`--overlays <dir>`) can be shown only after every image in the queue is marked done.
5. No change to the existing brush or mask modes; their outputs are byte-identical.

## Files
| What | Path |
|---|---|
| Test crops (annotate these) | `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/handcheck_crops/annotate_clean/` (20 PNGs, 512 × 512) |
| Instructions (rules for what to click) | `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/handcheck_crops/INSTRUCTIONS.md` |
| Manifest | `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/handcheck_crops/crop_manifest.csv` |
| Output folder (the maintainer's real annotations; the proof must NOT write here) | `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/handcheck_crops/annotations/` |
| Scorer that reads the points | `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/s3_phenotype/scripts/score_s3_nuclei.py` (lines 169–193: `glob annotations/*_points.csv`, `pd.read_csv`, columns `Y`, `X`) |
| Overlays to show after completion | `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/handcheck_crops/overlays/` and `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/s3_phenotype/out/crops/` |

## Proof check (pre-set)
- **Synthetic round trip in a scratch copy** of handcheck_crops (never in `annotations/`): scripted clicks at known positions → CSV → `score_s3_nuclei.py`'s hand-check block. The hit, point and object counts must be exactly as expected.
- **Format:** the CSV opens in pandas with the columns `X`, `Y`, `unsure`, and `Y`/`X` match the clicked pixel (row/column) exactly.
- **Resume:** closing and reopening the queue keeps every point and the done flags.
- **Brush and mask modes** are unchanged (the existing tests pass).
