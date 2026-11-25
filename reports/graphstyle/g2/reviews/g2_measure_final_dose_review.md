> Curated publication copy. Original source: `.codex-audit/g2_measure_final_dose_review.md`; original SHA256: `185ae4d10f68a5f07e15515545014a495a81490d256dd491b91f36cc90669176`. Original observations retain their recorded historical scope. Only selected artifacts are linked; omitted artifacts are labeled local-only and are never retargeted to a later pass. Unlinked context-relative paths and code blocks retain the original review context, which can include local-only omitted outputs; they do not select a later artifact. The validation index identifies the final selected subsets.

# Final Measure dose review — job 1668

Accepted for the frozen fixture in both themes. Both PNGs were individually inspected with `view_image`: the four blue observations at doses 0.1, 1, 10 and 100 are visibly drawn over the fitted curve and its contrast backing. The dark legend has a dark frame/background and readable pale text. The fitted curve, half-maximal reference, concentration labels and refused second panel remain readable, without new clipping or overlap. Both finalizer problem lists are empty; acceptance also rests on direct inspection, which identified the preceding v2 occlusion despite its empty lists.

The [v3 receipt](../evidence/measure-after-v3/receipt.json) identifies the actual validation-tree Measure source as SHA256 `4d5ca6af5c19585173731eb2ec88db2f85340a6a84e9ffa5938a75e950e45691`, using public saved preferences, the actual creator and the normal screen-mode writer, with `scientific_reruns: false`. Actual figure/axes backgrounds are white in light mode and `#181818` in dark mode; recorded text colors are black and `#F0F0F0`, respectively. Focused tests are frozen at `293719d2a71f36fd5800355a63e5685741d44c1c61b8ccb5031c17a5b259d699`; the unchanged evidence helper is `2131be9db3918842af6a3b792574513756947479c341183417b917aac6a36ab1`.

Baseline `measure-before` versus final `measure-after-v3` receipt comparison with `jq`, followed by direct `sha256sum` verification of all twelve input/payload/data files, establishes exact equality for both themes:

| File | SHA256, identical before/after and light/dark |
| --- | --- |
| `inputs.json` | `32cff66689792b378942ae86fd131a8a93b15770393ffaac0b2a24f04a3abd8b` |
| `payload.json` | `6ed4f31531850e8ccf27379e7b5aac79a31fa503e0356509bb23b1130918c9f9` |
| `data.csv` | `7d3a4458647dbb8136c3d2c94e37618808e579340364cbf897fc7ced225a2a39` |

Output file hashes were independently checked against the receipts:

| Output | SHA256 |
| --- | --- |
| [Light PNG](../evidence/measure-after-v3/light/viability_dose/viability_dose.png) | `ad7e5e5c938c51cebf06be13a85028df045412ae1afad48fb0fec2825507596b` |
| [Dark PNG](../evidence/measure-after-v3/dark/viability_dose/viability_dose.png) | `730e148a945ffd914b20a0ef2ef661c5cd7a0280bdb87aff2fa177d9ea5db6f4` |
| [Light PDF](../evidence/measure-after-v3/light/viability_dose/viability_dose.pdf) | `5b17704a9f82fdd2bbd869b321463c3e5b62298b8ec1783ff8b67912c3b1ff6b` |
| [Dark PDF](../evidence/measure-after-v3/dark/viability_dose/viability_dose.pdf) | `2ad7a1f87037f3befc74ab85e23daa4764aed16e4fa7b483b34a861e420c81c4` |

This closes the dose-marker visual blocker. Retain the other 12 accepted v2 images and original 10 unaffected images; no broad redraw is needed. The existing DNA observation remains: overlapping fitted components are not individually distinguishable everywhere, and the backed full model is visually dominant. No numerical or component rewrite is implied.

Only read-only inspection and this review note were performed here: no application edits, imports, tests, rendering or compute jobs. PDF files were hash-verified, not visually inspected.
