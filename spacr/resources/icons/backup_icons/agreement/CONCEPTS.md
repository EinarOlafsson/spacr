# agreement - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **agreement_01** - Two label columns joined pair by pair, one pair crossed out.
2. **agreement_02** - A 2x2 square of the two annotators' calls, agreements on the diagonal.
3. **agreement_03** - A kappa dial reading how far past chance the two raters got.
4. **agreement_04** - Two label sets overlapping, the agreed middle solid.
5. **agreement_05** - Two annotators nodding at the same call.
6. **agreement_06** - A disputed image pulled up for review, a tick against a cross.
7. **agreement_07** - Two label ribbons compared tile by tile, the odd tile flagged.
8. **agreement_08** - The two annotators' calls weighed against each other.
9. **agreement_09** - Three raters' columns resolved into one majority column.
10. **agreement_10** - Observed agreement with the chance share cut off the front.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_results_qc.py`
