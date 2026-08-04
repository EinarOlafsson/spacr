# dose_response - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **dose_response_01** - The full S, and the EC50 read off it at half the response.
2. **dose_response_02** - The concentration axis is logarithmic, and the curve is read on it.
3. **dose_response_03** - The inflection: the one place the response is actually changing.
4. **dose_response_04** - The EC50 with the interval it is known to, bracketed on the axis.
5. **dose_response_05** - Two compounds: the more potent one's midpoint sits further left.
6. **dose_response_06** - Still rising at the last dose: the answer is a bound, not a number.
7. **dose_response_07** - Up and back down again: not a dose-response, so not fitted.
8. **dose_response_08** - Both plateaus stated, and the span the response actually moves over.
9. **dose_response_09** - The fitted EC50 chosen as the dose the next experiment will use.
10. **dose_response_10** - One midpoint, two steepnesses: how sharply the response switches.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_pca_tabulate_dict_outliers_dose.py`
