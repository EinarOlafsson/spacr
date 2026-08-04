# profiler - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **profiler_01** - A response curve with the cursor riding on it above its slider.
2. **profiler_02** - One slider pushed along, and the output bar that grew with it.
3. **profiler_03** - A knob turned one way, the needle on the readout following it.
4. **profiler_04** - Three inputs held still, one moved off centre, one prediction out.
5. **profiler_05** - The local slope of the response, read off at the cursor.
6. **profiler_06** - An input pushed into the fitted model, a prediction coming out.
7. **profiler_07** - The cursor moved from here to there, and the step in the prediction.
8. **profiler_08** - The output column rising as the one handle is dragged right.
9. **profiler_09** - A lever pressed at one end, the readout swinging at the other.
10. **profiler_10** - One feature in the input row overwritten, and the prediction shifting.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_jobs_models.py`
