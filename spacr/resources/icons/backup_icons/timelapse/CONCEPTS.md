# timelapse - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **timelapse_01** - Filmstrip of three frames, the same cell further along in each.
2. **timelapse_02** - One track threading the same object through three frames in a row.
3. **timelapse_03** - Kymograph: the object's position streaked down a vertical time axis.
4. **timelapse_04** - One cell in the first frame and two in the last: a division caught over frames.
5. **timelapse_05** - A clock over the field: the cell now, and the dashed outline of where it was.
6. **timelapse_06** - A playhead running along a bar of frame ticks, the frame it lands on above.
7. **timelapse_07** - The same cell three times over: dashed where it was, solid where it is now.
8. **timelapse_08** - Two objects keeping their identity: matching links drawn from frame to frame.
9. **timelapse_09** - Frames stacked back into depth, the newest in front, along the time arrow.
10. **timelapse_10** - A film reel: the strip spooled up, one cell showing through a window.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_imaging_explore.py`
