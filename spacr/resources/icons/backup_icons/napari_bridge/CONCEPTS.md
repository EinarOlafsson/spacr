# napari_bridge - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **napari_bridge_01** - [napari mark - their trademark] The four-petal rosette on its own, monochrome.
2. **napari_bridge_02** - [napari mark - their trademark] The rosette open: petals as outlines round a clear centre.
3. **napari_bridge_03** - [napari mark - their trademark] The rosette inside another application's window.
4. **napari_bridge_04** - [napari mark - their trademark] A mask handed out to the rosette and taken back corrected.
5. **napari_bridge_05** - [napari mark - their trademark] Their brush, our data: a brush laid across the rosette.
6. **napari_bridge_06** - [original - no third-party mark] Two panes and a span between them, an object walking across.
7. **napari_bridge_07** - [original - no third-party mark] The mask goes out rough and comes back with its boundary fixed.
8. **napari_bridge_08** - [original - no third-party mark] Two windows overlapping, one object shared in the seam.
9. **napari_bridge_09** - [original - no third-party mark] It comes back as itself: the label number survives the crossing.
10. **napari_bridge_10** - [original - no third-party mark] Nothing crosses back unchecked: the wrong shape is turned away.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_trellis_gate_feature_napari.py`

Whose mark is on which candidate
--------------------------------

**01-05 carry napari's own visual identity** -- a monochrome four-petal
rosette in the manner of napari's mark. That mark is **napari's trademark, not
spaCR's**. Labelling a bridge *to* napari with it is ordinary nominative use
and is what most integrations do, but picking one of these means shipping a
monochrome derivative of a third-party trademark, so it should be a decision
rather than an accident.

**06-10 are original spaCR marks about the handoff** -- two panes and traffic
between them, a mask leaving rough and returning corrected, a label value that
survives the crossing, a return that is checked. They carry no third-party
mark at all.

Nothing was copied. No napari image file exists in this repository; the
rosette is built from the same `_draw` primitives as every other icon here.
