# cellpose_masks -- 10 candidate concepts

White artwork on transparent background, 1024x1024 RGBA, house style of `plaque.png` / `measure.png`.
Candidates for review only; nothing here is installed.

1. **cellpose_masks_01** - Raw signal on one side, finished mask on the other.
2. **cellpose_masks_02** - A polygon ROI with its vertices.
3. **cellpose_masks_03** - The mask as a stencil cut out of the field.
4. **cellpose_masks_04** - Instance labels: every object gets its own number.
5. **cellpose_masks_05** - The mask is per pixel (pixelated silhouette).
6. **cellpose_masks_06** - Cellpose flow vectors converging on the centre.
7. **cellpose_masks_07** - The mask layer sitting above the image layer.
8. **cellpose_masks_08** - The model diameter fitted to the object (Cellpose's `diameter`).
9. **cellpose_masks_09** - Touching objects split along their ridges.
10. **cellpose_masks_10** - Filled object in, boundary out.

Contact sheets: `_sheet_dark.png` (white ink on #14161a) and `_sheet_light.png` (the same alpha masks tinted dark on #f5f6f8, which is how they would have to be drawn in a light theme -- as shipped they are pure white and invisible there).
