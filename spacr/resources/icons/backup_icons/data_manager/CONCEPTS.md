# data_manager - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **data_manager_01** - A treemap of the project: the big folder dwarfing the rest.
2. **data_manager_02** - A disk-use ring with one wedge pulled clear of the rest.
3. **data_manager_03** - A drive whose fill level drops back down.
4. **data_manager_04** - Derived tiles going in the bin while the originals stay locked.
5. **data_manager_05** - A tall stack of files shrunk to a short one.
6. **data_manager_06** - A capacity gauge with its needle swung back off full.
7. **data_manager_07** - A folder list with a size bar against every entry.
8. **data_manager_08** - A block of data squeezed down between two jaws.
9. **data_manager_09** - The project folder measured end to end for what it costs.
10. **data_manager_10** - Two piles: what is kept, and what can go without losing anything.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_data_io.py`
