# db_browser - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **db_browser_01** - A lens held over the rows of a table.
2. **db_browser_02** - A query typed above the table, and the rows that answered it.
3. **db_browser_03** - Many rows into the filter, a few rows out.
4. **db_browser_04** - One column sorted: the rows shuffled into order under a caret.
5. **db_browser_05** - Rows poured out of the database file into a readable grid.
6. **db_browser_06** - One cell picked out, its row and its column banded across the sheet.
7. **db_browser_07** - A slice of the table walked out as a plain sheet of rows.
8. **db_browser_08** - The browser itself: tables listed down one side, rows on the other.
9. **db_browser_09** - Two tables joined on the key column they share.
10. **db_browser_10** - Paging through the rows: a long table with a grip on its scrollbar.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_data_io.py`
