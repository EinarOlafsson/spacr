# distributed_jobs - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **distributed_jobs_01** - A head node fanning the work out to three worker machines.
2. **distributed_jobs_02** - A rack cabinet of three blades, the top two running.
3. **distributed_jobs_03** - A job card lifting off into a cloud.
4. **distributed_jobs_04** - A terminal prompt wired down a cable to a machine somewhere else.
5. **distributed_jobs_05** - A scheduler in the middle with worker nodes spoked around it.
6. **distributed_jobs_06** - A scheduler allocation chart: jobs booked across node rows and time.
7. **distributed_jobs_07** - A laptop pushing a run down a long cable to a far bigger machine.
8. **distributed_jobs_08** - One submitted job splitting into three parallel lanes of work.
9. **distributed_jobs_09** - A remote machine reporting a live progress bar back home.
10. **distributed_jobs_10** - A submit button firing a paper plane at a queue of remote machines.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_jobs_models.py`
