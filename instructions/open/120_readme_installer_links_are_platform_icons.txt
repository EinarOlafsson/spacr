================================================================================
THE README'S INSTALLER LINKS ARE PLATFORM ICONS, IN WHITE LINE ART
================================================================================

Status:    filed 2026-08-16, not started.
Requested: 2026-08-16 - "add to instructions in the readme the links for the
           installer downloads should be icons for linux (the penguin, but
           just with white lines) osx (the apple, just a white apple) and
           windows (the windows sign, but just white)"

--------------------------------------------------------------------------------
WHAT TO BUILD
--------------------------------------------------------------------------------

Three download links in the README, each an icon rather than a text link:

    Linux      the penguin, drawn in white LINES -- outline, not a filled
               silhouette. The maintainer said "just with white lines".
    macOS      the apple, a plain white apple.
    Windows    the four-pane window mark, plain white.

One visual weight, one colour, three platforms. They are a row of choices,
so anything that makes one louder than the others is wrong.

--------------------------------------------------------------------------------
WHITE, ON A README THAT IS READ IN BOTH THEMES
--------------------------------------------------------------------------------

THE OBVIOUS TRAP, and it will be hit if it is not written down: GitHub
renders README.md on a white background in light mode. A white icon there is
invisible.

So "white" cannot be taken as a literal fill and shipped. Either

  (a) use GitHub's own theme switching -- `<picture>` with
      `<source media="(prefers-color-scheme: dark)">` and a dark-ink variant
      for light mode, which is the only way to honour "white" AND stay
      visible; or
  (b) draw the icon on a chip/badge whose background guarantees contrast in
      both themes.

(a) is the honest reading of the request: the maintainer wants white icons,
and white icons are what a dark-mode reader gets. Do not silently substitute
grey and call it done -- if (b) is chosen, say so here and why.

--------------------------------------------------------------------------------
THE ICONS THEMSELVES
--------------------------------------------------------------------------------

DRAW THEM, do not hotlink them. An `<img src="https://some-cdn/...">` in a
README is a dead icon the day that host moves, and it leaks a request to a
third party for every reader of the page.

Inline SVG or a committed asset under the repository. spaCR already ships
icons under `spacr/resources/icons/`; the same place, or `docs/`, is fine.

TRADEMARKS, stated rather than discovered later: Tux is free to use. The
Apple logo and the Windows logo are trademarks whose use is restricted --
GitHub READMEs do it constantly and nobody minds, but the SAFE version is a
generic laptop/window glyph if it ever matters. Use the real marks, note the
alternative here, and move on.

--------------------------------------------------------------------------------
THE LINKS MUST GO SOMEWHERE
--------------------------------------------------------------------------------

Each icon links to the actual installer for that platform. Instruction 53
covers building those installers; if a platform's installer does not yet
exist, the icon for it should not silently 404 -- either omit that platform
or point it at the instructions for installing by hand, and say which.

--------------------------------------------------------------------------------
HOW TO KNOW IT WORKED
--------------------------------------------------------------------------------

* The README shows three platform icons where the download links were.
* Every one is visible in BOTH GitHub themes -- checked by looking, in both,
  not by assuming.
* No icon is hotlinked from a third-party host.
* Every icon links to a URL that resolves, and any platform without an
  installer is handled deliberately rather than left broken.
* The three read as one row of equal choices -- same size, same weight.
