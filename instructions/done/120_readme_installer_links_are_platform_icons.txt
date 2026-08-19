================================================================================
THE README'S INSTALLER LINKS ARE PLATFORM ICONS, IN WHITE LINE ART
================================================================================

Status:    completed 2026-08-19.
Requested: 2026-08-16 - "add to instructions in the readme the links for the
           installer downloads should be icons for linux (the penguin, but
           just with white lines) osx (the apple, just a white apple) and
           windows (the windows sign, but just white)"
Requested: 2026-08-19 - "for the readme and github page i tried to add a linux an allpe and a windows icon to click on instead of just the blue text links, but they are not showing up. please look into this and fix them

           i weant 3 black and white icons for linux a linux penguine in white, for apple an applie logo in white, and for windows a windows 11 window (the 4 adjacent rectangles, you know what i mean)

           these should be clickable a down loade of each installer. theere should be an icon to the right of these with the spacer logo, that says legacy, if clicked the user is brought to a page with the old versions of spacr. also make sure that the old versions are saved when new versions are released, the generation of the installers is automatic, so it should also be automatic that with a new release the installer links are incremented. but the old links should still be available via this legacy page. and the old installers not deleted but just kept in the folder, it can be the same folder as the new versions as the names will increment with version."
Requested: 2026-08-19 - "ok great the icons are there now! but i want them to look slick not child like. i added 4 images to the platforms folder. use these on a square with rounded edges (but not rounded verry much, a small corner rounding) tat are slate gray."
Requested: 2026-08-19 - "the icons should be centered with no text including the spacr logo one. the icons should take be centered and take up 80% of the button. write legacy in white text under the spacr logo legacy button."
Requested: 2026-08-19 - "i added 3 not 4, you already have the white spacr logo."
Requested: 2026-08-19 - "that was much better, but change to 80% of the button and the color should be #2B2F3A. if possible when the mouse hovers over the button make it #1F5EFF. and take the legacy text out of the button and put it under."

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

--------------------------------------------------------------------------------
RESULT (2026-08-16)
--------------------------------------------------------------------------------

Done. The three installer downloads are now drawn platform icons in
``README.rst`` and in all nine translated READMEs, and both of the file's
premises needed correcting first.

CORRECTION 1 -- THE README IS ``README.rst``, NOT ``README.md``.
That changes the answer to the white-on-white trap. Option (a), a
``<picture>`` with ``prefers-color-scheme``, is UNAVAILABLE: github/markup
renders reStructuredText through docutils with ``raw_enabled=False``, so raw
HTML in a ``.rst`` README is dropped, and there is no other theme-switching
mechanism for RST on GitHub. The same README is also the PyPI
``long_description``, which readme_renderer renders on a permanently white
page. Option (b) was therefore taken, deliberately: each glyph sits on the
same dark rounded chip as ``spacr/resources/icons/app_icon.png`` -- fill
``rgb(0, 55, 55)``, corner radius 188/1024 -- so the artwork is genuinely
white, as asked, and the tile carries its own contrast into a white page
(contrast 13.1:1) and a dark one. Verified by rendering ``README.rst`` with
github/markup's exact docutils settings and with readme_renderer.

CORRECTION 2 -- "ONE VISUAL WEIGHT" AND "OUTLINE PENGUIN, FILLED APPLE,
FILLED WINDOWS" CANNOT BOTH BE HAD. A solid four-pane Windows mark covers
about 0.8 of its glyph box; an outlined penguin covers about 0.15. Measured,
not guessed. Since the file makes equal weight a success criterion and the
maintainer's own words were "just white" for all three -- a statement about
colour, Tux being normally orange-and-black and the Windows mark normally
blue -- all three are drawn as white LINE art at one stroke width. The three
now cover 0.113, 0.115 and 0.117 of their tiles.

WHAT SHIPPED

* ``packaging/generate_platform_icons.py`` draws the three marks and refuses
  to finish if one falls outside the shared coverage band.
* ``spacr/resources/icons/platforms/{windows,macos,linux}.png``, 512 px,
  committed here and served from raw.githubusercontent.com -- nothing is
  hotlinked from a third-party host.
* The README block became three ``image::`` substitutions with ``:width: 64``,
  a translated/localised ``:alt:``, and a ``:target:`` pointing at the same
  release asset the text link pointed at. No URL changed, so nothing that
  resolved before 404s now; all three were confirmed 200 against the live
  v1.5.0.4 release.
* ``tests/test_readme_installer_icons.py`` -- 14 tests asserting the rendered
  effect: GitHub's own docutils settings produce three linked images; the art
  is committed here; each icon is >90% opaque and its chip clears 3:1 against
  a white page; every opaque pixel is white, the chip, or a blend of exactly
  those two; the three coverages stay inside 0.02 of each other. Each guard
  was confirmed to fail on a deliberately broken icon or README.

BUG FOUND AND FIXED IN PASSING. ``packaging/release.py collect`` could not
have run at all: ``_updated_readme_text`` matched the asset stem
case-sensitively as ``spaCR-`` while the README (and the published v1.5.0.4
assets) spell it with a capital S, so the real README raised "malformed
installer link" for Windows. The matcher is now case-insensitive, matches the
URL rather than the ``\`text <url>\`_`` syntax around it -- so it handles the
icon form and a not-yet-converted text-link form alike -- requires all three
platforms to advertise one version, and rewrites the whole block so the
version in the URL, the file name and the label/alt text move together.

NOT DONE, DELIBERATELY

* The four-pane Windows mark and the apple are the real trademarks. The
  generic-glyph fallback is written down in the generator's docstring and was
  not taken; GitHub READMEs use the real marks and this is nominative use.
* No caption or version text was added under the icon row. The ``|Release|``
  badge directly above already shows the current installer version, and the
  version is in every ``:alt:``.
* TERRITORY: the nine ``docs/i18n/readme/README.*.rst`` files were converted
  even though that tree belongs to another worker, because
  ``test_localized_readmes_preserve_urls_code_and_table_shape`` requires every
  translated README to carry the canonical README's exact URL set -- the
  English change alone leaves the suite red. The conversion is mechanical and
  loses no translation: each ``:alt:`` is that file's own existing localised
  link text, verbatim. ``tests/test_documentation_i18n.py`` was updated for
  the new alt-text count (14 -> 17) and now also asserts the download alts are
  localised rather than English. Both were committed separately so they can be
  reverted on their own.

--------------------------------------------------------------------------------
REOPENED -- 2026-08-19
--------------------------------------------------------------------------------

The artwork exists on ``nightly`` but not on ``main``. Every README image URL
hard-codes ``main``, so all three live image requests return 404. The legacy
link also returns 404 because the generated installer page has not reached the
published GitHub Pages site. This is a delivery failure, not a drawing failure.

This reopened task absorbs instruction 123. Its earlier decision not to retain
the lightweight installers in ``spacr/application`` is explicitly superseded
by the maintainer's new request: versioned installers must accumulate there,
the generated archive must include the release being built without waiting for
a later release, and the fourth README tile must open that archive.

--------------------------------------------------------------------------------
RESULT -- 2026-08-19
--------------------------------------------------------------------------------

The broken delivery path is fixed. README artwork now comes from the shared
``nightly`` branch where the committed images exist, instead of ``main`` where
all three requests returned 404. The row is Linux, macOS, Windows, Legacy. The
new Legacy tile uses the white spaCR mark and a visible ``LEGACY`` label on the
same dark chip as the platform icons, and opens GitHub's rendered
``docs/source/installers.rst`` directly rather than the currently undeployed
Pages URL.

The archive has no bare extension links: every Linux, macOS, and Windows cell
is the corresponding clickable icon. The release workflow now merges the
just-built installers into the GitHub release history before rendering the
page, so a release adds its own row even though the GitHub release is created
later in the job graph.

``collect_installers`` no longer deletes older versions. It retains and hashes
everything already in ``spacr/application``; the 1.4.9.9 and 1.5.0.4 sets were
restored beside 1.5.0.1, for nine retained installers in total. Future release
commits add the newly versioned files and preserve those nine.

Verified with 39 focused tests covering GitHub-style RST rendering, linked
artwork and contrast, live release URLs, archive generation, pre-publication
row insertion, file retention, workflow wiring, and all localized README
copies. ``git diff --check`` is clean.

--------------------------------------------------------------------------------
REOPENED -- 2026-08-19, VISUAL REFINEMENT
--------------------------------------------------------------------------------

The four supplied images are now the source artwork. Preserve their design;
the generator should only place each mark on a slate-gray square with a small,
restrained corner radius. The previous teal, heavily rounded, hand-drawn tiles
are explicitly superseded.

Clarification: there are three supplied platform images; the existing white
spaCR logo is the fourth source. Platform marks have no text and fill 80% of
the tile. The legacy tile is the one exception: its centered spaCR logo has a
white ``LEGACY`` label beneath it.

--------------------------------------------------------------------------------
RESULT -- 2026-08-19, VISUAL REFINEMENT
--------------------------------------------------------------------------------

The README now uses the three supplied platform marks rather than the earlier
drawn approximations. Their originals are preserved under
``spacr/resources/icons/platforms/source``: the transparent Linux PNG, the
Windows PNG, and the Apple JPEG (stored with its correct ``.jpg`` extension).
The generator removes each source's particular background without modifying
or overwriting that source.

All four generated buttons are 512 px squares in slate gray ``#475569`` with
a restrained 32 px corner radius. Linux, Apple, and Windows are white,
text-free, centred, and fitted so the longer dimension occupies 80% of the
button. The fourth tile uses the existing white spaCR logo, centred above the
only label in the row: ``LEGACY`` in white Open Sans Bold.

The revised tests assert source provenance (including that Apple is JPEG),
the exact slate tile, small corner radius, 80% platform geometry, centring,
legacy-label placement, and byte-for-pixel agreement between committed assets
and a fresh generator render. All 43 focused README-icon and installer-archive
tests pass, and ``git diff --check`` is clean.

--------------------------------------------------------------------------------
REOPENED -- 2026-08-19, COLOUR AND LEGACY CAPTION
--------------------------------------------------------------------------------

Change the tile colour to ``#2B2F3A`` and keep every mark at 80% of the
512 px button. The legacy image must contain only the centred spaCR mark;
``Legacy`` belongs beneath the button rather than inside its slate square.

GitHub renders this reStructuredText README with raw HTML and custom CSS
disabled. A linked PNG therefore cannot react to pointer hover, and the page
cannot install a ``:hover`` rule that changes its pixels to ``#1F5EFF``. Check
the actual rendered HTML before closing this request, preserve a clickable
archive target, and document this platform limitation rather than pretending
that a static asset can hover.

--------------------------------------------------------------------------------
RESULT -- 2026-08-19, COLOUR AND LEGACY CAPTION
--------------------------------------------------------------------------------

All four visible buttons are 512 px squares filled exactly ``#2B2F3A``. Each
white mark, including the spaCR legacy mark, is centred and fitted to 410 px:
80% of the button. Every generated image shares a 600 px-high transparent
canvas so the four square buttons align on one row; only the legacy canvas
uses that space, drawing ``Legacy`` below the button rather than inside it.
The white caption has a narrow ``#2B2F3A`` edge so it remains legible against
both GitHub's light and dark page backgrounds.

The requested ``#1F5EFF`` hover state is deliberately not faked. GitHub's
reStructuredText README renderer disables raw HTML and custom CSS, while a
linked PNG cannot respond to pointer state. The desired blue is recorded as
``HOVER_BLUE`` in the generator for a future HTML/CSS surface where a genuine
hover interaction can be implemented.

The row was inspected at its actual 64 px README width on both light and dark
backgrounds. All 43 focused icon/archive tests pass; they now assert the exact
colour, 80% geometry, aligned output canvases, empty caption strips on the
three platform buttons, the external legacy caption, and reproducible output.
