# README deck sources

Sources of the slide deck published in `docs/source/_static/deck/`. `scripts/build.js`
(pptxgenjs) draws the 57 slides from `images/` and `scripts/sizes.json`; the other
scripts in `scripts/` made some of those images and need data that is not in the
repository. App screenshots in `images/app_1513/` come from the current application
(offscreen Qt, 1920x1080 at scale 2).

Rebuild, from the repository root:

    cd tools/readme_deck
    npm ci
    PUBLIC=1 node scripts/build.js
    cd ../..
    python tools/build_readme_deck.py tools/readme_deck/spacr_overview.pptx --out /private/rendered
    python tools/build_readme_deck.py --refresh-pages 8,32,48 --rendered /private/rendered --out /private/checked
    rsync -a /private/checked/ docs/source/_static/deck/
    python tools/refresh_alpha_deck.py

`python tools/build_readme_deck.py --out /private/rendered` runs the node steps itself;
`--pres DIR` selects other sources. `--out` must be a new folder. Choose the pages that
changed by comparing the rendered slides with the published ones.
