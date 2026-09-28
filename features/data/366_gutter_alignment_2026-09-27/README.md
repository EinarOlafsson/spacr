# Item 366: middle alignment candidate

The prior live measurement is preserved in
`../366_live_gutters_2026-09-27/`: the actual GitHub page had a 6 CSS pixel
vertical surplus at both 1440 and 1024 pixel viewport widths. The acceptance
tolerance remains 1 CSS pixel. That failure is not superseded by this local
preview.

GitHub removes the alignment class emitted by a normal docutils image
substitution. The candidate uses raw image substitutions with `align="middle"`
in the README; widths, image bytes, alt labels, API targets and explicit rows
are unchanged. Sphinx retains its ordinary RST image substitutions. The PyPI
metadata adapter converts raw tiles back to ordinary RST because PyPI disables
raw HTML.

Relevant upstream implementation:

- [GitHub rendering pipeline](https://github.com/github/markup): sanitization
  removes inline styles and classes.
- [GitHub RST renderer](https://github.com/github/markup/blob/master/lib/github/commands/rest2html):
  HTML4 writer and `raw_enabled=True`.
- [HTMLPipeline sanitization](https://github.com/gjtorikian/html-pipeline/blob/main/lib/html_pipeline/sanitization_filter.rb):
  image elements and the `align` attribute are allowed.
- [PyPI RST renderer](https://github.com/pypa/readme_renderer/blob/main/readme_renderer/rst.py):
  `raw_enabled=False`.

`local-preview-receipt.json` measures the generated HTML4 fragment inside a
local copy of the GitHub page shell and its loaded stylesheets. No stylesheet
was modified, and the original PNG bytes were loaded unchanged. This is a
local preview, **not live publication acceptance**. Visible horizontal and
vertical gaps were equal: 8.3798828125 CSS pixels at 1440 and 5.8193359375 at
1024, with zero spread at each width. Both screenshots contain all 21 tiles;
the 1024 screenshot received visual review.

`markup-receipt.json` records before/after hashes for all ten README files.
Generated tile substitutions retain the existing reviewed alt-label templates;
no translated prose or native-speaker review is claimed by this markup change.

Publication and another unmodified live GitHub measurement remain required.

Validation: 34 focused README contracts passed with the translation advisory
plugin disabled. The application environment lacked the optional PyPI renderer,
so the three actual PyPI rendering/conversion checks were then run successfully
in the existing base Conda documentation environment. They verify that all 49
original images and accessible labels survive packaging, links are absolute,
escaped attributes survive conversion and conversion is idempotent. All nine
current canonical README audit branches passed, including exact HTML link and
image attributes; the unchanged API audit was not repeated. `COVERAGE.md` was
regenerated against live sources and the new README byte counts.
