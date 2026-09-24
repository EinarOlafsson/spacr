488 — PYPI README IMAGES WITHOUT LOSING THE GITHUB PAGE
=====================================================
Status: DONE — implementation and package rendering verified; ships with1.5.1.0.

User request2026-09-23: keep every GitHub image and all content, making PyPI
compatible or providing a PyPI-specific equivalent instead of broken images.

The source README.rst is unchanged. setup.py prepares long_description by
resolving relative image/figure URLs to raw GitHub HTTPS resources and relative
page links/targets to the public repository. External badges, image dimensions,
alt text, layout and prose remain intact. This is generated at build time from
the same README, not a second hand-maintained page. No runtime dependencies,
network access or spaCR imports are needed to prepare package metadata.

URLs use the public nightly tree: main currently lacks three organism tiles.
Focused tests exercise actual setup metadata and PyPI's readme_renderer,
checking image count/alt text, absolute links and unchanged source content.
The public page for an existing release is not changed by committing this fix;
the next uploaded package must carry the corrected metadata.

Validation2026-09-23:54 focused packaging/README tests passed. All31 relative
images resolve publicly with HTTP200 and image Content-Type. A fresh build
environment using the declared setuptools>=77 floor generated real wheel
METADATA; PyPI's renderer accepted it and produced the complete HTML preview.
Existing GitHub README remains byte-for-byte unchanged. No PyPI upload made.

2026-09-24 — PACKAGE OUTPUT COMPLETE; PUBLICATION IS A RELEASE OPERATION
Candidate419bf0967's actual wheel README renders49image elements through
PyPI's readme_renderer, all with HTTPS sources. Wheel and source distribution
both pass strict Twine checks. The source archive contains all six release
contract fixtures and excludes development feature/instruction trees.
Receipt features/data/491_release_candidate_artifacts_2026-09-24.json.
Main/PyPI still carry1.5.0.9; the existing public page has not been changed yet.
Per the maintainer, observing the published page after the version update is
not a separate unfinished feature. Any actual publishing failure will be
recorded only if it occurs; this implementation item is complete.
