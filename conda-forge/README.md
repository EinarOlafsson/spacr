# spaCR on conda-forge

spaCR is published on [conda-forge](https://anaconda.org/conda-forge/spacr).
Create a separate environment to install it:

```bash
conda create -n spacr -c conda-forge spacr
conda activate spacr
spacr-run --list
spacr-qt
```

The authoritative build recipe and configuration live in
[`conda-forge/spacr-feedstock`](https://github.com/conda-forge/spacr-feedstock).
The one-time onboarding is complete. Version 1.5.0.8 was published on
2026-09-19 after [feedstock PR #3](https://github.com/conda-forge/spacr-feedstock/pull/3)
merged automatically. Its main-channel artifact and checksum were verified
on 2026-09-23; this check did not perform a fresh environment installation.

The reference recipe in this directory is a source-repository mirror;
conda-forge does not build from it. As of 2026-09-23, this mirror names the
published 1.5.0.9 PyPI archive, whose download and SHA-256 were verified;
the feedstock and conda package are still at 1.5.0.8. Their dependency pins
and build details also differ. The feedstock carries
`tensorboard-2.20.patch`. Compare against
the feedstock before proposing recipe or configuration changes; copying the
mirror over the feedstock would discard those changes.

## Automatic releases

The source repository publishes PyPI and GitHub first. The conda-forge bot
then detects the new PyPI version, updates the recipe version and source hash,
tests the feedstock on conda-forge infrastructure, and automatically merges a
passing version update. Conda-forge publishes the package from that merge.

Dependency-list changes still require an ordinary feedstock recipe edit.
Version-only releases require no manual conda command or upload token.
The bot's update can lag PyPI: check the feedstock pull requests and the
published package before describing an update as pending. Keep the mirror's
version and source hash current separately; changing the mirror does not
trigger a conda-forge release.
