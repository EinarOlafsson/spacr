# spaCR on conda-forge

Conda-forge requires one human-reviewed onboarding pull request before it can
publish a new package. The spaCR onboarding request is
[`conda-forge/staged-recipes#34352`](https://github.com/conda-forge/staged-recipes/pull/34352).
The reference v1 recipe in `recipe/recipe.yaml` uses the same immutable PyPI
source archive and license metadata.

## One-time onboarding

1. Keep the onboarding pull request green and respond to conda-forge review.
2. After merge, conda-forge creates
   `conda-forge/spacr-feedstock` and publishes the initial package.
3. Copy `conda-forge.yml` from this directory to the root of the generated
   feedstock and merge that feedstock change.

The recipe uses the PyPI source archive. Its dependency names are
translated to their conda-forge distribution names, including `torch` to
`pytorch`, `opencv-python-headless` to `opencv`, `tables` to `pytables`, and
`nvidia-ml-py` to `pynvml`.

## Automatic releases after onboarding

The source repository publishes PyPI and GitHub first. The conda-forge bot
then detects the new PyPI version, updates the recipe version and source hash,
tests the feedstock on conda-forge infrastructure, and automatically merges a
passing version update. Conda-forge publishes the package from that merge.

Dependency-list changes still require an ordinary feedstock recipe edit.
Version-only releases require no manual conda command or upload token.
