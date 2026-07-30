# spaCR on conda-forge

Conda-forge requires one human-reviewed onboarding pull request before it can
publish a new package. The ready-to-submit recipe is in `recipe/recipe.yaml`.

## One-time onboarding

1. Fork
   [`conda-forge/staged-recipes`](https://github.com/conda-forge/staged-recipes).
2. Create a branch in that fork and copy this repository's `recipe` directory
   to `recipes/spacr`.
3. Open a pull request to `conda-forge/staged-recipes`.
4. Respond to the conda-forge review. After merge, conda-forge creates
   `conda-forge/spacr-feedstock` and publishes the initial package.
5. Copy `conda-forge.yml` from this directory to the root of the generated
   feedstock and merge that feedstock change.

The recipe uses the tagged GitHub source archive. Its dependency names are
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
