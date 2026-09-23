# Publish documentation before every translation is finished

The package release and documentation deployment are separate workflows.
A `1.5.1.0` version bump does not publish private tutorial recordings.

## Publish the current English documentation and API

After the intended documentation changes are committed and pushed, run the
**docs** workflow with `api_language=english` on the chosen branch:

```sh
gh workflow run docs.yml --ref nightly -f api_language=english
```

An explicit dispatch on `nightly` publishes that exact run's checkout to the
public GitHub Pages site. It replaces the previous public documentation, so
choose `main` instead when the site must describe the released application.
Ordinary `nightly` pushes only build the site.

English mode regenerates the English API and runtime manifests from the exact
checkout being deployed and audits them against that source. It then builds
Sphinx with warnings treated as errors and checks the rendered API
links. API pages display an English-publication notice and do not offer stale
translations. Existing browser language preferences are preserved for a future
translated build. This mode does not alter the application's GUI languages.
The displayed documentation version comes from that checkout's matching
`setup.py` and `spacr/_version.py`, even when the build environment has an older
installed distribution.

The default `all` mode still requires every translation catalog to pass its
strict audit. Automatic `main` builds use that default. Until that complete
catalog gate passes, use an explicit English docs run after a package release;
the package version bump alone is insufficient to update the site.

## What a package release publishes

The release workflow accepts three- or four-component numeric versions, so
`1.5.1.0` is supported. Automatic release runs require an actual version
increase in `setup.py` pushed to `main`; a bump on `nightly` is not a release.
The manual release workflow also targets `main` and updates the version files
together. Merge the intended code, documentation and tutorial catalog before
releasing that version. Package publication has its own build checks and does
not depend on completion of all documentation translations.

## Tutorials can be delivered incrementally

The docs build publishes the tutorial player and catalog already checked into
`docs/source/_extra/tutorials`, with narration and masters pinned to their media
revision. A successful docs build does not promote the private authoring stage.

Ready lessons can be prepared as a separate verified candidate while unfinished
lessons remain unavailable. Check each included lesson's script, video, audio,
captions, links and browser playback against the same source snapshot before
promoting it. Missing voices must not be advertised as available. The candidate
publisher verifies file hashes and browser evidence, uploads a new media
revision, reads back the uploaded bytes and pins the website to that commit.
See `tools/tutorials/publish_release_candidate.py` for those separate steps.

Finishing every translation, every voice and all future tutorial improvements
is not a prerequisite for publishing an independently verified subset.
