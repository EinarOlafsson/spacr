# Editable translation reviews and generated catalogs

Editable production review records and their retired history live in the public
[spacr-translations](https://github.com/EinarOlafsson/spacr-translations) Git
repository. The initial migration preserves the approved 6,472 files, including
review-authoring instructions and retired records, at their original paths under
`docs/i18n/reviewed/` and `docs/i18n/retired/`. Fifteen historical fixtures,
receipt and installer evidence files remain in the spaCR source repository.

`docs/translation_reviews_revision.json` pins a full external commit and the
SHA-256 of its manifest. That manifest records each input's relative path, size
and SHA-256. Hydrated copies use the same paths in a spaCR checkout and are
ignored by Git; they are build inputs, not the editable authoritative repository.
The bootstrap reads verified Git objects and writes independent files. It never
hardlinks a working review file to a shared cache and refuses to overwrite local
differences or silently remove unpinned reviews.

## Bootstrap a source checkout

From the spaCR checkout, before source-level checks or documentation generation:

```sh
python tools/bootstrap_translation_reviews.py fetch
python tools/bootstrap_translation_reviews.py check
```

The bootstrap uses Git and the Python standard library. It needs no private
archive credentials or Hugging Face token. Public test and Docs workflows run
it explicitly before readers. Imports validate available canonical inputs
offline; they do not initiate network requests. Custom temporary review
directories used by isolated tests bypass canonical bootstrap.

For an offline machine, seed the cache and copies online first. An existing
verified Git cache can also restore missing copies without a connection:

```sh
python tools/bootstrap_translation_reviews.py fetch --offline
python tools/bootstrap_translation_reviews.py check --offline
```

`SPACR_TRANSLATIONS_CACHE` or `--cache /path/to/cache` selects the cache. Its
default is the platform user's cache home under `spacr/translation-reviews`.
A cold offline cache fails with setup guidance; it does not substitute empty
translations. Validation is cached once per process and pin. Explicit `check`
revalidates all pinned files. Kernel advisory locks release on process death;
complete files are staged and published atomically without clobbering local
files. An interrupted staging file is not treated as a reviewed record.

Normal source entrypoints include `build_documentation_i18n.py`,
`build_i18n_catalogs.py`, the reviewed API/runtime evidence checkers,
`audit_i18n_coverage.py`, Sphinx and the source-reading pytest cohorts. The two
catalog-builder CLIs bootstrap before reading production reviews. For imported
tools, standalone checkers and local pytest, run the explicit command above.

Installed spaCR, its bundled runtime catalogs, wheels and frozen applications
continue to use their packaged outputs; they do not fetch editorial inputs at
runtime. The generated API catalogs remain separately pinned in
`docs/media_revision.json`, group `api_memory`, on Hugging Face. Moving editable
review records does not move those catalogs or change that pin.

## Author and publish a reviewed change

Clone the translation repository into a writable checkout. Preserve each review
record's source/context hashes, attribution, review kind and retired history.
Obtain the required technical/language review against the exact English source;
AI peer review must not be described as native-speaker or human signoff.

The existing record writer can target that editable checkout while resolving
source blocks against this spaCR checkout:

```sh
TRANSLATION_CHECKOUT=/absolute/path/to/spacr-translations
python tools/write_reviewed_api_record.py --language de \
  --label 'spacr.qt.job_runner#3' --show
python tools/write_reviewed_api_record.py --language de \
  --label 'spacr.qt.job_runner#3' --name reviewed-change \
  --translation-file reviewed-de.txt \
  --review-root "$TRANSLATION_CHECKOUT/docs/i18n/reviewed"
```

The translation file and filename above are author-supplied examples. Do not
reuse a source label for a different paragraph. The writer also supports its
existing runtime `--table`/`--key` lane with the same external review root. New
review metadata and retirement decisions still need author/peer verification.
The CLI requires an explicit external review root once a source pin exists.

Regenerate the external input manifest after reviewing the final records:

```sh
python tools/bootstrap_translation_reviews.py manifest --source "$TRANSLATION_CHECKOUT"
```

Commit the intended review changes and `manifest.json` in the translation
repository, then publish that repository's commit. Record its full SHA and pin
it in the spaCR source:

```sh
python tools/bootstrap_translation_reviews.py pin \
  --revision "$(git -C "$TRANSLATION_CHECKOUT" rev-parse HEAD)"
```

`pin` verifies the published commit and manifest before updating metadata. It
does not alter hydrated or editable reviews. Validate the new pin in a fresh
spaCR worktree; existing copies from an older pin intentionally fail instead of
being overwritten. Preserve any local changes in the external editable checkout
before discarding an obsolete hydrated copy. Never use cleanup/reset on a shared
checkout to resolve this state.

Run the normal affected evidence checks and generators. API changes use
`python tools/build_documentation_i18n.py --assemble` and the existing
`tools/fetch_docs_media.py publish --group api_memory` only when generated API
outputs actually change; that publication still requires its normal credentials.
Runtime and Help outputs use their existing owning generators. Review and commit
the pin plus affected product outputs, then publish spaCR to nightly through the
normal integration owner. The external review commit, source pin and generated
outputs must describe the same accepted source/target change.
