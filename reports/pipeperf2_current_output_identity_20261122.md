# Current-tree pipeline output comparison

Base: `b9af1a272ba6f01a3aa8f6fd732e45c4cd2f0f3e`.

The two runs used the same physical private checkout and 10,963 byte-identical original input files. Both ran through sched at P1, four threads and 16 GiB RAM, with model imports blocked during runtime catalog assembly. No translation was generated.

Timings are one wall-clock measurement per variant on the shared workstation, excluding input downloads. They are measurements of these runs and do not establish a stable benchmark.

Each source-extraction operation includes three consecutive calls in one process and asserts that all three results agree. This measures the cold call followed by two repeat calls. Catalog assembly and each remaining operation run once per variant.

| Operation | Before (s) | After (s) | Before / after |
|---|---:|---:|---:|
| api-catalogs | 457.196 | 208.809 | 2.19 |
| api-source-extraction | 95.770 | 20.453 | 4.68 |
| caption-arrivals | 136.371 | 62.278 | 2.19 |
| coverage-and-review-scope | 973.640 | 262.812 | 3.70 |
| nested-helper-inventory | 26.446 | 13.936 | 1.90 |
| runtime-audit | 82.988 | 71.992 | 1.15 |
| runtime-catalogs | 348.564 | 252.315 | 1.38 |
| runtime-source-extraction | 28.061 | 12.045 | 2.33 |
| nested-helper-cli | 18.889 | 14.404 | 1.31 |

| Source extraction call | Before (s) | After (s) | Before / after |
|---|---:|---:|---:|
| runtime cold | 12.100848 | 10.427820 | 1.16 |
| runtime repeat 1 | 8.490225 | 0.753487 | 11.27 |
| runtime repeat 2 | 7.451525 | 0.849764 | 8.77 |
| api cold | 33.940660 | 20.161808 | 1.68 |
| api repeat 1 | 30.744895 | 0.128572 | 239.13 |
| api repeat 2 | 30.940329 | 0.104075 | 297.29 |

30 of 31 output files are byte-identical.

The assembly cache stamp includes SHA256 of the generator source in its input fingerprint. Its `inputs` value changes when the generator changes; its locale list remains identical. The proof retains both raw stamp hashes and leaves this invalidation rule unchanged.

| Output | Before SHA256 | After SHA256 | Identical |
|---|---|---|---|
| `canonical-sources.json` | `15ff2a2627a443f55a50cd82171dc76f8e3370ac8a98a9876f19753cd02faa8e` | `15ff2a2627a443f55a50cd82171dc76f8e3370ac8a98a9876f19753cd02faa8e` | yes |
| `docs/i18n/COVERAGE.md` | `2057475e8ca12cc9960b836168d15c718fedfe0d22e11c1c320845d8e4f245f2` | `2057475e8ca12cc9960b836168d15c718fedfe0d22e11c1c320845d8e4f245f2` | yes |
| `docs/i18n/REVIEW_SCOPE.md` | `aa5905377fb43ad663d4381dfc4e0b74e8c403c63838856199b89b7728f2c6da` | `aa5905377fb43ad663d4381dfc4e0b74e8c403c63838856199b89b7728f2c6da` | yes |
| `docs/i18n/api_source_history.json` | `8df21d044cc6e0ef4d2042c802e389972ef61bd51404eca0fda630bad8d7dc10` | `8df21d044cc6e0ef4d2042c802e389972ef61bd51404eca0fda630bad8d7dc10` | yes |
| `docs/source/_static/i18n/.api-assembled.json` | `2a8d1cd177a04cee38c80d158a87d1bc3b332eeed583987a2b3fd097f7721b9b` | `2bdfd80098bf9830b1bbb0ed8050aac7ea65c7bb44da2cd8cb4ddfae39e70f97` | no |
| `docs/source/_static/i18n/api/de.json` | `fd7852f18fc7f0876c44203aa60ce556d8bdca7d04d3ead8ac8fe1fe24b7ac69` | `fd7852f18fc7f0876c44203aa60ce556d8bdca7d04d3ead8ac8fe1fe24b7ac69` | yes |
| `docs/source/_static/i18n/api/en.json` | `3d6ddc8a56a7ff2b38ae065e16ab96dd7b42dfbcf521c8c096cef037873fe69c` | `3d6ddc8a56a7ff2b38ae065e16ab96dd7b42dfbcf521c8c096cef037873fe69c` | yes |
| `docs/source/_static/i18n/api/es.json` | `1424a420581ab3d2811df5e870ab5d764e659a18a7cf536f1ddb82849c028a3c` | `1424a420581ab3d2811df5e870ab5d764e659a18a7cf536f1ddb82849c028a3c` | yes |
| `docs/source/_static/i18n/api/fr.json` | `e99091877c7912ef03fdbbd0360d0cea4fad52d759e5b9a41818c2d3d18d2779` | `e99091877c7912ef03fdbbd0360d0cea4fad52d759e5b9a41818c2d3d18d2779` | yes |
| `docs/source/_static/i18n/api/hi.json` | `353fd54106e53d6ed3dc464b7b45da621c32e69955b4300ab901cfaeaafb2ce1` | `353fd54106e53d6ed3dc464b7b45da621c32e69955b4300ab901cfaeaafb2ce1` | yes |
| `docs/source/_static/i18n/api/is.json` | `f4e6c405dfa8b36895890f4011917976730dd4a2b6695ae8f136048933d8919d` | `f4e6c405dfa8b36895890f4011917976730dd4a2b6695ae8f136048933d8919d` | yes |
| `docs/source/_static/i18n/api/ko.json` | `9eba3a17cf3cc6a2f0c00319d5e02317ddb22bc9a9fef1ea954a7c9528f895de` | `9eba3a17cf3cc6a2f0c00319d5e02317ddb22bc9a9fef1ea954a7c9528f895de` | yes |
| `docs/source/_static/i18n/api/pt.json` | `5498fa5d2d8ddca5fe514cefccfa6e8bb923316c076a13a1c850346007bb6962` | `5498fa5d2d8ddca5fe514cefccfa6e8bb923316c076a13a1c850346007bb6962` | yes |
| `docs/source/_static/i18n/api/sv.json` | `9e7c970c451d18ceabbfb49345b776d13a56affa36f3c4667bf4df576c3e49d8` | `9e7c970c451d18ceabbfb49345b776d13a56affa36f3c4667bf4df576c3e49d8` | yes |
| `docs/source/_static/i18n/api/zh_CN.json` | `a01758077207bac67b11782ddb67ffa10eb3778308148b26f495af2c1bfe78eb` | `a01758077207bac67b11782ddb67ffa10eb3778308148b26f495af2c1bfe78eb` | yes |
| `nested-helper-cli.json` | `12bfc8761ff90c0442b6213b6236c87c7d85213c56951155fbe7dcc758fe6854` | `12bfc8761ff90c0442b6213b6236c87c7d85213c56951155fbe7dcc758fe6854` | yes |
| `nested-helper-inventory.json` | `3f9d3ead56f84888bc99612bc9233c0cf94dbba917c3370f6959071e26ace3ad` | `3f9d3ead56f84888bc99612bc9233c0cf94dbba917c3370f6959071e26ace3ad` | yes |
| `public-docstrings.json` | `7b4b7775dbeef1a2bc62a17be358215200ad2375c1b0adfcf1c9a15860ae56c5` | `7b4b7775dbeef1a2bc62a17be358215200ad2375c1b0adfcf1c9a15860ae56c5` | yes |
| `runtime-audit.txt` | `b1fcbc33dd50b66e99bc21bd8e80665f665735d09fd7e5f1885eb7df2267cffd` | `b1fcbc33dd50b66e99bc21bd8e80665f665735d09fd7e5f1885eb7df2267cffd` | yes |
| `spacr/qt/i18n_catalogs/__init__.py` | `ee9abc0a6a3caacf3332b06897edcd48c54c1fbf371a236ca894c4d598c1398f` | `ee9abc0a6a3caacf3332b06897edcd48c54c1fbf371a236ca894c4d598c1398f` | yes |
| `spacr/qt/i18n_catalogs/de.py` | `7bbbe874bbbf0c5cf36f8077d22d761c3dd3150c5a8b0cce8886d45551fcb409` | `7bbbe874bbbf0c5cf36f8077d22d761c3dd3150c5a8b0cce8886d45551fcb409` | yes |
| `spacr/qt/i18n_catalogs/en.py` | `0a3b9eaebcb0734fdb8ebb25b1113adb5341316aaf9b142e4450bf14da71c5d4` | `0a3b9eaebcb0734fdb8ebb25b1113adb5341316aaf9b142e4450bf14da71c5d4` | yes |
| `spacr/qt/i18n_catalogs/es.py` | `3880e03a7bc29c88266617a1872dda09af47762a43dd3e04eb65161e14e8c3b7` | `3880e03a7bc29c88266617a1872dda09af47762a43dd3e04eb65161e14e8c3b7` | yes |
| `spacr/qt/i18n_catalogs/fr.py` | `5fb3d65d52d03cf4ac0014849441feb91ec9d5bb3aa61f157ef97c68b11d66da` | `5fb3d65d52d03cf4ac0014849441feb91ec9d5bb3aa61f157ef97c68b11d66da` | yes |
| `spacr/qt/i18n_catalogs/hi.py` | `c92c34f447d99e4208c69e08841045d6166a7318bba70305ac874d1920a20f87` | `c92c34f447d99e4208c69e08841045d6166a7318bba70305ac874d1920a20f87` | yes |
| `spacr/qt/i18n_catalogs/is.py` | `0b7cbc7f5a2c63a13ce6e12d9182406ce4fe170a804406bc9e193b72c5ccedd0` | `0b7cbc7f5a2c63a13ce6e12d9182406ce4fe170a804406bc9e193b72c5ccedd0` | yes |
| `spacr/qt/i18n_catalogs/ko.py` | `df8dd942434b3e73089fe3b835723b71c324129b0e7ef37823f02fd5be41f063` | `df8dd942434b3e73089fe3b835723b71c324129b0e7ef37823f02fd5be41f063` | yes |
| `spacr/qt/i18n_catalogs/pt.py` | `19b9105372ecb5a34e18717878c066b8931aa3ed2fbd542adeda60aa54b9808e` | `19b9105372ecb5a34e18717878c066b8931aa3ed2fbd542adeda60aa54b9808e` | yes |
| `spacr/qt/i18n_catalogs/sv.py` | `d9f2f74d4501bcce31e5d5beb239cbd0537b373efb42013461da1051a3f5a4fb` | `d9f2f74d4501bcce31e5d5beb239cbd0537b373efb42013461da1051a3f5a4fb` | yes |
| `spacr/qt/i18n_catalogs/zh_CN.py` | `4a4096273704ff35c5772bb2e369ae9b10268c5078cfac04b8298bab1158c175` | `4a4096273704ff35c5772bb2e369ae9b10268c5078cfac04b8298bab1158c175` | yes |
| `tests/data/release_contracts/runtime_caption_arrivals_2026_10_08.json.gz` | `cb39a170551677304fd9ac7b656c39affd161831e1297be7e4322eed16eb8054` | `cb39a170551677304fd9ac7b656c39affd161831e1297be7e4322eed16eb8054` | yes |

## Failed stages

```json
{
  "before": {},
  "after": {}
}
```

## Validation on the landed pilot

Base: `b9af1a272ba6f01a3aa8f6fd732e45c4cd2f0f3e`.
Source performance branch: `40c345affd06f0275c4bd6006bb76778273d9666`.

All 157 required source ratchets passed in 155.82 seconds. All
154 focused caption-owner, callback-cache, nested-helper and API
extraction tests passed in 203.57 seconds: 311 distinct passing
tests, no skips or failures. The callback cases include the newly landed
`_ngff06_runtime_sources` and require a replacement callback to refresh an
already inventoried real source tree.

Both runtime audits returned zero. Exact source snapshots preserve all
16,619 runtime records and 14,961 API bodies. Both generated
coverage reports retain API freshness 14906/14961 per locale. No translation
was authored or added to a shipped catalog.

The full current 422-line per-file loop is preserved. Its only transformations
are reading the passed source text and returning the file set at six former
outer-loop continues. The entire original outer prefix and suffix remain
AST-identical, including every scan path and trailing owner callback. The
uncached public docstring parser is also AST-identical to the new baseline.
The source-preservation JSON records all six final tool hashes.

The complete builder change from the previous base `087c132313` to the new
landed pilot was reviewed; its SHA256 is `49b1b125ffb030e7462bb907e61c940590caa8456767490bdb49bd3c7b3809f2`. Retained changes
include:

- NGFF 0.6 owners in ome_zarr.py, image_store.py and ops_engine.py, plus the
  unchanged current manual setting-label records.
- The never_barcodes owner and every current never-calibration owner in
  ops_decode.py, including _never_settings.
- Flat-field, painting, crop, cytoskeleton and segmentation-input owners.
- Every earlier diameter, mask-worker, recorded-batch, saved-run,
  classification-pixel and measurement owner in the current body.

The cache policy adds the new NGFF callback dependency. Application code,
generated catalogs, translation pins and workflows are unchanged in this
branch. All generated proof files, original input hashes, raw logs and
reproduction scripts remain under
`spacr-codex-recovery-20261121/pipeperf2/` in the workspace.
