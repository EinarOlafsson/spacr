---
title: spaCR mask generator
sdk: gradio
sdk_version: 6.17.3
python_version: 3.12.12
app_file: tools/web_mask_generator/space_app.py
---

This is the managed Gradio Space configuration for the reviewed spaCR service.
Copy this file and its adjacent `requirements.txt` to the root of the selected
source staging bundle. Keep the project's `README.rst`, `setup.py`,
`pyproject.toml`, `MANIFEST.in`, `LICENSE`, all selected `spacr` files, the web
folder, `tools/build_model_cards.py`, `tools/build_score_cards.py` and setuptools'
Linux desktop/icon data inputs. Do not upload development-repository history,
private data, tutorials or credentials.

The managed build installs requirements before copying the Space source into
`/app`, so a local `.[plaque]` requirement cannot resolve there. Its pip Git
partial clone also failed against the public Hub. Requirements therefore use the
ordinary spaCR source distribution at immutable public Space revision
`85b41cf7084471679f8c1076df8ee4f0b18b09b6`, with independent SHA-256
`a2068b97a8973a64fa13febbd81b87e933befb7684779be50d356fe969ce457f`.
The HTTPS URL keeps the `plaque` extra and normal dependency resolution, with no
Git checkout, runtime pip invocation or dependency bypass. Normal pip validates
the archive's hash before building it; changed or unavailable bytes refuse.

The source archive was produced by ordinary `setuptools.build_meta.build_sdist`
from the reviewed minimal public source snapshot `66ac681d3040aab1823aed0bf16b4ed5c89589f7`
(code `35aeaa9ddc1f09cbd85fe75617a68df81adade0f`). It contains all 664 runtime
Python files, 969 unchanged original package files and eight normal generated
metadata files. Existing distribution policy prunes review generators, installer
copies and other nonpackage inputs; the archive inventory records those choices.
It contains no repository history or checkpoint weights. Its setup/build backend
does not read the old managed requirements file included by normal distribution
policy, so installation does not recurse. Required setup/README/pyproject and Linux
packaging data are included. The package metadata retains its actual version
1.5.1.4; the source snapshot is explicit and does not imply a release or hosted
acceptance. A future source/dependency change requires another reviewed archive,
immutable uploaded revision and checksum; do not replace the URL with a branch.

The Space's managed SDK supplies Gradio and spaces. Do not also include the
local-development `requirements-space.txt`, which targets a different SDK.
Gradio 6.17.3 requires Hugging Face Hub >=0.33.5,<2; the pinned Hub 0.36.2 also
satisfies spaCR's existing <1 bound. No project dependency bound is widened.
Torch 2.13.0 and Torchvision 0.28.0 are the selected matching pair; actual managed
build, CUDA placement and inference acceptance remain necessary.

Choose an eligible FREE personal ZeroGPU Space and select ZeroGPU hardware in
its actual settings. README metadata does not allocate hardware. No paid GPU,
PRO upgrade or paid fallback is configured. Keep the existing model-upload Space
separate. Set `SPACR_WEB_ZERO_GPU=1` only after selecting real ZeroGPU hardware.

Ordinary local startup does not download weights. Managed startup downloads
only when BOTH `SPACR_WEB_ZERO_GPU=1` and `SPACR_WEB_PROVISION_MODELS=1` explicitly
opt into the verified provisioning described below; leave `SPACR_WEB_MODELS_JSON`
unset for that route. Alternatively, without the provisioning opt-in, the
operator may provision the exact approved public model/dataset files at immutable
revisions, independently verify the catalogue SHA-256, and set
`SPACR_WEB_MODELS_JSON` to those trusted local paths. An empty mapping can
construct the UI but cannot process a model request. All advertised model versions
require their own verified files; do not imply that a missing version is available
or silently substitute another one. Public examples remain disabled until genuine
reviewed test-split records are supplied through `SPACR_WEB_EXAMPLES_JSON`.

Uploads require explicit consent. The result ZIP includes the original upload.
Job staging is removed when the result is copied to the download cache; cache
cleanup is configured for one hour, checked every five minutes. Public Gradio
sharing stays off, backend tracebacks stay hidden, strict CORS stays enabled,
and uploads remain limited to 64 MB. The pinned 6.17.3 SDK predates Gradio's run
history feature; only its unsupported `run_history` launch argument is omitted.
Other SDK versions retain explicit `run_history=False`.

Deployment and scientific parity are not established by a successful constructor
or source test. Record the actual Space/runtime identity, inspect its generated
API schema, exercise real inference/cancellation/cache cleanup and verify the
required desktop agreement before advertising a hosted service. Hand the actual
application URL and all 24 ordered inference inputs to the website owner; this
configuration contains no invented deployment URL.

### Explicit managed provisioning

The source bundle carries `models.json`, generated by the ordinary provisioning
preparation command from reviewed public Hub metadata. Its immutable revisions,
exact filenames/byte sizes/checksums, current advertised model set and relevant
source hashes must all agree. Rebuild it after changing its bound source inputs:

    python -m tools.web_mask_generator.provision \
      --prepare-manifest /operator/reviewed-public-weight-metadata.json \
      --output tools/web_mask_generator/space/models.json

This preparation command reads metadata only; it never downloads checkpoints.
To acquire weights at actual managed startup, set BOTH `SPACR_WEB_ZERO_GPU=1`
and `SPACR_WEB_PROVISION_MODELS=1`. Keep `SPACR_WEB_MODELS_JSON` unset; a
contradictory existing mapping refuses. Early spaces initialization remains
first. Only execution as the Space entrypoint invokes provisioning, and CPU,
ordinary local launches and importing the entrypoint never acquire files.

The private cache defaults to `~/.cache/spacr/web-managed`; the operator may
set `SPACR_WEB_PROVISION_CACHE` before startup. Do not point it at an existing
user data/model directory. Foreign/corrupt/symlink cache entries refuse instead
of being overwritten or deleted. A cache lease excludes concurrent startups.
All advertised versions must finish verification before the app receives a paths
map; partial acquisition cannot start an apparently complete inference service.
Verified files from an interrupted attempt can be reused by a later explicit
startup. Acquisition uses only public HTTPS Hub/CDN URLs, no ambient Hub token,
fixed immutable revisions and byte/checksum limits. Unknown redirects refuse.
No retry, default substitution or model execution occurs during provisioning.

The current seventeen weights total about17.1GB (decimal), in addition to the
source/environment. Available disk must cover missing files plus a one-file
staging margin. Each checkpoint is capped at2GiB, total declared data at24GiB,
and transfer work checks a twenty-minute startup deadline between bounded reads
(each underlying request/read also has an at-most60-second inactivity timeout).
These are resource bounds, not a measured hosted startup-time or capacity claim.
Operator-only startup failures use structured machine diagnostics. No personal
images are involved in this bootstrap. Actual free-host build, cache/queue/CUDA
behavior, scientific agreement and site integration still require acceptance.
