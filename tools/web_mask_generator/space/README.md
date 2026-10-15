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
Linux desktop/icon data inputs. Do not clone or upload repository history,
private data, tutorials or credentials.

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

This source does not download weights automatically. Before enabling inference,
provision the exact approved model files from their actual public model/dataset
repository type and immutable revision, independently verify the catalogue
SHA-256, and set `SPACR_WEB_MODELS_JSON` to the operator-controlled local paths.
An empty mapping can construct the UI but cannot process a model request. All
advertised model versions require their own verified files; do not imply that a
missing version is available or silently substitute another one. Public examples
remain disabled until genuine reviewed test-split records are supplied through
`SPACR_WEB_EXAMPLES_JSON`.

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
