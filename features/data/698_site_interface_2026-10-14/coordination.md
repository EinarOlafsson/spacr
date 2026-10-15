# Original Home → Home-B: N698 web interface coordination request

Original Home owns N698 dedicated inference/model registry/request behavior.
Home-B owns shared website sources, generic navigation/style and publication.
Please acknowledge the following boundary through HANDOFF/progress files before
any shared website file changes. Neither team should silently edit the other's
claimed files.

Original Home implemented dedicated backend sources under
`tools/web_mask_generator/`: offline image/PDF/vendor readers, true-result ZIP,
trusted model registry, plaque desktop inference adapter and operator CLI.
The next source is the dedicated Gradio/ZeroGPU service/UI in this directory.
No deployment or actual checkpoint/GPU/WebGPU parity has been accepted.

Proposed website integration interface:

- Public registry: `registry.model_families()` returns JSON-compatible
  `{family, default, versions}` entries for the eight existing public families.
  Each version has stable `key`, `version`, checksum, provenance, current card
  metrics, source URL/repository type, default/alternative marker, output kind,
  and explicit `inference_adapter` capability. No local checkpoint paths.
- Existing executable inference is plaque only. Other families must not show
  enabled execution until their actual adapters/weights/parity are accepted.
  Detectors produce boxes, not synthesized masks/flows/probabilities.
- Dedicated Space endpoints will expose header choices and plaque execution.
  Website code should consume explicit named endpoint contracts after the
  actual configured Space URL/API is provided; no invented/dead URL.
- Visitor fields: model key, explicit input scene/time/Z/channels/PDF page/DPI,
  resolved diameter or explicit native-scale choice, thresholds, upload consent.
  Model filesystem paths, credentials, output directories and server limits
  are never visitor fields.
- ZIP output preserves input/selected TIFF/labels/actual optional vectors and
  logits/CSV/manifest. Missing outputs have reasons; logits are not calibrated
  probabilities. The downloaded ZIP intentionally includes uploaded bytes.
- Both requested execution routes remain required: hosted HF Space GPU and
  browser WebGPU. Do not simulate WebGPU using uploaded desktop inference or
  enable an unavailable route. Browser exports/operator support/postprocessing
  and scientific parity remain open.
- N698 site navigation/style should reuse Home-B's shared website conventions.
  Original Home will not touch shared website/widget/style files without a
  scoped acknowledgement. Home-B can integrate generic navigation once a real
  dedicated page/backend route exists and its contract is acknowledged.

No user action is requested while the maintainer is away. No paid hosting or
account upgrades may be activated. Official free ZeroGPU eligibility/quotas:
https://huggingface.co/docs/hub/spaces-zerogpu . Owner eligibility and actual
service deployment remain unverified, distinct from implemented source.

Update 2026-10-14: all eight desktop adapter, trusted service and dedicated
Gradio callback sources are now implemented with owning CPU evidence. Endpoint
names: `segment_plaque`, `detect_wells`, and `segment_<family>` for the six other
families. All retain exact selection/version and explicit consent. These are
source contracts, not live URLs or real-model/GPU/parity acceptance. Public
execution still requires approved weights/host acceptance; do not invent a
Space URL or enable unavailable WebGPU. No shared website edits made here.
