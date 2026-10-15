# Web mask generator backend components (N698)

`results.build_result_zip` assembles a download from **real, caller-supplied
2-D segmentation outputs**. It does not load weights, run inference, upload
images, implement a web UI or claim desktop/browser parity.

The caller must supply the original uploaded file, the selected YX or CYX
image plane, its nonnegative integer label mask, the exact model ID/version,
a weights SHA-256, and the effective settings. Pass actual `(2, Y, X)` flow
vectors and `(Y, X)` backend scores. Set `probability_kind="logits"` for raw
Cellpose scores, rather than labelling them calibrated probabilities. For an
unavailable optional output, omit its array and supply an explicit reason in
`unavailable`. A detector cannot use this segmentation-only contract.

The ZIP contains the original uploaded bytes, selected-plane TIFF, uint32
label TIFF, actual optional arrays as pickle-free NumPy files, object CSV,
README and a provenance/hash manifest. CSV fields use pixels, zero-based
centroids and original selected-channel intensity means, through
`spacr.tabular`. Sparse large label IDs are preserved without allocating
measurement structures indexed by the maximum label.

The upload bytes are included intentionally; the caller must disclose this to
the user before packaging. No files leave the machine in this component. Do
not place access tokens or personal secrets in effective settings or model
metadata. ZIP publication is atomic: writer/disk failures preserve a previous
download. Temporary staging is cleaned on ordinary exceptions; abrupt process
termination can leave a private temporary folder for the service to reap.

Remaining N698 work includes model-class routing, actual inference integration,
image/PDF/multichannel readers, both chosen GPU paths, browser exports and
postprocessing, accessible localized site UI, random approved HF examples,
privacy/size/quota controls, hosting, and desktop/GPU/browser parity acceptance.

`inputs.describe_upload` reads upload choices from headers, and
`inputs.read_upload_plane` returns `image`, `image_axes`, and a `selection`
receipt suitable for recording under the effective settings. TIFF/OME-TIFF
uses authoritative axes, with explicit indices for every non-singleton T/Z or
other nonspatial dimension. Channels retain their chosen order; no projection,
normalization, RGB blending or automatic orientation is applied. PNG/JPEG/BMP
retain the actual decoder values and supported ordinary color modes. PDF pages
render through optional `pypdfium2` at the explicitly requested DPI, with a
receipt identifying these as page renderings, not calibrated acquisitions.

Defaults bound uploaded files to 64 MiB, decoded images to 256 MiB, selected
planes to 25 million pixels and PDFs to 256 pages. TIFF currently decodes a
whole selected series, so its preflight uses that series' complete decoded
size, even if only one T/Z plane is requested. Selection and size refusals
occur before TIFF decode or PDF render. Unsupported palette/multiframe ordinary
images and unsupported vendor reader versions are refused explicitly.

Python dependencies are the existing spaCR NumPy, pandas, tifffile,
scikit-image and Pillow packages, with `pypdfium2` required only for PDF input, scene-aware `czifile` for CZI,
and `nd2reader` for supported ND2 versions.
No dependency is installed automatically by these components.


Optional microscopy adapters retain explicit scene/position, T/Z and channel
choices. Current scene-aware czifile reads CZI scenes without blending them.
ND2Reader reads the selected position/time/Z/channel frames; reader-decoded
pixel values/dtypes are preserved. The decoded-byte preflight remains
conservative against the described whole series. Unsupported ND2 metadata
without declared X/Y dimensions is refused, and legacy czifile APIs without
scene metadata are refused rather than guessing scene identity. CPU acceptance
uses repository CZI fixtures and an already-cached public OME ND2 fixture;
CI skips the latter if unavailable, and never downloads it implicitly.

`plaque.run_plaque_download` wires the readers and packager to the actual
existing desktop plaque loader and `spacr.plaque.segment_plaque_image`.
An operator must provide a trusted local checkpoint, its expected SHA-256,
model identity/version, and already-resolved effective `diameter`,
`flow_threshold` and `CP_prob`. Weight bytes are verified before loading; no
weight download or alternate default selection is implemented. `diameter=None`
explicitly selects native scale; the caller must resolve the N696
magnification/Auto choice before invoking this adapter.

The existing desktop path performs inference once; the adapter captures its
actual raw vector field and cell-score logits from that same return value for
the ZIP. It does not substitute the RGB flow visualization for vectors, create
missing outputs, or alter labels. Input-selection/effective-setting provenance
is included. A future HTTP/Gradio layer must select trusted operator-controlled
checkpoints and enforce server limits itself; do not accept visitor filesystem
paths or forward arbitrary request dictionaries as loader/limit settings.

CPU adapter tests inject model construction/evaluation and exercise the real
desktop segmentation helper and ZIP path. They prove call/format/integration
contracts only. Actual trusted checkpoint availability, hosted GPU behavior,
scientific segmentation accuracy and desktop/browser IoU parity remain
unverified and required for N698 acceptance.

Run the backend locally as an operator (no service is started):

```bash
python -m tools.web_mask_generator.run uploaded.ome.tif --describe
python -m tools.web_mask_generator.run uploaded.ome.tif \
  --output result.zip --checkpoint /trusted/models/cpsam_plaque_r6 \
  --sha256 744aee3f71efa68f6a369a6f91fe5850dc61e35f3dc20d53dda2c04fce17b35f \
  --model-id toxoplasma_plaque_v4 --model-version v4 \
  --diameter 52 --index T=0 --index Z=1 --channel 0
```

The hash above is the existing bundled model catalogue's plaque-v4 entry, not
a claim that the checkpoint is installed or was run here. Choose the diameter
from the effective desktop settings for the image; `52` is only a command
example. Use `--native-scale` only when deliberately requesting native scale.
Duplicate axis choices, missing model provenance and implicit diameter defaults
are rejected. Zero-based `--series`, `--index AXIS=N`, ordered `--channel`, and
PDF `--page`/`--pdf-dpi` mirror the Python reader. The command prints the final
ZIP path only after successful publication. Deployments need a separate web
service/UI and both required GPU paths; this CLI does not provide those.

`registry.model_families()` exposes the eight current non-alpha web model
classes and all of their published versions. It reuses the existing model-card
family/default rules and current rescored card metrics, including alternatives,
rather than ranking historical raw catalogue numbers. Detector classes advertise
boxes; segmentation classes advertise labels. `inference_adapter` is currently
`"plaque"` only for the plaque family and `null` for the remaining families.
Metadata exposure is not a claim that weights or inference are available.

`registry.resolve_trusted_model(family, provisioned_paths, model_key=...)`
resolves only operator-provisioned model keys and verifies exact published
weight bytes. Public metadata contains no filesystem paths. Selecting an
unavailable default refuses rather than silently using an older installed
model. The operator mapping is private service configuration, never a visitor
request field. Existing alpha models remain excluded, and published dataset
repository types/URLs retain their actual type.

`service.PlaqueService` keeps the operator checkpoint mapping read-only and
allows only explicit visitor model/settings/selection/consent fields. Client
paths, output folders and resource-limit overrides are refused. The `job`
context copies bounded upload bytes into a private mode-700 folder and removes
that folder after delivery, inference failure or a transfer/consumer exception.
A web framework must serve the yielded ZIP during the context or copy it into
its own disclosed download cache before context exit. This component does not
claim that a future framework's upload/download caches are already configured.

`preload_default()` is an explicit operator startup action, not an import side
effect. It verifies the current default's exact published weight bytes and
warms a two-entry digest-keyed desktop model cache, supporting the documented
ZeroGPU startup-placement pattern. Replacing a path with different weight bytes
cannot reuse the old cached model under the new digest. Plaque requests with
more than three selected channels are refused before loading, preventing
Cellpose from silently ignoring additional channels.
