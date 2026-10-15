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
scikit-image and Pillow packages, with `pypdfium2` required only for PDF input, `czifile` with scene or subblock metadata for CZI,
and `nd2reader` for supported ND2 versions.
No dependency is installed automatically by these components.


Optional microscopy adapters retain explicit scene/position, T/Z and channel
choices. Current scene-aware czifile reads CZI scenes without blending them.
ND2Reader reads the selected position/time/Z/channel frames; reader-decoded
pixel values/dtypes are preserved. The decoded-byte preflight remains
conservative against the described whole series. Legacy czifile readers use
explicit subblock scene bounds and retain absolute origins. Since those readers
decode the complete multi-scene canvas, its full declared allocation must fit
the byte budget before scene metadata expansion or any pixel decode. Legacy
metadata also has an independent positive-integer `max_scenes` limit, default
4096 distinct scenes, checked before expanding subblock scene ranges. Repeated
subblocks for one scene do not count twice. Both `describe_upload` and
`read_upload_plane` accept an explicit limit override. This scene-count limit
applies only to the legacy reader route; the current CZI pixel route is unchanged.
Scene/time/Z/channel selection
never blends scenes or guesses missing metadata. Unsupported ND2 metadata
without declared X/Y dimensions and CZI without scene or subblock metadata
are refused. CPU acceptance
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
effect. It validates every configured public key and its exact published weight
checksum before constructing any model. It warms at most one configured
segmentation model: the current plaque default when provisioned, otherwise the
first supported configured segmentation key in lexical order. The existing
plaque or Mask digest-keyed cache retains that model; no other family is eagerly
loaded. An empty or detector-only configuration returns `None` without loading
an unrelated plaque model. Detectors keep their request-scoped desktop loader.
An unknown key, missing checkpoint or wrong checksum refuses startup before
any warm-up. This policy does not change request defaults: an unprovisioned
current default still refuses instead of using a warmed older version. It does
not certify actual ZeroGPU placement or cache lifetime; those require checks
on the intended host. Plaque requests with more than three selected channels
are refused before loading, preventing Cellpose from silently ignoring
additional channels.

`app.build_app(...)` constructs a real Gradio interface and named `models` /
`segment_plaque` API routes without starting a server or loading weights.
It has all eight model-family/version/provenance tabs; only the implemented
plaque adapter is executable. Users grant consent before the file picker is
enabled, choose actual channels/series/T/Z/PDF page, explicitly resolve diameter,
and receive the result ZIP plus a display-only mask-boundary preview. The queue
has one model worker and eight pending slots. Cancel addresses pending work;
it does not claim to interrupt an already-running CUDA kernel immediately.

`delete_cache=(300, 3600)` configures Gradio to reap registered upload/download
cache files older than an hour on its five-minute cleanup interval. Original
private job staging is removed immediately after the result is copied into
that cache. No input is sent to GitHub, training datasets or a model provider
by these handlers. Hosted processing necessarily transfers uploads to the
Space; the ZIP intentionally includes those bytes. Browser WebGPU remains a
separate unimplemented route, explicitly disclosed in this source UI.

Source-only Space launch preparation (no deployment has occurred):

1. Use an eligible **free** personal ZeroGPU Space. Verify the email/account-age
   requirements in [HF's official ZeroGPU guide](https://huggingface.co/docs/hub/spaces-zerogpu).
   Do not upgrade to paid hardware, PRO or an organization plan for this setup.
   If free ZeroGPU is unavailable, stop deployment and retain these sources;
   that eligibility gap is not solved by selecting paid hardware.
2. Supply a reviewed spaCR source checkout as the Space repository and install
   that checkout's project requirements. The additional UI requirements are
   `tools/web_mask_generator/requirements-space.txt`. They were checked with
   Gradio 6.30.0 / spaces 0.51.3 in a private local environment; the actual HF
   build, CUDA/checkpoint compatibility and scientific acceptance remain open.
3. The Space's root README metadata should use `sdk: gradio`,
   `sdk_version: 6.30.0`, `python_version: 3.12`, and
   `app_file: tools/web_mask_generator/space_app.py`. The supplied wrapper adds
   the source root for imports; ordinary direct execution of the relative-import
   library file is not the Space entry point.
4. Provision the approved checkpoint on the service from its actual published
   repository, respecting repository type. Set the operator-only environment
   variable `SPACR_WEB_MODELS_JSON` to a JSON mapping of approved keys to trusted
   local checkpoint paths. No visitor may choose these paths. Runtime verifies
   bytes against the current bundled catalogue; no default download occurs.
5. With free ZeroGPU actually selected in the Space settings, set
   `SPACR_WEB_ZERO_GPU=1`. This requests startup verification/preload, then a
   `spaces.GPU(duration=120)` call for execution. Basic consent/header/diameter
   checks happen before entering the decorated function. Startup validates all
   configured identities and warms at most one segmentation model as described
   above; detector-only startup does not preload a model. Actual model placement,
   quota, queue and execution availability must be tested on the intended account
   against the linked official guidance.
6. Launch only after the source/configuration review. `main()` keeps public
   Gradio sharing off, hides backend tracebacks, disables run history, retains
   strict CORS, and limits uploads. No server was launched while implementing
   or testing this source. The generic spaCR site must wait for an acknowledged
   Home-B interface and a real configured endpoint, rather than inventing a URL.
7. Before whole N698 acceptance, implement the remaining class adapters and
   browser exports/postprocessing, complete localization/style/accessibility,
   exercise hosted cache cleanup/cancellation, and compare actual model outputs
   against desktop results under the original IoU/object-count criterion.

`detector.run_well_download` runs the actual desktop `detect_wells` path with
verified operator weights, retaining the desktop's RGB→BGR handling, confidence,
reading order and axis-ratio refusal. Choose one grayscale or three RGB input
channels. It produces a detection ZIP containing input/selected TIFF, actual
boxes/confidences, canonical CSV, README and manifest. Bounding-box area/centre
are explicitly rectangle measurements, not segmented well area/centroids;
physical calibration is not assumed. Masks, vectors and per-pixel cell scores
are explicitly unavailable and never synthesized. Backend integration into the
public service/UI remains separate work; no hosted detector is claimed here.

`masks.run_mask_download` implements the other six segmentation family adapters
through the canonical N696 diameter/settings helpers and desktop normalization,
Cellpose evaluation, mask filtering and field-size restoration. Choose exactly
one explicit source channel. Blank diameter uses the desktop magnification rule;
`auto`/zero requests the model's automatic/native diameter; a positive value is
an explicit pixel diameter. The original image remains unchanged for intensity
measurements. Filtering may renumber retained IDs, exactly as on the desktop.
The ZIP records requested/resolved/evaluated diameter and actual role-specific
normalization, thresholds and filters, without exposing checkpoint paths.
This adapter requires the updated N696 backend; CPU tests exercise its actual
helpers with injected model evaluation and establish no real-model parity.
Service/UI enablement for these routes remains a separate integration step.

All eight dedicated Gradio tabs now have callable hosted-route interfaces.
Segmentation routes show label boundaries; the well detector shows actual
bounding rectangles and offers confidence/axis-ratio controls. Plaque diameter
requires explicit resolved pixels or native scale. Other Mask routes distinguish
blank (magnification default), Auto and explicit pixels, and accept optional
JSON desktop normalization/filter overrides. Each panel owns its callbacks so
selecting one family cannot execute another family's late-bound closure.
This source UI has CPU component/callback coverage, not deployed GPU acceptance.
ZeroGPU model placement/cache lifetime for all approved weights, real quota
behavior and scientific parity remain to be accepted on the intended host.

The operator CLI supports `--family` for every registered class. For generic
Mask models, omit diameter to use `--magnification` (default 20), choose
`--native-scale` for Auto, or give `--diameter` in pixels. Optional
`--settings-json '{"background": 100, "remove_border_objects": true}'`
passes only desktop normalization/filter settings. For the well detector,
`--confidence`, `--min-axis-ratio` and `--detector-image-size` retain desktop
defaults 0.25, 0.7 and 640. Detector settings do not fabricate pixel masks.

Random public test images are configured with `SPACR_WEB_EXAMPLES_JSON`, a JSON
list of operator-reviewed records, each with exactly: `family`, `repo_id`,
`revision` (40 lowercase hex immutable DATASET commit), `filename` (relative
path), `sha256`, `bytes`, `credit`, `split` (`"test"`). Review actual test-split
membership, scientific input suitability and redistribution rights before
provisioning records; the code cannot establish those facts from a filename.
No example catalogue is invented or downloaded implicitly. Buttons stay
disabled for unconfigured classes. Downloads use no ambient account token,
stream within the approved byte count and verify the entire checksum/header
before returning a cached image and attribution. Public example staging is
removed on success/failure; returned files use the disclosed Gradio cache.

The common background floor, signal-to-noise anchor, lower percentile, perimeter
merge and border/background switches have ordinary controls under
“Normalization and object filters”. Blank numbers and “Desktop default” keep
the canonical role settings; explicit controls take precedence over optional
JSON values. The detector also exposes a strictly integral image-size control.

### Managed ZeroGPU deployment candidate

`space/README.md` and `space/requirements.txt` are the files to stage at the
root of a **new**, reviewed managed Space bundle; they are not an existing
service. They pin managed Gradio 6.17.3 / Python 3.12.12 and compatible Hub 0.36.2,
Torch 2.13.0 / Torchvision 0.28.0 without widening the project's Hub bound. The
managed SDK supplies Gradio/spaces; do not combine these with the local-development
`requirements-space.txt` pins. Gradio 6.30 requires Hub >=1.16 whereas spaCR
requires <1, so that earlier launch preparation does not describe a resolvable
managed environment. Only the exact 6.17.3 launch omits its unsupported
`run_history` argument; every other version retains `run_history=False`. All
other sharing, consent, upload, error and CORS safeguards remain the same.
Weight provisioning, actual free-hardware selection, hosted execution and
scientific acceptance remain explicit deployment work.
