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
