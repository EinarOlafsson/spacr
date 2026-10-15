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
