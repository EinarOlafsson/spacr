F534 bounded paired-reference acquisition and registration-route check
2026-10-15

Reference dataset: Xenium_V1_hPancreas_Cancer_Add_on_FFPE, official 10x
post-Xenium H&E and IF of the same section. Original discovery/CC BY4.0
attribution: ../home-b-f534-paired-reference-20261015/owner-instructions.json.
No infected-tissue, Visium HD, marker identity or whole-slide claim.

Exact pixel acquisition:
- H&E full-resolution origin (10240,35840), size512x512 RGB uint8.
- IF full-resolution origin (32325,7612), size434x434 uint16, page/channel0.
- Both use one actual compressed1024x1024 tile from their official TIFF.
- Strict RangeGET206 and exact Content-Range; no full download/redirect fallback.
- Geometry, metadata and 4,115,621-byte total budget saved before pixel GETs.
- Actual transfer4,115,621bytes,2.56s/56,212KiB RSS,2G cap/thread1, no retries.
- The earlier failed IF OME-description requests were not retried. Channel0
  is not called DAPI or assigned a marker. No other channel was fetched.

Geometry source:
https://www.10xgenomics.com/support/software/xenium-explorer/latest/tutorials/xe-image-alignment
https://testkb.10xgenomics.com/s/article/35386990499853-How-can-I-convert-coordinates-between-H-E-image-and-Xenium-data
Each official3x3 matrix maps post-Xenium image pixels to Xenium image pixels.
H&E-to-IF is inverse(M_IF) @ M_HE. The plan records both full-image origins.

Actual production path (unmodified source76526f7a01):
- Four CSV columns source_x,source_y,target_x,target_y are read by the normal
  _st_read_landmarks -> spacr.tabular.read_table route.
- _st_register reads real H&E crop through a further local(32,32,448,448)
  TIFF window, and the real already-bounded IF user-image crop.
- A minimal adapter selects the existing Xenium registration branch. Its
  synthetic coordinate controls and pixel_size1.0 are a pixel-format fixture,
  NOT an actual transcript bundle or morphology-image import.
- Four supplied-matrix-derived fit pairs: RMS4.4558e-13 targetpixels.
- Four separate supplied-matrix-derived query controls: max5.2407e-13pixels.
- These tiny residuals are algebraic/format consistency only. None is an
  independently annotated anatomical landmark or held-out biological truth.
- Real input crops/source hashes unchanged; no threshold/algorithm changes.

Visual artifacts:
he-preview.png is the rawRGB crop. if-channel0-preview.png is display-only
1st–99th percentile normalization. paired-inspection.png rotates only the H&E
preview90degrees CCW and adds coordinate grids for a technical comparison;
it is not a new registered output or claimed anatomical match.

Independent anatomical point selection/accuracy remains open. No model,
GPU, segmentation, transcript counting, native GUI or new scientific result
was attempted. Product source was not edited. These assets are scratch only.
