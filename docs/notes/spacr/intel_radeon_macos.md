# Intel Radeon Mac qualification

Validated on 2026-10-23 with an iMac20,1, Radeon Pro 5300 (4 GiB), 16 GiB RAM,
macOS 26.5.2, Python 3.12.11, Torch 2.2.2, torchvision 0.17.2, NumPy 1.26.4,
Cellpose 4.2.1.1, PySide6 6.11.2, wgpu 0.32.0 and ModernGL 5.12.0.

## Supported execution

The shared accelerator selects Metal for segmentation, embeddings, DINO,
classification, supported attribution methods and motion models. PCA/KMeans
acceleration is opt-in. PCA fitting and sensitive projections retain CPU float64;
UMAP/t-SNE/DBSCAN remain CPU algorithms on Metal. Attribution feature extraction
and counterfactual generation stay on GPU, with unsupported reductions on CPU.

Radeon ambient rendering uses OpenGL 4.1 sampled tables. The known Intel Iris
context-probe exclusion remains. True 3D rendering and object picking were tested
through native OpenGL and Metal/WebGPU. During the Mac startup dissolve, an opaque
cover composites over a captured background to avoid repainting the entire Retina
window every frame. Live painting resumes when the cover closes.

Full Cellpose-SAM AdamW training requires at least 4,874,062,880 bytes for model,
gradients and optimizer state, before activations. A budget guard selects CPU
when that minimum exceeds dedicated Radeon VRAM. Decoder-head GPU training and
the real full-model device transfer were tested; full CPU SAM training was not.

## Environment compatibility

Keep Intel macOS Transformers below 4.58 and torchcam below 0.5 to retain Torch
2.2.2 compatibility. Transformers 4.57.6 and torchcam 0.4.1 were exercised with
real GPU computations. The final environment passed `pip check` and doctor
reported 17 passes, no warnings/errors and three optional checks skipped.

Python 3.9–3.12 passed isolated Torch forward/backward, native Qt/OpenGL and full
dependency-resolution checks. Python 3.12 received the full workflow battery.
Python 3.13–3.14 passed native Qt/OpenGL and WebGPU probes but cannot resolve the
full application because required Intel-macOS Torch wheels are unavailable.
The tested wgpu release requires Python 3.10 or newer.

## Validation and limits

The broad repository run passed 3,330 tests, with 45 skips and one obsolete
CUDA-only test assumption subsequently corrected and rechecked. Focused suites
covered Metal numerical parity, attribution, counterfactuals, embeddings,
training/checkpoint resume, Cellpose, persistent worker exit, real CZI conversion,
btrack optimization, all 54 GUI modules, native rendering and shutdown.
Some suites overlap; their counts must not be summed as unique tests.
The broad battery used nightly `6df3d8184` plus these hardware fixes; integration
onto subsequent nightly updates is checked with targeted workflow and GUI suites.

Raw Torch 2.2 Metal probes exposed unsupported float64/bfloat16/complex storage,
Conv3D, FFT, pinv and nondivisible adaptive pooling. These are upstream limits,
not universally supported GPU operations. Large batches can still exceed VRAM.
Tests using small real architectures establish device execution, not biological
accuracy of every downloadable pretrained model.

Three native close cycles and the inference process exited normally after the
worker transport fix. The historical close-event segfault was not reproduced.
The dissolve's median frame interval improved from approximately 128 ms to
32 ms; occasional long frames remain. A separate cold-start benchmark still
missed its targets (Home 6.42 s versus 5 s; worst stall 966 ms versus 500 ms).
This qualification does not establish perfectly smooth startup or exhaustive
support for other Radeon cards, operating systems or every optional environment.

Hardware-specific regressions live in `tests/test_metal_*.py`,
`tests/test_macos_loader_transport.py` and
`tests/qt/test_radeon_opening_layout.py`. Metal tests skip without a working MPS
device; native renderer qualification also requires a real desktop session.
