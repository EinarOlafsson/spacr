.. list-table::
   :header-rows: 1
   :widths: 32 18 18 22

   * - Hardware
     - Cellpose 4
     - Torch
     - UMAP / clustering
   * - NVIDIA (CUDA)
     - 🟢 GPU
     - 🟢 GPU
     - 🟢 GPU
   * - AMD on Linux (ROCm)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - AMD in an Intel Mac (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Apple Silicon (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Intel Arc/Xe (XPU)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - No GPU
     - 🟢 CPU
     - 🟢 CPU
     - 🟢 CPU

🟢 supported (stable)   🟣 implemented (beta)   🔴 not supported

Every cell is generated from ``spacr.accelerator.capabilities()``
with that backend's probe faked, so this table, the first setup
screen and ``spacr-doctor`` cannot disagree.

**No GPU is supported, not broken.** Every task runs on a CPU and
every result is identical; only the wall clock changes. On the
machine these were measured on, one 256x256 Cellpose image took
444.5 s on the CPU and 3.2 s on its Radeon.

*Beta* means implemented and dispatched to, but exercised on one
machine or none. CUDA is the only configuration with years behind
it.
