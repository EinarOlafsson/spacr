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

🟢 með stuðningi (stöðugt)   🟣 útfært (beta)   🔴 aðeins CPU-stuðningur

.. list-table::
   :header-rows: 1
   :widths: 32 58

   * - Þrívíddarteiknun
     - True 3D, reynt í röð
   * - Linux (x86-64, ARM64)
     - Vulkan (wgpu), OpenGL, CPU
   * - Windows (x64, ARM64)
     - Vulkan / Direct3D 12 (wgpu), OpenGL, CPU
   * - macOS (Intel, Apple silicon)
     - Metal (wgpu), OpenGL, CPU
   * - Án wgpu
     - OpenGL, CPU (Python 3.9, aðrir örgjörvar, Linux með glibc eldra en 2.28 eða sett upp án wgpu)
