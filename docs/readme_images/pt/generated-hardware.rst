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

🟢 suportado (estável)   🟣 implementado (beta)   🔴 apenas suporte a CPU

.. list-table::
   :header-rows: 1
   :widths: 32 58

   * - Renderização 3D
     - True 3D, tentados em ordem
   * - Linux (x86-64, ARM64)
     - Vulkan (wgpu), OpenGL, CPU
   * - Windows (x64, ARM64)
     - Vulkan / Direct3D 12 (wgpu), OpenGL, CPU
   * - macOS (Intel, Apple silicon)
     - Metal (wgpu), OpenGL, CPU
   * - Sem wgpu
     - OpenGL, CPU (Python 3.9, outros processadores, Linux com glibc anterior a 2.28 ou instalado sem wgpu)
