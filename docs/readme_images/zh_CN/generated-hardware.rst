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

🟢 支持（稳定）   🟣 已实现（beta）   🔴 仅支持 CPU

.. list-table::
   :header-rows: 1
   :widths: 32 58

   * - 3D 渲染
     - True 3D，按顺序尝试
   * - Linux (x86-64, ARM64)
     - Vulkan (wgpu), OpenGL, CPU
   * - Windows (x64, ARM64)
     - Vulkan / Direct3D 12 (wgpu), OpenGL, CPU
   * - macOS (Intel, Apple silicon)
     - Metal (wgpu), OpenGL, CPU
   * - 无 wgpu
     - OpenGL, CPU (Python 3.9、其他处理器、glibc 低于 2.28 的 Linux 或未安装 wgpu)
