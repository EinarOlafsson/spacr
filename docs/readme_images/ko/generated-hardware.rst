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

🟢 지원 (안정)   🟣 구현됨 (베타)   🔴 CPU만 지원

.. list-table::
   :header-rows: 1
   :widths: 32 58

   * - 3D 렌더링
     - True 3D, 순서대로 시도
   * - Linux (x86-64, ARM64)
     - Vulkan (wgpu), OpenGL, CPU
   * - Windows (x64, ARM64)
     - Vulkan / Direct3D 12 (wgpu), OpenGL, CPU
   * - macOS (Intel, Apple silicon)
     - Metal (wgpu), OpenGL, CPU
   * - wgpu 없음
     - OpenGL, CPU (Python 3.9, 기타 프로세서, glibc 2.28 미만 Linux 또는 wgpu 없이 설치)
