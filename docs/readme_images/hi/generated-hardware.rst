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

🟢 समर्थित (स्थिर)   🟣 लागू (बीटा)   🔴 केवल CPU समर्थन

.. list-table::
   :header-rows: 1
   :widths: 32 58

   * - 3D रेंडरिंग
     - True 3D, क्रम से आज़माए जाते हैं
   * - Linux (x86-64, ARM64)
     - Vulkan (wgpu), OpenGL, CPU
   * - Windows (x64, ARM64)
     - Vulkan / Direct3D 12 (wgpu), OpenGL, CPU
   * - macOS (Intel, Apple silicon)
     - Metal (wgpu), OpenGL, CPU
   * - wgpu के बिना
     - OpenGL, CPU (Python 3.9, अन्य प्रोसेसर, 2.28 से पुराने glibc वाला Linux या wgpu के बिना इंस्टॉल)
