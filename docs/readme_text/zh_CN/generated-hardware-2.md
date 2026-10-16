| 3D 渲染 | True 3D，按顺序尝试 |
| --- | --- |
| Linux (x86-64, ARM64) | Vulkan (wgpu), OpenGL, CPU |
| Windows (x64, ARM64) | Vulkan / Direct3D 12 (wgpu), OpenGL, CPU |
| macOS (Intel, Apple silicon) | Metal (wgpu), OpenGL, CPU |
| 无 wgpu | OpenGL, CPU (Python 3.9、其他处理器、glibc 低于 2.28 的 Linux 或未安装 wgpu) |
