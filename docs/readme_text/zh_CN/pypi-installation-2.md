spaCR 支持 Python <strong>3.9 through 3.14</strong>，但 Python 3.14.1 除外，torchvision 不包含该版本。最繁重的 CUDA 和 ROCm 工作流程建议使用 Linux；macOS 和 Windows 也受支持，两者都会使用各自的 GPU — macOS 通过 Metal，它涵盖 Apple Silicon 和 Intel Mac 中的 AMD 显卡，Windows 则通过 CUDA 或 DirectML。

标准安装包含 Qt 桌面界面。在服务器、集群或 CI 运行器上，可运行命令行流程而不打开该界面：
