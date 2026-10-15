### 核心工作流程

主要工作流程由六个模块组成：

- <strong>Mask</strong> 使用 Cellpose 分割细胞、细胞核、病原体和细胞器。
- <strong>Measure</strong> 将形态、强度、纹理、空间和共定位特征以及对象图像裁剪写入 SQLite。
- <strong>Annotate</strong> 在键盘驱动的网格中标注图像裁剪，并支持主动学习队列。
- <strong>Classify</strong> 训练基于图像或测量值的模型，并在每个检查点记录留出数据上的性能。
- <strong>Map Barcodes</strong> 将 FASTQ 读段映射到孔位和 gRNA，并提供丰度、碰撞和覆盖度质控。
- <strong>Regression</strong> 使用适合连续值、比例和计数响应的模型族估计向导 RNA、基因、条件和对照效应。
