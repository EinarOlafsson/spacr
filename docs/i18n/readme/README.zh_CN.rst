|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: 文档
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: 交互式教程
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI 版本
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 至 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: 测试套件
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt 界面
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub 源代码
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub 问题
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm 非商业许可证
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: 最新安装程序
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge 配方

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/logo_spacr.png
   :alt: spaCR
   :align: center
   :width: 360

spaCR
=====

语言: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

`翻译模型说明 <../TRANSLATION_MODELS.md>`_

**CRISPR 筛选的空间表型分析。**

spaCR 对高内涵显微镜图像中的单细胞进行分割和测量，将每个细胞与其获得的 gRNA 关联，并报告哪些基因改变了表型。输入为孔板图像和 FASTQ 读段；输出包括逐对象测量、训练后的分类器、逐向导 RNA 和逐基因效应量，以及按优先级排序的候选结果列表。

对于基于图像的混合 CRISPR 筛选，这涵盖了完整工作流程。如果只有高内涵显微镜数据而没有筛选实验，也可以单独运行分割、测量、标注和分类部分。

图像、掩膜、图像裁剪、测量值、标注、预测、条形码和孔位标识符都存储在同一个 SQLite 项目中，因此结果中的数值可以追溯到其来源对象。

spaCR 可作为桌面应用程序运行，也可在工作站、服务器或集群上以无图形界面模式运行。两种方式使用相同的模块；模块支持 CUDA 时会自动启用。


工作流程概览
--------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR 工作流程及输出结构
   :align: center

显微镜图像（TIFF、OME-TIFF、LIF、CZI、ND2）和测序读段（FASTQ）分别进入互补的图像分析与条形码映射流程。随后对对象表、图像裁剪、标注、预测、向导 RNA 身份、QC 结果和孔位级汇总进行联合分析。


快速开始
-----------

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR 支持 Python **3.9 至 3.14** （torchvision 不支持的 Python 3.14.1 除外）。Python 3.12 可选的科学计算软件包最齐全。涉及 CUDA 的工作流程建议使用 Linux；同时也支持 macOS 和 Windows。


安装详情
--------------------

|Release| |PyPI| |CondaRecipe|

**（测试版）轻量级桌面安装程序：**

.. spacr-installer-links-begin

* `Windows 10/11：下载 SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe>`_
* `macOS 11+（英特尔和苹果硅）：下载 SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg>`_
* `64 位 Linux：下载 SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

轻量级安装程序 — 无需 conda 或现有 Python 环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

安装过程中，安装程序会下载独立的 Python 3.12 运行时、Qt、PyTorch、spaCR 及科学计算依赖项，因此无需预先安装 conda 或 Python。默认使用便携式 CPU 版本，以免在未提示的情况下下载数 GB 的 CUDA 库。Windows 可将 NVIDIA 加速选为安装组件，Linux 接受 ``--torch-backend auto``，macOS 的标准 PyTorch wheel 则保留 Apple MPS 加速。

安装程序的帮助、进度和错误信息会根据操作系统的语言，使用 spaCR 支持的十种语言之一：英语、瑞典语、德语、西班牙语、简体中文、葡萄牙语、印地语、韩语、冰岛语和法语。不支持的语言环境会回退到英语。

在 Linux 上，打开已下载的安装程序前，先赋予其可执行权限：

.. code-block:: bash

   chmod +x spaCR-*-Linux-x86_64-Online.run
   ./spaCR-*-Linux-x86_64-Online.run

在 macOS 上，打开已下载的 ``.pkg`` 文件。如果当前测试版安装程序因未公证而被 Gatekeeper 阻止，请打开 **系统设置 → 隐私与安全性**，为 spaCR 选择 **仍要打开**，然后再次运行该安装包。

替换旧安装前，安装程序会验证 spaCR、Qt、PyTorch 以及依赖项是否一致，因此更新即使中断，原有可用环境也会保留。诊断日志以 ``install.log`` 保存于 spaCR 的独立安装目录中。

通过 PyPI 安装桌面应用程序
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

无图形界面或服务器安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

最新开发分支
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch nightly
   python -m pip install -e ".[qt]"

Conda 环境
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

可选功能
~~~~~~~~~~~~~~~~~~~~~

仅安装工作流程所需的可选依赖：

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

可安装的可选依赖取决于 Python 版本。在 Python 3.13 上，ultrack 的依赖限制会影响 ``spacr[all]``，TorchCAM 的 NumPy 限制会影响 ``attribution``；核心包和 Qt 应用不受影响。在 Python 3.14 上，btrack 可通过其可选依赖安装。pylibCZIrw CZI 转换器是可选且尚未测试的；基于 czifile 的 CZI 读取功能仍然可用。

旧版 Tk 界面仍会以 ``spacr-legacy`` 安装，但已不再继续开发。


命令行入口
-------------------------

.. code-block:: bash

   spacr                                      # Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

排查问题时，请设置 ``SPACR_LOG_LEVEL=DEBUG``。轮转日志写入 ``~/.spacr/logs/spacr.log``。


功能
--------

大多数筛选实验使用的六个模块
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** 使用 Cellpose 在二维图像、体积数据或时间序列中分割细胞、细胞核、病原体和细胞器。模型列表直接读取已安装的 Cellpose，而不是写死在代码中；运行开始前还会根据图像估算对象直径。掩膜可在图层查看器中手动修正，也可发送到 napari 中编辑后再导回。

**Measure** 将每个对象的形态、强度、纹理和共定位特征连同图像裁剪写入项目数据库。1.5.0.0 新增的照明校正会从孔板本身估算平场，并在提取任何强度特征前完成校正，从而消除孔板热图中表现为边缘效应的孔位偏差。分割 QC 横幅会在 Measure 运行前用简明文字说明掩膜质量；它只提供信息，不会阻止运行。绘制的多边形可将测量限制在感兴趣区域内。

**Annotate** 在键盘操作的网格中显示图像裁剪，并将标签直接写入 SQLite。它还能完成主动学习闭环：不离开当前界面即可用已有标注重新训练模型，按不确定性重新排列队列，查看学习曲线，并在继续标注已无法改善模型时给出停止建议。系统会按类别、孔位和孔板报告覆盖率，并记录每一轮过程。

**Classify** 在已标注的图像裁剪上训练 PyTorch CNN 和 Transformer，也可在测量表上训练经典模型或提升模型。现在每个 epoch 都会保留各类别的准确率，每个检查点还会生成模型卡，记录数据集、类别平衡、拆分规则和留出集指标。在评估界面中，混淆矩阵的单元格可直接查询：单击即可打开对应裁剪，并将高置信度错误与不确定样本分开列出。

**Map Barcodes** 从 FASTQ 读段中解码行、列和 gRNA 条形码，为孔位分配向导 RNA 身份，并将其与成像细胞关联。Barcode QC 会根据用户给出的每孔预期 gRNA 数进行范围评估，报告每孔读段数、冲突率和未映射比例，而不是采用固定阈值。

**Regression** 使用 17 类模型估计向导 RNA、基因、条件和对照效应，其中包括混合模型、Logistic、Probit、分位数、Beta、具有准二项方差的 GLM、Lasso、Ridge、Elastic Net、Hinge 和 Horseshoe。输出是经过排序并附有注释的候选结果列表，而不是未经整理的系数集合。

1.5.0.0 新增功能
~~~~~~~~~~~~~~~~~~

在筛选实验开始前，Power / Design 模块会计算所需的细胞数和孔数，并将测序错误以及成像细胞过少的孔造成的丢失纳入估算。实验设计工具排布孔板、对照和重复，并将版式导出到工作流程。实验完成后，QC 仪表板将分割、孔板、标注者一致性和数据泄漏检查汇总为一项结论；除 ``center`` 和 ``zscore`` 外，还可使用 ComBat 进行批次校正。

结果可以直接探索，无需导出后再重新导入。在 Graph Builder 中，将表格列拖到 x、y、颜色、大小和分面即可作图。在直方图或散点图上绘制的门会转换为过滤器。Feature Explorer 会按特征区分各类的能力进行排序。小多图、剂量–反应拟合、控制图和稳健异常值检测使用同一套坐标轴引擎。在一个视图中选择对象后，其他视图中也会同步选中；打开该选择可查看这些对象对应的图像裁剪。Layer Viewer 可叠加图像、标签、点和形状，并提供正交视图、同步比较网格以及从细胞到细胞核再到病原体的谱系树。

现在每次运行都可被明确追踪。每次运行都有运行 ID、随机种子和 ``on_error`` 策略；Mask、Measure、Classify 和 AnnData 导出会将各自产生的内容登记到工件注册表，因此可以从输出文件追溯到生成它的设置。模块会打开上一步实际写出的内容，流程图会标记已过期的输出，运行比较会列出两次运行在设置、对象数量和候选结果列表上的差异，而每次 GUI 运行都会生成等效的 Python 脚本。测量结果可导出为供 scanpy 使用的 ``.h5ad``；OME-Zarr 和 OMERO 可通过 Python API 使用。方法与结果导出器根据运行的结构化摘要起草论文的这两个部分：模型负责行文，但每个数字都必须来自摘要；包含摘要中不存在数字的草稿会被拒绝。安装出现问题时，``spacr-doctor`` 会报告实际运行的 spaCR、GPU 是否可用、Cellpose 是否匹配 spaCR 调用的 API，以及项目数据库和设置是否有效，并为每项失败的检查提供可复制的修复命令。

多语言桌面界面
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通过 **spaCR → 偏好设置 → 语言** 可在不重启的情况下，将正在运行的应用程序切换为英语、瑞典语、德语、西班牙语、简体中文、葡萄牙语、印地语、韩语、冰岛语或法语。该选择会被保存，之后打开的界面也会采用同一语言。

导航、偏好设置、AI 和 LIVE 控件、模块说明以及 spaCR 自身生成的控制台提示都会采用所选语言。工作进程输出、日志、回溯信息、路径、数据库值、标注、AI 回复、测量值和保存的结果不会被翻译，因此科学结果始终保留规范的英文形式。尚未经过人工审校的设置工具提示将保留英文，避免出现语言混杂的说明。`本地化指南 <https://einarolafsson.github.io/spacr/localization.html>`_ 介绍了此行为、环境变量覆盖方式以及随界面一同翻译的 `上下文帮助 <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_。

动画设置指南
~~~~~~~~~~~~~~~~~~~~~~~~~

94 个短动画展示了 143 项可视化设置会怎样影响图像。将鼠标悬停在某项设置上，然后单击工具提示中的 **动画**，即可播放文字旁的方形预览；再次单击可将其收起。动画只在用户请求时播放，也可在“偏好设置”中彻底关闭。`动画库 <https://einarolafsson.github.io/spacr/setting_animations.html>`_ 展示全部动画，`设置动画注册表 <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ 则记录每个动画对应的设置。

模块参考
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - 模块
     - 功能
     - 状态
     - 说明
   * - **桌面体验**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - 稳定
     - 可在十种内置语言之间即时重译已打开及按需创建的界面。
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - 稳定
     - 本地化模块摘要和设置帮助界面，同时保持 API URL 完全不变。
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - 稳定
     - 本地化 AI 和 LIVE 控件，但不改动用户内容或模型内容。
   * - |api-animations|_
     - |doc-animations|_
     - 稳定
     - 可从设置工具提示播放 94 个内置动画，说明 143 项可视化设置。
   * - |api-selection|_
     - |api-linked-views|_
     - Alpha
     - 在表格、孔板、嵌入、散点图和图形视图之间共享同一对象选择。
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alpha
     - 检查 GPU、Cellpose API、数据库和设置，并为每项失败的检查提供修复方法。
   * - **图像分析**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - 稳定
     - 在二维图像中分割细胞、细胞核、病原体和细胞器。
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - 分割三维体积图像和四维时间序列。
   * - |api-illumination|_
     - |api-flatfield|_
     - Alpha
     - 从整块孔板估算平场，并在测量强度前完成校正。
   * - |api-measure|_
     - |api-measure-2d|_
     - 稳定
     - 测量形态、强度、纹理和共定位特征，并保存图像裁剪。
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alpha
     - 在 Measure 运行前说明分割质量，但不会阻止运行。
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - 使用 IoU、Trackpy、btrack、Trackastra 或 ultrack 跟踪对象并量化运动性。
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alpha
     - 叠加图像、标签、点和形状图层，并提供正交视图和比较网格。
   * - |api-napari|_
     - |api-napari-curation|_
     - Alpha
     - 将掩膜交给 napari 修正后取回，并记录每一次编辑。
   * - **AI 与表型分析**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - 稳定
     - 在键盘操作的网格中审阅图像裁剪，并将标注保存到 SQLite。
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alpha
     - 在 Annotate 内重新训练模型，按不确定性重新排序，并提示何时可以停止标注。
   * - |api-classify|_
     - |api-classification|_
     - 稳定
     - 训练并应用 PyTorch CNN 和 Transformer 模型。
   * - |api-classify|_
     - |api-model-cards|_
     - Alpha
     - 为每个检查点记录数据集、类别平衡、拆分规则和留出集指标。
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alpha
     - 打开混淆矩阵单元格对应的图像裁剪，并将高置信度错误与不确定样本分开列出。
   * - |api-ml|_
     - |api-ml-models|_
     - 稳定
     - 在测量表上训练可解释的经典模型和提升模型。
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - 使用 Captum、SmoothGrad 和 TorchCAM 解释预测结果。
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - 以交互方式探索图像嵌入，并传播聚类标签。
   * - **测序与筛选分析**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - 稳定
     - 从 FASTQ 读段映射行、列和 gRNA 条形码，并为成像细胞分配向导 RNA。
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alpha
     - 根据每孔预期的 gRNA 数，报告每孔读段数、冲突率和未映射比例。
   * - |api-regression|_
     - |api-regression-models|_
     - 稳定
     - 使用 17 类模型估计向导 RNA、基因、条件和对照效应。
   * - |api-power|_
     - |api-power-design|_
     - Alpha
     - 在计入测序误差和孔位脱落后，估算筛选所需的细胞数和孔数。
   * - |api-graph|_
     - |api-graph-builder|_
     - Alpha
     - 通过将列拖到 x、y、颜色、大小和分面字段来生成图表。
   * - |api-artifacts|_
     - |api-provenance|_
     - Alpha
     - 记录 Mask、Measure、Classify 和导出结果对应的运行 ID、随机种子及设置。

.. |api-qt-app| replace:: **Qt 应用程序**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **十种语言本地化**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **本地化上下文帮助**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI 辅助控制台**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **设置动画注册表**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **可视化设置动画**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **选择**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **联动选择**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **安装诊断**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **二维掩膜生成**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **三维和四维掩膜生成**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **照明校正**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **平场校正**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **对象测量**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **分割质量控制**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **运行前评估**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **对象跟踪**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **图层**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **图层查看器**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari 桥接**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **掩膜校正**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **手动标注**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **主动学习**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **重新训练和排序**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **图像分类**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **模型卡**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **激活图**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **混淆矩阵下钻**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **机器学习**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **测量值分类**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **交互式嵌入**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **测序**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **条形码映射**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **条形码质量控制**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **孔位和冲突报告**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **筛选效应估计**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **统计功效与实验设计**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **工件**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **运行溯源**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


数据
----

参考数据集
~~~~~~~~~~~~~~~~~~

- `完整显微镜数据集：BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `测试数据集：Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `测序数据：NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `统计功效分析：spaCRPower <https://github.com/maomlab/spaCRPower>`_


贡献与支持
------------------------

欢迎通过 `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_ 提交错误报告和明确的功能请求。报告故障时，请附上 spaCR 版本、操作系统、Python 版本、模块设置及相关日志片段；``spacr-doctor`` 会自动收集其中的大部分信息。

许可
~~~~~~~~~

当前开发分支的源代码按 `PolyForm 非商业许可证 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_ 提供。商业使用需要另行获得版权持有者的许可。spaCR 1.4.9.9 及更早发布版仍按各自发布时附带的 MIT 许可证提供。

教程
~~~~~~~~~

`交互式 spaCR 教程库 <https://einarolafsson.github.io/spacr/tutorials/>`_ 以八种语言提供安装过程和各应用工作流程的配音、字幕操作指南。

引用 spaCR
~~~~~~~~~~~~

如果 spaCR 有助于您的研究，请引用：

Olafsson EB, *et al.* 一项汇集式图像 CRISPR 筛选将 EAF1 鉴定为 *T. gondii* 中 ESCRT 功能劫持的调控因子。

`bioRxiv 预印本 <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `软件归档 <https://doi.org/10.5281/zenodo.21343317>`_
