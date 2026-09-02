|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
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
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: 最新安装程序
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge 版本

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

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

**CRISPR 筛选的空间表型分析。**

spaCR 对高内涵显微镜图像中的单细胞进行分割和测量，将逐对象表型与测序得到的向导 RNA 丰度整合，并估计哪些基因与表型变化相关。以孔板图像和 FASTQ 读段为输入，它生成逐对象测量值、训练后的分类器、逐向导 RNA 和逐基因效应估计值，以及按优先级排序的命中结果列表。

分区、测量、标记和分类模块也没有序列手臂运行。

图像、掩膜、图像裁剪、测量、笔记、预测、条码和好识别器生活在一个 SQLite 项目中。

作为桌面应用程序或在工作站、服务器或集群上无图形界面运行。

硬件支持
~~~~~~~~~~~~~~~~

.. spacr-hardware-begin

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

🟢 supported (stable)   🟣 implemented (beta)   🔴 CPU support only

.. spacr-hardware-end


安装 spaCR
-------------

桌面应用程序
~~~~~~~~~~~~~~~~~~~

安装器包装自己的 Python. Conda 不需要。

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: 下载适用于 Windows 10/11 的 spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: 下载适用于 macOS 11+（Intel 和 Apple Silicon）的 spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 下载适用于 64 位 Linux 的 spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: 旧版 spaCR 安装程序
   :target: ../../source/installers.rst

.. spacr-installer-links-end

第一三个图标下载当前版本. spaCR 图标打开完整的安装档案. 安装链接和版本的文件名由发布工作流更新; 以前的安装者仍然在同一发布档案中。

在 Linux 上，将下载的文件设为可执行文件并运行：

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

在 macOS 中,打开 ``.pkg``. 目前的 beta 没有通知; 如果 Gatekeeper 阻止它,请选择 **系统设置 → 隐私和安全 → 打开 无论如何**。

请参见 `安装导游 <../../source/installer_guide.rst>`_ 更新、删除、离线和解决问题的指示。

使用 PyPI 安装
~~~~~~~~~~~~~~~~~

如需使用 PyPI 版本，请在 Conda 环境中通过 pip 安装 spaCR。Python 3.12 可选择的科学计算扩展包最为丰富：

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR supports Python **3.9 through 3.14**, except Python 3.14.1, which torchvision excludes. Linux is recommended for the heaviest CUDA and ROCm workflows; macOS and Windows are also supported, and both use their GPUs — macOS through Metal, which covers Apple Silicon and the AMD cards in Intel Macs, and Windows through CUDA or DirectML.

在服务器、集群或 CI 运行器上安装时，请省略 Qt：

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

可选集成单独安装,例如 ``spacr[zarr]``、 ``spacr[omero]``、``spacr[napari]`` 和 ``spacr[czi,nd2,lif]``. 查看完整的附件和 Python 版本兼容性表的 `安装导游 <../../source/installer_guide.rst>`_。

使用 conda-forge 安装
~~~~~~~~~~~~~~~~~~~~~~~~

官方 conda-forge 软件包会将 spaCR 及其桌面应用依赖项安装到当前环境中：

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

安装源
~~~~~~~~~~~~~~~~~~~

克隆存储库并将其安装在可编辑模式下,以便您的工作副本 *is* 安装的包和编辑有效,而无需重新安装::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

默认分支为 ``nightly``. 对于特定发布::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

以后来的变化,从克隆的内部::

    git pull
    pip install -e .

第二行只需要当依赖或输入点改变时; Python 代码在没有它的情况下获取。 如果命令在拖动后仍然运行旧代码,则 ``spacr-doctor`` 报告 ``spacr`` 实际上是您的路径,这是常见原因。

从源头安装(光)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

全克隆:427 MB 核心克隆:76 MB。

::

    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

Skips ``docs/``、``tests/``、 Cellpose 检查点、存档数字和扩展翻译目录。

Options: ``--dir``, ``--branch`` (default ``main``), ``--with-tests``, ``--with-docs``, ``--with-translations``, ``--no-install``.

``packaging/source_install_excludes.txt`` 列出每条路径。


命令行入口
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr                                      # launch the Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

排查问题时，请设置 ``SPACR_LOG_LEVEL=DEBUG``。轮转日志写入 ``~/.spacr/logs/spacr.log``。

``spacr-run --list`` 会列出具有无界面命令行入口的模块。仅在 GUI 中提供的标注、数据整理、比较和探索模块不会列出。


核心工作流程
-------------

主要工作流程由六个模块组成：

- **Mask** 使用 Cellpose 分割细胞、细胞核、病原体和细胞器。
- **Measure** 将形态、强度、纹理、空间和共定位特征以及对象图像裁剪写入 SQLite。
- **Annotate** 在键盘驱动的网格中标注图像裁剪，并支持主动学习队列。
- **Classify** 训练基于图像或测量值的模型，并在每个检查点记录留出数据上的性能。
- **Map Barcodes** 将 FASTQ 读段映射到孔位和 gRNA，并提供丰度、碰撞和覆盖度质控。
- **Regression** 使用适合连续值、比例和计数响应的模型族估计向导 RNA、基因、条件和对照效应。

同一项目还可以设计实验孔板、估算统计功效、校正批次效应、检查分割质量、浏览关联图表和图像裁剪、导出 AnnData、继续中断的工作，并记录生成各项结果时使用的设置。

spaCR 模块
-------------

.. spacr-workflow-begin

|Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

|Module_foreign|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

|Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|\ |Module_analyze_plaques|

|Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: 打开 Mask API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Module_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 16.0%
   :alt: 打开 Measure API
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Module_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 16.0%
   :alt: 打开 Annotate API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Module_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 16.0%
   :alt: 打开 Classify API
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Module_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 16.0%
   :alt: 打开 Map Barcodes API
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Module_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 16.0%
   :alt: 打开 Regression API
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Module_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.0%
   :alt: 打开 Import API
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |Module_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.0%
   :alt: 打开 Run Compare API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |Module_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.0%
   :alt: 打开 Experiment Design API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |Module_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.0%
   :alt: 打开 Power / Design API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |Module_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.0%
   :alt: 打开 Dose–Response API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |Module_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.0%
   :alt: 打开 QC API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |Module_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.0%
   :alt: 打开 Make Masks API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |Module_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.0%
   :alt: 打开 Align & Stitch API
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |Module_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.0%
   :alt: 打开 Image UMAP API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Module_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.0%
   :alt: 打开 Gate Editor API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |Module_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.0%
   :alt: 打开 Graph Builder API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |Module_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.0%
   :alt: 打开 Plaque Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: 打开 Recruitment API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: 打开 Invasion Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: 打开 Replication Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle

.. spacr-workflow-end

Every module spaCR ships, in the order the home screen lists them: the six pipeline modules first, then everything else. Select a tile to open that module's API page.


Make Masks
~~~~~~~~~~

Make Masks appears under **Tools** for manual correction of segmentation masks; its masthead opens the Cellpose workflows. Nine tools: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** and **Recrop**. Draw makes one filled label from a closed outline, Divide separates a merged object along a drawn line, Recrop turns one object in a crowded field into its own field.

Cellpose-SAM runs here show the cell-probability map and the flow field beside the mask. See the `feature guide <../../source/features.rst>`_ for each tool.

**其他资源**

- `互动教程 <https://einarolafsson.github.io/spacr/tutorials/>`_ — 从安装到成功调查的73个导向工作流。
- `Python API 快速启动 <../../source/python_api.rst>`_ - 从脚本、笔记本或集群运行和验证流程。
- `功能指南 <../../source/features.rst>`_ - 能力、成熟度和可选集成。
- `清理 API 参考 <https://einarolafsson.github.io/spacr/api/index.html>`_ - 按任务支持输入点,完整的模块参考一个级别更深。
- `语言与翻译指南 <../../source/localization.rst>`_ — 界面语言、上下文帮助和科学输出政策。

语言与翻译
~~~~~~~~~~~~~~~~~~~~~~

界面的导航和首选项支持十种语言。AI 和 LIVE 控件、模块说明以及经过审核的上下文帮助也会翻译。无需重启，即可在 **spaCR → 首选项 → 语言** 中更改语言。日志、路径、数据库值和测量结果不会被翻译；科学输出始终使用规范英语。请参阅 `上下文帮助政策 <../../source/localization.rst#contextual-help>`_。

动画设置指南
~~~~~~~~~~~~~~~~~~~~~~~~~

带有视觉说明的设置会在工具提示中提供 **Animation** 控件。浏览 `设置动画图库 <https://einarolafsson.github.io/spacr/setting_animations.html>`_ 或 `设置动画注册表 <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_。

数据
----

参考数据集
~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: 打开 BioStudies 显微镜数据集
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: 打开 Hugging Face 测试数据集
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: 打开 NCBI 测序数据集
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: 打开 spaCRPower
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: 打开 bioRxiv 预印本
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

动物园模型
~~~~~~~~~~

spaCR 附带一个训练好的模型目录，并在需要时下载。在主界面打开 **Model Zoo** 浏览并安装模型，或在设置文件中指定键名 -- ``pathogen_model: toxoplasma_pv_v1`` -- 模型会在首次需要时下载并校验其校验和。每个已发布条目都带有 SHA-256；没有校验和的条目会被拒绝而不是安装，因为被截断或被替换的检查点无法与真实文件区分。

.. spacr-model-zoo-begin

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Key
     - Trained on
     - Measured performance and limits
   * - ``toxoplasma_pv_v1``
       (cpsam_v2_toxo_r2)
     - Toxoplasma tachyzoite parasitophorous vacuoles stained with goat anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV lumen. 115 pairs (104 train / 11 test), 100 epochs, base cpsam_v2
     - F1 0.867 at IoU 0.5 against 0.713 for stock cpsam; AJI 0.808 against 0.426; accuracy falls sharply above IoU 0.8 -- suited to counting and area rather than precise morphometry
   * - ``toxoplasma_plaque_v1``
       (cpsam_plaque_r3)
     - Toxoplasma gondii plaque assays; round 3, evaluated in-domain (NAS) and against a literature generalisation set
     - F1 0.856 in-domain and 0.834 on the literature set, against 0.718 / 0.755 for round 1; round 3 trades precision (0.939 down to 0.858) for recall (0.631 up to 0.811) on the literature set, which is the right direction for a counting assay
   * - ``toxoplasma_well_detector_v1``
       (yolo_welldetect_v3.pt)
     - Whole-plate and multi-well Toxoplasma plaque-assay images; yolo11n base, 150 epochs, batch 16, imgsz 640
     - mAP50 0.993, mAP50-95 0.886, precision and recall both 0.987; locates WELLS, not plaques; it is the front half of a two-stage pipeline with toxoplasma_plaque_v1, and the well it finds also gives the diameter that makes areas comparable across microscopes

.. spacr-model-zoo-end

上面的数字是发布时测得的，并且与之一同给出了适用范围：模型只对其测量过的任务有效，而不是对所有任务都有效。``toxoplasma_well_detector_v1`` 和 ``toxoplasma_plaque_v1`` 是同一条流程的两个环节——检测器找到孔位，分割模型在孔内找到蚀斑，而孔径使不同显微镜之间的面积可以相互比较。

模型托管在各自作者本人的 Hugging Face 账户下，因此贡献一个模型并不意味着要交出他人账户的写入权限。``spacr.model_zoo`` 的 ``publish_model`` 会完成上传，并打印出需要添加的目录条目。


性能诊断
----------------------

生成硬件报告并将其附到性能相关问题中::

    python tools/spacr_hardware_report.py

节省到 ``~/.spacr/reports`` 并打印路径. ``--quick`` 将更长的基准标志; ``--out PATH`` 设置位置。

阅读没有项目数据. 时间进口,数字图书馆,窗户建设和动画. 报告处理器架构模拟(一个 x86_64 Python 构建在苹果硅)和 NumPy 的 BLAS 实施。

命令线参考
----------------------

下面的每个命令都以 ``pip install spacr`` 安装,所有命令都会接受 ``--help``。

启动申请
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-server`` 扫描模型设置筛选,否则会阻止未预期的工作。

``spacr-qt`` 和 ``spacr-nightly`` 是 ``spacr`` 的联盟。

当 spaCR 不开始时
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` 打印一个行每检查,每个故障运行一个命令. 它还报告哪个 ``spacr`` 在路径上,这是一个可编辑的旧安装的阴影。

``safespacr`` 读取每个偏好作为其默认的,并强迫背景,动画,字面登录和预载。

无图形界面发运行模块
~~~~~~~~~~~~~~~~~~~~~~~~~~

没有 Qt,没有显示器 - 用于集群、服务器和CI。

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` 读取相同的设置,并报告什么是缺乏,矛盾或指向什么。

``spacr-run --list`` 只显示无图形界面输入点的模块;标记、治疗和探索是互动的,被忽略了。

接下来的跑步检查
~~~~~~~~~~~~~~~~~~~~~~~~~~~

每个运行记录为 ``~/.spacr/runs`` 与其设置,加密输入,输出,警告,版本和种子。

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

数据审计与安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

环境
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

旋转日志写为 ``~/.spacr/logs/spacr.log``. 将此文件添加到错误报告中。


贡献与支持
------------------------

请通过 `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_ 提交错误报告和范围明确的功能请求。报告故障时，请提供 spaCR 版本、操作系统、Python 版本、模块设置和相关日志片段。``spacr-doctor`` 会收集其中的大部分信息；报告性能问题时还应附上硬件报告。

许可
~~~~~~~~~

spaCR is released under the `BSD 3 条款许可证 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

如果 spaCR 有助于发表作品,则引用被评估,并且不符合许可的条件,请参见下面的 `Citing spaCR`_。

教程
~~~~~~~~~

`spaCR 交互式教程库 <https://einarolafsson.github.io/spacr/tutorials/>`_ 提供安装和各应用工作流程的配音、字幕教程，共有 73 节课程、50 种语音，涵盖八种语言。

引用 spaCR
~~~~~~~~~~~~

如果 spaCR 对您的研究有所帮助，请引用：

Olafsson EB, *et al.* 一张以图像为基础的 CRISPR 筛选将 EAF1 定义为 *T. gondii* ESCRT 模块化器。

`生物Rxiv 预印 <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `软件档案 <https://doi.org/10.5281/zenodo.21343316>`_

致谢
~~~~~~~~~~~~~~~

spaCR 构建于开放科学软件之上，包括 NumPy、pandas、scikit-image、scikit-learn、Cellpose、PyTorch 和 Qt。有关多语言文档和界面目录所使用的模型，请参阅`翻译模型署名 <../TRANSLATION_MODELS.md>`_。
