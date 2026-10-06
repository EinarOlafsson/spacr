|Platforms| |Python| |Qt| |Tests| |Release| |Issues| |Source| |Conda| |PyPI| |Conda Downloads| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API%20Documentation
   :target: https://einarolafsson.github.io/spacr/
   :alt: API 文档
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
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Qt 界面
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub 源代码
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub 问题
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: BSD 3-Clause 许可证
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-2026.07.08.737057-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv 预印本
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: 最新安装程序
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge 版本
.. |Conda Downloads| image:: https://anaconda.org/conda-forge/spacr/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge 下载量
.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge 最新发布日期
.. |PyPI Downloads| image:: https://static.pepy.tech/personalized-badge/spacr?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads
   :target: https://pepy.tech/projects/spacr
   :alt: PyPI 下载量
.. |Platforms| image:: https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux、macOS 和 Windows
.. |Cite| image:: https://img.shields.io/badge/Cite-CITATION.cff-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: 引用 spaCR
.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=PyPI+rank+%2830d%29&color=brightgreen&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: spaCR 在过去 30 个完整日的 PyPI 下载量排名

.. image:: ../../source/_static/deck/slides/slide_01.jpg
   :alt: spaCR
   :width: 920
   :target: https://einarolafsson.github.io/spacr/_static/deck/

`← 上一页 <../../source/_static/deck/pages/57.md>`_   `下一页 → <../../source/_static/deck/pages/02.md>`_

spaCR
=====

.. spacr-language-picker-begin

语言: `🌐 简体中文 ▾ <README.md>`_

.. spacr-language-picker-end

**CRISPR 筛选的空间表型分析。**

spaCR 对高内涵显微镜图像中的单细胞进行分割和测量，将逐对象表型与测序得到的向导 RNA 丰度整合，并估计哪些基因与表型变化相关。以孔板图像和 FASTQ 读段为输入，它生成逐对象测量值、训练后的分类器、逐向导 RNA 和逐基因效应估计值，以及按优先级排序的命中结果列表。

分区、测量、标记和分类模块也没有序列手臂运行。

Make Masks 修正分割掩膜，并使用 **Box** 工具标注带类别标签的独立矩形以进行 YOLO 导出。框保留自己的标签和历史记录，不会更改原始图像或掩膜。

各工具的说明见 `功能指南 <../../source/features.rst>`_。

图像、掩膜、裁剪图像块、测量值、标注、预测、条形码和微孔标识符都存放在同一个 SQLite 项目中。

可作为桌面应用程序运行，也可在工作站、服务器或集群上以无图形界面方式运行。

试用 spaCR
~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

在 Import、Make Masks、Annotate 或实验分析界面中使用 **加载测试数据…** 下载示例数据。在终端中使用 ``spacr-download``。

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

支持(稳定) 实施(beta) 🔴 CPU 仅支持

.. spacr-hardware-end


安装 spaCR
~~~~~~~~~~~~~

桌面应用程序
-------------------

安装器包装自己的 Python. Conda 不需要。

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: 下载适用于 Windows 10/11 的 spaCR 1.5.1.3
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.3/spaCR-1.5.1.3-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: 下载适用于 macOS 11+（Intel 和 Apple Silicon）的 spaCR 1.5.1.3
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.3/spaCR-1.5.1.3-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 下载适用于 64 位 Linux 的 spaCR 1.5.1.3
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.3/spaCR-1.5.1.3-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: 旧版 spaCR 安装程序
   :target: ../../source/installers.rst

.. spacr-installer-links-end

在 Linux 上，将下载的文件设为可执行文件并运行：

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

在 macOS 中,打开 ``.pkg``. 目前的 beta 没有通知; 如果 Gatekeeper 阻止它,请选择 **系统设置 → 隐私和安全 → 打开 无论如何**。

有关更新、卸载、离线使用和故障排查的说明，请参阅 `安装指南 <../../source/installer_guide.rst>`_；有关工作站和服务器的配置建议以及 GPU 兼容性表，请参阅 `系统要求 <../../source/system_requirements.rst>`_。

使用 PyPI 安装
-----------------

如需使用 PyPI 版本，请在 Conda 环境中通过 pip 安装 spaCR。Python 3.12 可选择的科学计算扩展包最为丰富：

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

spaCR 支持 Python **3.9 through 3.14**，但 Python 3.14.1 除外，torchvision 不包含该版本。最繁重的 CUDA 和 ROCm 工作流程建议使用 Linux；macOS 和 Windows 也受支持，两者都会使用各自的 GPU — macOS 通过 Metal，它涵盖 Apple Silicon 和 Intel Mac 中的 AMD 显卡，Windows 则通过 CUDA 或 DirectML。

标准安装包含 Qt 桌面界面。在服务器、集群或 CI 运行器上，可运行命令行流程而不打开该界面：

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

可选集成单独安装,例如 ``spacr[zarr]``、 ``spacr[omero]``、``spacr[napari]`` 和 ``spacr[czi,nd2,lif]``. 查看完整的附件和 Python 版本兼容性表的 `安装导游 <../../source/installer_guide.rst>`_。

使用 conda-forge 安装
------------------------

官方 conda-forge 软件包会将 spaCR 及其桌面应用依赖项安装到当前环境中：

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

使用 Docker 安装
-------------------

使用已发布的 `GHCR 上的 Docker 镜像 <https://github.com/EinarOlafsson/spacr/pkgs/container/spacr>`_，在容器中运行 spaCR 的命令行处理流程。安装 `Docker <https://docs.docker.com/get-started/get-docker/>`_，然后使用此已发布的 CPU 镜像列出可用的处理流程：

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.0 spacr-run --list

对应的 NVIDIA GPU 镜像为 ``ghcr.io/einarolafsson/spacr:1.5.1.0-cuda12.4``。这两种镜像均面向 Linux x86-64 容器。有关 GPU 前提条件、数据和模型挂载、设置文件及完整处理流程命令，请参阅 `Docker 安装指南 <../../source/installer_guide.rst#container-images>`_。

从源代码安装
-------------------

克隆代码仓库并以可编辑模式安装。这样工作副本 *就是* 已安装的软件包，修改无需重新安装即可生效::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

这会克隆默认分支 ``main``，即最新的发布版本。开发在 ``nightly`` 分支上进行；如需改为克隆该分支，请添加 ``--branch nightly``。如需克隆特定版本::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

之后如需拉取更新，请在克隆目录中运行::

    git pull
    pip install -e .

只有在依赖项或入口点发生变化时才需要第二行；Python 代码无需它即可生效。如果拉取后某个命令仍在运行旧代码，``spacr-doctor`` 会报告路径中实际使用的是哪个 ``spacr``，这通常就是原因所在。

从源代码安装（精简版）
---------------------------

贡献者需要完整的提交历史；如果只想运行 spaCR，请选用以下任一方式。数据于 2026-09-15 由 ``packaging/measure_clone_forms.sh`` 测得::

    # One commit instead of every version: 540 MB downloaded, 69 s.
    # No history, so no git log, no git blame and no git bisect.
    # git pull still works, but stays shallow until git fetch --unshallow.
    git clone --depth 1 https://github.com/EinarOlafsson/spacr.git
    cd spacr && pip install -e .

    # Only the files spaCR runs from: 81 MB on disk, 39 s. No history
    # either, and no docs, tests, tools, features or example data.
    # --with-docs, --with-tests and --with-translations put those back;
    # --dir, --branch, --no-install and --help do the obvious things.
    # packaging/source_install_excludes.txt lists every skipped path.
    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

在 2026-09-15 的测量中，完整克隆下载了 5.8 GB。向浅克隆添加 ``--filter=blob:none`` 没有减少其测得的下载量。nightly 中受 Git 跟踪的文件检出后共占 1710 MB（2026-10-06 测量），不含 Git 历史。下载大小和耗时随分支而变化。


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
   spacr-download --list                      # what example data exists
   spacr-download measure annotate            # fetch example sets by name
   spacr-make-masks --folder DIR              # curate masks as a resumable queue
   spacr-make-masks --folder DIR --order easy --limit 50

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

spaCR 模块
-------------

.. spacr-workflow-begin

核心
^^^^

Core sequence from microscopy images through segmentation, measurements,
annotations, classification, barcode mapping and regression.

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

数据
^^^^

Import images and tables into spaCR projects and execute reproducible
multi-plate workflows.

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

工具
^^^^

Point these at a project: edit masks by hand, stitch tiles, read an
embedding, draw a gate, build a plot, check quality.

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

实验分析
^^^^^^^^

Quantitative readouts for biological assays.

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../../spacr/resources/icons/workflow/mask.png" width="16.0%" align="middle" alt="打开 Mask API"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../../spacr/resources/icons/workflow/measure.png" width="16.0%" align="middle" alt="打开 Measure API"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../../spacr/resources/icons/workflow/annotate.png" width="16.0%" align="middle" alt="打开 Annotate API"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../../spacr/resources/icons/workflow/classify_merged.png" width="16.0%" align="middle" alt="打开 Classify API"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../../spacr/resources/icons/workflow/map_barcodes.png" width="16.0%" align="middle" alt="打开 Map Barcodes API"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../../spacr/resources/icons/workflow/regression.png" width="16.0%" align="middle" alt="打开 Regression API"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../../spacr/resources/icons/workflow/apps/foreign.png" width="16.0%" align="middle" alt="打开 Import API"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../../spacr/resources/icons/workflow/apps/embeddings.png" width="16.0%" align="middle" alt="打开 Embeddings API"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../../spacr/resources/icons/workflow/apps/run_compare.png" width="16.0%" align="middle" alt="打开 Run Compare API"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../../spacr/resources/icons/workflow/apps/experiment_design.png" width="16.0%" align="middle" alt="打开 Experiment Design API"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../../spacr/resources/icons/workflow/apps/power.png" width="16.0%" align="middle" alt="打开 Power / Design API"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../../spacr/resources/icons/workflow/apps/dose_response.png" width="16.0%" align="middle" alt="打开 Dose–Response API"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../../spacr/resources/icons/workflow/apps/qc_dashboard.png" width="16.0%" align="middle" alt="打开 QC API"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../../spacr/resources/icons/workflow/apps/make_masks.png" width="16.0%" align="middle" alt="打开 Make Masks API"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../../spacr/resources/icons/workflow/apps/align.png" width="16.0%" align="middle" alt="打开 Align &amp; Stitch API"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../../spacr/resources/icons/workflow/apps/umap.png" width="16.0%" align="middle" alt="打开 Image UMAP API"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../../spacr/resources/icons/workflow/apps/gate_editor.png" width="16.0%" align="middle" alt="打开 Gate Editor API"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../../spacr/resources/icons/workflow/apps/graph_builder.png" width="16.0%" align="middle" alt="打开 Graph Builder API"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../../spacr/resources/icons/workflow/apps/toxoplasma.png" width="16.0%" align="middle" alt="打开 Toxoplasma API"></a>

.. spacr-workflow-end

这里列出所有在主界面上有入口卡片的模块，顺序与主界面一致：先是六个流程模块，然后是其余模块。选择卡片可打开对应模块的 API 页面。

其他资源
~~~~~~~~~~~~~~~

- `交互式教程 <https://einarolafsson.github.io/spacr/tutorials/>`_ — 从安装到筛选命中结果分析的引导式工作流程。
- `Python API 快速启动 <../../source/python_api.rst>`_ - 从脚本、笔记本或集群运行和验证流程。
- `特色引导 RNA <../../source/features.rst>`_ - 能力、成熟度和可选集成。
- `清理 API 参考 <https://einarolafsson.github.io/spacr/api/index.html>`_ - 按任务支持输入点,完整的模块参考一个级别更深。
- `语言与翻译引导 RNA <../../source/localization.rst>`_ — 界面语言、上下文帮助和科学输出政策。

语言与翻译
~~~~~~~~~~~~~~~~~~~~~~

界面的导航和首选项支持十种语言。AI 和 LIVE 控件、模块说明以及经过审核的上下文帮助也会翻译。无需重启，即可在 **spaCR → 首选项 → 语言** 中更改语言。日志、路径、数据库值和测量结果不会被翻译；科学输出始终使用规范英语。请参阅 `上下文帮助政策 <../../source/localization.rst#contextual-help>`_。

The nine non-English catalogs are machine-drafted and technically reviewed rather than read end to end by a native speaker. The `评论范围 <../REVIEW_SCOPE_2026-09-04.md>`_ records which languages have had a human pass and every term left in English by decision.

动画设置指南
~~~~~~~~~~~~~~~~~~~~~~~~~

带有视觉说明的设置会在工具提示中提供 **Animation** 控件。浏览 `设置动画图库 <https://einarolafsson.github.io/spacr/setting_animations.html>`_ 或 `设置动画注册表 <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_。

数据
~~~~

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

模型库
~~~~~~~~~

spaCR ships a catalogue of trained models and fetches them on demand. Open **Model Zoo** from the home screen to browse and install them, or name a key in a settings file -- ``pathogen_model: toxoplasma_pv_v1`` -- and the model is downloaded and checksum-verified the first time it is needed. Every published entry carries a SHA-256; an entry without one is refused rather than installed, because a truncated or substituted checkpoint cannot be told from the real one.

.. spacr-model-zoo-begin

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Model
     - Training data
     - Hold-out performance
   * - ``toxoplasma_pv_v1``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 229 images from 2 datasets, 104 round-1 and 125 newly curated
     - F1 0.864 against 0.713 for stock cpsam on 11 held-out in-house wells, at IoU 0.5; literature hold-out pending
   * - ``toxoplasma_plaque_v1``
       (Cellpose-SAM (cpsam))
     - crystal violet plaque wells; 184 wells from 3 datasets, 95 in-house and 89 literature
     - F1 0.856 in-domain; 0.806 on literature (3-fold cross-validated, SD 0.020)
   * - ``toxoplasma_plaque_v2``
       (Cellpose-SAM (cpsam_v2))
     - 488 curated fields across four domains -- 298 wells cropped from published figures, 96 phone-camera wells, 67 PFA and 27 methanol-fixed whole-well microscope scans; 27,582 plaques
     - not scored against stock; on 81 held-out fields it ties round 3 on literature (0.819 vs 0.820) and beats it by 0.166 on phone-camera wells (0.415 vs 0.249)
   * - ``toxoplasma_well_detector_v1``
       (YOLO11n)
     - whole-plate and multi-well crystal violet images; 562 images from 1 dataset, 190 of them with no well in them
     - mAP50 0.993 on its own held-out split; on the test set shared with v2 it scores mAP50 0.8838, against v2's 0.9457
   * - ``toxoplasma_well_detector_v2``
       (YOLO26n (ultralytics 8.4.155))
     - plate images and literature figures; 1,070 train / 254 val / 129 test, split by PMC article so no paper is in two sets; training data at einarolafsson/toxoplasma-plaque-well-detector-dataset
     - mAP50 0.9457 and mAP50-95 0.8341 against v1's (yolo_welldetect_v3.pt) 0.8838 and 0.7630 on the SAME test set; stock YOLO has no plaque-well class, so v1 is the baseline
   * - ``toxoplasma_from_cellmask_v1``
       (Cellpose-SAM (cpsam_v2))
     - Toxoplasma PV masks predicted from the HOST CELL MASK channel alone; 2567 training and 463 held-out fields, split by well, hosts HFF/HeLa/THP1
     - F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5
   * - ``toxoplasma_pv_v2``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 556 curated images accumulated over five rounds
     - F1 0.817 +/- 0.036 by 5-fold cross-validation over 619 pairs; ~0.86 against 0.713 for stock on the 11 in-house held-out wells
   * - ``toxoplasma_pv_v3``
       (Cellpose-SAM (cpsam_v2))
     - the 556 curated PV fields of round 5, split 437 train / 108 validation / 11 test; training data at einarolafsson/toxoplasma-pv-segmentation-dataset
     - F1 0.860 against stock cpsam_v2's 0.765 on the 11 anchor wells at IoU 0.5; AJI 0.803 against 0.505
   * - ``toxoplasma_pv_v4``
       (Cellpose-SAM (cpsam_v2))
     - round 6's 556 curated fields plus 80 hand-curated fields of a new plate (Anu revision, Replication09182026 plate 1); 502 train / 123 validation / 11 test; training data at einarolafsson/toxoplasma-pv-segmentation-dataset-r7
     - F1 0.854 against stock cpsam_v2's 0.765 on the 11 anchor wells at IoU 0.5; AJI 0.776 against 0.505
   * - ``live_cell_v1``
       (Cellpose-SAM (cpsam_v2))
     - 11,007 transmitted-light fields from 14 public datasets, split by acquisition 6,778 train / 2,030 validation / 2,199 test; training data at einarolafsson/live-cell-segmentation-dataset
     - on the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell
   * - ``nuclei_from_cellmask_v1``
       (Cellpose-SAM (cpsam_v2))
     - nuclei predicted from the HOST CELL MASK channel alone; 453 well-grouped held-out fields, hosts HFF/HeLa/THP1
     - F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5
   * - ``cell_from_hoechst_v1``
       (Cellpose-SAM (cpsam_v2))
     - the HOST CELL outline predicted from the Hoechst (nuclear) channel alone; 2,578 training fields and 451 held-out test fields, split by well so no well is on both sides
     - F1 0.870 against stock cpsam_v2's 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569
   * - ``toxoplasma_from_hoechst_v1``
       (Cellpose-SAM (cpsam_v2))
     - Toxoplasma PV masks predicted from the HOECHST channel alone; 2567 training and 463 held-out fields, split by well, hosts HFF/HeLa/THP1
     - F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5

.. spacr-model-zoo-end

上面的每个图像都是用模型从未在训练中看到的图像来测量的。

**精度**是模型的对象中有多少是真实的; **回忆**是它发现的真实对象中的多少。

**F1**是两个结合,并被引用,因为每个单独是三重播放 - 报告一个不可错误的板,以接近完美的精度,或每个黑暗的泡沫,以靠近完美的回报. 你会更喜欢失去取决于估计,并计算通常更好地通过过呼:板模型被接受的精度 0.858 与回报 0.811 上一个之前的轮子在 0.939 和 0.631.

**IoU**（交并比）是预测对象与参考对象的交集面积除以并集面积。解读分数时应同时查看阈值：“IoU 0.5 时 F1 为 0.864”表示，当交集面积达到并集面积的一半或以上时，该液泡才计为已检出。

**mAP50** 和 **mAP50-95** 属于探测器. 第一问孔是否被发现; 第二重复它在十个从 0.5 到 0.95 的边界,所以它也问每个盒子是多么紧紧地拖动。

**Cross-validated**,与一个**SD**,意味着得分是不同分区的三轮的平均值,而SD是它们移动到多远。

模型托管在各作者自己的 Hugging Face 账户中； ``spacr.model_zoo.publish_model`` 可上传模型，并输出要添加到目录中的条目。


性能诊断
----------------------

生成硬件报告并将其附到性能相关问题中::

    python tools/spacr_hardware_report.py

节省到 ``~/.spacr/reports`` 并打印路径. ``--quick`` 将更长的基准标志; ``--out PATH`` 设置位置。

不读取项目数据。测量导入、数值库运算、窗口构建和动画的耗时，并报告 Apple Silicon 上的 x86_64 仿真情况以及 NumPy 使用的 BLAS 实现。

命令线参考
----------------------

下面的每个命令都以 ``pip install spacr`` 安装,所有命令都会接受 ``--help``。

启动申请
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

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
~~~~~~~~~~~~~~~~~~~~~~~~

请通过 `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_ 提交错误报告和范围明确的功能请求。报告故障时，请提供 spaCR 版本、操作系统、Python 版本、模块设置和相关日志片段。``spacr-doctor`` 会收集其中的大部分信息；报告性能问题时还应附上硬件报告。

许可
~~~~~~~~~

spaCR is released under the `BSD 3 条款许可证 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

如果 spaCR 有助于发表作品,则引用被评估,并且不符合许可的条件,请参见下面的 `引用 spaCR`_。

教程
~~~~~~~~~

`spaCR 交互式教程库 <https://einarolafsson.github.io/spacr/tutorials/>`_ 提供安装和模块使用的分步教程。每节课程均列出可用的旁白和语言。

引用 spaCR
~~~~~~~~~~~~

如果 spaCR 对您的研究有所帮助，请引用：

Olafsson EB, *et al.* 一张以图像为基础的 CRISPR 筛选将 EAF1 定义为 *T. gondii* ESCRT 模块化器。

`生物Rxiv 预印 <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `软件档案 <https://doi.org/10.5281/zenodo.21343316>`_

其他作品引用 spaCR
~~~~~~~~~~~~~~~~~~~~~~~

.. spacr-citing-papers-begin

* `代谢适应性和Toxoplasma gondii中的营养素挖掘:从吸入路径缺乏突变物的见解。 <https://journals.asm.org/doi/full/10.1128/msphere.01011-24>`_
* `IRE1α promotes phagosomal calcium flux to enhance macrophage fungicidal activity. <https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00465-6>`_
* `Toxoplasma GRA8 包含主 ESCRT 配件蛋白质 ALG-2 并且对寄生虫的代谢完整性至关重要。 <https://www.biorxiv.org/content/10.64898/2026.07.20.739547v1.abstract>`_
* `spaCR: CRISPR-Cas9 筛选的空间现象分析(预印版本 1)。 <https://www.researchsquare.com/article/rs-7368254/v1>`_

.. spacr-citing-papers-end

致谢
~~~~~~~~~~~~~~~

spaCR 构建于开放科学软件之上，包括 NumPy、pandas、scikit-image、scikit-learn、Cellpose、PyTorch 和 Qt。有关多语言文档和界面目录所使用的模型，请参阅`翻译模型署名 <../TRANSLATION_MODELS.md>`_。
