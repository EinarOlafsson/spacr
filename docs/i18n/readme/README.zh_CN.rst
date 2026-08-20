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

.. image:: ../../../spacr/resources/icons/logo_spacr.png
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

**CRISPR 筛选的空间表型分析。**

spaCR 对高内涵显微镜图像中的单细胞进行分割和测量，将每个细胞与其获得的 gRNA 关联，并报告哪些基因改变了表型。输入为孔板图像和 FASTQ 读段；输出包括逐对象测量、训练后的分类器、逐向导 RNA 和逐基因效应量，以及按优先级排序的候选结果列表。

对于基于图像的混合 CRISPR 筛选，这涵盖了完整工作流程。如果只有高内涵显微镜数据而没有筛选实验，也可以单独运行分割、测量、标注和分类部分。

图像、掩膜、图像裁剪、测量值、标注、预测、条形码和孔位标识符都存储在同一个 SQLite 项目中，因此结果中的数值可以追溯到其来源对象。

spaCR 可作为桌面应用程序运行，也可在工作站、服务器或集群上以无图形界面模式运行。两种方式使用相同的模块；模块支持 CUDA 时会自动启用。


工作流程概览
--------------------

|WorkflowMask| |WorkflowArrow| |WorkflowMeasure| |WorkflowArrow| |WorkflowAnnotate| |WorkflowArrow| |WorkflowClassify| |WorkflowArrow| |WorkflowBarcodes| |WorkflowArrow| |WorkflowRegression|

.. |WorkflowMask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 96
   :alt: Mask API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |WorkflowMeasure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 96
   :alt: Measure API
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |WorkflowAnnotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 96
   :alt: Annotate API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/annotate_engine/index.html
   :align: middle
.. |WorkflowClassify| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 96
   :alt: Classify API
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |WorkflowBarcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 96
   :alt: Map Barcodes API
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |WorkflowRegression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 96
   :alt: Regression API
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |WorkflowArrow| image:: ../../../spacr/resources/icons/workflow/arrow.png
   :width: 18
   :align: middle

.. image:: ../../../spacr/resources/icons/workflow_home_apps.png
   :alt: spaCR 工作流程及输出结构
   :align: center

主要路径是按 → 测量 → 标记 → 分类 → 地图条形码 → 退缩. 下面的网格包含相同类别的所有其他应用程序,并在 spaCR 主界面上使用的顺序。


安装 spaCR
-------------

桌面应用程序
~~~~~~~~~~~~~~~~~~~

桌面安装器包含私人 Python 环境,因此不需要 conda 和现有 Python 安装。

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

在 Linux 上,使下载的文件可执行并运行:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

在 macOS 中,打开 ``.pkg``. 目前的 beta 没有通知; 如果 Gatekeeper 阻止它,请选择 **系统设置 → 隐私和安全 → 打开 无论如何**。

请参见 `安装导游 <../../source/installers.rst>`_ 更新、拆除、离线和解决问题的指示。

Python 安装
~~~~~~~~~~~~~~~~~~~

Python 3.12 有最广泛的选择可选的科学包:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR 支持 Python **3.9 至 3.14**,除 Python 3.14.1 外,除此之外, torchvision 除外. Linux 适用于 CUDA 工作流; macOS 和 Windows 也支持。

对于服务器、集群或 CI 运行器,请忽略 Qt:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

可选集成单独安装,例如 ``spacr[ome-zarr]``、 ``spacr[omero]``、``spacr[napari]`` 和 ``spacr[czi,nd2,lif]``. 查看完整的附件和 Python 版本兼容性表的 `安装导游 <../../source/installers.rst>`_。

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

在解决问题时设置 ``SPACR_LOG_LEVEL=DEBUG``. 旋转日志以 ``~/.spacr/logs/spacr.log`` 写作. 经典 Tk 界面仍然可用为 ``spacr-legacy`` 但不再开发。


您可以做什么
---------------

大多数筛选遵循六个模块:

- **Mask** 细胞、核、病原体和有机细胞与 Cellpose。
- **Measure** 写到 SQLite 的形状、强度、结构、空间和定位特征,以及对象图像裁剪。
- **Annotate** 标签在键盘驱动的网络中生长,并支持活跃学习曲线。
- **Classify** 列车以图像或测量为基础的模型和记录每个检查点的性能。
- **Map Barcodes** 地图 FASTQ 阅读到孔和 gRNAs,与丰富,碰撞和覆盖 QC。
- **Regression** estimates guide, gene, condition and control effects with model families suited to continuous, fractional and count responses.

同一项目还可以设计孔板、估算统计功效、校正批次效应、检查分割质量、探索相互关联的图表和图像裁剪、导出 AnnData、恢复中断的工作，并记录每项结果所使用的设置。

选择下一个页面,根据你想做的事情:

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


贡献与支持
------------------------

Bug reports and focused feature requests are welcome through `GitHub 问题 <https://github.com/EinarOlafsson/spacr/issues>`_. When reporting a failure, include the spaCR version, operating system, Python version, module settings and the relevant log excerpt. ``spacr-doctor`` collects most of that for you.

许可
~~~~~~~~~

目前的开发分支源可在 `非商用许可证 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. 商业使用需要从版权持有者的单独许可。 通过 spaCR 1.4.9.9 发布的版本仍然可在 MIT 许可证下,伴随这些出版物。

教程
~~~~~~~~~

`互动式 spaCR 教程图书馆 <https://einarolafsson.github.io/spacr/tutorials/>`_ 包含安装和每个应用程序工作流的描述、标签的步行路径,在 73 个课程中,在 8 种语言中有 50 个声音。

引用 spaCR
~~~~~~~~~~~~

如果 spaCR 有助于您的研究,请引用:

Olafsson EB, *et al.* 一张以图像为基础的 CRISPR 筛选将 EAF1 定义为 *T. gondii* ESCRT 模块化器。

`生物Rxiv 预印 <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `软件档案 <https://doi.org/10.5281/zenodo.21343317>`_

致谢
~~~~~~~~~~~~~~~

spaCR 构建于开放科学软件之上，包括 NumPy、pandas、scikit-image、scikit-learn、Cellpose、PyTorch 和 Qt。有关多语言文档和界面目录所使用的模型，请参阅`翻译模型署名 <../TRANSLATION_MODELS.md>`_。
