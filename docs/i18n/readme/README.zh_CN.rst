|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentation
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Interactive tutorials
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI version
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 through 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Test suite
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt interface
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub source
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub issues
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm Noncommercial license
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Latest installers
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge recipe

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
   :alt: spaCR workflow and output organization
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

spaCR supports Python **3.9 through 3.14** (except Python 3.14.1, which torchvision excludes). Python 3.12 has the widest choice of optional scientific packages. Linux is recommended for CUDA workflows; macOS and Windows are also supported.


安装详情
--------------------

此分類上一篇: |Release| |PyPI| |CondaRecipe|

**(beta) Lightweight 桌面安装器:**

.. spacr-installer-links-begin

* `Windows 10/11:下载 SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe>`_
* `macOS 11+ (英特尔和苹果硅):下载SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg>`_
* `64 位 Linux:下载 SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

轻量级安装程序 — 无需 conda 或现有 Python 环境
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The installer downloads a private Python 3.12 runtime, Qt, PyTorch, spaCR and the scientific dependencies during installation, so neither conda nor an existing Python is needed. The portable CPU build is the default, which keeps the installation from pulling several gigabytes of CUDA libraries unannounced. Windows offers NVIDIA acceleration as an optional installer component, Linux accepts ``--torch-backend auto``, and the standard macOS PyTorch wheel keeps Apple MPS acceleration.

安装帮助、进展和错误,以所有十种语言(spaCR)跟踪操作系统语言:英语、瑞典语、德语、西班牙语、简化中国语、葡萄牙语、印度语、韩语、冰岛语和法语。

在 Linux 上,在打开之前,让下载的安装程序执行:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

在 macOS 中,打开下载的 ``.pkg`` 如果 Gatekeeper 阻止当前 beta 安装程序,因为它没有被注册,打开 **系统设置 → 隐私和安全**,选择 ** 无论如何打开** 为 spaCR,然后重新运行包。

The installer validates spaCR, Qt, PyTorch and dependency consistency before replacing an older installation, so an interrupted update leaves the previous working environment in place. A diagnostic log is kept as ``install.log`` inside the private spaCR installation directory.

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

安装仅剩余的工作流需要:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

哪个额外的分辨率取决于 Python 版本. 在 Python 3.13 ,超极限限限为 ``spacr[all]`` 和 TorchCAM 的 NumPy 限制限为``attribution`` 额外; 核心包和 Qt 应用程序不受影响。 在 Python 3.14 , btrack 通过其额外可用。

遗传 Tk 界面仍然安装为 ``spacr-legacy`` 但不再开发。


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

设置 ``SPACR_LOG_LEVEL=DEBUG`` 在解决问题时. 旋转日志写为 ``~/.spacr/logs/spacr.log``。


功能
--------

大多数筛选实验使用的六个模块
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**按摩**细胞,核,病原体和有机细胞与Cellpose,在2D图像和量或时间序列数据。 模型列表是从安装的Cellpose而不是硬编码,而一个对象的直径是从图像之前开始的估计。

**测量**写到项目数据库的物质形态,强度,结构和配置特性,以及作物。新在 1.5.0.0:照明纠正估计板本身的平面,并分开它之前任何强度特性被采取,这取消了板温地图显示为边缘效果的良好位置比亚。 一个分区QC旗帜表明在平面语言中,面具看起来像什么,直到测量运行; 报告,它不会阻止。

** 注意** 显示基板驱动的网上的种子,并写标签直至 SQLite. 现在关闭了活跃学习圈:在没有离开屏幕的情况下,重新排序的尾巴,观看学习曲线,并获得停止判断,当进一步的标签停止改变模型时,它将对一个模型进行回归。

**Classify**列车 PyTorchCNN和转换器在注册的作物,和经典或增强的模型在测量表。 每个类的准确性现在保持每个时代而不是被排除,每个检查点得到一个模型卡记录其数据集,类平衡,分裂规则和持有的测量。 在评估屏幕上,一个混乱的矩阵细胞是一个问题:点击它打开这些作物,与安全错误的预测列出与不确定。

**Map Barcodes** decodes row, column and gRNA barcodes from FASTQ reads, assigns guide identities to wells, and joins them to imaged cells. Barcode QC reports reads per well, collision rate and unmapped fraction, sweeping around the number of gRNAs per well you say you expect rather than a fixed threshold.

** 回归** 估计指南,基因,状态和控制效果,使用17个模型家庭,包括混合模型,物流和 probit,量子,beta,GLM与量子比诺变量,拉索,雷吉,弹性网,环和马匹。

1.5.0.0 新增功能
~~~~~~~~~~~~~~

在屏幕存在之前,电源 / 设计模块回答需要多少细胞和多少井,以序列错误而定价,并以从图像过薄的井中产生的滴滴。 实验设计师将板块、控制器和复制器放出并出口管道的布局。 随后,一个 QC 板块将分区、孔板,笔记本商协议和泄漏检查到一个判决中, ComBat 可在 ``center`` 和 ``zscore`` 旁边进行包装纠正。

结果被探索而不是出口和重新进口. 一张图形建筑师将一个表格,拖拉列到x,y,颜色,尺寸和面孔. 门控拖在一个希斯托格拉或分散器变成过滤器. 一个功能探测器排序的特点是他们如何分开类。 小多元,剂量答案匹配,控制图表和强大的外部检测使用相同的轴发动机. 在一个视野中选择对象,并打开一个选项带来收获这些对象来自。 一个层观察员将图像,标签,点和形状,与正方形景,一个同步比较网,以及从核到病原的树线。

Runs are now identifiable. Each carries one run id, one seed and an ``on_error`` policy; Mask, Measure, Classify and the AnnData export register what they wrote in an artifact registry, so an output file leads back to the settings that produced it. A module opens on what the previous step actually wrote, the pipeline graph marks which outputs are stale, run comparison diffs the settings, object counts and hit lists of two runs, and every GUI run emits the equivalent Python script. Measurements export to ``.h5ad`` for scanpy; OME-Zarr and OMERO are available through the Python API. The methods-and-results exporter drafts those two manuscript sections from a structured digest of the run: the model writes the prose, but every number comes from the digest, and a draft containing a number the digest does not contain is rejected. 当安装错误时, ``spacr-doctor`` 报告 spaCR 实际上运行,是否 GPU 可用,是否Cellpose 符合 API spaCR 通话,以及项目数据库和设置是否有声音,并且每个线上都可以复制的修正,而不是通道。

多语言桌面界面
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → 偏好 → 语言** 将运行应用程序翻译成英语、瑞典语、德语、西班牙语、曼达林语、葡萄牙语、印度语、韩国语、冰岛语或法语,而无需重新启动。

Navigation, Preferences, AI and LIVE controls, module descriptions and spaCR-authored console notices follow the selected language. Worker output, logs, tracebacks, paths, database values, annotations, AI responses, measurements and saved results are never translated, so scientific output remains canonical English. Setting tooltips not yet reviewed in a language stay in English rather than becoming a mixed-language explanation. The `位置指南 <https://einarolafsson.github.io/spacr/localization.html>`_ documents the behavior, the environment override, and the `背景援助 <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ that is translated with it.

动画设置指南
~~~~~~~~~~~~~~~~~~~~~~~~~

94 short animations explain what 143 visual settings do to an image. Hover a setting and click **Animation** in its tooltip to play the square beside the text; click it again to fold it away. Animations are off until asked for, and can be disabled in Preferences. The `画廊 <https://einarolafsson.github.io/spacr/setting_animations.html>`_ shows all of them, and the `创建动画记录 <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ records which setting each one belongs to.

模块参考
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Module
     - Feature
     - State
     - Description
   * - **Desktop experience**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - Stable
     - Retranslates open and lazily created screens across ten bundled languages.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Stable
     - Localizes module summaries and setting-help chrome while preserving exact API URLs.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Stable
     - Localizes AI and LIVE controls without changing user or model content.
   * - |api-animations|_
     - |doc-animations|_
     - Stable
     - Plays 94 packaged animations for 143 visual settings from the setting tooltip.
   * - |api-selection|_
     - |api-linked-views|_
     - Alpha
     - Shares one object selection across the table, plate, embedding, scatter and graph views.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alpha
     - Diagnoses the install — GPU, Cellpose API, database, settings — with a fix per failing check.
   * - **Image analysis**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Stable
     - Segments cells, nuclei, pathogens and organelles in 2D images.
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - Segments volumetric images and 4D time series.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alpha
     - Estimates the flat-field from the plate and divides it out before intensity is measured.
   * - |api-measure|_
     - |api-measure-2d|_
     - Stable
     - Measures morphology, intensity, texture and colocalization, and writes the crops.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alpha
     - States what the segmentation looks like before Measure runs, without blocking it.
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - Tracks objects with IoU, Trackpy, btrack, Trackastra or ultrack, and quantifies motility.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alpha
     - Stacks image, label, point and shape layers, with orthogonal views and a comparison grid.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alpha
     - Hands a mask to napari for correction and takes it back, recording every edit.
   * - **AI and phenotyping**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Stable
     - Reviews crops on a keyboard-driven grid and saves annotations to SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alpha
     - Retrains inside Annotate, re-ranks by uncertainty, and says when labelling can stop.
   * - |api-classify|_
     - |api-classification|_
     - Stable
     - Trains and applies PyTorch CNN and transformer models.
   * - |api-classify|_
     - |api-model-cards|_
     - Alpha
     - Records dataset, class balance, split rule and held-out metrics beside each checkpoint.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alpha
     - Opens the crops behind a confusion cell, confident errors listed apart from uncertain ones.
   * - |api-ml|_
     - |api-ml-models|_
     - Stable
     - Trains interpretable classical and boosted models on measurement tables.
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - Explains predictions with Captum, SmoothGrad and TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - Explores image embeddings interactively and propagates cluster labels.
   * - **Sequencing and screen analysis**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Stable
     - Maps row, column and gRNA barcodes from FASTQ reads and assigns guides to imaged cells.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alpha
     - Reports reads per well, collision rate and unmapped fraction against the expected gRNAs per well.
   * - |api-regression|_
     - |api-regression-models|_
     - Stable
     - Estimates guide, gene, condition and control effects with 17 model families.
   * - |api-power|_
     - |api-power-design|_
     - Alpha
     - Answers how many cells and wells a screen needs, with sequencing error and well dropout priced in.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alpha
     - Builds a plot by dragging columns onto x, y, colour, size and facet.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alpha
     - Records the run id, seed and settings behind mask, measure, classify and export outputs.

.. |api-qt-app| replace:: **Qt application**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Ten-language localization**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Localized contextual help**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI-assisted console**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Setting animation registry**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Visual setting animations**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Selection**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Linked selection**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Installation diagnosis**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **2D mask generation**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **3D and 4D mask generation**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Illumination**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Flat-field correction**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Object measurements**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **Segmentation QC**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Pre-run verdict**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Object tracking**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Layers**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Layer viewer**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari bridge**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Mask curation**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Manual annotation**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Active learning**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Retrain and re-rank**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Image classification**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Model cards**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Activation maps**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Confusion drill-down**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Machine learning**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Measurement classification**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Interactive embedding**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Sequencing**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Map barcodes**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **Barcode QC**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Well and collision report**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Screen effect estimation**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Power and design**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Artifacts**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Run provenance**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


数据
----

参考数据集
~~~~~~~~~~~~~~~~~~

- `全微镜数据集:BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `测试数据集:Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `序列数据:NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Power analysis: spaCRPower <https://github.com/maomlab/spaCRPower>`_


贡献与支持
------------------------

Bug reports and focused feature requests are welcome through `GitHub 問題 <https://github.com/EinarOlafsson/spacr/issues>`_. When reporting a failure, include the spaCR version, operating system, Python version, module settings and the relevant log excerpt. ``spacr-doctor`` collects most of that for you.

许可
~~~~~~~~~

目前的开发分支在 `非商用许可证 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_ 下可用。 商业使用需要从版权持有者获得单独许可. 通过 spaCR 1.4.9.9 发布的版本仍然可用,根据 MIT 许可证,这些出版物伴随。

教程
~~~~~~~~~

`互动式 spaCR 教程图书馆 <https://einarolafsson.github.io/spacr/tutorials/>`_ 包含在八种语言中描述、标记的安装和每个应用程序工作流程的步行路径。

引用 spaCR
~~~~~~~~~~~~

如果 spaCR 有助于您的研究,请引用:

Olafsson EB, *et al.* 基于图像的集成屏幕 CRISPR 将 EAF1 定义为 ESCRT 子转换器的 *T. gondii* 模块化器。

`生物Rxiv 预印 <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `软件档案 <https://doi.org/10.5281/zenodo.21343317>`_
