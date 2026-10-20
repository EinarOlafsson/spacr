|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

网站： `einarolafsson.github.io/projects/spacr <https://einarolafsson.github.io/projects/spacr/>`_

.. |Website| image:: https://img.shields.io/badge/Website-spaCR-0A7BBB
   :target: https://einarolafsson.github.io/projects/spacr/
   :alt: spaCR 网站

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API&logo=readthedocs&logoColor=white
   :target: https://einarolafsson.github.io/spacr/
   :alt: API 文档

.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: 交互式教程

.. |PyPI| image:: https://img.shields.io/pypi/v/spacr?label=PyPI&logo=pypi&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: PyPI 版本

.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 至 3.14

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/main?logo=codecov&logoColor=white
   :target: https://app.codecov.io/github/EinarOlafsson/spacr
   :alt: Codecov 测试覆盖率

.. |Test counts| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FEinarOlafsson%2Fspacr%2Fnightly%2Fdocs%2Fsource%2F_static%2Ftest-counts.json&cacheSeconds=300&logo=pytest&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: 测试套件

.. |Qt| image:: https://img.shields.io/badge/GUI-Qt-41CD52?logo=qt&logoColor=white
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Qt 界面

.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr?logo=github
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub 问题

.. |License| image:: https://img.shields.io/badge/License-BSD--3-blue?logo=opensourceinitiative&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: BSD 3-Clause 许可证

.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv 预印本

.. |DOI| image:: https://img.shields.io/badge/DOI-blue?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/spacr?label=Conda&logo=anaconda&logoColor=white
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge 版本

.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge 最新发布日期

.. |Docker| image:: https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/pkgs/container/spacr
   :alt: GHCR 上的 Docker 镜像

.. |PyPI Downloads| image:: https://img.shields.io/pepy/dt/spacr?label=downloads&logo=pypi&logoColor=white
   :target: https://pepy.tech/projects/spacr
   :alt: PyPI 下载量

.. |Platforms| image:: https://img.shields.io/badge/Linux%20OSX%20Win-lightgrey?logo=linux&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux、macOS 和 Windows

.. |Cite| image:: https://img.shields.io/badge/Cite-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: 引用 spaCR

.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=Rank&color=brightgreen&logo=pypi&logoColor=white&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: spaCR 在过去 30 个完整日的 PyPI 下载量排名

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/_static/deck/"><picture><source media="(min-width: 1280px)" srcset="../../readme_assets/cover.desktop.svg"><img src="../../readme_assets/cover.svg" alt="spaCR"></picture></a></div>

.. raw:: html

   <div><a href="../../source/_static/deck/pages/57.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/intro-1.desktop.svg"><img src="../../readme_images/zh_CN/intro-1.svg" alt="← 上一页"></picture></a><a href="../../source/_static/deck/pages/02.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/intro-2.desktop.svg"><img src="../../readme_images/zh_CN/intro-2.svg" alt="下一页 →"></picture></a></div>

.. spacr-language-picker-begin

.. raw:: html

   <div><a href="README.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-language-picker.desktop.svg"><img src="../../readme_images/zh_CN/generated-language-picker.svg" alt="语言: 简体中文"></picture></a></div>

.. spacr-language-picker-end

.. raw:: html

   <div><a href="../../readme_text/zh_CN/spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/spacr.desktop.svg"><img src="../../readme_images/zh_CN/spacr.svg" alt="spaCR. CRISPR 筛选的空间表型分析。 spaCR 对显微镜图像中的单细胞进行分割和测量，将逐对象表型与测序得到的向导 RNA 丰度整合，并估计哪些基因与表型变化相关。以孔板图像和 FASTQ 读段为输入，它生成逐对象测量值、训练后的分类器、逐向导 RNA 和逐基因效应估计值，以及按优先级排序的命中结果列表。 分区、测量、标记和分类模块也没有序列手臂运行。 Make Masks 修正分割掩膜，并使用 Box 工具标注带类别标签的独立矩形以进行 YOLO 导出。框保留自己的标签和历史记录，不会更改原始图像或掩膜。 各工具的说明见 功能指南。 图像、掩膜和裁剪图像块是磁盘上的文件。测量值、标注和模型预测是项目 SQLite 数据库 measurements.db 中的表。 可作为桌面应用程序运行，也可在工作站、服务器或集群上以无图形界面方式运行。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/try-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/try-spacr.desktop.svg"><img src="../../readme_images/zh_CN/try-spacr.svg" alt="试用 spaCR."></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/zh_CN/hardware-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/hardware-support.desktop.svg"><img src="../../readme_images/zh_CN/hardware-support.svg" alt="硬件支持."></picture></a></div>

.. spacr-hardware-begin

.. raw:: html

   <div><a href="../../readme_text/zh_CN/generated-hardware-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-hardware-1.desktop.svg"><img src="../../readme_images/zh_CN/generated-hardware-1.svg" alt="NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU, 3D GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU, 3D CPU. 支持（稳定） 已实现（beta） 仅支持 CPU."></picture></a></div>

.. spacr-hardware-end

.. raw:: html

   <div><a href="../../readme_text/zh_CN/install-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/install-spacr.desktop.svg"><img src="../../readme_images/zh_CN/install-spacr.svg" alt="安装 spaCR. 请选择安装方式： 桌面应用程序：适用于 Linux、macOS 或 Windows 的安装程序，自带 Python。 pip：在 Conda 环境中运行 python -m pip install spacr。 conda-forge：在 Conda 环境中运行 conda install conda-forge::spacr。 Docker：在 Linux 容器中运行命令行处理流程，可使用 CPU 镜像或 NVIDIA GPU 镜像："></picture></a></div>

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.5 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.5-cuda12.4 spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/zh_CN/install-spacr-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/install-spacr-2.desktop.svg"><img src="../../readme_images/zh_CN/install-spacr-2.svg" alt="Docker 指南（nightly，英文） 介绍了 GPU 设置、数据挂载和处理流程命令。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/desktop-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/desktop-application.desktop.svg"><img src="../../readme_images/zh_CN/desktop-application.svg" alt="桌面应用程序. 安装器包装自己的 Python. Conda 不需要。"></picture></a></div>

.. spacr-installer-links-begin

| |InstallerLinux|\ |InstallerMacOS|\ |InstallerWindows|

.. |InstallerWindows| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-Windows-Online-Setup.exe"><img src="../../readme_assets/cells/platforms/windows.png" width="14.14%" align="middle" alt="下载适用于 Windows 10/11 的 spaCR 1.5.1.5"></a>
.. |InstallerMacOS| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-macOS-Universal-Online.pkg"><img src="../../readme_assets/cells/platforms/macos.png" width="14.14%" align="middle" alt="下载适用于 macOS 11+（Intel 和 Apple Silicon）的 spaCR 1.5.1.5"></a>
.. |InstallerLinux| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-Linux-x86_64-Online.run"><img src="../../readme_assets/cells/platforms/linux.png" width="14.14%" align="middle" alt="下载适用于 64 位 Linux 的 spaCR 1.5.1.5"></a>

.. spacr-installer-links-end

.. raw:: html

   <div><a href="../../readme_text/zh_CN/desktop-application-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/desktop-application-2.desktop.svg"><img src="../../readme_images/zh_CN/desktop-application-2.svg" alt="在 Linux 上，将下载的文件设为可执行文件并运行："></picture></a></div>

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. raw:: html

   <div><a href="../../readme_text/zh_CN/desktop-application-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/desktop-application-3.desktop.svg"><img src="../../readme_images/zh_CN/desktop-application-3.svg" alt="在 macOS 中,打开 .pkg. 目前的 beta 没有通知; 如果 Gatekeeper 阻止它,请选择 系统设置 → 隐私和安全 → 打开 无论如何。 有关安装、更新、重新安装、卸载、离线使用和故障排查的说明，请参阅 nightly 分支上的当前安装指南（英文）。有关工作站和服务器的配置建议以及 GPU 兼容性表，请参阅 系统要求。"></picture></a></div>

.. raw:: html

   <div><a href="https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installer_guide.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/desktop-application-3-link-1.desktop.svg"><img src="../../readme_images/zh_CN/desktop-application-3-link-1.svg" alt="nightly 分支上的当前安装指南（英文）"></picture></a><a href="../../source/system_requirements.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/desktop-application-3-link-2.desktop.svg"><img src="../../readme_images/zh_CN/desktop-application-3-link-2.svg" alt="系统要求"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/pypi-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/pypi-installation.desktop.svg"><img src="../../readme_images/zh_CN/pypi-installation.svg" alt="使用 PyPI 安装. 如需使用 PyPI 版本，请在 Conda 环境中通过 pip 安装 spaCR。Python 3.12 可选择的科学计算扩展包最为丰富："></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/zh_CN/pypi-installation-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/pypi-installation-2.desktop.svg"><img src="../../readme_images/zh_CN/pypi-installation-2.svg" alt="spaCR 支持 Python 3.9 through 3.14，但 Python 3.14.1 除外，torchvision 不包含该版本。最繁重的 CUDA 和 ROCm 工作流程建议使用 Linux；macOS 和 Windows 也受支持，两者都会使用各自的 GPU — macOS 通过 Metal，它涵盖 Apple Silicon 和 Intel Mac 中的 AMD 显卡，Windows 则通过 CUDA 或 DirectML。 标准安装包含 Qt 桌面界面。在服务器、集群或 CI 运行器上，可运行命令行流程而不打开该界面："></picture></a></div>

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/zh_CN/pypi-installation-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/pypi-installation-3.desktop.svg"><img src="../../readme_images/zh_CN/pypi-installation-3.svg" alt="可选集成单独安装,例如 spacr[zarr]、 spacr[omero]、spacr[napari] 和 spacr[czi,nd2,lif]. 查看完整的附件和 Python 版本兼容性表的 安装指南（nightly，英文）。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/conda-forge-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/conda-forge-installation.desktop.svg"><img src="../../readme_images/zh_CN/conda-forge-installation.svg" alt="使用 conda-forge 安装. 官方 conda-forge 软件包会将 spaCR 及其桌面应用依赖项安装到当前环境中："></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/zh_CN/install-from-source.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/install-from-source.desktop.svg"><img src="../../readme_images/zh_CN/install-from-source.svg" alt="从源代码安装. 克隆代码仓库并以可编辑模式安装。这样工作副本 就是 已安装的软件包，修改无需重新安装即可生效:"></picture></a></div>

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. raw:: html

   <div><a href="../../readme_text/zh_CN/install-from-source-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/install-from-source-2.desktop.svg"><img src="../../readme_images/zh_CN/install-from-source-2.svg" alt="这会克隆默认分支 main，即最新的发布版本。开发在 nightly 分支上进行；如需改为克隆该分支，请添加 --branch nightly。如需克隆特定版本:"></picture></a></div>

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. raw:: html

   <div><a href="../../readme_text/zh_CN/install-from-source-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/install-from-source-3.desktop.svg"><img src="../../readme_images/zh_CN/install-from-source-3.svg" alt="之后如需拉取更新，请在克隆目录中运行:"></picture></a></div>

.. code-block:: bash

   git pull
   pip install -e .

.. raw:: html

   <div><a href="../../readme_text/zh_CN/install-from-source-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/install-from-source-4.desktop.svg"><img src="../../readme_images/zh_CN/install-from-source-4.svg" alt="依赖项或入口点变化时才需重新安装。Python 代码修改会直接生效； spacr-doctor 可确定当前使用的安装。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/install-from-source-light.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/install-from-source-light.desktop.svg"><img src="../../readme_images/zh_CN/install-from-source-light.svg" alt="从源代码安装（精简版）. 贡献者需要版本历史；如果只需运行 spaCR，可选择下列方式。测量于 2026-10-07 对 nightly 的 05302fd5c 提交进行，使用 packaging/measure_clone_forms.sh:"></picture></a></div>

.. code-block:: bash

   # One commit instead of every version: 2048 MB downloaded, 140 s.
   # No history, so no git log, no git blame and no git bisect.
   # git pull still works, but stays shallow until git fetch --unshallow.
   git clone --depth 1 --branch nightly https://github.com/EinarOlafsson/spacr.git
   cd spacr && pip install -e .

   # Runtime files: 104 MB on disk, 6 s (Git: 38 MB; files: 66 MB).
   # No history, docs, tests, tools, features or example data.
   # --with-docs, --with-tests and --with-translations put those back;
   # --dir, --branch, --no-install and --help do the obvious things.
   # packaging/source_install_excludes.txt lists every skipped path.
   curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
   sh install_spacr.sh --branch nightly

.. raw:: html

   <div><a href="../../readme_text/zh_CN/install-from-source-light-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/install-from-source-light-2.desktop.svg"><img src="../../readme_images/zh_CN/install-from-source-light-2.svg" alt="nightly 的完整克隆下载了 9.25 GiB。向浅克隆添加 --filter=blob:none 不会缩小检出文件：其 Git 对象存储仍占用 2032 MB。Git 不报告按需延迟获取的对象，因此没有测得下载总量。nightly 的版本控制文件检出后占用 4657 MB（2026-10-15 测得），不含 Git 历史。下载大小和耗时随分支而变化。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/command-line-entry-points.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/command-line-entry-points.desktop.svg"><img src="../../readme_images/zh_CN/command-line-entry-points.svg" alt="命令行入口."></picture></a></div>

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

.. raw:: html

   <div><a href="../../readme_text/zh_CN/command-line-entry-points-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/command-line-entry-points-2.desktop.svg"><img src="../../readme_images/zh_CN/command-line-entry-points-2.svg" alt="spacr-run --list 会列出具有无界面命令行入口的模块。仅在 GUI 中提供的标注、数据整理、比较和探索模块不会列出。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/core-workflow.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/core-workflow.desktop.svg"><img src="../../readme_images/zh_CN/core-workflow.svg" alt="核心工作流程. 主要工作流程由六个模块组成： Mask 使用 Cellpose 分割细胞、细胞核、病原体和细胞器。 Measure 将形态、强度、纹理、空间和共定位特征以及对象图像裁剪写入 SQLite。 Annotate 在键盘驱动的网格中标注图像裁剪，并支持主动学习队列。 Classify 训练基于图像或测量值的模型，并在每个检查点记录留出数据上的性能。 Map Barcodes 将 FASTQ 读段映射到孔位和 gRNA，并提供丰度、碰撞和覆盖度质控。 Regression 使用适合连续值、比例和计数响应的模型族估计向导 RNA、基因、条件和对照效应。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/spacr-modules.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/spacr-modules.desktop.svg"><img src="../../readme_images/zh_CN/spacr-modules.svg" alt="spaCR 模块."></picture></a></div>

.. spacr-workflow-begin

.. raw:: html

   <div><a href="../../readme_text/zh_CN/generated-workflow-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-workflow-1.desktop.svg"><img src="../../readme_images/zh_CN/generated-workflow-1.svg" alt="核心. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression."></picture></a></div>

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. raw:: html

   <div><a href="../../readme_text/zh_CN/generated-workflow-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-workflow-2.desktop.svg"><img src="../../readme_images/zh_CN/generated-workflow-2.svg" alt="数据. Import images and tables into spaCR projects and execute reproducible multi-plate workflows."></picture></a></div>

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

.. raw:: html

   <div><a href="../../readme_text/zh_CN/generated-workflow-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-workflow-3.desktop.svg"><img src="../../readme_images/zh_CN/generated-workflow-3.svg" alt="工具. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality."></picture></a></div>

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. raw:: html

   <div><a href="../../readme_text/zh_CN/generated-workflow-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-workflow-4.desktop.svg"><img src="../../readme_images/zh_CN/generated-workflow-4.svg" alt="生物体. 针对特定生物体的图像分析和定量实验测定结果。"></picture></a></div>

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../readme_assets/cells/workflow/mask.png" width="14.14%" align="middle" alt="打开 Mask API"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../readme_assets/cells/workflow/measure.png" width="14.14%" align="middle" alt="打开 Measure API"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../readme_assets/cells/workflow/annotate.png" width="14.14%" align="middle" alt="打开 Annotate API"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../readme_assets/cells/workflow/classify_merged.png" width="14.14%" align="middle" alt="打开 Classify API"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../readme_assets/cells/workflow/map_barcodes.png" width="14.14%" align="middle" alt="打开 Map Barcodes API"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../readme_assets/cells/workflow/regression.png" width="14.14%" align="middle" alt="打开 Regression API"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../readme_assets/cells/workflow/apps/foreign.png" width="14.14%" align="middle" alt="打开 Import API"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../readme_assets/cells/workflow/apps/embeddings.png" width="14.14%" align="middle" alt="打开 Embeddings API"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../readme_assets/cells/workflow/apps/run_compare.png" width="14.14%" align="middle" alt="打开 Run Compare API"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../readme_assets/cells/workflow/apps/experiment_design.png" width="14.14%" align="middle" alt="打开 Experiment Design API"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../readme_assets/cells/workflow/apps/power.png" width="14.14%" align="middle" alt="打开 Power / Design API"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../readme_assets/cells/workflow/apps/dose_response.png" width="14.14%" align="middle" alt="打开 Dose–Response API"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../readme_assets/cells/workflow/apps/qc_dashboard.png" width="14.14%" align="middle" alt="打开 QC API"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../readme_assets/cells/workflow/apps/make_masks.png" width="14.14%" align="middle" alt="打开 Make Masks API"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../readme_assets/cells/workflow/apps/align.png" width="14.14%" align="middle" alt="打开 Align &amp; Stitch API"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../readme_assets/cells/workflow/apps/umap.png" width="14.14%" align="middle" alt="打开 Image UMAP API"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../readme_assets/cells/workflow/apps/gate_editor.png" width="14.14%" align="middle" alt="打开 Gate Editor API"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../readme_assets/cells/workflow/apps/graph_builder.png" width="14.14%" align="middle" alt="打开 Graph Builder API"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../readme_assets/cells/workflow/apps/toxoplasma.png" width="14.14%" align="middle" alt="打开 Toxoplasma API"></a>

.. spacr-workflow-end

.. raw:: html

   <div><a href="../../readme_text/zh_CN/spacr-modules-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/spacr-modules-2.desktop.svg"><img src="../../readme_images/zh_CN/spacr-modules-2.svg" alt="这里列出所有在主界面上有入口卡片的模块，顺序与主界面一致：先是六个流程模块，然后是其余模块。选择卡片可打开对应模块的 API 页面。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-resources.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-resources.desktop.svg"><img src="../../readme_images/zh_CN/other-resources.svg" alt="其他资源."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-resources-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-resources-2-1.desktop.svg"><img src="../../readme_images/zh_CN/other-resources-2-1.svg" alt="交互式教程 — 从安装到筛选命中结果分析的引导式工作流程。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-resources-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-resources-2-2.desktop.svg"><img src="../../readme_images/zh_CN/other-resources-2-2.svg" alt="Python API 快速启动 - 从脚本、笔记本或集群运行和验证流程。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-resources-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-resources-2-3.desktop.svg"><img src="../../readme_images/zh_CN/other-resources-2-3.svg" alt="特色引导 RNA - 能力、成熟度和可选集成。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-resources-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-resources-2-4.desktop.svg"><img src="../../readme_images/zh_CN/other-resources-2-4.svg" alt="清理 API 参考 - 按任务支持输入点,完整的模块参考一个级别更深。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-resources-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-resources-2-5.desktop.svg"><img src="../../readme_images/zh_CN/other-resources-2-5.svg" alt="语言与翻译引导 RNA — 界面语言、上下文帮助和科学输出政策。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/language-translation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/language-translation.desktop.svg"><img src="../../readme_images/zh_CN/language-translation.svg" alt="语言与翻译. 界面的导航和首选项支持十种语言。AI 和 LIVE 控件、模块说明以及经过审核的上下文帮助也会翻译。无需重启，即可在 spaCR → 首选项 → 语言 中更改语言。日志、路径、数据库值和测量结果不会被翻译；科学输出始终使用规范英语。请参阅 上下文帮助政策。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/language-translation-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/language-translation-2.desktop.svg"><img src="../../readme_images/zh_CN/language-translation-2.svg" alt="The nine non-English catalogs are machine-drafted and technically reviewed rather than read end to end by a native speaker. The 评论范围 records which languages have had a human pass and every term left in English by decision."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/animated-setting-guidance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/animated-setting-guidance.desktop.svg"><img src="../../readme_images/zh_CN/animated-setting-guidance.svg" alt="动画设置指南. 带有视觉说明的设置会在工具提示中提供 Animation 控件。浏览 设置动画图库 或 设置动画注册表。"></picture></a></div>

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/setting_animations.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/animated-setting-guidance-link-1.desktop.svg"><img src="../../readme_images/zh_CN/animated-setting-guidance-link-1.svg" alt="设置动画图库"></picture></a><a href="https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/animated-setting-guidance-link-2.desktop.svg"><img src="../../readme_images/zh_CN/animated-setting-guidance-link-2.svg" alt="设置动画注册表"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/data.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/data.desktop.svg"><img src="../../readme_images/zh_CN/data.svg" alt="数据. 参考数据集."></picture></a></div>

| |DataBioStudies|\ |DataHuggingFace|\ |DataNCBI|\ |DataSpaCRPower|\ |DataBioRxiv|

.. |DataBioStudies| raw:: html

   <a href="https://doi.org/10.6019/S-BIAD2135"><img src="../../readme_assets/cells/databanks/biostudies_button.png" width="14.14%" align="middle" alt="打开 BioStudies 显微镜数据集"></a>

.. |DataHuggingFace| raw:: html

   <a href="https://huggingface.co/datasets/einarolafsson/toxo_mito"><img src="../../readme_assets/cells/databanks/huggingface_button.png" width="14.14%" align="middle" alt="打开 Hugging Face 测试数据集"></a>

.. |DataNCBI| raw:: html

   <a href="https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935"><img src="../../readme_assets/cells/databanks/ncbi_button.png" width="14.14%" align="middle" alt="打开 NCBI 测序数据集"></a>

.. |DataSpaCRPower| raw:: html

   <a href="https://github.com/maomlab/spaCRPower"><img src="../../readme_assets/cells/databanks/spacrpower_button.png" width="14.14%" align="middle" alt="打开 spaCRPower"></a>

.. |DataBioRxiv| raw:: html

   <a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><img src="../../readme_assets/cells/databanks/biorxiv_button.png" width="14.14%" align="middle" alt="打开 bioRxiv 预印本"></a>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/model-zoo.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/model-zoo.desktop.svg"><img src="../../readme_images/zh_CN/model-zoo.svg" alt="模型库. spaCR ships a catalogue of trained models and fetches them on demand. Open Model Zoo from the home screen to browse and install them, or name a key in a settings file -- pathogen_model: toxoplasma_pv_v1 -- and the model is downloaded and checksum-verified the first time it is needed. Every published entry carries a SHA-256; an entry without one is refused rather than installed, because a truncated or substituted checkpoint cannot be told from the real one."></picture></a></div>

.. spacr-model-zoo-begin

.. raw:: html

   <div><a href="../../models/README.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/heading.svg"><img src="../../models/cards/heading.phone.svg" alt="Default models · held-out scores"></picture></a></div>
   <div><a href="../../models/toxoplasma_pv_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_pv_v4.svg"><img src="../../models/cards/toxoplasma_pv_v4.phone.svg" alt="Score card of Parasite vacuoles v4 (toxoplasma_pv_v4, default): F1 0.761 on 209 held-out fields against 0.377 for stock cpsam_v2. F1 0.761 against stock cpsam_v2&#x27;s 0.377 on the 209 strict hold-out fields at IoU 0.5; AJI 0.582 against 0.040."></picture></a><a href="../../models/toxoplasma_plaque_v5.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_plaque_v5.svg"><img src="../../models/cards/toxoplasma_plaque_v5.phone.svg" alt="Score card of Plaques v5 (toxoplasma_plaque_v5, default): F1 0.816 on 153 held-out fields against 0.227 for stock cpsam_v2. F1 0.816 against stock cpsam_v2&#x27;s 0.227 on the 153 strict hold-out fields at IoU 0.5; AJI 0.737 against 0.178."></picture></a><a href="../../models/toxoplasma_well_detector_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_well_detector_v3.svg"><img src="../../models/cards/toxoplasma_well_detector_v3.phone.svg" alt="Score card of Plaque wells v3 (toxoplasma_well_detector_v3, default): mAP50 0.994 on 73 held-out images. Stock was not evaluated in this run; see the named incumbent comparison on the model card."></picture></a><a href="../../models/toxoplasma_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_cellmask_v1.svg"><img src="../../models/cards/toxoplasma_from_cellmask_v1.phone.svg" alt="Score card of Vacuoles from cell mask v1 (toxoplasma_from_cellmask_v1, default): F1 0.606 on 463 held-out fields against 0.021 for stock cpsam_v2. F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/live_cell_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/live_cell_v1.svg"><img src="../../models/cards/live_cell_v1.phone.svg" alt="Score card of Live cells v1 (live_cell_v1, default): F1 0.694 on 2,199 held-out fields against 0.738 for stock cpsam_v2. On the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell."></picture></a><a href="../../models/nuclei_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/nuclei_from_cellmask_v1.svg"><img src="../../models/cards/nuclei_from_cellmask_v1.phone.svg" alt="Score card of Nuclei from cell mask v1 (nuclei_from_cellmask_v1, default): F1 0.888 on 453 held-out fields against 0.201 for stock cpsam_v2. F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/cell_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/cell_from_hoechst_v1.svg"><img src="../../models/cards/cell_from_hoechst_v1.phone.svg" alt="Score card of Cells from Hoechst v1 (cell_from_hoechst_v1, default): F1 0.870 on 451 held-out fields against 0.301 for stock cpsam_v2. F1 0.870 against stock cpsam_v2&#x27;s 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569."></picture></a><a href="../../models/toxoplasma_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_hoechst_v1.svg"><img src="../../models/cards/toxoplasma_from_hoechst_v1.phone.svg" alt="Score card of Vacuoles from Hoechst v1 (toxoplasma_from_hoechst_v1, default): F1 0.569 on 463 held-out fields against 0.002 for stock cpsam_v2. F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/generated-model-zoo-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-1.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-1.svg" alt="旧模型."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-1-link-1.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-1-link-1.svg" alt="toxoplasma_pv_v3"></picture></a><a href="../../models/toxoplasma_pv_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-1-link-2.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-1-link-2.svg" alt="toxoplasma_pv_v2"></picture></a><a href="../../models/toxoplasma_pv_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-1-link-3.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-1-link-3.svg" alt="toxoplasma_pv_v1"></picture></a><a href="../../models/toxoplasma_plaque_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-1-link-4.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-1-link-4.svg" alt="toxoplasma_plaque_v4"></picture></a><a href="../../models/toxoplasma_plaque_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-1-link-5.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-1-link-5.svg" alt="toxoplasma_plaque_v2"></picture></a><a href="../../models/toxoplasma_plaque_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-1-link-6.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-1-link-6.svg" alt="toxoplasma_plaque_v1"></picture></a><a href="../../models/toxoplasma_well_detector_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-1-link-7.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-1-link-7.svg" alt="toxoplasma_well_detector_v2"></picture></a><a href="../../models/toxoplasma_well_detector_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-1-link-8.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-1-link-8.svg" alt="toxoplasma_well_detector_v1"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/generated-model-zoo-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-2.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-2.svg" alt="替代模型."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-2-link-1.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-2-link-1.svg" alt="toxoplasma_pv_v5"></picture></a><a href="../../models/toxoplasma_plaque_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-2-link-2.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-2-link-2.svg" alt="toxoplasma_plaque_v3"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/generated-model-zoo-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/generated-model-zoo-3.desktop.svg"><img src="../../readme_images/zh_CN/generated-model-zoo-3.svg" alt="掩膜生成器"></picture></a></div>

.. spacr-model-zoo-end

.. raw:: html

   <div><a href="../../readme_text/zh_CN/diagnosing-performance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/diagnosing-performance.desktop.svg"><img src="../../readme_images/zh_CN/diagnosing-performance.svg" alt="性能诊断. 生成硬件报告并将其附到性能相关问题中:"></picture></a></div>

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. raw:: html

   <div><a href="../../readme_text/zh_CN/diagnosing-performance-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/diagnosing-performance-2.desktop.svg"><img src="../../readme_images/zh_CN/diagnosing-performance-2.svg" alt="节省到 ~/.spacr/reports 并打印路径. --quick 将更长的基准标志; --out PATH 设置位置。 不读取项目数据。测量导入、数值库运算、窗口构建和动画的耗时，并报告 Apple Silicon 上的 x86_64 仿真情况以及 NumPy 使用的 BLAS 实现。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/command-line-reference.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/command-line-reference.desktop.svg"><img src="../../readme_images/zh_CN/command-line-reference.svg" alt="命令线参考. 下面的每个命令都以 pip install spacr 安装,所有命令都会接受 --help。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/launching-the-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/launching-the-application.desktop.svg"><img src="../../readme_images/zh_CN/launching-the-application.svg" alt="启动申请."></picture></a></div>

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. raw:: html

   <div><a href="../../readme_text/zh_CN/when-spacr-will-not-start.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/when-spacr-will-not-start.desktop.svg"><img src="../../readme_images/zh_CN/when-spacr-will-not-start.svg" alt="当 spaCR 不开始时."></picture></a></div>

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. raw:: html

   <div><a href="../../readme_text/zh_CN/when-spacr-will-not-start-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/when-spacr-will-not-start-2.desktop.svg"><img src="../../readme_images/zh_CN/when-spacr-will-not-start-2.svg" alt="spacr-doctor 每项检查输出一行，并为每个失败项给出一条可运行的命令。它还会报告路径上的是哪个 spacr。 safespacr 以默认值读取每项偏好设置，并关闭背景、动画、详细日志和预加载。当已保存的偏好设置导致程序无法启动时使用它。在安全模式下保存的偏好设置会正常写入并保留。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/running-modules-headlessly.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/running-modules-headlessly.desktop.svg"><img src="../../readme_images/zh_CN/running-modules-headlessly.svg" alt="无图形界面发运行模块. 没有 Qt,没有显示器 - 用于集群、服务器和CI。"></picture></a></div>

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. raw:: html

   <div><a href="../../readme_text/zh_CN/running-modules-headlessly-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/running-modules-headlessly-2.desktop.svg"><img src="../../readme_images/zh_CN/running-modules-headlessly-2.svg" alt="validate 读取运行时会读取的同一组设置，并列出每个问题（例如缺少的设置或不存在的路径）及其修复方法。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/inspecting-a-run.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/inspecting-a-run.desktop.svg"><img src="../../readme_images/zh_CN/inspecting-a-run.svg" alt="检查一次运行. 每个运行记录为 ~/.spacr/runs 与其设置,加密输入,输出,警告,版本和种子。"></picture></a></div>

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. raw:: html

   <div><a href="../../readme_text/zh_CN/auditing-data-and-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/auditing-data-and-installation.desktop.svg"><img src="../../readme_images/zh_CN/auditing-data-and-installation.svg" alt="数据审计与安装."></picture></a></div>

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. raw:: html

   <div><a href="../../readme_text/zh_CN/environment.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/environment.desktop.svg"><img src="../../readme_images/zh_CN/environment.svg" alt="环境."></picture></a></div>

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. raw:: html

   <div><a href="../../readme_text/zh_CN/environment-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/environment-2.desktop.svg"><img src="../../readme_images/zh_CN/environment-2.svg" alt="旋转日志写为 ~/.spacr/logs/spacr.log. 将此文件添加到错误报告中。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/contributing-and-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/contributing-and-support.desktop.svg"><img src="../../readme_images/zh_CN/contributing-and-support.svg" alt="贡献与支持. 请通过 GitHub Issues 提交错误报告和范围明确的功能请求。报告故障时，请提供 spaCR 版本、操作系统、Python 版本、模块设置和相关日志片段。spacr-doctor 会收集其中的大部分信息；报告性能问题时还应附上硬件报告。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/licensing.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/licensing.desktop.svg"><img src="../../readme_images/zh_CN/licensing.svg" alt="许可. spaCR is released under the BSD 3 条款许可证. 如果 spaCR 有助于发表作品,则引用被评估,并且不符合许可的条件,请参见下面的 引用 spaCR。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/tutorials.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/tutorials.desktop.svg"><img src="../../readme_images/zh_CN/tutorials.svg" alt="教程. spaCR 交互式教程库 提供安装和模块使用的分步教程。每节课程均列出可用的旁白和语言。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/citing-spacr.desktop.svg"><img src="../../readme_images/zh_CN/citing-spacr.svg" alt="引用 spaCR. 如果 spaCR 对您的研究有所帮助，请引用： Olafsson EB, et al. 一张以图像为基础的 CRISPR 筛选将 EAF1 定义为 T. gondii ESCRT 模块化器。"></picture></a></div>

.. raw:: html

   <div><a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/citing-spacr-2-1.desktop.svg"><img src="../../readme_images/zh_CN/citing-spacr-2-1.svg" alt="生物Rxiv 预印"></picture></a><a href="https://doi.org/10.5281/zenodo.21343316"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/citing-spacr-2-2.desktop.svg"><img src="../../readme_images/zh_CN/citing-spacr-2-2.svg" alt="软件档案"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-work-citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-work-citing-spacr.desktop.svg"><img src="../../readme_images/zh_CN/other-work-citing-spacr.svg" alt="其他作品引用 spaCR."></picture></a></div>

.. spacr-citing-papers-begin

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-work-citing-spacr-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-work-citing-spacr-2-1.desktop.svg"><img src="../../readme_images/zh_CN/other-work-citing-spacr-2-1.svg" alt="代谢适应性和Toxoplasma gondii中的营养素挖掘:从吸入路径缺乏突变物的见解。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-work-citing-spacr-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-work-citing-spacr-2-2.desktop.svg"><img src="../../readme_images/zh_CN/other-work-citing-spacr-2-2.svg" alt="IRE1α promotes phagosomal calcium flux to enhance macrophage fungicidal activity."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-work-citing-spacr-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-work-citing-spacr-2-3.desktop.svg"><img src="../../readme_images/zh_CN/other-work-citing-spacr-2-3.svg" alt="Toxoplasma GRA8 包含主 ESCRT 配件蛋白质 ALG-2 并且对寄生虫的代谢完整性至关重要。"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-work-citing-spacr-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-work-citing-spacr-2-4.desktop.svg"><img src="../../readme_images/zh_CN/other-work-citing-spacr-2-4.svg" alt="Hydrogen sulfide dynamically upregulates copper uptake and localization."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/zh_CN/other-work-citing-spacr-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/other-work-citing-spacr-2-5.desktop.svg"><img src="../../readme_images/zh_CN/other-work-citing-spacr-2-5.svg" alt="spaCR: CRISPR-Cas9 筛选的空间现象分析(预印版本 1)。"></picture></a></div>

.. spacr-citing-papers-end

.. raw:: html

   <div><a href="../../readme_text/zh_CN/acknowledgments.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/zh_CN/acknowledgments.desktop.svg"><img src="../../readme_images/zh_CN/acknowledgments.svg" alt="致谢. spaCR 构建于开放科学软件之上，包括 NumPy、pandas、scikit-image、scikit-learn、Cellpose、PyTorch 和 Qt。有关多语言文档和界面目录所使用的模型，请参阅翻译模型署名。"></picture></a></div>
