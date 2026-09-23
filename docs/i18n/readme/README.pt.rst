|Platforms| |Python| |Qt| |Tests| |Release| |Issues| |Source| |Conda| |PyPI| |Conda Downloads| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API%20Documentation
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentação
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Tutoriais interativos
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: Versão no PyPI
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 a 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Suíte de testes
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Interface Qt
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: Código-fonte no GitHub
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: Problemas no GitHub
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: Licença BSD 3-Clause
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-2026.07.08.737057-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: Preprint no bioRxiv
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: DOI do Zenodo
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Instaladores mais recentes
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Versão no conda-forge
.. |Conda Downloads| image:: https://anaconda.org/conda-forge/spacr/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Downloads no conda-forge
.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Data da versão mais recente no conda-forge
.. |PyPI Downloads| image:: https://static.pepy.tech/personalized-badge/spacr?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads
   :target: https://pepy.tech/projects/spacr
   :alt: Downloads no PyPI
.. |Platforms| image:: https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS e Windows
.. |Cite| image:: https://img.shields.io/badge/Cite-CITATION.cff-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: Citar o spaCR
.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=PyPI+rank+%2830d%29&color=brightgreen&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: Classificação de downloads do spaCR no PyPI nos últimos 30 dias completos

.. image:: ../../source/_static/deck/slides/slide_01.jpg
   :alt: spaCR
   :width: 920
   :target: https://einarolafsson.github.io/spacr/_static/deck/

`← Anterior <../../source/_static/deck/pages/51.md>`_   `Próximo → <../../source/_static/deck/pages/02.md>`_

spaCR
=====

.. spacr-language-picker-begin

Idiomas: `🌐 Português ▾ <README.md>`_

.. spacr-language-picker-end

**Análise espacial de fenótipos em triagens CRISPR.**

O spaCR segmenta e mede células individuais em imagens de microscopia de alto conteúdo, integra fenótipos por objeto à abundância de guias derivada do sequenciamento e estima quais genes estão associados a alterações fenotípicas. A partir de imagens de placas e leituras FASTQ, ele produz medições por objeto, classificadores treinados, estimativas de efeito por guia e por gene e uma lista de resultados classificada.

Os módulos de segmentação, medição, anotação e classificação também são executados sem um braço de sequenciamento.

Imagens, máscaras, recortes, medições, anotações, previsões, códigos de barras e identificadores de poço ficam em um único projeto SQLite.

É executado como um aplicativo de desktop ou sem interface gráfica em uma estação de trabalho, servidor ou cluster.

Suporte de hardware
~~~~~~~~~~~~~~~~~~~

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


Instalar o spaCR
----------------

Aplicativo para desktop
~~~~~~~~~~~~~~~~~~~~~~~

Os instaladores empacotam seus próprios Python. Conda não é necessário.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Baixar o spaCR 1.5.0.9 para Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.9/spaCR-1.5.0.9-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Baixar o spaCR 1.5.0.9 para macOS 11+ (Intel e Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.9/spaCR-1.5.0.9-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Baixar o spaCR 1.5.0.9 para Linux de 64 bits
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.9/spaCR-1.5.0.9-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Instaladores anteriores do spaCR
   :target: ../../source/installers.rst

.. spacr-installer-links-end

Os três primeiros ícones baixam a versão atual. O ícone spaCR abre o arquivo completo do instalador. Os links do instaladores e os nomes dos arquivos versionados são atualizados pelo fluxo de trabalho da versão; os instaladores anteriores permanecem no mesmo arquivo de versão.

No Linux, torne o arquivo baixado executável e execute-o:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

No macOS, abra o arquivo ``.pkg``. A versão beta atual não é notarizada; se o Gatekeeper a bloquear, selecione **Ajustes do Sistema → Privacidade e Segurança → Abrir Mesmo Assim**.

Veja as instruções `guia do instalador <../../source/installer_guide.rst>`_ para atualização, desinstalação, off-line e solução de problemas.

Instalação pelo PyPI
~~~~~~~~~~~~~~~~~~~~

Para usar a versão publicada no PyPI, instale o spaCR com pip em um ambiente Conda. O Python 3.12 oferece a maior variedade de pacotes científicos opcionais:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

O spaCR oferece suporte ao Python **3.9 through 3.14**, exceto ao Python 3.14.1, que é excluído pelo torchvision. Recomenda-se Linux para os fluxos de trabalho mais pesados com CUDA e ROCm; macOS e Windows também são compatíveis, e ambos utilizam suas GPUs — macOS por meio do Metal, que abrange o Apple Silicon e as placas gráficas AMD dos Macs Intel, e Windows por meio de CUDA ou DirectML.

Em um servidor, cluster ou executor de CI, omita o Qt:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

As integrações opcionais são instaladas separadamente, por exemplo ``spacr[zarr]``,  ``spacr[omero]``,``spacr[napari]`` e ``spacr[czi,nd2,lif]``. Veja a tabela de compatibilidade  `guia de instalação <../../source/installer_guide.rst>`_ para os extras completos e ?Python-versão.

Instalação com conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~

O pacote oficial do conda-forge instala o spaCR e suas dependências de desktop no ambiente ativo:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Instalação a partir do código-fonte
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone o repositório e instale-o no modo editável, para que sua cópia de trabalho *seja* o pacote instalado e as alterações passem a valer sem reinstalar::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

Isso clona ``main``, o branch padrão, que contém a versão mais recente. O desenvolvimento acontece em ``nightly``; adicione ``--branch nightly`` para clonar esse branch no lugar dele. Para uma versão específica::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

Para obter alterações posteriores, execute de dentro do clone::

    git pull
    pip install -e .

A segunda linha só é necessária quando dependências ou pontos de entrada mudaram; o código Python é atualizado sem ela. Se um comando ainda executar código antigo depois do pull, o ``spacr-doctor`` informa qual ``spacr`` está de fato no seu caminho de busca, o que costuma ser a causa.

Instalação a partir do código-fonte (leve)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quem contribui precisa do histórico; para apenas executar o spaCR, use uma destas opções, medidas em 2026-09-15 com ``packaging/measure_clone_forms.sh``::

    # One commit instead of every version: 540 MB downloaded, 69 s.
    # No history, so no git log, no git blame and no git bisect.
    # git pull still works, but stays shallow until git fetch --unshallow.
    git clone --depth 1 https://github.com/EinarOlafsson/spacr.git
    cd spacr && pip install -e .

    # Only the files spaCR runs from: 81 MB on disk, 39 s. No history
    # either, and no docs, tests, tools or example data.
    # --with-docs, --with-tests and --with-translations put those back;
    # --dir, --branch, --no-install and --help do the obvious things.
    # packaging/source_install_excludes.txt lists every skipped path.
    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

O clone completo baixa 5,8 GB para um checkout de 1186 MB. Adicionar ``--filter=blob:none`` a esse clone não economiza nada: o checkout busca os blobs de qualquer forma.


Comandos de linha de comando
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Defina ``SPACR_LOG_LEVEL=DEBUG`` durante a solução de problemas. Os logs rotativos são gravados em ``~/.spacr/logs/spacr.log``.

``spacr-run --list`` lista os módulos com pontos de entrada de linha de comando para execução sem interface gráfica. Os módulos de anotação, curadoria, comparação e exploração disponíveis apenas na interface gráfica são omitidos.


Fluxo de trabalho principal
---------------------------

O fluxo de trabalho principal compreende seis módulos:

- **Mask** segmenta células, núcleos, patógenos e organelas com Cellpose.
- **Measure** grava no SQLite características de morfologia, intensidade, textura, espaciais e de colocalização, além de recortes dos objetos.
- **Annotate** rotula recortes em uma grade controlada pelo teclado e oferece suporte a filas de aprendizado ativo.
- **Classify** treina modelos baseados em imagens ou medições e registra, em cada checkpoint, o desempenho nos dados de validação reservados.
- **Map Barcodes** associa as leituras FASTQ aos poços e gRNAs, com controle de qualidade de abundância, colisões e cobertura.
- **Regression** estima efeitos de guias, genes, condições e controles com famílias de modelos adequadas a respostas contínuas, fracionárias e de contagem.

Módulos do spaCR
----------------

.. spacr-workflow-begin

Principal
^^^^^^^^^

Core sequence from microscopy images through segmentation, measurements,
annotations, classification, barcode mapping and regression.

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

Dados
^^^^^

Import images and tables into spaCR projects and execute reproducible
multi-plate workflows.

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

Ferramentas
^^^^^^^^^^^

Point these at a project: edit masks by hand, stitch tiles, read an
embedding, draw a gate, build a plot, check quality.

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

Ensaios
^^^^^^^

Quantitative readouts for biological assays.

| |Module_toxoplasma|\ |Module_plasmodium|\ |Module_candida|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: Abrir a API de Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks
   :align: middle
.. |Module_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 16.0%
   :alt: Abrir a API de Measure
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Module_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 16.0%
   :alt: Abrir a API de Annotate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Module_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 16.0%
   :alt: Abrir a API de Classify
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Module_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 16.0%
   :alt: Abrir a API de Map Barcodes
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Module_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 16.0%
   :alt: Abrir a API de Regression
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Module_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.0%
   :alt: Abrir a API de Import
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |Module_embeddings| image:: ../../../spacr/resources/icons/workflow/apps/embeddings.png
   :width: 16.0%
   :alt: Abrir a API de Embeddings
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html
   :align: middle
.. |Module_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.0%
   :alt: Abrir a API de Run Compare
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |Module_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.0%
   :alt: Abrir a API de Experiment Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |Module_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.0%
   :alt: Abrir a API de Power / Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |Module_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.0%
   :alt: Abrir a API de Dose–Response
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |Module_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.0%
   :alt: Abrir a API de QC
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |Module_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.0%
   :alt: Abrir a API de Make Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |Module_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.0%
   :alt: Abrir a API de Align & Stitch
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |Module_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.0%
   :alt: Abrir a API de Image UMAP
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap
   :align: middle
.. |Module_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.0%
   :alt: Abrir a API de Gate Editor
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |Module_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.0%
   :alt: Abrir a API de Graph Builder
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |Module_toxoplasma| image:: ../../../spacr/resources/icons/workflow/apps/toxoplasma.png
   :width: 16.0%
   :alt: Abrir a API de Toxoplasma
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html
   :align: middle
.. |Module_plasmodium| image:: ../../../spacr/resources/icons/workflow/apps/plasmodium.png
   :width: 16.0%
   :alt: Abrir a API de Plasmodium spp.
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html
   :align: middle
.. |Module_candida| image:: ../../../spacr/resources/icons/workflow/apps/candida.png
   :width: 16.0%
   :alt: Abrir a API de Candida spp.
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html
   :align: middle

.. spacr-workflow-end

Todos os módulos que acompanham o spaCR, na ordem em que a tela inicial os apresenta: primeiro os seis módulos do pipeline, depois todos os demais. Selecione um ícone para abrir a página da API desse módulo.

Consulte o `guia de funcionalidades <../../source/features.rst>`_ para conhecer cada ferramenta.

Outros recursos
~~~~~~~~~~~~~~~

- `Tutoriais interativos <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 guided workflows from installation por meio de hit investigation.
- `Python  API início rápido <../../source/python_api.rst>`_ — run and validate pipelines from scripts, notebooks or a cluster.
- `Guia de funcionalidades <../../source/features.rst>`_ capacidades, maturidade e integrações opcionais.
- `Referência API com curadoria <https://einarolafsson.github.io/spacr/api/index.html>`_ — supported entry points by task, with the complete module reference one level deeper.
- `Guia de idiomas e tradução <../../source/localization.rst>`_ linguagens de interface, ajuda contextual e política de saída científica.

Idioma e tradução
~~~~~~~~~~~~~~~~~~~~~~

A interface oferece dez idiomas na navegação e nas preferências. Os controles AI e LIVE, as descrições dos módulos e a ajuda contextual revisada também são traduzidos. Altere o idioma em **spaCR → Preferências → Idioma** sem reiniciar. Logs, caminhos, valores de banco de dados e medições nunca são traduzidos; a saída científica permanece em inglês canônico. Consulte a `política de ajuda contextual <../../source/localization.rst#contextual-help>`_.

Os nove catálogos não ingleses são elaborados por máquina e tecnicamente revisados em vez de lidos de ponta a ponta por um falante nativo de cada idioma. Os registros `âmbito de revisão <../REVIEW_SCOPE_2026-09-04.md>`_ que as línguas tiveram um passe humano, quanto do corpus que cobre e cada termo deixado em inglês por decisão.

Guia animado de configurações
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As configurações com uma explicação visual oferecem um controle **Animation** na dica de ferramenta. Consulte a `galeria de animações de configurações <https://einarolafsson.github.io/spacr/setting_animations.html>`_ ou o `registro de animações de configurações <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Dados
-----

Conjuntos de dados de referência
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: Abrir o conjunto de microscopia no BioStudies
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Abrir o conjunto de teste no Hugging Face
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: Abrir o conjunto de sequenciamento no NCBI
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: Abrir o spaCRPower
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: Abrir a pré-publicação no bioRxiv
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

Modelo zoo
~~~~~~~~~~

spaCR envia um catálogo de modelos treinados e os busca sob demanda. Abra  **Modelo Zoo** da tela inicial para navegar e instalá-los, ou nomeie uma chave em um arquivo de configurações -  ``pathogen_model: toxoplasma_pv_v1`` - e o modelo é baixado e verificado na primeira vez que é necessário. Cada entrada publicada carrega um SHA-256; uma entrada real sem um é recusada em vez de instalada, porque um truncado não pode ser informado

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

Cada figura acima é medida em imagens que o modelo nunca viu no treinamento.

**Precisão** é quantos dos objetos que um modelo relatou são reais;  **recall** é como muitos dos objetos reais que encontrou. Eles falham em direções opostas: má precisão inventa placas, má memória os perde.

**F1** é os dois combinados, e é citado porque cada um é trivialmente disputado -- reporte uma placa inconfundível para precisão quase perfeita, ou cada bolha escura para recall quase perfeito. O que você preferiria perder depende do ensaio, e a contagem é geralmente melhor servida por over-calling: o modelo de placa foi aceito com precisão 0,858 com recall 0,811 ao longo de uma rodada anterior em 0,939 e 0,631.

**IoU**, interseção sobre união, é o quanto um objeto previsto e o real se sobrepõem, dividido pela área que cobrem juntos. É a régua contra a qual o resto é lido, então uma pontuação não significa nada sem seu limite: "F1 0,864 em IoU 0,5" conta um vácuo como encontrado quando os dois contornos concordam com mais da metade de sua área combinada.

**mAP50** e **mAP50-95** pertencem ao detector. O primeiro pergunta se os poços foram encontrados; o segundo repete através de dez limiares de 0,5 a 0,95, por isso também pergunta com que força cada caixa é desenhada. A lacuna entre eles é a colocação, não a detecção.

**Cross-validated**, com um  **SD**, significa que a pontuação é a média de três execuções em divisões diferentes e o SD é o quão longe eles se afastaram. Uma divisão pode ter sorte: a figura da literatura deste modelo é 0,834 em uma única divisão de 19 poços e 0,806 em todos os três.

Os modelos são hospedados na conta Hugging Face do próprio autor, portanto, contribuir não significa entregar o acesso de gravação a qualquer outra pessoa.  ``spacr.model_zoo`` ``publish_model`` executa o upload e imprime a linha de catálogo para adicionar.


Diagnóstico de desempenho
-------------------------

Gere um relatório de hardware e anexe-o a uma issue relacionada ao desempenho::

    python tools/spacr_hardware_report.py

Salva em ``~/.spacr/reports`` e imprime o caminho.  ``--quick`` pula os benchmarks mais longos;  ``--out PATH`` define a localização.

Não lê dados do projeto. Importações de tempos, bibliotecas numéricas, construção de janelas e animação. Relata a emulação processador-arquitetura (uma compilação x86_64  Python no Apple Silicon) e a implementação BLAS do NumPy.

Referência da linha de comando
------------------------------

Cada comando abaixo é instalado por ``pip install spacr``. Todos eles aceitam  ``--help``.

Lançamento do aplicativo
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-server`` pula a triagem de configuração modal, que de outra forma bloquearia uma tarefa autônoma.

``spacr-qt`` e  ``spacr-nightly`` são aliases de ``spacr``.

Quando spaCR não será iniciado
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` imprime uma linha por verificação, com um comando a ser executado para cada falha. Ele também relata que  ``spacr`` está no caminho, que é o que uma antiga sombra de instalação editável.

``safespacr`` lê todas as preferências como padrão e força o pano de fundo, animações, logging verboso e pré-carregamento. Use-o quando uma preferência salva quebrar o lançamento. Ele não muda nada permanentemente.

Módulos em execução sem interface gráfica
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No Qt, no display para clusters, servidores e CI.

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` lê as mesmas configurações que a execução faria e relata o que está faltando, contraditório ou apontando para nada.

``spacr-run --list`` mostra apenas módulos com um ponto de entrada sem interface gráfica; a anotação, a curadoria e a exploração são interativas e omitidas.

Inspecionando uma corrida depois
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cada execução é registrada para ``~/.spacr/runs`` com suas configurações, entradas hashed, saídas, avisos, versões e sementes.

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

Auditoria de dados e instalação
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

Ambiente
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

Os logs rotativos são gravados em ``~/.spacr/logs/spacr.log``. Anexe esse arquivo a um relatório de bug.


Contribuições e suporte
------------------------

Envie relatos de erros e solicitações de recursos bem delimitadas pelo `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Ao relatar uma falha, inclua a versão do spaCR, o sistema operacional, a versão do Python, as configurações do módulo e o trecho relevante do log. O ``spacr-doctor`` coleta a maior parte dessas informações; inclua o relatório de hardware ao relatar problemas de desempenho.

Licença
~~~~~~~~~

spaCR é lançado sob o  `Licença BSD 3-Clause <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

Se spaCR contribuiu para o trabalho publicado, uma citação é apreciada e não é uma condição da licença veja  `Como citar o spaCR`_ abaixo.

Tutoriais
~~~~~~~~~

A `biblioteca interativa de tutoriais do spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contém demonstrações narradas e legendadas da instalação e de cada fluxo de trabalho: 73 lições, com 50 vozes em oito idiomas.

Como citar o spaCR
~~~~~~~~~~~~~~~~~~

Se o spaCR contribuir para sua pesquisa, cite:

Olafsson EB, *et al.* Um pooled image-based  CRISPR screen identifica o EAF1 como um modulador *T. gondii* da subversão ESCRT.

`bioRxiv pré-impressão <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_  · ? `arquivo de software <https://doi.org/10.5281/zenodo.21343316>`_

Agradecimentos
~~~~~~~~~~~~~~~

O spaCR utiliza software científico aberto, incluindo NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch e Qt. Consulte a `atribuição dos modelos de tradução <../TRANSLATION_MODELS.md>`_ para ver os modelos usados na documentação multilíngue e nos catálogos da interface.
