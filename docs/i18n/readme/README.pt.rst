|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
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
   :target: https://einarolafsson.github.io/spacr/
   :alt: Interface Qt
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: Código-fonte no GitHub
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: Problemas no GitHub
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: Licença PolyForm Noncommercial
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: DOI do Zenodo
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Instaladores mais recentes
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Versão no conda-forge

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

spaCR
=====

Idiomas: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

**Análise espacial de fenótipos em triagens CRISPR.**

O spaCR segmenta e mede células individuais em imagens de microscopia de alto conteúdo, integra fenótipos por objeto à abundância de guias derivada do sequenciamento e estima quais genes estão associados a alterações fenotípicas. A partir de imagens de placas e leituras FASTQ, ele produz medições por objeto, classificadores treinados, estimativas de efeito por guia e por gene e uma lista de resultados classificada.

Os módulos de segmentação, medição, anotação e classificação também são executados sem um braço de sequenciamento.

Imagens, máscaras, recortes, medições, anotações, previsões, códigos de barras e identificadores de poço vivem em um projeto SQLite.

Executa como um aplicativo de desktop ou sem interface gráfica em uma estação de trabalho, servidor ou cluster.

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
   :alt: Baixar o spaCR 1.5.0.4 para Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Baixar o spaCR 1.5.0.4 para macOS 11+ (Intel e Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Baixar o spaCR 1.5.0.4 para Linux de 64 bits
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
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

spaCR supports Python **3.9 through 3.14**, except Python 3.14.1, which torchvision excludes. Linux is recommended for the heaviest CUDA and ROCm workflows; macOS and Windows are also supported, and both use their GPUs — macOS through Metal, which covers Apple Silicon and the AMD cards in Intel Macs, and Windows through CUDA or DirectML.

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

Instalar a partir da fonte
~~~~~~~~~~~~~~~~~~~~~~~~~~

Clonar o repositório e instalá-lo no modo editável, para que sua cópia de trabalho *é* o pacote instalado e as edições entrem em vigor sem reinstalar::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

A ramificação padrão é ``nightly``. Para uma versão específica::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

Para puxar alterações posteriores, de dentro do clone::

    git pull
    pip install -e .

A segunda linha só é necessária quando as dependências ou os pontos de entrada são alterados; o código Python é captado sem ele. Se um comando ainda executa o código antigo depois de puxar,  ``spacr-doctor`` relata que ``spacr`` está realmente no seu caminho, que é a causa usual.

Instalar a partir da fonte (luz)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone completo: 427 MB. Clone principal: 76 MB.

::

    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

Ignora os pontos de verificação ``docs/``,  ``tests/``,Cellpose, figuras arquivadas e os catálogos de tradução estendidos. O resultado é um checkout normal.

Opções: ``--dir``,  ``--branch`` (padrão  ``main``), ``--with-tests``,``--with-docs``, -``--with-translations``,-``--no-install``.

``packaging/source_install_excludes.txt`` lista todos os caminhos ignorados.


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

O mesmo projeto também pode ser usado para criar placas, estimar o poder estatístico, corrigir efeitos de lote, inspecionar a qualidade da segmentação, explorar gráficos e recortes vinculados, exportar AnnData, retomar processamentos interrompidos e registrar as configurações associadas a cada resultado.

Módulos do spaCR
----------------

.. spacr-workflow-begin

|Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

|Module_foreign|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

|Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|\ |Module_analyze_plaques|

|Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: Abrir a API de Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
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
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
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
.. |Module_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.0%
   :alt: Abrir a API de Plaque Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: Abrir a API de Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: Abrir a API de Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: Abrir a API de Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle

.. spacr-workflow-end

Cada módulo spaCR é enviado, na ordem em que a tela inicial os lista: os seis módulos do pipeline primeiro, depois tudo o mais. Selecione um bloco para abrir a página  API desse módulo.


Make Masks
~~~~~~~~~~

Make Masks appears under **Tools** for manual correction of segmentation masks; its masthead opens the Cellpose workflows. Nine tools: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** and **Recrop**. Draw makes one filled label from a closed outline, Divide separates a merged object along a drawn line, Recrop turns one object in a crowded field into its own field.

Cellpose-SAM executa aqui mostrar o mapa de probabilidade de célula e o campo de fluxo ao lado da máscara. Veja o  `guia de recursos <../../source/features.rst>`_ para cada ferramenta.

**Outros recursos**

- `Tutoriais interativos <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 guided workflows from installation por meio de hit investigation.
- `Python  API início rápido <../../source/python_api.rst>`_ — run and validate pipelines from scripts, notebooks or a cluster.
- `Guia de funcionalidades <../../source/features.rst>`_ capacidades, maturidade e integrações opcionais.
- `Referência API com curadoria <https://einarolafsson.github.io/spacr/api/index.html>`_ — supported entry points by task, with the complete module reference one level deeper.
- `Guia de idiomas e tradução <../../source/localization.rst>`_ linguagens de interface, ajuda contextual e política de saída científica.

Idioma e tradução
~~~~~~~~~~~~~~~~~~~~~~

A interface oferece dez idiomas na navegação e nas preferências. Os controles AI e LIVE, as descrições dos módulos e a ajuda contextual revisada também são traduzidos. Altere o idioma em **spaCR → Preferências → Idioma** sem reiniciar. Logs, caminhos, valores de banco de dados e medições nunca são traduzidos; a saída científica permanece em inglês canônico. Consulte a `política de ajuda contextual <../../source/localization.rst#contextual-help>`_.

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

spaCR inclui um catálogo de modelos treinados e os obtém sob demanda. Abra **Model Zoo** na tela inicial para navegar por eles e instalá-los, ou indique uma chave num arquivo de configurações -- ``pathogen_model: toxoplasma_pv_v1`` -- e o modelo é baixado e sua soma de verificação conferida na primeira vez que é necessário. Cada entrada publicada carrega um SHA-256; uma entrada sem ele é recusada em vez de instalada, porque um checkpoint truncado ou substituído não pode ser distinguido do verdadeiro.

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

Os números acima são os medidos na publicação, e os limites são declarados com eles: um modelo é útil para o trabalho em que foi medido, não para cada trabalho. ``toxoplasma_well_detector_v1`` e  ``toxoplasma_plaque_v1`` são as duas metades de um fluxo de trabalho - o detector encontra os poços, o segmentador encontra as placas dentro deles, e o diâmetro do poço é o que torna as áreas comparáveis entre os microscópios.

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

Se spaCR contribuiu para o trabalho publicado, uma citação é apreciada e não é uma condição da licença veja  `Citing spaCR`_ abaixo.

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
