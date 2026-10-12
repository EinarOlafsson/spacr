|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

.. |Website| image:: https://img.shields.io/badge/Website-spaCR-0A7BBB
   :target: https://einarolafsson.github.io/projects/spacr/
   :alt: Site do spaCR

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API&logo=readthedocs&logoColor=white
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentação da API

.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Tutoriais interativos

.. |PyPI| image:: https://img.shields.io/pypi/v/spacr?label=PyPI&logo=pypi&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Versão no PyPI

.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 a 3.14

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/nightly?logo=codecov&logoColor=white
   :target: https://app.codecov.io/github/EinarOlafsson/spacr
   :alt: Cobertura de testes no Codecov

.. |Test counts| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FEinarOlafsson%2Fspacr%2Fnightly%2Fdocs%2Fsource%2F_static%2Ftest-counts.json&cacheSeconds=300&logo=pytest&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Suíte de testes

.. |Qt| image:: https://img.shields.io/badge/GUI-Qt-41CD52?logo=qt&logoColor=white
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Interface Qt

.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr?logo=github
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: Problemas no GitHub

.. |License| image:: https://img.shields.io/badge/License-BSD--3-blue?logo=opensourceinitiative&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: Licença BSD 3-Clause

.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: Preprint no bioRxiv

.. |DOI| image:: https://img.shields.io/badge/DOI-blue?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: DOI do Zenodo

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/spacr?label=Conda&logo=anaconda&logoColor=white
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Versão no conda-forge

.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Data da versão mais recente no conda-forge

.. |Docker| image:: https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/pkgs/container/spacr
   :alt: Imagens Docker no GHCR

.. |PyPI Downloads| image:: https://img.shields.io/pepy/dt/spacr?label=downloads&logo=pypi&logoColor=white
   :target: https://pepy.tech/projects/spacr
   :alt: Downloads no PyPI

.. |Platforms| image:: https://img.shields.io/badge/Linux%20OSX%20Win-lightgrey?logo=linux&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS e Windows

.. |Cite| image:: https://img.shields.io/badge/Cite-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: Citar o spaCR

.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=Rank&color=brightgreen&logo=pypi&logoColor=white&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: Classificação de downloads do spaCR no PyPI nos últimos 30 dias completos

.. image:: ../../source/_static/deck/slides/slide_01.jpg
   :alt: spaCR
   :width: 920
   :target: https://einarolafsson.github.io/spacr/_static/deck/

| |intro-1|\ |intro-2|

.. |intro-1| image:: ../../readme_images/pt/intro-1.svg
   :alt: ← Anterior
   :width: 49.5%
   :target: ../../source/_static/deck/pages/57.md

.. |intro-2| image:: ../../readme_images/pt/intro-2.svg
   :alt: Próximo →
   :width: 49.5%
   :target: ../../source/_static/deck/pages/02.md

.. spacr-language-picker-begin

.. image:: ../../readme_images/pt/generated-language-picker.svg
   :alt: Idiomas: Português
   :width: 32.8%
   :target: README.md

.. spacr-language-picker-end

.. image:: ../../readme_images/pt/spacr.svg
   :alt: spaCR. Análise espacial de fenótipos em triagens CRISPR. O spaCR segmenta e mede células individuais em imagens de microscopia, integra fenótipos por objeto à abundância de guias derivada do sequenciamento e estima quais genes estão associados a alterações fenotípicas. A partir de imagens de placas e leituras FASTQ, ele produz medições por objeto, classificadores treinados, estimativas de efeito por guia e por gene e uma lista de resultados classificada. Os módulos de segmentação, medição, anotação e classificação também são executados sem um braço de sequenciamento. Make Masks corrige máscaras de segmentação e anota retângulos independentes com rótulos de classe usando a ferramenta Box para exportação YOLO. Os retângulos mantêm seus próprios rótulos e histórico sem alterar as imagens ou máscaras de origem. Consulte o guia de funcionalidades para conhecer cada ferramenta. Imagens, máscaras e recortes são arquivos em disco. Medições, anotações e previsões do modelo são tabelas do banco de dados SQLite measurements.db do projeto. É executado como um aplicativo de desktop ou sem interface gráfica em uma estação de trabalho, servidor ou cluster.
   :width: 100%
   :target: ../../source/features.rst

.. image:: ../../readme_images/pt/try-spacr.svg
   :alt: Experimente o spaCR.
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. image:: ../../readme_images/pt/try-spacr-2.svg
   :alt: Use Carregar dados de teste… em Import, Make Masks, Annotate ou em uma tela de ensaio para baixar dados de exemplo. No terminal, use spacr-download.
   :width: 100%

.. image:: ../../readme_images/pt/hardware-support.svg
   :alt: Suporte de hardware.
   :width: 100%

.. spacr-hardware-begin

.. image:: ../../readme_images/pt/generated-hardware-1.svg
   :alt: NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU. suportado (estável) implementado (beta) apenas suporte a CPU.
   :width: 100%

.. spacr-hardware-end

.. image:: ../../readme_images/pt/install-spacr.svg
   :alt: Instalar o spaCR. Escolha uma forma de instalação: Aplicativo para desktop: um instalador para Linux, macOS ou Windows com Python próprio. pip: python -m pip install spacr em um ambiente Conda. conda-forge: conda install conda-forge::spacr em um ambiente Conda. Docker: os fluxos de processamento de linha de comando em um contêiner Linux, com a imagem para CPU ou a imagem para GPU NVIDIA:
   :width: 100%

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.4 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.4-cuda12.4 spacr-run --list

.. image:: ../../readme_images/pt/install-spacr-2.svg
   :alt: O guia do Docker explica a configuração da GPU, a montagem de dados e os comandos dos fluxos de processamento.
   :width: 100%
   :target: ../../source/installer_guide.rst#container-images

.. image:: ../../readme_images/pt/desktop-application.svg
   :alt: Aplicativo para desktop. Os instaladores empacotam seus próprios Python. Conda não é necessário.
   :width: 100%

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Baixar o spaCR 1.5.1.4 para Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Baixar o spaCR 1.5.1.4 para macOS 11+ (Intel e Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Baixar o spaCR 1.5.1.4 para Linux de 64 bits
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Linux-x86_64-Online.run

.. spacr-installer-links-end

.. image:: ../../readme_images/pt/desktop-application-2.svg
   :alt: No Linux, torne o arquivo baixado executável e execute-o:
   :width: 100%

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. image:: ../../readme_images/pt/desktop-application-3.svg
   :alt: No macOS, abra o arquivo .pkg. A versão beta atual não é notarizada; se o Gatekeeper a bloquear, selecione Ajustes do Sistema → Privacidade e Segurança → Abrir Mesmo Assim. Consulte o guia de instalação para obter instruções de atualização, desinstalação, uso offline e solução de problemas, e os requisitos do sistema para recomendações sobre estações de trabalho e servidores e tabelas de compatibilidade de GPU.
   :width: 100%

| |desktop-application-3-link-1|\ |desktop-application-3-link-2|

.. |desktop-application-3-link-1| image:: ../../readme_images/pt/desktop-application-3-link-1.svg
   :alt: guia de instalação
   :width: 49.5%
   :target: ../../source/installer_guide.rst

.. |desktop-application-3-link-2| image:: ../../readme_images/pt/desktop-application-3-link-2.svg
   :alt: requisitos do sistema
   :width: 49.5%
   :target: ../../source/system_requirements.rst

.. image:: ../../readme_images/pt/pypi-installation.svg
   :alt: Instalação pelo PyPI. Para usar a versão publicada no PyPI, instale o spaCR com pip em um ambiente Conda. O Python 3.12 oferece a maior variedade de pacotes científicos opcionais:
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. image:: ../../readme_images/pt/pypi-installation-2.svg
   :alt: O spaCR oferece suporte ao Python 3.9 through 3.14, exceto ao Python 3.14.1, que é excluído pelo torchvision. Recomenda-se Linux para os fluxos de trabalho mais pesados com CUDA e ROCm; macOS e Windows também são compatíveis, e ambos utilizam suas GPUs — macOS por meio do Metal, que abrange o Apple Silicon e as placas gráficas AMD dos Macs Intel, e Windows por meio de CUDA ou DirectML. A instalação padrão inclui a interface de desktop Qt. Em um servidor, cluster ou executor de CI, execute os fluxos de trabalho de linha de comando sem abri-la:
   :width: 100%

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. image:: ../../readme_images/pt/pypi-installation-3.svg
   :alt: As integrações opcionais são instaladas separadamente, por exemplo spacr[zarr], spacr[omero],spacr[napari] e spacr[czi,nd2,lif]. Veja a tabela de compatibilidade guia de instalação para os extras completos e ?Python-versão.
   :width: 100%
   :target: ../../source/installer_guide.rst

.. image:: ../../readme_images/pt/conda-forge-installation.svg
   :alt: Instalação com conda-forge. O pacote oficial do conda-forge instala o spaCR e suas dependências de desktop no ambiente ativo:
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. image:: ../../readme_images/pt/install-from-source.svg
   :alt: Instalação a partir do código-fonte. Clone o repositório e instale-o no modo editável, para que sua cópia de trabalho seja o pacote instalado e as alterações passem a valer sem reinstalar:
   :width: 100%

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. image:: ../../readme_images/pt/install-from-source-2.svg
   :alt: Isso clona main, o branch padrão, que contém a versão mais recente. O desenvolvimento acontece em nightly; adicione --branch nightly para clonar esse branch no lugar dele. Para uma versão específica:
   :width: 100%

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. image:: ../../readme_images/pt/install-from-source-3.svg
   :alt: Para obter alterações posteriores, execute de dentro do clone:
   :width: 100%

.. code-block:: bash

   git pull
   pip install -e .

.. image:: ../../readme_images/pt/install-from-source-4.svg
   :alt: Reinstale quando as dependências ou os pontos de entrada mudarem. As alterações no código Python têm efeito direto; spacr-doctor identifica a instalação ativa.
   :width: 100%

.. image:: ../../readme_images/pt/install-from-source-light.svg
   :alt: Instalação a partir do código-fonte (leve). Colaboradores precisam do histórico de versões; para executar o spaCR, escolha uma das opções abaixo. Medições de nightly em 05302fd5c no dia 2026-10-07, feitas com packaging/measure_clone_forms.sh:
   :width: 100%

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

.. image:: ../../readme_images/pt/install-from-source-light-2.svg
   :alt: O clone completo de nightly baixou 9.25 GiB. Adicionar --filter=blob:none ao clone superficial não reduz a cópia de trabalho: seu armazenamento de objetos Git ainda ocupa 2032 MB. O Git não informa as transferências adiadas de objetos, por isso nenhum total de download foi medido. Os arquivos versionados de nightly ocupam 3416 MB na cópia de trabalho (medidos em 2026-10-08), sem o histórico do Git. O tamanho e a duração da transferência variam conforme o ramo.
   :width: 100%

.. image:: ../../readme_images/pt/command-line-entry-points.svg
   :alt: Comandos de linha de comando.
   :width: 100%

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

.. image:: ../../readme_images/pt/command-line-entry-points-2.svg
   :alt: spacr-run --list lista os módulos com pontos de entrada de linha de comando para execução sem interface gráfica. Os módulos de anotação, curadoria, comparação e exploração disponíveis apenas na interface gráfica são omitidos.
   :width: 100%

.. image:: ../../readme_images/pt/core-workflow.svg
   :alt: Fluxo de trabalho principal. O fluxo de trabalho principal compreende seis módulos: Mask segmenta células, núcleos, patógenos e organelas com Cellpose. Measure grava no SQLite características de morfologia, intensidade, textura, espaciais e de colocalização, além de recortes dos objetos. Annotate rotula recortes em uma grade controlada pelo teclado e oferece suporte a filas de aprendizado ativo. Classify treina modelos baseados em imagens ou medições e registra, em cada checkpoint, o desempenho nos dados de validação reservados. Map Barcodes associa as leituras FASTQ aos poços e gRNAs, com controle de qualidade de abundância, colisões e cobertura. Regression estima efeitos de guias, genes, condições e controles com famílias de modelos adequadas a respostas contínuas, fracionárias e de contagem.
   :width: 100%

.. image:: ../../readme_images/pt/spacr-modules.svg
   :alt: Módulos do spaCR.
   :width: 100%

.. spacr-workflow-begin

.. image:: ../../readme_images/pt/generated-workflow-1.svg
   :alt: Principal. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression.
   :width: 100%

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. image:: ../../readme_images/pt/generated-workflow-2.svg
   :alt: Dados. Import images and tables into spaCR projects and execute reproducible multi-plate workflows.
   :width: 100%

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

.. image:: ../../readme_images/pt/generated-workflow-3.svg
   :alt: Ferramentas. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality.
   :width: 100%

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. image:: ../../readme_images/pt/generated-workflow-4.svg
   :alt: Organismo. Análise de imagens específica do organismo e resultados quantitativos de ensaios.
   :width: 100%

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../../spacr/resources/icons/workflow/mask.png" width="16.0%" align="middle" alt="Abrir a API de Mask"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../../spacr/resources/icons/workflow/measure.png" width="16.0%" align="middle" alt="Abrir a API de Measure"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../../spacr/resources/icons/workflow/annotate.png" width="16.0%" align="middle" alt="Abrir a API de Annotate"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../../spacr/resources/icons/workflow/classify_merged.png" width="16.0%" align="middle" alt="Abrir a API de Classify"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../../spacr/resources/icons/workflow/map_barcodes.png" width="16.0%" align="middle" alt="Abrir a API de Map Barcodes"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../../spacr/resources/icons/workflow/regression.png" width="16.0%" align="middle" alt="Abrir a API de Regression"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../../spacr/resources/icons/workflow/apps/foreign.png" width="16.0%" align="middle" alt="Abrir a API de Import"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../../spacr/resources/icons/workflow/apps/embeddings.png" width="16.0%" align="middle" alt="Abrir a API de Embeddings"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../../spacr/resources/icons/workflow/apps/run_compare.png" width="16.0%" align="middle" alt="Abrir a API de Run Compare"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../../spacr/resources/icons/workflow/apps/experiment_design.png" width="16.0%" align="middle" alt="Abrir a API de Experiment Design"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../../spacr/resources/icons/workflow/apps/power.png" width="16.0%" align="middle" alt="Abrir a API de Power / Design"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../../spacr/resources/icons/workflow/apps/dose_response.png" width="16.0%" align="middle" alt="Abrir a API de Dose–Response"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../../spacr/resources/icons/workflow/apps/qc_dashboard.png" width="16.0%" align="middle" alt="Abrir a API de QC"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../../spacr/resources/icons/workflow/apps/make_masks.png" width="16.0%" align="middle" alt="Abrir a API de Make Masks"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../../spacr/resources/icons/workflow/apps/align.png" width="16.0%" align="middle" alt="Abrir a API de Align &amp; Stitch"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../../spacr/resources/icons/workflow/apps/umap.png" width="16.0%" align="middle" alt="Abrir a API de Image UMAP"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../../spacr/resources/icons/workflow/apps/gate_editor.png" width="16.0%" align="middle" alt="Abrir a API de Gate Editor"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../../spacr/resources/icons/workflow/apps/graph_builder.png" width="16.0%" align="middle" alt="Abrir a API de Graph Builder"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../../spacr/resources/icons/workflow/apps/toxoplasma.png" width="16.0%" align="middle" alt="Abrir a API de Toxoplasma"></a>

.. spacr-workflow-end

.. image:: ../../readme_images/pt/spacr-modules-2.svg
   :alt: Todos os módulos com um bloco na tela inicial, na ordem dessa tela: primeiro os seis módulos do fluxo de processamento, depois os demais. Selecione um bloco para abrir a página da API do módulo.
   :width: 100%

.. image:: ../../readme_images/pt/other-resources.svg
   :alt: Outros recursos.
   :width: 100%

.. image:: ../../readme_images/pt/other-resources-2-1.svg
   :alt: Tutoriais interativos — fluxos de trabalho guiados, da instalação à investigação dos resultados positivos.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/tutorials/

.. image:: ../../readme_images/pt/other-resources-2-2.svg
   :alt: Python API início rápido — run and validate pipelines from scripts, notebooks or a cluster.
   :width: 100%
   :target: ../../source/python_api.rst

.. image:: ../../readme_images/pt/other-resources-2-3.svg
   :alt: Guia de funcionalidades capacidades, maturidade e integrações opcionais.
   :width: 100%
   :target: ../../source/features.rst

.. image:: ../../readme_images/pt/other-resources-2-4.svg
   :alt: Referência API com curadoria — supported entry points by task, with the complete module reference one level deeper.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/api/index.html

.. image:: ../../readme_images/pt/other-resources-2-5.svg
   :alt: Guia de idiomas e tradução linguagens de interface, ajuda contextual e política de saída científica.
   :width: 100%
   :target: ../../source/localization.rst

.. image:: ../../readme_images/pt/language-translation.svg
   :alt: Idioma e tradução. A interface oferece dez idiomas na navegação e nas preferências. Os controles AI e LIVE, as descrições dos módulos e a ajuda contextual revisada também são traduzidos. Altere o idioma em spaCR → Preferências → Idioma sem reiniciar. Logs, caminhos, valores de banco de dados e medições nunca são traduzidos; a saída científica permanece em inglês canônico. Consulte a política de ajuda contextual.
   :width: 100%
   :target: ../../source/localization.rst#contextual-help

.. image:: ../../readme_images/pt/language-translation-2.svg
   :alt: Os nove catálogos não ingleses são elaborados por máquina e tecnicamente revisados em vez de lidos de ponta a ponta por um falante nativo. Os registros âmbito de revisão que as línguas tiveram um passe humano e todos os termos deixados em inglês por decisão.
   :width: 100%
   :target: ../REVIEW_SCOPE_2026-09-04.md

.. image:: ../../readme_images/pt/animated-setting-guidance.svg
   :alt: Guia animado de configurações. As configurações com uma explicação visual oferecem um controle Animation na dica de ferramenta. Consulte a galeria de animações de configurações ou o registro de animações de configurações.
   :width: 100%

| |animated-setting-guidance-link-1|\ |animated-setting-guidance-link-2|

.. |animated-setting-guidance-link-1| image:: ../../readme_images/pt/animated-setting-guidance-link-1.svg
   :alt: galeria de animações de configurações
   :width: 49.5%
   :target: https://einarolafsson.github.io/spacr/setting_animations.html

.. |animated-setting-guidance-link-2| image:: ../../readme_images/pt/animated-setting-guidance-link-2.svg
   :alt: registro de animações de configurações
   :width: 49.5%
   :target: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. image:: ../../readme_images/pt/data.svg
   :alt: Dados. Conjuntos de dados de referência.
   :width: 100%

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

.. image:: ../../readme_images/pt/model-zoo.svg
   :alt: Biblioteca de modelos. spaCR envia um catálogo de modelos treinados e os busca sob demanda. Abra Modelo Zoo da tela inicial para navegar e instalá-los, ou nomeie uma chave em um arquivo de configurações - pathogen_model: toxoplasma_pv_v1 - e o modelo é baixado e verificado na primeira vez que é necessário. Cada entrada publicada carrega um SHA-256; uma entrada real sem um é recusada em vez de instalada, porque um truncado não pode ser informado
   :width: 100%

.. spacr-model-zoo-begin

| |zoo_heading|
| |zoo_card_toxoplasma_pv_v4|\ |zoo_card_toxoplasma_plaque_v4|
| |zoo_card_toxoplasma_well_detector_v3|\ |zoo_card_toxoplasma_from_cellmask_v1|
| |zoo_card_live_cell_v1|\ |zoo_card_nuclei_from_cellmask_v1|
| |zoo_card_cell_from_hoechst_v1|\ |zoo_card_toxoplasma_from_hoechst_v1|

.. |zoo_heading| image:: ../../models/cards/heading.svg
   :alt: Default models · held-out scores
   :target: ../../models/README.md
   :width: 99.5%

.. |zoo_card_toxoplasma_pv_v4| image:: ../../models/cards/toxoplasma_pv_v4.svg
   :alt: Score card of Parasite vacuoles v4 (toxoplasma_pv_v4, default): F1 0.761 on 209 held-out fields against 0.377 for stock cpsam_v2. F1 0.761 against stock cpsam_v2's 0.377 on the 209 strict hold-out fields at IoU 0.5; AJI 0.582 against 0.040.
   :target: ../../models/toxoplasma_pv_v4.md
   :width: 49.5%

.. |zoo_card_toxoplasma_plaque_v4| image:: ../../models/cards/toxoplasma_plaque_v4.svg
   :alt: Score card of Plaques v4 (toxoplasma_plaque_v4, default): F1 0.765 on 329 held-out fields against 0.338 for stock cpsam_v2. F1 0.765 against stock cpsam_v2's 0.338 on the 329 strict hold-out fields at IoU 0.5; AJI 0.662 against 0.261.
   :target: ../../models/toxoplasma_plaque_v4.md
   :width: 49.5%

.. |zoo_card_toxoplasma_well_detector_v3| image:: ../../models/cards/toxoplasma_well_detector_v3.svg
   :alt: Score card of Plaque wells v3 (toxoplasma_well_detector_v3, default): mAP50 0.994 on 73 held-out images. Stock was not evaluated in this run; see the named incumbent comparison on the model card.
   :target: ../../models/toxoplasma_well_detector_v3.md
   :width: 49.5%

.. |zoo_card_toxoplasma_from_cellmask_v1| image:: ../../models/cards/toxoplasma_from_cellmask_v1.svg
   :alt: Score card of Vacuoles from cell mask v1 (toxoplasma_from_cellmask_v1, default): F1 0.606 on 463 held-out fields against 0.021 for stock cpsam_v2. F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5.
   :target: ../../models/toxoplasma_from_cellmask_v1.md
   :width: 49.5%

.. |zoo_card_live_cell_v1| image:: ../../models/cards/live_cell_v1.svg
   :alt: Score card of Live cells v1 (live_cell_v1, default): F1 0.694 on 2,199 held-out fields against 0.738 for stock cpsam_v2. On the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell.
   :target: ../../models/live_cell_v1.md
   :width: 49.5%

.. |zoo_card_nuclei_from_cellmask_v1| image:: ../../models/cards/nuclei_from_cellmask_v1.svg
   :alt: Score card of Nuclei from cell mask v1 (nuclei_from_cellmask_v1, default): F1 0.888 on 453 held-out fields against 0.201 for stock cpsam_v2. F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5.
   :target: ../../models/nuclei_from_cellmask_v1.md
   :width: 49.5%

.. |zoo_card_cell_from_hoechst_v1| image:: ../../models/cards/cell_from_hoechst_v1.svg
   :alt: Score card of Cells from Hoechst v1 (cell_from_hoechst_v1, default): F1 0.870 on 451 held-out fields against 0.301 for stock cpsam_v2. F1 0.870 against stock cpsam_v2's 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569.
   :target: ../../models/cell_from_hoechst_v1.md
   :width: 49.5%

.. |zoo_card_toxoplasma_from_hoechst_v1| image:: ../../models/cards/toxoplasma_from_hoechst_v1.svg
   :alt: Score card of Vacuoles from Hoechst v1 (toxoplasma_from_hoechst_v1, default): F1 0.569 on 463 held-out fields against 0.002 for stock cpsam_v2. F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5.
   :target: ../../models/toxoplasma_from_hoechst_v1.md
   :width: 49.5%

.. image:: ../../readme_images/pt/generated-model-zoo-1.svg
   :alt: Modelos anteriores.
   :width: 100%

| |generated-model-zoo-1-link-1|\ |generated-model-zoo-1-link-2|\ |generated-model-zoo-1-link-3|
| |generated-model-zoo-1-link-4|\ |generated-model-zoo-1-link-5|\ |generated-model-zoo-1-link-6|
| |generated-model-zoo-1-link-7|

.. |generated-model-zoo-1-link-1| image:: ../../readme_images/pt/generated-model-zoo-1-link-1.svg
   :alt: toxoplasma_pv_v3
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v3.md

.. |generated-model-zoo-1-link-2| image:: ../../readme_images/pt/generated-model-zoo-1-link-2.svg
   :alt: toxoplasma_pv_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v2.md

.. |generated-model-zoo-1-link-3| image:: ../../readme_images/pt/generated-model-zoo-1-link-3.svg
   :alt: toxoplasma_pv_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v1.md

.. |generated-model-zoo-1-link-4| image:: ../../readme_images/pt/generated-model-zoo-1-link-4.svg
   :alt: toxoplasma_plaque_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v2.md

.. |generated-model-zoo-1-link-5| image:: ../../readme_images/pt/generated-model-zoo-1-link-5.svg
   :alt: toxoplasma_plaque_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v1.md

.. |generated-model-zoo-1-link-6| image:: ../../readme_images/pt/generated-model-zoo-1-link-6.svg
   :alt: toxoplasma_well_detector_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_well_detector_v2.md

.. |generated-model-zoo-1-link-7| image:: ../../readme_images/pt/generated-model-zoo-1-link-7.svg
   :alt: toxoplasma_well_detector_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_well_detector_v1.md

.. image:: ../../readme_images/pt/generated-model-zoo-2.svg
   :alt: Alternativas.
   :width: 100%

| |generated-model-zoo-2-link-1|\ |generated-model-zoo-2-link-2|

.. |generated-model-zoo-2-link-1| image:: ../../readme_images/pt/generated-model-zoo-2-link-1.svg
   :alt: toxoplasma_pv_v5
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v5.md

.. |generated-model-zoo-2-link-2| image:: ../../readme_images/pt/generated-model-zoo-2-link-2.svg
   :alt: toxoplasma_plaque_v3
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v3.md

.. spacr-model-zoo-end

.. image:: ../../readme_images/pt/diagnosing-performance.svg
   :alt: Diagnóstico de desempenho. Gere um relatório de hardware e anexe-o a uma issue relacionada ao desempenho:
   :width: 100%

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. image:: ../../readme_images/pt/diagnosing-performance-2.svg
   :alt: Salva em ~/.spacr/reports e imprime o caminho. --quick pula os benchmarks mais longos; --out PATH define a localização. Não lê dados do projeto. Mede o tempo de importações, bibliotecas numéricas, criação de janelas e animação, e informa a emulação x86_64 no Apple Silicon e a implementação de BLAS usada pelo NumPy.
   :width: 100%

.. image:: ../../readme_images/pt/command-line-reference.svg
   :alt: Referência da linha de comando. Cada comando abaixo é instalado por pip install spacr. Todos eles aceitam --help.
   :width: 100%

.. image:: ../../readme_images/pt/launching-the-application.svg
   :alt: Lançamento do aplicativo.
   :width: 100%

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. image:: ../../readme_images/pt/when-spacr-will-not-start.svg
   :alt: Quando spaCR não será iniciado.
   :width: 100%

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. image:: ../../readme_images/pt/when-spacr-will-not-start-2.svg
   :alt: spacr-doctor imprime uma linha por verificação, com um comando a ser executado para cada falha. Ele também informa qual spacr está no caminho. safespacr lê cada preferência com seu valor padrão e desativa o plano de fundo, as animações, o registro detalhado e o pré-carregamento. Use-o quando uma preferência salva impedir a inicialização. As preferências salvas no modo de segurança são gravadas normalmente e permanecem.
   :width: 100%

.. image:: ../../readme_images/pt/running-modules-headlessly.svg
   :alt: Módulos em execução sem interface gráfica. No Qt, no display para clusters, servidores e CI.
   :width: 100%

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. image:: ../../readme_images/pt/running-modules-headlessly-2.svg
   :alt: validate lê as mesmas configurações que a execução leria e lista cada problema, como uma configuração ausente ou um caminho que não existe, com uma correção.
   :width: 100%

.. image:: ../../readme_images/pt/inspecting-a-run.svg
   :alt: Inspecionar uma execução. Cada execução é registrada para ~/.spacr/runs com suas configurações, entradas hashed, saídas, avisos, versões e sementes.
   :width: 100%

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. image:: ../../readme_images/pt/auditing-data-and-installation.svg
   :alt: Auditoria de dados e instalação.
   :width: 100%

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. image:: ../../readme_images/pt/environment.svg
   :alt: Ambiente.
   :width: 100%

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. image:: ../../readme_images/pt/environment-2.svg
   :alt: Os logs rotativos são gravados em ~/.spacr/logs/spacr.log. Anexe esse arquivo a um relatório de bug.
   :width: 100%

.. image:: ../../readme_images/pt/contributing-and-support.svg
   :alt: Contribuições e suporte. Envie relatos de erros e solicitações de recursos bem delimitadas pelo GitHub Issues. Ao relatar uma falha, inclua a versão do spaCR, o sistema operacional, a versão do Python, as configurações do módulo e o trecho relevante do log. O spacr-doctor coleta a maior parte dessas informações; inclua o relatório de hardware ao relatar problemas de desempenho.
   :width: 100%
   :target: https://github.com/EinarOlafsson/spacr/issues

.. image:: ../../readme_images/pt/licensing.svg
   :alt: Licença. spaCR é lançado sob o Licença BSD 3-Clause. Se spaCR contribuiu para o trabalho publicado, uma citação é apreciada e não é uma condição da licença veja Como citar o spaCR abaixo.
   :width: 100%
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE

.. image:: ../../readme_images/pt/tutorials.svg
   :alt: Tutoriais. A biblioteca de tutoriais interativos do spaCR oferece guias de instalação e uso dos módulos. Cada aula indica a narração e os idiomas disponíveis.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/tutorials/

.. image:: ../../readme_images/pt/citing-spacr.svg
   :alt: Como citar o spaCR. Se o spaCR contribuir para sua pesquisa, cite: Olafsson EB, et al. Um pooled image-based CRISPR screen identifica o EAF1 como um modulador T. gondii da subversão ESCRT.
   :width: 100%

| |citing-spacr-2-1|\ |citing-spacr-2-2|

.. |citing-spacr-2-1| image:: ../../readme_images/pt/citing-spacr-2-1.svg
   :alt: bioRxiv pré-impressão
   :width: 49.5%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

.. |citing-spacr-2-2| image:: ../../readme_images/pt/citing-spacr-2-2.svg
   :alt: arquivo de software
   :width: 49.5%
   :target: https://doi.org/10.5281/zenodo.21343316

.. image:: ../../readme_images/pt/other-work-citing-spacr.svg
   :alt: Outros trabalhos citando spaCR.
   :width: 100%

.. spacr-citing-papers-begin

.. image:: ../../readme_images/pt/other-work-citing-spacr-2-1.svg
   :alt: Adaptação metabólica e eliminação de nutrientes em Toxoplasma gondii: insights de mutantes deficientes em vias de ingestão.
   :width: 100%
   :target: https://journals.asm.org/doi/full/10.1128/msphere.01011-24

.. image:: ../../readme_images/pt/other-work-citing-spacr-2-2.svg
   :alt: O IRE1 promove o fluxo de cálcio fagossomal para aumentar a atividade fungicida dos macrófagos.
   :width: 100%
   :target: https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00465-6

.. image:: ../../readme_images/pt/other-work-citing-spacr-2-3.svg
   :alt: O Toxoplasma GRA8 envolve a proteína acessória do hospedeiro ESCRT ALG-2 e é necessário para a integridade metabólica do parasita.
   :width: 100%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.20.739547v1.abstract

.. image:: ../../readme_images/pt/other-work-citing-spacr-2-4.svg
   :alt: Hydrogen sulfide dynamically upregulates copper uptake and localization.
   :width: 100%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.30.741779v1

.. image:: ../../readme_images/pt/other-work-citing-spacr-2-5.svg
   :alt: spaCR: Análise do fenótipo espacial das triagens CRISPR-Cas9 (versão de pré-impressão 1).
   :width: 100%
   :target: https://www.researchsquare.com/article/rs-7368254/v1

.. spacr-citing-papers-end

.. image:: ../../readme_images/pt/acknowledgments.svg
   :alt: Agradecimentos. O spaCR utiliza software científico aberto, incluindo NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch e Qt. Consulte a atribuição dos modelos de tradução para ver os modelos usados na documentação multilíngue e nos catálogos da interface.
   :width: 100%
   :target: ../TRANSLATION_MODELS.md
