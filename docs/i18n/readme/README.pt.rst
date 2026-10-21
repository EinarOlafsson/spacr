|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

Site: `einarolafsson.github.io/projects/spacr <https://einarolafsson.github.io/projects/spacr/>`_

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

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/main?logo=codecov&logoColor=white
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

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/_static/deck/"><picture><source media="(min-width: 1280px)" srcset="../../readme_assets/cover.desktop.svg"><img src="../../readme_assets/cover.svg" alt="spaCR"></picture></a></div>

.. raw:: html

   <div><a href="../../source/_static/deck/pages/57.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/intro-1.desktop.svg"><img src="../../readme_images/pt/intro-1.svg" alt="← Anterior"></picture></a><a href="../../source/_static/deck/pages/02.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/intro-2.desktop.svg"><img src="../../readme_images/pt/intro-2.svg" alt="Próximo →"></picture></a></div>

.. spacr-language-picker-begin

.. raw:: html

   <div><a href="README.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-language-picker.desktop.svg"><img src="../../readme_images/pt/generated-language-picker.svg" alt="Idiomas: Português"></picture></a></div>

.. spacr-language-picker-end

.. raw:: html

   <div><a href="../../readme_text/pt/spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/spacr.desktop.svg"><img src="../../readme_images/pt/spacr.svg" alt="spaCR. Análise espacial de fenótipos em triagens CRISPR. O spaCR segmenta e mede células individuais em imagens de microscopia, integra fenótipos por objeto à abundância de guias derivada do sequenciamento e estima quais genes estão associados a alterações fenotípicas. A partir de imagens de placas e leituras FASTQ, ele produz medições por objeto, classificadores treinados, estimativas de efeito por guia e por gene e uma lista de resultados classificada. Os módulos de segmentação, medição, anotação e classificação também podem ser usados sem um componente de sequenciamento. Make Masks corrige máscaras de segmentação e anota retângulos independentes com rótulos de classe usando a ferramenta Box para exportação YOLO. Os retângulos mantêm seus próprios rótulos e histórico sem alterar as imagens ou máscaras de origem. Consulte o guia de funcionalidades para conhecer cada ferramenta. Imagens, máscaras e recortes são arquivos em disco. Medições, anotações e previsões do modelo são tabelas do banco de dados SQLite measurements.db do projeto. É executado como um aplicativo de desktop ou sem interface gráfica em uma estação de trabalho, servidor ou cluster."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/try-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/try-spacr.desktop.svg"><img src="../../readme_images/pt/try-spacr.svg" alt="Experimente o spaCR."></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/pt/hardware-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/hardware-support.desktop.svg"><img src="../../readme_images/pt/hardware-support.svg" alt="Suporte de hardware."></picture></a></div>

.. spacr-hardware-begin

.. raw:: html

   <div><a href="../../readme_text/pt/generated-hardware-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-hardware-1.desktop.svg"><img src="../../readme_images/pt/generated-hardware-1.svg" alt="NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU, 3D GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU, 3D CPU. suportado (estável) implementado (beta) apenas suporte a CPU."></picture></a></div>

.. spacr-hardware-end

.. raw:: html

   <div><a href="../../readme_text/pt/install-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/install-spacr.desktop.svg"><img src="../../readme_images/pt/install-spacr.svg" alt="Instalar o spaCR. Escolha uma forma de instalação: Aplicativo para desktop: um instalador para Linux, macOS ou Windows com Python próprio. pip: python -m pip install spacr em um ambiente Conda. conda-forge: conda install conda-forge::spacr em um ambiente Conda. Docker: os fluxos de processamento de linha de comando em um contêiner Linux, com a imagem para CPU ou a imagem para GPU NVIDIA:"></picture></a></div>

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.5 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.5-cuda12.4 spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/pt/install-spacr-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/install-spacr-2.desktop.svg"><img src="../../readme_images/pt/install-spacr-2.svg" alt="O guia do Docker (nightly, em inglês) explica a configuração da GPU, a montagem de dados e os comandos dos fluxos de processamento."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/desktop-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/desktop-application.desktop.svg"><img src="../../readme_images/pt/desktop-application.svg" alt="Aplicativo para desktop. Os instaladores incluem seu próprio ambiente Python. Não é necessário usar Conda."></picture></a></div>

.. spacr-installer-links-begin

| |InstallerLinux|\ |InstallerMacOS|\ |InstallerWindows|

.. |InstallerWindows| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-Windows-Online-Setup.exe"><img src="../../readme_assets/cells/platforms/windows.png" width="14.14%" align="middle" alt="Baixar o spaCR 1.5.1.5 para Windows 10/11"></a>
.. |InstallerMacOS| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-macOS-Universal-Online.pkg"><img src="../../readme_assets/cells/platforms/macos.png" width="14.14%" align="middle" alt="Baixar o spaCR 1.5.1.5 para macOS 11+ (Intel e Apple Silicon)"></a>
.. |InstallerLinux| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-Linux-x86_64-Online.run"><img src="../../readme_assets/cells/platforms/linux.png" width="14.14%" align="middle" alt="Baixar o spaCR 1.5.1.5 para Linux de 64 bits"></a>

.. spacr-installer-links-end

.. raw:: html

   <div><a href="../../readme_text/pt/desktop-application-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/desktop-application-2.desktop.svg"><img src="../../readme_images/pt/desktop-application-2.svg" alt="No Linux, torne o arquivo baixado executável e execute-o:"></picture></a></div>

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. raw:: html

   <div><a href="../../readme_text/pt/desktop-application-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/desktop-application-3.desktop.svg"><img src="../../readme_images/pt/desktop-application-3.svg" alt="No macOS, abra o arquivo .pkg. A versão beta atual não é notarizada; se o Gatekeeper a bloquear, selecione Ajustes do Sistema → Privacidade e Segurança → Abrir Mesmo Assim. Para instruções de instalação, atualização, reinstalação, desinstalação, uso offline e solução de problemas, consulte o guia de instalação atual do nightly (em inglês). Consulte os requisitos do sistema para recomendações sobre estações de trabalho e servidores e tabelas de compatibilidade de GPU."></picture></a></div>

.. raw:: html

   <div><a href="https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installer_guide.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/desktop-application-3-link-1.desktop.svg"><img src="../../readme_images/pt/desktop-application-3-link-1.svg" alt="guia de instalação atual do nightly (em inglês)"></picture></a><a href="../../source/system_requirements.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/desktop-application-3-link-2.desktop.svg"><img src="../../readme_images/pt/desktop-application-3-link-2.svg" alt="requisitos do sistema"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/pypi-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/pypi-installation.desktop.svg"><img src="../../readme_images/pt/pypi-installation.svg" alt="Instalação pelo PyPI. Para usar a versão publicada no PyPI, instale o spaCR com pip em um ambiente Conda. O Python 3.12 oferece a maior variedade de pacotes científicos opcionais:"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/pt/pypi-installation-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/pypi-installation-2.desktop.svg"><img src="../../readme_images/pt/pypi-installation-2.svg" alt="O spaCR oferece suporte ao Python 3.9 through 3.14, exceto ao Python 3.14.1, que é excluído pelo torchvision. Recomenda-se Linux para os fluxos de trabalho mais pesados com CUDA e ROCm; macOS e Windows também são compatíveis, e ambos utilizam suas GPUs — macOS por meio do Metal, que abrange o Apple Silicon e as placas gráficas AMD dos Macs Intel, e Windows por meio de CUDA ou DirectML. A instalação padrão inclui a interface de desktop Qt. Em um servidor, cluster ou executor de CI, execute os fluxos de trabalho de linha de comando sem abri-la:"></picture></a></div>

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/pt/pypi-installation-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/pypi-installation-3.desktop.svg"><img src="../../readme_images/pt/pypi-installation-3.svg" alt="As integrações opcionais são instaladas separadamente, por exemplo spacr[zarr], spacr[omero], spacr[napari] e spacr[czi,nd2,lif]. Consulte o guia de instalação (nightly, em inglês) para ver a lista completa de extras e a tabela de compatibilidade das versões do Python."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/conda-forge-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/conda-forge-installation.desktop.svg"><img src="../../readme_images/pt/conda-forge-installation.svg" alt="Instalação com conda-forge. O pacote oficial do conda-forge instala o spaCR e suas dependências de desktop no ambiente ativo:"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/pt/install-from-source.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/install-from-source.desktop.svg"><img src="../../readme_images/pt/install-from-source.svg" alt="Instalação a partir do código-fonte. Clone o repositório e instale-o no modo editável, para que sua cópia de trabalho seja o pacote instalado e as alterações passem a valer sem reinstalar:"></picture></a></div>

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. raw:: html

   <div><a href="../../readme_text/pt/install-from-source-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/install-from-source-2.desktop.svg"><img src="../../readme_images/pt/install-from-source-2.svg" alt="Isso clona main, o branch padrão, que contém a versão mais recente. O desenvolvimento acontece em nightly; adicione --branch nightly para clonar esse branch no lugar dele. Para uma versão específica:"></picture></a></div>

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. raw:: html

   <div><a href="../../readme_text/pt/install-from-source-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/install-from-source-3.desktop.svg"><img src="../../readme_images/pt/install-from-source-3.svg" alt="Para obter alterações posteriores, execute de dentro do clone:"></picture></a></div>

.. code-block:: bash

   git pull
   pip install -e .

.. raw:: html

   <div><a href="../../readme_text/pt/install-from-source-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/install-from-source-4.desktop.svg"><img src="../../readme_images/pt/install-from-source-4.svg" alt="Reinstale quando as dependências ou os pontos de entrada mudarem. As alterações no código Python têm efeito direto; spacr-doctor identifica a instalação ativa."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/install-from-source-light.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/install-from-source-light.desktop.svg"><img src="../../readme_images/pt/install-from-source-light.svg" alt="Instalação a partir do código-fonte (leve). Colaboradores precisam do histórico de versões; para executar o spaCR, escolha uma das opções abaixo. Medições de nightly em 05302fd5c no dia 2026-10-07, feitas com packaging/measure_clone_forms.sh. Clone superficial: 2048 MB baixados em 140 s. Apenas o commit mais recente é mantido; assim, o histórico anterior não está disponível para git log, git blame e git bisect. git pull funciona, mas o clone continua superficial; git fetch --unshallow obtém o histórico completo."></picture></a></div>

.. code-block:: bash

   git clone --depth 1 --branch nightly https://github.com/EinarOlafsson/spacr.git
   cd spacr && pip install -e .

.. raw:: html

   <div><a href="../../readme_text/pt/install-from-source-light-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/install-from-source-light-2.desktop.svg"><img src="../../readme_images/pt/install-from-source-light-2.svg" alt="Somente arquivos de execução: 104 MB em disco em 6 s (Git: 38 MB; arquivos: 66 MB). São omitidos o histórico, docs, tests, tools, features, os dados de exemplo e as traduções ampliadas da interface. --with-docs, --with-tests e --with-translations mantêm a documentação, os testes e as traduções correspondentes. As traduções compactas da interface continuam disponíveis."></picture></a></div>

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
   sh install_spacr.sh --branch nightly

.. raw:: html

   <div><a href="../../readme_text/pt/install-from-source-light-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/install-from-source-light-3.desktop.svg"><img src="../../readme_images/pt/install-from-source-light-3.svg" alt="--dir escolhe o caminho do clone; --branch seleciona uma ramificação ou tag; --no-install baixa sem instalar; --help mostra as opções. packaging/source_install_excludes.txt lista todos os caminhos omitidos. O clone completo de nightly baixou 9.25 GiB. Adicionar --filter=blob:none ao clone superficial não reduz a cópia de trabalho: seu armazenamento de objetos Git ainda ocupa 2032 MB. O Git não informa as transferências adiadas de objetos, por isso nenhum total de download foi medido. Os arquivos versionados de nightly ocupam 720 MB na cópia de trabalho (medidos em 2026-10-21), sem o histórico do Git. O tamanho e a duração da transferência variam conforme o ramo."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/command-line-entry-points.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/command-line-entry-points.desktop.svg"><img src="../../readme_images/pt/command-line-entry-points.svg" alt="Comandos de linha de comando."></picture></a></div>

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

   <div><a href="../../readme_text/pt/command-line-entry-points-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/command-line-entry-points-2.desktop.svg"><img src="../../readme_images/pt/command-line-entry-points-2.svg" alt="spacr-run --list lista os módulos com pontos de entrada de linha de comando para execução sem interface gráfica. Os módulos de anotação, curadoria, comparação e exploração disponíveis apenas na interface gráfica são omitidos."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/core-workflow.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/core-workflow.desktop.svg"><img src="../../readme_images/pt/core-workflow.svg" alt="Fluxo de trabalho principal. O fluxo de trabalho principal compreende seis módulos: Mask segmenta células, núcleos, patógenos e organelas com Cellpose. Measure grava no SQLite características de morfologia, intensidade, textura, espaciais e de colocalização, além de recortes dos objetos. Annotate rotula recortes em uma grade controlada pelo teclado e oferece suporte a filas de aprendizado ativo. Classify treina modelos baseados em imagens ou medições e registra, em cada checkpoint, o desempenho nos dados de validação reservados. Map Barcodes associa as leituras FASTQ aos poços e gRNAs, com controle de qualidade de abundância, colisões e cobertura. Regression estima efeitos de guias, genes, condições e controles com famílias de modelos adequadas a respostas contínuas, fracionárias e de contagem."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/spacr-modules.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/spacr-modules.desktop.svg"><img src="../../readme_images/pt/spacr-modules.svg" alt="Módulos do spaCR."></picture></a></div>

.. spacr-workflow-begin

.. raw:: html

   <div><a href="../../readme_text/pt/generated-workflow-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-workflow-1.desktop.svg"><img src="../../readme_images/pt/generated-workflow-1.svg" alt="Principal. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression."></picture></a></div>

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. raw:: html

   <div><a href="../../readme_text/pt/generated-workflow-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-workflow-2.desktop.svg"><img src="../../readme_images/pt/generated-workflow-2.svg" alt="Dados. Import images and tables into spaCR projects and execute reproducible multi-plate workflows."></picture></a></div>

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

.. raw:: html

   <div><a href="../../readme_text/pt/generated-workflow-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-workflow-3.desktop.svg"><img src="../../readme_images/pt/generated-workflow-3.svg" alt="Ferramentas. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality."></picture></a></div>

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. raw:: html

   <div><a href="../../readme_text/pt/generated-workflow-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-workflow-4.desktop.svg"><img src="../../readme_images/pt/generated-workflow-4.svg" alt="Organismo. Análise de imagens específica do organismo e resultados quantitativos de ensaios."></picture></a></div>

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../readme_assets/cells/workflow/mask.png" width="14.14%" align="middle" alt="Abrir a API de Mask"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../readme_assets/cells/workflow/measure.png" width="14.14%" align="middle" alt="Abrir a API de Measure"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../readme_assets/cells/workflow/annotate.png" width="14.14%" align="middle" alt="Abrir a API de Annotate"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../readme_assets/cells/workflow/classify_merged.png" width="14.14%" align="middle" alt="Abrir a API de Classify"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../readme_assets/cells/workflow/map_barcodes.png" width="14.14%" align="middle" alt="Abrir a API de Map Barcodes"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../readme_assets/cells/workflow/regression.png" width="14.14%" align="middle" alt="Abrir a API de Regression"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../readme_assets/cells/workflow/apps/foreign.png" width="14.14%" align="middle" alt="Abrir a API de Import"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../readme_assets/cells/workflow/apps/embeddings.png" width="14.14%" align="middle" alt="Abrir a API de Embeddings"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../readme_assets/cells/workflow/apps/run_compare.png" width="14.14%" align="middle" alt="Abrir a API de Run Compare"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../readme_assets/cells/workflow/apps/experiment_design.png" width="14.14%" align="middle" alt="Abrir a API de Experiment Design"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../readme_assets/cells/workflow/apps/power.png" width="14.14%" align="middle" alt="Abrir a API de Power / Design"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../readme_assets/cells/workflow/apps/dose_response.png" width="14.14%" align="middle" alt="Abrir a API de Dose–Response"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../readme_assets/cells/workflow/apps/qc_dashboard.png" width="14.14%" align="middle" alt="Abrir a API de QC"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../readme_assets/cells/workflow/apps/make_masks.png" width="14.14%" align="middle" alt="Abrir a API de Make Masks"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../readme_assets/cells/workflow/apps/align.png" width="14.14%" align="middle" alt="Abrir a API de Align &amp; Stitch"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../readme_assets/cells/workflow/apps/umap.png" width="14.14%" align="middle" alt="Abrir a API de Image UMAP"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../readme_assets/cells/workflow/apps/gate_editor.png" width="14.14%" align="middle" alt="Abrir a API de Gate Editor"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../readme_assets/cells/workflow/apps/graph_builder.png" width="14.14%" align="middle" alt="Abrir a API de Graph Builder"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../readme_assets/cells/workflow/apps/toxoplasma.png" width="14.14%" align="middle" alt="Abrir a API de Toxoplasma"></a>

.. spacr-workflow-end

.. raw:: html

   <div><a href="../../readme_text/pt/spacr-modules-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/spacr-modules-2.desktop.svg"><img src="../../readme_images/pt/spacr-modules-2.svg" alt="Todos os módulos com um bloco na tela inicial, na ordem dessa tela: primeiro os seis módulos do fluxo de processamento, depois os demais. Selecione um bloco para abrir a página da API do módulo."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/other-resources.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-resources.desktop.svg"><img src="../../readme_images/pt/other-resources.svg" alt="Outros recursos."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/other-resources-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-resources-2-1.desktop.svg"><img src="../../readme_images/pt/other-resources-2-1.svg" alt="Tutoriais interativos — fluxos de trabalho guiados, da instalação à investigação dos resultados positivos."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/other-resources-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-resources-2-2.desktop.svg"><img src="../../readme_images/pt/other-resources-2-2.svg" alt="Início rápido da API Python — execute e valide fluxos de processamento a partir de scripts, notebooks ou de um cluster."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/other-resources-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-resources-2-3.desktop.svg"><img src="../../readme_images/pt/other-resources-2-3.svg" alt="Guia de funcionalidades — recursos, maturidade e integrações opcionais."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/other-resources-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-resources-2-4.desktop.svg"><img src="../../readme_images/pt/other-resources-2-4.svg" alt="Referência selecionada da API — pontos de entrada compatíveis organizados por tarefa, com a referência completa dos módulos um nível abaixo."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/other-resources-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-resources-2-5.desktop.svg"><img src="../../readme_images/pt/other-resources-2-5.svg" alt="Guia de idiomas e tradução — idiomas da interface, ajuda contextual e política de resultados científicos."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/language-translation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/language-translation.desktop.svg"><img src="../../readme_images/pt/language-translation.svg" alt="Idioma e tradução. A interface oferece dez idiomas na navegação e nas preferências. Os controles AI e LIVE, as descrições dos módulos e a ajuda contextual revisada também são traduzidos. Altere o idioma em spaCR → Preferências → Idioma sem reiniciar. Logs, caminhos, valores de banco de dados e medições nunca são traduzidos; a saída científica permanece em inglês canônico. Consulte a política de ajuda contextual."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/animated-setting-guidance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/animated-setting-guidance.desktop.svg"><img src="../../readme_images/pt/animated-setting-guidance.svg" alt="Guia animado de configurações. As configurações com uma explicação visual oferecem um controle Animation na dica de ferramenta. Consulte a galeria de animações de configurações ou o registro de animações de configurações."></picture></a></div>

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/setting_animations.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/animated-setting-guidance-link-1.desktop.svg"><img src="../../readme_images/pt/animated-setting-guidance-link-1.svg" alt="galeria de animações de configurações"></picture></a><a href="https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/animated-setting-guidance-link-2.desktop.svg"><img src="../../readme_images/pt/animated-setting-guidance-link-2.svg" alt="registro de animações de configurações"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/data.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/data.desktop.svg"><img src="../../readme_images/pt/data.svg" alt="Dados. Conjuntos de dados de referência."></picture></a></div>

| |DataBioStudies|\ |DataHuggingFace|\ |DataNCBI|\ |DataSpaCRPower|\ |DataBioRxiv|

.. |DataBioStudies| raw:: html

   <a href="https://doi.org/10.6019/S-BIAD2135"><img src="../../readme_assets/cells/databanks/biostudies_button.png" width="14.14%" align="middle" alt="Abrir o conjunto de microscopia no BioStudies"></a>

.. |DataHuggingFace| raw:: html

   <a href="https://huggingface.co/datasets/einarolafsson/toxo_mito"><img src="../../readme_assets/cells/databanks/huggingface_button.png" width="14.14%" align="middle" alt="Abrir o conjunto de teste no Hugging Face"></a>

.. |DataNCBI| raw:: html

   <a href="https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935"><img src="../../readme_assets/cells/databanks/ncbi_button.png" width="14.14%" align="middle" alt="Abrir o conjunto de sequenciamento no NCBI"></a>

.. |DataSpaCRPower| raw:: html

   <a href="https://github.com/maomlab/spaCRPower"><img src="../../readme_assets/cells/databanks/spacrpower_button.png" width="14.14%" align="middle" alt="Abrir o spaCRPower"></a>

.. |DataBioRxiv| raw:: html

   <a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><img src="../../readme_assets/cells/databanks/biorxiv_button.png" width="14.14%" align="middle" alt="Abrir a pré-publicação no bioRxiv"></a>

.. raw:: html

   <div><a href="../../readme_text/pt/model-zoo.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/model-zoo.desktop.svg"><img src="../../readme_images/pt/model-zoo.svg" alt="Biblioteca de modelos. O spaCR inclui um catálogo de modelos treinados e os baixa quando necessário. Abra Model Zoo na tela inicial para explorar e instalar modelos, ou indique uma chave no arquivo de configurações, como pathogen_model: toxoplasma_pv_v1. O modelo é baixado e verificado pela soma de verificação na primeira vez em que é necessário. Cada entrada publicada contém uma soma SHA-256. Entradas sem essa soma são recusadas, pois, sem ela, não seria possível distinguir um arquivo de modelo incompleto ou substituído do original."></picture></a></div>

.. spacr-model-zoo-begin

.. raw:: html

   <div><a href="../../models/README.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/heading.svg"><img src="../../models/cards/heading.phone.svg" alt="Default models · held-out scores"></picture></a></div>
   <div><a href="../../models/toxoplasma_pv_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_pv_v4.svg"><img src="../../models/cards/toxoplasma_pv_v4.phone.svg" alt="Score card of Parasite vacuoles v4 (toxoplasma_pv_v4, default): F1 0.761 on 209 held-out fields against 0.377 for stock cpsam_v2. F1 0.761 against stock cpsam_v2&#x27;s 0.377 on the 209 strict hold-out fields at IoU 0.5; AJI 0.582 against 0.040."></picture></a><a href="../../models/toxoplasma_plaque_v5.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_plaque_v5.svg"><img src="../../models/cards/toxoplasma_plaque_v5.phone.svg" alt="Score card of Plaques v5 (toxoplasma_plaque_v5, default): F1 0.816 on 153 held-out fields against 0.227 for stock cpsam_v2. F1 0.816 against stock cpsam_v2&#x27;s 0.227 on the 153 strict hold-out fields at IoU 0.5; AJI 0.737 against 0.178."></picture></a><a href="../../models/toxoplasma_well_detector_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_well_detector_v3.svg"><img src="../../models/cards/toxoplasma_well_detector_v3.phone.svg" alt="Score card of Plaque wells v3 (toxoplasma_well_detector_v3, default): mAP50 0.994 on 73 held-out images. Stock was not evaluated in this run; see the named incumbent comparison on the model card."></picture></a><a href="../../models/toxoplasma_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_cellmask_v1.svg"><img src="../../models/cards/toxoplasma_from_cellmask_v1.phone.svg" alt="Score card of Vacuoles from cell mask v1 (toxoplasma_from_cellmask_v1, default): F1 0.606 on 463 held-out fields against 0.021 for stock cpsam_v2. F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/nuclei_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/nuclei_from_cellmask_v1.svg"><img src="../../models/cards/nuclei_from_cellmask_v1.phone.svg" alt="Score card of Nuclei from cell mask v1 (nuclei_from_cellmask_v1, default): F1 0.888 on 453 held-out fields against 0.201 for stock cpsam_v2. F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/cell_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/cell_from_hoechst_v1.svg"><img src="../../models/cards/cell_from_hoechst_v1.phone.svg" alt="Score card of Cells from Hoechst v1 (cell_from_hoechst_v1, default): F1 0.870 on 451 held-out fields against 0.301 for stock cpsam_v2. F1 0.870 against stock cpsam_v2&#x27;s 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569."></picture></a><a href="../../models/toxoplasma_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_hoechst_v1.svg"><img src="../../models/cards/toxoplasma_from_hoechst_v1.phone.svg" alt="Score card of Vacuoles from Hoechst v1 (toxoplasma_from_hoechst_v1, default): F1 0.569 on 463 held-out fields against 0.002 for stock cpsam_v2. F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/generated-model-zoo-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-1.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-1.svg" alt="Modelos anteriores."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-1-link-1.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-1-link-1.svg" alt="toxoplasma_pv_v3"></picture></a><a href="../../models/toxoplasma_pv_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-1-link-2.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-1-link-2.svg" alt="toxoplasma_pv_v2"></picture></a><a href="../../models/toxoplasma_pv_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-1-link-3.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-1-link-3.svg" alt="toxoplasma_pv_v1"></picture></a><a href="../../models/toxoplasma_plaque_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-1-link-4.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-1-link-4.svg" alt="toxoplasma_plaque_v4"></picture></a><a href="../../models/toxoplasma_plaque_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-1-link-5.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-1-link-5.svg" alt="toxoplasma_plaque_v2"></picture></a><a href="../../models/toxoplasma_plaque_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-1-link-6.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-1-link-6.svg" alt="toxoplasma_plaque_v1"></picture></a><a href="../../models/toxoplasma_well_detector_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-1-link-7.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-1-link-7.svg" alt="toxoplasma_well_detector_v2"></picture></a><a href="../../models/toxoplasma_well_detector_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-1-link-8.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-1-link-8.svg" alt="toxoplasma_well_detector_v1"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/generated-model-zoo-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-2.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-2.svg" alt="Alternativas."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-2-link-1.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-2-link-1.svg" alt="toxoplasma_pv_v5"></picture></a><a href="../../models/toxoplasma_plaque_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-2-link-2.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-2-link-2.svg" alt="toxoplasma_plaque_v3"></picture></a><a href="../../models/live_cell_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-2-link-3.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-2-link-3.svg" alt="live_cell_v1"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/generated-model-zoo-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/generated-model-zoo-3.desktop.svg"><img src="../../readme_images/pt/generated-model-zoo-3.svg" alt="Gerador de máscaras"></picture></a></div>

.. spacr-model-zoo-end

.. raw:: html

   <div><a href="../../readme_text/pt/diagnosing-performance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/diagnosing-performance.desktop.svg"><img src="../../readme_images/pt/diagnosing-performance.svg" alt="Diagnóstico de desempenho. Gere um relatório de hardware e anexe-o a uma issue relacionada ao desempenho:"></picture></a></div>

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. raw:: html

   <div><a href="../../readme_text/pt/diagnosing-performance-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/diagnosing-performance-2.desktop.svg"><img src="../../readme_images/pt/diagnosing-performance-2.svg" alt="Salva em ~/.spacr/reports e mostra o caminho. --quick ignora os testes de desempenho mais longos; --out PATH define o local. Não lê dados do projeto. Mede o tempo de importações, bibliotecas numéricas, criação de janelas e animação, e informa a emulação x86_64 no Apple Silicon e a implementação de BLAS usada pelo NumPy."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/command-line-reference.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/command-line-reference.desktop.svg"><img src="../../readme_images/pt/command-line-reference.svg" alt="Referência da linha de comando. Todos os comandos abaixo são instalados por pip install spacr e aceitam --help."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/launching-the-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/launching-the-application.desktop.svg"><img src="../../readme_images/pt/launching-the-application.svg" alt="Lançamento do aplicativo."></picture></a></div>

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. raw:: html

   <div><a href="../../readme_text/pt/when-spacr-will-not-start.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/when-spacr-will-not-start.desktop.svg"><img src="../../readme_images/pt/when-spacr-will-not-start.svg" alt="Quando o spaCR não inicia."></picture></a></div>

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. raw:: html

   <div><a href="../../readme_text/pt/when-spacr-will-not-start-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/when-spacr-will-not-start-2.desktop.svg"><img src="../../readme_images/pt/when-spacr-will-not-start-2.svg" alt="spacr-doctor imprime uma linha por verificação, com um comando a ser executado para cada falha. Ele também informa qual spacr está no caminho. safespacr lê cada preferência com seu valor padrão e desativa o plano de fundo, as animações, o registro detalhado e o pré-carregamento. Use-o quando uma preferência salva impedir a inicialização. As preferências salvas no modo de segurança são gravadas normalmente e permanecem."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/running-modules-headlessly.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/running-modules-headlessly.desktop.svg"><img src="../../readme_images/pt/running-modules-headlessly.svg" alt="Executar módulos sem interface gráfica. Sem Qt nem tela — para clusters, servidores e CI."></picture></a></div>

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. raw:: html

   <div><a href="../../readme_text/pt/running-modules-headlessly-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/running-modules-headlessly-2.desktop.svg"><img src="../../readme_images/pt/running-modules-headlessly-2.svg" alt="validate lê as mesmas configurações que a execução leria e lista cada problema, como uma configuração ausente ou um caminho que não existe, com uma correção."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/inspecting-a-run.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/inspecting-a-run.desktop.svg"><img src="../../readme_images/pt/inspecting-a-run.svg" alt="Inspecionar uma execução. Cada execução é registrada em ~/.spacr/runs com suas configurações, hashes das entradas, saídas, avisos, versões e sementes aleatórias."></picture></a></div>

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. raw:: html

   <div><a href="../../readme_text/pt/auditing-data-and-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/auditing-data-and-installation.desktop.svg"><img src="../../readme_images/pt/auditing-data-and-installation.svg" alt="Auditoria dos dados e da instalação."></picture></a></div>

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. raw:: html

   <div><a href="../../readme_text/pt/environment.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/environment.desktop.svg"><img src="../../readme_images/pt/environment.svg" alt="Ambiente."></picture></a></div>

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. raw:: html

   <div><a href="../../readme_text/pt/environment-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/environment-2.desktop.svg"><img src="../../readme_images/pt/environment-2.svg" alt="Os logs rotativos são gravados em ~/.spacr/logs/spacr.log. Anexe esse arquivo a um relatório de bug."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/contributing-and-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/contributing-and-support.desktop.svg"><img src="../../readme_images/pt/contributing-and-support.svg" alt="Contribuições e suporte. Envie relatos de erros e solicitações de recursos bem delimitadas pelo GitHub Issues. Ao relatar uma falha, inclua a versão do spaCR, o sistema operacional, a versão do Python, as configurações do módulo e o trecho relevante do log. O spacr-doctor coleta a maior parte dessas informações; inclua o relatório de hardware ao relatar problemas de desempenho."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/licensing.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/licensing.desktop.svg"><img src="../../readme_images/pt/licensing.svg" alt="Licença. O spaCR é distribuído sob a licença BSD3-Clause. Se o spaCR contribuiu para um trabalho publicado, agradecemos uma citação, mas ela não é uma condição da licença; veja Como citar o spaCR abaixo."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/tutorials.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/tutorials.desktop.svg"><img src="../../readme_images/pt/tutorials.svg" alt="Tutoriais. A biblioteca de tutoriais interativos do spaCR oferece guias de instalação e uso dos módulos. Cada aula indica a narração e os idiomas disponíveis."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/citing-spacr.desktop.svg"><img src="../../readme_images/pt/citing-spacr.svg" alt="Como citar o spaCR. Se o spaCR contribuir para sua pesquisa, cite: Olafsson EB, et al. Um pooled image-based CRISPR screen identifica o EAF1 como um modulador T. gondii da subversão ESCRT."></picture></a></div>

.. raw:: html

   <div><a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/citing-spacr-2-1.desktop.svg"><img src="../../readme_images/pt/citing-spacr-2-1.svg" alt="bioRxiv pré-impressão"></picture></a><a href="https://doi.org/10.5281/zenodo.21343316"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/citing-spacr-2-2.desktop.svg"><img src="../../readme_images/pt/citing-spacr-2-2.svg" alt="arquivo de software"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/other-work-citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-work-citing-spacr.desktop.svg"><img src="../../readme_images/pt/other-work-citing-spacr.svg" alt="Outros trabalhos citando spaCR."></picture></a></div>

.. spacr-citing-papers-begin

.. raw:: html

   <div><a href="../../readme_text/pt/other-work-citing-spacr-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-work-citing-spacr-2-1.desktop.svg"><img src="../../readme_images/pt/other-work-citing-spacr-2-1.svg" alt="Adaptação metabólica e eliminação de nutrientes em Toxoplasma gondii: insights de mutantes deficientes em vias de ingestão."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/other-work-citing-spacr-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-work-citing-spacr-2-2.desktop.svg"><img src="../../readme_images/pt/other-work-citing-spacr-2-2.svg" alt="O IRE1 promove o fluxo de cálcio fagossomal para aumentar a atividade fungicida dos macrófagos."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/other-work-citing-spacr-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-work-citing-spacr-2-3.desktop.svg"><img src="../../readme_images/pt/other-work-citing-spacr-2-3.svg" alt="O Toxoplasma GRA8 envolve a proteína acessória do hospedeiro ESCRT ALG-2 e é necessário para a integridade metabólica do parasita."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/other-work-citing-spacr-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-work-citing-spacr-2-4.desktop.svg"><img src="../../readme_images/pt/other-work-citing-spacr-2-4.svg" alt="Hydrogen sulfide dynamically upregulates copper uptake and localization."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/pt/other-work-citing-spacr-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/other-work-citing-spacr-2-5.desktop.svg"><img src="../../readme_images/pt/other-work-citing-spacr-2-5.svg" alt="spaCR: Análise do fenótipo espacial das triagens CRISPR-Cas9 (versão de pré-impressão 1)."></picture></a></div>

.. spacr-citing-papers-end

.. raw:: html

   <div><a href="../../readme_text/pt/acknowledgments.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/pt/acknowledgments.desktop.svg"><img src="../../readme_images/pt/acknowledgments.svg" alt="Agradecimentos. O spaCR utiliza software científico aberto, incluindo NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch e Qt."></picture></a></div>
