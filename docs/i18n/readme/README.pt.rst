|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
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
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: DOI do Zenodo
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Instaladores mais recentes
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: Receita do conda-forge

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/logo_spacr.png
   :alt: spaCR
   :align: center
   :width: 360

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

`Informações sobre os modelos de tradução <../TRANSLATION_MODELS.md>`_

**Análise espacial de fenótipos em triagens CRISPR.**

O spaCR segmenta e mede células individuais em imagens de microscopia de alto conteúdo, associa cada célula ao gRNA que ela recebeu e informa quais genes alteraram o fenótipo. As entradas são imagens de placas e leituras FASTQ; as saídas incluem medições por objeto, classificadores treinados, tamanhos de efeito por guia e por gene e uma lista classificada de resultados.

Para triagens CRISPR agrupadas e baseadas em imagens, esse é o fluxo de trabalho completo. Se você tiver microscopia de alto conteúdo sem uma triagem, as etapas de segmentação, medição, anotação e classificação poderão ser executadas de forma independente.

Imagens, máscaras, recortes, medições, anotações, previsões, códigos de barras e identificadores de poço ficam em um único projeto SQLite, permitindo rastrear qualquer valor de resultado até o objeto de origem.

Execute o spaCR como aplicativo para desktop ou sem interface gráfica em uma estação de trabalho, servidor ou cluster. Os dois modos usam os mesmos módulos, e o CUDA é ativado automaticamente quando houver suporte no módulo.


Visão geral do fluxo de trabalho
--------------------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: Fluxo de trabalho e organização das saídas do spaCR
   :align: center

Imagens de microscopia (TIFF, OME-TIFF, LIF, CZI, ND2) e leituras de sequenciamento (FASTQ) entram em fluxos complementares de análise de imagens e mapeamento de códigos de barras. Em seguida, tabelas de objetos, recortes, anotações, previsões, identidades de guia, resultados de QC e resumos por poço são analisados em conjunto.


Instalar o spaCR
----------------

Aplicativo para desktop
~~~~~~~~~~~~~~~~~~~~~~~

Os instaladores de desktop incluem um ambiente Python privado, portanto, não é necessária uma instalação  Python existente.

.. spacr-installer-links-begin

|InstallerLinux|  |InstallerMacOS| ? |InstallerWindows| . |InstallerLegacy|

.. |InstallerWindows| image:: spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Baixar o spaCR 1.5.0.4 para Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Baixar o spaCR 1.5.0.4 para macOS 11+ (Intel e Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Baixar o spaCR 1.5.0.4 para Linux de 64 bits
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Instaladores anteriores do spaCR
   :target: docs/source/installers.rst

.. spacr-installer-links-end

Os três primeiros ícones baixam a versão atual. O ícone spaCR abre o arquivo completo do instalador. Os links do instaladores e os nomes dos arquivos versionados são atualizados pelo fluxo de trabalho da versão; os instaladores anteriores permanecem no mesmo arquivo de versão.

Em Linux, faça o executável do arquivo baixado e execute-o:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

No macOS, abra o arquivo ``.pkg``. A versão beta atual não é notarizada; se o Gatekeeper a bloquear, selecione **Ajustes do Sistema → Privacidade e Segurança → Abrir Mesmo Assim**.

Veja as instruções `guia do instalador <https://einarolafsson.github.io/spacr/installers.html>`_ para atualização, desinstalação, off-line e solução de problemas.

Instalação com Python
~~~~~~~~~~~~~~~~~~~~~

Python 3,12 tem a mais ampla escolha de pacotes científicos opcionais:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR supports Python **3.9 through 3.14**, except Python 3.14.1, which torchvision excludes. Linux is recommended for CUDA workflows; macOS and Windows are also supported.

Para um servidor, cluster ou corredor CI, omitir Qt:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

As integrações opcionais são instaladas separadamente, por exemplo ``spacr[ome-zarr]``,  ``spacr[omero]``,``spacr[napari]`` e ``spacr[czi,nd2,lif]``. Veja a tabela de compatibilidade  `guia de instalação <https://einarolafsson.github.io/spacr/installers.html>`_ para os extras completos e ?Python-versão.

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

Set ``SPACR_LOG_LEVEL=DEBUG`` when troubleshooting. Os logs rotativos são gravados em  ``~/.spacr/logs/spacr.log``. A interface clássica  Tk permanece disponível como ``spacr-legacy`` mas não está mais desenvolvida.


O que você pode fazer
---------------------

A maioria das triagens segue seis módulos:

- **Mask** segmenta células, núcleos, patógenos e organelas com  Cellpose.
- **Measure** escreve características de morfologia, intensidade, textura, espacial e localização, juntamente com recortes de objetos, para  SQLite.
- **Annotate** rotula recortes em uma grade orientada por teclado e suporta filas de aprendizado ativo.
- **Classify** treina modelos baseados em imagens ou medições e registra o desempenho com cada ponto de verificação.
- **Map Barcodes** mapas FASTQ lê para poços e  gRNAs, com abundância, colisão e cobertura  QC.
- **Regression** estima os efeitos de guia, gene, condição e controle com famílias de modelos adequadas a respostas contínuas, fracionárias e de contagem.

O mesmo projeto também pode projetar placas, estimar a potência, corrigir efeitos de lote, inspecionar a qualidade da segmentação, explorar parcelas e recortes vinculadas, exportar AnnData, retomar o trabalho interrompido e registrar as configurações por trás de cada resultado.

Escolha a próxima página pelo que você quer fazer:

- `Tutoriais interativos <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 guided workflows from installation por meio de hit investigation.
- `Python  API início rápido <https://einarolafsson.github.io/spacr/python_api.html>`_ — run and validate pipelines from scripts, notebooks or a cluster.
- `Guia de funcionalidades <https://einarolafsson.github.io/spacr/features.html>`_ capacidades, maturidade e integrações opcionais.
- `Referência API com curadoria <https://einarolafsson.github.io/spacr/api/index.html>`_ — supported entry points by task, with the complete module reference one level deeper.
- `Guia de localização <https://einarolafsson.github.io/spacr/localization.html>`_ linguagens de interface, ajuda contextual e política de saída científica.

Interface multilíngue para desktop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A localização em dez idiomas abrange navegação, preferências, controles AI e LIVE, descrições de módulos e ajuda contextual revisada. Altere o idioma em **spaCR → Preferências → Idioma** sem reiniciar. Logs, caminhos, valores de banco de dados e medições nunca são traduzidos; a saída científica permanece em inglês canônico. Consulte a `política de ajuda contextual <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_.

Guia animado de configurações
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As configurações com uma explicação visual oferecem um controle **Animation** na dica de ferramenta. Consulte a `galeria de animações de configurações <https://einarolafsson.github.io/spacr/setting_animations.html>`_ ou o `registro de animações de configurações <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Dados
-----

Conjuntos de dados de referência
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Conjunto de dados de microscopia completa: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Testando o conjunto de dados: Hugging Face  toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Dados de sequenciamento: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Análise de potência: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Contribuições e suporte
------------------------

Relatórios de bugs e solicitações de recursos focados são bem-vindos através de `GitHub Problemas <https://github.com/EinarOlafsson/spacr/issues>`_. Ao relatar uma falha, inclua a versão  spaCR, o sistema operacional, a versão do Python, as configurações do módulo e o trecho de log relevante.  ``spacr-doctor`` coleta a maior parte disso para você.

Licença
~~~~~~~~~

O ramo de desenvolvimento atual está disponível na fonte sob o `PolyForm Licença não comercial 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. O uso comercial requer uma licença separada do detentor dos direitos autorais. As versões liberadas através do  spaCR 1.4.9.9 permanecem disponíveis sob a Licença MIT que acompanhou esses lançamentos.

Tutoriais
~~~~~~~~~

O `biblioteca tutorial interativa spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contém passos narrados e legendados de instalação e de cada fluxo de trabalho de aplicativo, em 73 lições com 50 vozes em oito idiomas.

Como citar o spaCR
~~~~~~~~~~~~~~~~~~

Se spaCR contribuir para a sua pesquisa, cite:

Olafsson EB, *et al.* Um pooled image-based  CRISPR screen identifica o EAF1 como um modulador *T. gondii* da subversão ESCRT.

`bioRxiv pré-impressão <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_  · ? `arquivo de software <https://doi.org/10.5281/zenodo.21343317>`_
