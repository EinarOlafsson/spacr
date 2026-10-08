|Platforms| |Python| |Qt| |Tests| |Release| |Issues| |Source| |Conda| |PyPI| |Conda Downloads| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API%20Documentation
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentación de la API
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Tutoriales interactivos
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: Versión de PyPI
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 a 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Conjunto de pruebas
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Interfaz Qt
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: Código fuente en GitHub
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: Incidencias de GitHub
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: Licencia BSD 3-Clause
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-2026.07.08.737057-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: Preprint en bioRxiv
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: DOI de Zenodo
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Instaladores más recientes
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Versión en conda-forge
.. |Conda Downloads| image:: https://anaconda.org/conda-forge/spacr/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Descargas de conda-forge
.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Fecha de la última versión en conda-forge
.. |PyPI Downloads| image:: https://static.pepy.tech/personalized-badge/spacr?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads
   :target: https://pepy.tech/projects/spacr
   :alt: Descargas de PyPI
.. |Platforms| image:: https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS y Windows
.. |Cite| image:: https://img.shields.io/badge/Cite-CITATION.cff-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: Citar spaCR
.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=PyPI+rank+%2830d%29&color=brightgreen&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: Clasificación de descargas de spaCR en PyPI durante los últimos 30 días completos

.. image:: ../../source/_static/deck/slides/slide_01.jpg
   :alt: spaCR
   :width: 920
   :target: https://einarolafsson.github.io/spacr/_static/deck/

`← Anterior <../../source/_static/deck/pages/57.md>`_   `Siguiente → <../../source/_static/deck/pages/02.md>`_

spaCR
=====

.. spacr-language-picker-begin

Idiomas: `🌐 Español ▾ <README.md>`_

.. spacr-language-picker-end

**Análisis espacial del fenotipo en cribados CRISPR.**

spaCR segmenta y mide células individuales en imágenes de microscopía, integra los fenotipos por objeto con la abundancia de guías derivada de la secuenciación y estima qué genes están asociados con cambios fenotípicos. A partir de imágenes de placas y lecturas FASTQ, produce mediciones por objeto, clasificadores entrenados, estimaciones del efecto por guía y por gen y una lista ordenada de resultados.

Los módulos de segmentación, medición, anotación y clasificación también funcionan sin un brazo de secuenciación.

Make Masks corrige máscaras de segmentación y anota rectángulos independientes con etiquetas de clase mediante la herramienta **Box** para exportarlos a YOLO. Los recuadros conservan sus propias etiquetas e historial sin modificar las imágenes ni las máscaras de origen.

Consulte la `guía de funciones <../../source/features.rst>`_ para conocer cada herramienta.

Imágenes, máscaras, recortes, mediciones, anotaciones, predicciones, códigos de barras e identificadores de pocillo residen en un único proyecto SQLite.

Se ejecuta como una aplicación de escritorio o sin interfaz gráfica en una estación de trabajo, servidor o clúster.

Probar spaCR
~~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

Use **Cargar datos de prueba…** en Import, Make Masks, Annotate o una pantalla de ensayo para descargar datos de ejemplo. Desde una terminal, use ``spacr-download``.

Soporte de hardware
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


Instalar spaCR
~~~~~~~~~~~~~~

Aplicación de escritorio
------------------------

Los instaladores agrupan sus propios Python. No se requiere Conda.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11: descargar spaCR 1.5.1.3
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.3/spaCR-1.5.1.3-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel y Apple silicon): descargar spaCR 1.5.1.3
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.3/spaCR-1.5.1.3-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Linux de 64 bits: descargar spaCR 1.5.1.3
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.3/spaCR-1.5.1.3-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Instaladores anteriores de spaCR
   :target: ../../source/installers.rst

.. spacr-installer-links-end

En Linux, marque el archivo descargado como ejecutable y ejecútelo:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

En macOS, abra el archivo ``.pkg``. La beta actual no está notarizada; si Gatekeeper la bloquea, seleccione **Ajustes del Sistema → Privacidad y seguridad → Abrir igualmente**.

Consulte la `guía de instalación <../../source/installer_guide.rst>`_ para obtener instrucciones de actualización, desinstalación, uso sin conexión y solución de problemas, y los `requisitos del sistema <../../source/system_requirements.rst>`_ para ver recomendaciones sobre estaciones de trabajo y servidores, así como tablas de compatibilidad de GPU.

Instalación desde PyPI
----------------------

Para la versión de PyPI, instale spaCR con pip dentro de un entorno Conda. Python 3.12 ofrece la mayor variedad de paquetes científicos opcionales:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

spaCR admite Python **3.9 through 3.14**, salvo Python 3.14.1, que torchvision excluye. Se recomienda Linux para los flujos de trabajo CUDA y ROCm más exigentes; macOS y Windows también son compatibles, y ambos usan sus GPU — macOS mediante Metal, que cubre Apple Silicon y las tarjetas AMD de los Mac con Intel, y Windows mediante CUDA o DirectML.

La instalación estándar incluye la interfaz de escritorio Qt. En un servidor, clúster o ejecutor de CI, ejecute los flujos de trabajo de línea de comandos sin abrirla:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example ``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` y ``spacr[czi,nd2,lif]``. See the `Guía de instalación <../../source/installer_guide.rst>`_ for the complete extras y Python-version compatibility table.

Instalación con conda-forge
---------------------------

El paquete oficial de conda-forge instala spaCR y sus dependencias de escritorio en el entorno activo:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Instalación con Docker
----------------------

Ejecute los flujos de procesamiento de línea de comandos de spaCR en un contenedor con las `imágenes de Docker publicadas en GHCR <https://github.com/EinarOlafsson/spacr/pkgs/container/spacr>`_. Instale `Docker <https://docs.docker.com/get-started/get-docker/>`_ y, a continuación, enumere los flujos disponibles con esta imagen publicada para CPU:

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.0 spacr-run --list

La imagen correspondiente para GPU NVIDIA es ``ghcr.io/einarolafsson/spacr:1.5.1.0-cuda12.4``. Ambas imágenes están destinadas a contenedores Linux x86-64. Consulte la `guía de instalación con Docker <../../source/installer_guide.rst#container-images>`_ para conocer los requisitos de GPU, los montajes de datos y modelos, los archivos de configuración y los comandos completos de los flujos de procesamiento.

Instalación desde el código fuente
----------------------------------

Clone el repositorio e instálelo en modo editable, de modo que su copia de trabajo *sea* el paquete instalado y los cambios surtan efecto sin reinstalar::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

Así se clona ``main``, la rama predeterminada, que contiene la última versión publicada. El desarrollo se hace en ``nightly``; añada ``--branch nightly`` para clonar esa rama en su lugar. Para una versión concreta::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

Para incorporar cambios posteriores, ejecute desde dentro del clon::

    git pull
    pip install -e .

Reinstale cuando cambien las dependencias o los puntos de entrada. Los cambios en el código Python se aplican directamente; ``spacr-doctor`` identifica la instalación activa.

Instalación desde el código fuente (ligera)
-------------------------------------------

Los colaboradores necesitan el historial de versiones; para ejecutar spaCR, elija una de las opciones siguientes. Mediciones de ``nightly`` en ``05302fd5c`` el 2026-10-07, realizadas con ``packaging/measure_clone_forms.sh``::

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

El clon completo de nightly descargó 9.25 GiB. Añadir ``--filter=blob:none`` al clon superficial no reduce el tamaño de la copia de trabajo: su almacén de objetos Git aún ocupa 2032 MB. Las descargas silenciosas bajo demanda impiden medir todo el volumen transferido. Los archivos versionados de nightly ocupan 2839 MB en la copia de trabajo (medidos el 2026-10-07), sin el historial de Git. El tamaño y la duración de la descarga varían según la rama.


Comandos de línea de comandos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

``spacr-run --list`` enumera los módulos con puntos de entrada de línea de comandos para ejecutarse sin interfaz gráfica. Se omiten los módulos de anotación, curación, comparación y exploración disponibles únicamente en la interfaz gráfica.


Flujo de trabajo principal
--------------------------

El flujo de trabajo principal comprende seis módulos:

- **Mask** segmenta células, núcleos, patógenos y orgánulos con Cellpose.
- **Measure** guarda en SQLite características morfológicas, de intensidad, textura, espaciales y de colocalización, junto con recortes de objetos.
- **Annotate** etiqueta recortes en una cuadrícula controlada con el teclado y admite colas de aprendizaje activo.
- **Classify** entrena modelos basados en imágenes o mediciones y registra con cada punto de control el rendimiento en los datos reservados.
- **Map Barcodes** asigna las lecturas FASTQ a los pocillos y los gRNA, con controles de calidad de abundancia, colisiones y cobertura.
- **Regression** estima los efectos de guías, genes, condiciones y controles con familias de modelos adecuadas para respuestas continuas, fraccionarias y de recuento.

Módulos de spaCR
----------------

.. spacr-workflow-begin

Principal
^^^^^^^^^

Core sequence from microscopy images through segmentation, measurements,
annotations, classification, barcode mapping and regression.

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

Datos
^^^^^

Import images and tables into spaCR projects and execute reproducible
multi-plate workflows.

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

Herramientas
^^^^^^^^^^^^

Point these at a project: edit masks by hand, stitch tiles, read an
embedding, draw a gate, build a plot, check quality.

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

Organism
^^^^^^^^

Organism-specific image analysis and quantitative assay readouts.

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../../spacr/resources/icons/workflow/mask.png" width="16.0%" align="middle" alt="Abrir la API de Mask"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../../spacr/resources/icons/workflow/measure.png" width="16.0%" align="middle" alt="Abrir la API de Measure"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../../spacr/resources/icons/workflow/annotate.png" width="16.0%" align="middle" alt="Abrir la API de Annotate"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../../spacr/resources/icons/workflow/classify_merged.png" width="16.0%" align="middle" alt="Abrir la API de Classify"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../../spacr/resources/icons/workflow/map_barcodes.png" width="16.0%" align="middle" alt="Abrir la API de Map Barcodes"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../../spacr/resources/icons/workflow/regression.png" width="16.0%" align="middle" alt="Abrir la API de Regression"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../../spacr/resources/icons/workflow/apps/foreign.png" width="16.0%" align="middle" alt="Abrir la API de Import"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../../spacr/resources/icons/workflow/apps/embeddings.png" width="16.0%" align="middle" alt="Abrir la API de Embeddings"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../../spacr/resources/icons/workflow/apps/run_compare.png" width="16.0%" align="middle" alt="Abrir la API de Run Compare"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../../spacr/resources/icons/workflow/apps/experiment_design.png" width="16.0%" align="middle" alt="Abrir la API de Experiment Design"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../../spacr/resources/icons/workflow/apps/power.png" width="16.0%" align="middle" alt="Abrir la API de Power / Design"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../../spacr/resources/icons/workflow/apps/dose_response.png" width="16.0%" align="middle" alt="Abrir la API de Dose–Response"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../../spacr/resources/icons/workflow/apps/qc_dashboard.png" width="16.0%" align="middle" alt="Abrir la API de QC"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../../spacr/resources/icons/workflow/apps/make_masks.png" width="16.0%" align="middle" alt="Abrir la API de Make Masks"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../../spacr/resources/icons/workflow/apps/align.png" width="16.0%" align="middle" alt="Abrir la API de Align &amp; Stitch"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../../spacr/resources/icons/workflow/apps/umap.png" width="16.0%" align="middle" alt="Abrir la API de Image UMAP"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../../spacr/resources/icons/workflow/apps/gate_editor.png" width="16.0%" align="middle" alt="Abrir la API de Gate Editor"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../../spacr/resources/icons/workflow/apps/graph_builder.png" width="16.0%" align="middle" alt="Abrir la API de Graph Builder"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../../spacr/resources/icons/workflow/apps/toxoplasma.png" width="16.0%" align="middle" alt="Abrir la API de Toxoplasma"></a>

.. spacr-workflow-end

Todos los módulos con una tarjeta en la pantalla de inicio, en el orden de esa pantalla: primero los seis módulos de la secuencia de procesamiento y después los demás. Seleccione una tarjeta para abrir la página de API del módulo.

Otros recursos
~~~~~~~~~~~~~~~

- `Tutoriales interactivos <https://einarolafsson.github.io/spacr/tutorials/>`_ — flujos de trabajo guiados desde la instalación hasta la investigación de resultados positivos.
- `Inicio rápido Python API <../../source/python_api.rst>`_ — ejecutar y validar flujos de trabajo desde scripts, cuadernos o un clúster.
- `Guía de características <../../source/features.rst>`_ — capacidades, madurez e integraciones opcionales.
- `Referencia comisariada API <https://einarolafsson.github.io/spacr/api/index.html>`_ — puntos de entrada soportados por tarea, con el módulo completo de referencia un nivel más profundo.
- `Guía de idioma y traducción <../../source/localization.rst>`_ — lenguajes de interfaz, ayuda contextual y política de salida científica.

Idioma y traducción
~~~~~~~~~~~~~~~~~~~~~~

La interfaz admite diez idiomas en la navegación y las preferencias. Los controles AI y LIVE, las descripciones de los módulos y la ayuda contextual revisada también se traducen. Cambie el idioma en **spaCR → Preferencias → Idioma** sin reiniciar. Los registros, las rutas, los valores de la base de datos y las mediciones nunca se traducen; los resultados científicos permanecen en inglés canónico. Consulte la `política de ayuda contextual <../../source/localization.rst#contextual-help>`_.

Los nueve catálogos no ingleses son redactados por máquina y revisados técnicamente en lugar de leer de extremo a extremo por un hablante nativo. Los registros `ámbito de aplicación de la revisión <../REVIEW_SCOPE_2026-09-04.md>`_ que los idiomas han tenido un pase humano y cada término dejado en inglés por decisión.

Guía animada de ajustes
~~~~~~~~~~~~~~~~~~~~~~~~~

Los ajustes con una explicación visual incluyen un control **Animación** en su información emergente. Consulte la `galería de animaciones de ajustes <https://einarolafsson.github.io/spacr/setting_animations.html>`_ o el `registro de animaciones de ajustes <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Datos
~~~~~

Conjuntos de datos de referencia
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: Abrir el conjunto de microscopía en BioStudies
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Abrir el conjunto de prueba en Hugging Face
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: Abrir el conjunto de secuenciación en NCBI
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: Abrir spaCRPower
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: Abrir la prepublicación de bioRxiv
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

Biblioteca de modelos
~~~~~~~~~~~~~~~~~~~~~

spaCR envía un catálogo de modelos entrenados y los trae a pedido. Abra **Model Zoo** desde la pantalla de inicio para navegar e instalarlos, o nombre una clave en un archivo de configuración -- ``pathogen_model: toxoplasma_pv_v1`` -- y el modelo se descarga y comprueba la primera vez que es necesario. Cada entrada publicada lleva un SHA-256; una entrada sin uno se rechaza en lugar de instalarse, porque un puesto de control truncado o sustituido no se puede decir desde el real.

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

Cada figura de arriba se mide en imágenes que el modelo nunca vio en el entrenamiento.

**Precisión** es cuántos de los objetos reportados por un modelo son reales; **recordar** es cuantos de los verdaderos objetos que encontró. Fallan en direcciones opuestas: la mala precisión inventa placas, la mala memoria los echa en falta.

**F1** son los dos combinados, y se cita porque cada uno es trivialmente gamed -- reporta una placa inconfundible para la precisión casi perfecta, o cada mancha oscura para la memoria casi perfecta. Lo que preferirías perder depende del ensayo, y el conteo es generalmente mejor servido por sobrellamada: el modelo de placa fue aceptado con precisión 0,858 con memoria 0,811 sobre una ronda anterior en 0,939 y 0,631.

**IoU** (intersección sobre unión) divide el área de solapamiento entre el objeto predicho y el de referencia por el área de su unión. Lea las puntuaciones junto con su umbral: «F1 0.864 con IoU 0.5» cuenta una vacuola como detectada cuando el solapamiento alcanza al menos la mitad del área de la unión.

**mAP50** y **m AP50-95** pertenecen al detector. El primero pregunta si se encontraron los pozos; el segundo lo repite a través de diez umbrales de 0,5 a 0,95, por lo que también pregunta qué tan firmemente dibuja cada caja. La brecha entre ellos es la colocación, no la detección.

**Cross-validated**, con un **SD**, significa que la puntuación es la media de tres ejecuciones en diferentes divisiones y el SD es lo lejos que se alejaron. Una división puede tener suerte: la cifra de literatura de este modelo es 0,834 en una sola división de 19 pocillos y 0,806 en los tres.

Los modelos se alojan en la cuenta de Hugging Face de cada autor; ``spacr.model_zoo.publish_model`` sube un modelo e imprime la fila que se debe añadir al catálogo.


Diagnóstico del rendimiento
---------------------------

Genere un informe de hardware y adjúntelo a una incidencia relacionada con el rendimiento::

    python tools/spacr_hardware_report.py

Guarda en ``~/.spacr/reports`` e imprime la ruta. ``--quick`` omite los parámetros de referencia más largos; ``--out PATH`` establece la ubicación.

No lee datos del proyecto. Mide el tiempo de las importaciones, las bibliotecas numéricas, la creación de ventanas y la animación, e informa de la emulación x86_64 en Apple Silicon y de la implementación de BLAS que usa NumPy.

Referencia de la línea de órdenes
---------------------------------

Cada comando de abajo está instalado por ``pip install spacr``. Todos aceptan ``--help``.

Lanzamiento de la aplicación
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-qt`` y ``spacr-nightly`` son alias de ``spacr``.

Cuando spaCR no se iniciará
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` imprime una línea por cheque, con un comando para ejecutar por cada fallo. También informa que ``spacr`` está en la ruta, que es lo que una vieja instalación editable sombras.

``safespacr`` lee cada preferencia como por defecto y fuerza el telón de fondo, animaciones, registro verboso y precargar. Utilícela cuando una preferencia guardada rompa el lanzamiento. No cambia nada de forma permanente.

Módulos de ejecución sin interfaz gráfica
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No Qt, no display — para clusters, servidores e IC.

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` lee los mismos ajustes que la ejecución haría e informa de lo que falta, contradictorio o apuntando a nada.

Inspeccionar una carrera después
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cada ejecución se lleva a cabo a ``~/.spacr/runs`` con sus ajustes, entradas de hashed, salidas, advertencias, versiones y semillas.

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

Datos de auditoría e instalación
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

Entorno
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

Los registros de rotación se escriben en ``~/.spacr/logs/spacr.log``. Adjuntar ese archivo a un informe de fallo.


Contribuciones y soporte
~~~~~~~~~~~~~~~~~~~~~~~~

Envíe informes de errores y solicitudes de funciones concretas mediante `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Al informar de un fallo, incluya la versión de spaCR, el sistema operativo, la versión de Python, los ajustes del módulo y el fragmento de registro pertinente. ``spacr-doctor`` recopila la mayor parte de esta información; incluya el informe de hardware cuando notifique problemas de rendimiento.

Licencia
~~~~~~~~~

spaCR se libera bajo el `Licencia de 3-clausura BSD <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

Si spaCR contribuyó al trabajo publicado, una citación es apreciada y no es una condición de la licencia — véase `Citar spaCR`_ a continuación.

Tutoriales
~~~~~~~~~~

La `biblioteca de tutoriales interactivos de spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ ofrece guías de instalación y uso de los módulos. Cada lección indica la narración y los idiomas disponibles.

Citar spaCR
~~~~~~~~~~~~

Si spaCR contribuye a su investigación, cite:

Olafsson EB, *et al.* Una cribado de imagen agrupada basada en CRISPR identifica EAF1 como un modulador *T. gondii* de subversión ESCRT.

`preimpresión de bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `archivo de software <https://doi.org/10.5281/zenodo.21343316>`_

Otros trabajos citando spaCR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. spacr-citing-papers-begin

* `Adaptabilidad metabólica y recolección de nutrientes en Toxoplasma gondii: percepciones de mutantes deficientes en la vía de ingestión. <https://journals.asm.org/doi/full/10.1128/msphere.01011-24>`_
* `IRE1α promueve el flujo de calcio fagosomal para mejorar la actividad fungicida de los macrófagos. <https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00465-6>`_
* `El toxoplasma GRA8 activa la proteína accesoria ESCRT ALG-2 y es necesario para la integridad metabólica del parásito. <https://www.biorxiv.org/content/10.64898/2026.07.20.739547v1.abstract>`_
* `spaCR: Análisis espacial del fenotipo de cribados CRISPR-Cas9 (versión preliminar 1). <https://www.researchsquare.com/article/rs-7368254/v1>`_

.. spacr-citing-papers-end

Agradecimientos
~~~~~~~~~~~~~~~

spaCR se basa en software científico abierto, como NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch y Qt. Consulte la `atribución de los modelos de traducción <../TRANSLATION_MODELS.md>`_ para conocer los modelos utilizados en la documentación multilingüe y los catálogos de la interfaz.
