|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

.. |Website| image:: https://img.shields.io/badge/Website-spaCR-0A7BBB
   :target: https://einarolafsson.github.io/projects/spacr/
   :alt: Sitio web de spaCR

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API&logo=readthedocs&logoColor=white
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentación de la API

.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Tutoriales interactivos

.. |PyPI| image:: https://img.shields.io/pypi/v/spacr?label=PyPI&logo=pypi&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Versión de PyPI

.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 a 3.14

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/nightly?logo=codecov&logoColor=white
   :target: https://app.codecov.io/github/EinarOlafsson/spacr
   :alt: Cobertura de pruebas en Codecov

.. |Test counts| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FEinarOlafsson%2Fspacr%2Fnightly%2Fdocs%2Fsource%2F_static%2Ftest-counts.json&cacheSeconds=300&logo=pytest&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Conjunto de pruebas

.. |Qt| image:: https://img.shields.io/badge/GUI-Qt-41CD52?logo=qt&logoColor=white
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Interfaz Qt

.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr?logo=github
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: Incidencias de GitHub

.. |License| image:: https://img.shields.io/badge/License-BSD--3-blue?logo=opensourceinitiative&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: Licencia BSD 3-Clause

.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: Preprint en bioRxiv

.. |DOI| image:: https://img.shields.io/badge/DOI-blue?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: DOI de Zenodo

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/spacr?label=Conda&logo=anaconda&logoColor=white
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Versión en conda-forge

.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Fecha de la última versión en conda-forge

.. |Docker| image:: https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/pkgs/container/spacr
   :alt: Imágenes de Docker en GHCR

.. |PyPI Downloads| image:: https://img.shields.io/pepy/dt/spacr?label=downloads&logo=pypi&logoColor=white
   :target: https://pepy.tech/projects/spacr
   :alt: Descargas de PyPI

.. |Platforms| image:: https://img.shields.io/badge/Linux%20OSX%20Win-lightgrey?logo=linux&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS y Windows

.. |Cite| image:: https://img.shields.io/badge/Cite-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: Citar spaCR

.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=Rank&color=brightgreen&logo=pypi&logoColor=white&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: Clasificación de descargas de spaCR en PyPI durante los últimos 30 días completos

.. image:: ../../source/_static/deck/slides/slide_01.jpg
   :alt: spaCR
   :width: 920
   :target: https://einarolafsson.github.io/spacr/_static/deck/

| |intro-1|\ |intro-2|

.. |intro-1| image:: ../../readme_images/es/intro-1.svg
   :alt: ← Anterior
   :width: 49.5%
   :target: ../../source/_static/deck/pages/57.md

.. |intro-2| image:: ../../readme_images/es/intro-2.svg
   :alt: Siguiente →
   :width: 49.5%
   :target: ../../source/_static/deck/pages/02.md

.. spacr-language-picker-begin

.. image:: ../../readme_images/es/generated-language-picker.svg
   :alt: Idiomas: Español
   :width: 32.8%
   :target: README.md

.. spacr-language-picker-end

.. image:: ../../readme_images/es/spacr.svg
   :alt: spaCR. Análisis espacial del fenotipo en cribados CRISPR. spaCR segmenta y mide células individuales en imágenes de microscopía, integra los fenotipos por objeto con la abundancia de guías derivada de la secuenciación y estima qué genes están asociados con cambios fenotípicos. A partir de imágenes de placas y lecturas FASTQ, produce mediciones por objeto, clasificadores entrenados, estimaciones del efecto por guía y por gen y una lista ordenada de resultados. Los módulos de segmentación, medición, anotación y clasificación también funcionan sin un brazo de secuenciación. Make Masks corrige máscaras de segmentación y anota rectángulos independientes con etiquetas de clase mediante la herramienta Box para exportarlos a YOLO. Los recuadros conservan sus propias etiquetas e historial sin modificar las imágenes ni las máscaras de origen. Consulte la guía de funciones para conocer cada herramienta. Las imágenes, máscaras y recortes son archivos en disco. Las mediciones, anotaciones y predicciones del modelo son tablas de la base de datos SQLite measurements.db del proyecto. Se ejecuta como una aplicación de escritorio o sin interfaz gráfica en una estación de trabajo, servidor o clúster.
   :width: 100%
   :target: ../../source/features.rst

.. image:: ../../readme_images/es/try-spacr.svg
   :alt: Probar spaCR.
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. image:: ../../readme_images/es/try-spacr-2.svg
   :alt: Use Cargar datos de prueba… en Import, Make Masks, Annotate o una pantalla de ensayo para descargar datos de ejemplo. Desde una terminal, use spacr-download.
   :width: 100%

.. image:: ../../readme_images/es/hardware-support.svg
   :alt: Soporte de hardware.
   :width: 100%

.. spacr-hardware-begin

.. image:: ../../readme_images/es/generated-hardware-1.svg
   :alt: NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU. compatible (estable) implementado (beta) solo compatible con CPU.
   :width: 100%

.. spacr-hardware-end

.. image:: ../../readme_images/es/install-spacr.svg
   :alt: Instalar spaCR. Elija una vía de instalación: Aplicación de escritorio: un instalador para Linux, macOS o Windows con su propio Python. pip: python -m pip install spacr en un entorno de Conda. conda-forge: conda install conda-forge::spacr en un entorno de Conda. Docker: los flujos de procesamiento de línea de comandos en un contenedor Linux, con la imagen para CPU o la imagen para GPU NVIDIA:
   :width: 100%

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.4 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.4-cuda12.4 spacr-run --list

.. image:: ../../readme_images/es/install-spacr-2.svg
   :alt: La guía de Docker explica la configuración de la GPU, el montaje de datos y los comandos de los flujos de procesamiento.
   :width: 100%
   :target: ../../source/installer_guide.rst#container-images

.. image:: ../../readme_images/es/desktop-application.svg
   :alt: Aplicación de escritorio. Los instaladores agrupan sus propios Python. No se requiere Conda.
   :width: 100%

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11: descargar spaCR 1.5.1.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel y Apple silicon): descargar spaCR 1.5.1.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Linux de 64 bits: descargar spaCR 1.5.1.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Linux-x86_64-Online.run

.. spacr-installer-links-end

.. image:: ../../readme_images/es/desktop-application-2.svg
   :alt: En Linux, marque el archivo descargado como ejecutable y ejecútelo:
   :width: 100%

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. image:: ../../readme_images/es/desktop-application-3.svg
   :alt: En macOS, abra el archivo .pkg. La beta actual no está notarizada; si Gatekeeper la bloquea, seleccione Ajustes del Sistema → Privacidad y seguridad → Abrir igualmente. Consulte la guía de instalación para obtener instrucciones de actualización, desinstalación, uso sin conexión y solución de problemas, y los requisitos del sistema para ver recomendaciones sobre estaciones de trabajo y servidores, así como tablas de compatibilidad de GPU.
   :width: 100%

| |desktop-application-3-link-1|\ |desktop-application-3-link-2|

.. |desktop-application-3-link-1| image:: ../../readme_images/es/desktop-application-3-link-1.svg
   :alt: guía de instalación
   :width: 49.5%
   :target: ../../source/installer_guide.rst

.. |desktop-application-3-link-2| image:: ../../readme_images/es/desktop-application-3-link-2.svg
   :alt: requisitos del sistema
   :width: 49.5%
   :target: ../../source/system_requirements.rst

.. image:: ../../readme_images/es/pypi-installation.svg
   :alt: Instalación desde PyPI. Para la versión de PyPI, instale spaCR con pip dentro de un entorno Conda. Python 3.12 ofrece la mayor variedad de paquetes científicos opcionales:
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. image:: ../../readme_images/es/pypi-installation-2.svg
   :alt: spaCR admite Python 3.9 through 3.14, salvo Python 3.14.1, que torchvision excluye. Se recomienda Linux para los flujos de trabajo CUDA y ROCm más exigentes; macOS y Windows también son compatibles, y ambos usan sus GPU — macOS mediante Metal, que cubre Apple Silicon y las tarjetas AMD de los Mac con Intel, y Windows mediante CUDA o DirectML. La instalación estándar incluye la interfaz de escritorio Qt. En un servidor, clúster o ejecutor de CI, ejecute los flujos de trabajo de línea de comandos sin abrirla:
   :width: 100%

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. image:: ../../readme_images/es/pypi-installation-3.svg
   :alt: Optional integrations are installed separately, for example spacr[zarr], spacr[omero], spacr[napari] y spacr[czi,nd2,lif]. See the Guía de instalación for the complete extras y Python-version compatibility table.
   :width: 100%
   :target: ../../source/installer_guide.rst

.. image:: ../../readme_images/es/conda-forge-installation.svg
   :alt: Instalación con conda-forge. El paquete oficial de conda-forge instala spaCR y sus dependencias de escritorio en el entorno activo:
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. image:: ../../readme_images/es/install-from-source.svg
   :alt: Instalación desde el código fuente. Clone el repositorio e instálelo en modo editable, de modo que su copia de trabajo sea el paquete instalado y los cambios surtan efecto sin reinstalar:
   :width: 100%

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. image:: ../../readme_images/es/install-from-source-2.svg
   :alt: Así se clona main, la rama predeterminada, que contiene la última versión publicada. El desarrollo se hace en nightly; añada --branch nightly para clonar esa rama en su lugar. Para una versión concreta:
   :width: 100%

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. image:: ../../readme_images/es/install-from-source-3.svg
   :alt: Para incorporar cambios posteriores, ejecute desde dentro del clon:
   :width: 100%

.. code-block:: bash

   git pull
   pip install -e .

.. image:: ../../readme_images/es/install-from-source-4.svg
   :alt: Reinstale cuando cambien las dependencias o los puntos de entrada. Los cambios en el código Python se aplican directamente; spacr-doctor identifica la instalación activa.
   :width: 100%

.. image:: ../../readme_images/es/install-from-source-light.svg
   :alt: Instalación desde el código fuente (ligera). Los colaboradores necesitan el historial de versiones; para ejecutar spaCR, elija una de las opciones siguientes. Mediciones de nightly en 05302fd5c el 2026-10-07, realizadas con packaging/measure_clone_forms.sh:
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

.. image:: ../../readme_images/es/install-from-source-light-2.svg
   :alt: El clon completo de nightly descargó 9.25 GiB. Añadir --filter=blob:none al clon superficial no reduce la copia de trabajo: su almacén de objetos Git aún ocupa 2032 MB. Git no informa de las descargas diferidas de objetos, por lo que no se midió un total de descarga. Los archivos versionados de nightly ocupan 3416 MB en la copia de trabajo (medidos el 2026-10-08), sin el historial de Git. El tamaño y la duración de la descarga varían según la rama.
   :width: 100%

.. image:: ../../readme_images/es/command-line-entry-points.svg
   :alt: Comandos de línea de comandos.
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

.. image:: ../../readme_images/es/command-line-entry-points-2.svg
   :alt: spacr-run --list enumera los módulos con puntos de entrada de línea de comandos para ejecutarse sin interfaz gráfica. Se omiten los módulos de anotación, curación, comparación y exploración disponibles únicamente en la interfaz gráfica.
   :width: 100%

.. image:: ../../readme_images/es/core-workflow.svg
   :alt: Flujo de trabajo principal. El flujo de trabajo principal comprende seis módulos: Mask segmenta células, núcleos, patógenos y orgánulos con Cellpose. Measure guarda en SQLite características morfológicas, de intensidad, textura, espaciales y de colocalización, junto con recortes de objetos. Annotate etiqueta recortes en una cuadrícula controlada con el teclado y admite colas de aprendizaje activo. Classify entrena modelos basados en imágenes o mediciones y registra con cada punto de control el rendimiento en los datos reservados. Map Barcodes asigna las lecturas FASTQ a los pocillos y los gRNA, con controles de calidad de abundancia, colisiones y cobertura. Regression estima los efectos de guías, genes, condiciones y controles con familias de modelos adecuadas para respuestas continuas, fraccionarias y de recuento.
   :width: 100%

.. image:: ../../readme_images/es/spacr-modules.svg
   :alt: Módulos de spaCR.
   :width: 100%

.. spacr-workflow-begin

.. image:: ../../readme_images/es/generated-workflow-1.svg
   :alt: Principal. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression.
   :width: 100%

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. image:: ../../readme_images/es/generated-workflow-2.svg
   :alt: Datos. Import images and tables into spaCR projects and execute reproducible multi-plate workflows.
   :width: 100%

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

.. image:: ../../readme_images/es/generated-workflow-3.svg
   :alt: Herramientas. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality.
   :width: 100%

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. image:: ../../readme_images/es/generated-workflow-4.svg
   :alt: Organismo. Análisis de imágenes específico del organismo y resultados cuantitativos de ensayos.
   :width: 100%

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

.. image:: ../../readme_images/es/spacr-modules-2.svg
   :alt: Todos los módulos con una tarjeta en la pantalla de inicio, en el orden de esa pantalla: primero los seis módulos de la secuencia de procesamiento y después los demás. Seleccione una tarjeta para abrir la página de API del módulo.
   :width: 100%

.. image:: ../../readme_images/es/other-resources.svg
   :alt: Otros recursos.
   :width: 100%

.. image:: ../../readme_images/es/other-resources-2-1.svg
   :alt: Tutoriales interactivos — flujos de trabajo guiados desde la instalación hasta la investigación de resultados positivos.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/tutorials/

.. image:: ../../readme_images/es/other-resources-2-2.svg
   :alt: Inicio rápido Python API — ejecutar y validar flujos de trabajo desde scripts, cuadernos o un clúster.
   :width: 100%
   :target: ../../source/python_api.rst

.. image:: ../../readme_images/es/other-resources-2-3.svg
   :alt: Guía de características — capacidades, madurez e integraciones opcionales.
   :width: 100%
   :target: ../../source/features.rst

.. image:: ../../readme_images/es/other-resources-2-4.svg
   :alt: Referencia comisariada API — puntos de entrada soportados por tarea, con el módulo completo de referencia un nivel más profundo.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/api/index.html

.. image:: ../../readme_images/es/other-resources-2-5.svg
   :alt: Guía de idioma y traducción — lenguajes de interfaz, ayuda contextual y política de salida científica.
   :width: 100%
   :target: ../../source/localization.rst

.. image:: ../../readme_images/es/language-translation.svg
   :alt: Idioma y traducción. La interfaz admite diez idiomas en la navegación y las preferencias. Los controles AI y LIVE, las descripciones de los módulos y la ayuda contextual revisada también se traducen. Cambie el idioma en spaCR → Preferencias → Idioma sin reiniciar. Los registros, las rutas, los valores de la base de datos y las mediciones nunca se traducen; los resultados científicos permanecen en inglés canónico. Consulte la política de ayuda contextual.
   :width: 100%
   :target: ../../source/localization.rst#contextual-help

.. image:: ../../readme_images/es/language-translation-2.svg
   :alt: Los nueve catálogos no ingleses son redactados por máquina y revisados técnicamente en lugar de leer de extremo a extremo por un hablante nativo. Los registros ámbito de aplicación de la revisión que los idiomas han tenido un pase humano y cada término dejado en inglés por decisión.
   :width: 100%
   :target: ../REVIEW_SCOPE_2026-09-04.md

.. image:: ../../readme_images/es/animated-setting-guidance.svg
   :alt: Guía animada de ajustes. Los ajustes con una explicación visual incluyen un control Animación en su información emergente. Consulte la galería de animaciones de ajustes o el registro de animaciones de ajustes.
   :width: 100%

| |animated-setting-guidance-link-1|\ |animated-setting-guidance-link-2|

.. |animated-setting-guidance-link-1| image:: ../../readme_images/es/animated-setting-guidance-link-1.svg
   :alt: galería de animaciones de ajustes
   :width: 49.5%
   :target: https://einarolafsson.github.io/spacr/setting_animations.html

.. |animated-setting-guidance-link-2| image:: ../../readme_images/es/animated-setting-guidance-link-2.svg
   :alt: registro de animaciones de ajustes
   :width: 49.5%
   :target: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. image:: ../../readme_images/es/data.svg
   :alt: Datos. Conjuntos de datos de referencia.
   :width: 100%

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

.. image:: ../../readme_images/es/model-zoo.svg
   :alt: Biblioteca de modelos. spaCR envía un catálogo de modelos entrenados y los trae a pedido. Abra Model Zoo desde la pantalla de inicio para navegar e instalarlos, o nombre una clave en un archivo de configuración -- pathogen_model: toxoplasma_pv_v1 -- y el modelo se descarga y comprueba la primera vez que es necesario. Cada entrada publicada lleva un SHA-256; una entrada sin uno se rechaza en lugar de instalarse, porque un puesto de control truncado o sustituido no se puede decir desde el real.
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
   :alt: Score card of Parasite vacuoles v4 (toxoplasma_pv_v4, default): F1 0.854 on 11 held-out wells against 0.765 for stock cpsam_v2. F1 0.854 against stock cpsam_v2's 0.765 on the 11 anchor wells at IoU 0.5; AJI 0.776 against 0.505.
   :target: ../../models/toxoplasma_pv_v4.md
   :width: 49.5%

.. |zoo_card_toxoplasma_plaque_v4| image:: ../../models/cards/toxoplasma_plaque_v4.svg
   :alt: Score card of Plaques v4 (toxoplasma_plaque_v4, default): F1 0.815 on 263 held-out fields against 0.373 for stock cpsam_v2. F1 0.815 (toxoplasma_plaque_v3 0.844) on the 263 held-out fields at IoU 0.5; stock cpsam_v2 0.373 there with the Cellpose evaluation scorer.
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

.. image:: ../../readme_images/es/generated-model-zoo-1.svg
   :alt: Older models.
   :width: 100%

| |generated-model-zoo-1-link-1|\ |generated-model-zoo-1-link-2|\ |generated-model-zoo-1-link-3|
| |generated-model-zoo-1-link-4|\ |generated-model-zoo-1-link-5|\ |generated-model-zoo-1-link-6|
| |generated-model-zoo-1-link-7|

.. |generated-model-zoo-1-link-1| image:: ../../readme_images/es/generated-model-zoo-1-link-1.svg
   :alt: toxoplasma_pv_v3
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v3.md

.. |generated-model-zoo-1-link-2| image:: ../../readme_images/es/generated-model-zoo-1-link-2.svg
   :alt: toxoplasma_pv_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v2.md

.. |generated-model-zoo-1-link-3| image:: ../../readme_images/es/generated-model-zoo-1-link-3.svg
   :alt: toxoplasma_pv_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v1.md

.. |generated-model-zoo-1-link-4| image:: ../../readme_images/es/generated-model-zoo-1-link-4.svg
   :alt: toxoplasma_plaque_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v2.md

.. |generated-model-zoo-1-link-5| image:: ../../readme_images/es/generated-model-zoo-1-link-5.svg
   :alt: toxoplasma_plaque_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v1.md

.. |generated-model-zoo-1-link-6| image:: ../../readme_images/es/generated-model-zoo-1-link-6.svg
   :alt: toxoplasma_well_detector_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_well_detector_v2.md

.. |generated-model-zoo-1-link-7| image:: ../../readme_images/es/generated-model-zoo-1-link-7.svg
   :alt: toxoplasma_well_detector_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_well_detector_v1.md

.. image:: ../../readme_images/es/generated-model-zoo-2.svg
   :alt: Alternatives.
   :width: 100%

| |generated-model-zoo-2-link-1|\ |generated-model-zoo-2-link-2|

.. |generated-model-zoo-2-link-1| image:: ../../readme_images/es/generated-model-zoo-2-link-1.svg
   :alt: toxoplasma_pv_v5
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v5.md

.. |generated-model-zoo-2-link-2| image:: ../../readme_images/es/generated-model-zoo-2-link-2.svg
   :alt: toxoplasma_plaque_v3
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v3.md

.. spacr-model-zoo-end

.. image:: ../../readme_images/es/diagnosing-performance.svg
   :alt: Diagnóstico del rendimiento. Genere un informe de hardware y adjúntelo a una incidencia relacionada con el rendimiento:
   :width: 100%

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. image:: ../../readme_images/es/diagnosing-performance-2.svg
   :alt: Guarda en ~/.spacr/reports e imprime la ruta. --quick omite los parámetros de referencia más largos; --out PATH establece la ubicación. No lee datos del proyecto. Mide el tiempo de las importaciones, las bibliotecas numéricas, la creación de ventanas y la animación, e informa de la emulación x86_64 en Apple Silicon y de la implementación de BLAS que usa NumPy.
   :width: 100%

.. image:: ../../readme_images/es/command-line-reference.svg
   :alt: Referencia de la línea de órdenes. Cada comando de abajo está instalado por pip install spacr. Todos aceptan --help.
   :width: 100%

.. image:: ../../readme_images/es/launching-the-application.svg
   :alt: Lanzamiento de la aplicación.
   :width: 100%

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. image:: ../../readme_images/es/when-spacr-will-not-start.svg
   :alt: Cuando spaCR no se iniciará.
   :width: 100%

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. image:: ../../readme_images/es/when-spacr-will-not-start-2.svg
   :alt: spacr-doctor imprime una línea por cheque, con un comando para ejecutar por cada fallo. También informa qué spacr está en la ruta. safespacr lee cada preferencia con su valor predeterminado y desactiva el fondo, las animaciones, el registro detallado y la precarga. Úselo cuando una preferencia guardada impide el inicio. Las preferencias guardadas en modo seguro se escriben normalmente y se conservan.
   :width: 100%

.. image:: ../../readme_images/es/running-modules-headlessly.svg
   :alt: Módulos de ejecución sin interfaz gráfica. No Qt, no display — para clusters, servidores e IC.
   :width: 100%

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. image:: ../../readme_images/es/running-modules-headlessly-2.svg
   :alt: validate lee la misma configuración que leería la ejecución y enumera cada problema, como un ajuste ausente o una ruta que no existe, junto con una solución.
   :width: 100%

.. image:: ../../readme_images/es/inspecting-a-run.svg
   :alt: Inspeccionar una ejecución. Cada ejecución se lleva a cabo a ~/.spacr/runs con sus ajustes, entradas de hashed, salidas, advertencias, versiones y semillas.
   :width: 100%

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. image:: ../../readme_images/es/auditing-data-and-installation.svg
   :alt: Datos de auditoría e instalación.
   :width: 100%

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. image:: ../../readme_images/es/environment.svg
   :alt: Entorno.
   :width: 100%

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. image:: ../../readme_images/es/environment-2.svg
   :alt: Los registros de rotación se escriben en ~/.spacr/logs/spacr.log. Adjuntar ese archivo a un informe de fallo.
   :width: 100%

.. image:: ../../readme_images/es/contributing-and-support.svg
   :alt: Contribuciones y soporte. Envíe informes de errores y solicitudes de funciones concretas mediante GitHub Issues. Al informar de un fallo, incluya la versión de spaCR, el sistema operativo, la versión de Python, los ajustes del módulo y el fragmento de registro pertinente. spacr-doctor recopila la mayor parte de esta información; incluya el informe de hardware cuando notifique problemas de rendimiento.
   :width: 100%
   :target: https://github.com/EinarOlafsson/spacr/issues

.. image:: ../../readme_images/es/licensing.svg
   :alt: Licencia. spaCR se libera bajo el Licencia de 3-clausura BSD. Si spaCR contribuyó al trabajo publicado, una citación es apreciada y no es una condición de la licencia — véase Citar spaCR a continuación.
   :width: 100%
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE

.. image:: ../../readme_images/es/tutorials.svg
   :alt: Tutoriales. La biblioteca de tutoriales interactivos de spaCR ofrece guías de instalación y uso de los módulos. Cada lección indica la narración y los idiomas disponibles.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/tutorials/

.. image:: ../../readme_images/es/citing-spacr.svg
   :alt: Citar spaCR. Si spaCR contribuye a su investigación, cite: Olafsson EB, et al. Una cribado de imagen agrupada basada en CRISPR identifica EAF1 como un modulador T. gondii de subversión ESCRT.
   :width: 100%

| |citing-spacr-2-1|\ |citing-spacr-2-2|

.. |citing-spacr-2-1| image:: ../../readme_images/es/citing-spacr-2-1.svg
   :alt: preimpresión de bioRxiv
   :width: 49.5%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

.. |citing-spacr-2-2| image:: ../../readme_images/es/citing-spacr-2-2.svg
   :alt: archivo de software
   :width: 49.5%
   :target: https://doi.org/10.5281/zenodo.21343316

.. image:: ../../readme_images/es/other-work-citing-spacr.svg
   :alt: Otros trabajos citando spaCR.
   :width: 100%

.. spacr-citing-papers-begin

.. image:: ../../readme_images/es/other-work-citing-spacr-2-1.svg
   :alt: Adaptabilidad metabólica y recolección de nutrientes en Toxoplasma gondii: percepciones de mutantes deficientes en la vía de ingestión.
   :width: 100%
   :target: https://journals.asm.org/doi/full/10.1128/msphere.01011-24

.. image:: ../../readme_images/es/other-work-citing-spacr-2-2.svg
   :alt: IRE1α promueve el flujo de calcio fagosomal para mejorar la actividad fungicida de los macrófagos.
   :width: 100%
   :target: https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00465-6

.. image:: ../../readme_images/es/other-work-citing-spacr-2-3.svg
   :alt: El toxoplasma GRA8 activa la proteína accesoria ESCRT ALG-2 y es necesario para la integridad metabólica del parásito.
   :width: 100%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.20.739547v1.abstract

.. image:: ../../readme_images/es/other-work-citing-spacr-2-4.svg
   :alt: Hydrogen sulfide dynamically upregulates copper uptake and localization.
   :width: 100%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.30.741779v1

.. image:: ../../readme_images/es/other-work-citing-spacr-2-5.svg
   :alt: spaCR: Análisis espacial del fenotipo de cribados CRISPR-Cas9 (versión preliminar 1).
   :width: 100%
   :target: https://www.researchsquare.com/article/rs-7368254/v1

.. spacr-citing-papers-end

.. image:: ../../readme_images/es/acknowledgments.svg
   :alt: Agradecimientos. spaCR se basa en software científico abierto, como NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch y Qt. Consulte la atribución de los modelos de traducción para conocer los modelos utilizados en la documentación multilingüe y los catálogos de la interfaz.
   :width: 100%
   :target: ../TRANSLATION_MODELS.md
