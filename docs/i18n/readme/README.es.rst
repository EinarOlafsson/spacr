|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentación
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Tutoriales interactivos
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: Versión de PyPI
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 a 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Conjunto de pruebas
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Interfaz Qt
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: Código fuente en GitHub
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: Incidencias de GitHub
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: Licencia PolyForm Noncommercial
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: DOI de Zenodo
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Instaladores más recientes
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: Receta de conda-forge

.. image:: ../../../spacr/resources/icons/logo_spacr.png
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

**Análisis espacial del fenotipo en cribados CRISPR.**

spaCR segmenta y mide células individuales en imágenes de microscopía de alto contenido, vincula cada célula con el gRNA que recibió e indica qué genes modificaron el fenotipo. Las imágenes de placas y las lecturas FASTQ son la entrada; las mediciones por objeto, los clasificadores entrenados, los tamaños del efecto por guía y por gen y una lista ordenada de resultados son la salida.

Para los cribados CRISPR agrupados y basados en imágenes, este es el flujo de trabajo completo. Si dispone de microscopía de alto contenido sin cribado, las etapas de segmentación, medición, anotación y clasificación pueden ejecutarse por separado.

Las imágenes, máscaras, recortes, mediciones, anotaciones, predicciones, códigos de barras e identificadores de pocillo se guardan en un único proyecto SQLite, por lo que cualquier valor de un resultado puede rastrearse hasta su objeto de origen.

Ejecute spaCR como aplicación de escritorio o sin interfaz gráfica en una estación de trabajo, servidor o clúster. Ambos modos usan los mismos módulos y CUDA se utiliza automáticamente cuando el módulo lo admite.


Flujo de trabajo de un vistazo
------------------------------

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
   :alt: Flujo de trabajo y organización de resultados de spaCR
   :align: center

The main path is Mask → Measure → Annotate → Classify → Map Barcodes → Regression. The grid below it contains every other application in the same categories and order used on the spaCR home screen.


Instalar spaCR
--------------

Aplicación de escritorio
~~~~~~~~~~~~~~~~~~~~~~~~

Los instaladores de escritorio incluyen un entorno privado Python, por lo que conda y una instalación existente Python no son necesarios.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| =|InstallerWindows| +|InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11: descargar spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel y Apple silicon): descargar spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Linux de 64 bits: descargar spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Instaladores anteriores de spaCR
   :target: ../../source/installers.rst

.. spacr-installer-links-end

Los tres primeros iconos descargan la versión actual. El icono spaCR abre el archivo completo del instalador. Los enlaces al instalador y los nombres de archivos versionados se actualizan por el flujo de trabajo de la versión; los instaladores anteriores permanecen en el mismo archivo de lanzamiento.

En Linux, haga que el archivo descargado sea ejecutable y ejecútelo:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

En macOS, abra el archivo ``.pkg``. La beta actual no está notarizada; si Gatekeeper la bloquea, seleccione **Ajustes del Sistema → Privacidad y seguridad → Abrir igualmente**.

Consulte las instrucciones `Guía del instalador <../../source/installers.rst>`_ para actualizar, desinstalar, offline y solucionar problemas.

Instalación con Python
~~~~~~~~~~~~~~~~~~~~~~

Python 3.12 tiene la más amplia selección de envases científicos opcionales:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR supports Python **3.9 through 3.14**, except Python 3.14.1, which torchvision excludes. Linux is recommended for CUDA workflows; macOS and Windows are also supported.

Para un servidor, clúster o corredor CI, omita Qt:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example ``spacr[ome-zarr]``, ``spacr[omero]``, ``spacr[napari]`` y ``spacr[czi,nd2,lif]``. See the `Guía de instalación <../../source/installers.rst>`_ for the complete extras y Python-version compatibility table.

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

Establecer ``SPACR_LOG_LEVEL=DEBUG`` al solucionar problemas. Los registros de rotación se escriben en ``~/.spacr/logs/spacr.log``. La interfaz clásica Tk permanece disponible como ``spacr-legacy`` pero ya no está desarrollada.


Qué puede hacer
---------------

La mayoría de las cribados siguen seis módulos:

- **Mask** segmenta células, núcleos, patógenos y orgánelos con Cellpose.
- **Measure** escribe morfología, intensidad, textura, características espaciales y de colocalización, junto con recortes de objetos, a SQLite.
- **Annotate** recorte en una cuadrícula dirigida por teclado y soporta colas de aprendizaje activo.
- **Classify** forma modelos basados en imágenes o mediciones y registros de rendimiento retenido con cada puesto de control.
- **Map Barcodes** maps FASTQ lee a los pozos y gRNAs, con abundancia, colisión y cobertura QC.
- **Regression** guían, el gen, la condición y los efectos de control con familias modelo adecuadas para respuestas continuas, fraccionarias y de recuento.

El mismo proyecto también puede diseñar placas, estimar la potencia, corregir los efectos por lotes, inspeccionar la calidad de segmentación, explorar parcelas y recortes vinculados, exportar AnnData, reanudar el trabajo interrumpido y registrar los ajustes detrás de cada resultado.

Elija la siguiente página por lo que desea hacer:

- `Tutoriales interactivos <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 flujos de trabajo guiados desde la instalación hasta la investigación de impacto.
- `Inicio rápido Python API <../../source/python_api.rst>`_ — ejecutar y validar flujos de trabajo desde scripts, cuadernos o un clúster.
- `Guía de características <../../source/features.rst>`_ — capacidades, madurez e integraciones opcionales.
- `Referencia comisariada API <https://einarolafsson.github.io/spacr/api/index.html>`_ — puntos de entrada soportados por tarea, con el módulo completo de referencia un nivel más profundo.
- `Guía de idioma y traducción <../../source/localization.rst>`_ — lenguajes de interfaz, ayuda contextual y política de salida científica.

Idioma y traducción
~~~~~~~~~~~~~~~~~~~~~~

La interfaz admite diez idiomas en la navegación y las preferencias. Los controles AI y LIVE, las descripciones de los módulos y la ayuda contextual revisada también se traducen. Cambie el idioma en **spaCR → Preferencias → Idioma** sin reiniciar. Los registros, las rutas, los valores de la base de datos y las mediciones nunca se traducen; los resultados científicos permanecen en inglés canónico. Consulte la `política de ayuda contextual <../../source/localization.rst#contextual-help>`_.

Guía animada de ajustes
~~~~~~~~~~~~~~~~~~~~~~~~~

Los ajustes con una explicación visual incluyen un control **Animación** en su información emergente. Consulte la `galería de animaciones de ajustes <https://einarolafsson.github.io/spacr/setting_animations.html>`_ o el `registro de animaciones de ajustes <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Datos
-----

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


Contribuciones y soporte
------------------------

Bug reports y focused feature requests are welcome through `GitHub Cuestiones <https://github.com/EinarOlafsson/spacr/issues>`_. When reporting a failure, include the spaCR version, operating system, Python version, module settings y the relevant log excerpt. ``spacr-doctor`` collects most of that for you.

Licencia
~~~~~~~~~

The current development branch is source-available under the `Licencia PolyForm no comercial 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Commercial use requires a separate license from the copyright holder. Released versions through spaCR 1.4.9.9 remain available under the MIT License that accompanied those releases.

Tutoriales
~~~~~~~~~~

El `biblioteca interactiva de tutoriales spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contiene recorridos narrados, con subtítulos de instalación y de cada flujo de trabajo de aplicaciones, en 73 lecciones con 50 voces en ocho idiomas.

Citar spaCR
~~~~~~~~~~~~

Si spaCR contribuye a su investigación, cite:

Olafsson EB, *et al.* Una cribado de imagen agrupada basada en CRISPR identifica EAF1 como un modulador *T. gondii* de subversión ESCRT.

`preimpresión de bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `archivo de software <https://doi.org/10.5281/zenodo.21343317>`_

Agradecimientos
~~~~~~~~~~~~~~~

spaCR se basa en software científico abierto, como NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch y Qt. Consulte la `atribución de los modelos de traducción <../TRANSLATION_MODELS.md>`_ para conocer los modelos utilizados en la documentación multilingüe y los catálogos de la interfaz.
