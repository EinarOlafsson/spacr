|Docs| |Tutorials| =|PyPI| +|Python| ×|Tests| ≤|Qt| >|Source| ±|Issues| °|License| ≥|DOI|

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

Idiomas: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

`Información sobre los modelos de traducción <../TRANSLATION_MODELS.md>`_

**Análisis espacial del fenotipo en cribados CRISPR.**

spaCR segmenta y mide células individuales en imágenes de microscopía de alto contenido, vincula cada célula con el gRNA que recibió e indica qué genes modificaron el fenotipo. Las imágenes de placas y las lecturas FASTQ son la entrada; las mediciones por objeto, los clasificadores entrenados, los tamaños del efecto por guía y por gen y una lista ordenada de resultados son la salida.

Para los cribados CRISPR agrupados y basados en imágenes, este es el flujo de trabajo completo. Si dispone de microscopía de alto contenido sin cribado, las etapas de segmentación, medición, anotación y clasificación pueden ejecutarse por separado.

Las imágenes, máscaras, recortes, mediciones, anotaciones, predicciones, códigos de barras e identificadores de pocillo se guardan en un único proyecto SQLite, por lo que cualquier valor de un resultado puede rastrearse hasta su objeto de origen.

Ejecute spaCR como aplicación de escritorio o sin interfaz gráfica en una estación de trabajo, servidor o clúster. Ambos modos usan los mismos módulos y CUDA se utiliza automáticamente cuando el módulo lo admite.


Flujo de trabajo de un vistazo
------------------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR workflow and output organization
   :align: center

Las imágenes de microscopía (TIFF, OME-TIFF, LIF, CZI, ND2) y las lecturas de secuenciación (FASTQ) pasan por flujos complementarios de análisis de imágenes y asignación de códigos de barras. Después se analizan conjuntamente las tablas de objetos, los recortes, las anotaciones, las predicciones, las identidades de guía, los resultados de QC y los resúmenes por pocillo.


Inicio rápido
-------------

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR soporta Python **3.9 a 3.14** (excepto Python3.14.1, que la antorcha excluye). Python3.12 tiene la más amplia selección de paquetes científicos opcionales. Linux se recomienda para flujos de trabajo CUDA; macOS y Windows también son compatibles.


Detalles de instalación
-----------------------

|Release| |PyPI| =|CondaRecipe|

**(beta) Instaladores ligeros de escritorio: **

.. spacr-installer-links-begin

* `Windows 10/11: descargar SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe>`_
* `macOS 11+ (Intel y silicio Apple): descargar SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg>`_
* `64-bit Linux: descargar SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

Instaladores ligeros — no requieren conda ni una instalación de Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

El instalador descarga un tiempo de ejecución privado Python 3.12, Qt, PyTorch, spaCR y las dependencias científicas durante la instalación, por lo que no se necesita ni conda ni un Python. La construcción portátil CPU es la predeterminada, lo que evita que la instalación tire de varios gigabytes de bibliotecas CUDA sin anunciarse. Windows ofrece aceleración NVIDIA como componente opcional del instalador, Linux acepta ``--torch-backend auto``, y la rueda estándar macOS PyTorch mantiene la aceleración de Apple MPS.

Ayuda del instalador, progreso y errores siguen el idioma del sistema operativo en los diez idiomas spaCR: inglés, sueco, alemán, español, chino simplificado, portugués, hindi, coreano, islandés y francés.

En Linux, haga que el instalador descargado sea ejecutable antes de abrirlo:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

En macOS, abra el ``.pkg`` descargado. Si Gatekeeper bloquea el instalador beta actual porque no está notariado, abra **System Settings → Privacidad y seguridad**, elija **Open Daughth** for spaCR, luego ejecute el paquete de nuevo.

El instalador valida spaCR, Qt, PyTorch y la consistencia de la dependencia antes de reemplazar una instalación anterior, por lo que una actualización interrumpida deja el entorno de trabajo anterior en su lugar. Un registro de diagnóstico se mantiene como ``install.log`` dentro del directorio de instalación privado spaCR.

Aplicación de escritorio desde PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

Instalación sin interfaz gráfica o en servidor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Rama de desarrollo más reciente
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch nightly
   python -m pip install -e ".[qt]"

Entornos conda
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

Funciones opcionales
~~~~~~~~~~~~~~~~~~~~~

Instale sólo los extras que necesita su flujo de trabajo:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Qué resolución extras depende de la versión Python. En Python 3.13, ultrack limita ``spacr[all]`` y TorchCAM NumPy limita la restricción adicional ``attribution``; el paquete principal y la aplicación Qt no se ven afectados. En Python 3.14, buck está disponible a través de su extra. El convertidor PylibCZIrw CZI es opcional y no está probado; La lectura CZI basada en czifile sigue disponible.

La interfaz Tk legado todavía está instalada como ``spacr-legacy`` pero ya no se desarrolla.


Comandos de línea de comandos
-----------------------------

.. code-block:: bash

   spacr                                      # Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Establecer ``SPACR_LOG_LEVEL=DEBUG`` al solucionar problemas. Los registros de rotación se escriben en ``~/.spacr/logs/spacr.log``.


Funciones
---------

Los seis módulos más usados en los cribados
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** segmentos células, núcleos, patógenos y orgánulos con Cellpose, en imágenes 2D y en datos volumétricos o series temporales. La lista de modelos se lee desde el Cellpose instalado en lugar de codificado de forma dura, y se estima un diámetro de objeto a partir de las imágenes antes de que comience la ejecución. Las máscaras se pueden corregir a mano en el visor de capas, o enviar a napari y atrás.

**Measure** escribe características de morfología, intensidad, textura y colocalización por objeto en la base de datos del proyecto, junto con los recortes. Nuevo en 1.5.0.0: la corrección de iluminación estima el campo plano de la placa en sí y lo divide antes de que se tome cualquier característica de intensidad, lo que elimina el sesgo de posición que muestran los mapas de calor de placas como efectos de borde. Una segmentación QC banner indica en lenguaje sencillo cómo se ven las máscaras antes de que Mida se ejecute; informa, no bloquea. Un polígono dibujado restringe la medición a una región de interés.

**Anotate** muestra los recortes en una cuadrícula dirigida por teclado y escribe etiquetas directamente a SQLite. Ahora cierra el bucle de aprendizaje activo: reentrena un modelo en lo que has etiquetado sin salir de la cribado, vuelve a poner la cola en orden por incertidumbre, observa la curva de aprendizaje y consigue un veredicto de parada cuando las etiquetas más dejan de cambiar el modelo. La cobertura se reporta por clase, por pozo y por placa, y cada ronda se registra.

**Classify** entrena CNNs y transformadores PyTorch en recortes anotados, y modelos clásicos o mejorados en tablas de medición. La precisión por clase se mantiene ahora en cada época en lugar de ser descartada, y cada punto de control obtiene una tarjeta modelo registrando sus conjuntos de datos, balance de clase, regla de división y métricas ocultas. En la cribado de evaluación, una celda de matriz de confusión es una consulta: haga clic en ella para abrir esos recortes, con predicciones con confianza erróneas listadas aparte de las inciertas.

**Map Barcodes** decodifica la fila, columna y gRNA de códigos de barras de FASTQ lee, asigna identidades de guía a pozos y los une a celdas de imágenes. Los informes de Barcode QC lee por pozo, tasa de colisión y fracción no mapeada, barrendo alrededor del número de gRNAs por pozo que usted dice que espera en lugar de un umbral fijo.

**Regresión** estima guía, gen, condición y efectos de control utilizando 17 familias de modelos, incluyendo modelos mixtos, logística y probit, cuantil, beta, GLMs con varianza cuasi-binomio, lazo, cresta, red elástica, bisagra y herradura. El resultado es una lista de resultados anotados y clasificados en lugar de un volcado coeficiente.

Novedades de 1.5.0.0
~~~~~~~~~~~~~~~~~~~~

Antes de que exista una cribado, el módulo Power / Design responde a cuántas celdas y a cuántos pozos necesita, a precios de error de secuenciación y con la deserción que proviene de pozos que se han visualizado demasiado finamente. Un diseñador de experimentos expone la placa, sus controles y sus réplicas y exporta el diseño para el gasoducto. Posteriormente, un tablero QC recoge la segmentación, placa, acuerdo de anotación y fugas en un veredicto, y ComBat está disponible junto a ``center`` y ``zscore`` para la corrección por lotes.

Los resultados se exploran en lugar de exportarse y reimportarse. Un Graph Builder traza una tabla arrastrando columnas a x, y, color, tamaño y faceta. Las puertas dibujadas en un histograma o un scatter se convierten en filtros. Un explorador de características clasifica características por la manera en que separan las clases. Pequeños múltiplos, ajustes de dosis, gráficos de control y una detección atípica robusta utilizan el mismo motor de eje. Seleccionar objetos en una vista los selecciona en todos ellos, y abrir una selección trae los recortes de los objetos que provienen. Un espectador de capas acumula imágenes, etiquetas, puntos y formas, con vistas ortogonales, una cuadrícula de comparación sincronizada, y un árbol de linaje de celda a núcleo a patógeno.

Cada uno lleva un identificador de ejecución, una semilla y una política ``on_error``; Mascara, Medida, Classify y el registro de exportación AnnData lo que escribieron en un registro de artefactos, por lo que un archivo de salida conduce de nuevo a los ajustes que lo produjeron. Un módulo abre sobre lo que el paso anterior realmente escribió, las marcas de las gráficas de flujos de trabajo que las salidas están rancias, ejecutan comparaciones diffs las configuraciones, recuentos de objetos y listas de resultados de dos ejecuciones, y cada ejecución de GUI emite el script equivalente Python. Las mediciones de exportación a ``.h5ad`` para scanpy; OME-Zarr y OMERO están disponibles a través del Python API. Los métodos y resultados exportadores redactan esas dos secciones de manuscritos de una digestión estructurada de la ejecución: el modelo escribe la prosa, pero cada número viene del digest, y un borrador que contiene un número del digest no es rechazado. Cuando algo está mal con la instalación, ``spacr-doctor`` informa que spaCR se está ejecutando, si el GPU es utilizable, si Cellpose coincide con las llamadas API spaCR, y si la base de datos y los ajustes del proyecto son sólidos, con una corrección copiable en cada línea que no es un pase.

Interfaz de escritorio multilingüe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Preferencias → Language** retransmite la aplicación en ejecución al inglés, sueco, alemán, español, chino mandarín, portugués, hindi, coreano, islandés o francés sin reiniciar. La elección persiste, y las cribados se abren más tarde heredan.

Navigation, Preferences, AI and LIVE controls, module descriptions and spaCR-authored console notices follow the selected language. Worker output, logs, tracebacks, paths, database values, annotations, AI responses, measurements and saved results are never translated, so scientific output remains canonical English. Setting tooltips not yet reviewed in a language stay in English rather than becoming a mixed-language explanation. The `Guía de localización <https://einarolafsson.github.io/spacr/localization.html>`_ documents the behavior, the environment override, and the `ayuda contextual <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ that is translated with it.

Orientación de ajuste animado
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 short animations explain what 143 visual settings do to an image. Hover a setting and click **Animation** in its tooltip to play the square beside the text; click it again to fold it away. Animations are off until asked for, and can be disabled in Preferences. The `galería <https://einarolafsson.github.io/spacr/setting_animations.html>`_ shows all of them, and the `Configuración del registro de animación <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ records which setting each one belongs to.

Referencia de módulos
~~~~~~~~~~~~~~~~~~~~~

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


Datos
-----

Conjuntos de datos de referencia
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Conjunto completo de datos de microscopía: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Conjunto de datos de pruebas: cara de agarre toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Datos de secuenciación: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Análisis de potencia: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Contribuciones y soporte
------------------------

Bug reports and focused feature requests are welcome through `GitHub Cuestiones <https://github.com/EinarOlafsson/spacr/issues>`_. When reporting a failure, include the spaCR version, operating system, Python version, module settings and the relevant log excerpt. ``spacr-doctor`` collects most of that for you.

Licencia
~~~~~~~~~

The current development branch is source-available under the `Licencia PolyForm no comercial 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Commercial use requires a separate license from the copyright holder. Released versions through spaCR 1.4.9.9 remain available under the MIT License that accompanied those releases.

Tutoriales
~~~~~~~~~~

The `biblioteca interactiva de tutoriales spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contains narrated, captioned walkthroughs of installation and of each application workflow, in eight languages.

Citar spaCR
~~~~~~~~~~~~

Si spaCR contribuye a su investigación, cite:

Olafsson EB, *et al.* Una cribado de imagen agrupada basada en CRISPR identifica EAF1 como un modulador *T. gondii* de subversión ESCRT.

`preimpresión de bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `archivo de software <https://doi.org/10.5281/zenodo.21343317>`_
