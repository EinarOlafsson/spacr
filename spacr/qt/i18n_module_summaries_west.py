"""Western-language translations of the built-in module summaries.

The keys mirror the built-in rows in :data:`spacr.qt.app.APPS`.  Keeping this
catalog separate from the general UI catalog makes the longer, scientific
module descriptions easier for fluent speakers to review.
"""

from __future__ import annotations

#: TRAIN CELLPOSE IS NOT HERE EITHER, for the same event and the other half
#: of the same reason. The merge did not delete its row, it REWROTE it: the
#: tile went from "Train custom Cellpose models" to "Fine-tune a Cellpose
#: model on your own labelled fields, then segment a folder of images with it
#: or with a stock model", and the nine reviewed sentences still described the
#: training half alone. ``module_summary`` had already stopped using them --
#: the source hash has not matched since -- so removing them changes nothing
#: a user sees and stops the table asserting a review that no longer applies.
#: Translating the new sentence puts the row back.
#:
#: CELLPOSE MASKS IS NOT HERE. Its row went when Train Cellpose and Cellpose
#: Masks became one Cellpose Workbench page, and the applying half is that
#: page's Apply tab -- a tab carries its label, not a one-line summary. With
#: no English sentence left for it anywhere, the reviewed translations were
#: bound to a source that no longer exists, which is the exact condition
#: `REVIEWED_SOURCE_HASHES` exists to detect: a fluent translation of
#: something the app no longer says. Give the module a row or a fold button
#: again and its sentence comes back with it, reviewed against that sentence.
_BUILTIN_APP_KEYS = (
    "mask",
    "timelapse",
    "motility",
    "measure",
    "annotate",
    "classify_merged",
    "map_barcodes",
    "regression",
    "align",
    "convert",
    "foreign",
    "external_masks",
    "queue",
    "batch",
    "distributed_jobs",
    "db_browser",
    "model_compare",
    "model_zoo",
    "plate_view",
    "agreement",
    "umap",
    "activation",
    "train_compare",
    "classifier_evaluation",
    "run_history",
    "report",
    "analyze_plaques",
    "recruitment",
    "invasion",
    "replication",
)


#: Swedish, German and Spanish one-line summaries for every built-in spaCR
#: module.  Product names, file formats and established model/metric names are
#: intentionally left unchanged so they match the rest of the user interface.
MODULE_SUMMARIES_WEST: dict[str, dict[str, str]] = {
    "sv": {
        "mask": "Generera segmenteringsmasker för celler, kärnor, patogener och organeller från mikroskopibilder med Cellpose och andra stödda metoder",
        "timelapse": "Segmentera och spåra objekt genom bildrutorna i en tidsserie",
        "motility": "Kvantifiera spårhastighet och rakhet och stratifiera resultaten efter infektionsstatus",
        "measure": "Kvantifiera intensitets- och morfologiegenskaper för varje objekt",
        "annotate": "Tilldela annoteringar till bilder av enskilda objekt och lagra dem i projektdatabasen",
        "classify_merged": "Träna klassificerare på bildutsnitt med PyTorch eller på uppmätta egenskaper med gradient boosting",
        "map_barcodes": "Koppla sekvenseringsstreckkoder till screeningdata",
        "regression": "Regressionsanalys av screeningspoäng",
        "align": "Registrera och sammanfoga bildplattor till en mosaik som skrivs stegvis med begränsad minnesanvändning",
        "convert": "Konvertera ND2-, CZI-, LIF- och OME-TIFF-bilder till Yokogawa-TIFF-layout och registrera kopplingarna till källfilerna",
        "foreign": "Importera externa bilder, masker och en mättabell till ett spaCR-projekt och mappa källkolumnerna till spaCR-fält",
        "external_masks": "Importera bilder och externt genererade etikettmasker som ett uppmätt spaCR-projekt som är redo för annotering",
        "queue": "Kör samma bearbetningspipeline på flera plattor",
        "batch": "Köa moduler, plattor och inställningar för obevakad sekventiell körning",
        "distributed_jobs": "Skicka och övervaka spaCR-körningar på SSH-arbetsstationer, Slurm eller moln-/HPC-system",
        "db_browser": "Bläddra i, filtrera och exportera tabeller från measurements.db",
        "model_compare": "Jämför två Cellpose-modeller på samma synfält med masker sida vid sida, skillnader i objektantal och justerat Rand-index (ARI)",
        "model_zoo": "Bläddra bland, verifiera, hämta och prestandatesta Cellpose- och klassificeringsmodeller på utvalda synfält",
        "plate_view": "Visualisera mätvärden som plattvärmekartor och detektera kanteffekter",
        "agreement": "Beräkna Cohens eller Fleiss κ mellan annoteringskolumner och granska bildutsnitt med avvikande annoteringar",
        "umap": "Visualisera UMAP-inbäddningar med bilder som markörer",
        "activation": "Generera klassaktiveringskartor för bildklassificerarens prediktioner",
        "train_compare": "Jämför träningskurvor och inställningar mellan flera körningar",
        "classifier_evaluation": "Utvärdera prediktioner på undanhållna data, nästlad CV, kalibrering, dataläckage och mått per platta",
        "run_history": "Sök i körningsinställningar, utdata, varningar, fel och prestandamått",
        "report": "Generera delbara HTML- eller PDF-rapporter med QC-resultat, figurer, statistik, inställningar och programvaruversioner",
        "analyze_plaques": "Kvantifiera mätvärden från plackanalyser",
        "recruitment": "Kvantifiera mätvärden för molekylär rekrytering",
        "invasion": "Kvantifiera bundna och invaderade parasiter med tvåfärgad differentialfärgning och beräkna invasionseffektivitet per brunn",
        "replication": "Kvantifiera parasiter per vakuol och beräkna replikeringshastigheter per betingelse",
    },
    "de": {
        "mask": "Segmentierungsmasken für Zellen, Zellkerne, Pathogene und Organellen aus Mikroskopiebildern mit Cellpose und unterstützten Alternativen erzeugen",
        "timelapse": "Objekte über die Einzelbilder einer Zeitreihe segmentieren und verfolgen",
        "motility": "Spurgeschwindigkeit und Geradlinigkeit quantifizieren und Ergebnisse nach Infektionsstatus stratifizieren",
        "measure": "Intensitäts- und Morphologiemerkmale pro Objekt quantifizieren",
        "annotate": "Bilder einzelner Objekte annotieren und die Annotationen in der Projektdatenbank speichern",
        "classify_merged": "Klassifikatoren mit PyTorch auf Bildausschnitten oder mit Gradient Boosting auf gemessenen Merkmalen trainieren",
        "map_barcodes": "Sequenzierungs-Barcodes den Screening-Daten zuordnen",
        "regression": "Regressionsanalyse von Screening-Scores",
        "align": "Bildkacheln registrieren und mit begrenztem Speicherverbrauch zu einem schrittweise geschriebenen Mosaik zusammenfügen",
        "convert": "ND2-, CZI-, LIF- und OME-TIFF-Bilder in ein Yokogawa-TIFF-Layout konvertieren und die Zuordnungen zu den Quelldateien protokollieren",
        "foreign": "Externe Bilder, Masken und eine Messtabelle in ein spaCR-Projekt importieren und Quellspalten den spaCR-Feldern zuordnen",
        "external_masks": "Bilder und extern erzeugte Labelmasken als vermessenes, annotationsbereites spaCR-Projekt importieren",
        "queue": "Dieselbe Verarbeitungspipeline auf mehreren Platten ausführen",
        "batch": "Module, Platten und Einstellungen zur unbeaufsichtigten sequenziellen Ausführung einreihen",
        "distributed_jobs": "spaCR-Läufe auf SSH-Arbeitsstationen, Slurm oder Cloud-/HPC-Systemen übermitteln und überwachen",
        "db_browser": "Tabellen aus measurements.db durchsuchen, filtern und exportieren",
        "model_compare": "Zwei Cellpose-Modelle auf denselben Bildfeldern mithilfe nebeneinander dargestellter Masken, Unterschieden in der Objektzahl und des adjustierten Rand-Index (ARI) vergleichen",
        "model_zoo": "Cellpose- und Klassifikationsmodelle durchsuchen, verifizieren, herunterladen und auf ausgewählten Bildfeldern benchmarken",
        "plate_view": "Messwerte als Platten-Heatmaps visualisieren und Randeffekte erkennen",
        "agreement": "Cohens oder Fleiss’ κ über Annotationsspalten berechnen und Bildausschnitte mit abweichenden Annotationen überprüfen",
        "umap": "UMAP-Einbettungen mit Bildern als Symbolen visualisieren",
        "activation": "Klassenaktivierungskarten für Vorhersagen eines Bildklassifikators erzeugen",
        "train_compare": "Trainingskurven und Einstellungen mehrerer Läufe vergleichen",
        "classifier_evaluation": "Vorhersagen auf zurückgehaltenen Daten, verschachtelte CV, Kalibrierung, Datenleckage und Kennzahlen pro Platte auswerten",
        "run_history": "Laufeinstellungen, Ausgaben, Warnungen, Fehler und Leistungskennzahlen durchsuchen",
        "report": "Teilbare HTML- oder PDF-Berichte mit QC-Ergebnissen, Abbildungen, Statistiken, Einstellungen und Softwareversionen erzeugen",
        "analyze_plaques": "Messwerte aus Plaque-Assays quantifizieren",
        "recruitment": "Messwerte der molekularen Rekrutierung quantifizieren",
        "invasion": "Angeheftete und eingedrungene Parasiten mit einer zweifarbigen Differenzialfärbung quantifizieren und die Invasionseffizienz pro Well berechnen",
        "replication": "Parasiten pro Vakuole quantifizieren und Replikationsraten nach Bedingung berechnen",
    },
    "es": {
        "mask": "Generar máscaras de segmentación para células, núcleos, patógenos y orgánulos a partir de imágenes microscópicas mediante Cellpose y alternativas compatibles",
        "timelapse": "Segmentar y rastrear objetos a lo largo de los fotogramas de una serie temporal",
        "motility": "Cuantificar la velocidad y la rectitud de las trayectorias y estratificar los resultados por estado de infección",
        "measure": "Cuantificar las características de intensidad y morfología de cada objeto",
        "annotate": "Asignar anotaciones a imágenes de objetos individuales y guardarlas en la base de datos del proyecto",
        "classify_merged": "Entrenar clasificadores con PyTorch sobre recortes de imagen o con gradient boosting sobre características medidas",
        "map_barcodes": "Asignar códigos de barras de secuenciación a los datos del cribado",
        "regression": "Análisis de regresión de las puntuaciones del cribado",
        "align": "Registrar y ensamblar teselas de imagen en un mosaico escrito de forma incremental con un uso de memoria acotado",
        "convert": "Convertir imágenes ND2, CZI, LIF y OME-TIFF al formato TIFF de Yokogawa y registrar sus correspondencias con los archivos de origen",
        "foreign": "Importar imágenes, máscaras y una tabla de mediciones externas a un proyecto spaCR, asignando las columnas de origen a los campos de spaCR",
        "external_masks": "Importar imágenes y máscaras de etiquetas externas como un proyecto spaCR medido y listo para la anotación",
        "queue": "Ejecutar la misma secuencia de procesamiento en varias placas",
        "batch": "Poner en cola módulos, placas y ajustes para su ejecución secuencial sin supervisión",
        "distributed_jobs": "Enviar y supervisar ejecuciones de spaCR en estaciones de trabajo SSH, Slurm o sistemas de nube/HPC",
        "db_browser": "Explorar, filtrar y exportar tablas de measurements.db",
        "model_compare": "Comparar dos modelos de Cellpose en los mismos campos mediante máscaras en paralelo, diferencias en el número de objetos y el índice Rand ajustado (ARI)",
        "model_zoo": "Explorar, verificar, descargar y evaluar modelos de Cellpose y de clasificación en campos seleccionados",
        "plate_view": "Visualizar mediciones como mapas de calor de placas y detectar efectos de borde",
        "agreement": "Calcular κ de Cohen o de Fleiss entre columnas de anotación y revisar los recortes con anotaciones discordantes",
        "umap": "Visualizar representaciones UMAP con imágenes como símbolos",
        "activation": "Generar mapas de activación de clase para las predicciones de un clasificador de imágenes",
        "train_compare": "Comparar las curvas y los ajustes de varias ejecuciones de entrenamiento",
        "classifier_evaluation": "Evaluar predicciones sobre datos reservados, CV anidada, calibración, fugas de datos y métricas por placa",
        "run_history": "Buscar ajustes, resultados, advertencias, fallos y métricas de rendimiento de las ejecuciones",
        "report": "Generar informes HTML o PDF compartibles con resultados de QC, figuras, estadísticas, ajustes y versiones del software",
        "analyze_plaques": "Cuantificar las mediciones de ensayos de placas de lisis",
        "recruitment": "Cuantificar las mediciones de reclutamiento molecular",
        "invasion": "Cuantificar los parásitos adheridos e invadidos mediante tinción diferencial de dos colores y calcular la eficiencia de invasión por pocillo",
        "replication": "Cuantificar los parásitos por vacuola y calcular las tasas de replicación por condición",
    },
}


def validate_module_summaries_west() -> None:
    """Raise :class:`AssertionError` if this parallel catalog is incomplete."""
    expected_languages = {"sv", "de", "es"}
    assert set(MODULE_SUMMARIES_WEST) == expected_languages
    assert len(_BUILTIN_APP_KEYS) == len(set(_BUILTIN_APP_KEYS)) == 30

    expected_keys = set(_BUILTIN_APP_KEYS)
    for language_code, summaries in MODULE_SUMMARIES_WEST.items():
        assert len(summaries) == 30, language_code
        assert set(summaries) == expected_keys, language_code
        assert all(isinstance(text, str) and text.strip() for text in summaries.values())
        assert all("http://" not in text and "https://" not in text for text in summaries.values())


validate_module_summaries_west()
