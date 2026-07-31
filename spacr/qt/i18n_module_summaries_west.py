"""Western-language translations of the built-in module summaries.

The keys mirror the built-in rows in :data:`spacr.qt.app.APPS`.  Keeping this
catalog separate from the general UI catalog makes the longer, scientific
module descriptions easier for fluent speakers to review.
"""

from __future__ import annotations


_BUILTIN_APP_KEYS = (
    "mask",
    "timelapse",
    "motility",
    "measure",
    "annotate",
    "classify",
    "ml_analyze",
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
    "make_masks",
    "train_cellpose",
    "cellpose_masks",
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
        "mask": "Generera Cellpose-masker för celler, kärnor och patogener",
        "timelapse": "Segmentera och spåra objekt genom bildrutorna i en tidsserie",
        "motility": "Automatiserad motilitetsanalys: spåra hastighet och kvalitetskontrollera infektion",
        "measure": "Mät intensitets- och morfologiegenskaper för enskilda objekt",
        "annotate": "Annotera bilder av enskilda objekt i ett rutnät och spara dem i databasen",
        "classify": "Träna Torch-CNN:er/Transformer-modeller för att klassificera enskilda objekt",
        "ml_analyze": "Klassisk maskininlärning (XGBoost/random forest/…) på screeningegenskaper",
        "map_barcodes": "Koppla sekvenseringsstreckkoder till screeningdata",
        "regression": "Regressionsanalys av screeningspoäng",
        "align": "Registrera bildplattor till en sammanfogad bildyta, som skrivs stegvis så att en mosaik på 20000 × 20000 aldrig måste få plats i RAM",
        "convert": "Konvertera ND2/CZI/LIF/OME-TIFF till Yokogawa-TIFF: förhandsgranska mappningen och skapa sedan en mappningsfil tillbaka till originalen",
        "foreign": "Importera bilder, masker och en mättabell från en annan källa till ett spaCR-projekt, med kolumnerna mappade till spaCR:s struktur",
        "external_masks": "Omvandla bilder och externt genererade etikettmasker till ett uppmätt spaCR-projekt som är redo för annotering",
        "queue": "Kör flera plattor genom samma arbetsflöde i följd",
        "batch": "Köa valfria moduler, plattor och inställningar och kör dem över natten",
        "distributed_jobs": "Skicka och övervaka spaCR-körningar på SSH-arbetsstationer, Slurm eller moln-/HPC-system",
        "db_browser": "Bläddra i och exportera measurements.db utan sqlite3 CLI",
        "make_masks": "Finjustera Cellpose-modeller för ditt dataset",
        "train_cellpose": "Träna anpassade Cellpose-modeller",
        "cellpose_masks": "Generera masker med Cellpose",
        "model_compare": "Kör två Cellpose-modeller på samma synfält: jämför masker sida vid sida samt skillnader i objektantal och ARI",
        "model_zoo": "Bläddra bland, verifiera, hämta och prestandatesta Cellpose- och klassificeringsmodeller på tre av dina synfält",
        "plate_view": "Visa valfritt mätvärde som en plattvärmekarta med detektering av kanteffekter",
        "agreement": "Beräkna Cohen/Fleiss kappa mellan annoteringskolumner och granska oenigheter",
        "umap": "Generera UMAP-inbäddningar med bilder som markörer",
        "activation": "Generera aktiveringskartor",
        "train_compare": "Lägg kurvor från flera träningskörningar ovanpå varandra och jämför deras inställningsskillnader sida vid sida",
        "classifier_evaluation": "Utvärdera prediktioner på undanhållna data, nästlad CV, kalibrering, dataläckage och mått per platta",
        "run_history": "Sök bland alla körningars inställningar, filer, varningar, fel och prestanda",
        "report": "Skapa delbar HTML/PDF med ett klick: QC-resultat, figurer, statistik, inställningar och versioner",
        "analyze_plaques": "Analysera data från plackanalyser",
        "recruitment": "Analysera rekryteringsdata",
        "invasion": "Tvåfärgad utanför-/innanförfärgning: bundna respektive invaderade parasiter och invasionseffektivitet per brunn",
        "replication": "Endodyogeni: parasiter per vakuol, beräknade som replikeringshastighet per betingelse",
    },
    "de": {
        "mask": "Cellpose-Masken für Zellen, Zellkerne und Pathogene erzeugen",
        "timelapse": "Objekte über die Einzelbilder einer Zeitreihe segmentieren und verfolgen",
        "motility": "Automatisierter Motilitätsassay: Geschwindigkeit verfolgen und Infektions-QC durchführen",
        "measure": "Intensitäts- und Morphologiemerkmale einzelner Objekte messen",
        "annotate": "Bilder einzelner Objekte in einem Raster annotieren und in der Datenbank speichern",
        "classify": "Torch-CNNs/Transformer-Modelle zur Klassifizierung einzelner Objekte trainieren",
        "ml_analyze": "Klassisches ML (XGBoost/Random Forest/…) auf Screening-Merkmalen",
        "map_barcodes": "Sequenzierungs-Barcodes den Screening-Daten zuordnen",
        "regression": "Regressionsanalyse von Screening-Scores",
        "align": "Bildkacheln zu einer zusammengefügten Bildfläche registrieren; diese wird schrittweise geschrieben, sodass ein 20000 × 20000-Mosaik nie vollständig in den RAM passen muss",
        "convert": "ND2/CZI/LIF/OME-TIFF in Yokogawa-TIFFs konvertieren: Zuordnung in der Vorschau prüfen und anschließend eine Zuordnungsdatei zu den Originalen erstellen",
        "foreign": "Bilder, Masken und eine Messtabelle aus einer anderen Quelle in ein spaCR-Projekt importieren und ihre Spalten der spaCR-Struktur zuordnen",
        "external_masks": "Bilder und extern erzeugte Labelmasken in ein vermessenes spaCR-Projekt umwandeln, das zur Annotation bereit ist",
        "queue": "Mehrere Platten nacheinander mit derselben Pipeline verarbeiten",
        "batch": "Beliebige Module, Platten und Einstellungen einreihen und über Nacht ausführen",
        "distributed_jobs": "spaCR-Läufe auf SSH-Arbeitsstationen, Slurm oder Cloud-/HPC-Systemen übermitteln und überwachen",
        "db_browser": "measurements.db ohne sqlite3 CLI durchsuchen und exportieren",
        "make_masks": "Cellpose-Modelle für den eigenen Datensatz feinabstimmen",
        "train_cellpose": "Benutzerdefinierte Cellpose-Modelle trainieren",
        "cellpose_masks": "Masken mit Cellpose erzeugen",
        "model_compare": "Zwei Cellpose-Modelle auf denselben Bildfeldern ausführen: Masken nebeneinander sowie Differenzen bei Objektzahl und ARI vergleichen",
        "model_zoo": "Cellpose- und Klassifikationsmodelle durchsuchen, verifizieren, herunterladen und auf drei eigenen Bildfeldern benchmarken",
        "plate_view": "Beliebige Messwerte als Platten-Heatmap mit Erkennung von Randeffekten anzeigen",
        "agreement": "Cohen/Fleiss-Kappa zwischen Annotationsspalten berechnen und Abweichungen überprüfen",
        "umap": "UMAP-Einbettungen mit Bildern als Symbolen erzeugen",
        "activation": "Aktivierungskarten erzeugen",
        "train_compare": "Kurven mehrerer Trainingsläufe überlagern und die Unterschiede ihrer Einstellungen nebeneinander vergleichen",
        "classifier_evaluation": "Vorhersagen auf zurückgehaltenen Daten, verschachtelte CV, Kalibrierung, Datenleckage und Kennzahlen pro Platte auswerten",
        "run_history": "Einstellungen, Dateien, Warnungen, Fehler und Leistung aller Aufträge durchsuchen",
        "report": "Mit einem Klick teilbares HTML/PDF erstellen: QC-Bewertung, Abbildungen, Statistik, Einstellungen und Versionen",
        "analyze_plaques": "Daten aus Plaque-Assays analysieren",
        "recruitment": "Recruitment-Daten analysieren",
        "invasion": "Zweifarbige Außen-/Innenfärbung: angeheftete gegenüber eingedrungenen Parasiten und Invasionseffizienz pro Well",
        "replication": "Endodyogenie: Parasiten pro Vakuole, ausgewertet als Replikationsrate pro Bedingung",
    },
    "es": {
        "mask": "Generar máscaras de Cellpose para células, núcleos y patógenos",
        "timelapse": "Segmentar y rastrear objetos a lo largo de los fotogramas de una serie temporal",
        "motility": "Ensayo de motilidad automatizado: seguimiento de la velocidad y control de calidad de la infección",
        "measure": "Medir características de intensidad y morfología de objetos individuales",
        "annotate": "Anotar imágenes de objetos individuales en una cuadrícula y guardarlas en la base de datos",
        "classify": "Entrenar CNN/Transformers de Torch para clasificar objetos individuales",
        "ml_analyze": "ML clásico (XGBoost/random forest/…) sobre características del cribado",
        "map_barcodes": "Asignar códigos de barras de secuenciación a los datos del cribado",
        "regression": "Análisis de regresión de las puntuaciones del cribado",
        "align": "Registrar teselas en un único lienzo ensamblado, escrito de forma incremental para que un mosaico de 20000 × 20000 nunca tenga que caber por completo en RAM",
        "convert": "Convertir ND2/CZI/LIF/OME-TIFF a TIFF de Yokogawa: previsualizar la correspondencia y después crear un archivo que remita a los originales",
        "foreign": "Importar imágenes, máscaras y una tabla de mediciones de otra fuente a un proyecto spaCR, asignando sus columnas a la estructura de spaCR",
        "external_masks": "Convertir imágenes y máscaras de etiquetas generadas externamente en un proyecto spaCR medido y listo para la anotación",
        "queue": "Procesar varias placas consecutivamente mediante el mismo flujo de trabajo",
        "batch": "Poner en cola cualquier combinación de módulos, placas y ajustes y ejecutarla durante la noche",
        "distributed_jobs": "Enviar y supervisar ejecuciones de spaCR en estaciones de trabajo SSH, Slurm o sistemas de nube/HPC",
        "db_browser": "Explorar y exportar measurements.db sin usar sqlite3 CLI",
        "make_masks": "Ajustar con precisión modelos de Cellpose para su conjunto de datos",
        "train_cellpose": "Entrenar modelos personalizados de Cellpose",
        "cellpose_masks": "Generar máscaras con Cellpose",
        "model_compare": "Ejecutar dos modelos de Cellpose en los mismos campos: comparar las máscaras en paralelo y las diferencias de recuento de objetos y ARI",
        "model_zoo": "Explorar, verificar, descargar y evaluar modelos de Cellpose y de clasificación en tres de sus campos",
        "plate_view": "Mostrar cualquier medición como mapa de calor de la placa con detección de efectos de borde",
        "agreement": "Calcular kappa de Cohen/Fleiss entre columnas de anotación y revisar los desacuerdos",
        "umap": "Generar representaciones UMAP con imágenes como símbolos",
        "activation": "Generar mapas de activación",
        "train_compare": "Superponer las curvas de varias ejecuciones de entrenamiento y comparar en paralelo las diferencias entre sus ajustes",
        "classifier_evaluation": "Evaluar predicciones sobre datos reservados, CV anidada, calibración, fugas de datos y métricas por placa",
        "run_history": "Buscar los ajustes, archivos, advertencias, fallos y rendimiento de todas las tareas",
        "report": "Crear con un clic un HTML/PDF que se pueda compartir: dictamen de QC, figuras, estadísticas, ajustes y versiones",
        "analyze_plaques": "Analizar datos de ensayos de placas de lisis",
        "recruitment": "Analizar datos de reclutamiento",
        "invasion": "Tinción exterior/interior con dos colores: parásitos adheridos frente a invadidos y eficiencia de invasión por pocillo",
        "replication": "Endodiogenia: parásitos por vacuola, convertidos en una tasa de replicación por condición",
    },
}


def validate_module_summaries_west() -> None:
    """Raise :class:`AssertionError` if this parallel catalog is incomplete."""
    expected_languages = {"sv", "de", "es"}
    assert set(MODULE_SUMMARIES_WEST) == expected_languages
    assert len(_BUILTIN_APP_KEYS) == len(set(_BUILTIN_APP_KEYS)) == 34

    expected_keys = set(_BUILTIN_APP_KEYS)
    for language_code, summaries in MODULE_SUMMARIES_WEST.items():
        assert len(summaries) == 34, language_code
        assert set(summaries) == expected_keys, language_code
        assert all(isinstance(text, str) and text.strip() for text in summaries.values())
        assert all("http://" not in text and "https://" not in text for text in summaries.values())


validate_module_summaries_west()
