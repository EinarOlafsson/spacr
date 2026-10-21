# spaCR

<strong>Räumliche Phänotypanalyse von CRISPR-Screens.</strong>

spaCR segmentiert und vermisst einzelne Zellen in Mikroskopiebildern, integriert Phänotypen einzelner Objekte mit sequenzierungsbasierten Guide-Häufigkeiten und schätzt, welche Gene mit phänotypischen Veränderungen assoziiert sind. Ausgehend von Plattenbildern und FASTQ-Reads erzeugt es Messungen pro Objekt, trainierte Klassifikatoren, Effektschätzungen pro Guide und Gen sowie eine Rangliste der Treffer.

Die Module für Segmentierung, Messung, Annotation und Klassifizierung können auch ohne einen Sequenzierungsteil verwendet werden.

Make Masks korrigiert Segmentierungsmasken und annotiert mit dem Werkzeug <strong>Box</strong> unabhängige Rechtecke mit Klassenlabels für den YOLO-Export. Die Rechtecke behalten ihre eigenen Labels und ihren Verlauf, ohne Quellbilder oder Masken zu ändern.

Einzelheiten zu jedem Werkzeug stehen im [Funktionsleitfaden](<../../source/features.rst>).

Bilder, Masken und Bildausschnitte sind Dateien auf der Festplatte. Messungen, Annotationen und Modellvorhersagen sind Tabellen in der SQLite-Datenbank ` measurements.db ` des Projekts.

Läuft als Desktop-Anwendung oder ohne grafische Oberfläche auf einer Workstation, einem Server oder Cluster.
