# spaCR

<strong>Rumslig fenotypanalys av CRISPR-screeningar.</strong>

spaCR segmenterar och mäter enskilda celler i mikroskopibilder, integrerar fenotyper per objekt med sekvenseringshärledd guideförekomst och uppskattar vilka gener som är associerade med fenotypiska förändringar. Med plattbilder och FASTQ-läsningar som utgångspunkt producerar programmet mätningar per objekt, tränade klassificerare, effektskattningar per guide och gen samt en rangordnad träfflista.

Segmenterings-, mät-, annoterings- och klassificeringsmodulerna körs även utan en sekvensarm.

Make Masks korrigerar segmenteringsmasker och annoterar oberoende, klassmärkta rektanglar med verktyget <strong>Box</strong> för YOLO-export. Rutorna har egna etiketter och egen historik utan att ändra källbilder eller masker.

Se [funktionsguiden](<../../source/features.rst>) för varje verktyg.

Bilder, masker och bildutsnitt är filer på disk. Mätningar, annoteringar och modellprediktioner är tabeller i projektets SQLite-databas ` measurements.db `.

Körs som ett skrivbordsprogram eller utan grafiskt gränssnitt på en arbetsstation, server eller kluster.
