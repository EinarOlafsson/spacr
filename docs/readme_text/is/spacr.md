# spaCR

<strong>Rýmisbundin svipgerðargreining á CRISPR-skimunum.</strong>

spaCR aðgreinir og mælir stakar frumur í smásjármyndum, samþættir svipgerðir einstakra viðfanga við magn leiðarsameinda sem fæst úr raðgreiningu og metur hvaða gen tengjast svipgerðarbreytingum. Út frá plötumyndum og FASTQ-röðum býr það til mælingar fyrir hvert viðfang, þjálfaða flokkara, áhrifamat fyrir hverja leiðarsameind og hvert gen og forgangsraðaðan lista yfir niðurstöður.

Segmingu, mæling, notkun og flokksmiðju mótmælur virkar einnig án sekkunararms.

Make Masks leiðréttir skiptingargrímur og merkir sjálfstæða rétthyrninga með flokkamerkjum með <strong>Box</strong> verkfærinu fyrir YOLO-útflutning. Rammarnir halda eigin merkjum og sögu án þess að breyta upprunamyndum eða grímum.

Hvert verkfæri er útskýrt í [eiginleikahandbókinni](<../../source/features.rst>).

Myndir, grímur og myndúrklippur eru skrár á diski. Mælingar, merkingar og spár líkana eru töflur í SQLite-gagnagrunni verkefnisins, ` measurements.db `.

Keyrist sem skjáborðsforrit eða án grafísks viðmóts á vinnustöð, þjóni eða reikniklasa.
