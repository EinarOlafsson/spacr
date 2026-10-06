from pathlib import Path
import json

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
targets = {
 'da': {
  'description': 'Tegn, rediger og eksportér separate YOLO-bokse med klasser uden at ændre billedet eller segmenteringsmasken.',
  'objective': 'Tegn og rediger separate bokse med klasser, gem deres projekt og eksportér normaliserede YOLO-etiketter.',
  'scenes': [
   'Vælg Box ved siden af Draw for at annotere separate YOLO-afgrænsningsbokse. Boksene har deres egne klasser og deres egen historik; de erstatter ikke segmenteringsmasken og ændrer ikke det optagne billede. Her demonstrerer boksene betjeningen og er ikke et biologisk træningsfacit.',
   'Klik på Add class, og indtast et klassenavn. Dette eksempel bruger demonstration. Vælg den ønskede klasse, før du tegner; hold klassenavne og deres numeriske identifikatorer ensartede i hele træningsdatasættet.',
   'Træk hen over billedet for at tegne en boks. Omridset og klassens navn viser den nye annotering. For rigtige træningsdata skal målobjektet afgrænses præcist; dette øvelsesrektangel viser kun bevægelsen.',
   'Træk inde i en eksisterende boks for at flytte den. Boksen beholder sin klasse, mens dens billedkoordinater ændres. Kontrollér placeringen, før du gemmer.',
   'Træk i et hjørne af boksen for at ændre størrelsen. Placér kanterne omkring det ønskede objekt, og kontrollér, at rektanglet bliver inden for billedet.',
   'Hold Control nede, mens du trækker, for at tegne endnu en boks inde i en eksisterende. Det opretter en separat annotering i stedet for at flytte den ydre boks. Begge bokse beholder deres egne koordinater og deres klasse.',
   'Højreklik på en boks for at slette den. Kontrollér, hvilket omrids der forsvinder, især når boksene overlapper. Fjernelse af en boks sletter ingen pixels fra segmenteringsmasken.',
   'Klik på Undo for at gendanne den slettede boks. De to uafhængige annoteringer kommer tilbage. Brug historikken til at fortryde en fejlagtig redigering, før du gemmer.',
   'Redo udfører sletningen igen. Fortryd endnu en gang for at beholde begge demonstrationsbokse til eksporten. Gennemgå de endelige annoteringer, før du skriver dem til disk.',
   'Klik på Save boxes for at gemme annoteringerne i projektfilen ved siden af billederne. Bokskoordinater og klasser gemmes adskilt fra masken. Opbevar projektfilen sammen med de tilsvarende kildebilleder.',
   'Klik på Export YOLO labels, og vælg tekstfilens destination. Hver række indeholder en klasseidentifikator og normaliserede værdier for centrum X, centrum Y, bredde og højde i forhold til hele billedet. Den tilhørende fil med klassenavne bevarer identifikatorernes tilknytning.',
   'Et billedfelt uden annoterede bokse eksporterer en tom etiketfil. Dette andet mikroskopifelt illustrerer filformatet og er ikke et biologisk negativt eksempel. Brug kun negative træningseksempler efter at have kontrolleret, at der ikke er målobjekter.',
   'Gå tilbage til det første billede. De gemte bokse indlæses med de samme klasser og koordinater, mens det optagne billede og segmenteringsmasken er uændrede. Opbevar YOLO-etiketter, klassetilknytning og kildebilleder samlet til træning.',
  ]},
 'nb': {
  'description': 'Tegn, rediger og eksporter separate YOLO-bokser med klasser uten å endre bildet eller segmenteringsmasken.',
  'objective': 'Tegn og rediger separate bokser med klasser, lagre prosjektet og eksporter normaliserte YOLO-etiketter.',
  'scenes': [
   'Velg Box ved siden av Draw for å annotere separate YOLO-avgrensningsbokser. Boksene har egne klasser og egen historikk; de erstatter ikke segmenteringsmasken og endrer ikke det innsamlede bildet. Her demonstrerer boksene betjeningen og er ikke en biologisk treningsfasit.',
   'Klikk på Add class og skriv inn et klassenavn. Dette eksemplet bruker demonstration. Velg den tiltenkte klassen før du tegner; hold klassenavn og numeriske identifikatorer konsistente i hele treningsdatasettet.',
   'Dra over bildet for å tegne en boks. Omrisset og klasseetiketten viser den nye annoteringen. For virkelige treningsdata skal målobjektet avgrenses nøyaktig; dette øvelsesrektangelet viser bare bevegelsen.',
   'Dra inne i en eksisterende boks for å flytte den. Boksen beholder klassen mens bildekoordinatene endres. Kontroller plasseringen før du lagrer.',
   'Dra et hjørne av boksen for å endre størrelsen. Plasser kantene rundt det tiltenkte objektet og kontroller at rektangelet holder seg innenfor bildet.',
   'Hold Control mens du drar for å tegne en ny boks inne i en eksisterende. Dette oppretter en separat annotering i stedet for å flytte den ytre boksen. Begge boksene beholder egne koordinater og egen klasse.',
   'Høyreklikk på en boks for å slette den. Kontroller hvilket omriss som forsvinner, særlig når boksene overlapper. Å fjerne en boks sletter ingen piksler fra segmenteringsmasken.',
   'Klikk på Undo for å gjenopprette den slettede boksen. De to uavhengige annoteringene kommer tilbake. Bruk historikken til å rette en feilaktig redigering før du lagrer.',
   'Redo utfører slettingen på nytt. Angre én gang til for å beholde begge demonstrasjonsboksene til eksporten. Gå gjennom de endelige annoteringene før du skriver dem til disk.',
   'Klikk på Save boxes for å lagre annoteringene i prosjektfilen ved siden av bildene. Bokskoordinater og klasser lagres separat fra masken. Oppbevar prosjektfilen sammen med de tilhørende kildebildene.',
   'Klikk på Export YOLO labels og velg hvor tekstfilen skal lagres. Hver rad inneholder en klasseidentifikator og normaliserte verdier for sentrum X, sentrum Y, bredde og høyde i forhold til hele bildet. Den tilhørende filen med klassenavn bevarer koblingen mellom identifikatorer og navn.',
   'Et bildefelt uten annoterte bokser eksporterer en tom etikettfil. Dette andre mikroskopifeltet illustrerer filformatet og er ikke et biologisk negativt eksempel. Bruk negative treningseksempler bare etter å ha kontrollert at ingen målobjekter finnes.',
   'Gå tilbake til det første bildet. De lagrede boksene lastes inn med de samme klassene og koordinatene, mens det innsamlede bildet og segmenteringsmasken er uendret. Oppbevar YOLO-etiketter, klassekobling og kildebilder samlet til trening.',
  ]},
 'it': {
  'description': 'Disegna, modifica ed esporta riquadri YOLO separati con le relative classi senza cambiare l’immagine o la maschera di segmentazione.',
  'objective': 'Disegnare e modificare riquadri separati con le relative classi, salvare il progetto ed esportare etichette YOLO normalizzate.',
  'scenes': [
   'Scegli Box accanto a Draw per annotare riquadri di delimitazione YOLO separati. Questi riquadri hanno classi e cronologia proprie; non sostituiscono la maschera di segmentazione né cambiano l’immagine acquisita. Qui i riquadri illustrano l’interazione, non costituiscono un riferimento biologico per l’addestramento.',
   'Fai clic su Add class e inserisci un nome di classe. Questo esempio usa demonstration. Seleziona la classe desiderata prima di disegnare; mantieni coerenti nomi di classe e identificatori numerici nell’intero dataset di addestramento.',
   'Trascina sull’immagine per disegnare un riquadro. Il contorno e l’etichetta della classe mostrano la nuova annotazione. Per dati reali di addestramento, racchiudi accuratamente l’oggetto bersaglio; questo rettangolo di prova illustra solo il gesto.',
   'Trascina all’interno di un riquadro esistente per spostarlo. Il riquadro mantiene la classe mentre cambiano le sue coordinate nell’immagine. Controlla la posizione prima di salvare.',
   'Trascina un angolo del riquadro per ridimensionarlo. Posiziona i bordi attorno all’oggetto desiderato e verifica che il rettangolo rimanga dentro l’immagine.',
   'Tieni premuto Control mentre trascini per disegnare un altro riquadro dentro uno esistente. Questo crea un’annotazione separata invece di spostare il riquadro esterno. Entrambi mantengono coordinate e classe proprie.',
   'Fai clic con il pulsante destro su un riquadro per eliminarlo. Controlla quale contorno scompare, soprattutto quando i riquadri si sovrappongono. Rimuovere un riquadro non cancella pixel dalla maschera di segmentazione.',
   'Fai clic su Undo per ripristinare il riquadro eliminato. Tornano le due annotazioni indipendenti. Usa la cronologia per correggere una modifica accidentale prima di salvare.',
   'Redo applica nuovamente l’eliminazione. Annulla ancora una volta per conservare entrambi i riquadri dimostrativi per l’esportazione. Controlla le annotazioni finali prima di scriverle su disco.',
   'Fai clic su Save boxes per memorizzare le annotazioni nel file di progetto accanto alle immagini. Coordinate e classi dei riquadri vengono salvate separatamente dalla maschera. Conserva il file di progetto con le immagini sorgente corrispondenti.',
   'Fai clic su Export YOLO labels e scegli la destinazione del file di testo. Ogni riga contiene un identificatore di classe e i valori normalizzati di centro X, centro Y, larghezza e altezza rispetto all’intera immagine. Il file associato con i nomi delle classi conserva la corrispondenza degli identificatori.',
   'Un campo senza riquadri annotati esporta un file di etichette vuoto. Questo secondo campo di microscopia illustra il formato del file, non un negativo biologico. Usa esempi negativi di addestramento solo dopo aver verificato l’assenza di oggetti bersaglio.',
   'Torna alla prima immagine. I riquadri salvati vengono ricaricati con le stesse classi e coordinate, mentre l’immagine acquisita e la maschera di segmentazione restano invariate. Conserva insieme etichette YOLO, corrispondenza delle classi e immagini sorgente per l’addestramento.',
  ]},
}
(scratch / 'yolo-tutorial-reviewed-targets-eu2.json').write_text(json.dumps(targets, ensure_ascii=False, indent=2) + '\n')
print('Prepared Danish, Norwegian and Italian YOLO additions privately', flush=True)
