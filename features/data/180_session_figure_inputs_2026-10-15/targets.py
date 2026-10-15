TARGETS = {
'sv': [
'Registrera figurernas filer i ordning, titlar, körningsavsnitt och den valda platsen.',
'PNG-förhandsvisningar och tillhörande PDF-sidor som ägs av kön och ryms inom arbetsytans aktuella storleksgräns per fil markeras för medtagning även i referensläge. Externa och större filer förblir referenser. Python-figurer i minnet och deras cachefiler i pickle-format inkluderas aldrig.',
'vanligt tillstånd för :meth:`apply_workspace_state`, inklusive arbetsytans uttryckliga förteckning över filer som ska tas med.',
'Återställ sparade förhandsvisningar utan att köra eller återskapa figurer.',
'Tillgängliga PNG- och PDF-filer kopieras till nytt lagringsutrymme som ägs av kön; arbetsytepaketet och originalfilerna lämnas orörda. Saknade förhandsvisningar behåller sina platser för den vanliga visningen av en otillgänglig bild. Arbetsytans återställningsrapport anger saknade filer separat. Ogiltigt tillstånd eller ogiltiga bildformat lämnar kön oförändrad.',
'Tillstånd från :meth:`workspace_state`, där arbetsytans återställare redan har löst sökvägarna till medtagna filer.',
'om tillståndet accepterades. Filer som överskrider den aktuella kopieringsgränsen per fil utlöser ett fel som återställaren rapporterar.',
'Sparade figurfiler överskrider arbetsytans kopieringsgräns.'
],
'de': [
'Dateien der Abbildungen in ihrer Reihenfolge, Titel, Laufabschnitte und den ausgewählten Platz erfassen.',
'Der Warteschlange gehörende PNG-Vorschauen und zugehörige PDF-Seiten innerhalb der aktuellen Dateigrößengrenze des Arbeitsbereichs werden auch im Referenzmodus zur Mitnahme markiert. Externe und größere Dateien bleiben Referenzen. Python-Abbildungsobjekte im Speicher und ihre Pickle-Cachedateien werden niemals aufgenommen.',
'einfache Zustandsdaten für :meth:`apply_workspace_state`, einschließlich der ausdrücklichen Liste mitzuführender Dateien des Arbeitsbereichs.',
'Gespeicherte Vorschauen wiederherstellen, ohne Abbildungen auszuführen oder neu zu erzeugen.',
'Verfügbare PNG- und PDF-Dateien werden in einen neuen, der Warteschlange gehörenden Speicherbereich kopiert; das Arbeitsbereichspaket und die Originaldateien bleiben unberührt. Fehlende Vorschauen behalten ihre Plätze für die übliche Anzeige eines nicht verfügbaren Bildes. Der Wiederherstellungsbericht des Arbeitsbereichs nennt fehlende Dateien separat. Ungültige Zustandsdaten oder Bildformate lassen diese Warteschlange unverändert.',
'Zustandsdaten aus :meth:`workspace_state`, deren Pfade zu mitgeführten Dateien bereits von der Arbeitsbereichswiederherstellung aufgelöst wurden.',
'ob die Zustandsdaten angenommen wurden. Dateien über der aktuellen Kopiergrenze pro Datei lösen einen Fehler aus, den die Wiederherstellung meldet.',
'Gespeicherte Abbildungsdateien überschreiten die Kopiergrenze des Arbeitsbereichs.'
],
'es': [
'Registrar los archivos de figuras en orden, sus títulos, las secciones de ejecución y la posición seleccionada.',
'Las vistas previas PNG y las páginas PDF asociadas que pertenecen a la cola y no superan el límite actual por archivo del espacio de trabajo se marcan para incluirlas incluso en modo de referencia. Los archivos externos y los de mayor tamaño permanecen como referencias. Nunca se incluyen los objetos de figura de Python en memoria ni sus archivos de caché serializados con pickle.',
'datos de estado simples para :meth:`apply_workspace_state`, incluido el inventario explícito de archivos que se incorporan al espacio de trabajo.',
'Restaurar las vistas previas guardadas sin ejecutar ni reconstruir figuras.',
'Los archivos PNG y PDF disponibles se copian a un nuevo almacenamiento que pertenece a la cola; el paquete del espacio de trabajo y los archivos originales permanecen intactos. Las vistas previas ausentes conservan sus posiciones para la visualización habitual de una imagen no disponible. El informe de restauración del espacio de trabajo identifica por separado los archivos ausentes. Un estado o un formato de imagen no válidos dejan esta cola sin cambios.',
'estado de :meth:`workspace_state`, con las rutas de los archivos incluidos ya resueltas por el restaurador del espacio de trabajo.',
'si se aceptó el estado. Los archivos que superan el límite actual de copia por archivo provocan un error que comunica el restaurador.',
'Los archivos de figuras guardados superan el límite de copia del espacio de trabajo.'
],
}
TARGETS.update({
'fr': [
'Enregistrer les fichiers des figures dans l’ordre, leurs titres, les sections d’exécution et l’emplacement sélectionné.',
'Les aperçus PNG et les pages PDF associées appartenant à la file et respectant la limite actuelle par fichier de l’espace de travail sont marqués pour être inclus, même en mode référence. Les fichiers externes et les fichiers plus volumineux restent des références. Les objets de figure Python en mémoire et leurs fichiers de cache sérialisés avec pickle ne sont jamais inclus.',
'des données d’état simples pour :meth:`apply_workspace_state`, comprenant l’inventaire explicite des fichiers à emporter avec l’espace de travail.',
'Restaurer les aperçus enregistrés sans exécuter ni reconstruire les figures.',
'Les fichiers PNG et PDF disponibles sont copiés dans un nouvel espace de stockage appartenant à la file ; le paquet de l’espace de travail et les fichiers originaux restent intacts. Les aperçus manquants conservent leur emplacement pour l’affichage habituel d’une image indisponible. Le rapport de restauration de l’espace de travail signale séparément les fichiers manquants. Un état ou un format d’image invalide laisse cette file inchangée.',
'l’état provenant de :meth:`workspace_state`, dont les chemins des fichiers inclus ont déjà été résolus par le restaurateur de l’espace de travail.',
'si l’état a été accepté. Les fichiers dépassant la limite actuelle de copie par fichier déclenchent une erreur signalée par le restaurateur.',
'Les fichiers de figures enregistrés dépassent la limite de copie de l’espace de travail.'
],
'pt': [
'Registrar os arquivos de figuras em ordem, os títulos, as seções de execução e a posição selecionada.',
'As prévias PNG e as páginas PDF associadas que pertencem à fila e respeitam o limite atual por arquivo do espaço de trabalho são marcadas para inclusão mesmo no modo de referência. Arquivos externos e arquivos maiores permanecem como referências. Objetos de figura Python em memória e seus arquivos de cache serializados com pickle nunca são incluídos.',
'dados de estado simples para :meth:`apply_workspace_state`, incluindo o inventário explícito de arquivos a incluir no espaço de trabalho.',
'Restaurar as prévias salvas sem executar nem reconstruir figuras.',
'Os arquivos PNG e PDF disponíveis são copiados para um novo armazenamento pertencente à fila; o pacote do espaço de trabalho e os arquivos originais permanecem intactos. Prévias ausentes mantêm suas posições para a exibição habitual de uma imagem indisponível. O relatório de restauração do espaço de trabalho identifica separadamente os arquivos ausentes. Um estado ou formato de imagem inválido deixa esta fila inalterada.',
'estado de :meth:`workspace_state`, com os caminhos dos arquivos incluídos já resolvidos pelo restaurador do espaço de trabalho.',
'se o estado foi aceito. Arquivos que excedem o limite atual de cópia por arquivo geram um erro relatado pelo restaurador.',
'Os arquivos de figuras salvos excedem o limite de cópia do espaço de trabalho.'
],
'is': [
'Skrá myndritaskrár í röð, titla, keyrsluhluta og valið sæti.',
'PNG-forskoðanir og tilheyrandi PDF-síður í eigu biðraðarinnar, sem eru innan núverandi stærðarmarka vinnusvæðisins fyrir hverja skrá, eru merktar til að fylgja með jafnvel í tilvísunarham. Ytri skrár og stærri skrár verða áfram tilvísanir. Python-myndritahlutir í minni og skyndiminnisskrár þeirra á pickle-sniði eru aldrei teknar með.',
'einföld stöðugögn fyrir :meth:`apply_workspace_state`, þar á meðal skýr skrá yfir þær skrár sem fylgja vinnusvæðinu.',
'Endurheimta vistaðar forskoðanir án þess að keyra eða endurgera myndrit.',
'Tiltækar PNG- og PDF-skrár eru afritaðar í nýtt geymslusvæði í eigu biðraðarinnar; vinnusvæðispakkinn og upprunalegu skrárnar eru ósnert. Forskoðanir sem vantar halda sætum sínum fyrir hefðbundna birtingu þegar mynd er ekki tiltæk. Endurheimtarskýrsla vinnusvæðisins tilgreinir skrár sem vantar sérstaklega. Ógild stöðugögn eða ógild myndsnið skilja þessa biðröð eftir óbreytta.',
'staða frá :meth:`workspace_state`, þar sem endurheimtir vinnusvæðisins hefur þegar leyst slóðir að skrám sem fylgja með.',
'hvort staðan var samþykkt. Skrár sem fara yfir núverandi stærðarmörk afritunar fyrir hverja skrá valda villu sem endurheimtirinn tilkynnir.',
'Vistaðar myndritaskrár fara yfir afritunarmörk vinnusvæðisins.'
],
})
TARGETS.update({
'zh_CN': [
'记录按顺序排列的图文件、标题、运行分区和选中的位置。',
'属于队列的 PNG 预览及其配套 PDF 页面，若未超过工作区当前的单文件大小限额，即使在引用模式下也会标记为随包携带。外部文件和较大文件仍保留为引用。内存中的 Python 图对象及其 pickle 缓存文件绝不会包含在内。',
'供 :meth:`apply_workspace_state` 使用的普通状态数据，包括工作区明确列出的随包携带文件清单。',
'恢复保存的预览，而不执行或重建图对象。',
'可用的 PNG 和 PDF 文件会复制到队列所有的新存储位置；工作区包和原始文件保持不变。缺失的预览保留其位置，以显示通常的图像不可用提示。工作区恢复报告会单独列出缺失的文件。无效的状态或图像格式不会改变此队列。',
'来自 :meth:`workspace_state` 的状态，其中随包携带文件的路径已由工作区恢复程序解析。',
'是否接受了该状态。超过当前单文件复制限额的文件会引发错误，由恢复程序报告。',
'保存的图文件超过了工作区的复制限额。'
],
'hi': [
'चित्र फ़ाइलों का क्रम, शीर्षक, रन के अनुभाग और चुना गया स्थान दर्ज करें।',
'कतार के स्वामित्व वाले PNG पूर्वावलोकन और संबद्ध PDF पृष्ठ, जो कार्यक्षेत्र की वर्तमान प्रति-फ़ाइल आकार सीमा के भीतर हैं, संदर्भ मोड में भी साथ ले जाने के लिए चिह्नित किए जाते हैं। बाहरी और बड़ी फ़ाइलें संदर्भ के रूप में बनी रहती हैं। स्मृति में मौजूद Python चित्र ऑब्जेक्ट और उनकी pickle प्रारूप वाली कैश फ़ाइलें कभी शामिल नहीं की जातीं।',
':meth:`apply_workspace_state` के लिए साधारण स्थिति डेटा, जिसमें कार्यक्षेत्र के साथ ले जाने वाली फ़ाइलों की स्पष्ट सूची शामिल है।',
'चित्रों को निष्पादित या पुनर्निर्मित किए बिना सहेजे गए पूर्वावलोकन पुनर्स्थापित करें।',
'उपलब्ध PNG और PDF फ़ाइलें कतार के स्वामित्व वाले नए संग्रहण में कॉपी की जाती हैं; कार्यक्षेत्र बंडल और मूल फ़ाइलें अछूती रहती हैं। अनुपस्थित पूर्वावलोकन अपने स्थान बनाए रखते हैं, जहाँ चित्र उपलब्ध न होने का सामान्य संकेत दिखता है। कार्यक्षेत्र पुनर्स्थापन रिपोर्ट अनुपस्थित फ़ाइलों को अलग से बताती है। अमान्य स्थिति या अमान्य चित्र प्रारूप इस कतार को अपरिवर्तित छोड़ते हैं।',
':meth:`workspace_state` से मिली स्थिति, जिसमें साथ ले जाई गई फ़ाइलों के पथ कार्यक्षेत्र पुनर्स्थापक पहले ही निर्धारित कर चुका है।',
'क्या स्थिति स्वीकार की गई। वर्तमान प्रति-फ़ाइल कॉपी सीमा से बड़ी फ़ाइलें एक त्रुटि उत्पन्न करती हैं, जिसे पुनर्स्थापक रिपोर्ट करता है।',
'सहेजी गई चित्र फ़ाइलें कार्यक्षेत्र की कॉपी सीमा से अधिक हैं।'
],
'ko': [
'그림 파일의 순서, 제목, 실행 구역과 선택된 위치를 기록합니다.',
'큐가 소유한 PNG 미리보기와 짝을 이루는 PDF 페이지 중 현재 작업 공간의 파일별 크기 한도 이내인 파일은 참조 모드에서도 함께 보관하도록 표시됩니다. 외부 파일과 더 큰 파일은 참조로 남습니다. 메모리에 있는 Python 그림 객체와 해당 pickle 형식 캐시 파일은 절대 포함되지 않습니다.',
':meth:`apply_workspace_state` 에 전달할 일반 상태 데이터이며, 작업 공간에 함께 보관할 파일의 명시적 목록을 포함합니다.',
'그림을 실행하거나 재구성하지 않고 저장된 미리보기를 복원합니다.',
'사용 가능한 PNG 및 PDF 파일을 큐가 소유한 새 저장 공간에 복사하며, 작업 공간 묶음과 원본 파일은 그대로 둡니다. 누락된 미리보기는 기존 위치를 유지하여 일반적인 이미지 사용 불가 표시를 보여 줍니다. 작업 공간 복원 보고서는 누락된 파일을 별도로 알려 줍니다. 잘못된 상태 또는 이미지 형식은 이 큐를 변경하지 않습니다.',
':meth:`workspace_state` 에서 가져온 상태이며, 함께 보관된 파일의 경로는 작업 공간 복원기가 이미 해석한 상태입니다.',
'상태를 수락했는지 여부입니다. 현재 파일별 복사 한도를 초과하는 파일은 오류를 발생시키며, 복원기가 이를 보고합니다.',
'저장된 그림 파일이 작업 공간 복사 한도를 초과합니다.'
],
})
