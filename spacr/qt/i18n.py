"""Runtime localization for the spaCR Qt application.

The application historically embedded English text directly in its widgets.
Replacing every call site at once would make localization brittle, so this
module provides two complementary layers:

* :func:`tr` translates text at construction time and always falls back to
  the original English source string.
* :func:`retranslate_widget_tree` safely updates already-created static Qt
  labels, buttons, menus, tabs, combo-box choices and accessibility text.
  Original English strings are retained as dynamic Qt properties, allowing a
  window to switch from Swedish to Korean (for example) without translating a
  translation. Editable values and user-entered paths are never touched.

Ten languages ship without optional dependencies or network access: English,
Swedish, German, Spanish, Simplified Chinese (Mandarin), Portuguese, Hindi,
Korean, Icelandic and French. Catalog entries cover application navigation,
every registered module, Preferences, common actions and common settings
terminology. Uncatalogued scientific or third-party terms remain in English
rather than being guessed.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class Language:
    """One selectable UI language.

    :param code: stable persisted language code.
    :param native_name: language name written in that language.
    :param english_name: language name written in English.
    """

    code: str
    native_name: str
    english_name: str

    @property
    def display_name(self) -> str:
        """Return an unambiguous native/English selector label."""
        if self.native_name == self.english_name:
            return self.native_name
        return f"{self.native_name} — {self.english_name}"


LANGUAGES = (
    Language("en", "English", "English"),
    Language("sv", "Svenska", "Swedish"),
    Language("de", "Deutsch", "German"),
    Language("es", "Español", "Spanish"),
    Language("zh_CN", "简体中文", "Mandarin Chinese"),
    Language("pt", "Português", "Portuguese"),
    Language("hi", "हिन्दी", "Hindi"),
    Language("ko", "한국어", "Korean"),
    Language("is", "Íslenska", "Icelandic"),
    Language("fr", "Français", "French"),
)

LANGUAGE_BY_CODE = {language.code: language for language in LANGUAGES}
VALID_LANGUAGE_CODES = tuple(LANGUAGE_BY_CODE)
DEFAULT_LANGUAGE = "en"
ENV_LANGUAGE = "SPACR_LANGUAGE"

# The compact row format makes catalog review practical: every row is ordered
# sv, de, es, zh_CN, pt, hi, ko, is, fr. English is the source key.
_TRANSLATED_CODES = VALID_LANGUAGE_CODES[1:]


def _row(*values: str) -> tuple[str, ...]:
    """Validate and return one parallel translation row."""
    if len(values) != len(_TRANSLATED_CODES):
        raise ValueError(
            f"translation row has {len(values)} values; "
            f"expected {len(_TRANSLATED_CODES)}"
        )
    return tuple(values)


_ROWS: Dict[str, tuple[str, ...]] = {
    # Navigation and common actions.
    "Home": _row(
        "Hem", "Startseite", "Inicio", "主页", "Início",
        "मुखपृष्ठ", "홈", "Heim", "Accueil"),
    "All apps": _row(
        "Alla appar", "Alle Module", "Todas las aplicaciones", "所有应用",
        "Todos os aplicativos", "सभी ऐप", "모든 앱", "Öll forrit",
        "Toutes les applications"),
    "Preferences…": _row(
        "Inställningar…", "Einstellungen…", "Preferencias…", "首选项…",
        "Preferências…", "प्राथमिकताएँ…", "환경설정…", "Stillingar…",
        "Préférences…"),
    "Quit": _row(
        "Avsluta", "Beenden", "Salir", "退出", "Sair",
        "बाहर निकलें", "종료", "Hætta", "Quitter"),
    "Demos": _row(
        "Demon", "Demos", "Demostraciones", "演示", "Demonstrações",
        "डेमो", "데모", "Sýnishorn", "Démonstrations"),
    "Help": _row(
        "Hjälp", "Hilfe", "Ayuda", "帮助", "Ajuda",
        "सहायता", "도움말", "Hjálp", "Aide"),
    # The "(web)" was dropped from both labels; the keys move with them or
    # nine languages lose the row. Each translation loses its own bracket
    # rather than keeping a parenthesis the English no longer has.
    "Tutorial": _row(
        "Handledning", "Tutorial", "Tutorial",
        "教程", "Tutorial", "ट्यूटोरियल",
        "튜토리얼", "Kennsla", "Tutoriel"),
    "Documentation": _row(
        "Dokumentation", "Dokumentation",
        "Documentación", "文档", "Documentação",
        "दस्तावेज़", "문서", "Skjölun",
        "Documentation"),
    "About spaCR": _row(
        "Om spaCR", "Über spaCR", "Acerca de spaCR", "关于 spaCR",
        "Sobre o spaCR", "spaCR के बारे में", "spaCR 정보", "Um spaCR",
        "À propos de spaCR"),
    # Keyboard-shortcut map. The bindings themselves remain platform-native
    # key identifiers; only their labels, categories and scopes are copy.
    "Background": _row(
        "Bakgrund", "Hintergrund", "Fondo", "背景", "Plano de fundo",
        "पृष्ठभूमि", "배경", "Bakgrunnur", "Arrière-plan"),
    "Blank the background": _row(
        "Töm bakgrunden", "Hintergrund leeren", "Vaciar el fondo", "清空背景",
        "Limpar o plano de fundo", "पृष्ठभूमि खाली करें", "배경 비우기",
        "Tæma bakgrunninn", "Effacer l’arrière-plan"),
    "Brush": _row(
        "Pensel", "Pinsel", "Pincel", "画笔", "Pincel",
        "ब्रश", "브러시", "Pensill", "Pinceau"),
    "Divide an object": _row(
        "Dela ett objekt", "Objekt teilen", "Dividir un objeto", "拆分对象",
        "Dividir um objeto", "ऑब्जेक्ट को विभाजित करें", "객체 나누기",
        "Skipta hlut", "Diviser un objet"),
    "Draw an object": _row(
        "Rita ett objekt", "Objekt zeichnen", "Dibujar un objeto", "绘制对象",
        "Desenhar um objeto", "ऑब्जेक्ट आरेखित करें", "객체 그리기",
        "Teikna hlut", "Dessiner un objet"),
    "Erase": _row(
        "Suddgummi", "Radierer", "Goma de borrar", "橡皮擦", "Borracha",
        "इरेज़र", "지우개", "Strokleður", "Gomme"),
    "Field Browser": _row(
        "Fältbläddrare", "Feldbrowser", "Explorador de campos", "视野浏览器",
        "Navegador de campos", "फ़ील्ड ब्राउज़र", "필드 브라우저",
        "Reitavafri", "Explorateur de champs"),
    "Full screen": _row(
        "Helskärm", "Vollbild", "Pantalla completa", "全屏", "Tela cheia",
        "पूर्ण स्क्रीन", "전체 화면", "Heilskjár", "Plein écran"),
    "Go to home": _row(
        "Gå till startsidan", "Zur Startseite wechseln", "Ir a Inicio",
        "转到主页", "Ir para Início", "मुखपृष्ठ पर जाएँ", "홈으로 이동",
        "Fara á heimaskjáinn", "Aller à l’accueil"),
    "Jump to the newest console line": _row(
        "Gå till den senaste konsolraden",
        "Zur neuesten Konsolenzeile springen",
        "Ir a la línea más reciente de la consola",
        "跳转到控制台最新一行",
        "Ir para a linha mais recente do console",
        "कंसोल की नवीनतम पंक्ति पर जाएँ",
        "콘솔의 최신 줄로 이동",
        "Fara í nýjustu línu stjórnborðsins",
        "Aller à la dernière ligne de la console"),
    "Magic wand — add": _row(
        "Trollstav — lägg till", "Zauberstab — hinzufügen",
        "Varita mágica — añadir", "魔棒 — 添加",
        "Varinha mágica — adicionar", "मैजिक वैंड — जोड़ें",
        "자동 선택 도구 — 추가", "Töfrasproti — bæta við",
        "Baguette magique — ajouter"),
    "Next image": _row(
        "Nästa bild", "Nächstes Bild", "Imagen siguiente", "下一张图像",
        "Próxima imagem", "अगली छवि", "다음 이미지", "Næsta mynd",
        "Image suivante"),
    "Open command palette": _row(
        "Öppna kommandopaletten", "Befehlspalette öffnen",
        "Abrir la paleta de comandos", "打开命令面板",
        "Abrir a paleta de comandos", "कमांड पैलेट खोलें", "명령 팔레트 열기",
        "Opna skipanaspjaldið", "Ouvrir la palette de commandes"),
    "Open preferences": _row(
        "Öppna inställningarna", "Einstellungen öffnen",
        "Abrir las preferencias", "打开首选项", "Abrir as preferências",
        "प्राथमिकताएँ खोलें", "환경설정 열기", "Opna stillingar",
        "Ouvrir les préférences"),
    "Pause or resume the animated background": _row(
        "Pausa eller återuppta den animerade bakgrunden",
        "Animierten Hintergrund anhalten oder fortsetzen",
        "Pausar o reanudar el fondo animado",
        "暂停或继续动态背景",
        "Pausar ou retomar o plano de fundo animado",
        "एनिमेटेड पृष्ठभूमि रोकें या फिर से चलाएँ",
        "애니메이션 배경 일시 중지 또는 재개",
        "Gera hlé á hreyfanlega bakgrunninum eða halda honum áfram",
        "Mettre en pause ou reprendre l’arrière-plan animé"),
    "Previous image": _row(
        "Föregående bild", "Vorheriges Bild", "Imagen anterior", "上一张图像",
        "Imagem anterior", "पिछली छवि", "이전 이미지", "Fyrri mynd",
        "Image précédente"),
    "Recrop an object": _row(
        "Beskär ett objekt på nytt", "Objekt neu zuschneiden",
        "Volver a recortar un objeto", "重新裁剪对象",
        "Recortar novamente um objeto", "ऑब्जेक्ट को फिर से क्रॉप करें",
        "객체 다시 자르기", "Skera hlut aftur", "Recadrer un objet"),
    "Redo": _row(
        "Gör om", "Wiederherstellen", "Rehacer", "重做", "Refazer",
        "फिर से करें", "다시 실행", "Endurtaka", "Rétablir"),
    "Reset the zoom": _row(
        "Återställ zoomningen", "Zoom zurücksetzen", "Restablecer el zoom",
        "重置缩放", "Redefinir o zoom", "ज़ूम रीसेट करें", "확대/축소 초기화",
        "Endurstilla aðdrátt", "Réinitialiser le zoom"),
    "Restart the background": _row(
        "Starta om bakgrunden", "Hintergrund neu starten",
        "Reiniciar el fondo", "重新启动背景动画", "Reiniciar o plano de fundo",
        "पृष्ठभूमि फिर से शुरू करें", "배경 다시 시작",
        "Endurræsa bakgrunninn", "Redémarrer l’arrière-plan"),
    "Save the mask": _row(
        "Spara masken", "Maske speichern", "Guardar la máscara", "保存掩膜",
        "Salvar a máscara", "मास्क सहेजें", "마스크 저장", "Vista grímuna",
        "Enregistrer le masque"),
    "Search this module's settings": _row(
        "Sök i inställningarna för den här modulen",
        "Einstellungen dieses Moduls durchsuchen",
        "Buscar en la configuración de este módulo",
        "搜索此模块的设置",
        "Pesquisar nas configurações deste módulo",
        "इस मॉड्यूल की सेटिंग्स में खोजें",
        "이 모듈의 설정 검색",
        "Leita í stillingum þessarar einingar",
        "Rechercher dans les paramètres de ce module"),
    "Settings recipes": _row(
        "Inställningsrecept", "Einstellungsrezepte",
        "Recetas de configuración", "设置方案", "Receitas de configuração",
        "सेटिंग रेसिपी", "설정 레시피", "Stillingauppskriftir",
        "Recettes de paramètres"),
    "Show only the background full screen": _row(
        "Visa endast bakgrunden i helskärm",
        "Nur den Hintergrund im Vollbild anzeigen",
        "Mostrar solo el fondo a pantalla completa",
        "仅全屏显示背景",
        "Mostrar somente o plano de fundo em tela cheia",
        "केवल पृष्ठभूमि को पूर्ण स्क्रीन में दिखाएँ",
        "배경만 전체 화면으로 표시",
        "Sýna aðeins bakgrunninn á öllum skjánum",
        "Afficher uniquement l’arrière-plan en plein écran"),
    "Show the full app list": _row(
        "Visa hela listan över appar", "Vollständige App-Liste anzeigen",
        "Mostrar la lista completa de aplicaciones", "显示完整应用列表",
        "Mostrar a lista completa de aplicativos", "पूरी ऐप सूची दिखाएँ",
        "전체 앱 목록 표시", "Sýna allan forritalistann",
        "Afficher la liste complète des applications"),
    "Show this cheat sheet": _row(
        "Visa den här översikten över kortkommandon",
        "Diese Tastenkürzelübersicht anzeigen",
        "Mostrar este resumen de atajos de teclado",
        "显示此快捷键表",
        "Mostrar este resumo dos atalhos de teclado",
        "यह त्वरित संदर्भ दिखाएँ",
        "이 단축키 안내 표시",
        "Sýna þetta yfirlit yfir flýtilykla",
        "Afficher cette fiche récapitulative des raccourcis"),
    "Switch to 1st app": _row(
        "Växla till den första appen", "Zur ersten App wechseln",
        "Cambiar a la primera aplicación", "切换到第一个应用",
        "Mudar para o primeiro aplicativo", "पहले ऐप पर जाएँ",
        "첫 번째 앱으로 전환", "Skipta yfir í fyrsta forritið",
        "Passer à la première application"),
    "Switch to 2nd app": _row(
        "Växla till den andra appen", "Zur zweiten App wechseln",
        "Cambiar a la segunda aplicación", "切换到第二个应用",
        "Mudar para o segundo aplicativo", "दूसरे ऐप पर जाएँ",
        "두 번째 앱으로 전환", "Skipta yfir í annað forritið",
        "Passer à la deuxième application"),
    "Switch to 3rd app": _row(
        "Växla till den tredje appen", "Zur dritten App wechseln",
        "Cambiar a la tercera aplicación", "切换到第三个应用",
        "Mudar para o terceiro aplicativo", "तीसरे ऐप पर जाएँ",
        "세 번째 앱으로 전환", "Skipta yfir í þriðja forritið",
        "Passer à la troisième application"),
    "Switch to 4th app": _row(
        "Växla till den fjärde appen", "Zur vierten App wechseln",
        "Cambiar a la cuarta aplicación", "切换到第四个应用",
        "Mudar para o quarto aplicativo", "चौथे ऐप पर जाएँ",
        "네 번째 앱으로 전환", "Skipta yfir í fjórða forritið",
        "Passer à la quatrième application"),
    "Switch to 5th app": _row(
        "Växla till den femte appen", "Zur fünften App wechseln",
        "Cambiar a la quinta aplicación", "切换到第五个应用",
        "Mudar para o quinto aplicativo", "पाँचवें ऐप पर जाएँ",
        "다섯 번째 앱으로 전환", "Skipta yfir í fimmta forritið",
        "Passer à la cinquième application"),
    "Switch to 6th app": _row(
        "Växla till den sjätte appen", "Zur sechsten App wechseln",
        "Cambiar a la sexta aplicación", "切换到第六个应用",
        "Mudar para o sexto aplicativo", "छठे ऐप पर जाएँ",
        "여섯 번째 앱으로 전환", "Skipta yfir í sjötta forritið",
        "Passer à la sixième application"),
    "Switch to 7th app": _row(
        "Växla till den sjunde appen", "Zur siebten App wechseln",
        "Cambiar a la séptima aplicación", "切换到第七个应用",
        "Mudar para o sétimo aplicativo", "सातवें ऐप पर जाएँ",
        "일곱 번째 앱으로 전환", "Skipta yfir í sjöunda forritið",
        "Passer à la septième application"),
    "Switch to 8th app": _row(
        "Växla till den åttonde appen", "Zur achten App wechseln",
        "Cambiar a la octava aplicación", "切换到第八个应用",
        "Mudar para o oitavo aplicativo", "आठवें ऐप पर जाएँ",
        "여덟 번째 앱으로 전환", "Skipta yfir í áttunda forritið",
        "Passer à la huitième application"),
    "Switch to 9th app": _row(
        "Växla till den nionde appen", "Zur neunten App wechseln",
        "Cambiar a la novena aplicación", "切换到第九个应用",
        "Mudar para o nono aplicativo", "नौवें ऐप पर जाएँ",
        "아홉 번째 앱으로 전환", "Skipta yfir í níunda forritið",
        "Passer à la neuvième application"),
    "Toggle AI Console": _row(
        "Slå på/av AI-konsolen", "KI-Konsole ein-/ausschalten",
        "Activar o desactivar la Consola de IA", "启用或停用人工智能控制台",
        "Ativar/desativar o Console de IA", "एआई कंसोल चालू या बंद करें",
        "AI 콘솔 켜기/끄기", "Virkja eða óvirkja Gervigreindarstjórnborð",
        "Activer ou désactiver la Console IA"),
    "Toggle field quarantine": _row(
        "Växla fältkarantän", "Feldquarantäne umschalten",
        "Alternar la cuarentena del campo", "切换视野隔离状态",
        "Alternar a quarentena do campo", "फ़ील्ड क्वारंटीन टॉगल करें",
        "필드 격리 전환", "Víxla sóttkví reits",
        "Activer ou désactiver la quarantaine du champ"),
    "Quarantine or restore this field": _row(
        "Sätt detta fält i karantän eller återställ det",
        "Dieses Feld unter Quarantäne stellen oder wiederherstellen",
        "Poner en cuarentena o restaurar este campo", "隔离或恢复此视野",
        "Colocar este campo em quarentena ou restaurá-lo",
        "इस फ़ील्ड को क्वारंटीन करें या पुनर्स्थापित करें",
        "이 필드를 격리하거나 복원", "Setja þennan reit í sóttkví eða endurheimta hann",
        "Mettre ce champ en quarantaine ou le restaurer"),
    "Field browser": _row(
        "Fältbläddrare", "Feldbrowser", "Explorador de campos", "视野浏览器",
        "Navegador de campos", "फ़ील्ड ब्राउज़र", "필드 브라우저",
        "Reitavafri", "Explorateur de champs"),
    "Toggle full screen": _row(
        "Växla helskärmsläge", "Vollbildmodus ein-/ausschalten",
        "Activar o desactivar el modo de pantalla completa", "切换全屏模式",
        "Ativar/desativar o modo de tela cheia", "पूर्ण स्क्रीन मोड टॉगल करें",
        "전체 화면 전환", "Víxla skjáfylli",
        "Activer ou désactiver le mode plein écran"),
    "Undo": _row(
        "Ångra", "Rückgängig", "Deshacer", "撤销", "Desfazer",
        "पूर्ववत करें", "실행 취소", "Afturkalla", "Annuler"),
    "Zoom": _row(
        "Zooma", "Zoom", "Zoom", "缩放", "Zoom",
        "ज़ूम", "확대/축소", "Aðdráttur", "Zoom"),
    "anywhere in spaCR": _row(
        "var som helst i spaCR", "überall in spaCR",
        "en cualquier parte de spaCR", "spaCR 中的任意位置",
        "em qualquer lugar no spaCR", "spaCR में कहीं भी", "spaCR 어디서나",
        "hvar sem er í spaCR", "partout dans spaCR"),
    "the Annotate and Make Masks screens": _row(
        "skärmarna Annotering och Skapa masker",
        "die Bildschirme Annotieren und Masken erstellen",
        "las pantallas Anotación y Crear máscaras",
        "标注和创建掩膜屏幕",
        "as telas Anotação e Criar máscaras",
        "एनोटेशन और मास्क बनाएँ स्क्रीन",
        "어노테이션 및 마스크 만들기 화면",
        "skjáirnir Merking og Búa til grímur",
        "les écrans Annotation et Créer des masques"),
    "the Annotate and Make Masks screens and the QC field browser": _row(
        "skärmarna Annotering och Skapa masker samt QC-fältbläddraren",
        "die Bildschirme Annotieren und Masken erstellen sowie der QC-Feldbrowser",
        "las pantallas Anotación y Crear máscaras y el Explorador de campos de QC",
        "标注和创建掩膜屏幕以及 QC 视野浏览器",
        "as telas Anotação e Criar máscaras e o Navegador de campos de QC",
        "एनोटेशन और मास्क बनाएँ स्क्रीन तथा QC फ़ील्ड ब्राउज़र",
        "어노테이션 및 마스크 만들기 화면과 QC 필드 브라우저",
        "skjáirnir Merking og Búa til grímur og QC-reitavafrinn",
        "les écrans Annotation et Créer des masques et l’Explorateur de champs QC"),
    "the Annotate screen": _row(
        "skärmen Annotering", "der Bildschirm Annotieren",
        "la pantalla Anotación", "标注屏幕", "a tela Anotação",
        "एनोटेशन स्क्रीन", "어노테이션 화면", "skjárinn Merking",
        "l’écran Annotation"),
    "the Field Browser": _row(
        "Fältbläddraren", "der Feldbrowser", "el Explorador de campos",
        "视野浏览器", "o Navegador de campos", "फ़ील्ड ब्राउज़र",
        "필드 브라우저", "Reitavafrinn", "l’Explorateur de champs"),
    "the QC field browser": _row(
        "QC-fältbläddraren", "der QC-Feldbrowser",
        "el Explorador de campos de QC", "QC 视野浏览器",
        "o Navegador de campos de QC", "QC फ़ील्ड ब्राउज़र",
        "QC 필드 브라우저", "QC-reitavafrinn",
        "l’Explorateur de champs QC"),
    "the Make Masks screen": _row(
        "skärmen Skapa masker", "der Bildschirm Masken erstellen",
        "la pantalla Crear máscaras", "创建掩膜屏幕",
        "a tela Criar máscaras", "मास्क बनाएँ स्क्रीन", "마스크 만들기 화면",
        "skjárinn Búa til grímur", "l’écran Créer des masques"),
    # Terms/setup chrome. The agreement document remains in English because
    # a translated summary is not the governing licence; every instruction
    # and control around it is translated exactly.
    "Terms of use": _row(
        "Användningsvillkor", "Nutzungsbedingungen", "Condiciones de uso",
        "使用条款", "Termos de uso", "उपयोग की शर्तें", "이용 약관",
        "Notkunarskilmálar", "Conditions d’utilisation"),
    "Review the terms of use and scroll to the end to enable acceptance. "
    "Use the license link to read the full PolyForm Noncommercial License "
    "1.0.0.": _row(
        "Läs igenom användningsvillkoren och rulla till slutet för att "
        "aktivera godkännandet. Använd licenslänken för att läsa hela "
        "PolyForm Noncommercial License 1.0.0.",
        "Lesen Sie die Nutzungsbedingungen und scrollen Sie bis zum Ende, "
        "um die Zustimmung zu aktivieren. Über den Lizenzlink können Sie die "
        "vollständige PolyForm Noncommercial License 1.0.0 lesen.",
        "Revise las condiciones de uso y desplácese hasta el final para "
        "habilitar la aceptación. Utilice el enlace de la licencia para leer "
        "la PolyForm Noncommercial License 1.0.0 completa.",
        "请查看使用条款并滚动到末尾以启用接受选项。使用许可证链接可阅读完整的 "
        "PolyForm Noncommercial License 1.0.0。",
        "Revise os termos de uso e role até o final para habilitar a "
        "aceitação. Use o link da licença para ler a PolyForm Noncommercial "
        "License 1.0.0 completa.",
        "उपयोग की शर्तों की समीक्षा करें और स्वीकृति सक्षम करने के लिए अंत तक स्क्रॉल करें। "
        "पूर्ण PolyForm Noncommercial License 1.0.0 पढ़ने के लिए लाइसेंस लिंक का उपयोग करें।",
        "이용 약관을 검토하고 끝까지 스크롤하여 동의 항목을 활성화하십시오. "
        "라이선스 링크에서 전체 PolyForm Noncommercial License 1.0.0을 확인할 수 있습니다.",
        "Farðu yfir notkunarskilmálana og skrunaðu til enda til að "
        "virkja samþykki. Notaðu leyfistengilinn til að lesa PolyForm "
        "Noncommercial License 1.0.0 í heild.",
        "Consultez les conditions d’utilisation et faites défiler jusqu’à la "
        "fin pour activer l’acceptation. Utilisez le lien de licence pour lire "
        "l’intégralité de la PolyForm Noncommercial License 1.0.0."),
    "I have read and agree to these terms": _row(
        "Jag har läst och godkänner dessa villkor",
        "Ich habe diese Bedingungen gelesen und stimme ihnen zu",
        "He leído y acepto estos términos",
        "我已阅读并同意这些条款", "Li e aceito estes termos",
        "मैंने इन शर्तों को पढ़ लिया है और मैं इन्हें स्वीकार करता हूँ",
        "이 약관을 읽었으며 이에 동의합니다",
        "Ég hef lesið og samþykki þessa skilmála",
        "J’ai lu et j’accepte ces conditions"),
    "Scroll to the end of the terms to enable the acceptance checkbox.": _row(
        "Rulla till slutet av villkoren för att aktivera kryssrutan för "
        "godkännande.",
        "Scrollen Sie bis zum Ende der Nutzungsbedingungen, um das "
        "Kontrollkästchen zur Zustimmung zu aktivieren.",
        "Desplácese hasta el final de las condiciones para habilitar la "
        "casilla de aceptación.",
        "滚动到条款末尾以启用接受复选框。",
        "Role até o final dos termos para habilitar a caixa de seleção de "
        "aceitação.",
        "स्वीकृति चेकबॉक्स सक्षम करने के लिए शर्तों के अंत तक स्क्रॉल करें।",
        "약관 끝까지 스크롤하여 동의 확인란을 활성화하십시오.",
        "Skrunaðu til enda skilmálanna til að virkja samþykkisreitinn.",
        "Faites défiler jusqu’à la fin des conditions pour activer la case "
        "d’acceptation."),
    "Accept the terms of use to complete setup. If you close this window "
    "without accepting, spaCR will present the terms again at the next "
    "startup.": _row(
        "Godkänn användningsvillkoren för att slutföra konfigurationen. Om "
        "du stänger fönstret utan att godkänna dem visar spaCR villkoren igen "
        "vid nästa start.",
        "Akzeptieren Sie die Nutzungsbedingungen, um die Einrichtung "
        "abzuschließen. Wenn Sie dieses Fenster ohne Zustimmung schließen, "
        "zeigt spaCR die Bedingungen beim nächsten Start erneut an.",
        "Acepte las condiciones de uso para completar la configuración. Si "
        "cierra esta ventana sin aceptarlas, spaCR volverá a mostrar las "
        "condiciones en el próximo inicio.",
        "接受使用条款以完成设置。如果未接受就关闭此窗口，spaCR 将在下次启动时再次显示这些条款。",
        "Aceite os termos de uso para concluir a configuração. Se fechar "
        "esta janela sem aceitá-los, o spaCR apresentará os termos novamente "
        "na próxima inicialização.",
        "सेटअप पूरा करने के लिए उपयोग की शर्तें स्वीकार करें। यदि आप बिना स्वीकार किए यह विंडो "
        "बंद करते हैं, तो spaCR अगली बार शुरू होने पर शर्तें फिर दिखाएगा।",
        "설정을 완료하려면 이용 약관에 동의하십시오. 동의하지 않고 이 창을 닫으면 spaCR가 "
        "다음 시작 시 약관을 다시 표시합니다.",
        "Samþykktu notkunarskilmálana til að ljúka uppsetningu. Ef þú "
        "lokar þessum glugga án þess að samþykkja birtir spaCR skilmálana "
        "aftur við næstu ræsingu.",
        "Acceptez les conditions d’utilisation pour terminer la configuration. "
        "Si vous fermez cette fenêtre sans les accepter, spaCR les présentera "
        "de nouveau au prochain démarrage."),
    "Illumination Correction": _row(
        "Belysningskorrigering", "Beleuchtungskorrektur",
        "Corrección de iluminación", "照明校正",
        "Correção de iluminação", "प्रदीपन सुधार", "조명 보정",
        "Lýsingarleiðrétting", "Correction de l’éclairage"),
    # The strap line, one phase per row. Split rather than translated as one
    # sentence because the loading screen lights the phases INDIVIDUALLY, so
    # each has to stand alone -- and a language that reorders the clauses
    # would otherwise light them in the wrong order.
    # See spacr.qt.widgets.loading_screen.STRAP_PHASES.
    "End-to-end microscopy": _row(
        "Mikroskopi från början till slut", "Mikroskopie von Anfang bis Ende",
        "Microscopía de extremo a extremo", "端到端显微成像",
        "Microscopia de ponta a ponta", "आद्यंत माइक्रोस्कोपी",
        "엔드투엔드 현미경 분석", "Smásjárgreining frá upphafi til enda",
        "Microscopie de bout en bout"),
    "single-cell image analysis": _row(
        "bildanalys på encellsnivå", "Einzelzell-Bildanalyse",
        "análisis de imágenes de células individuales", "单细胞图像分析",
        "análise de imagens de células individuais",
        "एकल-कोशिका छवि विश्लेषण", "단일 세포 이미지 분석",
        "myndgreining einstakra frumna", "analyse d'images unicellulaires"),
    "genotype-to-phenotype mapping": _row(
        "kartläggning från genotyp till fenotyp",
        "Genotyp-zu-Phänotyp-Zuordnung", "mapeo de genotipo a fenotipo",
        "基因型到表型的映射", "mapeamento de genótipo para fenótipo",
        "जीनोटाइप-से-फेनोटाइप मानचित्रण", "유전형-표현형 매핑",
        "vörpun frá arfgerð til svipgerðar",
        "cartographie génotype-phénotype"),
    "Check for updates…": _row(
        "Sök efter uppdateringar…", "Nach Updates suchen…",
        "Buscar actualizaciones…", "检查更新…", "Verificar atualizações…",
        "अपडेट जाँचें…", "업데이트 확인…", "Leita að uppfærslum…",
        "Rechercher des mises à jour…"),
    "Open log folder…": _row(
        "Öppna loggmapp…", "Protokollordner öffnen…",
        "Abrir carpeta de registros…", "打开日志文件夹…",
        "Abrir pasta de logs…", "लॉग फ़ोल्डर खोलें…",
        "로그 폴더 열기…", "Opna annálamöppu…",
        "Ouvrir le dossier des journaux…"),
    "Ready": _row(
        "Klar", "Bereit", "Listo", "就绪", "Pronto",
        "तैयार", "준비됨", "Tilbúið", "Prêt"),
    "Opened {name}": _row(
        "Öppnade {name}", "{name} geöffnet", "Se abrió {name}",
        "已打开 {name}", "{name} aberto", "{name} खोला गया",
        "{name} 열림", "{name} opnað", "{name} ouvert"),
    "spaCR — Command palette": _row(
        "spaCR — Kommandopalett", "spaCR — Befehlspalette",
        "spaCR — Paleta de comandos", "spaCR — 命令面板",
        "spaCR — Paleta de comandos", "spaCR — कमांड पैलेट",
        "spaCR — 명령 팔레트", "spaCR — Skipanaspjald",
        "spaCR — Palette de commandes"),
    "Type to filter — Enter to run, Esc to cancel": _row(
        "Skriv för att filtrera — Enter kör, Esc avbryter",
        "Zum Filtern tippen — Eingabe startet, Esc bricht ab",
        "Escriba para filtrar — Intro ejecuta, Esc cancela",
        "输入以筛选 — 回车运行，Esc 取消",
        "Digite para filtrar — Enter executa, Esc cancela",
        "फ़िल्टर करने के लिए लिखें — Enter चलाता है, Esc रद्द करता है",
        "필터 입력 — Enter 실행, Esc 취소",
        "Sláðu inn til að sía — Enter keyrir, Esc hættir við",
        "Saisissez pour filtrer — Entrée exécute, Échap annule"),
    "Go to  {name}": _row(
        "Gå till {name}", "Zu {name}", "Ir a {name}", "前往{name}",
        "Ir para {name}", "{name} पर जाएँ", "{name}(으)로 이동",
        "Fara í {name}", "Aller à {name}"),
    "Apps · {section}": _row(
        "Appar · {section}", "Module · {section}",
        "Aplicaciones · {section}", "应用 · {section}",
        "Aplicativos · {section}", "ऐप · {section}",
        "앱 · {section}", "Forrit · {section}",
        "Applications · {section}"),
    "Navigation": _row(
        "Navigering", "Navigation", "Navegación", "导航", "Navegação",
        "नेविगेशन", "탐색", "Leiðsögn", "Navigation"),
    "Actions": _row(
        "Åtgärder", "Aktionen", "Acciones", "操作", "Ações",
        "क्रियाएँ", "작업", "Aðgerðir", "Actions"),
    "Open Preferences…": _row(
        "Öppna inställningar…", "Einstellungen öffnen…",
        "Abrir preferencias…", "打开首选项…", "Abrir preferências…",
        "प्राथमिकताएँ खोलें…", "환경설정 열기…", "Opna stillingar…",
        "Ouvrir les préférences…"),
    "Open AI Providers…": _row(
        "Öppna AI-leverantörer…", "KI-Anbieter öffnen…",
        "Abrir proveedores de IA…", "打开 AI 提供商…",
        "Abrir provedores de IA…", "AI प्रदाता खोलें…",
        "AI 제공업체 열기…", "Opna AI-veitur…",
        "Ouvrir les fournisseurs d’IA…"),
    "Keyboard shortcuts…": _row(
        "Kortkommandon…", "Tastenkürzel…", "Atajos de teclado…",
        "键盘快捷键…", "Atalhos de teclado…", "कीबोर्ड शॉर्टकट…",
        "키보드 단축키…", "Flýtilyklar…", "Raccourcis clavier…"),
    "Run": _row(
        "Kör", "Ausführen", "Ejecutar", "运行", "Executar",
        "चलाएँ", "실행", "Keyra", "Exécuter"),
    "Stop": _row(
        "Stoppa", "Stopp", "Detener", "停止", "Parar",
        "रोकें", "중지", "Stöðva", "Arrêter"),
    "Propagate": _row(
        "Överför", "Übernehmen", "Propagar", "应用", "Propagar",
        "लागू करें", "전파", "Yfirfæra", "Propager"),
    "Close": _row(
        "Stäng", "Schließen", "Cerrar", "关闭", "Fechar",
        "बंद करें", "닫기", "Loka", "Fermer"),
    "Open": _row(
        "Öppna", "Öffnen", "Abrir", "打开", "Abrir",
        "खोलें", "열기", "Opna", "Ouvrir"),
    "Pause": _row(
        "Pausa", "Pause", "Pausa", "暂停", "Pausar",
        "रोकें", "일시정지", "Gera hlé", "Pause"),
    "Browse": _row(
        "Bläddra", "Durchsuchen", "Examinar", "浏览", "Procurar",
        "ब्राउज़ करें", "찾아보기", "Velja", "Parcourir"),
    "Add": _row(
        "Lägg till", "Hinzufügen", "Añadir", "添加", "Adicionar",
        "जोड़ें", "추가", "Bæta við", "Ajouter"),
    "Remove": _row(
        "Ta bort", "Entfernen", "Eliminar", "移除", "Remover",
        "हटाएँ", "제거", "Fjarlægja", "Supprimer"),
    "Save": _row(
        "Spara", "Speichern", "Guardar", "保存", "Salvar",
        "सहेजें", "저장", "Vista", "Enregistrer"),
    "Cancel": _row(
        "Avbryt", "Abbrechen", "Cancelar", "取消", "Cancelar",
        "रद्द करें", "취소", "Hætta við", "Annuler"),
    "Settings": _row(
        "Inställningar", "Einstellungen", "Configuración", "设置",
        "Configurações", "सेटिंग्स", "설정", "Stillingar", "Paramètres"),
    "Console": _row(
        "Konsol", "Konsole", "Consola", "控制台", "Console",
        "कंसोल", "콘솔", "Stjórnborð", "Console"),
    "Preview": _row(
        "Förhandsvisning", "Vorschau", "Vista previa", "预览",
        "Pré-visualização", "पूर्वावलोकन", "미리보기", "Forskoðun",
        "Aperçu"),
    "Search": _row(
        "Sök", "Suchen", "Buscar", "搜索", "Pesquisar",
        "खोजें", "검색", "Leita", "Rechercher"),
    "Refresh": _row(
        "Uppdatera", "Aktualisieren", "Actualizar", "刷新", "Atualizar",
        "ताज़ा करें", "새로 고침", "Endurnýja", "Actualiser"),
    "All modules": _row(
        "Alla moduler", "Alle Module", "Todos los módulos", "所有模块",
        "Todos os módulos", "सभी मॉड्यूल", "모든 모듈", "Allar einingar",
        "Tous les modules"),
    "All statuses": _row(
        "Alla statusar", "Alle Status", "Todos los estados", "所有状态",
        "Todos os estados", "सभी स्थितियाँ", "모든 상태", "Allar stöður",
        "Tous les états"),
    "Success": _row(
        "Lyckades", "Erfolgreich", "Correcto", "成功", "Sucesso",
        "सफल", "성공", "Tókst", "Réussi"),
    "Failed": _row(
        "Misslyckades", "Fehlgeschlagen", "Fallido", "失败", "Falhou",
        "विफल", "실패", "Mistókst", "Échec"),
    "Running": _row(
        "Körs", "Läuft", "En ejecución", "运行中", "Em execução",
        "चल रहा है", "실행 중", "Í gangi", "En cours"),
    "Corrupt": _row(
        "Skadad", "Beschädigt", "Dañado", "已损坏", "Corrompido",
        "दूषित", "손상됨", "Skemmt", "Corrompu"),
    "Open run folder": _row(
        "Öppna körningsmapp", "Laufordner öffnen",
        "Abrir carpeta de ejecución", "打开运行文件夹",
        "Abrir pasta da execução", "रन फ़ोल्डर खोलें",
        "실행 폴더 열기", "Opna keyrslumöppu",
        "Ouvrir le dossier d’exécution"),
    "Copy path": _row(
        "Kopiera sökväg", "Pfad kopieren", "Copiar ruta", "复制路径",
        "Copiar caminho", "पथ कॉपी करें", "경로 복사", "Afrita slóð",
        "Copier le chemin"),
    "Load settings in module": _row(
        "Läs in inställningar i modulen",
        "Einstellungen im Modul laden",
        "Cargar ajustes en el módulo", "在模块中加载设置",
        "Carregar configurações no módulo", "मॉड्यूल में सेटिंग लोड करें",
        "모듈에 설정 불러오기", "Hlaða stillingum í einingu",
        "Charger les paramètres dans le module"),
    "Overview": _row(
        "Översikt", "Übersicht", "Resumen", "概览", "Visão geral",
        "अवलोकन", "개요", "Yfirlit", "Vue d’ensemble"),
    "Files & models": _row(
        "Filer och modeller", "Dateien & Modelle", "Archivos y modelos",
        "文件和模型", "Arquivos e modelos", "फ़ाइलें और मॉडल",
        "파일 및 모델", "Skrár og líkön", "Fichiers et modèles"),
    "Warnings & failure": _row(
        "Varningar och fel", "Warnungen & Fehler", "Avisos y fallo",
        "警告和失败", "Avisos e falha", "चेतावनियाँ और विफलता",
        "경고 및 실패", "Viðvaranir og bilun", "Avertissements et échec"),
    "Environment": _row(
        "Miljö", "Umgebung", "Entorno", "环境", "Ambiente",
        "परिवेश", "환경", "Umhverfi", "Environnement"),
    "Language": _row(
        "Språk", "Sprache", "Idioma", "语言", "Idioma",
        "भाषा", "언어", "Tungumál", "Langue"),

    # Application registry — every user-visible module.
    "Mask": _row(
        "Masker", "Masken", "Máscaras", "掩膜", "Máscaras",
        "मास्क", "마스크", "Grímur", "Masques"),
    "Timelapse": _row(
        "Tidsserie", "Zeitraffer", "Serie temporal", "延时成像",
        "Série temporal", "टाइमलैप्स", "타임랩스", "Tímaraðir",
        "Série temporelle"),
    "Motility Assay": _row(
        "Motilitetsanalys", "Motilitätsassay", "Ensayo de motilidad",
        "运动性分析", "Ensaio de motilidade", "गतिशीलता परीक्षण",
        "운동성 분석", "Hreyfanleikapróf", "Test de motilité"),
    "Measure": _row(
        "Mätning", "Messen", "Medición", "测量", "Medição",
        "मापन", "측정", "Mæling", "Mesure"),
    "Annotate": _row(
        "Annotering", "Annotieren", "Anotación", "标注", "Anotação",
        "एनोटेशन", "어노테이션", "Merking", "Annotation"),
    "Classify": _row(
        "Klassificering", "Klassifizieren", "Clasificar", "分类", "Classificar",
        "वर्गीकरण", "분류", "Flokkun", "Classification"),
    "Classify (CV)": _row(
        "Klassificering (CV)", "Klassifizieren (CV)", "Clasificar (CV)",
        "分类（CV）", "Classificar (CV)", "वर्गीकरण (CV)",
        "분류(CV)", "Flokkun (CV)", "Classification (CV)"),
    "Classify (ML)": _row(
        "Klassificering (ML)", "Klassifizieren (ML)", "Clasificar (ML)",
        "分类（ML）", "Classificar (ML)", "वर्गीकरण (ML)",
        "분류(ML)", "Flokkun (ML)", "Classification (ML)"),
    "Map Barcodes": _row(
        "Kartlägg streckkoder", "Barcodes zuordnen", "Mapear códigos de barras",
        "映射条形码", "Mapear códigos de barras", "बारकोड मैप करें",
        "바코드 매핑", "Varpa strikamerkjum", "Mapper les codes-barres"),
    "Regression": _row(
        "Regression", "Regression", "Regresión", "回归", "Regressão",
        "प्रतिगमन", "회귀", "Aðhvarf", "Régression"),
    "Align & Stitch": _row(
        "Justera och sammanfoga", "Ausrichten und zusammenfügen",
        "Alinear y unir", "对齐与拼接", "Alinhar e unir",
        "संरेखित और जोड़ें", "정렬 및 스티칭", "Jafna og sauma",
        "Aligner et assembler"),
    "Format Converter": _row(
        "Formatkonverterare", "Formatkonverter", "Convertidor de formatos",
        "格式转换器", "Conversor de formatos", "प्रारूप परिवर्तक",
        "형식 변환기", "Sniðbreytir", "Convertisseur de formats"),
    "Import Project": _row(
        "Importera projekt", "Projekt importieren", "Importar proyecto",
        "导入项目", "Importar projeto", "प्रोजेक्ट आयात करें",
        "프로젝트 가져오기", "Flytja inn verkefni", "Importer un projet"),
    "External Masks": _row(
        "Externa masker", "Externe Masken", "Máscaras externas", "外部掩膜",
        "Máscaras externas", "बाहरी मास्क", "외부 마스크", "Ytri grímur",
        "Masques externes"),
    "Plate Queue": _row(
        "Plattkö", "Plattenwarteschlange", "Cola de placas", "孔板队列",
        "Fila de placas", "प्लेट कतार", "플레이트 대기열", "Plöturöð",
        "File de plaques"),
    "Batch Runner": _row(
        "Batchkörning", "Stapelverarbeitung", "Ejecución por lotes",
        "批处理运行器", "Executor em lote", "बैच रनर",
        "배치 실행기", "Lotukeyrsla", "Exécution par lots"),
    "Database Browser": _row(
        "Databasutforskare", "Datenbankbrowser", "Explorador de base de datos",
        "数据库浏览器", "Navegador de banco de dados", "डेटाबेस ब्राउज़र",
        "데이터베이스 브라우저", "Gagnagrunnsvafri",
        "Explorateur de base de données"),
    "Make Masks": _row(
        "Skapa masker", "Masken erstellen", "Crear máscaras", "创建掩膜",
        "Criar máscaras", "मास्क बनाएँ", "마스크 만들기", "Búa til grímur",
        "Créer des masques"),
    "Train Cellpose": _row(
        "Träna Cellpose", "Cellpose trainieren", "Entrenar Cellpose",
        "训练 Cellpose", "Treinar Cellpose", "Cellpose प्रशिक्षित करें",
        "Cellpose 학습", "Þjálfa Cellpose", "Entraîner Cellpose"),
    # The name `train_cellpose` registers under. "Train Cellpose" above is
    # what the module used to be called and still appears in older prose,
    # so both rows stand.
    "Cellpose Workbench": _row(
        "Cellpose-verkstad", "Cellpose-Werkbank", "Banco de trabajo Cellpose",
        "Cellpose 工作台", "Bancada Cellpose", "Cellpose वर्कबेंच",
        "Cellpose 워크벤치", "Cellpose-vinnuborð", "Atelier Cellpose"),
    "Cellpose Masks": _row(
        "Cellpose-masker", "Cellpose-Masken", "Máscaras Cellpose",
        "Cellpose 掩膜", "Máscaras Cellpose", "Cellpose मास्क",
        "Cellpose 마스크", "Cellpose-grímur", "Masques Cellpose"),
    "Model Compare": _row(
        "Modelljämförelse", "Modellvergleich", "Comparar modelos",
        "模型比较", "Comparar modelos", "मॉडल तुलना",
        "모델 비교", "Líkansamanburður", "Comparer les modèles"),
    "Model Zoo": _row(
        "Modellbibliotek", "Modellzoo", "Biblioteca de modelos", "模型库",
        "Biblioteca de modelos", "मॉडल संग्रह", "모델 라이브러리",
        "Líkanasafn", "Bibliothèque de modèles"),
    "Plate Viewer": _row(
        "Plattvisare", "Plattenansicht", "Visor de placas", "孔板查看器",
        "Visualizador de placas", "प्लेट व्यूअर", "플레이트 뷰어",
        "Plötuskoðari", "Visionneuse de plaques"),
    "Annotator Agreement": _row(
        "Annotatörsöverensstämmelse", "Annotatorenübereinstimmung",
        "Concordancia de anotadores", "标注者一致性",
        "Concordância de anotadores", "एनोटेटर सहमति",
        "주석자 일치도", "Samræmi merkjara", "Accord des annotateurs"),
    "Image UMAP": _row(
        "Bild-UMAP", "Bild-UMAP", "UMAP de imágenes", "图像 UMAP",
        "UMAP de imagens", "इमेज UMAP", "이미지 UMAP", "Mynda-UMAP",
        "UMAP d’images"),
    "Activation": _row(
        "Aktivering", "Aktivierung", "Activación", "激活", "Ativação",
        "सक्रियण", "활성화", "Virkjun", "Activation"),
    "Training Runs": _row(
        "Träningskörningar", "Trainingsläufe", "Ejecuciones de entrenamiento",
        "训练运行", "Execuções de treinamento", "प्रशिक्षण रन",
        "학습 실행", "Þjálfunarkeyrslur", "Exécutions d’entraînement"),
    "Classifier Evaluation": _row(
        "Klassificerarutvärdering", "Klassifikatorauswertung",
        "Evaluación del clasificador", "分类器评估",
        "Avaliação do classificador", "वर्गीकारक मूल्यांकन",
        "분류기 평가", "Mat á flokkara", "Évaluation du classificateur"),
    # NAMES THAT ARRIVED THROUGH `register_app(translations=...)`.
    # Folding a module into a host screen deletes its registry row, and
    # with it the only call that put its name in these nine catalogs --
    # so a Korean window would head the folded page in English. The names
    # are the module's, not the tile's, and the page still wears them, so
    # they are written down here where the other module names are.
    # `add_translation` is a no-op for a source already catalogued, so a
    # module that still registers is unaffected.
    "Barcode QC": _row(
        "Streckkods-QC", "Barcode-QC", "CC de códigos de barras",
        "条形码质控", "CQ de código de barras", "बारकोड QC",
        "바코드 QC", "Strikamerkja-QC", "CQ des codes-barres"),
    "Illumination": _row(
        "Belysning", "Beleuchtung", "Iluminación", "照明",
        "Iluminação", "प्रकाश", "조명", "Lýsing", "Éclairage"),
    "Explain CV Model": _row(
        "Förklara CV-modell", "CV-Modell erklären", "Explicar modelo CV",
        "解释 CV 模型", "Explicar modelo de VC", "CV मॉडल समझाएँ",
        "CV 모델 설명", "Skýra CV-líkan", "Expliquer le modèle CV"),
    "Feature Dictionary": _row(
        "Egenskapsordlista", "Merkmalswörterbuch",
        "Diccionario de características", "特征词典",
        "Dicionário de características", "विशेषता शब्दकोश", "특성 사전",
        "Eiginleikaorðabók", "Dictionnaire des caractéristiques"),
    "AnnData Export": _row(
        "AnnData-export", "AnnData-Export", "Exportar a AnnData",
        "导出 AnnData", "Exportar para AnnData", "AnnData निर्यात",
        "AnnData 내보내기", "AnnData-útflutningur", "Export AnnData"),
    # Fold buttons no longer have an application-registry row to contribute
    # their display name.  Keep every current folded name in the compact
    # catalog: the icon-only button exposes this text through its tooltip and
    # accessible name, so an English fallback there is still visible UI.
    "Curate": _row(
        "Kurera", "Kuratieren", "Curación", "校正", "Curadoria",
        "क्यूरेट", "큐레이트", "Grisja", "Curation"),
    "Image Scatter": _row(
        "Bildspridningsdiagram", "Bild-Streudiagramm",
        "Dispersión de imágenes", "图像散点图",
        "Dispersão de imagens", "छवि स्कैटर प्लॉट", "이미지 산점도",
        "Mynddreifirit", "Nuage d’images"),
    "Mask the whole folder": _row(
        "Maskera hela mappen", "Gesamten Ordner maskieren",
        "Generar máscaras para toda la carpeta", "为整个文件夹生成掩膜",
        "Gerar máscaras para toda a pasta", "पूरे फ़ोल्डर के लिए मास्क बनाएँ",
        "전체 폴더의 마스크 생성", "Búa til grímur fyrir alla möppuna",
        "Générer les masques de tout le dossier"),
    "Napari Bridge": _row(
        "Napari-brygga", "Napari-Brücke", "Puente con napari",
        "napari 桥接", "Ponte para o napari", "नैपारी ब्रिज",
        "나파리 브리지", "Napari-brú", "Passerelle napari"),
    "PCA": _row(
        "PCA", "PCA", "PCA", "PCA", "PCA", "PCA", "PCA", "PCA",
        "ACP"),
    "Volcano Explorer": _row(
        "Utforska vulkandiagram", "Vulkanplot-Explorer",
        "Explorador de gráficos volcán", "火山图浏览器",
        "Explorador de gráficos vulcão", "वोल्केनो प्लॉट एक्सप्लोरर",
        "볼케이노 플롯 탐색기", "Eldfjallaritskönnun",
        "Explorateur de graphiques volcan"),
    "Run History": _row(
        "Körningshistorik", "Ausführungsverlauf", "Historial de ejecuciones",
        "运行历史", "Histórico de execuções", "रन इतिहास",
        "실행 기록", "Keyrslusaga", "Historique des exécutions"),
    "Distributed Jobs": _row(
        "Distribuerade jobb", "Verteilte Aufträge", "Trabajos distribuidos",
        "分布式作业", "Trabalhos distribuídos", "वितरित कार्य",
        "분산 작업", "Dreifð verk", "Tâches distribuées"),
    "Report": _row(
        "Rapport", "Bericht", "Informe", "报告", "Relatório",
        "रिपोर्ट", "보고서", "Skýrsla", "Rapport"),

    # Distributed execution.
    "Execution profile": _row(
        "Körprofil", "Ausführungsprofil", "Perfil de ejecución", "执行配置",
        "Perfil de execução", "निष्पादन प्रोफ़ाइल", "실행 프로필",
        "Keyrslusnið", "Profil d’exécution"),
    "Profile name": _row(
        "Profilnamn", "Profilname", "Nombre del perfil", "配置名称",
        "Nome do perfil", "प्रोफ़ाइल नाम", "프로필 이름",
        "Heiti sniðs", "Nom du profil"),
    "SSH workstation": _row(
        "SSH-arbetsstation", "SSH-Arbeitsstation", "Estación SSH", "SSH 工作站",
        "Estação SSH", "SSH कार्यस्थान", "SSH 워크스테이션",
        "SSH-vinnustöð", "Station SSH"),
    "Slurm cluster": _row(
        "Slurm-kluster", "Slurm-Cluster", "Clúster Slurm", "Slurm 集群",
        "Cluster Slurm", "Slurm क्लस्टर", "Slurm 클러스터",
        "Slurm-klasi", "Grappe Slurm"),
    "Cloud / custom command": _row(
        "Moln / eget kommando", "Cloud / eigener Befehl",
        "Nube / comando personalizado", "云 / 自定义命令",
        "Nuvem / comando personalizado", "क्लाउड / कस्टम कमांड",
        "클라우드 / 사용자 지정 명령", "Ský / sérsniðin skipun",
        "Cloud / commande personnalisée"),
    "Backend": _row(
        "Körsystem", "Backend", "Sistema de ejecución", "后端",
        "Sistema de execução", "बैकएंड", "백엔드", "Bakendi",
        "Moteur d’exécution"),
    "SSH host": _row(
        "SSH-värd", "SSH-Host", "Host SSH", "SSH 主机",
        "Host SSH", "SSH होस्ट", "SSH 호스트", "SSH-hýsill", "Hôte SSH"),
    "Remote work directory": _row(
        "Fjärrarbetskatalog", "Remote-Arbeitsverzeichnis",
        "Directorio de trabajo remoto", "远程工作目录",
        "Diretório de trabalho remoto", "दूरस्थ कार्य निर्देशिका",
        "원격 작업 디렉터리", "Fjartengd vinnumappa",
        "Répertoire de travail distant"),
    "Local dataset root": _row(
        "Lokal datarot", "Lokaler Datenstamm", "Raíz local de datos",
        "本地数据根目录", "Raiz local dos dados", "स्थानीय डेटा मूल",
        "로컬 데이터 루트", "Staðbundin gagnarót",
        "Racine locale des données"),
    "Remote dataset root": _row(
        "Fjärrdatarot", "Remote-Datenstamm", "Raíz remota de datos",
        "远程数据根目录", "Raiz remota dos dados", "दूरस्थ डेटा मूल",
        "원격 데이터 루트", "Fjartengd gagnarót",
        "Racine distante des données"),
    "spaCR runner": _row(
        "spaCR-körare", "spaCR-Runner", "Ejecutor spaCR", "spaCR 运行程序",
        "Executor spaCR", "spaCR रनर", "spaCR 실행기",
        "spaCR-keyrari", "Exécutable spaCR"),
    "Slurm options": _row(
        "Slurm-alternativ", "Slurm-Optionen", "Opciones de Slurm",
        "Slurm 选项", "Opções do Slurm", "Slurm विकल्प",
        "Slurm 옵션", "Slurm-valkostir", "Options Slurm"),
    "Submit command": _row(
        "Skicka-kommando", "Übermittlungsbefehl", "Comando de envío",
        "提交命令", "Comando de envio", "सबमिट कमांड",
        "제출 명령", "Innsendingarskipun", "Commande d’envoi"),
    "Status command": _row(
        "Statuskommando", "Statusbefehl", "Comando de estado", "状态命令",
        "Comando de status", "स्थिति कमांड", "상태 명령",
        "Stöðuskipun", "Commande d’état"),
    "Cancel command": _row(
        "Avbryt-kommando", "Abbruchbefehl", "Comando de cancelación",
        "取消命令", "Comando de cancelamento", "रद्द कमांड",
        "취소 명령", "Afturköllunarskipun", "Commande d’annulation"),
    "Log command (optional)": _row(
        "Loggkommando (valfritt)", "Protokollbefehl (optional)",
        "Comando de registro (opcional)", "日志命令（可选）",
        "Comando de log (opcional)", "लॉग कमांड (वैकल्पिक)",
        "로그 명령(선택)", "Annálaskipun (valfrjálst)",
        "Commande de journal (facultative)"),
    "Job-ID pattern (optional)": _row(
        "Jobb-ID-mönster (valfritt)", "Job-ID-Muster (optional)",
        "Patrón de ID de trabajo (opcional)", "作业 ID 模式（可选）",
        "Padrão de ID do trabalho (opcional)",
        "कार्य-ID पैटर्न (वैकल्पिक)", "작업 ID 패턴(선택)",
        "Verk-ID-mynstur (valfrjálst)",
        "Motif d’ID de tâche (facultatif)"),
    "Poll interval": _row(
        "Kontrollintervall", "Abfrageintervall", "Intervalo de consulta",
        "轮询间隔", "Intervalo de consulta", "पोल अंतराल",
        "폴링 간격", "Könnunarbil", "Intervalle d’interrogation"),
    "Submission": _row(
        "Inskickning", "Übermittlung", "Envío", "提交",
        "Envio", "सबमिशन", "제출", "Innsending", "Soumission"),
    "New profile…": _row(
        "Ny profil…", "Neues Profil…", "Nuevo perfil…", "新建配置…",
        "Novo perfil…", "नई प्रोफ़ाइल…", "새 프로필…",
        "Nýtt snið…", "Nouveau profil…"),
    "Edit profile…": _row(
        "Redigera profil…", "Profil bearbeiten…", "Editar perfil…",
        "编辑配置…", "Editar perfil…", "प्रोफ़ाइल संपादित करें…",
        "프로필 편집…", "Breyta sniði…", "Modifier le profil…"),
    "Delete profile": _row(
        "Ta bort profil", "Profil löschen", "Eliminar perfil", "删除配置",
        "Excluir perfil", "प्रोफ़ाइल हटाएँ", "프로필 삭제",
        "Eyða sniði", "Supprimer le profil"),
    "Execution target": _row(
        "Körmål", "Ausführungsziel", "Destino de ejecución", "执行目标",
        "Destino de execução", "निष्पादन लक्ष्य", "실행 대상",
        "Keyrslumark", "Cible d’exécution"),
    "Drop or choose a spaCR settings CSV/JSON": _row(
        "Släpp eller välj en spaCR-inställningsfil (CSV/JSON)",
        "spaCR-Einstellungen als CSV/JSON ablegen oder wählen",
        "Suelte o elija ajustes spaCR CSV/JSON",
        "拖放或选择 spaCR 设置 CSV/JSON",
        "Solte ou escolha configurações spaCR CSV/JSON",
        "spaCR सेटिंग CSV/JSON छोड़ें या चुनें",
        "spaCR 설정 CSV/JSON을 놓거나 선택하세요",
        "Slepptu eða veldu spaCR-stillingar CSV/JSON",
        "Déposez ou choisissez des réglages spaCR CSV/JSON"),
    "Submit": _row(
        "Skicka", "Senden", "Enviar", "提交", "Enviar",
        "सबमिट करें", "제출", "Senda", "Soumettre"),
    "Submit remote…": _row(
        "Skicka till fjärrsystem…", "Remote übermitteln…",
        "Enviar en remoto…", "远程提交…", "Enviar remotamente…",
        "दूरस्थ रूप से सबमिट करें…", "원격 제출…",
        "Senda fjarvinnslu…", "Soumettre à distance…"),
    "Browse…": _row(
        "Bläddra…", "Durchsuchen…", "Examinar…", "浏览…", "Procurar…",
        "ब्राउज़ करें…", "찾아보기…", "Fletta…", "Parcourir…"),
    "Module": _row(
        "Modul", "Modul", "Módulo", "模块", "Módulo",
        "मॉड्यूल", "모듈", "Eining", "Module"),
    "Cancel job": _row(
        "Avbryt jobb", "Auftrag abbrechen", "Cancelar trabajo", "取消作业",
        "Cancelar trabalho", "कार्य रद्द करें", "작업 취소",
        "Hætta við verk", "Annuler la tâche"),
    "Refresh log": _row(
        "Uppdatera logg", "Protokoll aktualisieren", "Actualizar registro",
        "刷新日志", "Atualizar log", "लॉग रीफ़्रेश करें",
        "로그 새로고침", "Endurnýja annál", "Actualiser le journal"),
    "Open local record": _row(
        "Öppna lokal post", "Lokalen Datensatz öffnen",
        "Abrir registro local", "打开本地记录", "Abrir registro local",
        "स्थानीय रिकॉर्ड खोलें", "로컬 기록 열기",
        "Opna staðbundna færslu", "Ouvrir le dossier local"),
    "Job": _row(
        "Jobb", "Auftrag", "Trabajo", "作业", "Trabalho",
        "कार्य", "작업", "Verk", "Tâche"),
    "Status": _row(
        "Status", "Status", "Estado", "状态", "Status",
        "स्थिति", "상태", "Staða", "État"),
    "Profile": _row(
        "Profil", "Profil", "Perfil", "配置", "Perfil",
        "प्रोफ़ाइल", "프로필", "Snið", "Profil"),
    "Remote ID": _row(
        "Fjärr-ID", "Remote-ID", "ID remoto", "远程 ID", "ID remoto",
        "दूरस्थ ID", "원격 ID", "Fjartengt ID", "ID distant"),
    "Submitted": _row(
        "Skickat", "Übermittelt", "Enviado", "已提交", "Enviado",
        "सबमिट किया", "제출됨", "Sent", "Soumise"),
    "Updated": _row(
        "Uppdaterat", "Aktualisiert", "Actualizado", "已更新", "Atualizado",
        "अपडेट किया", "업데이트됨", "Uppfært", "Mise à jour"),
    "Plaque Assay": _row(
        "Plackanalys", "Plaque-Assay", "Ensayo de placas", "空斑分析",
        "Ensaio de placas", "प्लाक परीक्षण", "플라크 분석", "Skellugreining",
        "Test de plaques"),
    "Recruitment": _row(
        "Rekryteringsanalys", "Rekrutierungsanalyse",
        "Ensayo de reclutamiento", "募集分析", "Ensaio de recrutamento",
        "रिक्रूटमेंट विश्लेषण", "리크루트먼트 분석", "Söfnunargreining",
        "Test de recrutement"),
    "Invasion Assay": _row(
        "Invasionsanalys", "Invasionsassay", "Ensayo de invasión", "侵袭分析",
        "Ensaio de invasão", "आक्रमण परीक्षण", "침입 분석", "Innrásarpróf",
        "Test d’invasion"),
    "Replication Assay": _row(
        "Replikationsanalys", "Replikationsassay", "Ensayo de replicación",
        "复制分析", "Ensaio de replicação", "प्रतिकृति परीक्षण",
        "복제 분석", "Fjölgunarpróf", "Test de réplication"),

    # Registry sections.
    "Core": _row(
        "Kärna", "Kern", "Principal", "核心", "Principal",
        "मुख्य", "핵심", "Kjarni", "Cœur"),
    "Data": _row(
        "Data", "Daten", "Datos", "数据", "Dados",
        "डेटा", "데이터", "Gögn", "Données"),
    "Segmentation models": _row(
        "Segmenteringsmodeller", "Segmentierungsmodelle",
        "Modelos de segmentación", "分割模型", "Modelos de segmentação",
        "खंडन मॉडल", "분할 모델", "Hlutunarlíkön",
        "Modèles de segmentation"),
    "Toxoplasma": _row(
        "Toxoplasma", "Toxoplasma", "Toxoplasma", "弓形虫",
        "Toxoplasma", "टोक्सोप्लाज़्मा", "톡소플라스마",
        "Toxoplasma", "Toxoplasma"),
    # Data Manager registers itself but predates the `translations=`
    # keyword on `register_app`, so its nine live here rather than beside
    # its screen. Moving them into that call is a two-line change in
    # `spacr/qt/screens/data_manager.py` and deleting this row.
    "Data Manager": _row(
        "Datahanterare", "Datenverwaltung", "Gestor de datos", "数据管理器",
        "Gerenciador de dados", "डेटा प्रबंधक", "데이터 관리자",
        "Gagnastjóri", "Gestionnaire de données"),
    # Declared in `SECTION_ORDER` and drawn the day their first app
    # registers. Written here when the section was named rather than when
    # its tab appeared, for the same reason `_SECTION_NOTE_LIBRARY` is:
    # the first module to claim an empty section must get a described,
    # translated tab, not an English heading in nine languages.
    "Explore": _row(
        "Utforska", "Erkunden", "Explorar", "探索", "Explorar",
        "अन्वेषण", "탐색", "Kanna", "Explorer"),
    "Design": _row(
        "Design", "Entwurf", "Diseño", "实验设计", "Planejamento",
        "डिज़ाइन", "설계", "Hönnun", "Conception"),
    "Core Pipeline": _row(
        "Kärnflöde", "Kernpipeline", "Flujo principal", "核心流程",
        "Fluxo principal", "मुख्य पाइपलाइन", "핵심 파이프라인",
        "Kjarnavinnsla", "Flux principal"),
    "Data & Batch": _row(
        "Data och batch", "Daten und Stapel", "Datos y lotes", "数据与批处理",
        "Dados e lote", "डेटा और बैच", "데이터 및 배치",
        "Gögn og lotur", "Données et lots"),
    "Segmentation Models": _row(
        "Segmenteringsmodeller", "Segmentierungsmodelle",
        "Modelos de segmentación", "分割模型", "Modelos de segmentação",
        "खंडन मॉडल", "분할 모델", "Hlutunarlíkön",
        "Modèles de segmentation"),
    "Results & QC": _row(
        "Resultat och QC", "Ergebnisse und QC", "Resultados y CC",
        "结果与质控", "Resultados e CQ", "परिणाम और QC",
        "결과 및 QC", "Niðurstöður og gæðaeftirlit",
        "Résultats et CQ"),
    "Toxoplasma Assays": _row(
        "Toxoplasma-analyser", "Toxoplasma-Assays", "Ensayos de Toxoplasma",
        "弓形虫分析", "Ensaios de Toxoplasma", "टोक्सोप्लाज़्मा परीक्षण",
        "톡소플라스마 분석", "Toxoplasma-próf", "Tests Toxoplasma"),

    # Preferences and frequent dialog text.
    "spaCR — Preferences": _row(
        "spaCR — Inställningar", "spaCR — Einstellungen",
        "spaCR — Preferencias", "spaCR — 首选项", "spaCR — Preferências",
        "spaCR — प्राथमिकताएँ", "spaCR — 환경설정", "spaCR — Stillingar",
        "spaCR — Préférences"),
    "Theme": _row(
        "Tema", "Design", "Tema", "主题", "Tema",
        "थीम", "테마", "Þema", "Thème"),
    "Font scale": _row(
        "Textskala", "Schriftgröße", "Escala de fuente", "字体缩放",
        "Escala da fonte", "फ़ॉन्ट आकार", "글꼴 배율", "Leturstærð",
        "Échelle de police"),
    "App dock": _row(
        "Appanel", "App-Leiste", "Panel de aplicaciones", "应用栏",
        "Painel de aplicativos", "ऐप पैनल", "앱 패널", "Forritaspjald",
        "Volet des applications"),
    "Reveal on hover": _row(
        "Visa vid hovring", "Beim Überfahren anzeigen",
        "Mostrar al pasar el cursor", "悬停时显示", "Mostrar ao passar o mouse",
        "होवर पर दिखाएँ", "마우스를 올리면 표시", "Sýna við yfirferð",
        "Afficher au survol"),
    "Locked open": _row(
        "Låst öppen", "Offen fixiert", "Fijado abierto", "固定打开",
        "Fixado aberto", "खुला रखें", "열린 상태로 고정", "Læst opið",
        "Verrouillé ouvert"),
    "Hidden": _row(
        "Dold", "Ausgeblendet", "Oculto", "隐藏", "Oculto",
        "छिपा हुआ", "숨김", "Falið", "Masqué"),
    "Page opacity": _row(
        "Sidopacitet", "Seitendeckkraft", "Opacidad de página", "页面不透明度",
        "Opacidade da página", "पृष्ठ अपारदर्शिता", "페이지 불투명도",
        "Ógagnsæi síðu", "Opacité de page"),
    "Colour-blind mode": _row(
        "Färgblindhetsläge", "Farbenblindmodus", "Modo daltonismo",
        "色觉辅助模式", "Modo daltonismo", "रंग-दृष्टि मोड",
        "색각 보정 모드", "Litblinduviðmót", "Mode daltonien"),
    "Off": _row(
        "Av", "Aus", "Desactivado", "关闭", "Desativado",
        "बंद", "꺼짐", "Slökkt", "Désactivé"),
    "Diagnostics": _row(
        "Diagnostik", "Diagnose", "Diagnóstico", "诊断", "Diagnóstico",
        "निदान", "진단", "Greining", "Diagnostic"),
    "Enable verbose logging": _row(
        "Aktivera detaljerad loggning", "Ausführliche Protokollierung",
        "Activar registro detallado", "启用详细日志",
        "Ativar logs detalhados", "विस्तृत लॉगिंग चालू करें",
        "상세 로깅 활성화", "Virkja ítarlega annála",
        "Activer la journalisation détaillée"),
    "Allow editing in the Database Browser": _row(
        "Tillåt redigering i databasutforskaren",
        "Bearbeiten im Datenbankbrowser erlauben",
        "Permitir edición en el explorador de base de datos",
        "允许在数据库浏览器中编辑",
        "Permitir edição no navegador de banco de dados",
        "डेटाबेस ब्राउज़र में संपादन की अनुमति दें",
        "데이터베이스 브라우저에서 편집 허용",
        "Leyfa breytingar í gagnagrunnsvafra",
        "Autoriser la modification dans l’explorateur de base de données"),
    "Module visibility": _row(
        "Funktionsmognad", "Funktionsreife", "Madurez de funciones",
        "功能成熟度", "Maturidade dos recursos", "फ़ीचर परिपक्वता",
        "기능 성숙도", "Þroski eiginleika", "Maturité des fonctions"),
    "Show Alpha modules and settings": _row(
        "Visa Alpha-moduler och inställningar",
        "Alpha-Module und -Einstellungen anzeigen",
        "Mostrar módulos y ajustes Alpha", "显示 Alpha 模块和设置",
        "Mostrar módulos e configurações Alpha",
        "Alpha मॉड्यूल और सेटिंग्स दिखाएँ",
        "Alpha 모듈 및 설정 표시", "Sýna Alpha-einingar og stillingar",
        "Afficher les modules et paramètres Alpha"),
    "Show Beta modules and settings": _row(
        "Visa Beta-moduler och inställningar",
        "Beta-Module und -Einstellungen anzeigen",
        "Mostrar módulos y ajustes Beta", "显示 Beta 模块和设置",
        "Mostrar módulos e configurações Beta",
        "Beta मॉड्यूल और सेटिंग्स दिखाएँ",
        "Beta 모듈 및 설정 표시", "Sýna Beta-einingar og stillingar",
        "Afficher les modules et paramètres Bêta"),
    "Figure format": _row(
        "Figurformat", "Abbildungsformat", "Formato de figura", "图形格式",
        "Formato da figura", "चित्र प्रारूप", "그림 형식", "Myndsnið",
        "Format de figure"),
    "PNG resolution": _row(
        "PNG-upplösning", "PNG-Auflösung", "Resolución PNG", "PNG 分辨率",
        "Resolução PNG", "PNG रिज़ॉल्यूशन", "PNG 해상도", "PNG-upplausn",
        "Résolution PNG"),

    # AI/chat, console presentation and linked-help chrome. These strings are
    # deliberately UI-only: provider replies, pipeline stdout, tracebacks,
    # paths and generated outputs never pass through the translator.
    "Hover a tile to see what it does.": _row(
        "Håll pekaren över en modul för att se vad den gör.",
        "Bewegen Sie den Zeiger über ein Modul, um seine Funktion zu sehen.",
        "Pase el cursor sobre un módulo para ver qué hace.",
        "将指针悬停在模块上可查看其功能。",
        "Passe o ponteiro sobre um módulo para ver o que ele faz.",
        "मॉड्यूल का कार्य देखने के लिए उस पर पॉइंटर रखें।",
        "모듈 위에 포인터를 올려 기능을 확인하세요.",
        "Haltu bendlinum yfir einingu til að sjá hvað hún gerir.",
        "Survolez un module pour voir ce qu’il fait."),
    "AI": _row(
        "AI", "KI", "IA", "人工智能", "IA",
        "एआई", "AI", "Gervigreind", "IA"),
    "Live": _row(
        "Live", "Live", "En vivo", "实时", "Ao vivo",
        "लाइव", "실시간", "Beint", "Direct"),
    "Provider": _row(
        "Leverantör", "Anbieter", "Proveedor", "提供商", "Provedor",
        "प्रदाता", "제공업체", "Veitandi", "Fournisseur"),
    "Providers": _row(
        "Leverantörer", "Anbieter", "Proveedores", "提供商", "Provedores",
        "प्रदाता", "제공업체", "Veitendur", "Fournisseurs"),
    "Providers…": _row(
        "Leverantörer…", "Anbieter…", "Proveedores…", "提供商…",
        "Provedores…", "प्रदाता…", "제공업체…", "Veitendur…",
        "Fournisseurs…"),
    "Send": _row(
        "Skicka", "Senden", "Enviar", "发送", "Enviar",
        "भेजें", "보내기", "Senda", "Envoyer"),
    "Clear": _row(
        "Rensa", "Leeren", "Limpiar", "清除", "Limpar",
        "साफ़ करें", "지우기", "Hreinsa", "Effacer"),
    "Copy": _row(
        "Kopiera", "Kopieren", "Copiar", "复制", "Copiar",
        "कॉपी करें", "복사", "Afrita", "Copier"),
    "Install:": _row(
        "Installera:", "Installieren:", "Instalar:", "安装：", "Instalar:",
        "इंस्टॉल करें:", "설치:", "Setja upp:", "Installer :"),
    "Login:": _row(
        "Logga in:", "Anmelden:", "Iniciar sesión:", "登录：", "Entrar:",
        "लॉग इन:", "로그인:", "Innskráning:", "Connexion :"),
    "installed": _row(
        "installerad", "installiert", "instalado", "已安装", "instalado",
        "इंस्टॉल है", "설치됨", "uppsett", "installé"),
    "missing": _row(
        "saknas", "fehlt", "falta", "缺失", "ausente",
        "अनुपलब्ध", "없음", "vantar", "manquant"),
    "Font size": _row(
        "Textstorlek", "Schriftgröße", "Tamaño de fuente", "字体大小",
        "Tamanho da fonte", "फ़ॉन्ट आकार", "글꼴 크기", "Leturstærð",
        "Taille de police"),
    "Console font size": _row(
        "Konsolens textstorlek", "Konsolenschriftgröße",
        "Tamaño de fuente de la consola", "控制台字体大小",
        "Tamanho da fonte do console", "कंसोल फ़ॉन्ट आकार",
        "콘솔 글꼴 크기", "Leturstærð stjórnborðs",
        "Taille de police de la console"),
    "spaCR output": _row(
        "spaCR-utdata", "spaCR-Ausgabe", "salida de spaCR", "spaCR 输出",
        "saída do spaCR", "spaCR आउटपुट", "spaCR 출력", "spaCR-úttak",
        "sortie spaCR"),
    "spaCR ERROR": _row(
        "spaCR-FEL", "spaCR-FEHLER", "ERROR de spaCR", "spaCR 错误",
        "ERRO do spaCR", "spaCR त्रुटि", "spaCR 오류", "spaCR-VILLA",
        "ERREUR spaCR"),
    "spaCR user": _row(
        "spaCR-användare", "spaCR-Benutzer", "usuario de spaCR",
        "spaCR 用户", "usuário do spaCR", "spaCR उपयोगकर्ता",
        "spaCR 사용자", "spaCR-notandi", "utilisateur spaCR"),
    "spaCR AI": _row(
        "spaCR AI", "spaCR-KI", "IA de spaCR", "spaCR 人工智能",
        "IA do spaCR", "spaCR एआई", "spaCR AI", "spaCR-gervigreind",
        "IA spaCR"),
    "Open spaCR API documentation": _row(
        "Öppna spaCR:s API-dokumentation",
        "spaCR-API-Dokumentation öffnen",
        "Abrir la documentación de la API de spaCR",
        "打开 spaCR API 文档", "Abrir a documentação da API do spaCR",
        "spaCR API दस्तावेज़ खोलें", "spaCR API 문서 열기",
        "Opna API-skjölun spaCR", "Ouvrir la documentation de l’API spaCR"),
    "Open API reference for {name}": _row(
        "Öppna API-referens för {name}",
        "API-Referenz für {name} öffnen",
        "Abrir la referencia de la API para {name}",
        "打开 {name} 的 API 参考", "Abrir a referência da API para {name}",
        "{name} के लिए API संदर्भ खोलें", "{name} API 참조 열기",
        "Opna API-tilvísun fyrir {name}",
        "Ouvrir la référence API de {name}"),
    "API: {url}": _row(
        "API: {url}", "API: {url}", "API: {url}", "API：{url}",
        "API: {url}", "API: {url}", "API: {url}", "API: {url}",
        "API : {url}"),
    "integer": _row(
        "heltal", "Ganzzahl", "entero", "整数", "inteiro",
        "पूर्णांक", "정수", "heiltala", "entier"),
    "float": _row(
        "decimaltal", "Gleitkommazahl", "decimal", "浮点数", "decimal",
        "दशमलव", "실수", "fleytitala", "nombre décimal"),
    "boolean": _row(
        "boolesk", "boolesch", "booleano", "布尔值", "booleano",
        "बूलियन", "불리언", "rökbreyta", "booléen"),
    "string": _row(
        "text", "Zeichenfolge", "texto", "字符串", "texto",
        "स्ट्रिंग", "문자열", "strengur", "chaîne"),
    "list": _row(
        "lista", "Liste", "lista", "列表", "lista",
        "सूची", "목록", "listi", "liste"),
    "tuple": _row(
        "tupel", "Tupel", "tupla", "元组", "tupla",
        "ट्यूपल", "튜플", "tvennd", "tuple"),
    "dictionary": _row(
        "ordbok", "Wörterbuch", "diccionario", "字典", "dicionário",
        "शब्दकोश", "사전", "orðabók", "dictionnaire"),
    "optional": _row(
        "valfri", "optional", "opcional", "可选", "opcional",
        "वैकल्पिक", "선택 사항", "valfrjálst", "facultatif"),
    "Controls this setting.": _row(
        "Styr den här inställningen.", "Steuert diese Einstellung.",
        "Controla este ajuste.", "控制此设置。", "Controla esta configuração.",
        "इस सेटिंग को नियंत्रित करता है।", "이 설정을 제어합니다.",
        "Stýrir þessari stillingu.", "Contrôle ce paramètre."),
    "Type here and hit Enter…  (toggle AI at the bottom-right to route through your chat subscription)": _row(
        "Skriv här och tryck Enter…  (slå på AI nere till höger för att skicka via din chattprenumeration)",
        "Hier eingeben und Enter drücken…  (KI unten rechts aktivieren, um über Ihr Chat-Abonnement zu senden)",
        "Escriba aquí y pulse Intro…  (active la IA abajo a la derecha para enviar mediante su suscripción de chat)",
        "在此输入并按回车…（打开右下角的人工智能，通过您的聊天订阅发送）",
        "Digite aqui e pressione Enter…  (ative a IA no canto inferior direito para enviar pela sua assinatura de chat)",
        "यहाँ लिखें और Enter दबाएँ…  (अपनी चैट सदस्यता से भेजने के लिए नीचे दाईं ओर एआई चालू करें)",
        "여기에 입력하고 Enter를 누르세요…  (오른쪽 아래 AI를 켜 채팅 구독을 통해 전송)",
        "Sláðu inn hér og ýttu á Enter…  (kveiktu á gervigreind neðst til hægri til að senda með spjalláskriftinni)",
        "Saisissez ici puis appuyez sur Entrée…  (activez l’IA en bas à droite pour envoyer via votre abonnement de chat)"),
    "Ask a question (Enter to send · Shift+Enter for newline)": _row(
        "Ställ en fråga (Enter skickar · Skift+Enter ger ny rad)",
        "Eine Frage stellen (Enter sendet · Umschalt+Enter für neue Zeile)",
        "Haga una pregunta (Intro envía · Mayús+Intro añade una línea)",
        "提出问题（回车发送 · Shift+回车换行）",
        "Faça uma pergunta (Enter envia · Shift+Enter cria nova linha)",
        "प्रश्न पूछें (Enter से भेजें · Shift+Enter से नई पंक्ति)",
        "질문하기 (Enter 전송 · Shift+Enter 줄바꿈)",
        "Spyrðu spurningar (Enter sendir · Shift+Enter fyrir nýja línu)",
        "Posez une question (Entrée envoie · Maj+Entrée ajoute une ligne)"),
    "Ready.": _row(
        "Klar.", "Bereit.", "Listo.", "就绪。", "Pronto.",
        "तैयार।", "준비됨.", "Tilbúið.", "Prêt."),
    "No provider configured.": _row(
        "Ingen leverantör är konfigurerad.",
        "Kein Anbieter konfiguriert.",
        "No hay ningún proveedor configurado.",
        "未配置提供商。", "Nenhum provedor configurado.",
        "कोई प्रदाता कॉन्फ़िगर नहीं है।", "구성된 제공업체가 없습니다.",
        "Enginn veitandi er stilltur.", "Aucun fournisseur configuré."),
    "Cancelling…": _row(
        "Avbryter…", "Wird abgebrochen…", "Cancelando…", "正在取消…",
        "Cancelando…", "रद्द किया जा रहा है…", "취소 중…", "Hætti við…",
        "Annulation…"),
    "Connecting to {provider}…": _row(
        "Ansluter till {provider}…", "Verbindung zu {provider}…",
        "Conectando con {provider}…", "正在连接 {provider}…",
        "Conectando a {provider}…", "{provider} से कनेक्ट हो रहा है…",
        "{provider}에 연결 중…", "Tengist {provider}…",
        "Connexion à {provider}…"),
    "Streaming from {provider}…": _row(
        "Strömmar från {provider}…", "Stream von {provider}…",
        "Recibiendo de {provider}…", "正在接收 {provider} 的响应…",
        "Recebendo de {provider}…", "{provider} से उत्तर आ रहा है…",
        "{provider} 응답 수신 중…", "Streymir frá {provider}…",
        "Réception depuis {provider}…"),
    "Click to toggle AI. When ON (blue), pressing Enter in the console routes your message through your chat subscription via the selected provider.": _row(
        "Klicka för att slå på eller av AI. När den är PÅ (blå) skickas meddelandet i konsolen via din chattprenumeration och vald leverantör när du trycker Enter.",
        "Klicken, um die KI ein- oder auszuschalten. Wenn sie EIN (blau) ist, wird Ihre Konsolennachricht beim Drücken der Eingabetaste über Ihr Chat-Abonnement und den gewählten Anbieter gesendet.",
        "Haga clic para activar o desactivar la IA. Cuando está ACTIVADA (azul), al pulsar Intro el mensaje de la consola se envía mediante su suscripción de chat y el proveedor seleccionado.",
        "点击以打开或关闭人工智能。打开（蓝色）后，在控制台中按回车会通过您的聊天订阅和所选提供商发送消息。",
        "Clique para ativar ou desativar a IA. Quando ATIVA (azul), pressionar Enter no console envia a mensagem pela sua assinatura de chat e pelo provedor selecionado.",
        "एआई चालू या बंद करने के लिए क्लिक करें। चालू (नीला) होने पर कंसोल में Enter दबाने से संदेश आपकी चैट सदस्यता और चुने हुए प्रदाता के माध्यम से भेजा जाता है।",
        "AI를 켜거나 끄려면 클릭하세요. 켜짐(파란색) 상태에서 콘솔의 Enter를 누르면 선택한 제공업체와 채팅 구독을 통해 메시지가 전송됩니다.",
        "Smelltu til að kveikja eða slökkva á gervigreind. Þegar hún er KVEIKT (blá) sendir Enter skilaboðin í stjórnborðinu með spjalláskriftinni og valda veitandanum.",
        "Cliquez pour activer ou désactiver l’IA. Lorsqu’elle est ACTIVÉE (bleu), appuyer sur Entrée dans la console envoie le message via votre abonnement de chat et le fournisseur sélectionné."),
    "Click to toggle Live Preview. When ON (blue), the interactive Cellpose preview appears above the console.": _row(
        "Klicka för att visa eller dölja liveförhandsvisningen. När den är PÅ (blå) visas den interaktiva Cellpose-förhandsvisningen ovanför konsolen.",
        "Klicken, um die Live-Vorschau ein- oder auszublenden. Wenn sie EIN (blau) ist, erscheint die interaktive Cellpose-Vorschau über der Konsole.",
        "Haga clic para mostrar u ocultar la vista previa en vivo. Cuando está ACTIVADA (azul), la vista interactiva de Cellpose aparece sobre la consola.",
        "点击以显示或隐藏实时预览。打开（蓝色）后，交互式 Cellpose 预览会显示在控制台上方。",
        "Clique para mostrar ou ocultar a pré-visualização ao vivo. Quando ATIVA (azul), a pré-visualização interativa do Cellpose aparece acima do console.",
        "लाइव पूर्वावलोकन दिखाने या छिपाने के लिए क्लिक करें। चालू (नीला) होने पर इंटरैक्टिव Cellpose पूर्वावलोकन कंसोल के ऊपर दिखाई देता है।",
        "실시간 미리보기를 표시하거나 숨기려면 클릭하세요. 켜짐(파란색)일 때 대화형 Cellpose 미리보기가 콘솔 위에 나타납니다.",
        "Smelltu til að sýna eða fela lifandi forskoðun. Þegar hún er KVEIKT (blá) birtist gagnvirk Cellpose-forskoðun fyrir ofan stjórnborðið.",
        "Cliquez pour afficher ou masquer l’aperçu en direct. Lorsqu’il est ACTIVÉ (bleu), l’aperçu Cellpose interactif apparaît au-dessus de la console."),
    "Click to toggle Track Preview for the timelapse.": _row(
        "Klicka för att visa eller dölja spårförhandsvisningen för tidsserien.",
        "Klicken, um die Spurvorschau für die Zeitreihe ein- oder auszublenden.",
        "Haga clic para mostrar u ocultar la vista previa de trayectorias de la serie temporal.",
        "点击以显示或隐藏时间序列的轨迹预览。",
        "Clique para mostrar ou ocultar a pré-visualização de trajetórias da série temporal.",
        "टाइमलैप्स का ट्रैक पूर्वावलोकन दिखाने या छिपाने के लिए क्लिक करें।",
        "타임랩스 추적 미리보기를 표시하거나 숨기려면 클릭하세요.",
        "Smelltu til að sýna eða fela rakningarforskoðun tímaraðarinnar.",
        "Cliquez pour afficher ou masquer l’aperçu des trajectoires de la série temporelle."),
    "Click to toggle Track Preview for the motility analysis.": _row(
        "Klicka för att visa eller dölja spårförhandsvisningen för motilitetsanalysen.",
        "Klicken, um die Spurvorschau für die Motilitätsanalyse ein- oder auszublenden.",
        "Haga clic para mostrar u ocultar la vista previa de trayectorias del análisis de motilidad.",
        "点击以显示或隐藏运动分析的轨迹预览。",
        "Clique para mostrar ou ocultar a pré-visualização de trajetórias da análise de motilidade.",
        "गतिशीलता विश्लेषण का ट्रैक पूर्वावलोकन दिखाने या छिपाने के लिए क्लिक करें।",
        "운동성 분석의 추적 미리보기를 표시하거나 숨기려면 클릭하세요.",
        "Smelltu til að sýna eða fela rakningarforskoðun hreyfanleikagreiningarinnar.",
        "Cliquez pour afficher ou masquer l’aperçu des trajectoires de l’analyse de motilité."),
    "Click to toggle Measurement Preview.": _row(
        "Klicka för att visa eller dölja mätningsförhandsvisningen.",
        "Klicken, um die Messvorschau ein- oder auszublenden.",
        "Haga clic para mostrar u ocultar la vista previa de mediciones.",
        "点击以显示或隐藏测量预览。",
        "Clique para mostrar ou ocultar a pré-visualização das medições.",
        "मापन पूर्वावलोकन दिखाने या छिपाने के लिए क्लिक करें।",
        "측정 미리보기를 표시하거나 숨기려면 클릭하세요.",
        "Smelltu til að sýna eða fela mælingaforskoðun.",
        "Cliquez pour afficher ou masquer l’aperçu des mesures."),
    "Toggle the interactive image UMAP. When ON (blue), click a point to preview its image, draw around a cluster, and write manual or automatic labels to the database.": _row(
        "Slå på eller av den interaktiva bild-UMAP-vyn. När den är PÅ (blå) kan du klicka på en punkt för att förhandsvisa bilden, rita runt ett kluster och skriva manuella eller automatiska etiketter till databasen.",
        "Schaltet die interaktive Bild-UMAP-Ansicht ein oder aus. Wenn sie EIN (blau) ist, können Sie einen Punkt zur Bildvorschau anklicken, einen Cluster umzeichnen und manuelle oder automatische Beschriftungen in die Datenbank schreiben.",
        "Active o desactive el UMAP interactivo de imágenes. Cuando está ACTIVADO (azul), puede pulsar un punto para previsualizar su imagen, dibujar alrededor de un clúster y guardar etiquetas manuales o automáticas en la base de datos.",
        "打开或关闭交互式图像 UMAP。打开（蓝色）后，可点击点预览图像、圈选聚类，并将手动或自动标签写入数据库。",
        "Ativa ou desativa o UMAP interativo de imagens. Quando ATIVO (azul), você pode clicar em um ponto para pré-visualizar a imagem, contornar um cluster e gravar rótulos manuais ou automáticos no banco de dados.",
        "इंटरैक्टिव इमेज UMAP चालू या बंद करें। चालू (नीला) होने पर किसी बिंदु पर क्लिक करके उसकी छवि देखें, क्लस्टर के चारों ओर रेखा बनाएँ और मैन्युअल या स्वचालित लेबल डेटाबेस में लिखें।",
        "대화형 이미지 UMAP을 켜거나 끕니다. 켜짐(파란색)일 때 점을 클릭해 이미지를 미리 보고, 군집을 둘러 그린 뒤 수동 또는 자동 레이블을 데이터베이스에 기록할 수 있습니다.",
        "Kveikir eða slekkur á gagnvirku mynd-UMAP. Þegar það er KVEIKT (blátt) geturðu smellt á punkt til að forskoða myndina, teiknað utan um klasa og skrifað handvirk eða sjálfvirk merki í gagnagrunninn.",
        "Active ou désactive l’UMAP d’images interactif. Lorsqu’il est ACTIVÉ (bleu), cliquez sur un point pour prévisualiser son image, entourez un groupe et écrivez des étiquettes manuelles ou automatiques dans la base de données."),
    "Pick provider · Providers…": _row(
        "Välj leverantör · Leverantörer…",
        "Anbieter wählen · Anbieter…",
        "Elegir proveedor · Proveedores…",
        "选择提供商 · 提供商…", "Escolher provedor · Provedores…",
        "प्रदाता चुनें · प्रदाता…", "제공업체 선택 · 제공업체…",
        "Veldu veitanda · Veitendur…",
        "Choisir un fournisseur · Fournisseurs…"),
    "Drag to resize this console section. Double-click for auto height.": _row(
        "Dra för att ändra storlek på den här konsolsektionen. Dubbelklicka för automatisk höjd.",
        "Ziehen, um die Größe dieses Konsolenabschnitts zu ändern. Doppelklicken für automatische Höhe.",
        "Arrastre para cambiar el tamaño de esta sección de la consola. Haga doble clic para ajustar la altura automáticamente.",
        "拖动以调整此控制台区域的大小。双击以自动调整高度。",
        "Arraste para redimensionar esta seção do console. Clique duas vezes para altura automática.",
        "इस कंसोल अनुभाग का आकार बदलने के लिए खींचें। स्वचालित ऊँचाई के लिए डबल-क्लिक करें।",
        "이 콘솔 영역의 크기를 조정하려면 드래그하세요. 자동 높이는 두 번 클릭하세요.",
        "Dragðu til að breyta stærð þessa stjórnborðshluta. Tvísmelltu fyrir sjálfvirka hæð.",
        "Faites glisser pour redimensionner cette section de console. Double-cliquez pour une hauteur automatique."),

    # Exact messages authored by spaCR. Dynamic values are deliberately kept
    # as placeholders so paths, provider names, tracebacks and function names
    # remain verbatim after formatting.
    "[AI] No provider configured. Open Providers…": _row(
        "[AI] Ingen leverantör är konfigurerad. Öppna Leverantörer…",
        "[KI] Kein Anbieter konfiguriert. Öffnen Sie Anbieter…",
        "[IA] No hay ningún proveedor configurado. Abra Proveedores…",
        "[人工智能] 未配置提供商。请打开“提供商”…",
        "[IA] Nenhum provedor configurado. Abra Provedores…",
        "[एआई] कोई प्रदाता कॉन्फ़िगर नहीं है। प्रदाता… खोलें।",
        "[AI] 구성된 제공업체가 없습니다. 제공업체…를 여세요.",
        "[Gervigreind] Enginn veitandi er stilltur. Opnaðu Veitendur…",
        "[IA] Aucun fournisseur configuré. Ouvrez Fournisseurs…"),
    "(empty response — try again or switch provider)": _row(
        "(tomt svar — försök igen eller byt leverantör)",
        "(leere Antwort – versuchen Sie es erneut oder wechseln Sie den Anbieter)",
        "(respuesta vacía — inténtelo de nuevo o cambie de proveedor)",
        "（响应为空——请重试或更换提供商）",
        "(resposta vazia — tente novamente ou troque de provedor)",
        "(खाली उत्तर — फिर कोशिश करें या प्रदाता बदलें)",
        "(빈 응답 — 다시 시도하거나 제공업체를 변경하세요)",
        "(tómt svar — reyndu aftur eða skiptu um veitanda)",
        "(réponse vide — réessayez ou changez de fournisseur)"),
    "[AI error] {detail}": _row(
        "[AI-fel] {detail}", "[KI-Fehler] {detail}",
        "[error de IA] {detail}", "[人工智能错误] {detail}",
        "[erro de IA] {detail}", "[एआई त्रुटि] {detail}",
        "[AI 오류] {detail}", "[villa í gervigreind] {detail}",
        "[erreur IA] {detail}"),
    "[AI] Enable AI in the actions row + pick a provider first.": _row(
        "[AI] Slå på AI på åtgärdsraden och välj först en leverantör.",
        "[KI] Aktivieren Sie zuerst die KI in der Aktionsleiste und wählen Sie einen Anbieter.",
        "[IA] Active primero la IA en la fila de acciones y elija un proveedor.",
        "[人工智能] 请先在操作栏中启用人工智能并选择提供商。",
        "[IA] Primeiro ative a IA na barra de ações e escolha um provedor.",
        "[एआई] पहले कार्रवाई पंक्ति में एआई चालू करें और प्रदाता चुनें।",
        "[AI] 먼저 작업 행에서 AI를 켜고 제공업체를 선택하세요.",
        "[Gervigreind] Kveiktu fyrst á gervigreind í aðgerðalínunni og veldu veitanda.",
        "[IA] Activez d’abord l’IA dans la barre d’actions et choisissez un fournisseur."),
    "An error occurred — asking spaCR AI to explain it. (Ask the AI to \"show the raw error\" to see the traceback.)": _row(
        "Ett fel inträffade — spaCR AI ombeds att förklara det. (Be AI att \"visa det råa felet\" för att se stackspårningen.)",
        "Ein Fehler ist aufgetreten – spaCR-KI wird um eine Erklärung gebeten. (Bitten Sie die KI, den \"Rohfehler anzuzeigen\", um den Traceback zu sehen.)",
        "Se produjo un error — se pedirá a la IA de spaCR que lo explique. (Pida a la IA que \"muestre el error sin procesar\" para ver el rastreo.)",
        "发生错误——正在请 spaCR 人工智能进行解释。（请要求人工智能“显示原始错误”以查看回溯。）",
        "Ocorreu um erro — a IA do spaCR vai explicá-lo. (Peça à IA para \"mostrar o erro bruto\" e ver o rastreamento.)",
        "एक त्रुटि हुई — spaCR एआई से उसकी व्याख्या पूछी जा रही है। (ट्रेसबैक देखने के लिए एआई से \"कच्ची त्रुटि दिखाओ\" कहें।)",
        "오류가 발생했습니다. spaCR AI에게 설명을 요청합니다. (트레이스백을 보려면 AI에게 \"원시 오류를 보여 줘\"라고 요청하세요.)",
        "Villa kom upp — spaCR-gervigreind er beðin um að útskýra hana. (Biddu gervigreindina að \"sýna hráu villuna\" til að sjá rakninguna.)",
        "Une erreur s’est produite — l’IA spaCR va l’expliquer. (Demandez à l’IA d’\"afficher l’erreur brute\" pour voir la trace.)"),

    "(no vendor CLI installed)": _row(
        "(inget leverantörs-CLI installerat)",
        "(keine Anbieter-CLI installiert)",
        "(no hay ninguna CLI de proveedor instalada)",
        "（未安装任何提供商 CLI）",
        "(nenhuma CLI de provedor instalada)",
        "(कोई प्रदाता CLI इंस्टॉल नहीं है)",
        "(설치된 제공업체 CLI 없음)",
        "(ekkert CLI veitanda uppsett)",
        "(aucune CLI de fournisseur installée)"),
    "[AI] No vendor CLI installed. Click ▾ next to the AI switch → Providers…": _row(
        "[AI] Inget leverantörs-CLI är installerat. Klicka på ▾ bredvid AI-reglaget → Leverantörer…",
        "[KI] Keine Anbieter-CLI installiert. Klicken Sie neben dem KI-Schalter auf ▾ → Anbieter…",
        "[IA] No hay ninguna CLI de proveedor instalada. Pulse ▾ junto al interruptor de IA → Proveedores…",
        "[人工智能] 未安装提供商 CLI。请点击人工智能开关旁的 ▾ → 提供商…",
        "[IA] Nenhuma CLI de provedor instalada. Clique em ▾ ao lado do controle de IA → Provedores…",
        "[एआई] कोई प्रदाता CLI इंस्टॉल नहीं है। एआई स्विच के बगल में ▾ क्लिक करें → प्रदाता…",
        "[AI] 설치된 제공업체 CLI가 없습니다. AI 스위치 옆의 ▾를 클릭하세요 → 제공업체…",
        "[Gervigreind] Ekkert CLI veitanda er uppsett. Smelltu á ▾ við hlið gervigreindarrofa → Veitendur…",
        "[IA] Aucune CLI de fournisseur n’est installée. Cliquez sur ▾ à côté de l’interrupteur IA → Fournisseurs…"),
    "→ Starting {module} ({function}) with src={src} + {count} settings…": _row(
        "→ Startar {module} ({function}) med src={src} + {count} inställningar…",
        "→ Starte {module} ({function}) mit src={src} + {count} Einstellungen…",
        "→ Iniciando {module} ({function}) con src={src} + {count} ajustes…",
        "→ 正在启动 {module} ({function})，src={src} + {count} 项设置…",
        "→ Iniciando {module} ({function}) com src={src} + {count} configurações…",
        "→ {module} ({function}) src={src} + {count} सेटिंग्स के साथ शुरू हो रहा है…",
        "→ {module} ({function}) 시작 중: src={src} + 설정 {count}개…",
        "→ Ræsi {module} ({function}) með src={src} + {count} stillingum…",
        "→ Démarrage de {module} ({function}) avec src={src} + {count} paramètres…"),
    "■ Stopped safely at a field, trial, or job boundary": _row(
        "■ Stoppade säkert vid en fält-, försöks- eller jobbgräns",
        "■ Sicher an einer Feld-, Versuchs- oder Auftragsgrenze gestoppt",
        "■ Detenido de forma segura en el límite de un campo, ensayo o trabajo",
        "■ 已在视野、试验或作业边界安全停止",
        "■ Interrompido com segurança no limite de um campo, ensaio ou trabalho",
        "■ फ़ील्ड, ट्रायल या जॉब सीमा पर सुरक्षित रूप से रोका गया",
        "■ 필드, 시험 또는 작업 경계에서 안전하게 중지됨",
        "■ Stöðvað örugglega við mörk sviðs, tilraunar eða verks",
        "■ Arrêt sécurisé à la limite d’un champ, d’un essai ou d’une tâche"),
    "✓ Finished": _row(
        "✓ Klar", "✓ Abgeschlossen", "✓ Finalizado", "✓ 已完成",
        "✓ Concluído", "✓ पूरा हुआ", "✓ 완료", "✓ Lokið",
        "✓ Terminé"),
    "✗ Failed — see traceback above": _row(
        "✗ Misslyckades — se stackspårningen ovan",
        "✗ Fehlgeschlagen – siehe Traceback oben",
        "✗ Falló — consulte el rastreo anterior",
        "✗ 失败——请查看上方回溯",
        "✗ Falhou — consulte o rastreamento acima",
        "✗ विफल — ऊपर ट्रेसबैक देखें",
        "✗ 실패 — 위의 트레이스백을 확인하세요",
        "✗ Mistókst — sjá rakningu hér að ofan",
        "✗ Échec — consultez la trace ci-dessus"),
    "Requesting stop. The current field/trial/job will finish, then the resumable run will stop at its next safe boundary.": _row(
        "Begär stopp. Det aktuella fältet, försöket eller jobbet slutförs, sedan stannar den återupptagningsbara körningen vid nästa säkra gräns.",
        "Stopp wird angefordert. Das aktuelle Feld, der Versuch oder Auftrag wird beendet; anschließend stoppt der fortsetzbare Lauf an der nächsten sicheren Grenze.",
        "Solicitando la detención. El campo, ensayo o trabajo actual terminará y la ejecución reanudable se detendrá en el siguiente límite seguro.",
        "正在请求停止。当前视野、试验或作业完成后，可恢复运行将在下一个安全边界停止。",
        "Solicitando a interrupção. O campo, ensaio ou trabalho atual será concluído; depois, a execução retomável parará no próximo limite seguro.",
        "रोकने का अनुरोध किया गया है। वर्तमान फ़ील्ड, ट्रायल या जॉब पूरा होगा, फिर दोबारा शुरू की जा सकने वाली रन अगली सुरक्षित सीमा पर रुकेगी।",
        "중지를 요청했습니다. 현재 필드, 시험 또는 작업이 완료되면 재개 가능한 실행이 다음 안전 경계에서 중지됩니다.",
        "Beðið er um stöðvun. Núverandi svið, tilraun eða verk klárast og síðan stöðvast endurræsanlega keyrslan við næstu öruggu mörk.",
        "Arrêt demandé. Le champ, l’essai ou la tâche en cours va se terminer, puis l’exécution reprenable s’arrêtera à la prochaine limite sûre."),
    "Close deferred: the current field is still finishing. The window will remain open so its worker is not destroyed mid-write; close it again after Stop completes.": _row(
        "Stängning uppskjuten: det aktuella fältet slutförs fortfarande. Fönstret förblir öppet så att arbetaren inte förstörs mitt under en skrivning; stäng det igen när stoppet är klart.",
        "Schließen aufgeschoben: Das aktuelle Feld wird noch abgeschlossen. Das Fenster bleibt geöffnet, damit der Worker nicht während eines Schreibvorgangs zerstört wird; schließen Sie es erneut, nachdem der Stopp abgeschlossen ist.",
        "Cierre aplazado: el campo actual todavía está terminando. La ventana permanecerá abierta para que el proceso no se destruya durante una escritura; ciérrela de nuevo cuando finalice la detención.",
        "已延迟关闭：当前视野仍在收尾。窗口将保持打开，以免工作线程在写入中途被销毁；停止完成后请再次关闭。",
        "Fechamento adiado: o campo atual ainda está terminando. A janela permanecerá aberta para que o processo não seja destruído durante uma gravação; feche-a novamente após a conclusão da interrupção.",
        "बंद करना स्थगित है: वर्तमान फ़ील्ड अभी पूरा हो रहा है। विंडो खुली रहेगी ताकि लिखते समय वर्कर नष्ट न हो; रुकना पूरा होने के बाद इसे फिर बंद करें।",
        "닫기가 연기되었습니다. 현재 필드가 아직 마무리 중입니다. 쓰기 중 작업자가 소멸되지 않도록 창을 열어 둡니다. 중지가 완료된 후 다시 닫으세요.",
        "Lokun frestað: núverandi svið er enn að klárast. Glugginn verður opinn svo vinnsluþráðurinn eyðileggist ekki í miðri skrifun; lokaðu honum aftur þegar stöðvun lýkur.",
        "Fermeture différée : le champ actuel est encore en cours de finalisation. La fenêtre restera ouverte afin que le processus ne soit pas détruit pendant une écriture ; refermez-la une fois l’arrêt terminé."),
    "Loaded {count} settings from {path}": _row(
        "Läste in {count} inställningar från {path}",
        "{count} Einstellungen aus {path} geladen",
        "Se cargaron {count} ajustes desde {path}",
        "已从 {path} 加载 {count} 项设置",
        "{count} configurações carregadas de {path}",
        "{path} से {count} सेटिंग्स लोड हुईं",
        "{path}에서 설정 {count}개를 불러왔습니다",
        "Hlóð {count} stillingum úr {path}",
        "{count} paramètres chargés depuis {path}"),
    "[settings] {note}": _row(
        "[inställningar] {note}", "[Einstellungen] {note}",
        "[ajustes] {note}", "[设置] {note}", "[configurações] {note}",
        "[सेटिंग्स] {note}", "[설정] {note}", "[stillingar] {note}",
        "[paramètres] {note}"),
    "[issue] auto-file failed: {detail}": _row(
        "[ärende] automatisk rapportering misslyckades: {detail}",
        "[Issue] Automatisches Melden fehlgeschlagen: {detail}",
        "[incidencia] falló el envío automático: {detail}",
        "[问题] 自动提交失败：{detail}",
        "[problema] falha no envio automático: {detail}",
        "[इश्यू] स्वचालित रिपोर्ट विफल: {detail}",
        "[이슈] 자동 등록 실패: {detail}",
        "[mál] sjálfvirk skráning mistókst: {detail}",
        "[ticket] échec du signalement automatique : {detail}"),
    "[issue] opened pre-filled report in your browser — review + submit to complete filing.\n{url}...": _row(
        "[ärende] öppnade en förifylld rapport i webbläsaren — granska och skicka för att slutföra rapporteringen.\n{url}...",
        "[Issue] Ein vorausgefüllter Bericht wurde im Browser geöffnet – prüfen und senden Sie ihn, um die Meldung abzuschließen.\n{url}...",
        "[incidencia] se abrió un informe prellenado en el navegador — revíselo y envíelo para completar el registro.\n{url}...",
        "[问题] 已在浏览器中打开预填报告——请审核并提交以完成登记。\n{url}...",
        "[problema] um relatório pré-preenchido foi aberto no navegador — revise e envie para concluir o registro.\n{url}...",
        "[इश्यू] ब्राउज़र में पहले से भरी रिपोर्ट खोली गई — रिपोर्ट पूरी करने के लिए इसकी समीक्षा करके सबमिट करें।\n{url}...",
        "[이슈] 미리 채워진 보고서를 브라우저에서 열었습니다. 검토하고 제출하여 등록을 완료하세요.\n{url}...",
        "[mál] forútyllt skýrsluform var opnað í vafranum — farðu yfir það og sendu til að ljúka skráningu.\n{url}...",
        "[ticket] un rapport prérempli a été ouvert dans votre navigateur — vérifiez-le et envoyez-le pour terminer le signalement.\n{url}..."),

    "Import settings…": _row(
        "Importera inställningar…", "Einstellungen importieren…",
        "Importar ajustes…", "导入设置…", "Importar configurações…",
        "सेटिंग्स आयात करें…", "설정 가져오기…",
        "Flytja inn stillingar…", "Importer des paramètres…"),
    "Clear console": _row(
        "Rensa konsolen", "Konsole leeren", "Limpiar consola", "清空控制台",
        "Limpar console", "कंसोल साफ़ करें", "콘솔 지우기",
        "Hreinsa stjórnborð", "Effacer la console"),
    "File as issue": _row(
        "Rapportera som ärende", "Als Issue melden", "Registrar como incidencia",
        "登记为问题", "Registrar como problema", "इश्यू के रूप में दर्ज करें",
        "이슈로 등록", "Skrá sem mál", "Signaler comme ticket"),
    "Open a pre-filled GitHub issue with the last traceback + environment. You review before submitting. Toggle on/off in AI Settings → Report errors as GitHub issues.": _row(
        "Öppna ett förifyllt GitHub-ärende med den senaste stackspårningen och miljön. Du granskar det innan det skickas. Slå på eller av i AI-inställningar → Rapportera fel som GitHub-ärenden.",
        "Öffnet ein vorausgefülltes GitHub-Issue mit dem letzten Traceback und der Umgebung. Sie prüfen es vor dem Senden. Ein-/ausschalten unter KI-Einstellungen → Fehler als GitHub-Issues melden.",
        "Abra una incidencia de GitHub prellenada con el último rastreo y el entorno. Revísela antes de enviarla. Active o desactive esta opción en Ajustes de IA → Informar errores como incidencias de GitHub.",
        "打开一个预填的 GitHub 问题，其中包含最近的回溯和环境信息。提交前由您审核。可在人工智能设置 → 将错误报告为 GitHub 问题中开启或关闭。",
        "Abre um problema do GitHub pré-preenchido com o último rastreamento e o ambiente. Você o revisa antes de enviar. Ative ou desative em Configurações de IA → Relatar erros como problemas do GitHub.",
        "अंतिम ट्रेसबैक और एनवायरनमेंट के साथ पहले से भरा GitHub इश्यू खोलता है। सबमिट करने से पहले आप इसकी समीक्षा करते हैं। एआई सेटिंग्स → त्रुटियों को GitHub इश्यू के रूप में रिपोर्ट करें में इसे चालू या बंद करें।",
        "최근 트레이스백과 환경 정보가 채워진 GitHub 이슈를 엽니다. 제출 전에 사용자가 검토합니다. AI 설정 → 오류를 GitHub 이슈로 보고에서 켜거나 끄세요.",
        "Opnar forútyllt GitHub-mál með síðustu rakningu og umhverfisupplýsingum. Þú ferð yfir það áður en það er sent. Kveiktu eða slökktu í gervigreindarstillingum → Tilkynna villur sem GitHub-mál.",
        "Ouvre un ticket GitHub prérempli avec la dernière trace et l’environnement. Vous le vérifiez avant de l’envoyer. Activez ou désactivez cette option dans Paramètres de l’IA → Signaler les erreurs comme tickets GitHub."),

    # AI provider/setup dialog and the legacy standalone chat surface. Rich
    # text translations retain the exact markup consumed by Qt.
    "AI Console — providers & settings": _row(
        "AI-konsol — leverantörer och inställningar",
        "KI-Konsole – Anbieter und Einstellungen",
        "Consola de IA — proveedores y ajustes",
        "人工智能控制台 — 提供商与设置",
        "Console de IA — provedores e configurações",
        "एआई कंसोल — प्रदाता और सेटिंग्स",
        "AI 콘솔 — 제공업체 및 설정",
        "Gervigreindarstjórnborð — veitendur og stillingar",
        "Console IA — fournisseurs et paramètres"),
    "The AI Console talks to the <b>vendor coding-agent CLI</b> for each provider, using your chat subscription (Claude.ai Pro, ChatGPT Plus/Pro/Team, Google account).<br><br>For each provider you want to use, install the CLI then log in <em>once</em>. Copy the commands below into a terminal.": _row(
        "AI-konsolen kommunicerar med varje leverantörs <b>CLI för kodningsagenter</b> och använder din chattprenumeration (Claude.ai Pro, ChatGPT Plus/Pro/Team, Google-konto).<br><br>Installera CLI-verktyget för varje leverantör du vill använda och logga sedan in <em>en gång</em>. Kopiera kommandona nedan till en terminal.",
        "Die KI-Konsole kommuniziert über die <b>CLI des Coding-Agent-Anbieters</b> mit jedem Anbieter und verwendet dabei Ihr Chat-Abonnement (Claude.ai Pro, ChatGPT Plus/Pro/Team, Google-Konto).<br><br>Installieren Sie für jeden gewünschten Anbieter die CLI und melden Sie sich dann <em>einmal</em> an. Kopieren Sie die folgenden Befehle in ein Terminal.",
        "La consola de IA se comunica con la <b>CLI del agente de programación del proveedor</b> y utiliza su suscripción de chat (Claude.ai Pro, ChatGPT Plus/Pro/Team, cuenta de Google).<br><br>Para cada proveedor que quiera usar, instale la CLI e inicie sesión <em>una sola vez</em>. Copie los comandos siguientes en una terminal.",
        "人工智能控制台通过各提供商的<b>编码代理 CLI</b>进行通信，并使用您的聊天订阅（Claude.ai Pro、ChatGPT Plus/Pro/Team 或 Google 帐户）。<br><br>对于要使用的每个提供商，请安装 CLI，然后只需登录<em>一次</em>。将下方命令复制到终端中。",
        "O Console de IA se comunica com a <b>CLI do agente de programação do provedor</b> e usa sua assinatura de chat (Claude.ai Pro, ChatGPT Plus/Pro/Team, conta do Google).<br><br>Para cada provedor que você quiser usar, instale a CLI e entre <em>uma vez</em>. Copie os comandos abaixo para um terminal.",
        "एआई कंसोल हर प्रदाता के <b>कोडिंग-एजेंट CLI</b> से आपकी चैट सदस्यता (Claude.ai Pro, ChatGPT Plus/Pro/Team, Google खाता) के ज़रिए संवाद करता है।<br><br>जिस भी प्रदाता का उपयोग करना हो, उसका CLI इंस्टॉल करें और फिर <em>एक बार</em> लॉग इन करें। नीचे दिए कमांड टर्मिनल में कॉपी करें।",
        "AI 콘솔은 각 제공업체의 <b>코딩 에이전트 CLI</b>와 채팅 구독(Claude.ai Pro, ChatGPT Plus/Pro/Team, Google 계정)을 통해 통신합니다.<br><br>사용할 제공업체별로 CLI를 설치한 다음 <em>한 번</em> 로그인하세요. 아래 명령을 터미널에 복사하세요.",
        "Gervigreindarstjórnborðið hefur samskipti við <b>CLI-forritunaraðstoð hvers veitanda</b> og notar spjalláskriftina þína (Claude.ai Pro, ChatGPT Plus/Pro/Team, Google-reikning).<br><br>Settu upp CLI fyrir hvern veitanda sem þú vilt nota og skráðu þig svo inn <em>einu sinni</em>. Afritaðu skipanirnar hér fyrir neðan í skjáhermi.",
        "La console IA communique avec la <b>CLI de l’agent de programmation du fournisseur</b> et utilise votre abonnement de chat (Claude.ai Pro, ChatGPT Plus/Pro/Team, compte Google).<br><br>Pour chaque fournisseur souhaité, installez la CLI puis connectez-vous <em>une seule fois</em>. Copiez les commandes ci-dessous dans un terminal."),
    "Once a CLI is installed <em>and</em> you're logged in, hit <b>Refresh</b> below and it will appear in the provider dropdown.": _row(
        "När ett CLI är installerat <em>och</em> du är inloggad klickar du på <b>Uppdatera</b> nedan, så visas det i leverantörslistan.",
        "Sobald eine CLI installiert ist <em>und</em> Sie angemeldet sind, klicken Sie unten auf <b>Aktualisieren</b>; sie erscheint dann in der Anbieterliste.",
        "Cuando la CLI esté instalada <em>y</em> haya iniciado sesión, pulse <b>Actualizar</b> y aparecerá en la lista de proveedores.",
        "CLI 安装完成<em>且</em>登录后，请点击下方的<b>刷新</b>，它将出现在提供商下拉列表中。",
        "Depois que uma CLI estiver instalada <em>e</em> você tiver entrado, clique em <b>Atualizar</b> abaixo para que ela apareça na lista de provedores.",
        "CLI इंस्टॉल होने <em>और</em> लॉग इन करने के बाद नीचे <b>ताज़ा करें</b> दबाएँ; वह प्रदाता सूची में दिखेगा।",
        "CLI를 설치하고 <em>로그인한</em> 다음 아래의 <b>새로 고침</b>을 누르면 제공업체 목록에 표시됩니다.",
        "Þegar CLI er uppsett <em>og</em> þú ert innskráð/ur skaltu smella á <b>Endurnýja</b> hér fyrir neðan; það birtist þá í veitalistanum.",
        "Une fois la CLI installée <em>et</em> la connexion effectuée, cliquez sur <b>Actualiser</b> ci-dessous ; elle apparaîtra dans la liste des fournisseurs."),
    "<b>Response speed</b><br><span style='color:gray;'>Same three levels for every provider. Faster = snappier + cheaper; Deep = more thorough reasoning.</span>": _row(
        "<b>Svarshastighet</b><br><span style='color:gray;'>Samma tre nivåer för alla leverantörer. Snabbare = rappare + billigare; Djup = grundligare resonemang.</span>",
        "<b>Antwortgeschwindigkeit</b><br><span style='color:gray;'>Dieselben drei Stufen für jeden Anbieter. Schneller = zügiger + günstiger; Tief = gründlichere Schlussfolgerungen.</span>",
        "<b>Velocidad de respuesta</b><br><span style='color:gray;'>Los mismos tres niveles para todos los proveedores. Más rápida = más ágil + económica; Profunda = razonamiento más exhaustivo.</span>",
        "<b>响应速度</b><br><span style='color:gray;'>所有提供商均使用相同的三个级别。更快 = 响应更迅速且成本更低；深入 = 推理更周密。</span>",
        "<b>Velocidade da resposta</b><br><span style='color:gray;'>Os mesmos três níveis para todos os provedores. Mais rápida = mais ágil + econômica; Profunda = raciocínio mais completo.</span>",
        "<b>उत्तर की गति</b><br><span style='color:gray;'>हर प्रदाता के लिए वही तीन स्तर। तेज़ = झटपट + सस्ता; गहरा = अधिक संपूर्ण तर्क।</span>",
        "<b>응답 속도</b><br><span style='color:gray;'>모든 제공업체에 동일한 세 단계를 적용합니다. 빠름 = 민첩하고 저렴함; 심층 = 더 철저한 추론.</span>",
        "<b>Svarhraði</b><br><span style='color:gray;'>Sömu þrjú stig fyrir alla veitendur. Hraðara = snöggvara + ódýrara; Djúpt = ítarlegri röksemdafærsla.</span>",
        "<b>Vitesse de réponse</b><br><span style='color:gray;'>Les trois mêmes niveaux pour chaque fournisseur. Plus rapide = plus réactif + moins cher ; Approfondi = raisonnement plus complet.</span>"),
    "Fast — snappy replies, smallest model": _row(
        "Snabb — rappa svar, minsta modellen",
        "Schnell – zügige Antworten, kleinstes Modell",
        "Rápida — respuestas ágiles, modelo más pequeño",
        "快速 — 响应迅速，最小模型",
        "Rápida — respostas ágeis, menor modelo",
        "तेज़ — त्वरित उत्तर, सबसे छोटा मॉडल",
        "빠름 — 신속한 응답, 가장 작은 모델",
        "Hratt — snögg svör, minnsta líkanið",
        "Rapide — réponses réactives, plus petit modèle"),
    "Balanced — default, mid-tier model": _row(
        "Balanserad — standard, mellanstor modell",
        "Ausgewogen – Standard, mittelgroßes Modell",
        "Equilibrada — predeterminada, modelo intermedio",
        "平衡 — 默认，中等规模模型",
        "Equilibrada — padrão, modelo intermediário",
        "संतुलित — डिफ़ॉल्ट, मध्यम श्रेणी का मॉडल",
        "균형 — 기본값, 중간 급 모델",
        "Jafnvægi — sjálfgefið, miðstært líkan",
        "Équilibré — valeur par défaut, modèle intermédiaire"),
    "Deep — most thorough, largest model": _row(
        "Djup — grundligast, största modellen",
        "Tief – am gründlichsten, größtes Modell",
        "Profunda — la más exhaustiva, modelo más grande",
        "深入 — 最周密，最大模型",
        "Profunda — mais completa, maior modelo",
        "गहरा — सबसे संपूर्ण, सबसे बड़ा मॉडल",
        "심층 — 가장 철저한 응답, 가장 큰 모델",
        "Djúpt — ítarlegast, stærsta líkanið",
        "Approfondi — le plus complet, plus grand modèle"),
    "<b>Report errors as GitHub issues</b><br><span style='color:gray;'>Adds a \"File as GitHub issue\" button to the Explain-error flow. Clicking it opens your browser at a pre-filled issue on the spaCR repo — you review the payload and hit Submit yourself.</span>": _row(
        "<b>Rapportera fel som GitHub-ärenden</b><br><span style='color:gray;'>Lägger till knappen \"Rapportera som GitHub-ärende\" i flödet Förklara fel. Ett klick öppnar webbläsaren med ett förifyllt ärende i spaCR-arkivet — du granskar innehållet och klickar själv på Skicka.</span>",
        "<b>Fehler als GitHub-Issues melden</b><br><span style='color:gray;'>Fügt dem Ablauf zur Fehlererklärung die Schaltfläche \"Als GitHub-Issue melden\" hinzu. Ein Klick öffnet im Browser ein vorausgefülltes Issue im spaCR-Repository – Sie prüfen die Angaben und klicken selbst auf Senden.</span>",
        "<b>Informar errores como incidencias de GitHub</b><br><span style='color:gray;'>Añade el botón \"Registrar como incidencia de GitHub\" al flujo de explicación de errores. Al pulsarlo se abre en el navegador una incidencia prellenada del repositorio de spaCR; usted revisa los datos y pulsa Enviar.</span>",
        "<b>将错误报告为 GitHub 问题</b><br><span style='color:gray;'>在错误解释流程中添加“登记为 GitHub 问题”按钮。点击后，浏览器会打开 spaCR 仓库中预填的问题——由您审核内容并亲自点击提交。</span>",
        "<b>Relatar erros como problemas do GitHub</b><br><span style='color:gray;'>Adiciona o botão \"Registrar como problema do GitHub\" ao fluxo de explicação de erros. Ao clicar, o navegador abre um problema pré-preenchido no repositório do spaCR — você revisa os dados e clica em Enviar.</span>",
        "<b>त्रुटियों को GitHub इश्यू के रूप में रिपोर्ट करें</b><br><span style='color:gray;'>त्रुटि-व्याख्या प्रवाह में \"GitHub इश्यू के रूप में दर्ज करें\" बटन जोड़ता है। क्लिक करने पर spaCR रिपॉज़िटरी में पहले से भरा इश्यू ब्राउज़र में खुलता है — आप विवरण की समीक्षा करके खुद सबमिट करते हैं।</span>",
        "<b>오류를 GitHub 이슈로 보고</b><br><span style='color:gray;'>오류 설명 흐름에 \"GitHub 이슈로 등록\" 버튼을 추가합니다. 클릭하면 spaCR 저장소의 미리 채워진 이슈가 브라우저에서 열리며, 사용자가 내용을 검토하고 직접 제출합니다.</span>",
        "<b>Tilkynna villur sem GitHub-mál</b><br><span style='color:gray;'>Bætir hnappnum \"Skrá sem GitHub-mál\" við villuútskýringuna. Smellur opnar forútyllt mál í spaCR-geymslunni í vafranum — þú ferð yfir gögnin og smellir sjálf/ur á Senda.</span>",
        "<b>Signaler les erreurs comme tickets GitHub</b><br><span style='color:gray;'>Ajoute le bouton \"Signaler comme ticket GitHub\" au parcours d’explication des erreurs. Un clic ouvre dans votre navigateur un ticket prérempli sur le dépôt spaCR — vous vérifiez les informations et cliquez vous-même sur Envoyer.</span>"),
    "Enable — one-click issue filing from the error dialog": _row(
        "Aktivera — rapportera ett ärende med ett klick från feldialogrutan",
        "Aktivieren – Issue mit einem Klick aus dem Fehlerdialog melden",
        "Activar — registrar una incidencia con un clic desde el cuadro de error",
        "启用 — 从错误对话框一键登记问题",
        "Ativar — registrar um problema com um clique na caixa de erro",
        "चालू करें — त्रुटि डायलॉग से एक क्लिक में इश्यू दर्ज करें",
        "활성화 — 오류 대화 상자에서 한 번의 클릭으로 이슈 등록",
        "Virkja — skrá mál með einum smelli úr villuglugganum",
        "Activer — créer un ticket en un clic depuis la boîte de dialogue d’erreur"),
    "Route errors through AI — show the AI's explanation instead of the raw traceback": _row(
        "Skicka fel via AI — visa AI-förklaringen i stället för den råa stackspårningen",
        "Fehler über KI leiten – KI-Erklärung statt des rohen Tracebacks anzeigen",
        "Enviar errores a la IA — mostrar la explicación de la IA en lugar del rastreo sin procesar",
        "通过人工智能处理错误 — 显示人工智能解释而非原始回溯",
        "Encaminhar erros pela IA — mostrar a explicação da IA em vez do rastreamento bruto",
        "त्रुटियाँ एआई के माध्यम से भेजें — कच्चे ट्रेसबैक के बजाय एआई की व्याख्या दिखाएँ",
        "AI를 통해 오류 처리 — 원시 트레이스백 대신 AI 설명 표시",
        "Beina villum í gegnum gervigreind — sýna skýringu hennar í stað hrárrar rakningar",
        "Acheminer les erreurs vers l’IA — afficher l’explication de l’IA plutôt que la trace brute"),
    "<b>GitHub sign-in</b><br><span style='color:gray;'>Sign in so auto-filed issues send directly (no browser). Paste a Personal Access Token with <i>repo/issues</i> scope, or install + log in to the GitHub CLI (<code>gh auth login</code>) and spaCR will use it automatically.</span>": _row(
        "<b>GitHub-inloggning</b><br><span style='color:gray;'>Logga in så att automatiska ärenden skickas direkt (utan webbläsare). Klistra in en personlig åtkomsttoken med omfånget <i>repo/issues</i>, eller installera och logga in i GitHub CLI (<code>gh auth login</code>), så använder spaCR det automatiskt.</span>",
        "<b>GitHub-Anmeldung</b><br><span style='color:gray;'>Melden Sie sich an, damit automatisch erstellte Issues direkt gesendet werden (ohne Browser). Fügen Sie ein persönliches Zugriffstoken mit dem Bereich <i>repo/issues</i> ein oder installieren Sie die GitHub-CLI und melden Sie sich dort an (<code>gh auth login</code>); spaCR verwendet sie dann automatisch.</span>",
        "<b>Inicio de sesión en GitHub</b><br><span style='color:gray;'>Inicie sesión para enviar directamente las incidencias automáticas (sin navegador). Pegue un token de acceso personal con el ámbito <i>repo/issues</i>, o instale la CLI de GitHub e inicie sesión (<code>gh auth login</code>); spaCR la usará automáticamente.</span>",
        "<b>GitHub 登录</b><br><span style='color:gray;'>登录后，自动登记的问题可直接发送（无需浏览器）。请粘贴具有 <i>repo/issues</i> 范围的个人访问令牌，或安装 GitHub CLI 并登录（<code>gh auth login</code>），spaCR 将自动使用它。</span>",
        "<b>Entrada no GitHub</b><br><span style='color:gray;'>Entre para que problemas registrados automaticamente sejam enviados diretamente (sem navegador). Cole um Token de Acesso Pessoal com escopo <i>repo/issues</i>, ou instale e entre na CLI do GitHub (<code>gh auth login</code>); o spaCR a usará automaticamente.</span>",
        "<b>GitHub साइन-इन</b><br><span style='color:gray;'>साइन इन करने पर स्वचालित इश्यू सीधे भेजे जाएँगे (बिना ब्राउज़र)। <i>repo/issues</i> स्कोप वाला Personal Access Token पेस्ट करें, या GitHub CLI इंस्टॉल करके लॉग इन करें (<code>gh auth login</code>); spaCR इसका स्वचालित उपयोग करेगा।</span>",
        "<b>GitHub 로그인</b><br><span style='color:gray;'>로그인하면 자동 등록된 이슈가 브라우저 없이 바로 전송됩니다. <i>repo/issues</i> 범위의 개인용 액세스 토큰을 붙여넣거나 GitHub CLI를 설치하고 로그인하세요(<code>gh auth login</code>). spaCR이 자동으로 사용합니다.</span>",
        "<b>GitHub-innskráning</b><br><span style='color:gray;'>Skráðu þig inn svo sjálfvirk mál sendist beint (án vafra). Límdu inn persónulegan aðgangslykil með <i>repo/issues</i>-heimild, eða settu upp og skráðu þig inn í GitHub CLI (<code>gh auth login</code>); spaCR notar það sjálfkrafa.</span>",
        "<b>Connexion à GitHub</b><br><span style='color:gray;'>Connectez-vous pour envoyer directement les tickets créés automatiquement (sans navigateur). Collez un jeton d’accès personnel avec la portée <i>repo/issues</i>, ou installez la CLI GitHub et connectez-vous (<code>gh auth login</code>) ; spaCR l’utilisera automatiquement.</span>"),
    "Personal Access Token (ghp_… / github_pat_…)": _row(
        "Personlig åtkomsttoken (ghp_… / github_pat_…)",
        "Persönliches Zugriffstoken (ghp_… / github_pat_…)",
        "Token de acceso personal (ghp_… / github_pat_…)",
        "个人访问令牌 (ghp_… / github_pat_…)",
        "Token de Acesso Pessoal (ghp_… / github_pat_…)",
        "Personal Access Token (ghp_… / github_pat_…)",
        "개인용 액세스 토큰 (ghp_… / github_pat_…)",
        "Persónulegur aðgangslykill (ghp_… / github_pat_…)",
        "Jeton d’accès personnel (ghp_… / github_pat_…)"),
    "Save token": _row(
        "Spara token", "Token speichern", "Guardar token", "保存令牌",
        "Salvar token", "टोकन सहेजें", "토큰 저장",
        "Vista lykil", "Enregistrer le jeton"),
    "<b>System prompt</b><br><span style='color:gray;'>The persona spaCR sends to the assistant before your first message. Edit to change how answers are framed, then Save. Reset restores the default.</span>": _row(
        "<b>Systeminstruktion</b><br><span style='color:gray;'>Personan som spaCR skickar till assistenten före ditt första meddelande. Redigera för att ändra hur svar formuleras och klicka sedan på Spara. Återställ återgår till standarden.</span>",
        "<b>Systemanweisung</b><br><span style='color:gray;'>Die Persona, die spaCR vor Ihrer ersten Nachricht an den Assistenten sendet. Bearbeiten Sie sie, um die Formulierung der Antworten zu ändern, und klicken Sie dann auf Speichern. Zurücksetzen stellt den Standard wieder her.</span>",
        "<b>Instrucción del sistema</b><br><span style='color:gray;'>La personalidad que spaCR envía al asistente antes de su primer mensaje. Edítela para cambiar cómo se formulan las respuestas y pulse Guardar. Restablecer recupera el valor predeterminado.</span>",
        "<b>系统提示</b><br><span style='color:gray;'>spaCR 在您发送第一条消息前传给助手的角色设定。编辑后可改变回答的表述方式，然后点击保存。重置将恢复默认值。</span>",
        "<b>Prompt do sistema</b><br><span style='color:gray;'>A persona que o spaCR envia ao assistente antes da sua primeira mensagem. Edite para mudar como as respostas são formuladas e clique em Salvar. Redefinir restaura o padrão.</span>",
        "<b>सिस्टम प्रॉम्प्ट</b><br><span style='color:gray;'>आपके पहले संदेश से पहले spaCR सहायक को जो व्यक्तित्व भेजता है। उत्तरों के प्रस्तुतिकरण को बदलने के लिए इसे संपादित करें, फिर सहेजें। रीसेट डिफ़ॉल्ट को वापस लाता है।</span>",
        "<b>시스템 프롬프트</b><br><span style='color:gray;'>spaCR이 첫 메시지 전에 어시스턴트에게 전달하는 페르소나입니다. 응답의 표현 방식을 바꾸려면 편집한 뒤 저장하세요. 재설정하면 기본값으로 복원됩니다.</span>",
        "<b>Kerfiskveðja</b><br><span style='color:gray;'>Persónan sem spaCR sendir aðstoðarmanninum fyrir fyrstu skilaboðin þín. Breyttu henni til að stýra framsetningu svara og vistaðu síðan. Endurstilling endurheimtir sjálfgefið gildi.</span>",
        "<b>Invite système</b><br><span style='color:gray;'>Le profil que spaCR envoie à l’assistant avant votre premier message. Modifiez-le pour changer la formulation des réponses, puis enregistrez. La réinitialisation restaure la valeur par défaut.</span>"),
    "Save prompt": _row(
        "Spara instruktion", "Anweisung speichern", "Guardar instrucción",
        "保存提示", "Salvar prompt", "प्रॉम्प्ट सहेजें",
        "프롬프트 저장", "Vista kveðju", "Enregistrer l’invite"),
    "Reset to default": _row(
        "Återställ standard", "Auf Standard zurücksetzen",
        "Restablecer valor predeterminado", "重置为默认值",
        "Redefinir para o padrão", "डिफ़ॉल्ट पर रीसेट करें",
        "기본값으로 재설정", "Endurstilla á sjálfgefið",
        "Rétablir la valeur par défaut"),
    "Using your custom prompt (overrides default).": _row(
        "Din anpassade instruktion används (ersätter standarden).",
        "Ihre benutzerdefinierte Anweisung wird verwendet (überschreibt den Standard).",
        "Se está usando su instrucción personalizada (sustituye la predeterminada).",
        "正在使用您的自定义提示（覆盖默认值）。",
        "Usando seu prompt personalizado (substitui o padrão).",
        "आपके कस्टम प्रॉम्प्ट का उपयोग हो रहा है (यह डिफ़ॉल्ट को बदलता है)।",
        "사용자 정의 프롬프트 사용 중(기본값 대체).",
        "Sérsniðna kveðjan þín er notuð (hún yfirskrífar sjálfgefið gildi).",
        "Votre invite personnalisée est utilisée (elle remplace la valeur par défaut)."),
    "Using the default spaCR-aware prompt.": _row(
        "Standardinstruktionen med spaCR-kännedom används.",
        "Die spaCR-spezifische Standardanweisung wird verwendet.",
        "Se está usando la instrucción predeterminada con conocimiento de spaCR.",
        "正在使用了解 spaCR 的默认提示。",
        "Usando o prompt padrão com conhecimento do spaCR.",
        "spaCR की जानकारी वाले डिफ़ॉल्ट प्रॉम्प्ट का उपयोग हो रहा है।",
        "spaCR을 이해하는 기본 프롬프트 사용 중.",
        "Sjálfgefna kveðjan með spaCR-þekkingu er notuð.",
        "L’invite par défaut connaissant spaCR est utilisée."),
    "a saved token": _row(
        "en sparad token", "ein gespeichertes Token", "un token guardado",
        "已保存的令牌", "um token salvo", "सहेजा गया टोकन",
        "저장된 토큰", "vistaðan lykil", "un jeton enregistré"),
    "the GITHUB_TOKEN env var": _row(
        "miljövariabeln GITHUB_TOKEN", "die Umgebungsvariable GITHUB_TOKEN",
        "la variable de entorno GITHUB_TOKEN", "GITHUB_TOKEN 环境变量",
        "a variável de ambiente GITHUB_TOKEN", "GITHUB_TOKEN एनवायरनमेंट वेरिएबल",
        "GITHUB_TOKEN 환경 변수", "umhverfisbreytuna GITHUB_TOKEN",
        "la variable d’environnement GITHUB_TOKEN"),
    "the GitHub CLI": _row(
        "GitHub CLI", "die GitHub-CLI", "la CLI de GitHub", "GitHub CLI",
        "a CLI do GitHub", "GitHub CLI", "GitHub CLI", "GitHub CLI",
        "la CLI GitHub"),
    "✓ Signed in via {source} — issues send directly.": _row(
        "✓ Inloggad via {source} — ärenden skickas direkt.",
        "✓ Über {source} angemeldet – Issues werden direkt gesendet.",
        "✓ Sesión iniciada mediante {source} — las incidencias se envían directamente.",
        "✓ 已通过 {source} 登录——问题将直接发送。",
        "✓ Conectado por {source} — os problemas são enviados diretamente.",
        "✓ {source} के ज़रिए साइन इन हैं — इश्यू सीधे भेजे जाएँगे।",
        "✓ {source}로 로그인됨 — 이슈가 바로 전송됩니다.",
        "✓ Innskráð/ur með {source} — mál sendast beint.",
        "✓ Connecté via {source} — les tickets sont envoyés directement."),
    "Not signed in — issues open in your browser. Add a token or run gh auth login.": _row(
        "Inte inloggad — ärenden öppnas i webbläsaren. Lägg till en token eller kör gh auth login.",
        "Nicht angemeldet – Issues werden im Browser geöffnet. Fügen Sie ein Token hinzu oder führen Sie gh auth login aus.",
        "No ha iniciado sesión — las incidencias se abren en el navegador. Añada un token o ejecute gh auth login.",
        "未登录——问题将在浏览器中打开。请添加令牌或运行 gh auth login。",
        "Não conectado — os problemas abrem no navegador. Adicione um token ou execute gh auth login.",
        "साइन इन नहीं है — इश्यू ब्राउज़र में खुलेंगे। टोकन जोड़ें या gh auth login चलाएँ।",
        "로그인되지 않음 — 이슈가 브라우저에서 열립니다. 토큰을 추가하거나 gh auth login을 실행하세요.",
        "Ekki innskráð/ur — mál opnast í vafranum. Bættu við lykli eða keyrðu gh auth login.",
        "Non connecté — les tickets s’ouvrent dans votre navigateur. Ajoutez un jeton ou exécutez gh auth login."),
    "Install + login instructions for the vendor coding-agent CLIs.": _row(
        "Installations- och inloggningsanvisningar för leverantörernas CLI-verktyg för kodningsagenter.",
        "Installations- und Anmeldeanweisungen für die Coding-Agent-CLIs der Anbieter.",
        "Instrucciones de instalación e inicio de sesión para las CLI de agentes de programación de los proveedores.",
        "提供商编码代理 CLI 的安装和登录说明。",
        "Instruções de instalação e entrada para as CLIs de agentes de programação dos provedores.",
        "प्रदाता के कोडिंग-एजेंट CLI के लिए इंस्टॉल और लॉगिन निर्देश।",
        "제공업체 코딩 에이전트 CLI의 설치 및 로그인 안내.",
        "Leiðbeiningar um uppsetningu og innskráningu fyrir CLI-forritunaraðstoð veitenda.",
        "Instructions d’installation et de connexion pour les CLI d’agents de programmation des fournisseurs."),
    "Install a vendor CLI to chat": _row(
        "Installera ett leverantörs-CLI för att chatta",
        "Installieren Sie eine Anbieter-CLI, um zu chatten",
        "Instale una CLI de proveedor para chatear",
        "安装提供商 CLI 以开始聊天",
        "Instale uma CLI de provedor para conversar",
        "चैट करने के लिए प्रदाता CLI इंस्टॉल करें",
        "채팅하려면 제공업체 CLI를 설치하세요",
        "Settu upp CLI veitanda til að spjalla",
        "Installez une CLI de fournisseur pour discuter"),
    "The AI Console uses your chat subscription via the vendor coding-agent CLIs: `claude`, `codex`, or `gemini`. Install any one of them and log in. Open Providers ▸ Copy the commands and paste in a terminal.": _row(
        "AI-konsolen använder din chattprenumeration via leverantörernas CLI-verktyg för kodningsagenter: `claude`, `codex` eller `gemini`. Installera ett av dem och logga in. Öppna Leverantörer ▸ Kopiera kommandona och klistra in dem i en terminal.",
        "Die KI-Konsole verwendet Ihr Chat-Abonnement über die Coding-Agent-CLIs der Anbieter: `claude`, `codex` oder `gemini`. Installieren Sie eine davon und melden Sie sich an. Öffnen Sie Anbieter ▸ Kopieren Sie die Befehle und fügen Sie sie in ein Terminal ein.",
        "La consola de IA usa su suscripción de chat mediante las CLI de agentes de programación de los proveedores: `claude`, `codex` o `gemini`. Instale una e inicie sesión. Abra Proveedores ▸ Copie los comandos y péguelos en una terminal.",
        "人工智能控制台通过提供商的编码代理 CLI（`claude`、`codex` 或 `gemini`）使用您的聊天订阅。请安装其中任意一个并登录。打开“提供商”▸ 复制命令并粘贴到终端中。",
        "O Console de IA usa sua assinatura de chat pelas CLIs de agentes de programação dos provedores: `claude`, `codex` ou `gemini`. Instale uma delas e entre. Abra Provedores ▸ Copie os comandos e cole em um terminal.",
        "एआई कंसोल प्रदाता के कोडिंग-एजेंट CLI: `claude`, `codex` या `gemini` के ज़रिए आपकी चैट सदस्यता का उपयोग करता है। इनमें से कोई एक इंस्टॉल करके लॉग इन करें। प्रदाता खोलें ▸ कमांड कॉपी करके टर्मिनल में पेस्ट करें।",
        "AI 콘솔은 제공업체의 코딩 에이전트 CLI인 `claude`, `codex` 또는 `gemini`를 통해 채팅 구독을 사용합니다. 하나를 설치하고 로그인하세요. 제공업체 열기 ▸ 명령을 복사하여 터미널에 붙여넣으세요.",
        "Gervigreindarstjórnborðið notar spjalláskriftina þína í gegnum CLI-forritunaraðstoð veitenda: `claude`, `codex` eða `gemini`. Settu eitt þeirra upp og skráðu þig inn. Opnaðu Veitendur ▸ Afritaðu skipanirnar og límdu í skjáhermi.",
        "La console IA utilise votre abonnement de chat via les CLI d’agents de programmation des fournisseurs : `claude`, `codex` ou `gemini`. Installez-en une et connectez-vous. Ouvrez Fournisseurs ▸ Copiez les commandes et collez-les dans un terminal."),
    "A response is already streaming — hit Cancel to interrupt.": _row(
        "Ett svar strömmas redan — klicka på Avbryt för att avbryta.",
        "Eine Antwort wird bereits gestreamt – klicken Sie zum Unterbrechen auf Abbrechen.",
        "Ya se está recibiendo una respuesta — pulse Cancelar para interrumpirla.",
        "正在接收一个响应——请点击取消以中断。",
        "Uma resposta já está sendo recebida — clique em Cancelar para interromper.",
        "एक उत्तर पहले से आ रहा है — रोकने के लिए रद्द करें दबाएँ।",
        "이미 응답을 수신 중입니다. 중단하려면 취소를 누르세요.",
        "Svar er þegar að streyma — smelltu á Hætta við til að rjúfa.",
        "Une réponse est déjà en cours de réception — cliquez sur Annuler pour l’interrompre."),
    "Failed: {detail}": _row(
        "Misslyckades: {detail}", "Fehlgeschlagen: {detail}",
        "Falló: {detail}", "失败：{detail}", "Falhou: {detail}",
        "विफल: {detail}", "실패: {detail}", "Mistókst: {detail}",
        "Échec : {detail}"),
    "Install a vendor CLI first (Providers…).": _row(
        "Installera först ett leverantörs-CLI (Leverantörer…).",
        "Installieren Sie zuerst eine Anbieter-CLI (Anbieter…).",
        "Instale primero una CLI de proveedor (Proveedores…).",
        "请先安装提供商 CLI（提供商…）。",
        "Primeiro instale uma CLI de provedor (Provedores…).",
        "पहले प्रदाता CLI इंस्टॉल करें (प्रदाता…)।",
        "먼저 제공업체 CLI를 설치하세요(제공업체…).",
        "Settu fyrst upp CLI veitanda (Veitendur…).",
        "Installez d’abord une CLI de fournisseur (Fournisseurs…)."),

    # ---- The first-run setup screen ---------------------------------
    # Its own captions, which used to be the one screen in the program
    # that never translated: it asks which language to use and then went
    # on asking the rest in English.
    "How it runs": _row("Hur den körs", "Wie es läuft", "Cómo se ejecuta", "运行方式", "Como funciona", "यह कैसे चलता है", "실행 방식", "Hvernig hún keyrir", "Comment il s’exécute"),
    "The assistant": _row("Assistenten", "Der Assistent", "El asistente", "助手", "O assistente", "सहायक", "어시스턴트", "Aðstoðarmaðurinn", "L’assistant"),
    "When something breaks": _row("När något går fel", "Wenn etwas schiefgeht", "Cuando algo falla", "出现问题时", "Quando algo falha", "जब कुछ गड़बड़ हो", "문제가 생겼을 때", "Þegar eitthvað bilar", "Quand quelque chose casse"),
    "Done": _row("Klart", "Fertig", "Listo", "完成", "Concluído", "पूर्ण", "완료", "Lokið", "Terminé"),
    "spaCR mode": _row("spaCR-läge", "spaCR-Modus", "Modo spaCR", "spaCR 模式", "Modo spaCR", "spaCR मोड", "spaCR 모드", "spaCR-hamur", "Mode spaCR"),
    "Reproducibility hash": _row("Reproducerbarhetshash", "Reproduzierbarkeits-Hash", "Hash de reproducibilidad", "可复现性哈希", "Hash de reprodutibilidade", "पुनरुत्पादन हैश", "재현성 해시", "Endurgerðarhash", "Empreinte de reproductibilité"),
    "One-click issue filing": _row("Felrapport med ett klick", "Fehlermeldung mit einem Klick", "Informe de problemas con un clic", "一键提交问题", "Relato de problemas com um clique", "एक-क्लिक समस्या रिपोर्ट", "원클릭 이슈 등록", "Villutilkynning með einum smelli", "Signalement en un clic"),
    "AI assistant on at launch": _row("AI-assistent på vid start", "KI-Assistent beim Start aktiv", "Asistente de IA activo al iniciar", "启动时开启 AI 助手", "Assistente de IA ativo ao iniciar", "शुरू होते ही AI सहायक चालू", "시작 시 AI 어시스턴트 켜기", "AI-aðstoð virk við ræsingu", "Assistant IA actif au démarrage"),
    "Include recent logs in a report": _row("Inkludera senaste loggarna i en rapport", "Aktuelle Protokolle in den Bericht aufnehmen", "Incluir registros recientes en el informe", "在报告中附上最近的日志", "Incluir os registos recentes no relatório", "रिपोर्ट में हाल के लॉग शामिल करें", "보고서에 최근 로그 포함", "Hafa nýlegar annálar með í skýrslu", "Inclure les journaux récents dans un rapport"),
    "AI provider": _row("AI-leverantör", "KI-Anbieter", "Proveedor de IA", "AI 提供方", "Fornecedor de IA", "AI प्रदाता", "AI 제공자", "AI-veita", "Fournisseur d’IA"),
    "{n} of {total}": _row("{n} av {total}", "{n} von {total}", "{n} de {total}", "第 {n} / {total} 步", "{n} de {total}", "{total} में से {n}", "{total} 중 {n}", "{n} af {total}", "{n} sur {total}"),
    "Every label, tooltip and message spaCR shows you. You can change it later in Preferences, and nothing about your data depends on it.": _row(
        "Varje etikett, verktygstips och meddelande som spaCR visar. Du kan ändra det senare i Inställningar, och inget i dina data beror på det.",
        "Jede Beschriftung, jeder Tooltip und jede Meldung, die spaCR anzeigt. Sie können das später in den Einstellungen ändern; Ihre Daten hängen nicht davon ab.",
        "Cada etiqueta, descripción emergente y mensaje que muestra spaCR. Puede cambiarlo después en Preferencias, y nada de sus datos depende de ello.",
        "spaCR 显示的每个标签、提示和消息。之后可以在首选项中更改，你的数据不受影响。",
        "Cada rótulo, dica e mensagem que o spaCR mostra. Pode alterar depois nas Preferências, e nada nos seus dados depende disso.",
        "spaCR द्वारा दिखाया जाने वाला हर लेबल, टूलटिप और संदेश। इसे बाद में प्राथमिकताओं में बदला जा सकता है, और आपके डेटा पर कोई असर नहीं पड़ता।",
        "spaCR가 보여주는 모든 라벨, 툴팁, 메시지에 적용됩니다. 나중에 환경설정에서 바꿀 수 있으며 데이터에는 영향이 없습니다.",
        "Sérhver merking, ábending og skilaboð sem spaCR sýnir. Þú getur breytt þessu síðar í Stillingum og ekkert í gögnunum þínum veltur á því.",
        "Chaque libellé, infobulle et message affiché par spaCR. Vous pouvez le changer plus tard dans les Préférences, et rien dans vos données n’en dépend."),
    "That is everything. All of it is in Preferences if you change your mind.": _row(
        "Det var allt. Allt finns i Inställningar om du ändrar dig.",
        "Das war alles. Alles davon steht in den Einstellungen, falls Sie es sich anders überlegen.",
        "Eso es todo. Todo está en Preferencias si cambia de opinión.",
        "就是这些。如果改变主意，所有选项都在首选项里。",
        "É tudo. Está tudo nas Preferências, caso mude de ideias.",
        "बस इतना ही। मन बदले तो सब कुछ प्राथमिकताओं में मिलेगा।",
        "이것이 전부입니다. 마음이 바뀌면 모두 환경설정에 있습니다.",
        "Það var allt. Allt er í Stillingum ef þú skiptir um skoðun.",
        "C’est tout. Tout se retrouve dans les Préférences si vous changez d’avis."),

    "How spaCR looks, and whether its colours are chosen to stay distinguishable without colour vision. Both take effect as you pick them, so you can see what you are choosing.": _row(
        "Hur spaCR ser ut, och om färgerna väljs så att de går att skilja åt utan färgseende. Båda träder i kraft medan du väljer, så att du ser vad du väljer.",
        "Wie spaCR aussieht und ob seine Farben so gewählt werden, dass sie ohne Farbsehen unterscheidbar bleiben. Beides wirkt sofort, sodass Sie sehen, was Sie wählen.",
        "El aspecto de spaCR y si sus colores se eligen para seguir siendo distinguibles sin visión del color. Ambos se aplican al elegirlos, para que pueda ver lo que está seleccionando.",
        "spaCR 的外观，以及是否选择在无色觉时仍可区分的配色。两者在选择时即时生效，让你看到自己的选择。",
        "O aspeto do spaCR e se as suas cores são escolhidas para se manterem distinguíveis sem visão das cores. Ambos entram em vigor à medida que escolhe, para que veja o que está a escolher.",
        "spaCR कैसा दिखता है, और क्या इसके रंग ऐसे चुने जाएँ जो वर्ण-दृष्टि के बिना भी अलग दिखें। दोनों चुनते ही लागू हो जाते हैं, ताकि आप देख सकें कि क्या चुन रहे हैं।",
        "spaCR의 모습과, 색각 없이도 구분되도록 색을 고를지 여부입니다. 둘 다 고르는 즉시 적용되므로 무엇을 고르는지 바로 보입니다.",
        "Hvernig spaCR lítur út og hvort litirnir eru valdir svo þeir greinist án litaskyns. Hvort tveggja tekur gildi um leið og þú velur, svo þú sérð hvað þú ert að velja.",
        "L’apparence de spaCR, et si ses couleurs sont choisies pour rester distinguables sans vision des couleurs. Les deux s’appliquent au moment du choix, pour que vous voyiez ce que vous choisissez."),
    "The mode decides how much of this machine spaCR keeps for itself: how many background processes it starts and holds, and whether it hands back its caches and GPU memory between runs. Balanced keeps them, which is fastest; the other two give them up so the rest of your machine has more to work with. The reproducibility hash records what went into a run, so a result can be traced back to the exact inputs that produced it.": _row(
        "Läget avgör hur mycket av datorn spaCR behåller för sig själv: hur många bakgrundsprocesser den startar och håller kvar, och om den lämnar tillbaka sina cacher och GPU-minne mellan körningar. Balanserat behåller dem, vilket är snabbast; de andra två lämnar dem ifrån sig så att resten av datorn får mer att arbeta med. Reproducerbarhetshashen registrerar vad som gick in i en körning, så att ett resultat kan spåras tillbaka till exakt de indata som gav det.",
        "Der Modus bestimmt, wie viel dieses Rechners spaCR für sich behält: wie viele Hintergrundprozesse es startet und hält und ob es Caches und GPU-Speicher zwischen Läufen zurückgibt. Ausgewogen behält sie, was am schnellsten ist; die anderen beiden geben sie ab, damit dem übrigen Rechner mehr bleibt. Der Reproduzierbarkeits-Hash hält fest, was in einen Lauf einging, sodass ein Ergebnis auf genau die Eingaben zurückgeführt werden kann, die es erzeugt haben.",
        "El modo decide cuánto de esta máquina se reserva spaCR: cuántos procesos en segundo plano inicia y mantiene, y si devuelve sus cachés y la memoria de GPU entre ejecuciones. Equilibrado los conserva, que es lo más rápido; los otros dos los ceden para que el resto de la máquina disponga de más. El hash de reproducibilidad registra qué entró en una ejecución, de modo que un resultado pueda rastrearse hasta las entradas exactas que lo produjeron.",
        "该模式决定 spaCR 为自己保留多少本机资源：启动并保持多少后台进程，以及在两次运行之间是否交还缓存和 GPU 显存。均衡会保留它们，速度最快；另外两种会交还，让机器的其余部分有更多可用资源。可复现性哈希记录一次运行的输入，因此结果可以追溯到产生它的确切输入。",
        "O modo decide quanto desta máquina o spaCR guarda para si: quantos processos em segundo plano inicia e mantém, e se devolve as suas caches e memória de GPU entre execuções. Equilibrado mantém-nas, o que é mais rápido; os outros dois libertam-nas para que o resto da máquina tenha mais com que trabalhar. O hash de reprodutibilidade regista o que entrou numa execução, para que um resultado possa ser rastreado até às entradas exatas que o produziram.",
        "मोड तय करता है कि spaCR इस मशीन का कितना हिस्सा अपने पास रखे: वह कितनी पृष्ठभूमि प्रक्रियाएँ शुरू कर के रोके रखता है, और क्या वह रन के बीच अपनी कैश और GPU मेमोरी लौटाता है। संतुलित उन्हें रखे रहता है, जो सबसे तेज़ है; अन्य दो उन्हें छोड़ देते हैं ताकि बाकी मशीन को अधिक मिले। पुनरुत्पादन हैश यह दर्ज करता है कि रन में क्या गया, ताकि परिणाम को ठीक उन्हीं इनपुट तक पहुँचाया जा सके जिन्होंने उसे बनाया।",
        "이 모드는 spaCR가 이 컴퓨터를 얼마나 차지할지 정합니다: 백그라운드 프로세스를 몇 개나 띄워 두는지, 실행 사이에 캐시와 GPU 메모리를 반납하는지입니다. 균형은 그대로 유지해 가장 빠르고, 나머지 둘은 반납해 컴퓨터의 다른 작업에 여유를 줍니다. 재현성 해시는 실행에 들어간 것을 기록하므로, 결과를 그것을 만든 정확한 입력까지 되짚을 수 있습니다.",
        "Hamurinn ræður hversu mikið af þessari vél spaCR heldur eftir: hversu mörg bakgrunnsferli það ræsir og heldur, og hvort það skilar skyndiminni og GPU-minni milli keyrslna. Jafnvægi heldur þeim, sem er hraðast; hinir tveir sleppa þeim svo restin af vélinni hafi meira svigrúm. Endurgerðarhashið skráir hvað fór inn í keyrslu, svo rekja megi niðurstöðu til nákvæmlega þeirra gagna sem bjuggu hana til.",
        "Le mode décide de la part de cette machine que spaCR garde pour lui : combien de processus d’arrière-plan il lance et conserve, et s’il rend ses caches et la mémoire GPU entre les exécutions. Équilibré les conserve, ce qui est le plus rapide ; les deux autres les rendent pour laisser plus de ressources au reste de la machine. L’empreinte de reproductibilité enregistre ce qui est entré dans une exécution, afin qu’un résultat puisse être retracé jusqu’aux entrées exactes qui l’ont produit."),
    "spaCR can explain an error or a result through a coding assistant you already subscribe to. It uses the vendor's own command-line tool, so nothing is sent anywhere you have not already logged in to.": _row(
        "spaCR kan förklara ett fel eller ett resultat genom en kodassistent du redan prenumererar på. Den använder leverantörens eget kommandoradsverktyg, så ingenting skickas någonstans du inte redan är inloggad hos.",
        "spaCR kann einen Fehler oder ein Ergebnis über einen Coding-Assistenten erklären, den Sie bereits abonniert haben. Es nutzt das Kommandozeilenwerkzeug des Anbieters, sodass nichts irgendwohin geht, wo Sie nicht bereits angemeldet sind.",
        "spaCR puede explicar un error o un resultado mediante un asistente de código al que ya está suscrito. Utiliza la herramienta de línea de comandos del proveedor, así que nada se envía a ningún servicio en el que no haya iniciado sesión.",
        "spaCR 可以通过你已订阅的编程助手来解释错误或结果。它使用厂商自己的命令行工具，因此不会把任何内容发往你尚未登录的地方。",
        "O spaCR pode explicar um erro ou um resultado através de um assistente de código que já subscreve. Usa a ferramenta de linha de comandos do próprio fornecedor, pelo que nada é enviado para onde ainda não tenha sessão iniciada.",
        "spaCR किसी त्रुटि या परिणाम को उस कोडिंग सहायक के जरिए समझा सकता है जिसकी सदस्यता आपके पास पहले से है। यह विक्रेता के अपने कमांड-लाइन टूल का उपयोग करता है, इसलिए कुछ भी वहाँ नहीं भेजा जाता जहाँ आप पहले से लॉग इन न हों।",
        "spaCR는 이미 구독 중인 코딩 어시스턴트를 통해 오류나 결과를 설명할 수 있습니다. 공급업체의 명령줄 도구를 그대로 사용하므로, 이미 로그인한 곳 외에는 아무것도 전송되지 않습니다.",
        "spaCR getur útskýrt villu eða niðurstöðu í gegnum kóðunaraðstoð sem þú ert þegar áskrifandi að. Það notar skipanalínutól framleiðandans sjálfs, svo ekkert er sent neitt þangað sem þú ert ekki þegar innskráð(ur).",
        "spaCR peut expliquer une erreur ou un résultat via un assistant de code auquel vous êtes déjà abonné. Il utilise l’outil en ligne de commande de l’éditeur, donc rien n’est envoyé là où vous n’êtes pas déjà connecté."),
    "What may leave this machine, and under whose name. Nothing is ever sent without you seeing it first and pressing send yourself.": _row(
        "Vad som får lämna den här datorn, och i vems namn. Ingenting skickas någonsin utan att du först ser det och själv trycker på skicka.",
        "Was diesen Rechner verlassen darf und unter wessen Namen. Nichts wird jemals gesendet, ohne dass Sie es zuvor sehen und selbst auf Senden drücken.",
        "Qué puede salir de esta máquina y a nombre de quién. Nunca se envía nada sin que lo vea primero y pulse personalmente Enviar.",
        "什么内容可以离开这台机器，以及以谁的名义。任何内容都不会在你先看到并亲自点击发送之前被发送。",
        "O que pode sair desta máquina, e em nome de quem. Nada é enviado sem que o veja primeiro e carregue em enviar você mesmo.",
        "इस मशीन से क्या बाहर जा सकता है, और किसके नाम से। कुछ भी तब तक नहीं भेजा जाता जब तक आप उसे देख कर स्वयं भेजें न दबाएँ।",
        "이 컴퓨터에서 무엇이 나갈 수 있는지, 그리고 누구의 이름으로 나가는지입니다. 먼저 내용을 보고 직접 보내기를 누르기 전에는 아무것도 전송되지 않습니다.",
        "Hvað má fara af þessari vél og í hvers nafni. Ekkert er nokkurn tímann sent nema þú sjáir það fyrst og ýtir sjálf(ur) á senda.",
        "Ce qui peut quitter cette machine, et sous quel nom. Rien n’est jamais envoyé sans que vous l’ayez d’abord vu et appuyé vous-même sur envoyer."),

    # ---- What the machine can run, on the first slide ----------------
    "Segmentation and object classification need an NVIDIA GPU. Everything else runs without one.": _row(
        "Segmentering och objektklassificering kräver en NVIDIA-GPU. Allt annat fungerar utan en.",
        "Segmentierung und Objektklassifizierung brauchen eine NVIDIA-GPU. Alles andere läuft auch ohne.",
        "La segmentación y la clasificación de objetos necesitan una GPU NVIDIA. Todo lo demás funciona sin ella.",
        "分割和对象分类需要 NVIDIA GPU。其余功能没有它也能运行。",
        "A segmentação e a classificação de objetos precisam de uma GPU NVIDIA. Todo o resto funciona sem ela.",
        "सेगमेंटेशन और ऑब्जेक्ट वर्गीकरण के लिए NVIDIA GPU चाहिए। बाकी सब उसके बिना भी चलता है।",
        "분할과 객체 분류에는 NVIDIA GPU 가 필요합니다. 나머지는 없어도 실행됩니다.",
        "Hlutun og flokkun hluta krefjast NVIDIA-skjákorts. Allt annað keyrir án þess.",
        "La segmentation et la classification d’objets nécessitent un GPU NVIDIA. Tout le reste fonctionne sans."),
    "The card is there but torch cannot use it. Run spacr-doctor to find out which part of CUDA is missing.": _row(
        "Kortet finns men torch kan inte använda det. Kör spacr-doctor för att ta reda på vilken del av CUDA som saknas.",
        "Die Karte ist vorhanden, aber torch kann sie nicht nutzen. Führen Sie spacr-doctor aus, um herauszufinden, welcher Teil von CUDA fehlt.",
        "La tarjeta está presente pero torch no puede usarla. Ejecute spacr-doctor para averiguar qué parte de CUDA falta.",
        "显卡在，但 torch 无法使用它。运行 spacr-doctor 查看缺少哪一部分 CUDA。",
        "A placa existe mas o torch não a consegue usar. Execute spacr-doctor para descobrir que parte do CUDA falta.",
        "कार्ड मौजूद है पर torch उसे उपयोग नहीं कर पा रहा। यह जानने के लिए कि CUDA का कौन-सा हिस्सा अनुपस्थित है, spacr-doctor चलाएँ।",
        "카드는 있지만 torch 가 사용할 수 없습니다. CUDA 의 어느 부분이 빠졌는지 확인하려면 spacr-doctor 를 실행하세요.",
        "Kortið er til staðar en torch getur ekki notað það. Keyrðu spacr-doctor til að sjá hvaða hluta CUDA vantar.",
        "La carte est présente mais torch ne peut pas l’utiliser. Lancez spacr-doctor pour savoir quelle partie de CUDA manque."),
    "Compatible GPU": _row(
        "Kompatibel GPU", "Kompatible GPU", "GPU compatible", "兼容的 GPU",
        "GPU compatível", "संगत GPU", "호환되는 GPU", "Samhæft skjákort",
        "GPU compatible"),
    "No compatible GPU": _row(
        "Ingen kompatibel GPU", "Keine kompatible GPU", "Sin GPU compatible",
        "没有兼容的 GPU", "Sem GPU compatível", "कोई संगत GPU नहीं",
        "호환되는 GPU 없음", "Ekkert samhæft skjákort", "Aucun GPU compatible"),
    "No compatible GPU — none detected": _row(
        "Ingen kompatibel GPU — ingen hittades",
        "Keine kompatible GPU – keine gefunden",
        "Sin GPU compatible: no se detectó ninguna",
        "没有兼容的 GPU —— 未检测到",
        "Sem GPU compatível — nenhuma detetada",
        "कोई संगत GPU नहीं — कोई नहीं मिला",
        "호환되는 GPU 없음 — 감지되지 않음",
        "Ekkert samhæft skjákort — ekkert fannst",
        "Aucun GPU compatible — aucun détecté"),

    # Compact first-run choices and captions.  These values are assembled
    # from registries rather than literal widget constructors, so the large
    # generated Qt catalog cannot discover them from the AST.  Keep them in
    # the exact, hand-reviewed catalog and let the caption ratchet below the
    # test suite catch the next choice added without a row.
    "ask": _row(
        "Fråga först", "Vorher fragen", "Preguntar antes", "提交前询问",
        "Perguntar antes", "पहले पूछें", "먼저 묻기", "Spyrja fyrst",
        "Demander d’abord"),
    "off": _row(
        "Av", "Aus", "Desactivado", "关闭", "Desativado",
        "बंद", "꺼짐", "Slökkt", "Désactivé"),
    "Dark": _row(
        "Mörkt", "Dunkel", "Oscuro", "深色", "Escuro",
        "गहरा", "어두움", "Dökkt", "Sombre"),
    "Next": _row(
        "Nästa", "Weiter", "Siguiente", "下一步", "Seguinte",
        "अगला", "다음", "Áfram", "Suivant"),
    "None": _row(
        "Ingen", "Keine", "Ninguna", "无", "Nenhuma",
        "कोई नहीं", "없음", "Engin", "Aucune"),
    "Skip": _row(
        "Hoppa över", "Überspringen", "Omitir", "跳过", "Ignorar",
        "छोड़ें", "건너뛰기", "Sleppa", "Passer"),
    "Blobs": _row(
        "Blobbar", "Blobs", "Formas fluidas", "流动色块",
        "Formas fluidas", "तरल आकृतियाँ", "블롭", "Klessur",
        "Formes fluides"),
    "Bokeh": _row(
        "Bokeh", "Bokeh", "Bokeh", "散景", "Bokeh",
        "बोकेह", "보케", "Bokeh", "Bokeh"),
    "Cells": _row(
        "Celler", "Zellen", "Células", "细胞", "Células",
        "कोशिकाएँ", "세포", "Frumur", "Cellules"),
    "Glass": _row(
        "Glas", "Glas", "Cristal", "玻璃", "Vidro",
        "काँच", "유리", "Gler", "Verre"),
    "Later": _row(
        "Senare", "Später", "Más tarde", "稍后", "Mais tarde",
        "बाद में", "나중에", "Síðar", "Plus tard"),
    "Light": _row(
        "Ljust", "Hell", "Claro", "浅色", "Claro",
        "हल्का", "밝음", "Ljóst", "Clair"),
    "never": _row(
        "Aldrig", "Nie", "Nunca", "从不", "Nunca",
        "कभी नहीं", "안 함", "Aldrei", "Jamais"),
    "Aurora": _row(
        "Norrsken", "Polarlicht", "Aurora", "极光", "Aurora",
        "ध्रुवीय ज्योति", "오로라", "Norðurljós", "Aurore"),
    "Finish": _row(
        "Slutför", "Fertigstellen", "Finalizar", "完成", "Concluir",
        "समाप्त करें", "마침", "Ljúka", "Terminer"),
    "Next ›": _row(
        "Nästa ›", "Weiter ›", "Siguiente ›", "下一步 ›", "Seguinte ›",
        "अगला ›", "다음 ›", "Áfram ›", "Suivant ›"),
    "always": _row(
        "Alltid", "Immer", "Siempre", "始终", "Sempre",
        "हमेशा", "항상", "Alltaf", "Toujours"),
    "‹ Back": _row(
        "‹ Tillbaka", "‹ Zurück", "‹ Atrás", "‹ 返回", "‹ Voltar",
        "‹ वापस", "‹ 뒤로", "‹ Til baka", "‹ Retour"),
    "Ripples": _row(
        "Krusningar", "Wellen", "Ondas", "涟漪", "Ondulações",
        "लहरें", "물결", "Gárur", "Ondulations"),
    "Tubules": _row(
        "Tubuli", "Tubuli", "Túbulos", "管状结构", "Túbulos",
        "नलिकाएँ", "세관", "Píplur", "Tubules"),
    "Balanced": _row(
        "Balanserat", "Ausgewogen", "Equilibrado", "均衡", "Equilibrado",
        "संतुलित", "균형", "Jafnvægi", "Équilibré"),
    "Starfield": _row(
        "Stjärnfält", "Sternenfeld", "Campo estelar", "星空",
        "Campo estelar", "तारों का क्षेत्र", "별빛", "Stjörnusvið",
        "Champ d’étoiles"),
    "Demos menu": _row(
        "Demomeny", "Demo-Menü", "Menú de demostraciones", "演示菜单",
        "Menu de demonstrações", "डेमो मेनू", "데모 메뉴",
        "Sýnishornavalmynd", "Menu Démonstrations"),
    "protanopia": _row(
        "protanopi", "Protanopie", "protanopía", "红色盲", "protanopia",
        "प्रोटैनोपिया", "제1색맹", "rauðblinda", "protanopie"),
    "tritanopia": _row(
        "tritanopi", "Tritanopie", "tritanopía", "蓝黄色盲", "tritanopia",
        "ट्राइटैनोपिया", "제3색맹", "blá-gul litblinda", "tritanopie"),
    "Drag & drop": _row(
        "Dra och släpp", "Ziehen und ablegen", "Arrastrar y soltar", "拖放",
        "Arrastar e soltar", "खींचें और छोड़ें", "끌어서 놓기",
        "Draga og sleppa", "Glisser-déposer"),
    "Performance": _row(
        "Prestanda", "Leistung", "Rendimiento", "性能", "Desempenho",
        "प्रदर्शन", "성능", "Afköst", "Performances"),
    "Sign in now": _row(
        "Logga in nu", "Jetzt anmelden", "Iniciar sesión ahora", "立即登录",
        "Iniciar sessão agora", "अभी साइन इन करें", "지금 로그인",
        "Skrá inn núna", "Se connecter maintenant"),
    "Start spaCR": _row(
        "Starta spaCR", "spaCR starten", "Iniciar spaCR", "启动 spaCR",
        "Iniciar o spaCR", "spaCR शुरू करें", "spaCR 시작", "Ræsa spaCR",
        "Démarrer spaCR"),
    "Cytoskeleton": _row(
        "Cytoskelett", "Zytoskelett", "Citoesqueleto", "细胞骨架",
        "Citoesqueleto", "कोशिका कंकाल", "세포골격", "Frumugrind",
        "Cytosquelette"),
    "Save choices": _row(
        "Spara val", "Auswahl speichern", "Guardar opciones", "保存选择",
        "Guardar escolhas", "विकल्प सहेजें", "선택 저장", "Vista val",
        "Enregistrer les choix"),
    "Set spaCR up": _row(
        "Konfigurera spaCR", "spaCR einrichten", "Configurar spaCR",
        "设置 spaCR", "Configurar o spaCR", "spaCR सेट करें", "spaCR 설정",
        "Stilla spaCR", "Configurer spaCR"),
    "deuteranopia": _row(
        "deuteranopi", "Deuteranopie", "deuteranopía", "绿色盲",
        "deuteranopia", "ड्यूटेरैनोपिया", "제2색맹", "grænblinda",
        "deutéranopie"),
    "Follow system": _row(
        "Följ systemet", "Systemeinstellung folgen", "Seguir el sistema",
        "跟随系统", "Seguir o sistema", "सिस्टम के अनुसार", "시스템 설정 따르기",
        "Fylgja kerfinu", "Suivre le système"),
    "Open the page": _row(
        "Öppna sidan", "Seite öffnen", "Abrir la página", "打开页面",
        "Abrir a página", "पृष्ठ खोलें", "페이지 열기", "Opna síðuna",
        "Ouvrir la page"),
    "Command palette": _row(
        "Kommandopalett", "Befehlspalette", "Paleta de comandos", "命令面板",
        "Paleta de comandos", "कमांड पैलेट", "명령 팔레트", "Skipanaspjald",
        "Palette de commandes"),
    "Copy the command": _row(
        "Kopiera kommandot", "Befehl kopieren", "Copiar el comando", "复制命令",
        "Copiar o comando", "कमांड कॉपी करें", "명령 복사", "Afrita skipunina",
        "Copier la commande"),
    "Extra Performance": _row(
        "Extra prestanda", "Maximale Leistung", "Máximo rendimiento", "极致性能",
        "Desempenho máximo", "अधिकतम प्रदर्शन", "최대 성능", "Hámarksafköst",
        "Performances maximales"),
    "Skip — keep all off": _row(
        "Hoppa över — låt allt vara av", "Überspringen – alles ausgeschaltet lassen",
        "Omitir — mantener todo desactivado", "跳过 — 全部保持关闭",
        "Ignorar — manter tudo desativado", "छोड़ें — सब कुछ बंद रखें",
        "건너뛰기 — 모두 끈 상태로 유지", "Sleppa — hafa allt óvirkt",
        "Passer — tout laisser désactivé"),
    "whatever is available": _row(
        "det som är tillgängligt", "was verfügbar ist", "lo que esté disponible",
        "任意可用项", "o que estiver disponível", "जो उपलब्ध हो",
        "사용 가능한 항목", "það sem er tiltækt", "ce qui est disponible"),
    "Sidebar — apps by category": _row(
        "Sidofält — appar efter kategori", "Seitenleiste – Module nach Kategorie",
        "Barra lateral — aplicaciones por categoría", "侧边栏 — 按类别列出应用",
        "Barra lateral — aplicações por categoria", "साइडबार — श्रेणी के अनुसार ऐप",
        "사이드바 — 카테고리별 앱", "Hliðarstika — forrit eftir flokki",
        "Barre latérale — applications par catégorie"),
    "spaCR privacy and optional account setup": _row(
        "spaCR-integritet och valfri kontokonfiguration",
        "spaCR-Datenschutz und optionale Kontoeinrichtung",
        "Privacidad de spaCR y configuración opcional de cuentas",
        "spaCR 隐私与可选账户设置",
        "Privacidade do spaCR e configuração opcional de contas",
        "spaCR गोपनीयता और वैकल्पिक खाता सेटअप",
        "spaCR 개인정보 보호 및 선택적 계정 설정",
        "Persónuvernd spaCR og valfrjáls reikningsuppsetning",
        "Confidentialité de spaCR et configuration facultative des comptes"),
    "Enable the public GitHub issue-report action": _row(
        "Aktivera åtgärden för offentliga GitHub-felrapporter",
        "Aktion zum öffentlichen GitHub-Problembericht aktivieren",
        "Activar la acción para informar de incidencias públicas en GitHub",
        "启用公开 GitHub 问题报告操作",
        "Ativar a ação de relatório público de problemas no GitHub",
        "सार्वजनिक GitHub समस्या रिपोर्ट कार्रवाई चालू करें",
        "공개 GitHub 이슈 신고 작업 활성화",
        "Virkja opinbera GitHub-villutilkynningu",
        "Activer l’action de signalement public sur GitHub"),
    "Set up GitHub, Claude, GPT/Codex, and Gemini now": _row(
        "Konfigurera GitHub, Claude, GPT/Codex och Gemini nu",
        "GitHub, Claude, GPT/Codex und Gemini jetzt einrichten",
        "Configurar GitHub, Claude, GPT/Codex y Gemini ahora",
        "立即设置 GitHub、Claude、GPT/Codex 和 Gemini",
        "Configurar GitHub, Claude, GPT/Codex e Gemini agora",
        "GitHub, Claude, GPT/Codex और Gemini अभी सेट करें",
        "지금 GitHub, Claude, GPT/Codex 및 Gemini 설정",
        "Stilla GitHub, Claude, GPT/Codex og Gemini núna",
        "Configurer GitHub, Claude, GPT/Codex et Gemini maintenant"),
    "Include redacted diagnostic logs in report previews": _row(
        "Ta med rensade diagnostikloggar i rapportförhandsvisningar",
        "Bereinigte Diagnoseprotokolle in Berichtsvorschauen aufnehmen",
        "Incluir registros de diagnóstico censurados en las vistas previas",
        "在报告预览中包含已脱敏的诊断日志",
        "Incluir registos de diagnóstico editados nas pré-visualizações",
        "रिपोर्ट पूर्वावलोकन में संपादित निदान लॉग शामिल करें",
        "보고서 미리보기에 민감 정보가 제거된 진단 로그 포함",
        "Hafa hreinsaða greiningarannála með í forskoðun skýrslu",
        "Inclure les journaux de diagnostic expurgés dans les aperçus"),
    "Load a synthetic demo dataset for a selected core workflow in one click — no data of your own required. Use it to try spaCR before loading an experiment.": _row(
        "Läs in syntetiska demodata för ett valt kärnflöde med ett klick — inga egna data krävs. Använd dem för att prova spaCR innan du läser in ett experiment.",
        "Laden Sie mit einem Klick synthetische Demodaten für einen ausgewählten Kernablauf – eigene Daten sind nicht erforderlich. Probieren Sie damit spaCR aus, bevor Sie ein Experiment laden.",
        "Cargue con un clic datos sintéticos de demostración para uno de los flujos principales disponibles, sin necesidad de aportar datos propios. Utilícelos para probar spaCR antes de cargar un experimento.",
        "一键为选定的核心流程加载合成演示数据，无需使用自己的数据。可在加载实验之前用它试用 spaCR。",
        "Carregue com um clique dados sintéticos de demonstração para um dos fluxos principais disponíveis — não precisa dos seus próprios dados. Use-os para experimentar o spaCR antes de carregar uma experiência.",
        "चुने हुए मुख्य वर्कफ़्लो के लिए एक क्लिक में सिंथेटिक डेमो डेटा लोड करें — अपने डेटा की आवश्यकता नहीं है। कोई प्रयोग लोड करने से पहले spaCR आज़माने के लिए इसका उपयोग करें।",
        "선택한 핵심 워크플로의 합성 데모 데이터를 클릭 한 번으로 불러옵니다. 사용자 데이터는 필요하지 않습니다. 실험을 불러오기 전에 spaCR를 시험해 보세요.",
        "Hladdu tilbúnum sýnigögnum fyrir valið kjarnavinnsluferli með einum smelli — eigin gögn þarf ekki. Notaðu þau til að prófa spaCR áður en tilraun er hlaðin inn.",
        "Chargez en un clic des données synthétiques de démonstration pour l’un des flux principaux proposés, sans fournir vos propres données. Utilisez-les pour essayer spaCR avant de charger une expérience."),
    "Ctrl+K opens a searchable list of every app, every recent run, and every menu action. Ctrl+, opens Preferences. F1 shows the shortcut cheat sheet.": _row(
        "Ctrl+K öppnar en sökbar lista över alla appar, senaste körningar och menyåtgärder. Ctrl+, öppnar Inställningar. F1 visar kortkommandona.",
        "Ctrl+K öffnet eine durchsuchbare Liste aller Module, letzten Läufe und Menüaktionen. Ctrl+, öffnet die Einstellungen. F1 zeigt die Tastenkürzel.",
        "Ctrl+K abre una lista con búsqueda de todas las aplicaciones, ejecuciones recientes y acciones de menú. Ctrl+, abre Preferencias. F1 muestra los atajos.",
        "Ctrl+K 打开可搜索的应用、最近运行和菜单操作列表。Ctrl+, 打开首选项。F1 显示快捷键速查表。",
        "Ctrl+K abre uma lista pesquisável de todas as aplicações, execuções recentes e ações de menu. Ctrl+, abre as Preferências. F1 mostra os atalhos.",
        "Ctrl+K सभी ऐप, हाल की रन और मेनू कार्रवाइयों की खोज योग्य सूची खोलता है। Ctrl+, प्राथमिकताएँ खोलता है। F1 शॉर्टकट सूची दिखाता है।",
        "Ctrl+K는 모든 앱, 최근 실행 및 메뉴 작업을 검색할 수 있는 목록을 엽니다. Ctrl+,는 환경설정을 엽니다. F1은 단축키 안내를 표시합니다.",
        "Ctrl+K opnar leitanlegan lista yfir öll forrit, nýlegar keyrslur og valmyndaraðgerðir. Ctrl+, opnar Stillingar. F1 sýnir flýtilyklana.",
        "Ctrl+K ouvre une liste consultable de toutes les applications, exécutions récentes et actions de menu. Ctrl+, ouvre les Préférences. F1 affiche les raccourcis."),
    "Drop a folder of acquisition images onto Mask to set its input; Mask detects the filename regex and displays a metadata validation summary in the Console. Measure, Annotate and other modules accept the files or folders described by their input controls.": _row(
        "Släpp en mapp med insamlade bilder på Mask för att ange dess indata; Mask identifierar filnamnsmönstret och visar en sammanfattning av metadatavalideringen i konsolen. Measure, Annotate och övriga moduler tar emot de filer eller mappar som beskrivs vid deras indatakontroller.",
        "Legen Sie einen Ordner mit Aufnahmen auf Mask ab, um dessen Eingabe festzulegen; Mask erkennt das Dateinamensmuster und zeigt eine Zusammenfassung der Metadatenvalidierung in der Konsole. Measure, Annotate und andere Module nehmen die Dateien oder Ordner an, die an ihren Eingabefeldern beschrieben sind.",
        "Suelte una carpeta de imágenes adquiridas sobre Mask para definir su entrada; Mask detecta el patrón de nombres y muestra un resumen de validación de metadatos en la Consola. Measure, Annotate y los demás módulos aceptan los archivos o carpetas descritos junto a sus controles de entrada.",
        "将采集图像文件夹拖放到 Mask 以设置其输入；Mask 会检测文件名正则表达式，并在控制台中显示元数据验证摘要。Measure、Annotate 及其他模块接受其输入控件所说明的文件或文件夹。",
        "Largue uma pasta de imagens adquiridas em Mask para definir a sua entrada; Mask deteta o padrão dos nomes e mostra um resumo da validação dos metadados na Consola. Measure, Annotate e os outros módulos aceitam os ficheiros ou pastas descritos nos respetivos controlos de entrada.",
        "अधिग्रहण छवियों का फ़ोल्डर Mask पर छोड़कर उसका इनपुट तय करें; Mask फ़ाइलनाम रेगेक्स पहचानता है और कंसोल में मेटाडेटा सत्यापन सारांश दिखाता है। Measure, Annotate और अन्य मॉड्यूल अपने इनपुट नियंत्रणों में बताए गए फ़ाइल या फ़ोल्डर स्वीकार करते हैं।",
        "획득 이미지 폴더를 Mask에 놓아 입력을 설정합니다. Mask는 파일명 정규식을 감지하고 콘솔에 메타데이터 검증 요약을 표시합니다. Measure, Annotate 및 다른 모듈은 각 입력 컨트롤에 설명된 파일이나 폴더를 받습니다.",
        "Slepptu möppu með myndatökum á Mask til að velja inntakið; Mask greinir skráarnafnamynstrið og sýnir samantekt á sannprófun lýsigagna í stjórnborðinu. Measure, Annotate og aðrar einingar taka við þeim skrám eða möppum sem inntaksstýringar þeirra lýsa.",
        "Déposez un dossier d’images acquises sur Mask pour définir son entrée ; Mask détecte l’expression régulière des noms de fichiers et affiche un résumé de validation des métadonnées dans la console. Measure, Annotate et les autres modules acceptent les fichiers ou dossiers décrits par leurs contrôles d’entrée."),
    "Primary modules are grouped here into Core, Data, Results & QC, Explore, Assays and Design; related workflows are reached from their host module. Click any name to open it. Ctrl+1 through Ctrl+9 opens the first nine apps in sidebar order.": _row(
        "Primära moduler är grupperade här i Kärna, Data, Resultat och QC, Utforska, Analyser och Design; relaterade arbetsflöden nås från sin värdmodul. Klicka på ett namn för att öppna det. Ctrl+1 till Ctrl+9 öppnar de första nio apparna i sidofältets ordning.",
        "Die Hauptmodule sind hier in Kern, Daten, Ergebnisse und QC, Erkunden, Assays und Entwurf gruppiert; zugehörige Arbeitsabläufe erreichen Sie über ihr übergeordnetes Modul. Klicken Sie auf einen Namen, um ihn zu öffnen. Ctrl+1 bis Ctrl+9 öffnet die ersten neun Apps in der Reihenfolge der Seitenleiste.",
        "Los módulos principales se agrupan aquí en Principal, Datos, Resultados y CC, Explorar, Ensayos y Diseño; los flujos relacionados se abren desde su módulo anfitrión. Haga clic en un nombre para abrirlo. De Ctrl+1 a Ctrl+9 se abren las nueve primeras aplicaciones en el orden de la barra lateral.",
        "主要模块在这里分为核心、数据、结果与质控、探索、实验分析和实验设计；相关流程可从其宿主模块进入。点击名称即可打开。Ctrl+1 至 Ctrl+9 按侧边栏顺序打开前九个应用。",
        "Os módulos principais estão agrupados aqui em Principal, Dados, Resultados e CQ, Explorar, Ensaios e Planejamento; os fluxos relacionados são acedidos a partir do respetivo módulo anfitrião. Clique num nome para o abrir. De Ctrl+1 a Ctrl+9 abrem as primeiras nove aplicações pela ordem da barra lateral.",
        "मुख्य मॉड्यूल यहाँ मुख्य, डेटा, परिणाम और QC, अन्वेषण, एसे और डिज़ाइन में समूहित हैं; संबंधित वर्कफ़्लो उनके होस्ट मॉड्यूल से खोले जाते हैं। किसी नाम पर क्लिक करके उसे खोलें। Ctrl+1 से Ctrl+9 साइडबार क्रम में पहले नौ ऐप खोलते हैं।",
        "주요 모듈은 핵심, 데이터, 결과 및 QC, 탐색, 어세이 및 설계로 그룹화되어 있습니다. 관련 워크플로는 호스트 모듈에서 열 수 있습니다. 이름을 클릭하면 열립니다. Ctrl+1부터 Ctrl+9까지는 사이드바 순서대로 처음 아홉 개 앱을 엽니다.",
        "Aðaleiningar eru flokkaðar hér í Kjarna, Gögn, Niðurstöður og gæðaeftirlit, Kanna, Prófanir og Hönnun; tengd vinnsluferli eru opnuð úr hýsingareiningunni. Smelltu á heiti til að opna það. Ctrl+1 til Ctrl+9 opnar fyrstu níu forritin í röð hliðarstikunnar.",
        "Les modules principaux sont regroupés ici dans Cœur, Données, Résultats et CQ, Explorer, Essais et Conception ; les flux associés sont accessibles depuis leur module hôte. Cliquez sur un nom pour l’ouvrir. De Ctrl+1 à Ctrl+9 ouvrent les neuf premières applications dans l’ordre de la barre latérale."),
    "Crash reports go to the PUBLIC spaCR GitHub repository. They are world-readable, indexed, and cannot be reliably unpublished. A report is redacted, shown in an editable preview, and sent only when you press Send for that specific report. Account setup uses the official GitHub, Claude, Codex (GPT), and Gemini CLIs; spaCR does not store their passwords or tokens. All choices are optional and revocable in Preferences.": _row(
        "Kraschrapporter skickas till spaCR:s OFFENTLIGA GitHub-arkiv. De kan läsas av alla, indexeras och kan inte tas bort på ett tillförlitligt sätt. Rapporten rensas, visas i en redigerbar förhandsvisning och skickas bara när du trycker på Skicka för just den rapporten. Kontokonfigurationen använder de officiella kommandoradsverktygen för GitHub, Claude, Codex (GPT) och Gemini; spaCR lagrar inte deras lösenord eller token. Alla val är frivilliga och kan återkallas i Inställningar.",
        "Absturzberichte werden an das ÖFFENTLICHE spaCR-Repository auf GitHub gesendet. Sie sind weltweit lesbar, werden indexiert und können nicht zuverlässig zurückgenommen werden. Ein Bericht wird bereinigt, in einer bearbeitbaren Vorschau angezeigt und nur gesendet, wenn Sie bei diesem Bericht auf Senden klicken. Die Kontoeinrichtung verwendet die offiziellen CLIs von GitHub, Claude, Codex (GPT) und Gemini; spaCR speichert weder Passwörter noch Token. Alle Optionen sind freiwillig und können in den Einstellungen widerrufen werden.",
        "Los informes de fallos se envían al repositorio PÚBLICO de spaCR en GitHub. Cualquiera puede leerlos, se indexan y no se pueden retirar de forma fiable. El informe se censura, se muestra en una vista previa editable y solo se envía cuando pulse Enviar para ese informe concreto. La configuración de cuentas usa las CLI oficiales de GitHub, Claude, Codex (GPT) y Gemini; spaCR no almacena sus contraseñas ni tokens. Todas las opciones son voluntarias y se pueden revocar en Preferencias.",
        "崩溃报告会提交到公开的 spaCR GitHub 仓库。任何人都能阅读，搜索引擎也会收录，而且无法保证彻底撤回。报告会先脱敏并显示在可编辑的预览中；只有当你为该报告按下“发送”时才会提交。账户设置使用 GitHub、Claude、Codex（GPT）和 Gemini 的官方命令行工具；spaCR 不存储其密码或令牌。所有选项均为自愿选择，并可在首选项中撤销。",
        "Os relatórios de falhas são enviados para o repositório PÚBLICO do spaCR no GitHub. Podem ser lidos por qualquer pessoa, são indexados e não podem ser retirados de forma fiável. O relatório é editado, mostrado numa pré-visualização alterável e só é enviado quando carrega em Enviar nesse relatório específico. A configuração de contas usa as CLI oficiais do GitHub, Claude, Codex (GPT) e Gemini; o spaCR não guarda palavras-passe nem tokens. Todas as opções são voluntárias e podem ser revogadas nas Preferências.",
        "क्रैश रिपोर्ट सार्वजनिक spaCR GitHub रिपॉज़िटरी में जाती हैं। उन्हें दुनिया भर में पढ़ा और अनुक्रमित किया जा सकता है तथा उनका प्रकाशन भरोसेमंद तरीके से वापस नहीं लिया जा सकता। रिपोर्ट से संवेदनशील जानकारी हटाकर संपादन योग्य पूर्वावलोकन दिखाया जाता है और वह तभी भेजी जाती है जब आप उसी रिपोर्ट के लिए भेजें दबाते हैं। खाता सेटअप आधिकारिक GitHub, Claude, Codex (GPT) और Gemini CLI का उपयोग करता है; spaCR उनके पासवर्ड या टोकन संग्रहीत नहीं करता। सभी विकल्प वैकल्पिक हैं और प्राथमिकताओं में वापस लिए जा सकते हैं।",
        "충돌 보고서는 공개 spaCR GitHub 저장소로 전송됩니다. 누구나 읽을 수 있고 검색에 노출되며, 게시 후 완전히 회수된다고 보장할 수 없습니다. 보고서는 민감 정보가 제거된 뒤 편집 가능한 미리보기에 표시되며, 해당 보고서에서 보내기를 눌렀을 때만 전송됩니다. 계정 설정에는 공식 GitHub, Claude, Codex(GPT), Gemini CLI를 사용하며 spaCR는 비밀번호나 토큰을 저장하지 않습니다. 모든 선택 사항은 선택적이며 환경설정에서 철회할 수 있습니다.",
        "Hrunskýrslur fara í OPINBERT spaCR-safn á GitHub. Allir geta lesið þær, þær eru skráðar í leitarvélum og ekki er hægt að tryggja að þær verði afturkallaðar. Skýrslan er hreinsuð, sýnd í breytanlegri forskoðun og aðeins send þegar þú ýtir á Senda fyrir þá tilteknu skýrslu. Reikningsuppsetning notar opinber skipanalínuverkfæri GitHub, Claude, Codex (GPT) og Gemini; spaCR geymir hvorki lykilorð né aðgangslykla þeirra. Öll val eru valfrjáls og má afturkalla í Stillingum.",
        "Les rapports de plantage sont envoyés au dépôt GitHub PUBLIC de spaCR. Ils sont lisibles partout, indexés et ne peuvent pas être retirés de manière fiable. Le rapport est expurgé, affiché dans un aperçu modifiable et envoyé uniquement lorsque vous cliquez sur Envoyer pour ce rapport précis. La configuration des comptes utilise les interfaces en ligne de commande officielles de GitHub, Claude, Codex (GPT) et Gemini ; spaCR ne conserve ni mots de passe ni jetons. Tous les choix sont facultatifs et révocables dans les Préférences."),

    # ---- Home screen chrome -----------------------------------------
    "Hit List": _row(
        "Träfflista", "Trefferliste", "Lista de aciertos", "命中列表",
        "Lista de acertos", "हिट सूची", "히트 목록", "Niðurstöðulisti",
        "Liste des résultats"),
    "Methods & Results": _row(
        "Metod och resultat", "Methoden und Ergebnisse",
        "Métodos y resultados", "方法与结果", "Métodos e resultados",
        "विधियाँ और परिणाम", "방법 및 결과", "Aðferðir og niðurstöður",
        "Méthodes et résultats"),
    "Assays": _row("Analyser", "Assays", "Ensayos", "实验分析", "Ensaios", "एसे", "어세이", "Prófanir", "Essais"),
    "Alpha": _row("Alfa", "Alpha", "Alfa", "内测", "Alfa", "अल्फा", "알파", "Alfa", "Alpha"),
    "Beta": _row("Beta", "Beta", "Beta", "公测", "Beta", "बीटा", "베타", "Beta", "Bêta"),
    "Stable": _row("Stabil", "Stabil", "Estable", "稳定", "Estável", "स्थिर", "안정", "Stöðugt", "Stable"),
    "SYSTEM": _row("SYSTEM", "SYSTEM", "SISTEMA", "系统", "SISTEMA", "सिस्टम", "시스템", "KERFI", "SYSTÈME"),
    "System": _row("System", "System", "Sistema", "系统", "Sistema", "सिस्टम", "시스템", "Kerfi", "Système"),
    "RECENT RUNS": _row("SENASTE KÖRNINGAR", "LETZTE LÄUFE", "EJECUCIONES RECIENTES", "最近运行", "EXECUÇÕES RECENTES", "हाल के रन", "최근 실행", "NÝLEGAR KEYRSLUR", "EXÉCUTIONS RÉCENTES"),
    "TOTALS": _row("TOTALER", "SUMMEN", "TOTALES", "合计", "TOTAIS", "कुल", "합계", "SAMTÖLUR", "TOTAUX"),
    "QUEUED": _row("I KÖ", "IN WARTESCHLANGE", "EN COLA", "队列中", "EM FILA", "कतार में", "대기 중", "Í BIÐRÖÐ", "EN FILE"),
    "queued": _row("i kö", "in Warteschlange", "en cola", "队列中", "em fila", "कतार में", "대기 중", "í biðröð", "en file"),
    "MODULE STATE": _row("MODULSTATUS", "MODULSTATUS", "ESTADO DEL MÓDULO", "模块状态", "ESTADO DO MÓDULO", "मॉड्यूल स्थिति", "모듈 상태", "STAÐA EININGA", "ÉTAT DES MODULES"),
    "Disk": _row("Disk", "Datenträger", "Disco", "磁盘", "Disco", "डिस्क", "디스크", "Diskur", "Disque"),
    "job": _row("jobb", "Auftrag", "trabajo", "任务", "tarefa", "जॉब", "작업", "verk", "tâche"),
    "Walkthroughs": _row("Genomgångar", "Rundgänge", "Recorridos", "分步导览", "Percursos guiados", "मार्गदर्शिकाएँ", "안내 둘러보기", "Leiðsagnir", "Visites guidées"),
    "Welcome to spaCR": _row("Välkommen till spaCR", "Willkommen bei spaCR", "Bienvenido a spaCR", "欢迎使用 spaCR", "Bem-vindo ao spaCR", "spaCR में आपका स्वागत है", "spaCR에 오신 것을 환영합니다", "Velkomin í spaCR", "Bienvenue dans spaCR"),
    "Readouts that measure a biological assay rather than a pipeline stage.": _row(
        "Mätvärden som mäter en biologisk analys snarare än ett pipelinesteg.",
        "Messgrößen, die einen biologischen Assay messen statt einer Pipeline-Stufe.",
        "Lecturas que miden un ensayo biológico en lugar de una etapa del flujo.",
        "衡量生物学实验本身而非流程步骤的读出指标。",
        "Leituras que medem um ensaio biológico em vez de uma etapa do fluxo.",
        "ऐसे रीडआउट जो पाइपलाइन चरण के बजाय जैविक एसे को मापते हैं।",
        "파이프라인 단계가 아니라 생물학적 어세이를 측정하는 판독값입니다.",
        "Mælingar sem mæla líffræðilega prófun fremur en þrep í vinnsluferli.",
        "Mesures qui évaluent un essai biologique plutôt qu’une étape du flux."),

    # ---- The GitHub row on the setup screen -------------------------
    "signed in through the GitHub CLI": _row("inloggad via GitHub CLI", "über die GitHub-CLI angemeldet", "sesión iniciada con la CLI de GitHub", "已通过 GitHub CLI 登录", "sessão iniciada através da CLI do GitHub", "GitHub CLI के माध्यम से साइन इन", "GitHub CLI로 로그인됨", "innskráð(ur) gegnum GitHub CLI", "connecté via la CLI GitHub"),
    "signed in through GITHUB_TOKEN": _row("inloggad via GITHUB_TOKEN", "über GITHUB_TOKEN angemeldet", "sesión iniciada con GITHUB_TOKEN", "已通过 GITHUB_TOKEN 登录", "sessão iniciada através de GITHUB_TOKEN", "GITHUB_TOKEN के माध्यम से साइन इन", "GITHUB_TOKEN으로 로그인됨", "innskráð(ur) gegnum GITHUB_TOKEN", "connecté via GITHUB_TOKEN"),
    "signed in with a stored token": _row("inloggad med en sparad token", "mit gespeichertem Token angemeldet", "sesión iniciada con un token guardado", "已使用已保存的令牌登录", "sessão iniciada com um token guardado", "संग्रहीत टोकन से साइन इन", "저장된 토큰으로 로그인됨", "innskráð(ur) með vistuðum aðgangslykli", "connecté avec un jeton enregistré"),
    "not signed in — reports open in your browser": _row("inte inloggad — rapporter öppnas i din webbläsare", "nicht angemeldet – Berichte öffnen sich im Browser", "sin sesión iniciada: los informes se abren en su navegador", "未登录 — 报告将在浏览器中打开", "sem sessão iniciada — os relatórios abrem no seu navegador", "साइन इन नहीं — रिपोर्ट आपके ब्राउज़र में खुलेंगी", "로그인되지 않음 — 보고서는 브라우저에서 열립니다", "ekki innskráð(ur) — skýrslur opnast í vafranum þínum", "non connecté — les rapports s’ouvrent dans votre navigateur"),
    "the GitHub CLI is not installed — reports open in your browser": _row("GitHub CLI är inte installerat — rapporter öppnas i din webbläsare", "die GitHub-CLI ist nicht installiert – Berichte öffnen sich im Browser", "la CLI de GitHub no está instalada: los informes se abren en su navegador", "未安装 GitHub CLI — 报告将在浏览器中打开", "a CLI do GitHub não está instalada — os relatórios abrem no seu navegador", "GitHub CLI संस्थापित नहीं है — रिपोर्ट ब्राउज़र में खुलेंगी", "GitHub CLI가 설치되어 있지 않음 — 보고서는 브라우저에서 열립니다", "GitHub CLI er ekki uppsett — skýrslur opnast í vafranum þínum", "la CLI GitHub n’est pas installée — les rapports s’ouvrent dans votre navigateur"),
    "starting GitHub sign-in…": _row("startar GitHub-inloggning…", "GitHub-Anmeldung wird gestartet…", "iniciando el acceso a GitHub…", "正在开始 GitHub 登录…", "a iniciar a autenticação no GitHub…", "GitHub साइन-इन शुरू हो रहा है…", "GitHub 로그인을 시작하는 중…", "ræsi GitHub-innskráningu…", "démarrage de la connexion GitHub…"),
    "`gh auth login` would not start — run it in a terminal": _row("`gh auth login` startade inte — kör det i en terminal", "`gh auth login` ließ sich nicht starten – führen Sie es im Terminal aus", "`gh auth login` no se pudo iniciar: ejecútelo en una terminal", "`gh auth login` 无法启动 — 请在终端中运行", "`gh auth login` não arrancou — execute-o num terminal", "`gh auth login` शुरू नहीं हुआ — इसे टर्मिनल में चलाएँ", "`gh auth login`을 시작할 수 없습니다 — 터미널에서 실행하세요", "`gh auth login` ræstist ekki — keyrðu það í skel", "`gh auth login` n’a pas démarré — lancez-le dans un terminal"),
    "enter {code} in {where}": _row("ange {code} i {where}", "{code} in {where} eingeben", "introduzca {code} en {where}", "在 {where} 中输入 {code}", "introduza {code} em {where}", "{where} में {code} दर्ज करें", "{where}에 {code} 입력", "sláðu {code} inn í {where}", "saisissez {code} dans {where}"),

    # ---- The Demos menu ---------------------------------------------
    "Mask demo…": _row("Maskdemo…", "Masken-Demo…", "Demo de máscaras…", "掩膜演示…", "Demonstração de máscaras…", "मास्क डेमो…", "마스크 데모…", "Maskasýnishorn…", "Démo de masques…"),
    "Measure demo…": _row("Mätdemo…", "Mess-Demo…", "Demo de medición…", "测量演示…", "Demonstração de medição…", "मापन डेमो…", "측정 데모…", "Mælingasýnishorn…", "Démo de mesure…"),
    "Crop demo…": _row("Beskärningsdemo…", "Ausschnitt-Demo…", "Demo de recortes…", "裁剪演示…", "Demonstração de recortes…", "क्रॉप डेमो…", "크롭 데모…", "Útklippusýnishorn…", "Démo de découpe…"),
    "Classify demo…": _row("Klassificeringsdemo…", "Klassifizierungs-Demo…", "Demo de clasificación…", "分类演示…", "Demonstração de classificação…", "वर्गीकरण डेमो…", "분류 데모…", "Flokkunarsýnishorn…", "Démo de classification…"),
    "Timelapse demo…": _row("Tidsseriedemo…", "Zeitraffer-Demo…", "Demo de lapso de tiempo…", "延时演示…", "Demonstração de time-lapse…", "टाइमलैप्स डेमो…", "타임랩스 데모…", "Tímaraðarsýnishorn…", "Démo time-lapse…"),
    "Sequencing demo…": _row("Sekvenseringsdemo…", "Sequenzierungs-Demo…", "Demo de secuenciación…", "测序演示…", "Demonstração de sequenciação…", "सीक्वेंसिंग डेमो…", "시퀀싱 데모…", "Raðgreiningarsýnishorn…", "Démo de séquençage…"),
    "Open Demos menu": _row("Öppna Demo-menyn", "Demos-Menü öffnen", "Abrir el menú Demostraciones", "打开演示菜单", "Abrir o menu Demonstrações", "डेमो मेनू खोलें", "데모 메뉴 열기", "Opna Sýnishorn-valmynd", "Ouvrir le menu Démonstrations"),

    # ---- The settings-count line under every panel -------------------
    "Showing {shown} of {total} settings": _row("Visar {shown} av {total} inställningar", "{shown} von {total} Einstellungen angezeigt", "Mostrando {shown} de {total} ajustes", "显示 {total} 项设置中的 {shown} 项", "A mostrar {shown} de {total} definições", "{total} में से {shown} सेटिंग दिख रही हैं", "설정 {total}개 중 {shown}개 표시", "Sýni {shown} af {total} stillingum", "Affichage de {shown} réglages sur {total}"),
    "Showing all {total} settings.": _row("Visar alla {total} inställningar.", "Alle {total} Einstellungen werden angezeigt.", "Mostrando los {total} ajustes.", "显示全部 {total} 项设置。", "A mostrar todas as {total} definições.", "सभी {total} सेटिंग दिख रही हैं।", "설정 {total}개 모두 표시.", "Sýni allar {total} stillingarnar.", "Affichage des {total} réglages."),
    "{total} settings.": _row("{total} inställningar.", "{total} Einstellungen.", "{total} ajustes.", "{total} 项设置。", "{total} definições.", "{total} सेटिंग।", "설정 {total}개.", "{total} stillingar.", "{total} réglages."),
    "{n} more under All settings": _row("{n} till under Alla inställningar", "{n} weitere unter „Alle Einstellungen“", "{n} más en Todos los ajustes", "另有 {n} 项在“全部设置”中", "mais {n} em Todas as definições", "‘सभी सेटिंग’ में {n} और", "‘모든 설정’에 {n}개 더", "{n} til viðbótar undir Allar stillingar", "{n} de plus sous Tous les réglages"),
    "modified only": _row("endast ändrade", "nur geänderte", "solo modificados", "仅已修改", "apenas modificadas", "केवल बदली हुई", "변경된 항목만", "aðeins breyttar", "modifiés uniquement"),
    "No setting matches. Clear the search box, or switch to All settings.": _row(
        "Ingen inställning matchar. Rensa sökrutan eller byt till Alla inställningar.",
        "Keine Einstellung passt. Leeren Sie das Suchfeld oder wechseln Sie zu „Alle Einstellungen“.",
        "Ningún ajuste coincide. Borre el cuadro de búsqueda o cambie a Todos los ajustes.",
        "没有匹配的设置。请清空搜索框，或切换到“全部设置”。",
        "Nenhuma definição corresponde. Limpe a caixa de pesquisa ou mude para Todas as definições.",
        "कोई सेटिंग मेल नहीं खाती। खोज बॉक्स साफ़ करें, या ‘सभी सेटिंग’ पर जाएँ।",
        "일치하는 설정이 없습니다. 검색창을 비우거나 ‘모든 설정’으로 전환하세요.",
        "Engin stilling passar. Hreinsaðu leitarreitinn eða skiptu yfir í Allar stillingar.",
        "Aucun réglage ne correspond. Videz le champ de recherche ou passez à Tous les réglages."),

    # ---- Figure and settings panel chrome ---------------------------
    "Figures": _row("Figurer", "Abbildungen", "Figuras", "图表", "Figuras", "आकृतियाँ", "그림", "Myndir", "Figures"),
    "Live preview": _row("Direktförhandsvisning", "Live-Vorschau", "Vista previa en vivo", "实时预览", "Pré-visualização ao vivo", "लाइव पूर्वावलोकन", "실시간 미리보기", "Bein forskoðun", "Aperçu en direct"),
    "Clear figures": _row("Rensa figurer", "Abbildungen leeren", "Borrar figuras", "清除图表", "Limpar figuras", "आकृतियाँ हटाएँ", "그림 지우기", "Hreinsa myndir", "Effacer les figures"),
    "Normalise": _row("Normalisera", "Normalisieren", "Normalizar", "归一化", "Normalizar", "सामान्यीकृत करें", "정규화", "Staðla", "Normaliser"),
    # FOUR PLOT WORDS THAT ARE ORDINARY ENGLISH WORDS TOO, and the bulk
    # catalog picked the ordinary sense of each: "Legend" as the myth, "Grid"
    # as a network (and, in Icelandic, as a person's name), "Opacity" as
    # ruthlessness, "Colour" as something else entirely. A row here is read
    # before the bulk catalog, so this is where the chart sense is pinned.
    "Colour": _row(
        "Färg", "Farbe", "Color", "颜色", "Cor", "रंग", "색상", "Litur",
        "Couleur"),
    "Opacity": _row(
        "Opacitet", "Deckkraft", "Opacidad", "不透明度", "Opacidade",
        "अपारदर्शिता", "불투명도", "Ógegnsæi", "Opacité"),
    "Legend": _row(
        "Teckenförklaring", "Legende", "Leyenda", "图例", "Legenda",
        "लेजेंड", "범례", "Skýringar", "Légende"),
    "Grid": _row(
        "Rutnät", "Gitter", "Cuadrícula", "网格", "Grade", "ग्रिड", "격자",
        "Hnitanet", "Grille"),
    "Hover any setting for details, or select ⓘ for documentation.": _row(
        "Håll pekaren över en inställning för detaljer, eller välj ⓘ för dokumentation.",
        "Zeigen Sie auf eine Einstellung für Details, oder wählen Sie ⓘ für die Dokumentation.",
        "Pase el cursor por un ajuste para ver detalles, o seleccione ⓘ para la documentación.",
        "将指针悬停在某项设置上可查看详情，或选择 ⓘ 查看文档。",
        "Passe o cursor sobre uma definição para ver detalhes, ou selecione ⓘ para a documentação.",
        "विवरण के लिए किसी सेटिंग पर कर्सर ले जाएँ, या दस्तावेज़ के लिए ⓘ चुनें।",
        "설정 위에 마우스를 올리면 설명이, ⓘ를 선택하면 문서가 나옵니다.",
        "Haltu bendlinum yfir stillingu til að sjá nánar, eða veldu ⓘ fyrir leiðbeiningar.",
        "Survolez un réglage pour les détails, ou choisissez ⓘ pour la documentation."),
    "Hover a settings category for what the group decides, or open one to keep it here.": _row(
        "Håll pekaren över en inställningskategori för vad gruppen avgör, eller öppna en för att behålla den här.",
        "Zeigen Sie auf eine Einstellungskategorie, um zu sehen, worüber die Gruppe entscheidet, oder öffnen Sie eine, um sie hier zu behalten.",
        "Pase el cursor por una categoría de ajustes para ver qué decide el grupo, o abra una para mantenerla aquí.",
        "将指针悬停在设置类别上可查看该组决定什么，或打开一个以将其保留在此处。",
        "Passe o cursor sobre uma categoria de definições para ver o que o grupo decide, ou abra uma para a manter aqui.",
        "समूह क्या तय करता है यह देखने के लिए किसी सेटिंग श्रेणी पर कर्सर ले जाएँ, या इसे यहाँ रखने के लिए कोई एक खोलें।",
        "설정 범주 위에 마우스를 올리면 그 묶음이 무엇을 정하는지 보이고, 하나를 열면 여기에 고정됩니다.",
        "Haltu bendlinum yfir stillingaflokk til að sjá hvað hópurinn ræður, eða opnaðu einn til að halda honum hér.",
        "Survolez une catégorie de réglages pour voir ce que le groupe décide, ou ouvrez-en une pour la garder ici."),

    # ---- What a sweep of six screens in Swedish found still English
    # Section headings, menu entries, status lines and the hint strip:
    # captions that reach tr() but had no row, and captions that were
    # composed before the lookup so no row could ever have matched.
    "Click to fold {name} away, and click again to bring it back. The panel above takes the space.": _row(
        "Klicka för att fälla ihop {name}, och klicka igen för att fälla ut den. Panelen ovanför tar utrymmet.",
        "Klicken, um {name} einzuklappen, und erneut klicken, um es zurückzuholen. Der Bereich darüber nimmt den Platz ein.",
        "Haga clic para contraer {name}, y haga clic de nuevo para volver a mostrarlo. El panel de arriba ocupa el espacio.",
        "点击可折叠 {name}，再次点击可将其重新展开。上方的面板会占用腾出的空间。",
        "Clique para recolher {name} e clique novamente para restaurar. O painel acima ocupa o espaço.",
        "{name} को समेटने के लिए क्लिक करें, और वापस लाने के लिए फिर से क्लिक करें। ऊपर वाला पैनल यह जगह ले लेता है।",
        "클릭하면 {name}이(가) 접히고, 다시 클릭하면 되돌아옵니다. 위쪽 패널이 그 공간을 차지합니다.",
        "Smelltu til að fella {name} saman og smelltu aftur til að opna það. Spjaldið fyrir ofan tekur plássið.",
        "Cliquez pour replier {name}, et cliquez encore pour l’afficher de nouveau. Le panneau du dessus prend la place."),
    "Acquisition & Axes": _row(
        "Insamling & axlar",
        "Aufnahme & Achsen",
        "Adquisición y ejes",
        "采集与坐标轴",
        "Aquisição e eixos",
        "अधिग्रहण और अक्ष",
        "획득 및 축",
        "Myndataka og ásar",
        "Acquisition et axes"),
    "Additional Settings": _row(
        "Ytterligare inställningar",
        "Weitere Einstellungen",
        "Ajustes adicionales",
        "其他设置",
        "Configurações adicionais",
        "अतिरिक्त सेटिंग्स",
        "추가 설정",
        "Viðbótarstillingar",
        "Paramètres supplémentaires"),
    "Assay Inputs": _row(
        "Analysindata",
        "Assay-Eingaben",
        "Entradas del ensayo",
        "实验输入",
        "Entradas do ensaio",
        "परीक्षण इनपुट",
        "어세이 입력",
        "Inntak prófunar",
        "Entrées du test"),
    "Attribution Method": _row(
        "Metod för tilldelning",
        "Attributionsmethode",
        "Método de atribución",
        "归因方法",
        "Método de atribuição",
        "एट्रिब्यूशन विधि",
        "기여도 방법",
        "Eignunaraðferð",
        "Méthode d’attribution"),
    "Background & Denoising": _row(
        "Bakgrund & brusreducering",
        "Hintergrund & Entrauschen",
        "Fondo y reducción de ruido",
        "背景与去噪",
        "Fundo e remoção de ruído",
        "पृष्ठभूमि और शोर निवारण",
        "배경 및 노이즈 제거",
        "Bakgrunnur og suðhreinsun",
        "Fond et débruitage"),
    "Barcode References": _row(
        "Streckkodsreferenser",
        "Barcode-Referenzen",
        "Referencias de códigos de barras",
        "条形码参考表",
        "Referências de códigos de barras",
        "बारकोड संदर्भ",
        "바코드 참조",
        "Strikamerkjatilvísanir",
        "Références de codes-barres"),
    "Detection Thresholds": _row(
        "Detektionströsklar",
        "Erkennungsschwellen",
        "Umbrales de detección",
        "检测阈值",
        "Limiares de detecção",
        "पहचान सीमाएँ",
        "검출 임계값",
        "Greiningarþröskuldar",
        "Seuils de détection"),
    "Dimensionality Reduction": _row(
        "Dimensionsreduktion",
        "Dimensionsreduktion",
        "Reducción de dimensionalidad",
        "降维",
        "Redução de dimensionalidade",
        "विमीयता न्यूनीकरण",
        "차원 축소",
        "Víddafækkun",
        "Réduction de dimensionnalité"),
    "Effect & Prevalence": _row(
        "Effekt & prevalens",
        "Effekt & Prävalenz",
        "Efecto y prevalencia",
        "效应与发生率",
        "Efeito e prevalência",
        "प्रभाव और व्यापकता",
        "효과 및 유병률",
        "Áhrif og algengi",
        "Effet et prévalence"),
    "Embedding Search": _row(
        "Inbäddningssökning",
        "Einbettungssuche",
        "Búsqueda de incrustación",
        "嵌入搜索",
        "Pesquisa de incorporação",
        "एम्बेडिंग खोज",
        "임베딩 검색",
        "Innfellingarleit",
        "Recherche de plongement"),
    "Estimator Tuning": _row(
        "Estimatorinställning",
        "Schätzer-Abstimmung",
        "Ajuste del estimador",
        "估计器调优",
        "Ajuste do estimador",
        "एस्टिमेटर ट्यूनिंग",
        "추정기 튜닝",
        "Fínstilling metils",
        "Réglage de l’estimateur"),
    "Feature Preparation": _row(
        "Förberedelse av egenskaper",
        "Merkmalsaufbereitung",
        "Preparación de características",
        "特征准备",
        "Preparação das características",
        "विशेषता तैयारी",
        "특징 준비",
        "Undirbúningur eiginleika",
        "Préparation des caractéristiques"),
    "Feature Selection & Importance": _row(
        "Egenskapsurval & betydelse",
        "Merkmalsauswahl & Bedeutung",
        "Selección de características e importancia",
        "特征选择与重要性",
        "Seleção de características e importância",
        "विशेषता चयन और महत्व",
        "특징 선택 및 중요도",
        "Val eiginleika og mikilvægi",
        "Sélection et importance des caractéristiques"),
    "Field Sampling": _row(
        "Fältprovtagning",
        "Feldstichprobe",
        "Muestreo de campos",
        "视野采样",
        "Amostragem de campos",
        "फ़ील्ड नमूनाकरण",
        "필드 샘플링",
        "Úrtak sviða",
        "Échantillonnage des champs"),
    "Importance & diagnostics": _row(
        "Betydelse & diagnostik",
        "Bedeutung & Diagnose",
        "Importancia y diagnóstico",
        "重要性与诊断",
        "Importância e diagnóstico",
        "महत्व और निदान",
        "중요도 및 진단",
        "Mikilvægi og greining",
        "Importance et diagnostic"),
    "Library Design": _row(
        "Biblioteksdesign",
        "Bibliotheksentwurf",
        "Diseño de la biblioteca",
        "文库设计",
        "Planejamento da biblioteca",
        "लाइब्रेरी डिज़ाइन",
        "라이브러리 설계",
        "Hönnun safns",
        "Conception de la banque"),
    "Map Quantification": _row(
        "Kartkvantifiering",
        "Kartenquantifizierung",
        "Cuantificación del mapa",
        "图谱定量",
        "Quantificação do mapa",
        "मानचित्र परिमाणीकरण",
        "지도 정량화",
        "Magngreining korta",
        "Quantification des cartes"),
    "Position & Collision Checks": _row(
        "Positions- & kollisionskontroller",
        "Positions- & Kollisionsprüfungen",
        "Comprobaciones de posición y colisión",
        "位置与碰撞检查",
        "Verificações de posição e colisão",
        "स्थिति और टकराव जाँच",
        "위치 및 충돌 검사",
        "Staðsetningar- og árekstraathuganir",
        "Vérifications de position et de collision"),
    "Post-processing": _row(
        "Efterbearbetning",
        "Nachbearbeitung",
        "Posprocesamiento",
        "后处理",
        "Pós-processamento",
        "पश्च-प्रसंस्करण",
        "후처리",
        "Eftirvinnsla",
        "Post-traitement"),
    "Preview & Diagnostics": _row(
        "Förhandsvisning & diagnostik",
        "Vorschau & Diagnose",
        "Vista previa y diagnóstico",
        "预览与诊断",
        "Pré-visualização e diagnóstico",
        "पूर्वावलोकन और निदान",
        "미리보기 및 진단",
        "Forskoðun og greining",
        "Aperçu et diagnostic"),
    "QC & Failure Handling": _row(
        "QC & felhantering",
        "QC & Fehlerbehandlung",
        "QC y gestión de fallos",
        "质控与失败处理",
        "CQ e tratamento de falhas",
        "QC और विफलता प्रबंधन",
        "QC 및 실패 처리",
        "Gæðaeftirlit og meðhöndlun bilana",
        "CQ et gestion des échecs"),
    "Read Parsing": _row(
        "Tolkning av läsningar",
        "Read-Parsing",
        "Análisis de lecturas",
        "读段解析",
        "Análise das leituras",
        "रीड पार्सिंग",
        "리드 파싱",
        "Þáttun raðlesa",
        "Analyse des lectures"),
    "Reference & Count Tables": _row(
        "Referens- & räknetabeller",
        "Referenz- & Zähltabellen",
        "Tablas de referencia y de recuento",
        "参考表与计数表",
        "Tabelas de referência e de contagem",
        "संदर्भ और गिनती तालिकाएँ",
        "참조 및 카운트 테이블",
        "Tilvísana- og talningatöflur",
        "Tables de référence et de comptage"),
    "Replication Scoring": _row(
        "Poängsättning av replikation",
        "Replikationsbewertung",
        "Puntuación de la replicación",
        "复制评分",
        "Pontuação da replicação",
        "प्रतिकृति स्कोरिंग",
        "복제 점수 산출",
        "Stigagjöf fjölgunar",
        "Score de réplication"),
    "Rows & Missing Values": _row(
        "Rader & saknade värden",
        "Zeilen & fehlende Werte",
        "Filas y valores faltantes",
        "行与缺失值",
        "Linhas e valores ausentes",
        "पंक्तियाँ और अनुपलब्ध मान",
        "행 및 결측값",
        "Raðir og gildi sem vantar",
        "Lignes et valeurs manquantes"),
    "Selected hit": _row(
        "Vald träff",
        "Ausgewählter Treffer",
        "Acierto seleccionado",
        "所选命中",
        "Acerto selecionado",
        "चयनित हिट",
        "선택한 히트",
        "Valin niðurstaða",
        "Hit sélectionné"),
    "Sequencing Depth": _row(
        "Sekvenseringsdjup",
        "Sequenzierungstiefe",
        "Profundidad de secuenciación",
        "测序深度",
        "Profundidade de sequenciamento",
        "सीक्वेंसिंग गहराई",
        "시퀀싱 깊이",
        "Raðgreiningardýpt",
        "Profondeur de séquençage"),
    "Show the columns": _row(
        "Visa kolumnerna",
        "Die Spalten anzeigen",
        "Mostrar las columnas",
        "显示各列",
        "Mostrar as colunas",
        "स्तंभ दिखाएँ",
        "열 표시",
        "Sýna dálkana",
        "Afficher les colonnes"),
    "Spectral Embedding": _row(
        "Spektral inbäddning",
        "Spektrale Einbettung",
        "Incrustación espectral",
        "谱嵌入",
        "Incorporação espectral",
        "स्पेक्ट्रल एम्बेडिंग",
        "스펙트럼 임베딩",
        "Rófinnfelling",
        "Plongement spectral"),
    "Starting Point": _row(
        "Utgångspunkt",
        "Ausgangspunkt",
        "Punto de partida",
        "起点",
        "Ponto de partida",
        "प्रारंभिक बिंदु",
        "시작 지점",
        "Upphafspunktur",
        "Point de départ"),
    "Starvation & Exclusion": _row(
        "Svält & uteslutning",
        "Unterversorgung & Ausschluss",
        "Pozos hambrientos y exclusión",
        "读数匮乏与排除",
        "Privação e exclusão",
        "भुखमरी और बहिष्करण",
        "결핍 및 제외",
        "Svelti og útilokun",
        "Privation et exclusion"),
    "Threshold Sweep": _row(
        "Tröskelsvep",
        "Schwellenwert-Sweep",
        "Barrido de umbrales",
        "阈值扫描",
        "Varredura de limiar",
        "सीमा स्वीप",
        "임계값 스윕",
        "Þröskuldssveip",
        "Balayage de seuils"),
    "Thresholding": _row(
        "Tröskelsättning",
        "Schwellenwertbildung",
        "Umbralización",
        "阈值处理",
        "Limiarização",
        "सीमा निर्धारण",
        "임계값 처리",
        "Þröskuldun",
        "Seuillage"),
    "Vacuole Assignment": _row(
        "Tilldelning av vakuoler",
        "Vakuolenzuordnung",
        "Asignación de vacuolas",
        "空泡归属",
        "Designação de vacúolos",
        "वैक्यूओल आवंटन",
        "액포 할당",
        "Úthlutun vakúóla",
        "Affectation des vacuoles"),
    "Visualization & Diagnostics": _row(
        "Visualisering & diagnostik",
        "Visualisierung & Diagnose",
        "Visualización y diagnóstico",
        "可视化与诊断",
        "Visualização e diagnóstico",
        "विज़ुअलाइज़ेशन और निदान",
        "시각화 및 진단",
        "Myndræn framsetning og greining",
        "Visualisation et diagnostic"),
    "Volumetric Processing (Beta)": _row(
        "Volymetrisk bearbetning (Beta)",
        "Volumetrische Verarbeitung (Beta)",
        "Procesamiento volumétrico (Beta)",
        "体积处理 (公测)",
        "Processamento volumétrico (Beta)",
        "वॉल्यूमेट्रिक प्रसंस्करण (Beta)",
        "볼륨 처리 (베타)",
        "Rúmmálsvinnsla (Beta)",
        "Traitement volumétrique (Bêta)"),
    "Well Expectations": _row(
        "Förväntningar per brunn",
        "Erwartungen pro Well",
        "Expectativas por pozo",
        "每孔预期",
        "Expectativas por poço",
        "वेल अपेक्षाएँ",
        "웰 기대치",
        "Væntingar um brunna",
        "Attentes par puits"),
    "Appearance": _row(
        "Utseende",
        "Erscheinungsbild",
        "Apariencia",
        "外观",
        "Aparência",
        "रूप-रंग",
        "모양",
        "Útlit",
        "Apparence"),
    "Axis scale": _row(
        "Axelskala",
        "Achsenskala",
        "Escala del eje",
        "坐标轴刻度",
        "Escala do eixo",
        "अक्ष स्केल",
        "축 스케일",
        "Ásakvarði",
        "Échelle des axes"),
    "Group colours": _row(
        "Gruppfärger",
        "Gruppenfarben",
        "Colores de los grupos",
        "分组颜色",
        "Cores dos grupos",
        "समूह के रंग",
        "그룹 색상",
        "Litir hópa",
        "Couleurs des groupes"),
    "Colour every mark belonging to {group}.": _row(
        "Färga varje markering som hör till {group}.",
        "Alle Markierungen einfärben, die zu {group} gehören.",
        "Colorear todas las marcas que pertenecen a {group}.",
        "为属于 {group} 的所有标记着色。",
        "Colorir todas as marcas pertencentes a {group}.",
        "{group} से संबंधित हर चिह्न को रंग दें।",
        "{group}에 속한 모든 마크의 색상을 지정합니다.",
        "Lita öll merki sem tilheyra {group}.",
        "Colorer toutes les marques appartenant à {group}."),
    "({count} more groups not listed)": _row(
        "({count} fler grupper visas inte)",
        "({count} weitere Gruppen nicht aufgeführt)",
        "({count} grupos más no listados)",
        "（另有 {count} 个组未列出）",
        "(mais {count} grupos não listados)",
        "({count} और समूह सूची में नहीं)",
        "(표시되지 않은 그룹 {count}개 더 있음)",
        "({count} hópar til viðbótar eru ekki sýndir)",
        "({count} autres groupes non affichés)"),
    "Colour for {group}": _row(
        "Färg för {group}",
        "Farbe für {group}",
        "Color para {group}",
        "{group} 的颜色",
        "Cor para {group}",
        "{group} के लिए रंग",
        "{group} 색상",
        "Litur fyrir {group}",
        "Couleur de {group}"),
    "Line colour": _row(
        "Linjens färg",
        "Linienfarbe",
        "Color de línea",
        "线条颜色",
        "Cor da linha",
        "लाइन का रंग",
        "라인 색상",
        "Línulitur",
        "Couleur de la ligne"),
    "Font colour": _row(
        "Teckensnittsfärg",
        "Schriftfarbe",
        "Color de fuente",
        "字体颜色",
        "Cor da fonte",
        "फ़ॉन्ट का रंग",
        "글꼴 색상",
        "Leturlitur",
        "Couleur de la police"),
    "Generate a synthetic {app} dataset and open it in the matching app.": _row(
        "Skapa en syntetisk datauppsättning för {app} och öppna den i motsvarande app.",
        "Einen synthetischen {app}-Datensatz erzeugen und im passenden Modul öffnen.",
        "Genere un conjunto de datos sintético de {app} y ábralo en la aplicación correspondiente.",
        "生成一个合成的 {app} 数据集，并在对应的应用中打开。",
        "Gerar um conjunto de dados sintético de {app} e abri-lo no aplicativo correspondente.",
        "एक सिंथेटिक {app} डेटासेट बनाएँ और उसे संबंधित ऐप में खोलें।",
        "합성 {app} 데이터셋을 생성하고 해당 앱에서 엽니다.",
        "Búa til tilbúið {app}-gagnasafn og opna það í samsvarandi forriti.",
        "Génère un ensemble de données {app} synthétique et l’ouvre dans l’application correspondante."),
    "Spatial phenotype analysis of CRISPR&#8209;Cas9 screens": _row(
        "Rumslig fenotypanalys av CRISPR&#8209;Cas9-screeningar",
        "Räumliche Phänotypanalyse von CRISPR&#8209;Cas9-Screens",
        "Análisis espacial de fenotipos en cribados CRISPR&#8209;Cas9",
        "CRISPR&#8209;Cas9 筛选的空间表型分析",
        "Análise espacial de fenótipos em triagens CRISPR&#8209;Cas9",
        "CRISPR&#8209;Cas9 स्क्रीन का स्थानिक फेनोटाइप विश्लेषण",
        "CRISPR&#8209;Cas9 스크린의 공간 표현형 분석",
        "Rúmræn svipgerðargreining á CRISPR&#8209;Cas9-skimunum",
        "Analyse spatiale du phénotype des criblages CRISPR&#8209;Cas9"),
    "Licensed under the {name}.": _row(
        "Licensierad under {name}.",
        "Lizenziert unter der {name}.",
        "Licenciado bajo la {name}.",
        "依据 {name} 授权。",
        "Licenciado sob a {name}.",
        "{name} के तहत लाइसेंस प्राप्त।",
        "{name}에 따라 라이선스가 부여됩니다.",
        "Gefið út með leyfinu {name}.",
        "Sous licence {name}."),
    "Free for research and other noncommercial use.": _row(
        "Fri för forskning och annan icke-kommersiell användning.",
        "Kostenlos für die Forschung und andere nichtkommerzielle Nutzung.",
        "Gratuito para investigación y otros usos no comerciales.",
        "供研究及其他非商业用途免费使用。",
        "Gratuito para pesquisa e outros usos não comerciais.",
        "शोध और अन्य गैर-व्यावसायिक उपयोग के लिए निःशुल्क।",
        "연구 및 기타 비상업적 용도로는 무료입니다.",
        "Ókeypis til rannsókna og annarra nota sem ekki eru í viðskiptaskyni.",
        "Gratuit pour la recherche et tout autre usage non commercial."),
    "Checking for updates…": _row(
        "Söker efter uppdateringar…",
        "Wird nach Updates gesucht…",
        "Buscando actualizaciones…",
        "正在检查更新…",
        "Verificando atualizações…",
        "अपडेट जाँचे जा रहे हैं…",
        "업데이트 확인 중…",
        "Leita að uppfærslum…",
        "Recherche de mises à jour…"),
    "Upgrading spaCR…": _row(
        "Uppgraderar spaCR…",
        "spaCR wird aktualisiert…",
        "Actualizando spaCR…",
        "正在升级 spaCR…",
        "Atualizando o spaCR…",
        "spaCR अपग्रेड हो रहा है…",
        "spaCR 업그레이드 중…",
        "Uppfæri spaCR…",
        "Mise à niveau de spaCR…"),
    "Console context off": _row(
        "Konsolkontext av",
        "Konsole-Kontext aus",
        "Contexto de Consola desactivado",
        "控制台 上下文已关闭",
        "Contexto do Console desativado",
        "कंसोल संदर्भ बंद",
        "콘솔 컨텍스트 꺼짐",
        "Slökkt á Stjórnborð-samhengi",
        "Contexte de console désactivé"),
    "Console context: {n} chars sent": _row(
        "Konsolkontext: {n} tecken skickade",
        "Konsole-Kontext: {n} Zeichen gesendet",
        "Contexto de Consola: {n} caracteres enviados",
        "控制台 上下文：已发送 {n} 个字符",
        "Contexto do Console: {n} caracteres enviados",
        "कंसोल संदर्भ: {n} वर्ण भेजे गए",
        "콘솔 컨텍스트: {n}자 전송됨",
        "Stjórnborð-samhengi: {n} stafir sendir",
        "Contexte de console : {n} caractères envoyés"),
    ", {n} dropped": _row(
        ", {n} borttagna",
        ", {n} verworfen",
        ", {n} descartados",
        "，已丢弃 {n} 个字符",
        ", {n} descartados",
        ", {n} छोड़े गए",
        ", {n}자 잘림",
        ", {n} sleppt",
        ", {n} écartés"),
    "Drop a folder of images anywhere on this window, or {offer}. You can also type a path into the src field below.": _row(
        "Släpp en mapp med bilder var som helst i det här fönstret, eller {offer}. Du kan också skriva en sökväg i fältet src nedan.",
        "Legen Sie einen Bildordner irgendwo auf diesem Fenster ab, oder {offer}. Sie können auch einen Pfad in das Feld src unten eingeben.",
        "Suelte una carpeta de imágenes en cualquier parte de esta ventana, o {offer}. También puede escribir una ruta en el campo src de abajo.",
        "将图像文件夹拖放到本窗口的任意位置，或{offer}。您也可以在下方的 src 字段中输入路径。",
        "Solte uma pasta de imagens em qualquer lugar desta janela, ou {offer}. Você também pode digitar um caminho no campo src abaixo.",
        "छवियों का कोई फ़ोल्डर इस विंडो में कहीं भी छोड़ें, या {offer}। नीचे दिए src फ़ील्ड में पथ भी टाइप कर सकते हैं।",
        "이미지 폴더를 이 창 아무 곳에나 끌어다 놓거나, {offer}. 아래의 src 필드에 경로를 직접 입력할 수도 있습니다.",
        "Slepptu möppu með myndum hvar sem er í þessum glugga, eða {offer}. Þú getur líka slegið slóð inn í src-reitinn hér fyrir neðan.",
        "Déposez un dossier d’images n’importe où sur cette fenêtre, ou {offer}. Vous pouvez aussi saisir un chemin dans le champ src ci-dessous."),
    "use Demos → {demo} for a synthetic dataset": _row(
        "använd Demon → {demo} för en syntetisk datauppsättning",
        "verwenden Sie Demos → {demo} für einen synthetischen Datensatz",
        "utilice Demostraciones → {demo} para un conjunto de datos sintético",
        "使用“演示 → {demo}”获取合成数据集",
        "use Demonstrações → {demo} para um conjunto de dados sintético",
        "सिंथेटिक डेटासेट के लिए डेमो → {demo} का उपयोग करें",
        "합성 데이터셋이 필요하면 데모 → {demo}를 사용하세요",
        "notaðu Sýnishorn → {demo} fyrir tilbúið gagnasafn",
        "utilisez Démonstrations → {demo} pour un ensemble de données synthétique"),
    "pick a dataset from the Demos menu": _row(
        "välj en datauppsättning i menyn Demon",
        "wählen Sie einen Datensatz aus dem Demos-Menü",
        "elija un conjunto de datos del menú Demostraciones",
        "从“演示”菜单中选择一个数据集",
        "escolha um conjunto de dados no menu Demonstrações",
        "डेमो मेनू से कोई डेटासेट चुनें",
        "데모 메뉴에서 데이터셋을 선택하세요",
        "veldu gagnasafn úr Sýnishorn-valmyndinni",
        "choisissez un ensemble de données dans le menu Démonstrations"),
    "Remove {value}": _row(
        "Ta bort {value}",
        "{value} entfernen",
        "Eliminar {value}",
        "移除 {value}",
        "Remover {value}",
        "{value} हटाएँ",
        "{value} 제거",
        "Fjarlægja {value}",
        "Supprimer {value}"),
    "Computer Vision": _row(
        "Datorseende",
        "Maschinelles Sehen",
        "Visión por ordenador",
        "计算机视觉",
        "Visão computacional",
        "कंप्यूटर विज़न",
        "컴퓨터 비전",
        "Tölvusjón",
        "Vision par ordinateur"),
    "Machine Learning": _row(
        "Maskininlärning",
        "Maschinelles Lernen",
        "Aprendizaje automático",
        "机器学习",
        "Aprendizado de máquina",
        "मशीन लर्निंग",
        "머신러닝",
        "Vélnám",
        "Apprentissage automatique"),
    "Live settings": _row(
        "Liveinställningar",
        "Live-Einstellungen",
        "Ajustes en vivo",
        "实时设置",
        "Configurações ao vivo",
        "लाइव सेटिंग्स",
        "라이브 설정",
        "Beinar stillingar",
        "Paramètres en direct"),
    "News": _row(
        "Nyheter",
        "Neuigkeiten",
        "Novedades",
        "新闻",
        "Notícias",
        "समाचार",
        "새 소식",
        "Fréttir",
        "Actualités"),
    "+{n} more": _row(
        "+{n} till",
        "+{n} weitere",
        "+{n} más",
        "+{n} 项",
        "+{n} mais",
        "+{n} और",
        "+{n}개 더",
        "+{n} til viðbótar",
        "+{n} de plus"),
    "Merge the tables inside each database": _row(
        "Slå ihop tabellerna i varje databas",
        "Die Tabellen in jeder Datenbank zusammenführen",
        "Fusionar las tablas dentro de cada base de datos",
        "合并每个数据库内的表",
        "Mesclar as tabelas dentro de cada banco de dados",
        "हर डेटाबेस के भीतर तालिकाएँ मिलाएँ",
        "각 데이터베이스 안의 테이블 병합하기",
        "Sameina töflurnar í hverjum gagnagrunni",
        "Fusionner les tables dans chaque base de données"),
    "Merge the databases into one frame": _row(
        "Slå ihop databaserna till en dataram",
        "Die Datenbanken zu einem Frame zusammenführen",
        "Fusionar las bases de datos en un único marco",
        "将各数据库合并为一个数据框",
        "Mesclar os bancos de dados em um único quadro",
        "सभी डेटाबेस को एक फ़्रेम में मिलाएँ",
        "데이터베이스를 하나의 프레임으로 병합하기",
        "Sameina gagnagrunnana í einn gagnaramma",
        "Fusionner les bases de données en un seul tableau"),
    "Pick a column and regress on it": _row(
        "Välj en kolumn och regressera på den",
        "Eine Spalte wählen und darauf regressieren",
        "Elegir una columna y hacer la regresión sobre ella",
        "选择一列并对其进行回归",
        "Escolher uma coluna e fazer a regressão sobre ela",
        "कोई स्तंभ चुनें और उस पर प्रतिगमन चलाएँ",
        "열을 선택해 회귀 분석하기",
        "Veldu dálk og keyrðu aðhvarf á hann",
        "Choisir une colonne et effectuer la régression"),
    "Load a table, or press SQL to read the column names out of the database, to fill classes in from a column.": _row(
        "Läs in en tabell, eller tryck på SQL för att läsa ut kolumnnamnen ur databasen, för att fylla i klasser från en kolumn.",
        "Laden Sie eine Tabelle oder drücken Sie SQL, um die Spaltennamen aus der Datenbank zu lesen, damit die Klassen aus einer Spalte gefüllt werden können.",
        "Cargue una tabla, o pulse SQL para leer los nombres de las columnas de la base de datos, y así rellenar las clases a partir de una columna.",
        "加载一张表，或点击 SQL 从数据库中读取列名，以便根据某一列填充类别。",
        "Carregue uma tabela, ou pressione SQL para ler os nomes das colunas do banco de dados, para preencher as classes a partir de uma coluna.",
        "किसी स्तंभ से वर्ग भरने के लिए कोई तालिका लोड करें, या डेटाबेस से स्तंभों के नाम पढ़ने के लिए SQL दबाएँ।",
        "테이블을 불러오거나 SQL 버튼을 눌러 데이터베이스에서 열 이름을 읽어오면, 열의 값으로 클래스를 채울 수 있습니다.",
        "Hlaðu töflu, eða ýttu á SQL til að lesa dálkaheitin úr gagnagrunninum, svo hægt sé að fylla flokkana út frá dálki.",
        "Chargez un tableau, ou appuyez sur SQL pour lire les noms de colonnes dans la base de données, afin de remplir les classes à partir d’une colonne."),
    "give the class a name first": _row(
        "ge klassen ett namn först",
        "geben Sie der Klasse zuerst einen Namen",
        "primero dé un nombre a la clase",
        "请先为该类别命名",
        "dê um nome à classe primeiro",
        "पहले वर्ग को कोई नाम दें",
        "먼저 클래스에 이름을 지정하세요",
        "gefðu flokknum fyrst nafn",
        "donnez d’abord un nom à la classe"),
    "choose the column the value comes from": _row(
        "välj kolumnen som värdet kommer från",
        "wählen Sie die Spalte, aus der der Wert stammt",
        "elija la columna de la que procede el valor",
        "请选择该值取自的列",
        "escolha a coluna de onde vem o valor",
        "वह स्तंभ चुनें जिससे मान आता है",
        "값을 가져올 열을 선택하세요",
        "veldu dálkinn sem gildið kemur úr",
        "choisissez la colonne d’où vient la valeur"),
    "there is already a random-rest class; two classes both meaning 'everything else' have no boundary between them": _row(
        "det finns redan en slumpmässig restklass; två klasser som båda betyder 'allt annat' har ingen gräns mellan sig",
        "es gibt bereits eine Zufallsrest-Klasse; zwei Klassen, die beide 'alles andere' bedeuten, haben keine Grenze zwischen sich",
        "ya existe una clase de resto aleatorio; dos clases que significan 'todo lo demás' no tienen frontera entre ellas",
        "已经存在一个“随机剩余”类别；两个都表示“其余全部”的类别之间没有界限",
        "já existe uma classe de restante aleatório; duas classes que significam 'todo o resto' não têm fronteira entre si",
        "एक random-rest वर्ग पहले से मौजूद है; दो वर्ग जिनका अर्थ एक ही है — ‘बाकी सब’ — उनके बीच कोई सीमा नहीं रहती",
        "이미 무작위 나머지 클래스가 있습니다; 둘 다 '그 밖의 전부'를 뜻하는 클래스 사이에는 경계가 없습니다",
        "það er þegar til flokkur fyrir slembiafgang; tveir flokkar sem báðir merkja 'allt hitt' hafa engin mörk sín á milli",
        "il existe déjà une classe reste aléatoire ; deux classes signifiant toutes deux 'tout le reste' n’ont aucune frontière entre elles"),

    # The first-run tour's own chrome.
    "Step {n} / {total}": _row(
        "Steg {n} / {total}", "Schritt {n} / {total}", "Paso {n} / {total}",
        "第 {n} / {total} 步", "Passo {n} / {total}", "चरण {n} / {total}",
        "{total}단계 중 {n}단계", "Skref {n} / {total}",
        "Étape {n} / {total}"),
    "This quick 5-step tour will show you the home layout. Press Esc at any time to skip.": _row(
        "Den här korta rundturen i fem steg visar hemskärmens upplägg. Tryck Esc när som helst för att hoppa över den.",
        "Diese kurze Tour in fünf Schritten zeigt Ihnen den Aufbau der Startseite. Mit Esc können Sie sie jederzeit überspringen.",
        "Este breve recorrido de cinco pasos le muestra la disposición de la pantalla de inicio. Pulse Esc en cualquier momento para omitirlo.",
        "这个五步快速导览会介绍主页的布局。随时按 Esc 可以跳过。",
        "Este percurso rápido de cinco passos mostra a disposição do ecrã inicial. Prima Esc a qualquer momento para o ignorar.",
        "यह पाँच चरणों की छोटी झलक मुखपृष्ठ का ढाँचा दिखाती है। छोड़ने के लिए कभी भी Esc दबाएँ।",
        "이 다섯 단계짜리 짧은 둘러보기가 홈 화면 구성을 안내합니다. 언제든 Esc 를 누르면 건너뜁니다.",
        "Þessi stutta fimm skrefa kynning sýnir uppsetningu heimaskjásins. Ýttu á Esc hvenær sem er til að sleppa henni.",
        "Cette courte visite en cinq étapes présente la disposition de l’accueil. Appuyez sur Échap à tout moment pour la passer."),

    # HALF A TRANSLATION READS WORSE THAN NONE. Each of these
    # matched a TERM inside itself, so the word-by-word fallback
    # produced things like "Load the Mätning databases". An exact
    # row is what overrides a term match.
    "Console context: no new output": _row(
        "Konsolkontext: ingen ny utdata",
        "Konsole-Kontext: keine neue Ausgabe",
        "Contexto de Consola: sin salida nueva",
        "控制台 上下文：无新输出",
        "Contexto do Console: nenhuma saída nova",
        "कंसोल संदर्भ: कोई नया आउटपुट नहीं",
        "콘솔 컨텍스트: 새 출력 없음",
        "Stjórnborð-samhengi: ekkert nýtt úttak",
        "Contexte de console : aucune nouvelle sortie"),
    "Point {module} at some data": _row(
        "Rikta {module} mot data",
        "{module} auf Daten richten",
        "Indica a {module} dónde están los datos",
        "为 {module} 指定数据",
        "Aponte {module} para alguns dados",
        "{module} को कुछ डेटा दिखाएँ",
        "{module}에 사용할 데이터를 지정하세요",
        "{module} þarf gögn",
        "Pointez {module} vers des données"),
    "Load the measurement databases": _row(
        "Läs in mätdatabaserna",
        "Die Messdatenbanken laden",
        "Cargar las bases de datos de mediciones",
        "加载测量数据库",
        "Carregar os bancos de dados de medições",
        "मापन डेटाबेस लोड करें",
        "측정 데이터베이스 불러오기",
        "Hlaða mælingagagnagrunnunum",
        "Charger les bases de données de mesures"),
    # THE HEADINGS OF THE ADVANCED-SETTINGS TREE. Every one of them is a
    # phrase the word-by-word fallback half-translates -- "Objekt Filtration
    # (all Objekt)", "Bild Preprocessing (per Objekt)", "Avancerat
    # settings" -- so each needs an exact row, the same way "Intensity
    # Handling (all objects)" already has one in the external catalog. They
    # are looked up in the case written here and uppercased on the way to
    # the header, so a row spelled in capitals would never be found.
    "Advanced settings": _row(
        "Avancerade inställningar", "Erweiterte Einstellungen",
        "Configuración avanzada", "高级设置", "Configurações avançadas",
        "उन्नत सेटिंग्स", "고급 설정", "Ítarlegar stillingar",
        "Paramètres avancés"),
    "Image Preprocessing (per object)": _row(
        "Bildförbehandling (per objekt)", "Bildvorverarbeitung (pro Objekt)",
        "Preprocesamiento de imagen (por objeto)", "图像预处理（每个对象）",
        "Pré-processamento de imagem (por objeto)",
        "छवि पूर्व-प्रसंस्करण (प्रति ऑब्जेक्ट)", "이미지 전처리 (객체별)",
        "Forvinnsla myndar (á hvern hlut)",
        "Prétraitement de l’image (par objet)"),
    "Object Filtration (all objects)": _row(
        "Objektfiltrering (alla objekt)", "Objektfilterung (alle Objekte)",
        "Filtrado de objetos (todos los objetos)", "对象筛选（所有对象）",
        "Filtragem de objetos (todos os objetos)",
        "ऑब्जेक्ट फ़िल्टरिंग (सभी ऑब्जेक्ट)", "객체 필터링 (모든 객체)",
        "Hlutasíun (allir hlutir)", "Filtrage des objets (tous les objets)"),
    "Organelle Segmentation (advanced)": _row(
        "Organellsegmentering (avancerat)",
        "Organellen-Segmentierung (erweitert)",
        "Segmentación de orgánulos (avanzada)", "细胞器分割（高级）",
        "Segmentação de organelas (avançada)",
        "कोशिकांग विभाजन (उन्नत)", "소기관 분할 (고급)",
        "Hlutun frumulíffæra (ítarlegt)",
        "Segmentation des organites (avancée)"),
    # The per-object sub-headings of that tree. Cell, Nucleus and Pathogen
    # already resolve exactly; the four organelle slots did not, and a
    # heading composed by the word-by-word fallback is one word away from
    # reading half English the day a term row changes.
    "Organelle 1": _row(
        "Organell 1", "Organelle 1", "Orgánulo 1", "细胞器 1", "Organela 1", "कोशिकांग 1", "소기관 1", "Frumulíffæri 1", "Organite 1"),
    "Organelle 2": _row(
        "Organell 2", "Organelle 2", "Orgánulo 2", "细胞器 2", "Organela 2", "कोशिकांग 2", "소기관 2", "Frumulíffæri 2", "Organite 2"),
    "Organelle 3": _row(
        "Organell 3", "Organelle 3", "Orgánulo 3", "细胞器 3", "Organela 3", "कोशिकांग 3", "소기관 3", "Frumulíffæri 3", "Organite 3"),
    "Organelle 4": _row(
        "Organell 4", "Organelle 4", "Orgánulo 4", "细胞器 4", "Organela 4", "कोशिकांग 4", "소기관 4", "Frumulíffæri 4", "Organite 4"),
    # Captions a HANDLER writes, and the templates it writes them from.
    # A sentence composed first and translated afterwards matches nothing --
    # the finished line carries a count, a module name or a stage name that
    # no catalog can hold -- so the sentence is a row with a placeholder and
    # the value is substituted after the lookup.
    "Copied {count} lines": _row(
        "Kopierade {count} rader", "{count} Zeilen kopiert",
        "Se copiaron {count} líneas", "已复制 {count} 行",
        "{count} linhas copiadas", "{count} पंक्तियाँ कॉपी की गईं",
        "{count}줄을 복사했습니다", "Afritaði {count} línur",
        "{count} lignes copiées"),
    "Fetching {count} file(s)…": _row(
        "Hämtar {count} fil(er)…", "{count} Datei(en) werden geladen…",
        "Descargando {count} archivo(s)…", "正在获取 {count} 个文件…",
        "Baixando {count} arquivo(s)…", "{count} फ़ाइल(ें) प्राप्त की जा रही हैं…",
        "{count}개 파일을 가져오는 중…", "Sæki {count} skrá/skrár…",
        "Téléchargement de {count} fichier(s)…"),
    "Alpha and Beta": _row(
        "Alfa och Beta", "Alpha und Beta", "Alfa y Beta", "内测和公测",
        "Alfa e Beta", "अल्फा और बीटा", "알파 및 베타", "Alfa og Beta",
        "Alpha et Bêta"),
    "{stages} settings are hidden by Preferences. Enable them in "
    "Preferences → Feature maturity.": _row(
        "{stages}-inställningar döljs av Inställningar. Aktivera dem under "
        "Inställningar → Funktionsmognad.",
        "{stages}-Einstellungen werden von den Einstellungen ausgeblendet. "
        "Aktivieren Sie sie unter Einstellungen → Funktionsreife.",
        "Las preferencias ocultan los ajustes {stages}. Actívelos en "
        "Preferencias → Madurez de las funciones.",
        "{stages} 设置已被首选项隐藏。请在“首选项 → 功能成熟度”中启用。",
        "As preferências ocultam as configurações {stages}. Ative-as em "
        "Preferências → Maturidade dos recursos.",
        "{stages} सेटिंग्स प्राथमिकताओं द्वारा छिपाई गई हैं। उन्हें "
        "प्राथमिकताएँ → फ़ीचर परिपक्वता में सक्षम करें।",
        "{stages} 설정이 환경설정에 의해 숨겨져 있습니다. "
        "환경설정 → 기능 성숙도에서 사용하도록 설정하세요.",
        "{stages}-stillingar eru faldar af Stillingum. Kveiktu á þeim í "
        "Stillingar → Þroski eiginleika.",
        "Les préférences masquent les réglages {stages}. Activez-les dans "
        "Préférences → Maturité des fonctionnalités."),
    "The '{app}' app is interactive-only in this Qt build. Use the classic "
    "Tk GUI (`spacr`) for now.": _row(
        "Modulen ”{app}” är enbart interaktiv i det här Qt-bygget. Använd "
        "det klassiska Tk-gränssnittet (`spacr`) tills vidare.",
        "Das Modul „{app}“ ist in diesem Qt-Build nur interaktiv. Verwenden "
        "Sie vorerst die klassische Tk-Oberfläche (`spacr`).",
        "El módulo «{app}» solo es interactivo en esta compilación de Qt. "
        "Utilice por ahora la interfaz clásica de Tk (`spacr`).",
        "在此 Qt 版本中，“{app}”模块仅支持交互操作。请暂时使用经典的 Tk 界面（`spacr`）。",
        "O módulo “{app}” é apenas interativo nesta compilação Qt. Use por "
        "enquanto a interface clássica Tk (`spacr`).",
        "इस Qt बिल्ड में ‘{app}’ मॉड्यूल केवल इंटरैक्टिव है। फ़िलहाल क्लासिक "
        "Tk इंटरफ़ेस (`spacr`) का उपयोग करें।",
        "이 Qt 빌드에서 '{app}' 모듈은 대화형으로만 동작합니다. 당분간 기존 "
        "Tk 인터페이스(`spacr`)를 사용하세요.",
        "Einingin „{app}“ er aðeins gagnvirk í þessari Qt-útgáfu. Notaðu "
        "klassíska Tk-viðmótið (`spacr`) í bili.",
        "Le module « {app} » est uniquement interactif dans cette version "
        "Qt. Utilisez pour l’instant l’interface Tk classique (`spacr`)."),
    "All files": _row(
        "Alla filer", "Alle Dateien", "Todos los archivos", "所有文件",
        "Todos os arquivos", "सभी फ़ाइलें", "모든 파일", "Allar skrár",
        "Tous les fichiers"),
    # AN EXACT ROW IS WHAT BEATS THE WORD-BY-WORD FALLBACK. Without one,
    # `_term_translation` finds "image" in the middle of this caption and
    # leaves "Choose Bild…" on the live preview's button -- half English,
    # half German, and the same shape in every other language.
    "Choose image…": _row(
        "Välj bild…", "Bild auswählen…", "Elegir imagen…", "选择图像…",
        "Escolher imagem…", "छवि चुनें…", "이미지 선택…", "Velja mynd…",
        "Choisir une image…"),
    "No source selected — click Open source…": _row(
        "Ingen källa vald — klicka på Öppna källa…",
        "Keine Quelle ausgewählt — klicken Sie auf Quelle öffnen…",
        "No se ha seleccionado ninguna fuente — haga clic en Abrir fuente…",
        "未选择来源 — 请点击“打开来源”…",
        "Nenhuma origem selecionada — clique em Abrir origem…",
        "कोई स्रोत नहीं चुना गया — ‘स्रोत खोलें…’ पर क्लिक करें",
        "선택된 소스가 없습니다 — 소스 열기…를 클릭하세요",
        "Enginn uppruni valinn — smelltu á Opna uppruna…",
        "Aucune source sélectionnée — cliquez sur Ouvrir la source…"),

    # ---- THE LIVE PREVIEW'S VALUE-CARRYING DROPDOWNS ------------------
    # Every entry below is BOTH a caption the user reads and a value the
    # panel matches on, so the value lives in the entry's item data and
    # only the caption is translated -- see :func:`set_translatable_items`.
    # An exact row is what makes the caption right. Left to the word-by-word
    # fallback, "auto" came back as the vehicle ("자동차", "汽车", "कार"),
    # "Overlay" as "Surprise" in French, and "All channels" as the
    # half-English "All Kanaler".
    #
    # The outline-colour dropdown.
    "auto": _row(
        "automatisk", "automatisch", "automático", "自动", "automático",
        "स्वचालित", "자동", "sjálfvirkt", "automatique"),
    "color (random)": _row(
        "färg (slumpmässig)", "Farbe (zufällig)", "color (aleatorio)",
        "颜色（随机）", "cor (aleatória)", "रंग (यादृच्छिक)", "색상(무작위)",
        "litur (slembinn)", "couleur (aléatoire)"),
    "green": _row(
        "grön", "Grün", "verde", "绿色", "verde", "हरा", "녹색", "grænn",
        "vert"),
    "magenta": _row(
        "magenta", "Magenta", "magenta", "品红", "magenta", "मैजेंटा",
        "자홍색", "magenta", "magenta"),
    "yellow": _row(
        "gul", "Gelb", "amarillo", "黄色", "amarelo", "पीला", "노란색",
        "gulur", "jaune"),
    "cyan": _row(
        "cyan", "Cyan", "cian", "青色", "ciano", "सियान", "청록색",
        "blágrænn", "cyan"),
    "white": _row(
        "vit", "Weiß", "blanco", "白色", "branco", "सफ़ेद", "흰색", "hvítur",
        "blanc"),
    "red": _row(
        "röd", "Rot", "rojo", "红色", "vermelho", "लाल", "빨간색", "rauður",
        "rouge"),
    # What the right-hand canvas shows.
    "View:": _row(
        "Vy:", "Ansicht:", "Vista:", "视图：", "Vista:", "दृश्य:", "보기:",
        "Sýn:", "Affichage :"),
    "Overlay": _row(
        "Överlägg", "Überlagerung", "Superposición", "叠加", "Sobreposição",
        "ओवरले", "오버레이", "Yfirlag", "Superposition"),
    "Masks": _row(
        "Masker", "Masken", "Máscaras", "掩膜", "Máscaras", "मास्क", "마스크",
        "Grímur", "Masques"),
    "Flows": _row(
        "Flöden", "Flüsse", "Flujos", "流场", "Fluxos", "प्रवाह", "흐름",
        "Flæði", "Flux"),
    # The channel view control. "Ch 3" names a plane and stays as written --
    # a word touching a digit is part of an identifier, not prose.
    "All channels": _row(
        "Alla kanaler", "Alle Kanäle", "Todos los canales", "所有通道",
        "Todos os canais", "सभी चैनल", "모든 채널", "Allar rásir",
        "Tous les canaux"),
    # THE SEGMENTATION COMPARTMENTS, whose English spelling is the key the
    # worker and every `{object}_…` setting are written with. The bulk
    # catalog had read them as everyday words -- "cell" as a spreadsheet
    # celda, "nucleus" as an atomic nucleus (परमाणु), "organelle" as an
    # organ, "pathogen" as pathology -- so these are the biological senses.
    "cell": _row(
        "cell", "Zelle", "célula", "细胞", "célula", "कोशिका", "세포",
        "fruma", "cellule"),
    "nucleus": _row(
        "kärna", "Zellkern", "núcleo", "细胞核", "núcleo", "केंद्रक", "핵",
        "kjarni", "noyau"),
    "pathogen": _row(
        "patogen", "Pathogen", "patógeno", "病原体", "patógeno", "रोगजनक",
        "병원체", "sýkill", "pathogène"),
    "organelle": _row(
        "organell", "Organell", "orgánulo", "细胞器", "organela", "कोशिकांग",
        "세포소기관", "frumulíffæri", "organite"),
    "cell + nucleus": _row(
        "cell + kärna", "Zelle + Zellkern", "célula + núcleo", "细胞 + 细胞核",
        "célula + núcleo", "कोशिका + केंद्रक", "세포 + 핵", "fruma + kjarni",
        "cellule + noyau"),
    # How a compartment's intensity threshold is computed. Statistics, not
    # prose: the fallback had offered "meaning" (意思, "Að segja") for the
    # average and "a hundred percent" (百分之百) for the percentile.
    "mean": _row(
        "medelvärde", "Mittelwert", "media", "均值", "média", "माध्य", "평균",
        "meðaltal", "moyenne"),
    "percentile": _row(
        "percentil", "Perzentil", "percentil", "百分位", "percentil",
        "प्रतिशतक", "백분위수", "hundraðsmark", "centile"),
}


# Common settings words provide broad, conservative coverage for short labels
# and section headings not yet promoted to an exact phrase above. Technical
# identifiers (UMAP, Cellpose, XGBoost, CUDA, SQL, file suffixes) are retained.
_TERM_ROWS: Dict[str, tuple[str, ...]] = {
    "Input": _row("Indata", "Eingabe", "Entrada", "输入", "Entrada", "इनपुट", "입력", "Inntak", "Entrée"),
    "Output": _row("Utdata", "Ausgabe", "Salida", "输出", "Saída", "आउटपुट", "출력", "Úttak", "Sortie"),
    "Data": _row("Data", "Daten", "Datos", "数据", "Dados", "डेटा", "데이터", "Gögn", "Données"),
    "Model": _row("Modell", "Modell", "Modelo", "模型", "Modelo", "मॉडल", "모델", "Líkan", "Modèle"),
    "Models": _row("Modeller", "Modelle", "Modelos", "模型", "Modelos", "मॉडल", "모델", "Líkön", "Modèles"),
    "Training": _row("Träning", "Training", "Entrenamiento", "训练", "Treinamento", "प्रशिक्षण", "학습", "Þjálfun", "Entraînement"),
    "Validation": _row("Validering", "Validierung", "Validación", "验证", "Validação", "सत्यापन", "검증", "Staðfesting", "Validation"),
    "Runtime": _row("Körtid", "Laufzeit", "Tiempo de ejecución", "运行时", "Tempo de execução", "रनटाइम", "실행 시간", "Keyrslutími", "Temps d’exécution"),
    "Reliability": _row("Tillförlitlighet", "Zuverlässigkeit", "Fiabilidad", "可靠性", "Confiabilidade", "विश्वसनीयता", "신뢰성", "Áreiðanleiki", "Fiabilité"),
    "Image": _row("Bild", "Bild", "Imagen", "图像", "Imagem", "छवि", "이미지", "Mynd", "Image"),
    "Images": _row("Bilder", "Bilder", "Imágenes", "图像", "Imagens", "छवियाँ", "이미지", "Myndir", "Images"),
    "Cell": _row("Cell", "Zelle", "Célula", "细胞", "Célula", "कोशिका", "세포", "Fruma", "Cellule"),
    "Cells": _row("Celler", "Zellen", "Células", "细胞", "Células", "कोशिकाएँ", "세포", "Frumur", "Cellules"),
    "Nucleus": _row("Cellkärna", "Zellkern", "Núcleo", "细胞核", "Núcleo", "नाभिक", "핵", "Kjarni", "Noyau"),
    "Pathogen": _row("Patogen", "Pathogen", "Patógeno", "病原体", "Patógeno", "रोगजनक", "병원체", "Sýkill", "Pathogène"),
    "Organelle": _row("Organell", "Organelle", "Orgánulo", "细胞器", "Organela", "कोशिकांग", "소기관", "Frumulíffæri", "Organite"),
    "Segmentation": _row("Segmentering", "Segmentierung", "Segmentación", "分割", "Segmentação", "छवि विभाजन", "분할", "Hlutun", "Segmentation"),
    "Measurement": _row("Mätning", "Messung", "Medición", "测量", "Medição", "मापन", "측정", "Mæling", "Mesure"),
    "Measurements": _row("Mätningar", "Messungen", "Mediciones", "测量", "Medições", "मापन", "측정", "Mælingar", "Mesures"),
    "Features": _row("Egenskaper", "Merkmale", "Características", "特征", "Características", "विशेषताएँ", "특징", "Eiginleikar", "Caractéristiques"),
    "Filtering": _row("Filtrering", "Filterung", "Filtrado", "筛选", "Filtragem", "फ़िल्टरिंग", "필터링", "Síun", "Filtrage"),
    "Object": _row("Objekt", "Objekt", "Objeto", "对象", "Objeto", "ऑब्जेक्ट", "객체", "Hlutur", "Objet"),
    "Objects": _row("Objekt", "Objekte", "Objetos", "对象", "Objetos", "ऑब्जेक्ट", "객체", "Hlutir", "Objets"),
    "Channel": _row("Kanal", "Kanal", "Canal", "通道", "Canal", "चैनल", "채널", "Rás", "Canal"),
    "Channels": _row("Kanaler", "Kanäle", "Canales", "通道", "Canais", "चैनल", "채널", "Rásir", "Canaux"),
    "Source": _row("Källa", "Quelle", "Origen", "来源", "Origem", "स्रोत", "소스", "Uppruni", "Source"),
    "Metadata": _row("Metadata", "Metadaten", "Metadatos", "元数据", "Metadados", "मेटाडेटा", "메타데이터", "Lýsigögn", "Métadonnées"),
    "Workflow": _row("Arbetsflöde", "Arbeitsablauf", "Flujo de trabajo", "工作流", "Fluxo de trabalho", "कार्यप्रवाह", "워크플로", "Vinnuflæði", "Flux de travail"),
    "Test": _row("Test", "Test", "Prueba", "测试", "Teste", "परीक्षण", "테스트", "Próf", "Test"),
    "Display": _row("Visning", "Anzeige", "Visualización", "显示", "Exibição", "प्रदर्शन", "표시", "Birting", "Affichage"),
    "Plot": _row("Diagram", "Diagramm", "Gráfico", "绘图", "Gráfico", "प्लॉट", "플롯", "Graf", "Graphique"),
    "Plots": _row("Diagram", "Diagramme", "Gráficos", "绘图", "Gráficos", "प्लॉट", "플롯", "Gröf", "Graphiques"),
    "Cluster": _row("Kluster", "Cluster", "Clúster", "聚类", "Cluster", "क्लस्टर", "클러스터", "Klasi", "Cluster"),
    "Clustering": _row("Klustring", "Clustering", "Agrupamiento", "聚类", "Agrupamento", "क्लस्टरिंग", "군집화", "Klösun", "Partitionnement"),
    "Advanced": _row("Avancerat", "Erweitert", "Avanzado", "高级", "Avançado", "उन्नत", "고급", "Ítarlegt", "Avancé"),
    "General": _row("Allmänt", "Allgemein", "General", "常规", "Geral", "सामान्य", "일반", "Almennt", "Général"),
    "Paths": _row("Sökvägar", "Pfade", "Rutas", "路径", "Caminhos", "पथ", "경로", "Slóðir", "Chemins"),
    "Controls": _row("Kontroller", "Kontrollen", "Controles", "控件", "Controles", "नियंत्रण", "컨트롤", "Stýringar", "Contrôles"),
    "Plate": _row("Platta", "Platte", "Placa", "孔板", "Placa", "प्लेट", "플레이트", "Plata", "Plaque"),
    "Plates": _row("Plattor", "Platten", "Placas", "板", "Placas", "प्लेट", "플레이트", "Plötur", "Plaques"),
    "Batch": _row("Batch", "Batch", "Lote", "批次", "Lote", "बैच", "배치", "Lota", "Lot"),
    "Results": _row("Resultat", "Ergebnisse", "Resultados", "结果", "Resultados", "परिणाम", "결과", "Niðurstöður", "Résultats"),
    "Quality": _row("Kvalitet", "Qualität", "Calidad", "质量", "Qualidade", "गुणवत्ता", "품질", "Gæði", "Qualité"),
    "Control": _row("Kontroll", "Kontrolle", "Control", "对照", "Controle", "नियंत्रण", "대조군", "Viðmið", "Contrôle"),
    "Tracking": _row("Spårning", "Verfolgung", "Seguimiento", "跟踪", "Rastreamento", "ट्रैकिंग", "추적", "Rakning", "Suivi"),
    "Time": _row("Tid", "Zeit", "Tiempo", "时间", "Tempo", "समय", "시간", "Tími", "Temps"),
    "Calibration": _row("Kalibrering", "Kalibrierung", "Calibración", "校准", "Calibração", "अंशांकन", "보정", "Kvörðun", "Étalonnage"),
    "Summary": _row("Sammanfattning", "Zusammenfassung", "Resumen", "摘要", "Resumo", "सारांश", "요약", "Samantekt", "Résumé"),
    "Confusion matrix": _row("Förväxlingsmatris", "Konfusionsmatrix", "Matriz de confusión", "混淆矩阵", "Matriz de confusão", "भ्रम मैट्रिक्स", "혼동 행렬", "Ruglingsfylki", "Matrice de confusion"),
    "Per-plate metrics": _row("Mätvärden per platta", "Kennzahlen pro Platte", "Métricas por placa", "每板指标", "Métricas por placa", "प्रति-प्लेट मेट्रिक्स", "플레이트별 지표", "Mæligildi á plötu", "Métriques par plaque"),
    "Predictions": _row("Förutsägelser", "Vorhersagen", "Predicciones", "预测", "Previsões", "पूर्वानुमान", "예측", "Spár", "Prédictions"),
    "Leakage audit": _row("Dataläckagegranskning", "Datenleckprüfung", "Auditoría de fuga de datos", "数据泄漏审计", "Auditoria de vazamento de dados", "डेटा लीकेज ऑडिट", "데이터 누수 감사", "Gagnalekagreining", "Audit des fuites de données"),
    "Results folder": _row("Resultatmapp", "Ergebnisordner", "Carpeta de resultados", "结果文件夹", "Pasta de resultados", "परिणाम फ़ोल्डर", "결과 폴더", "Niðurstöðumappa", "Dossier de résultats"),
    "Evaluation run": _row("Utvärderingskörning", "Auswertungslauf", "Ejecución de evaluación", "评估运行", "Execução de avaliação", "मूल्यांकन रन", "평가 실행", "Matskeyrsla", "Exécution d’évaluation"),
    "Scan": _row("Sök", "Scannen", "Escanear", "扫描", "Escanear", "स्कैन करें", "스캔", "Skanna", "Scanner"),
    "Open folder": _row("Öppna mapp", "Ordner öffnen", "Abrir carpeta", "打开文件夹", "Abrir pasta", "फ़ोल्डर खोलें", "폴더 열기", "Opna möppu", "Ouvrir le dossier"),
    "Classification": _row("Klassificering", "Klassifizierung", "Clasificación", "分类", "Classificação", "वर्गीकरण", "분류", "Flokkun", "Classification"),
    "Annotation": _row("Annotering", "Annotation", "Anotación", "标注", "Anotação", "एनोटेशन", "어노테이션", "Merking", "Annotation"),
    "Classes": _row("Klasser", "Klassen", "Clases", "类别", "Classes", "वर्ग", "클래스", "Flokkar", "Classes"),
    "Optimization": _row("Optimering", "Optimierung", "Optimización", "优化", "Otimização", "अनुकूलन", "최적화", "Bestun", "Optimisation"),
    "stability repeats": _row("stabilitetsupprepningar", "Stabilitätswiederholungen", "repeticiones de estabilidad", "稳定性重复次数", "repetições de estabilidade", "स्थिरता पुनरावृत्तियाँ", "안정성 반복", "stöðugleikaendurtekningar", "répétitions de stabilité"),
    "neighborhood weight": _row("grannskapsvikt", "Nachbarschaftsgewicht", "peso de vecindad", "邻域权重", "peso da vizinhança", "पड़ोस भार", "이웃 가중치", "vægi nágrennis", "poids du voisinage"),
    "stability weight": _row("stabilitetsvikt", "Stabilitätsgewicht", "peso de estabilidad", "稳定性权重", "peso da estabilidade", "स्थिरता भार", "안정성 가중치", "vægi stöðugleika", "poids de stabilité"),
    "cluster weight": _row("klustervikt", "Clustergewicht", "peso de clúster", "聚类权重", "peso do cluster", "क्लस्टर भार", "클러스터 가중치", "vægi klasa", "poids des clusters"),
    "Loss": _row("Förlust", "Verlust", "Pérdida", "损失", "Perda", "हानि", "손실", "Tap", "Perte"),
    "Inference": _row("Inferens", "Inferenz", "Inferencia", "推理", "Inferência", "इंफरेंस", "추론", "Ályktun", "Inférence"),
    "Storage": _row("Lagring", "Speicher", "Almacenamiento", "存储", "Armazenamento", "भंडारण", "저장", "Geymsla", "Stockage"),
}


def _build_catalogs(
    rows: Mapping[str, tuple[str, ...]],
) -> Dict[str, Dict[str, str]]:
    """Convert parallel translation rows to language-keyed catalogs."""
    catalogs = {code: {} for code in _TRANSLATED_CODES}
    for source, values in rows.items():
        for code, value in zip(_TRANSLATED_CODES, values):
            catalogs[code][source] = value
    return catalogs


CATALOGS = _build_catalogs(_ROWS)
TERM_CATALOGS = _build_catalogs(_TERM_ROWS)


def add_translation(source: str, values: Iterable[str]) -> bool:
    """Add one parallel translation row after the catalogs are built.

    The half of the app-registration seam that lands here. Every app name
    and every section name has to appear in every one of the nine
    catalogs — ``tests/qt/test_i18n.py`` walks ``spacr.qt.app.APPS`` and
    asserts it — so an app registered from its own module used to need
    nine hand-edits in this file. It now gives its translations once, to
    :func:`spacr.qt.app.register_app`, and they arrive here.

    :data:`_ROWS` and :data:`CATALOGS` are both updated IN PLACE.
    Rebinding either would strand every module that imported the name,
    and ``retranslate_widget_tree`` holds one for the life of a window.

    :param source: the English string, exactly as the UI spells it.
    :param values: its translations, in :data:`LANGUAGES` order after
        English (sv, de, es, zh_CN, pt, hi, ko, is, fr).
    :returns: ``True`` if the row was added, ``False`` if the same ``source``
        and translations were already catalogued.
    :raises ValueError: if ``values`` is not one string per language, or
        any of them is blank. A missing translation fails here, where the
        app name is in the message, rather than as a blank sidebar row in
        Korean.
    """
    source = str(source)
    row = _row(*[str(value) for value in values])
    if not all(value.strip() for value in row):
        raise ValueError(f"translation row for {source!r} has a blank entry")
    if source in _ROWS:
        if _ROWS[source] != row:
            raise ValueError(
                f"translation row for {source!r} conflicts with the "
                "catalogued row"
            )
        return False
    _ROWS[source] = row
    for code, value in zip(_TRANSLATED_CODES, row):
        CATALOGS[code][source] = value
    return True


def _absorb_registered_app_names() -> None:
    """Catalogue the display name of every app registered so far.

    The PULL half of the seam: :func:`spacr.qt.app.register_app` pushes a
    new app's name into the catalogs above when this module is already
    imported, and this picks up the apps that registered before it was.
    Between them the order of the two imports stops mattering.

    Read out of :data:`sys.modules` rather than imported: ``spacr.qt.app``
    imports the widget package, which imports this module, so importing it
    from here would be a cycle. An unregistered process simply finds
    nothing.
    """
    app = sys.modules.get("spacr.qt.app")
    # `getattr(..., None)`, not a bare attribute read: `spacr.qt.app`
    # imports the widget package (which imports this module) at its line
    # 41, so it is present in `sys.modules` and only PARTIALLY built
    # while this runs. There is nothing registered yet at that point --
    # the push half delivers it later.
    pull = getattr(app, "registered_metadata", None) if app else None
    if pull is None:
        return
    metadata = getattr(app, "APP_META", {})
    for key, values in pull("translations").items():
        name = (metadata.get(key) or {}).get("name") or key
        try:
            add_translation(name, values)
        except ValueError:
            # A bad row costs that app its translations, not the app.
            pass


_absorb_registered_app_names()


def normalize_language(code: object) -> str:
    """Return a supported language code, falling back to English.

    Locale-shaped values such as ``pt_BR`` and ``zh-CN`` resolve to their
    bundled base/catalog variants. This also makes a manually edited
    ``QSettings`` file harmless.
    """
    raw = str(code or "").strip().replace("-", "_")
    if raw in LANGUAGE_BY_CODE:
        return raw
    lower = raw.lower()
    exact = {item.lower(): item for item in VALID_LANGUAGE_CODES}
    if lower in exact:
        return exact[lower]
    base = lower.split("_", 1)[0]
    if base == "zh":
        return "zh_CN"
    if base in LANGUAGE_BY_CODE:
        return base
    return DEFAULT_LANGUAGE


def current_language() -> str:
    """Return the active persisted language without creating an import cycle."""
    env = os.environ.get(ENV_LANGUAGE)
    if env:
        return normalize_language(env)
    try:
        from .preferences import get_language
        return normalize_language(get_language())
    except Exception:
        return DEFAULT_LANGUAGE


def language_choices() -> tuple[tuple[str, str], ...]:
    """Return ``(display label, code)`` choices for Preferences."""
    return tuple((language.display_name, language.code)
                 for language in LANGUAGES)


def _exact_translation(source: str, language: str) -> Optional[str]:
    """Return an exact catalog translation, including case/mnemonic variants."""
    stripped = source.strip()
    if stripped != source and stripped:
        translated = _exact_translation(stripped, language)
        if translated is not None:
            leading = source[:len(source) - len(source.lstrip())]
            trailing = source[len(source.rstrip()):]
            return f"{leading}{translated}{trailing}"

    catalog = CATALOGS.get(language, {})
    if source in catalog:
        return catalog[source]
    # TERM_CATALOGS also contains reviewed multi-word scientific phrases.
    # The word-by-word fallback below cannot ever match those dictionary keys,
    # so take an exact phrase before decomposing a short label into tokens.
    term_catalog = TERM_CATALOGS.get(language, {})
    if source in term_catalog:
        return term_catalog[source]
    try:
        from .i18n_catalogs import ui_text
        translated = ui_text(source, language)
        if translated is not None:
            return translated
    except (ImportError, AttributeError):
        # External catalogs add coverage; their absence must not make the
        # compact core catalog unavailable.
        pass
    try:
        from spacr.plugins import discover_plugins
        for plugin in discover_plugins():
            translated = plugin.translations.get(language, {}).get(source)
            if translated:
                return translated
    except Exception:
        # A plugin translation is optional metadata; core localization must
        # remain available if discovery fails.
        pass

    # Qt uses '&' for keyboard mnemonics and '&&' for a literal ampersand.
    literal = source.replace("&&", "&")
    mnemonic = literal.startswith("&") and not literal.startswith("&&")
    lookup = literal[1:] if mnemonic else literal
    if lookup in catalog:
        translated = catalog[lookup]
        if "&&" in source:
            translated = translated.replace("&", "&&")
        return f"&{translated}" if mnemonic else translated
    try:
        from .i18n_catalogs import ui_text
        translated = ui_text(lookup, language)
        if translated is not None:
            if "&&" in source:
                translated = translated.replace("&", "&&")
            return f"&{translated}" if mnemonic else translated
    except (ImportError, AttributeError):
        pass

    # Section headers are often uppercased before they reach QLabel.
    if source.isupper():
        for english, translated in catalog.items():
            if english.upper() == source:
                return translated.upper()
    return None


_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")


def _term_translation(source: str, language: str) -> Optional[str]:
    """Translate known words in a short static label conservatively."""
    if len(source) > 80 or "\n" in source or source.lstrip().startswith("<"):
        return None
    if "/" in source or "\\" in source or "://" in source:
        return None
    terms = TERM_CATALOGS.get(language, {})
    if not terms:
        return None
    lookup = {key.casefold(): value for key, value in terms.items()}
    changed = False

    def _inside_an_identifier(text: str, start: int, end: int) -> bool:
        """Whether the word at ``start:end`` is part of a code name.

        A word touching ``_`` or a digit is a piece of an identifier --
        ``cell_area``, ``channel_1``, ``image_path`` -- and not a word of
        prose. Translating it rewrites a column name, a settings key or an
        SQL example into something that no longer names anything: the
        database browser's own example predicate,
        ``cell_area > 1000``, was shown to a Swedish user as
        ``Cell_area > 1000``, and the search hint offered ``'Kanal_1'`` for
        a column called ``channel_1``.
        """
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        return any(char == "_" or char.isdigit()
                   for char in (before, after) if char)

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        word = match.group(0)
        if _inside_an_identifier(source, match.start(), match.end()):
            return word
        translated = lookup.get(word.casefold())
        if translated is None:
            return word
        changed = True
        if word.isupper() and language not in {"zh_CN", "hi", "ko"}:
            return translated.upper()
        return translated

    result = _WORD_RE.sub(replace, source)
    return result if changed else None


def tr(text: object, language: Optional[str] = None, **values: object) -> str:
    """Translate one English UI string.

    Missing entries intentionally remain English. Keyword values are applied
    with ``str.format`` *after* translation, allowing catalogs to reorder
    placeholders safely.
    """
    source = str(text)
    code = normalize_language(language or current_language())
    translated = source
    if code != DEFAULT_LANGUAGE:
        translated = (_exact_translation(source, code)
                      or _term_translation(source, code)
                      or source)
    if values:
        try:
            return translated.format(**values)
        except (KeyError, IndexError, ValueError):
            return translated
    return translated


def has_translation(text: object, language: Optional[str] = None) -> bool:
    """Return whether ``text`` has an exact or conservative term translation."""
    source = str(text)
    code = normalize_language(language or current_language())
    if code == DEFAULT_LANGUAGE:
        return source in _ROWS or source in _TERM_ROWS
    return (_exact_translation(source, code) is not None
            or _term_translation(source, code) is not None)


def catalog_coverage(
    sources: Iterable[str], language: Optional[str] = None,
) -> tuple[int, int]:
    """Return ``(translated, total)`` for an iterable of source strings."""
    items = tuple(dict.fromkeys(str(source) for source in sources))
    code = normalize_language(language or current_language())
    return sum(has_translation(item, code) for item in items), len(items)


def _translate_qt_text(obj, getter_name: str, setter_name: str,
                       property_name: str, language: str) -> None:
    """Translate one Qt string property while retaining its English source.

    If application code changes a label after an earlier translation pass, it
    is dynamic data rather than the cached static caption.  Preserve that new
    value and automatically opt the label out instead of restoring stale UI
    text over a path, progress value or result.
    """
    getter = getattr(obj, getter_name, None)
    setter = getattr(obj, setter_name, None)
    if not callable(getter) or not callable(setter):
        return
    try:
        current = str(getter() or "")
        source = obj.property(property_name)
        rendered_property = f"{property_name}_last_rendered"
        last_rendered = obj.property(rendered_property)
        if source is None:
            source = current
            if not source:
                return
            obj.setProperty(property_name, str(source))
        elif last_rendered is not None and current != str(last_rendered):
            # A setter outside the translator replaced the rendered value.
            # QLabel/QAbstractButton contents can carry paths, metrics and
            # provider output, so preserve them byte-for-byte until callers
            # explicitly opt in with set_translatable_text().
            obj.setProperty(property_name, current)
            if property_name == "_spacr_i18n_text":
                obj.setProperty("i18nSkipText", True)
                obj.setProperty(rendered_property, current)
                return
            source = current
        rendered = tr(source, language)
        setter(rendered)
        obj.setProperty(rendered_property, rendered)
    except (AttributeError, RuntimeError, TypeError):
        # A deferred-delete Qt wrapper may remain in findChildren briefly.
        return


def set_translatable_text(
    widget,
    source: str,
    language: Optional[str] = None,
    **values: object,
) -> None:
    """Set dynamic UI text while retaining its canonical template and values.

    This is for application chrome such as ``Connecting to {provider}…``.
    User text, AI replies, worker output and scientific results must not use
    this helper because they intentionally remain untouched by localization.
    """
    widget.setProperty("_spacr_i18n_text_template", str(source))
    # Python attributes reliably retain arbitrary values across all supported
    # PySide versions; QVariant conversion of a dict is less consistent.
    widget._spacr_i18n_text_values = dict(values)
    widget.setText(tr(source, language, **values))


def set_translatable_items(
    combo,
    sources: Iterable[str],
    values: Optional[Iterable[object]] = None,
    language: Optional[str] = None,
) -> None:
    """Fill a dropdown with translated captions over untranslatable values.

    A combo box whose entries a handler reads back with ``currentText()``
    cannot be translated: the caption moves and every comparison misses.
    That is why the live preview's dropdowns were marked untranslatable
    outright, and why the ones that were not marked went wrong quietly --
    the segmentation object box handed ``cellen`` to a worker that only
    knows ``cell``, and the threshold method wrote ``medelvärde`` into a
    settings key that only accepts ``mean``.

    Each entry here carries what the code matches on in its item DATA, so
    ``currentData()`` answers the same English value whatever the caption
    reads. The English sources are recorded on the widget, so the ordinary
    language pass re-renders the captions on every later change instead of
    freezing the language the dropdown happened to be built in.

    The selected entry is kept by its value, never by its caption, and
    signals stay blocked while the entries are replaced.

    :param combo: the dropdown to fill; its existing entries are replaced.
    :param sources: the English captions, in order.
    :param values: what each entry means to the code, in the same order;
        defaults to ``sources`` itself.
    :param language: language to render in; the current one by default.
    :raises ValueError: if ``values`` is not one value per caption.
    """
    captions = [str(source) for source in sources]
    data = list(captions) if values is None else list(values)
    if len(data) != len(captions):
        raise ValueError(
            f"{len(captions)} captions but {len(data)} values")
    code = normalize_language(language or current_language())
    previous = combo.currentData()
    blocked = combo.blockSignals(True)
    try:
        combo.clear()
        for caption, value in zip(captions, data):
            combo.addItem(tr(caption, code), value)
        if previous is not None:
            index = combo.findData(previous)
            if index >= 0:
                combo.setCurrentIndex(index)
    finally:
        combo.blockSignals(blocked)
    combo._spacr_i18n_item_sources = list(captions)
    # An explicit False, because the property may already be True from the
    # widget's class -- FlatComboBox marks every entry untranslatable,
    # which is right for the file names it usually lists and wrong for a
    # caption that now keeps its value somewhere else.
    combo.setProperty("i18nSkipItems", False)


def _refresh_dynamic_text(widget, language: str) -> bool:
    """Retranslate a template set by :func:`set_translatable_text`."""
    try:
        source = widget.property("_spacr_i18n_text_template")
        if source is None:
            return False
        values = getattr(widget, "_spacr_i18n_text_values", {})
        widget.setText(tr(str(source), language, **dict(values or {})))
        return True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _refresh_module_help(obj, language: str) -> None:
    """Regenerate semantic module help without word-level prose mixing."""
    try:
        app_key = obj.property("moduleAppKey")
        name_source = obj.property("moduleNameSource")
        summary_source = obj.property("moduleSummarySource")
    except (AttributeError, RuntimeError):
        return
    if not app_key or not summary_source:
        return

    from .i18n_module_summaries import module_summary

    name = tr(str(name_source or app_key), language)
    summary = module_summary(str(app_key), str(summary_source), language)
    style = str(obj.property("moduleTooltipStyle") or "")
    if style == "sidebar":
        obj.setToolTip(f"{name} — {summary}")
        obj.setAccessibleName(name)
        obj.setAccessibleDescription(summary)
    elif style == "tile":
        stage_source = str(obj.property("moduleStageSource") or "")
        stage = tr(stage_source, language) if stage_source else ""
        suffix = f" ({stage.lower()})" if stage else ""
        obj.setToolTip(f"{name}{suffix} — {summary}")
        obj.setAccessibleName(name)
        obj.setAccessibleDescription(
            f"{stage} — {summary}" if stage else summary)
    elif hasattr(obj, "setStatusTip"):
        # QAction module entries have a status tip but no QWidget tooltip.
        obj.setStatusTip(summary)


def _follow_qt_own_catalogs(code: str) -> None:
    """Load Qt's own catalog for ``code`` when it is not the one loaded.

    ``&Copy``, ``Select All`` and ``Close Tab`` are Qt's strings rather than
    spaCR's, and they come from ``qtbase_<lang>.qm`` rather than from any
    catalog here. That file is loaded once at startup, so choosing a
    different language while the application is running left every Qt menu,
    file dialog and message box in the language the application STARTED in.
    The language pass carries it now: one load per change, and none at all
    when the language has not moved.
    """
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:                                        # noqa: BLE001
        return
    app = QApplication.instance()
    if app is None:
        return
    if getattr(app, "_spacr_qt_translator_code", None) == code:
        return
    install_qt_translations(app, code)


def retranslate_widget_tree(root, language: Optional[str] = None) -> None:
    """Retranslate static text in ``root`` and all existing descendants.

    The function is intentionally best-effort and idempotent. It never edits
    line-edit contents, text editors, table cells, model data, filenames or
    console output.

    Qt's OWN text follows too -- see :func:`_follow_qt_own_catalogs` -- so a
    language chosen after launch reaches the right-click menu of every text
    field, not only the captions spaCR wrote.
    """
    if root is None:
        return
    code = normalize_language(language or current_language())
    _follow_qt_own_catalogs(code)
    try:
        from PySide6.QtGui import QAction
        from PySide6.QtWidgets import (
            QAbstractButton,
            QComboBox,
            QGroupBox,
            QLabel,
            QLineEdit,
            QPlainTextEdit,
            QTableWidget,
            QTabWidget,
            QTextEdit,
            QTreeWidget,
            QWidget,
        )
    except Exception:
        return

    widgets = [root] if isinstance(root, QWidget) else []
    try:
        widgets.extend(root.findChildren(QWidget))
    except (AttributeError, RuntimeError):
        pass

    for widget in widgets:
        _translate_qt_text(
            widget, "windowTitle", "setWindowTitle",
            "_spacr_i18n_window_title", code)
        semantic_help = (
            widget.property("moduleAppKey")
            or widget.property("settingsAppKey")
        )
        if not semantic_help:
            _translate_qt_text(
                widget, "toolTip", "setToolTip",
                "_spacr_i18n_tooltip", code)
            _translate_qt_text(
                widget, "accessibleName", "setAccessibleName",
                "_spacr_i18n_accessible_name", code)
            _translate_qt_text(
                widget, "accessibleDescription", "setAccessibleDescription",
                "_spacr_i18n_accessible_description", code)

        # Chat messages, generated output and other dynamic labels opt out.
        # Translating their contents during a language switch would mutate
        # user/provider text rather than application chrome.
        dynamic_text = _refresh_dynamic_text(widget, code)
        semantic_setting_text = False
        setting_key = widget.property("settingKey")
        settings_app_key = widget.property("settingsAppKey")
        if (isinstance(widget, (QLabel, QAbstractButton))
                and setting_key and settings_app_key
                and not dynamic_text
                and not widget.property("i18nSkipText")):
            try:
                current = str(widget.text() or "")
                source = widget.property("_spacr_i18n_setting_text")
                if source is None:
                    source = current
                    if source:
                        widget.setProperty(
                            "_spacr_i18n_setting_text", str(source))
                if source:
                    rendered = None
                    # The compact catalog contains manually reviewed terms and
                    # therefore remains authoritative when it has this exact
                    # visible label.  The context-keyed catalog fills the much
                    # larger settings surface and app-specific labels.
                    if str(source) in _ROWS or str(source) in _TERM_ROWS:
                        rendered = tr(str(source), code)
                    else:
                        from .i18n_catalogs import setting_label
                        rendered = setting_label(
                            str(setting_key), str(source), code,
                            str(settings_app_key),
                        )
                        if rendered is None:
                            rendered = tr(str(source), code)
                    if rendered:
                        widget.setText(str(rendered))
                        semantic_setting_text = True
            except (AttributeError, RuntimeError, TypeError):
                pass
        if (isinstance(widget, (QLabel, QAbstractButton))
                and not dynamic_text
                and not semantic_setting_text
                and not widget.property("i18nSkipText")):
            _translate_qt_text(
                widget, "text", "setText", "_spacr_i18n_text", code)
        if isinstance(widget, QGroupBox):
            _translate_qt_text(
                widget, "title", "setTitle", "_spacr_i18n_title", code)
        if isinstance(widget, (QLineEdit, QPlainTextEdit, QTextEdit)):
            _translate_qt_text(
                widget, "placeholderText", "setPlaceholderText",
                "_spacr_i18n_placeholder", code)
        if isinstance(widget, QTabWidget):
            sources = getattr(widget, "_spacr_i18n_tab_sources", None)
            if sources is None or len(sources) != widget.count():
                sources = [widget.tabText(i) for i in range(widget.count())]
                widget._spacr_i18n_tab_sources = sources
            for index, source in enumerate(sources):
                widget.setTabText(index, tr(source, code))
        if isinstance(widget, QTableWidget):
            sources = getattr(widget, "_spacr_i18n_header_sources", None)
            if sources is None or len(sources) != widget.columnCount():
                sources = []
                for index in range(widget.columnCount()):
                    item = widget.horizontalHeaderItem(index)
                    sources.append(item.text() if item is not None else "")
                widget._spacr_i18n_header_sources = sources
            for index, source in enumerate(sources):
                item = widget.horizontalHeaderItem(index)
                if item is not None and source:
                    item.setText(tr(source, code))
        if isinstance(widget, QTreeWidget):
            sources = getattr(widget, "_spacr_i18n_header_sources", None)
            if sources is None or len(sources) != widget.columnCount():
                header = widget.headerItem()
                sources = [
                    header.text(index) if header is not None else ""
                    for index in range(widget.columnCount())
                ]
                widget._spacr_i18n_header_sources = sources
            header = widget.headerItem()
            if header is not None:
                for index, source in enumerate(sources):
                    if source:
                        header.setText(index, tr(source, code))
        if isinstance(widget, QComboBox) and not widget.isEditable():
            if widget.property("i18nSkipItems"):
                continue
            sources = getattr(widget, "_spacr_i18n_item_sources", None)
            if sources is None or len(sources) != widget.count():
                sources = [widget.itemText(i) for i in range(widget.count())]
                widget._spacr_i18n_item_sources = sources
            for index, source in enumerate(sources):
                translated = tr(source, code)
                if translated != source or source in _ROWS:
                    widget.setItemText(index, translated)
        _refresh_module_help(widget, code)
        try:
            module_api_key = widget.property("moduleApiAppKey")
        except (AttributeError, RuntimeError):
            module_api_key = None
        set_url = getattr(widget, "set_url", None)
        if module_api_key and callable(set_url):
            from .screens.settings_model import api_docs_url
            set_url(api_docs_url(str(module_api_key), language=code))
        retranslate_content = getattr(
            widget, "retranslate_dynamic_content", None)
        if callable(retranslate_content):
            try:
                retranslate_content(code)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

    actions = []
    try:
        actions = root.findChildren(QAction)
    except (AttributeError, RuntimeError):
        pass
    for action in actions:
        _translate_qt_text(
            action, "text", "setText", "_spacr_i18n_text", code)
        _translate_qt_text(
            action, "toolTip", "setToolTip", "_spacr_i18n_tooltip", code)
        if not action.property("moduleAppKey"):
            _translate_qt_text(
                action, "statusTip", "setStatusTip",
                "_spacr_i18n_status_tip", code)
        _refresh_module_help(action, code)

    # Settings tooltips are structured HTML (name, type, scientific prose and
    # API link), so rebuild them semantically after the generic Qt pass.
    #
    # The same catalog errors as the settings-label block above, which is
    # what this guard was missing: a malformed record raised TypeError out
    # of `setting_label` there and was caught, and raised it out of the same
    # catalog here and was not -- so a language switch stopped part way and
    # left the window half English with no way back except another switch.
    #
    # NOT IMPORTED UNLESS IT ALREADY IS. `refresh_api_tooltips` rebuilds the
    # tooltips that `settings_model` itself attached, so a tree can only hold
    # one if that module has already been imported -- and if it has not, there
    # is provably nothing here to refresh.
    #
    # Importing it anyway cost 0.3 s of a 1.4 s launch: the module reaches
    # external_mask_inputs, which reaches external_masks, which reaches
    # convert, which imports pandas. All of it paid at startup, to retranslate
    # a Home page that has no settings on it.
    if "spacr.qt.screens.settings_model" not in sys.modules:
        return
    try:
        from .screens.settings_model import refresh_api_tooltips
        refresh_api_tooltips(root, code)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass


#: Our language codes mapped to the ones Qt names its own catalogs by.
#:
#: Qt ships `qtbase_<code>.qm` for its OWN strings -- the words in a file
#: dialog, a message box's buttons, and the Copy/Paste/Select All that
#: every text field offers on right-click. None of those are spaCR's to
#: translate, and without this they stay English on a Swedish screen.
#:
#: Hindi and Icelandic are absent because Qt does not ship them. Their
#: users get English in Qt's own menus and spaCR's own text translated,
#: which is the best that can be done without writing those catalogs.
QT_CATALOGS = {
    "sv": "sv", "de": "de", "es": "es", "zh_CN": "zh_CN",
    "pt": "pt_BR", "ko": "ko", "fr": "fr",
}


def install_qt_translations(app, language: Optional[str] = None) -> bool:
    """Load Qt's own translations for ``language``. True if one loaded.

    Idempotent: a translator installed by an earlier call is removed
    first, so switching language twice does not leave the first one
    underneath answering for strings the second does not carry.
    """
    if app is None:
        return False
    code = normalize_language(language or current_language())
    try:
        from PySide6.QtCore import QLibraryInfo, QTranslator
    except Exception:                                        # noqa: BLE001
        return False

    previous = getattr(app, "_spacr_qt_translator", None)
    if previous is not None:
        try:
            app.removeTranslator(previous)
        except Exception:                                    # noqa: BLE001
            pass
        app._spacr_qt_translator = None
    # WHICH LANGUAGE IS LOADED, recorded whether or not one could be. Qt
    # ships no catalog for Hindi or Icelandic, and without this the
    # language pass would try to load one again on every dialog it sees.
    app._spacr_qt_translator_code = code

    catalog = QT_CATALOGS.get(code)
    if catalog is None:
        return False
    try:
        path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        translator = QTranslator(app)
        if not translator.load(f"qtbase_{catalog}", path):
            return False
        app.installTranslator(translator)
        # HELD ON THE APPLICATION. A QTranslator that is garbage collected
        # is a QTranslator Qt goes on asking and getting nothing from.
        app._spacr_qt_translator = translator
        return True
    except Exception:                                        # noqa: BLE001
        return False


def install_dialog_translation(app) -> None:
    """Translate transient Qt dialogs when they are shown.

    File pickers, message boxes, input prompts and progress dialogs are often
    constructed and executed in one expression, so they do not exist during
    the main-window language pass.  An application event filter catches only
    top-level ``QDialog`` show events and applies the same conservative exact
    catalog translation to their title, labels, buttons and accessible text.
    Dynamic paths, table data and user text remain outside that traversal.
    """
    if app is None or getattr(app, "_spacr_dialog_i18n_filter", None) is not None:
        return
    try:
        from PySide6.QtCore import QEvent, QObject
        from PySide6.QtWidgets import QDialog
    except Exception:
        return

    class _DialogTranslationFilter(QObject):
        def eventFilter(self, watched, event):  # noqa: N802
            if event.type() == QEvent.Show and isinstance(watched, QDialog):
                retranslate_widget_tree(watched)
            return False

    event_filter = _DialogTranslationFilter(app)
    app._spacr_dialog_i18n_filter = event_filter
    app.installEventFilter(event_filter)


__all__ = [
    "CATALOGS",
    "DEFAULT_LANGUAGE",
    "ENV_LANGUAGE",
    "LANGUAGES",
    "LANGUAGE_BY_CODE",
    "Language",
    "TERM_CATALOGS",
    "VALID_LANGUAGE_CODES",
    "catalog_coverage",
    "current_language",
    "has_translation",
    "install_dialog_translation",
    "install_qt_translations",
    "language_choices",
    "normalize_language",
    "retranslate_widget_tree",
    "set_translatable_items",
    "set_translatable_text",
    "tr",
]
