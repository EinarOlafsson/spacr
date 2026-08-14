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

from dataclasses import dataclass
import os
import re
import sys
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
    "Tutorial (web)": _row(
        "Handledning (webb)", "Tutorial (Web)", "Tutorial (web)",
        "教程（网页）", "Tutorial (web)", "ट्यूटोरियल (वेब)",
        "튜토리얼(웹)", "Kennsla (vefur)", "Tutoriel (web)"),
    "Documentation (web)": _row(
        "Dokumentation (webb)", "Dokumentation (Web)",
        "Documentación (web)", "文档（网页）", "Documentação (web)",
        "दस्तावेज़ (वेब)", "문서(웹)", "Skjölun (vefur)",
        "Documentation (web)"),
    "About spaCR": _row(
        "Om spaCR", "Über spaCR", "Acerca de spaCR", "关于 spaCR",
        "Sobre o spaCR", "spaCR के बारे में", "spaCR 정보", "Um spaCR",
        "À propos de spaCR"),
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
        "Activa o desactiva el UMAP interactivo de imágenes. Cuando está ACTIVADO (azul), puede pulsar un punto para previsualizar su imagen, dibujar alrededor de un clúster y guardar etiquetas manuales o automáticas en la base de datos.",
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
        "Abre una incidencia de GitHub prellenada con el último rastreo y el entorno. Usted la revisa antes de enviarla. Active o desactive esta opción en Ajustes de IA → Informar errores como incidencias de GitHub.",
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
    :returns: ``True`` if the row was added, ``False`` if ``source`` was
        already catalogued — registering the same app name twice is a
        no-op, not a conflict.
    :raises ValueError: if ``values`` is not one string per language, or
        any of them is blank. A missing translation fails here, where the
        app name is in the message, rather than as a blank sidebar row in
        Korean.
    """
    source = str(source)
    if source in _ROWS:
        return False
    row = _row(*[str(value) for value in values])
    if not all(value.strip() for value in row):
        raise ValueError(f"translation row for {source!r} has a blank entry")
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

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        word = match.group(0)
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


def retranslate_widget_tree(root, language: Optional[str] = None) -> None:
    """Retranslate static text in ``root`` and all existing descendants.

    The function is intentionally best-effort and idempotent. It never edits
    line-edit contents, text editors, table cells, model data, filenames or
    console output.
    """
    if root is None:
        return
    code = normalize_language(language or current_language())
    try:
        from PySide6.QtGui import QAction
        from PySide6.QtWidgets import (
            QAbstractButton, QComboBox, QGroupBox, QLabel, QLineEdit,
            QPlainTextEdit, QTableWidget, QTabWidget, QTextEdit, QTreeWidget,
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
    try:
        from .screens.settings_model import refresh_api_tooltips
        refresh_api_tooltips(root, code)
    except (ImportError, AttributeError, RuntimeError):
        pass


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
    "language_choices",
    "normalize_language",
    "retranslate_widget_tree",
    "set_translatable_text",
    "tr",
]
