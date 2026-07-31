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
        "एनोटेशन", "주석", "Merking", "Annotation"),
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
        "Plattkö", "Plattenwarteschlange", "Cola de placas", "板队列",
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
        "Plattvisare", "Plattenansicht", "Visor de placas", "板查看器",
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
        "Rekrytering", "Rekrutierung", "Reclutamiento", "募集",
        "Recrutamento", "भर्ती", "모집", "Söfnun", "Recrutement"),
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
    "Feature maturity": _row(
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
    "Runtime": _row("Körtid", "Laufzeit", "Ejecución", "运行时", "Execução", "रनटाइम", "실행", "Keyrslutími", "Exécution"),
    "Reliability": _row("Tillförlitlighet", "Zuverlässigkeit", "Fiabilidad", "可靠性", "Confiabilidade", "विश्वसनीयता", "신뢰성", "Áreiðanleiki", "Fiabilité"),
    "Image": _row("Bild", "Bild", "Imagen", "图像", "Imagem", "छवि", "이미지", "Mynd", "Image"),
    "Images": _row("Bilder", "Bilder", "Imágenes", "图像", "Imagens", "छवियाँ", "이미지", "Myndir", "Images"),
    "Cell": _row("Cell", "Zelle", "Célula", "细胞", "Célula", "कोशिका", "세포", "Fruma", "Cellule"),
    "Cells": _row("Celler", "Zellen", "Células", "细胞", "Células", "कोशिकाएँ", "세포", "Frumur", "Cellules"),
    "Nucleus": _row("Cellkärna", "Zellkern", "Núcleo", "细胞核", "Núcleo", "नाभिक", "핵", "Kjarni", "Noyau"),
    "Pathogen": _row("Patogen", "Pathogen", "Patógeno", "病原体", "Patógeno", "रोगजनक", "병원체", "Sýkill", "Pathogène"),
    "Organelle": _row("Organell", "Organelle", "Orgánulo", "细胞器", "Organela", "कोशिकांग", "소기관", "Frumulíffæri", "Organite"),
    "Segmentation": _row("Segmentering", "Segmentierung", "Segmentación", "分割", "Segmentação", "खंडन", "분할", "Hlutun", "Segmentation"),
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
    "Cluster": _row("Kluster", "Cluster", "Clúster", "聚类", "Cluster", "क्लस्टर", "클러스터", "Klasi", "Groupe"),
    "Clustering": _row("Klustring", "Clustering", "Agrupamiento", "聚类", "Agrupamento", "क्लस्टरिंग", "군집화", "Klösun", "Regroupement"),
    "Advanced": _row("Avancerat", "Erweitert", "Avanzado", "高级", "Avançado", "उन्नत", "고급", "Ítarlegt", "Avancé"),
    "General": _row("Allmänt", "Allgemein", "General", "常规", "Geral", "सामान्य", "일반", "Almennt", "Général"),
    "Paths": _row("Sökvägar", "Pfade", "Rutas", "路径", "Caminhos", "पथ", "경로", "Slóðir", "Chemins"),
    "Controls": _row("Kontroller", "Kontrollen", "Controles", "控件", "Controles", "नियंत्रण", "컨트롤", "Stýringar", "Contrôles"),
    "Plate": _row("Platta", "Platte", "Placa", "板", "Placa", "प्लेट", "플레이트", "Plata", "Plaque"),
    "Plates": _row("Plattor", "Platten", "Placas", "板", "Placas", "प्लेट", "플레이트", "Plötur", "Plaques"),
    "Batch": _row("Batch", "Stapel", "Lote", "批次", "Lote", "बैच", "배치", "Lota", "Lot"),
    "Results": _row("Resultat", "Ergebnisse", "Resultados", "结果", "Resultados", "परिणाम", "결과", "Niðurstöður", "Résultats"),
    "Quality": _row("Kvalitet", "Qualität", "Calidad", "质量", "Qualidade", "गुणवत्ता", "품질", "Gæði", "Qualité"),
    "Control": _row("Kontroll", "Kontrolle", "Control", "控制", "Controle", "नियंत्रण", "관리", "Stýring", "Contrôle"),
    "Tracking": _row("Spårning", "Verfolgung", "Seguimiento", "跟踪", "Rastreamento", "ट्रैकिंग", "추적", "Rekjanleiki", "Suivi"),
    "Time": _row("Tid", "Zeit", "Tiempo", "时间", "Tempo", "समय", "시간", "Tími", "Temps"),
    "Calibration": _row("Kalibrering", "Kalibrierung", "Calibración", "校准", "Calibração", "अंशांकन", "보정", "Kvörðun", "Étalonnage"),
    "Summary": _row("Sammanfattning", "Zusammenfassung", "Resumen", "摘要", "Resumo", "सारांश", "요약", "Samantekt", "Résumé"),
    "Confusion matrix": _row("Förväxlingsmatris", "Konfusionsmatrix", "Matriz de confusión", "混淆矩阵", "Matriz de confusão", "भ्रम मैट्रिक्स", "혼동 행렬", "Ruglingsfylki", "Matrice de confusion"),
    "Per-plate metrics": _row("Mätvärden per platta", "Kennzahlen pro Platte", "Métricas por placa", "每板指标", "Métricas por placa", "प्रति-प्लेट मेट्रिक्स", "플레이트별 지표", "Mæligildi á plötu", "Métriques par plaque"),
    "Predictions": _row("Förutsägelser", "Vorhersagen", "Predicciones", "预测", "Previsões", "पूर्वानुमान", "예측", "Spár", "Prédictions"),
    "Leakage audit": _row("Läckagegranskning", "Datenleckprüfung", "Auditoría de fugas", "泄漏审计", "Auditoria de vazamento", "लीकेज ऑडिट", "누출 감사", "Lekagreining", "Audit des fuites"),
    "Results folder": _row("Resultatmapp", "Ergebnisordner", "Carpeta de resultados", "结果文件夹", "Pasta de resultados", "परिणाम फ़ोल्डर", "결과 폴더", "Niðurstöðumappa", "Dossier de résultats"),
    "Evaluation run": _row("Utvärderingskörning", "Auswertungslauf", "Ejecución de evaluación", "评估运行", "Execução de avaliação", "मूल्यांकन रन", "평가 실행", "Matskeyrsla", "Exécution d’évaluation"),
    "Scan": _row("Sök", "Scannen", "Escanear", "扫描", "Verificar", "स्कैन करें", "스캔", "Skanna", "Analyser"),
    "Open folder": _row("Öppna mapp", "Ordner öffnen", "Abrir carpeta", "打开文件夹", "Abrir pasta", "फ़ोल्डर खोलें", "폴더 열기", "Opna möppu", "Ouvrir le dossier"),
    "Classification": _row("Klassificering", "Klassifizierung", "Clasificación", "分类", "Classificação", "वर्गीकरण", "분류", "Flokkun", "Classification"),
    "Annotation": _row("Annotering", "Annotation", "Anotación", "标注", "Anotação", "एनोटेशन", "주석", "Merking", "Annotation"),
    "Classes": _row("Klasser", "Klassen", "Clases", "类别", "Classes", "वर्ग", "클래스", "Flokkar", "Classes"),
    "Optimization": _row("Optimering", "Optimierung", "Optimización", "优化", "Otimização", "अनुकूलन", "최적화", "Bestun", "Optimisation"),
    "stability repeats": _row("stabilitetsupprepningar", "Stabilitätswiederholungen", "repeticiones de estabilidad", "稳定性重复次数", "repetições de estabilidade", "स्थिरता पुनरावृत्तियाँ", "안정성 반복", "stöðugleikaendurtekningar", "répétitions de stabilité"),
    "neighborhood weight": _row("grannskapsvikt", "Nachbarschaftsgewicht", "peso de vecindad", "邻域权重", "peso da vizinhança", "पड़ोस भार", "이웃 가중치", "vægi nágrennis", "poids du voisinage"),
    "stability weight": _row("stabilitetsvikt", "Stabilitätsgewicht", "peso de estabilidad", "稳定性权重", "peso da estabilidade", "स्थिरता भार", "안정성 가중치", "vægi stöðugleika", "poids de stabilité"),
    "cluster weight": _row("klustervikt", "Clustergewicht", "peso de clúster", "聚类权重", "peso do cluster", "क्लस्टर भार", "클러스터 가중치", "vægi klasa", "poids des groupes"),
    "Loss": _row("Förlust", "Verlust", "Pérdida", "损失", "Perda", "हानि", "손실", "Tap", "Perte"),
    "Inference": _row("Inferens", "Inferenz", "Inferencia", "推理", "Inferência", "अनुमान", "추론", "Ályktun", "Inférence"),
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
    """Translate one Qt string property while retaining its English source."""
    getter = getattr(obj, getter_name, None)
    setter = getattr(obj, setter_name, None)
    if not callable(getter) or not callable(setter):
        return
    try:
        source = obj.property(property_name)
        if source is None:
            source = getter()
            if not source:
                return
            obj.setProperty(property_name, str(source))
        setter(tr(source, language))
    except RuntimeError:
        # A deferred-delete Qt wrapper may remain in findChildren briefly.
        return


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
            QTableWidget, QTabWidget, QWidget,
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
        _translate_qt_text(
            widget, "toolTip", "setToolTip",
            "_spacr_i18n_tooltip", code)
        _translate_qt_text(
            widget, "accessibleName", "setAccessibleName",
            "_spacr_i18n_accessible_name", code)
        _translate_qt_text(
            widget, "accessibleDescription", "setAccessibleDescription",
            "_spacr_i18n_accessible_description", code)

        if isinstance(widget, (QLabel, QAbstractButton)):
            _translate_qt_text(
                widget, "text", "setText", "_spacr_i18n_text", code)
        if isinstance(widget, QGroupBox):
            _translate_qt_text(
                widget, "title", "setTitle", "_spacr_i18n_title", code)
        if isinstance(widget, QLineEdit):
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
        _translate_qt_text(
            action, "statusTip", "setStatusTip",
            "_spacr_i18n_status_tip", code)


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
    "language_choices",
    "normalize_language",
    "retranslate_widget_tree",
    "tr",
]
