#!/usr/bin/env python3
"""Build separate translated README and API-docstring catalogs.

English docstrings remain beside their Python symbols.  Translations are
stored below ``docs/i18n`` and copied into Sphinx's static tree, keyed by the
fully-qualified symbol plus a hash of the canonical English text.  The hash
makes a changed English docstring an explicit stale-catalog failure rather
than silently displaying an obsolete translation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Mapping

from build_i18n_catalogs import (
    MODEL_SPECS,
    _TOKEN_RE,
    _looks_degenerate,
    _translate_batches,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "i18n"
README_DIR = SOURCE_DIR / "readme"
STATIC_API_DIR = ROOT / "docs" / "source" / "_static" / "i18n" / "api"
API_DIR = STATIC_API_DIR
README_SOURCE = ROOT / "README.rst"

LANGUAGE_PICKER_LABELS = {
    "sv": "Språk",
    "de": "Sprachen",
    "es": "Idiomas",
    "zh_CN": "语言",
    "pt": "Idiomas",
    "hi": "भाषाएँ",
    "ko": "언어",
    "is": "Tungumál",
    "fr": "Langues",
}

# The project summary is the first prose a reader sees on GitHub and contains
# several domain-sensitive senses (pooled screen, plate, guide and hit).  Keep
# these two blocks human-reviewed instead of trusting a generic model to pick
# the scientific meaning from a short sentence.
_SUMMARY_SOURCE = (
    "spaCR segments and measures single cells in high-content microscopy "
    "images, links each cell to the gRNA it received, and reports which genes "
    "changed the phenotype. Plate images and FASTQ reads go in; per-object "
    "measurements, trained classifiers, per-guide and per-gene effect sizes, "
    "and a ranked hit list come out."
)
_SCOPE_SOURCE = (
    "If you run image-based pooled CRISPR screens, that is the whole path. If "
    "you have high-content microscopy and no screen, the segmentation, "
    "measurement, annotation and classification half runs on its own."
)
_TAGLINE_SOURCE = "**Spatial phenotype analysis of CRISPR screens.**"
_ATTRIBUTION_SOURCE = "Translation model attribution"
_STORAGE_SOURCE = (
    "Images, masks, crops, measurements, annotations, predictions, barcodes "
    "and well identifiers live in one SQLite project, so a number in a "
    "result can be traced back to the object it came from."
)
_EXECUTION_SOURCE = (
    "Run spaCR as a desktop application or headlessly on a workstation, "
    "server or cluster. Both drive the same modules, and CUDA is used "
    "automatically where a module supports it."
)
_WORKFLOW_SOURCE = (
    "Microscopy images (TIFF, OME-TIFF, LIF, CZI, ND2) and sequencing reads "
    "(FASTQ) enter complementary image-analysis and barcode-mapping "
    "pipelines. Object tables, crops, annotations, predictions, guide "
    "identities, QC results and well-level summaries are then analyzed "
    "together."
)
REVIEWED_README_BLOCKS = {
    _SUMMARY_SOURCE: {
        "sv": "spaCR segmenterar och mäter enskilda celler i mikroskopibilder med högt innehåll, kopplar varje cell till den gRNA den fick och rapporterar vilka gener som förändrade fenotypen. Plattbilder och FASTQ-läsningar matas in; ut kommer mätningar per objekt, tränade klassificerare, effektstorlekar per guide och gen samt en rangordnad träfflista.",
        "de": "spaCR segmentiert und vermisst einzelne Zellen in High-Content-Mikroskopiebildern, verknüpft jede Zelle mit der erhaltenen gRNA und berichtet, welche Gene den Phänotyp verändert haben. Plattenbilder und FASTQ-Reads dienen als Eingabe; ausgegeben werden Messungen pro Objekt, trainierte Klassifikatoren, Effektgrößen pro Guide und Gen sowie eine Rangliste der Treffer.",
        "es": "spaCR segmenta y mide células individuales en imágenes de microscopía de alto contenido, vincula cada célula con el gRNA que recibió e indica qué genes modificaron el fenotipo. Las imágenes de placas y las lecturas FASTQ son la entrada; las mediciones por objeto, los clasificadores entrenados, los tamaños del efecto por guía y por gen y una lista ordenada de resultados son la salida.",
        "zh_CN": "spaCR 对高内涵显微镜图像中的单细胞进行分割和测量，将每个细胞与其获得的 gRNA 关联，并报告哪些基因改变了表型。输入为孔板图像和 FASTQ 读段；输出包括逐对象测量、训练后的分类器、逐向导 RNA 和逐基因效应量，以及按优先级排序的候选结果列表。",
        "pt": "O spaCR segmenta e mede células individuais em imagens de microscopia de alto conteúdo, associa cada célula ao gRNA que ela recebeu e informa quais genes alteraram o fenótipo. As entradas são imagens de placas e leituras FASTQ; as saídas incluem medições por objeto, classificadores treinados, tamanhos de efeito por guia e por gene e uma lista classificada de resultados.",
        "hi": "spaCR उच्च-सामग्री माइक्रोस्कोपी छवियों में एकल कोशिकाओं का विभाजन और मापन करता है, प्रत्येक कोशिका को मिले gRNA से जोड़ता है और बताता है कि किन जीनों ने फीनोटाइप बदला। इनपुट के रूप में प्लेट छवियाँ और FASTQ रीड आती हैं; आउटपुट में प्रति-वस्तु मापन, प्रशिक्षित वर्गीकारक, प्रति-गाइड और प्रति-जीन प्रभाव आकार तथा प्राथमिकता के अनुसार परिणामों की सूची मिलती है।",
        "ko": "spaCR는 고함량 현미경 영상에서 단일 세포를 분할하고 측정하며, 각 세포를 전달받은 gRNA와 연결하고 어떤 유전자가 표현형을 바꾸었는지 보고합니다. 플레이트 영상과 FASTQ 리드를 입력하면 객체별 측정값, 학습된 분류기, 가이드별·유전자별 효과 크기와 우선순위가 지정된 후보 목록이 출력됩니다.",
        "is": "spaCR aðgreinir og mælir stakar frumur í afkastamiklum smásjármyndum, tengir hverja frumu við gRNA-ið sem hún fékk og greinir frá því hvaða gen breyttu svipgerðinni. Plötumyndir og FASTQ-raðir eru inntak; mælingar fyrir hvert viðfang, þjálfaðir flokkarar, áhrifastærðir fyrir hverja leiðarsameind og hvert gen og forgangsraðaður niðurstöðulisti eru úttak.",
        "fr": "spaCR segmente et mesure les cellules individuelles dans des images de microscopie à haut contenu, associe chaque cellule au gRNA qu’elle a reçu et indique quels gènes ont modifié le phénotype. Les images de plaques et les lectures FASTQ constituent les entrées ; les mesures par objet, les classificateurs entraînés, les tailles d’effet par guide et par gène et une liste de résultats classés constituent les sorties.",
    },
    _SCOPE_SOURCE: {
        "sv": "För bildbaserade poolade CRISPR-screeningar täcker detta hela arbetsflödet. Om du har mikroskopi med högt innehåll men ingen screening kan delarna för segmentering, mätning, annotering och klassificering köras fristående.",
        "de": "Für bildbasierte gepoolte CRISPR-Screens deckt dies den gesamten Arbeitsablauf ab. Bei High-Content-Mikroskopie ohne Screen können Segmentierung, Messung, Annotation und Klassifizierung eigenständig ausgeführt werden.",
        "es": "Para los cribados CRISPR agrupados y basados en imágenes, este es el flujo de trabajo completo. Si dispone de microscopía de alto contenido sin cribado, las etapas de segmentación, medición, anotación y clasificación pueden ejecutarse por separado.",
        "zh_CN": "对于基于图像的混合 CRISPR 筛选，这涵盖了完整工作流程。如果只有高内涵显微镜数据而没有筛选实验，也可以单独运行分割、测量、标注和分类部分。",
        "pt": "Para triagens CRISPR agrupadas e baseadas em imagens, esse é o fluxo de trabalho completo. Se você tiver microscopia de alto conteúdo sem uma triagem, as etapas de segmentação, medição, anotação e classificação poderão ser executadas de forma independente.",
        "hi": "छवि-आधारित पूल्ड CRISPR स्क्रीनिंग के लिए यह पूरा कार्यप्रवाह है। यदि आपके पास उच्च-सामग्री माइक्रोस्कोपी है लेकिन कोई स्क्रीनिंग नहीं है, तो विभाजन, मापन, एनोटेशन और वर्गीकरण वाले भाग स्वतंत्र रूप से चलाए जा सकते हैं।",
        "ko": "영상 기반 풀드 CRISPR 스크리닝에서는 이것이 전체 작업 흐름입니다. 고함량 현미경 데이터만 있고 스크리닝 실험은 없는 경우에도 분할, 측정, 주석 및 분류 단계를 독립적으로 실행할 수 있습니다.",
        "is": "Fyrir myndgreindar samsettar CRISPR-skimanir nær þetta yfir allt verkflæðið. Ef þú ert með afkastamiklar smásjármyndir en enga skimun er hægt að keyra aðgreiningu, mælingar, merkingar og flokkun sjálfstætt.",
        "fr": "Pour les criblages CRISPR groupés fondés sur l’imagerie, ce flux couvre l’ensemble du parcours. Avec des images de microscopie à haut contenu mais sans criblage, les étapes de segmentation, de mesure, d’annotation et de classification peuvent être exécutées indépendamment.",
    },
    _TAGLINE_SOURCE: {
        "sv": "**Rumslig fenotypanalys av CRISPR-screeningar.**",
        "de": "**Räumliche Phänotypanalyse von CRISPR-Screens.**",
        "es": "**Análisis espacial del fenotipo en cribados CRISPR.**",
        "zh_CN": "**CRISPR 筛选的空间表型分析。**",
        "pt": "**Análise espacial de fenótipos em triagens CRISPR.**",
        "hi": "**CRISPR स्क्रीनिंग का स्थानिक फीनोटाइप विश्लेषण।**",
        "ko": "**CRISPR 스크리닝의 공간 표현형 분석.**",
        "is": "**Rýmisbundin svipgerðargreining á CRISPR-skimunum.**",
        "fr": "**Analyse spatiale des phénotypes de criblages CRISPR.**",
    },
    _ATTRIBUTION_SOURCE: {
        "sv": "Information om översättningsmodellerna",
        "de": "Angaben zu den Übersetzungsmodellen",
        "es": "Información sobre los modelos de traducción",
        "zh_CN": "翻译模型说明",
        "pt": "Informações sobre os modelos de tradução",
        "hi": "अनुवाद मॉडल की जानकारी",
        "ko": "번역 모델 정보",
        "is": "Upplýsingar um þýðingarlíkön",
        "fr": "Informations sur les modèles de traduction",
    },
    _STORAGE_SOURCE: {
        "sv": "Bilder, masker, bildutsnitt, mätningar, annoteringar, prediktioner, streckkoder och brunnsidentifierare lagras i ett enda SQLite-projekt, så ett värde i ett resultat kan spåras tillbaka till objektet det kom från.",
        "de": "Bilder, Masken, Bildausschnitte, Messungen, Annotationen, Vorhersagen, Barcodes und Well-Kennungen liegen in einem einzigen SQLite-Projekt. Dadurch lässt sich ein Ergebniswert bis zu seinem Ursprungsobjekt zurückverfolgen.",
        "es": "Las imágenes, máscaras, recortes, mediciones, anotaciones, predicciones, códigos de barras e identificadores de pocillo se guardan en un único proyecto SQLite, por lo que cualquier valor de un resultado puede rastrearse hasta su objeto de origen.",
        "zh_CN": "图像、掩膜、图像裁剪、测量值、标注、预测、条形码和孔位标识符都存储在同一个 SQLite 项目中，因此结果中的数值可以追溯到其来源对象。",
        "pt": "Imagens, máscaras, recortes, medições, anotações, previsões, códigos de barras e identificadores de poço ficam em um único projeto SQLite, permitindo rastrear qualquer valor de resultado até o objeto de origem.",
        "hi": "छवियाँ, मास्क, इमेज क्रॉप, मापन, एनोटेशन, पूर्वानुमान, बारकोड और वेल पहचानकर्ता एक ही SQLite प्रोजेक्ट में रहते हैं, इसलिए किसी परिणाम के मान को उसके स्रोत ऑब्जेक्ट तक वापस खोजा जा सकता है।",
        "ko": "영상, 마스크, 이미지 크롭, 측정값, 주석, 예측, 바코드 및 웰 식별자는 하나의 SQLite 프로젝트에 저장되므로 결과의 값을 그 출처 객체까지 추적할 수 있습니다.",
        "is": "Myndir, grímur, myndúrklippur, mælingar, merkingar, spár, strikamerki og brunnaauðkenni eru geymd í einu SQLite-verkefni, þannig að rekja má niðurstöðugildi aftur til viðfangsins sem það kom frá.",
        "fr": "Les images, masques, recadrages, mesures, annotations, prédictions, codes-barres et identifiants de puits sont conservés dans un même projet SQLite, ce qui permet de relier chaque valeur d’un résultat à son objet d’origine.",
    },
    _EXECUTION_SOURCE: {
        "sv": "Kör spaCR som skrivbordsprogram eller utan grafiskt gränssnitt på en arbetsstation, server eller beräkningskluster. Båda sätten använder samma moduler, och CUDA används automatiskt när modulen stöder det.",
        "de": "Führen Sie spaCR als Desktopanwendung oder ohne grafische Oberfläche auf einer Workstation, einem Server oder Cluster aus. Beide Varianten verwenden dieselben Module; CUDA wird automatisch genutzt, wenn das jeweilige Modul es unterstützt.",
        "es": "Ejecute spaCR como aplicación de escritorio o sin interfaz gráfica en una estación de trabajo, servidor o clúster. Ambos modos usan los mismos módulos y CUDA se utiliza automáticamente cuando el módulo lo admite.",
        "zh_CN": "spaCR 可作为桌面应用程序运行，也可在工作站、服务器或集群上以无图形界面模式运行。两种方式使用相同的模块；模块支持 CUDA 时会自动启用。",
        "pt": "Execute o spaCR como aplicativo para desktop ou sem interface gráfica em uma estação de trabalho, servidor ou cluster. Os dois modos usam os mesmos módulos, e o CUDA é ativado automaticamente quando houver suporte no módulo.",
        "hi": "spaCR को डेस्कटॉप एप्लिकेशन के रूप में या वर्कस्टेशन, सर्वर अथवा क्लस्टर पर बिना ग्राफ़िकल इंटरफ़ेस के चलाएँ। दोनों तरीके समान मॉड्यूल चलाते हैं और समर्थित मॉड्यूल में CUDA अपने आप उपयोग होता है।",
        "ko": "spaCR를 데스크톱 애플리케이션으로 실행하거나 워크스테이션, 서버 또는 클러스터에서 그래픽 인터페이스 없이 실행할 수 있습니다. 두 방식 모두 동일한 모듈을 사용하며, 모듈이 지원하면 CUDA가 자동으로 활성화됩니다.",
        "is": "Keyrðu spaCR sem skjáborðsforrit eða án grafísks viðmóts á vinnustöð, þjóni eða reikniklasa. Báðar leiðir nota sömu einingar og CUDA er virkjað sjálfkrafa þegar einingin styður það.",
        "fr": "Exécutez spaCR comme application de bureau ou sans interface graphique sur une station de travail, un serveur ou un cluster. Les deux modes utilisent les mêmes modules et CUDA est activé automatiquement lorsqu’un module le prend en charge.",
    },
    _WORKFLOW_SOURCE: {
        "sv": "Mikroskopibilder (TIFF, OME-TIFF, LIF, CZI, ND2) och sekvenseringsläsningar (FASTQ) matas in i kompletterande arbetsflöden för bildanalys och streckkodsmappning. Objekttabeller, bildutsnitt, annoteringar, prediktioner, guideidentiteter, QC-resultat och sammanfattningar per brunn analyseras sedan tillsammans.",
        "de": "Mikroskopiebilder (TIFF, OME-TIFF, LIF, CZI, ND2) und Sequenzierungs-Reads (FASTQ) durchlaufen einander ergänzende Pipelines für Bildanalyse und Barcode-Zuordnung. Objekttabellen, Bildausschnitte, Annotationen, Vorhersagen, Guide-Identitäten, QC-Ergebnisse und Zusammenfassungen auf Well-Ebene werden anschließend gemeinsam analysiert.",
        "es": "Las imágenes de microscopía (TIFF, OME-TIFF, LIF, CZI, ND2) y las lecturas de secuenciación (FASTQ) pasan por flujos complementarios de análisis de imágenes y asignación de códigos de barras. Después se analizan conjuntamente las tablas de objetos, los recortes, las anotaciones, las predicciones, las identidades de guía, los resultados de QC y los resúmenes por pocillo.",
        "zh_CN": "显微镜图像（TIFF、OME-TIFF、LIF、CZI、ND2）和测序读段（FASTQ）分别进入互补的图像分析与条形码映射流程。随后对对象表、图像裁剪、标注、预测、向导 RNA 身份、QC 结果和孔位级汇总进行联合分析。",
        "pt": "Imagens de microscopia (TIFF, OME-TIFF, LIF, CZI, ND2) e leituras de sequenciamento (FASTQ) entram em fluxos complementares de análise de imagens e mapeamento de códigos de barras. Em seguida, tabelas de objetos, recortes, anotações, previsões, identidades de guia, resultados de QC e resumos por poço são analisados em conjunto.",
        "hi": "माइक्रोस्कोपी छवियाँ (TIFF, OME-TIFF, LIF, CZI, ND2) और सीक्वेंसिंग रीड (FASTQ) पूरक इमेज-विश्लेषण तथा बारकोड-मैपिंग कार्यप्रवाह में जाती हैं। इसके बाद ऑब्जेक्ट तालिकाएँ, इमेज क्रॉप, एनोटेशन, पूर्वानुमान, गाइड पहचान, QC परिणाम और प्रति-वेल सारांश एक साथ विश्लेषित किए जाते हैं।",
        "ko": "현미경 영상(TIFF, OME-TIFF, LIF, CZI, ND2)과 시퀀싱 리드(FASTQ)는 서로 보완적인 영상 분석 및 바코드 매핑 작업 흐름으로 들어갑니다. 그런 다음 객체 테이블, 이미지 크롭, 주석, 예측, 가이드 식별 정보, QC 결과 및 웰 단위 요약을 함께 분석합니다.",
        "is": "Smásjármyndir (TIFF, OME-TIFF, LIF, CZI, ND2) og raðgreiningarlestur (FASTQ) fara í samverkandi ferli fyrir myndgreiningu og strikamerkjavörpun. Síðan eru viðfangstöflur, myndúrklippur, merkingar, spár, auðkenni leiðarsameinda, QC-niðurstöður og samantektir fyrir hvern brunn greind saman.",
        "fr": "Les images de microscopie (TIFF, OME-TIFF, LIF, CZI, ND2) et les lectures de séquençage (FASTQ) alimentent des flux complémentaires d’analyse d’images et d’association des codes-barres. Les tables d’objets, recadrages, annotations, prédictions, identités des guides, résultats de QC et résumés par puits sont ensuite analysés ensemble.",
    },
}

REVIEWED_README_HEADINGS = {
    "Workflow at a glance": {
        "sv": "Arbetsflödet i korthet", "de": "Workflow auf einen Blick",
        "es": "Flujo de trabajo de un vistazo", "zh_CN": "工作流程概览",
        "pt": "Visão geral do fluxo de trabalho", "hi": "कार्यप्रवाह का अवलोकन",
        "ko": "작업 흐름 개요", "is": "Yfirlit yfir verkflæðið",
        "fr": "Vue d’ensemble du flux de travail",
    },
    "Quick start": {
        "sv": "Snabbstart", "de": "Schnellstart", "es": "Inicio rápido",
        "zh_CN": "快速开始", "pt": "Início rápido", "hi": "त्वरित शुरुआत",
        "ko": "빠른 시작", "is": "Flýtiræsing", "fr": "Démarrage rapide",
    },
    "Installation details": {
        "sv": "Installationsinformation", "de": "Installationsdetails",
        "es": "Detalles de instalación", "zh_CN": "安装详情",
        "pt": "Detalhes da instalação", "hi": "स्थापना विवरण",
        "ko": "설치 세부 정보", "is": "Upplýsingar um uppsetningu",
        "fr": "Détails de l’installation",
    },
    "Lightweight installers — no conda or existing Python required": {
        "sv": "Lätta installationsprogram — varken conda eller befintlig Python krävs",
        "de": "Leichte Installationsprogramme — weder conda noch vorhandenes Python erforderlich",
        "es": "Instaladores ligeros — no requieren conda ni una instalación de Python",
        "zh_CN": "轻量级安装程序 — 无需 conda 或现有 Python 环境",
        "pt": "Instaladores leves — não exigem conda nem uma instalação existente do Python",
        "hi": "हल्के इंस्टॉलर — conda या पहले से स्थापित Python की आवश्यकता नहीं",
        "ko": "경량 설치 프로그램 — conda 또는 기존 Python 환경 불필요",
        "is": "Létt uppsetningarforrit — hvorki conda né fyrirliggjandi Python þarf",
        "fr": "Programmes d’installation légers — ni conda ni installation Python existante requis",
    },
    "Desktop application from PyPI": {
        "sv": "Skrivbordsprogram från PyPI", "de": "Desktopanwendung von PyPI",
        "es": "Aplicación de escritorio desde PyPI", "zh_CN": "通过 PyPI 安装桌面应用程序",
        "pt": "Aplicativo para desktop pelo PyPI", "hi": "PyPI से डेस्कटॉप एप्लिकेशन",
        "ko": "PyPI에서 데스크톱 애플리케이션 설치", "is": "Skjáborðsforrit frá PyPI",
        "fr": "Application de bureau depuis PyPI",
    },
    "Headless or server installation": {
        "sv": "Installation utan grafiskt gränssnitt eller på server",
        "de": "Installation ohne grafische Oberfläche oder auf einem Server",
        "es": "Instalación sin interfaz gráfica o en servidor",
        "zh_CN": "无图形界面或服务器安装", "pt": "Instalação sem interface gráfica ou em servidor",
        "hi": "बिना ग्राफ़िकल इंटरफ़ेस या सर्वर पर स्थापना",
        "ko": "그래픽 인터페이스 없이 또는 서버에 설치",
        "is": "Uppsetning án grafísks viðmóts eða á þjóni",
        "fr": "Installation sans interface graphique ou sur serveur",
    },
    "Latest development branch": {
        "sv": "Senaste utvecklingsgrenen", "de": "Neuester Entwicklungszweig",
        "es": "Rama de desarrollo más reciente", "zh_CN": "最新开发分支",
        "pt": "Ramificação de desenvolvimento mais recente", "hi": "नवीनतम विकास शाखा",
        "ko": "최신 개발 브랜치", "is": "Nýjasta þróunargrein",
        "fr": "Branche de développement la plus récente",
    },
    "Conda environments": {
        "sv": "Conda-miljöer", "de": "Conda-Umgebungen", "es": "Entornos conda",
        "zh_CN": "Conda 环境", "pt": "Ambientes conda", "hi": "Conda वातावरण",
        "ko": "Conda 환경", "is": "Conda-umhverfi", "fr": "Environnements conda",
    },
    "Optional capabilities": {
        "sv": "Valfria funktioner", "de": "Optionale Funktionen",
        "es": "Funciones opcionales", "zh_CN": "可选功能",
        "pt": "Recursos opcionais", "hi": "वैकल्पिक सुविधाएँ",
        "ko": "선택 기능", "is": "Valfrjálsir eiginleikar",
        "fr": "Fonctionnalités facultatives",
    },
    "Command-line entry points": {
        "sv": "Kommandoradskommandon", "de": "Befehle für die Kommandozeile",
        "es": "Comandos de línea de comandos", "zh_CN": "命令行入口",
        "pt": "Comandos de linha de comando", "hi": "कमांड-लाइन प्रवेश बिंदु",
        "ko": "명령줄 진입점", "is": "Skipanalínuskipanir",
        "fr": "Points d’entrée en ligne de commande",
    },
    "Features": {
        "sv": "Funktioner", "de": "Funktionen", "es": "Funciones",
        "zh_CN": "功能", "pt": "Recursos", "hi": "विशेषताएँ",
        "ko": "기능", "is": "Eiginleikar", "fr": "Fonctionnalités",
    },
    "The six modules most screens use": {
        "sv": "De sex moduler som används i de flesta screeningar",
        "de": "Die sechs Module, die in den meisten Screens verwendet werden",
        "es": "Los seis módulos más usados en los cribados",
        "zh_CN": "大多数筛选实验使用的六个模块",
        "pt": "Os seis módulos mais usados nas triagens",
        "hi": "अधिकांश स्क्रीनिंग में उपयोग होने वाले छह मॉड्यूल",
        "ko": "대부분의 스크리닝에서 사용하는 6개 모듈",
        "is": "Einingarnar sex sem flestar skimanir nota",
        "fr": "Les six modules les plus utilisés dans les criblages",
    },
    "New in 1.5.0.0": {
        "sv": "Nytt i 1.5.0.0", "de": "Neu in 1.5.0.0", "es": "Novedades de 1.5.0.0",
        "zh_CN": "1.5.0.0 新增功能", "pt": "Novidades na 1.5.0.0",
        "hi": "1.5.0.0 में नया", "ko": "1.5.0.0의 새로운 기능",
        "is": "Nýtt í 1.5.0.0", "fr": "Nouveautés de la version 1.5.0.0",
    },
    "Internationalized desktop interface": {
        "sv": "Flerspråkigt skrivbordsgränssnitt", "de": "Mehrsprachige Desktopoberfläche",
        "es": "Interfaz de escritorio multilingüe", "zh_CN": "多语言桌面界面",
        "pt": "Interface multilíngue para desktop", "hi": "बहुभाषी डेस्कटॉप इंटरफ़ेस",
        "ko": "다국어 데스크톱 인터페이스", "is": "Fjöltyngt skjáborðsviðmót",
        "fr": "Interface de bureau multilingue",
    },
    "Animated settings guidance": {
        "sv": "Animerad hjälp för inställningar", "de": "Animierte Einstellungshilfe",
        "es": "Guía animada de ajustes", "zh_CN": "动画设置指南",
        "pt": "Guia animado de configurações", "hi": "एनिमेटेड सेटिंग मार्गदर्शन",
        "ko": "애니메이션 설정 안내", "is": "Hreyfimyndaleiðbeiningar fyrir stillingar",
        "fr": "Guide animé des paramètres",
    },
    "Module reference": {
        "sv": "Modulreferens", "de": "Modulreferenz", "es": "Referencia de módulos",
        "zh_CN": "模块参考", "pt": "Referência dos módulos", "hi": "मॉड्यूल संदर्भ",
        "ko": "모듈 참조", "is": "Tilvísun eininga", "fr": "Référence des modules",
    },
    "Data": {
        "sv": "Data", "de": "Daten", "es": "Datos", "zh_CN": "数据",
        "pt": "Dados", "hi": "डेटा", "ko": "데이터", "is": "Gögn", "fr": "Données",
    },
    "Reference datasets": {
        "sv": "Referensdatauppsättningar", "de": "Referenzdatensätze",
        "es": "Conjuntos de datos de referencia", "zh_CN": "参考数据集",
        "pt": "Conjuntos de dados de referência", "hi": "संदर्भ डेटासेट",
        "ko": "참조 데이터세트", "is": "Viðmiðunargagnasöfn",
        "fr": "Jeux de données de référence",
    },
    "Contributing and support": {
        "sv": "Bidrag och support", "de": "Beiträge und Support",
        "es": "Contribuciones y soporte", "zh_CN": "贡献与支持",
        "pt": "Contribuições e suporte", "hi": "योगदान और सहायता",
        "ko": "기여 및 지원", "is": "Framlög og aðstoð",
        "fr": "Contributions et assistance",
    },
    "Licensing": {
        "sv": "Licens", "de": "Lizenz", "es": "Licencia", "zh_CN": "许可",
        "pt": "Licença", "hi": "लाइसेंस", "ko": "라이선스", "is": "Leyfi",
        "fr": "Licence",
    },
    "Tutorials": {
        "sv": "Handledningar", "de": "Tutorials", "es": "Tutoriales",
        "zh_CN": "教程", "pt": "Tutoriais", "hi": "ट्यूटोरियल",
        "ko": "튜토리얼", "is": "Kennsluefni", "fr": "Tutoriels",
    },
    "Citing spaCR": {
        "sv": "Citera spaCR", "de": "spaCR zitieren", "es": "Citar spaCR",
        "zh_CN": "引用 spaCR", "pt": "Como citar o spaCR", "hi": "spaCR का संदर्भ",
        "ko": "spaCR 인용", "is": "Tilvísun í spaCR", "fr": "Citer spaCR",
    },
}

_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+")
_UNDERLINE_RE = re.compile(r"^\s*[=~^`'\-:#*+]{3,}\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_FIELD_RE = re.compile(
    r"^(:(?:param|parameter|arg|argument|keyword|kwarg|type|returns?|"
    r"rtype|raises?|ivar|vartype|cvar|var)\b[^:]*:)\s*(.*)$"
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _clean_doc(node: ast.AST) -> str:
    value = ast.get_docstring(node, clean=False) or ""
    return inspect.cleandoc(value).strip()


def public_docstrings() -> dict[str, str]:
    """Extract public module, class, function and method docstrings."""
    docs: dict[str, str] = {}
    for path in sorted((ROOT / "spacr").rglob("*.py")):
        if any(part in {"tests", "__pycache__"} for part in path.parts):
            continue
        # These are generated translation payloads, not Python API.  Including
        # their module headers makes every locale regeneration stale every API
        # locale and needlessly exposes generator metadata in the API picker.
        if "i18n_catalogs" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        module = _module_name(path)
        module_doc = _clean_doc(tree)
        if module_doc:
            docs[module] = module_doc
        for node in tree.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            key = f"{module}.{node.name}"
            doc = _clean_doc(node)
            if doc:
                docs[key] = doc
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if child.name.startswith("_"):
                        continue
                    child_doc = _clean_doc(child)
                    if child_doc:
                        docs[f"{key}.{child.name}"] = child_doc
    return dict(sorted(docs.items()))


def _split_long(text: str, limit: int = 1000) -> list[str]:
    """Split prose below the OPUS models' 480-token generation ceiling.

    A 1,000-character ceiling leaves room for German/Portuguese expansion and
    for protected RST markers.  Silent tokenizer truncation is unacceptable
    here because it can produce a fluent-looking translation with the end of
    a docstring missing.
    """
    if len(text) <= limit:
        return [text]
    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    bounded: list[str] = []
    for chunk in chunks:
        while len(chunk) > limit:
            split_at = chunk.rfind(" ", 0, limit)
            if split_at < limit // 2:
                split_at = limit
            bounded.append(chunk[:split_at].strip())
            chunk = chunk[split_at:].strip()
        if chunk:
            bounded.append(chunk)
    return bounded


def translatable_blocks(text: str) -> tuple[list[str], list[tuple[str, object]]]:
    """Split reStructuredText into prose blocks and lossless layout tokens."""
    lines = text.splitlines()
    blocks: list[str] = []
    layout: list[tuple[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            layout.append(("raw", ""))
            index += 1
            continue
        # The README language picker is navigation markup, not prose.  Passing
        # all ten RST links through a translation model can reorder or damage
        # their delimiters even when the visible language names are unchanged.
        # Keep it byte-for-byte here and localize only its leading label after
        # the translated document has been rebuilt.
        if line.startswith("Languages:") and "README" in line:
            literal = [line]
            index += 1
            while index < len(lines) and lines[index].strip():
                literal.append(lines[index])
                index += 1
            layout.append(("raw_lines", literal))
            continue
        field_match = _FIELD_RE.match(line)
        if field_match:
            prefix, first = field_match.groups()
            field = [first] if first else []
            index += 1
            while (index < len(lines) and lines[index].strip()
                   and lines[index].startswith((" ", "\t"))):
                field.append(lines[index].strip())
                index += 1
            prose = " ".join(field)
            positions = []
            for piece in _split_long(prose):
                positions.append(len(blocks))
                blocks.append(piece)
            layout.append(("translated_prefixed", (prefix, positions)))
            continue
        bullet_match = re.match(r"^(\s*(?:[*-]|#\.)\s+)(.*)$", line)
        if bullet_match:
            prefix, first = bullet_match.groups()
            base_indent = len(prefix) - len(prefix.lstrip())
            item = [first]
            index += 1
            while index < len(lines) and lines[index].strip():
                following = lines[index]
                following_bullet = re.match(
                    r"^(\s*)(?:[*-]|#\.)\s+", following
                )
                if following_bullet and len(following_bullet.group(1)) <= base_indent:
                    break
                following_indent = len(following) - len(following.lstrip())
                if following_indent <= base_indent:
                    break
                item.append(lines[index].strip())
                index += 1
            prose = " ".join(item)
            positions = []
            for piece in _split_long(prose):
                positions.append(len(blocks))
                blocks.append(piece)
            layout.append(("translated_prefixed", (prefix, positions)))
            continue
        # Doctest prompts, their continuation lines and expected output are
        # executable documentation.  Translating any token in that block can
        # turn valid Python into convincing-looking broken code, so retain the
        # complete example through the next blank line byte-for-byte.
        if line.lstrip().startswith((">>>", "...")):
            literal = [line]
            index += 1
            while index < len(lines) and lines[index].strip():
                literal.append(lines[index])
                index += 1
            layout.append(("raw_lines", literal))
            continue
        if line.startswith((" ", "\t")):
            literal = [line]
            index += 1
            while index < len(lines) and (
                not lines[index].strip() or lines[index].startswith((" ", "\t"))
            ):
                literal.append(lines[index])
                index += 1
            layout.append(("raw_lines", literal))
            continue
        if _DIRECTIVE_RE.match(line):
            literal = [line]
            index += 1
            while index < len(lines) and (
                not lines[index].strip() or lines[index].startswith((" ", "\t"))
            ):
                literal.append(lines[index])
                index += 1
            layout.append(("raw_lines", literal))
            continue
        if _UNDERLINE_RE.match(line):
            layout.append(("raw", line))
            index += 1
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines):
            following = lines[index]
            if not following.strip() or _DIRECTIVE_RE.match(following):
                break
            if following.lstrip().startswith((">>>", "...")):
                break
            if _UNDERLINE_RE.match(following):
                break
            if following.lstrip().startswith(("* ", "- ", "#. ")):
                break
            if _FIELD_RE.match(following):
                break
            paragraph.append(following.strip())
            index += 1
        prose = " ".join(paragraph)
        pieces = _split_long(prose)
        positions = []
        for piece in pieces:
            positions.append(len(blocks))
            blocks.append(piece)
        layout.append(("translated", positions))
    return blocks, layout


def rebuild_document(layout: Iterable[tuple[str, object]], translated: list[str]) -> str:
    lines: list[str] = []
    for kind, payload in layout:
        if kind == "raw":
            lines.append(str(payload))
        elif kind == "raw_lines":
            lines.extend(str(line) for line in payload)
        elif kind == "translated_prefixed":
            prefix, positions = payload
            separator = (
                "" if not positions or str(prefix).endswith((" ", "\t"))
                else " "
            )
            lines.append(
                str(prefix) + separator
                + " ".join(translated[index] for index in positions)
            )
        else:
            lines.append(" ".join(translated[index] for index in payload))
    for index in range(1, len(lines)):
        underline = lines[index].strip()
        if _UNDERLINE_RE.match(underline) and lines[index - 1].strip():
            character = underline[0]
            lines[index] = character * max(
                len(underline), len(lines[index - 1].strip())
            )
    return "\n".join(lines).strip()


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _english_manifest(docs: Mapping[str, str]) -> dict[str, object]:
    return {
        "schema": 1,
        "language": "en",
        "symbols": {
            key: {"source_sha256": _source_hash(value), "text": value}
            for key, value in docs.items()
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _translate_documents(
    documents: Mapping[str, str], language: str, model_root: Path, args,
) -> dict[str, str]:
    block_map: dict[str, tuple[list[str], list[tuple[str, object]]]] = {}
    unique: set[str] = set()
    for key, value in documents.items():
        blocks, layout = translatable_blocks(value)
        block_map[key] = (blocks, layout)
        unique.update(blocks)
    translations = _translate_batches(
        sorted(unique), language, model_root,
        device=args.device, batch_size=args.batch_size, beams=args.beams,
        threads=args.threads,
    )
    reviewed_blocks = {**REVIEWED_README_BLOCKS, **REVIEWED_README_HEADINGS}
    for source, reviewed in reviewed_blocks.items():
        if source in translations:
            translations[source] = reviewed[language]
    result: dict[str, str] = {}
    for key, (blocks, layout) in block_map.items():
        result[key] = rebuild_document(
            layout, [translations[block] for block in blocks],
        )
    return result


def write_language(
    docs: Mapping[str, str], language: str, translations: Mapping[str, str],
) -> None:
    model, _folder, license_name, _prefix = MODEL_SPECS[language]
    payload = {
        "schema": 1,
        "language": language,
        "generator": model,
        "license": license_name,
        "symbols": {
            key: {
                "source_sha256": _source_hash(source),
                "text": translations[key],
            }
            for key, source in docs.items()
        },
    }
    path = API_DIR / f"{language}.json"
    _write_json(path, payload)


def reusable_api_translations(
    docs: Mapping[str, str], language: str,
) -> dict[str, str]:
    """Return reviewed/generated entries whose English source is unchanged."""
    path = API_DIR / f"{language}.json"
    try:
        symbols = json.loads(path.read_text(encoding="utf-8")).get(
            "symbols", {}
        )
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return {}
    reusable: dict[str, str] = {}
    for key, source in docs.items():
        record = symbols.get(key, {})
        text = str(record.get("text", "")).strip()
        if record.get("source_sha256") == _source_hash(source) and text:
            reusable[key] = text
    return reusable


def audit(docs: Mapping[str, str], languages: Iterable[str]) -> int:
    failures: list[str] = []
    expected = set(docs)
    protected_pattern = re.compile(
        r"``[^`]+``|:(?:class|func|mod|meth|attr|data|doc):`[^`]+`|"
        r"https?://[^\s)>}\]]+"
    )
    field_pattern = re.compile(
        r"(?m)^(:(?:param|parameter|arg|argument|keyword|kwarg|type|"
        r"returns?|rtype|raises?|ivar|vartype|cvar|var)\b[^:]*:)"
    )
    rst_link_pattern = re.compile(r"`[^`\n]+\s+<([^>\n]+)>`_")

    def protected_values(text: str) -> list[str]:
        values = []
        for value in protected_pattern.findall(text):
            # Paragraph reflow may collapse a source line break inside an
            # inline literal or an RST role; its referenced value is still
            # unchanged and the one-line rendering is valid RST.
            values.append(re.sub(r"\s+", " ", value).rstrip(".,;:!?"))
        return sorted(values)

    def syntax_contract(source: str, translated: str, label: str) -> None:
        if protected_values(source) != protected_values(translated):
            failures.append(f"{label}: code/link/RST roles changed")
        if sorted(field_pattern.findall(source)) != sorted(
            field_pattern.findall(translated)
        ):
            failures.append(f"{label}: RST fields changed")
        if sorted(rst_link_pattern.findall(source)) != sorted(
            rst_link_pattern.findall(translated)
        ):
            failures.append(f"{label}: RST link targets changed")
        source_doctest = [
            line for line in source.splitlines()
            if line.lstrip().startswith((">>>", "..."))
        ]
        if any(line not in translated.splitlines() for line in source_doctest):
            failures.append(f"{label}: doctest code changed")
        if _TOKEN_RE.search(translated) or re.search(
            r"Z\s*X\s*Q\s*\d", translated
        ):
            failures.append(f"{label}: leaked protection token")

    readme_source = README_SOURCE.read_text(encoding="utf-8")
    script_pattern = {
        "zh_CN": re.compile(r"[\u3400-\u9fff]"),
        "hi": re.compile(r"[\u0900-\u097f]"),
        "ko": re.compile(r"[\uac00-\ud7af]"),
    }
    for language in languages:
        path = API_DIR / f"{language}.json"
        if not path.is_file():
            failures.append(f"{language}: API catalog is missing")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        symbols = payload.get("symbols", {})
        missing = expected - set(symbols)
        stale = set(symbols) - expected
        if missing:
            failures.append(f"{language}: {len(missing)} API symbols missing")
        if stale:
            failures.append(f"{language}: {len(stale)} stale API symbols")
        unchanged_prose = 0
        missing_target_script = 0
        for key, source in docs.items():
            record = symbols.get(key, {})
            if record.get("source_sha256") != _source_hash(source):
                failures.append(f"{language}: stale source hash for {key}")
            if not str(record.get("text", "")).strip():
                failures.append(f"{language}: blank translation for {key}")
            else:
                translated_text = str(record.get("text", ""))
                is_prose = len(source) >= 80 and bool(
                    re.search(r"[A-Za-z]{4}", source)
                )
                if is_prose and translated_text == source:
                    unchanged_prose += 1
                if (is_prose and language in script_pattern
                        and not script_pattern[language].search(translated_text)):
                    missing_target_script += 1
                if _looks_degenerate(source, translated_text, language):
                    failures.append(f"{language}/{key}: degenerate translation")
                syntax_contract(
                    source, translated_text,
                    f"{language}/{key}",
                )
        prose_limit = max(100, len(docs) // 10)
        if unchanged_prose > prose_limit:
            failures.append(
                f"{language}: {unchanged_prose} API docstrings remain English"
            )
        if missing_target_script > prose_limit:
            failures.append(
                f"{language}: {missing_target_script} API docstrings lack target script"
            )
        readme_path = README_DIR / f"README.{language}.rst"
        if not readme_path.is_file():
            failures.append(f"{language}: translated README is missing")
        else:
            readme = readme_path.read_text(encoding="utf-8")
            if len(readme) < 10_000:
                failures.append(f"{language}: translated README is too short")
            contract_readme = readme.replace(
                "<../../../README.rst>", "<README.rst>"
            ).replace(
                "<../TRANSLATION_MODELS.md>",
                "<docs/i18n/TRANSLATION_MODELS.md>",
            )
            contract_readme = re.sub(
                r"<README\.([A-Za-z_]+)\.rst>",
                r"<docs/i18n/readme/README.\1.rst>",
                contract_readme,
            )
            syntax_contract(
                readme_source, contract_readme, f"{language}/README"
            )
            if "../../../README.rst" not in readme:
                failures.append(f"{language}: English README link is broken")
    if failures:
        print("\n".join(failures[:200]), file=sys.stderr)
        return 1
    print(f"verified API catalogs: languages={len(tuple(languages))} symbols={len(docs)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument(
        "--model-root", type=Path,
        default=Path("/mnt/firecuda2/Claude/toxoplasma_projects/tutorials/project/translation_models/opus"),
    )
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="retranslate current API entries and README instead of reusing them",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    docs = public_docstrings()
    _write_json(API_DIR / "en.json", _english_manifest(docs))
    print(f"wrote English API manifest: symbols={len(docs)}")
    if args.sources_only:
        return 0
    if args.audit:
        return audit(docs, args.languages)

    readme = README_SOURCE.read_text(encoding="utf-8")
    readme_links: list[tuple[str, str, str]] = []
    for index, match in enumerate(re.finditer(
        r"`([^`<>]+?)\s+<([^>]+)>`_", readme
    )):
        label, target = match.group(1), match.group(2)
        # The language picker deliberately shows every language in its own
        # spelling. All other prose labels (not their destinations) belong to
        # the translated GitHub page.
        if target == "README.rst" or target.startswith(
            "docs/i18n/readme/README."
        ):
            continue
        key = f"__readme_link_{index}__"
        readme_links.append((key, label, target))
    for language in args.languages:
        reusable = {} if args.force else reusable_api_translations(
            docs, language,
        )
        pending = {key: source for key, source in docs.items() if key not in reusable}
        translated = dict(reusable)
        if pending:
            translated.update(
                _translate_documents(pending, language, args.model_root, args)
            )
        write_language(docs, language, translated)

        readme_path = README_DIR / f"README.{language}.rst"
        rebuild_readme = args.force or not readme_path.is_file()
        if rebuild_readme:
            documents = {"__readme__": readme}
            documents.update({key: label for key, label, _target in readme_links})
            readme_translation = _translate_documents(
                documents, language, args.model_root, args,
            )
            README_DIR.mkdir(parents=True, exist_ok=True)
            localized_readme = readme_translation["__readme__"]
            localized_readme = localized_readme.replace(
                "Languages:", f"{LANGUAGE_PICKER_LABELS[language]}:", 1
            )
            for key, label, target in readme_links:
                localized_readme = localized_readme.replace(
                    f"`{label} <{target}>`_",
                    f"`{readme_translation[key]} <{target}>`_",
                )
            localized_readme = localized_readme.replace(
                "docs/i18n/readme/README.", "README."
            ).replace(
                "docs/i18n/TRANSLATION_MODELS.md", "../TRANSLATION_MODELS.md"
            ).replace(
                "<README.rst>", "<../../../README.rst>"
            )
            readme_path.write_text(localized_readme + "\n", encoding="utf-8")
        print(
            f"wrote {language}: API={len(docs)} "
            f"translated={len(pending)} README={int(rebuild_readme)}"
        )
    return audit(docs, args.languages)


if __name__ == "__main__":
    raise SystemExit(main())
