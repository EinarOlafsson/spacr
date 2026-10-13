#!/usr/bin/env python3
"""Localize visible tutorial titles and section breadcrumbs exactly.

Tutorial prose and speech are maintained independently.  This tool limits
itself to the two catalog fields rendered as navigation chrome, reusing the
reviewed Qt app-name rows wherever that locale is supported by the desktop
application and explicit reviewed rows for Italian and Japanese.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "docs/source/_extra/tutorials/catalog"

SUPPORTED_CODES = {
    "es": "es",
    "fr": "fr",
    "hi": "hi",
    "pt-BR": "pt",
    "zh-CN": "zh_CN",
}

TITLE_OVERRIDES = {
    "es": (
        "PyPI, GitHub y conda-forge", "Instalación con Conda",
        "Instalación con pip", "Instaladores por plataforma",
        "Pantalla de inicio y navegación",
        "API de Python y flujos de trabajo sin interfaz gráfica",
        "Clasificación por visión artificial",
        "Clasificación por aprendizaje automático", "Mapas de activación",
        "Ensayo de motilidad", "Alinear y unir",
        "Corrección de iluminación", "Exploración de parámetros",
    ),
    "fr": (
        "PyPI, GitHub et conda-forge", "Installation avec Conda",
        "Installation avec pip", "Programmes d’installation par plateforme",
        "Écran d’accueil et navigation",
        "API Python et flux de travail sans interface graphique",
        "Classification par vision artificielle",
        "Classification par apprentissage automatique", "Cartes d’activation",
        "Test de motilité", "Aligner et assembler",
        "Correction de l’illumination", "Exploration des paramètres",
    ),
    "hi": (
        "PyPI, GitHub और conda-forge", "Conda से इंस्टॉलेशन",
        "pip से इंस्टॉलेशन", "प्लेटफ़ॉर्म इंस्टॉलर",
        "होम स्क्रीन और नेविगेशन", "Python API और हेडलेस वर्कफ़्लो",
        "कंप्यूटर विज़न वर्गीकरण", "मशीन लर्निंग वर्गीकरण",
        "एक्टिवेशन मैप", "गतिशीलता परीक्षण", "संरेखण और स्टिचिंग",
        "इल्यूमिनेशन सुधार", "पैरामीटर स्वीप",
    ),
    "pt-BR": (
        "PyPI, GitHub e conda-forge", "Instalação com Conda",
        "Instalação com pip", "Instaladores por plataforma",
        "Tela inicial e navegação",
        "API Python e fluxos de trabalho sem interface gráfica",
        "Classificação por visão computacional",
        "Classificação por aprendizado de máquina", "Mapas de ativação",
        "Ensaio de motilidade", "Alinhar e unir",
        "Correção de iluminação", "Varredura de parâmetros",
    ),
    "zh-CN": (
        "PyPI、GitHub 和 conda-forge", "使用 Conda 安装", "使用 pip 安装",
        "平台安装程序", "主页和导航", "Python API 和无界面工作流",
        "计算机视觉分类", "机器学习分类", "激活图", "运动性测定",
        "对齐和拼接", "照明校正", "参数扫描",
    ),
}

OVERRIDE_SOURCES = (
    "PyPI, GitHub, and conda-forge",
    "Installation with Conda",
    "Installation with pip",
    "Platform installers",
    "Home screen and navigation",
    "Python API and headless workflows",
    "Classify, computer vision",
    "Classify, machine learning",
    "Activation maps",
    "Motility assay",
    "Align and Stitch",
    "Illumination",
    "Parameter Sweep",
)

# Feature Dictionary self-registers after the base Home table is imported.
# Keep its visible tutorial title explicit here so a cold documentation build
# cannot depend on whether that late registry row was imported first.
FEATURE_DICTIONARY_TITLES = {
    "es": "Diccionario de características",
    "fr": "Dictionnaire des caractéristiques",
    "hi": "विशेषता शब्दकोश",
    "pt-BR": "Dicionário de características",
    "zh-CN": "特征词典",
}

SECTION_ROWS = {
    "es": {
        "spaCR": "spaCR", "Core": "Módulos principales",
        "Segmentation models": "Modelos de segmentación",
        "Results and quality control": "Resultados y control de calidad",
        "Toxoplasma assays": "Ensayos de Toxoplasma",
        "Data and batch runs": "Datos y ejecuciones por lotes",
        "Data": "Datos", "Explore": "Explorar", "Design": "Diseño",
    },
    "fr": {
        "spaCR": "spaCR", "Core": "Modules principaux",
        "Segmentation models": "Modèles de segmentation",
        "Results and quality control": "Résultats et contrôle qualité",
        "Toxoplasma assays": "Essais Toxoplasma",
        "Data and batch runs": "Données et traitements par lots",
        "Data": "Données", "Explore": "Exploration",
        "Design": "Planification expérimentale",
    },
    "hi": {
        "spaCR": "spaCR", "Core": "मुख्य मॉड्यूल",
        "Segmentation models": "सेगमेंटेशन मॉडल",
        "Results and quality control": "परिणाम और गुणवत्ता नियंत्रण",
        "Toxoplasma assays": "टॉक्सोप्लाज़्मा परीक्षण",
        "Data and batch runs": "डेटा और बैच रन", "Data": "डेटा",
        "Explore": "अन्वेषण", "Design": "प्रायोगिक डिज़ाइन",
    },
    "pt-BR": {
        "spaCR": "spaCR", "Core": "Módulos principais",
        "Segmentation models": "Modelos de segmentação",
        "Results and quality control": "Resultados e controle de qualidade",
        "Toxoplasma assays": "Ensaios de Toxoplasma",
        "Data and batch runs": "Dados e execuções em lote", "Data": "Dados",
        "Explore": "Explorar", "Design": "Planejamento experimental",
    },
    "zh-CN": {
        "spaCR": "spaCR", "Core": "核心模块",
        "Segmentation models": "分割模型",
        "Results and quality control": "结果与质量控制",
        "Toxoplasma assays": "弓形虫测定",
        "Data and batch runs": "数据与批量运行", "Data": "数据",
        "Explore": "探索", "Design": "实验设计",
    },
    "it": {
        "spaCR": "spaCR", "Core": "Moduli principali",
        "Segmentation models": "Modelli di segmentazione",
        "Results and quality control": "Risultati e controllo qualità",
        "Toxoplasma assays": "Saggi su Toxoplasma",
        "Data and batch runs": "Dati ed esecuzioni batch", "Data": "Dati",
        "Explore": "Esplorazione", "Design": "Progettazione sperimentale",
    },
    "ja": {
        "spaCR": "spaCR", "Core": "コアモジュール",
        "Segmentation models": "セグメンテーションモデル",
        "Results and quality control": "結果と品質管理",
        "Toxoplasma assays": "トキソプラズマアッセイ",
        "Data and batch runs": "データとバッチ実行", "Data": "データ",
        "Explore": "探索", "Design": "実験計画",
    },
}

ITALIAN_TITLES = (
    "PyPI, GitHub e conda-forge", "Installazione con Conda",
    "Installazione con pip", "Programmi di installazione per piattaforma",
    "Schermata iniziale e navigazione",
    "API Python e flussi di lavoro senza interfaccia grafica", "Maschere",
    "Misurazione", "Annotazione", "Classificazione con visione artificiale",
    "Classificazione con apprendimento automatico",
    "Mappatura dei codici a barre", "Regressione", "Creazione delle maschere",
    "UMAP delle immagini", "Mappe di attivazione", "Serie temporali",
    "Saggio di motilità", "Addestramento di Cellpose", "Maschere Cellpose",
    "Confronto dei modelli", "Raccolta dei modelli",
    "Concordanza tra annotatori", "Saggio delle placche", "Reclutamento",
    "Saggio di invasione", "Saggio di replicazione",
    "Esecuzioni di addestramento", "Rapporto", "Coda delle piastre",
    "Maschere esterne", "Allineamento e composizione",
    "Visualizzatore di piastre", "Esplorazione del database",
    "Conversione di formato", "Importazione del progetto",
    "Esecuzione in batch", "Processi distribuiti",
    "Valutazione del classificatore", "Cronologia delle esecuzioni",
    "Classificazione", "Correzione manuale", "Correzione dell’illuminazione",
    "Gestione dei dati", "Esplorazione dei progetti", "Ponte napari",
    "Controllo qualità dei codici a barre", "Elenco dei risultati",
    "Metodi e risultati", "Confronto delle esecuzioni", "Carte di controllo",
    "Grafo della pipeline", "Profilo delle predizioni",
    "Pannello di controllo qualità", "Dispersione di immagini", "Lignaggio",
    "Visualizzatore dei livelli", "Costruttore di grafi",
    "Esportazione AnnData", "PCA", "Tabulazione",
    "Dizionario delle caratteristiche", "Piccoli multipli", "Editor dei gate",
    "Esplorazione delle caratteristiche", "Valori anomali",
    "Progettazione dell’esperimento", "Potenza / progettazione",
    "Dose–risposta", "Spiegazione del modello CV", "Analisi del risultato",
    "Esplorazione del volcano plot", "Esplorazione dei parametri",
)

JAPANESE_TITLES = (
    "PyPI、GitHub、conda-forge", "Condaによるインストール",
    "pipによるインストール", "プラットフォーム別インストーラー",
    "ホーム画面とナビゲーション", "Python APIとヘッドレスワークフロー",
    "マスク生成", "測定", "アノテーション", "コンピュータービジョン分類",
    "機械学習分類", "バーコードマッピング", "回帰分析", "マスク作成",
    "画像UMAP", "活性化マップ", "タイムラプス", "運動性アッセイ",
    "Cellposeの学習", "Cellposeマスク", "モデル比較", "モデルライブラリ",
    "アノテーター間一致度", "プラークアッセイ", "リクルートメント",
    "侵入アッセイ", "複製アッセイ", "学習履歴", "レポート",
    "プレートキュー", "外部マスク", "位置合わせとスティッチ",
    "プレートビューア", "データベースブラウザー", "形式変換",
    "プロジェクトのインポート", "バッチ実行", "分散ジョブ",
    "分類器の評価", "実行履歴", "分類", "手動修正", "照明補正",
    "データ管理", "プロジェクトブラウザー", "napari連携", "バーコードQC",
    "ヒットリスト", "メソッドと結果", "実行比較", "管理図",
    "パイプライングラフ", "予測プロファイラー", "QCダッシュボード",
    "画像散布図", "系譜", "レイヤービューア", "グラフ作成",
    "AnnDataエクスポート", "PCA", "集計", "特徴量辞書",
    "スモールマルチプル", "ゲートエディター", "特徴量エクスプローラー",
    "外れ値", "実験計画", "検出力 / 設計", "用量反応",
    "CVモデルの説明", "ヒットの検証", "ボルケーノプロット探索",
    "パラメータ探索",
)


def _supported_title_maps(source: dict) -> dict[str, dict[str, str]]:
    import spacr.qt
    from spacr.qt.i18n import tr

    spacr.qt.register_self_registering_modules()
    result: dict[str, dict[str, str]] = {}
    for suffix, code in SUPPORTED_CODES.items():
        mapping = {
            lesson["id"]: tr(lesson["title"], code)
            for lesson in source["lessons"]
        }
        overrides = dict(zip(OVERRIDE_SOURCES, TITLE_OVERRIDES[suffix]))
        for lesson in source["lessons"]:
            mapping[lesson["id"]] = overrides.get(
                lesson["title"], mapping[lesson["id"]]
            )
        mapping["62_feature_dictionary"] = FEATURE_DICTIONARY_TITLES[suffix]
        result[suffix] = mapping
    return result


def main() -> int:
    source = json.loads((CATALOG_DIR / "lessons_en.json").read_text())
    ids = [lesson["id"] for lesson in source["lessons"]]
    if len(ids) != 73 or len(set(ids)) != 73:
        raise RuntimeError("review title rows before moving the 73-lesson ratchet")
    if len(ITALIAN_TITLES) != 73 or len(JAPANESE_TITLES) != 73:
        raise RuntimeError("Italian/Japanese title rows are not aligned")

    title_maps = _supported_title_maps(source)
    title_maps["it"] = dict(zip(ids, ITALIAN_TITLES))
    title_maps["ja"] = dict(zip(ids, JAPANESE_TITLES))
    for suffix, titles in title_maps.items():
        path = CATALOG_DIR / f"lessons_{suffix}.json"
        catalog = json.loads(path.read_text())
        target_ids = [lesson["id"] for lesson in catalog["lessons"]]
        if target_ids != ids:
            raise RuntimeError(f"lesson alignment differs in {path.name}")
        section_rows = SECTION_ROWS[suffix]
        localized_sections = set(section_rows.values())
        for lesson in catalog["lessons"]:
            lesson["title"] = titles[lesson["id"]]
            section = lesson["section"]
            if section in section_rows:
                lesson["section"] = section_rows[section]
            elif section not in localized_sections:
                raise RuntimeError(
                    f"unreviewed section {section!r} in {path.name}"
                )
        path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
