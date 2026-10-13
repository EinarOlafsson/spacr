#!/usr/bin/env python3
"""Apply the microscopy glossary and preserve exact in-app module labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from translate_catalog import speech_text

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog"
LANGUAGES = ("es", "fr", "hi", "it", "pt-BR", "ja", "zh-CN")

OPEN_PREFIX = {
    "es": "Abra {title} en la sección {section}.",
    "fr": "Ouvrez {title} dans la section {section}.",
    "hi": "{section} अनुभाग से {title} खोलें।",
    "it": "Apri {title} dalla sezione {section}.",
    "pt-BR": "Abra {title} na seção {section}.",
    "ja": "{section} セクションから {title} を開きます。",
    "zh-CN": "从 {section} 部分打开 {title}。",
}

GENERIC_METADATA = {
    "hi": {
        "prerequisite": "यदि इस मॉड्यूल को अपस्ट्रीम परिणाम चाहिए, तो पहले के कार्यप्रवाह ट्यूटोरियल पूरे करें।",
        "objectives": (
            "समझें कि {title} का उपयोग कब करना है।",
            "महत्वपूर्ण इनपुट और सेटिंग्स कॉन्फ़िगर करें।",
            "कार्यप्रवाह चलाएँ और उसके आउटपुट का स्थान पहचानें।",
        ),
    },
    "ja": {
        "prerequisite": "このモジュールで上流の結果が必要な場合は、先に該当するワークフローのチュートリアルを完了してください。",
        "objectives": (
            "{title} を使用する場面を理解する。",
            "重要な入力と設定を構成する。",
            "ワークフローを実行し、その出力場所を確認する。",
        ),
    },
    "zh-CN": {
        "prerequisite": "如果该模块需要上游结果，请先完成前面的相关工作流教程。",
        "objectives": (
            "了解何时使用 {title}。",
            "配置重要输入和设置。",
            "运行工作流并找到其输出位置。",
        ),
    },
}

GENERIC_PREREQUISITE = (
    "Follow the earlier workflow lessons when this module needs upstream results."
)
GENERIC_OBJECTIVES = (
    "Understand when to use {title}.",
    "Configure the important inputs and settings.",
    "Run the workflow and locate its outputs.",
)

GLOSSARY = {
    "ja": {
        "オブジェクトの作物": "オブジェクトクロップ",
        "オブジェクト作物": "オブジェクトクロップ",
        "対象作物": "オブジェクトクロップ",
        "画像作物": "画像クロップ",
        "作物画像": "クロップ画像",
        "作物覆い地図": "クロップオーバーレイマップ",
        "作物": "クロップ",
        "臓器細胞": "オルガネラ",
        "臓器": "オルガネラ",
        "マレイ": "配列",
        "測定点 db": "measurements.db",
        "訓練前の重量": "学習済み重み",
        "規則化": "正則化",
        "正常化": "正規化",
        "基礎の真実": "グラウンドトゥルース",
        "作業流": "ワークフロー",
        "作業空間": "ワークスペース",
        "機能ファミリー": "特徴量群",
        "特徴ファミリー": "特徴量群",
        "測定機能": "測定特徴量",
        "機能ベース": "特徴量ベース",
        "測定された機能": "測定特徴量",
        "属性方法": "アトリビューション手法",
        "属性強度": "アトリビューション強度",
        "配属": "アトリビューション",
        "真空体": "液胞",
        "人口": "集団",
        "門口": "ゲート",
        "井戸": "ウェル",
        "覆盖": "カバレッジ",
        "豊富な限界": "存在量しきい値",
        "平面の場": "フラットフィールド",
        "濃度特性": "輝度特徴量",
        "起源地図": "来歴マップ",
        "出身の情報": "来歴情報",
        "出産概要": "来歴概要",
        "定例画像": "標準形式の画像",
        "外国データ": "外部データ",
        "再生可能": "再現可能",
        "再起動可能": "再実行可能",
        "地図化": "マッピング",
        "輸入": "インポート",
        "輸出": "エクスポート",
        "折り畳み": "フォールド",
        "図面パネル": "図パネル",
        "診断数字": "診断図",
        "表と数字": "表と図",
        "品質管理の数字": "品質管理図",
        "数字制限": "図の上限",
        "ラベルの飛行機": "ラベル平面",
        "ラベル飛行機": "ラベル平面",
        "マスク飛行機": "マスク平面",
        "真空": "液胞",
        "列車セルポース": "Train Cellpose",
        "学習速率,増やし,バッチ,時代": "学習率,データ拡張,バッチ,エポック",
        "学習速率,正則化,配分,増や,時代": "学習率,正則化,バッチ,データ拡張,エポック",
        "学習速率": "学習率",
        "訓練前のチェックポイント": "学習済みチェックポイント",
        "パッケージされた重量": "同梱された重み",
        "汚れのチャンネル": "染色チャネル",
        "汚れのチャネル": "染色チャネル",
        "掃除機の任務": "液胞への割り当て",
        "実験根": "実験ルート",
        "排行表": "順位表",
        "基線": "ベースライン",
        "漂移": "ドリフト",
        "機動板": "ダッシュボード",
        "機能探査機": "Feature Explorer",
        "機能スケーリング": "特徴量スケーリング",
        "配分数": "寄与数",
        "重量化": "重み付け",
        "地図が正しく": "マッピングが正しく",
        "直接推算されたマッピング": "推定されたマッピング",
        "推算されたマッピング": "推定されたマッピング",
        "図面の測定を図り": "測定値をプロットし",
        "アーテファクト": "アーティファクト",
        "ソース取得": "元の取得データ",
        "校准": "較正",
        "基礎真実": "グラウンドトゥルース",
        "分類者": "分類器",
        "機能の準備": "特徴量の準備",
        "選択された機能": "選択された特徴量",
        "種子": "シード",
        "試算予算": "試行予算",
        "交叉認証": "クロスバリデーション",
        "維持された分割": "ホールドアウト分割",
        "分類元": "分類器",
        "実行の実行": "ホールドアウト",
        "レーンCV": "Train CV",
        "レーンXG": "Train XG",
        "スケーミング": "スキーム",
        "時代,早期停止": "エポック,早期停止",
        "訓練された分類器": "学習済み分類器",
        "高い信頼性,低い信頼性": "高信頼度,低信頼度",
        "豊富な量 限界": "存在量しきい値",
        "エンドプラズマ網膜": "小胞体",
        "内プラズマ網膜": "小胞体",
        "カーネルの部分": "Core セクション",
        "図書館": "ライブラリ",
        "検出 限界値": "検出しきい値",
        "限界値": "しきい値",
        "流れ合意": "フロー整合性",
        "訓練ランス": "Training Runs",
        "トレーニングランス": "Training Runs",
        "熱地図": "ヒートマップ",
        "数値機能": "数値特徴量",
        "数値特征": "数値特徴量",
        "特征": "特徴量",
        "機能をランク": "特徴量をランク",
        "機能を準備": "特徴量を準備",
        "機能範囲": "特徴量の範囲",
        "買収前": "取得前",
        "画面ヒット": "スクリーニングヒット",
        "画面結果": "スクリーニング結果",
        "プレック画像": "プラーク画像",
        "制御測定": "対照測定",
        "前向きの合意": "前景一致度",
        "横に横たわる重叠": "並列オーバーレイ",
        "重叠": "重なり",
        "基礎的な真実": "グラウンドトゥルース",
        "背景と否定": "Background and Denoising",
        "強度清掃": "強度ノイズ除去",
        "維持されたパフォーマンス": "ホールドアウト性能",
        "目的地": "出力先",
        "地図": "マップ",
        "授業": "クラス",
        "証拠": "エビデンス",
    },
    "hi": {
        "वस्तु फसलों": "ऑब्जेक्ट क्रॉप",
        "वस्तु फसल": "ऑब्जेक्ट क्रॉप",
        "छवि फसल": "इमेज क्रॉप",
        "फसल छवियों": "क्रॉप इमेज",
        "फसलों": "क्रॉप",
        "फसल": "क्रॉप",
        "माप बिंदु db": "measurements.db",
        "मूल सत्य": "ग्राउंड-ट्रुथ",
        "कलाकृतियों": "आर्टिफ़ैक्ट",
        "कलाकृति": "आर्टिफ़ैक्ट",
        "टिप्पणी स्तंभ": "एनोटेशन कॉलम",
        "टिप्पणी परतें": "एनोटेशन लेयर",
        "टिप्पणी": "एनोटेशन",
        "कुएं": "वेल",
        "कुओं": "वेल",
        "एक कुएं": "एक वेल",
        "पकड़ से बाहर": "होल्ड-आउट",
        "सिर रहित": "हेडलेस",
        "मापी गई सुविधाओं": "मापे गए फीचर",
        "माप सुविधाओं": "मापन फीचर",
        "सुविधाओं परिवारों": "फीचर समूहों",
        "सुविधाओं": "फीचर",
        "विशेषताएं": "फीचर",
        "रोगज़नक़": "रोगजनक",
        "वैक्यूलल्स": "वैक्यूल",
        "वैक्यूलल": "वैक्यूल",
        "वैक्यूओल": "वैक्यूल",
        "बीज बोने": "सीडिंग",
    },
    "zh-CN": {
        "脑膜网膜": "内质网",
        "内质网膜": "内质网",
        "对象级别的功能": "对象级特征",
        "对象级功能": "对象级特征",
        "功能家庭": "特征组",
        "功能扩展": "特征缩放",
        "功能探险器": "Feature Explorer",
        "测量功能": "测量特征",
        "功能准备": "特征准备",
        "选定的功能": "选定的特征",
        "数值功能": "数值特征",
        "图像作物": "图像裁剪图",
        "物体作物": "对象裁剪图",
        "对象作物": "对象裁剪图",
        "作物图像": "裁剪图像",
        "作物图库": "裁剪图库",
        "作物框架": "裁剪框",
        "作物": "裁剪图",
        "机器人": "细胞器",
        "真空细胞": "液泡",
        "真空体": "液泡",
        "真空": "液泡",
        "污染道": "染色通道",
        "染色道": "染色通道",
        "收获的道": "采集通道",
        "图像道": "图像通道",
        "输入道": "输入通道",
        "注册道": "配准通道",
        "频道": "通道",
        "掩膜飞机": "掩膜图层",
        "掩膜平面": "掩膜图层",
        "标签飞机": "标签图层",
        "标签平面": "标签图层",
        "衍生的隔离器": "派生区室",
        "隔离器": "区室",
        "测量点 db": "measurements.db",
        "测量点db": "measurements.db",
        "列车Cellpose": "Train Cellpose",
        "列车CV": "Train CV",
        "列车XG": "Train XG",
        "编程简历": "Classify CV",
        "学习速度": "学习率",
        "规律化": "正则化",
        "正常化": "归一化",
        "预训练的重量": "预训练权重",
        "已训练的重量": "已训练权重",
        "批量,时代": "批次,训练轮数",
        "时代和早期停止": "训练轮数和早停",
        "平面场": "平场",
        "属性批量": "归因批次",
        "属性验证": "归因验证",
        "属性方法": "归因方法",
        "属性强度": "归因强度",
        "空间文物": "空间伪影",
        "衍生文物": "派生文件",
        "陈旧的文物": "过期的分析产物",
        "转移的文物": "已传输的分析产物",
        "文物库存": "分析产物清单",
        "来源地图": "来源映射图",
        "来源地地图": "来源映射图",
        "条码地图": "条码映射",
        "地图深度": "映射深度",
        "地图运行": "映射运行",
        "控制地图": "对照映射",
        "图书馆": "文库",
        "板块": "板",
        "井口": "孔",
        "井底": "孔",
        "井标识符": "孔标识符",
        "一个井": "一个孔",
        "井": "孔",
        "过器": "过滤器",
        "过和排序": "过滤和排序",
        "过图表": "过滤图表",
        "收购": "采集",
        "源获取": "原始采集数据",
        "控制器": "对照",
        "人口门": "群体门",
        "人口": "群体",
        "丰富度门": "丰度阈值",
        "丰富度": "丰度",
        "屏幕结果": "筛选结果",
        "屏幕击中": "筛选命中项",
        "屏幕视图器": "Plate Viewer",
        "高温地图": "热图",
        "父母和儿童类": "父对象和子对象类别",
        "控制关系": "包含关系",
        "稀缺存储": "稀疏存储",
        "类调用": "类别判定",
        "延期的指标": "留出集指标",
        "已保留的指标": "留出集指标",
        "持久分离": "留出集划分",
        "板源和工作流量": "板数据源与工作流",
        "分区化掩膜": "分割掩膜",
        "分区分部分": "分割部分",
        "排行榜": "排名表",
        "面具": "掩膜",
        "口罩": "掩膜",
        " 道": " 通道",
    },
    "fr": {
        "section du noyau": "section Core",
    },
}


def replace_glossary(value: str, language: str) -> str:
    for wrong, correct in GLOSSARY.get(language, {}).items():
        value = value.replace(wrong, correct)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES)
    )
    parser.add_argument(
        "--lessons", nargs="+", default=None,
        help="Polish only the listed lesson ids; the default is every lesson.",
    )
    args = parser.parse_args()
    requested_lessons = set(args.lessons) if args.lessons else None
    english = json.loads((CATALOG / "lessons_en.json").read_text())
    english_by_id = {lesson["id"]: lesson for lesson in english["lessons"]}
    for language in args.languages:
        path = CATALOG / f"lessons_{language}.json"
        localized = json.loads(path.read_text())
        for series in localized["series"]:
            series["title"] = replace_glossary(series["title"], language)
        for lesson in localized["lessons"]:
            if requested_lessons is not None and lesson["id"] not in requested_lessons:
                continue
            source = english_by_id[lesson["id"]]
            # Identity and routing are generated structure, never translated
            # prose. Copy them from English on every polish pass so a new
            # fold cannot leave otherwise reviewed languages pointing at a
            # retired Home tile.
            for key in ("number", "slug", "series", "app_key"):
                lesson[key] = source[key]
            if "host_app_key" in source:
                lesson["host_app_key"] = source["host_app_key"]
            else:
                lesson.pop("host_app_key", None)
            # These strings are literal UI labels; translating them makes the
            # tutorial disagree with the application users are looking at.
            lesson["title"] = source["title"]
            lesson["section"] = source["section"]
            for key in ("description", "prerequisite"):
                lesson[key] = replace_glossary(lesson[key], language)
            lesson["objectives"] = [replace_glossary(item, language)
                                    for item in lesson["objectives"]]
            generic = GENERIC_METADATA.get(language)
            if (
                generic
                and source["prerequisite"] == GENERIC_PREREQUISITE
                and source["objectives"] == [
                    item.format(title=source["title"])
                    for item in GENERIC_OBJECTIVES
                ]
            ):
                lesson["prerequisite"] = generic["prerequisite"]
                lesson["objectives"] = [
                    item.format(title=source["title"])
                    for item in generic["objectives"]
                ]
            for index, scene in enumerate(lesson["scenes"]):
                scene["narration"] = replace_glossary(scene["narration"], language)
                # Generic five-scene module lessons use an abstract ``input``
                # frame and need an explicit navigation sentence. Captured
                # lessons already show the open module and often use scene 2
                # for a real settings category; prepending "Open …" there
                # makes the narration disagree with the highlighted control.
                if index == 1 and source["scenes"][index]["visual"] == "input":
                    prefix = OPEN_PREFIX[language].format(
                        title=source["title"], section=source["section"])
                    if scene["narration"].startswith(prefix):
                        scene["narration"] = scene["narration"][len(prefix):].strip()
                    # Earlier catalogs carried a translated navigation sentence
                    # here.  Every maintained catalog has now been normalized to
                    # ``prefix`` above, so removing an arbitrary first sentence
                    # would instead discard real reviewed instructions whenever
                    # the English scene contains two sentences.
                    if not scene["narration"].startswith(prefix):
                        scene["narration"] = f"{prefix} {scene['narration']}"
                scene["speech_text"] = speech_text(scene["narration"], language)
        path.write_text(json.dumps(localized, indent=2, ensure_ascii=False) + "\n")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
