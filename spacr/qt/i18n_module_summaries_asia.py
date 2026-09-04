"""Asian-language summaries for every built-in spaCR application module.

``MODULE_SUMMARIES_ASIA`` is indexed first by spaCR language code and then by
the stable application key from :data:`spacr.qt.app.APPS`.  Its values are
localized versions of the concise, one-line English descriptions shown for
those applications.  Technical names and interoperability identifiers remain
unchanged so that they agree with the UI, documentation, and file formats.
"""
from __future__ import annotations

from typing import Dict

MODULE_SUMMARIES_ASIA: Dict[str, Dict[str, str]] = {
    "zh_CN": {
        "mask": "使用 Cellpose 及支持的替代方法，从显微图像中生成细胞、细胞核、病原体和细胞器的分割掩膜",
        "timelapse": "对时间序列各帧中的对象进行分割和跟踪",
        "motility": "量化轨迹速度和直线度，并按感染状态对结果分层",
        "measure": "定量分析每个对象的强度和形态学特征",
        "annotate": "为单对象图像指定标注，并将标注存储在项目数据库中",
        "classify_merged": "使用 PyTorch 在图像裁剪上训练分类器，或使用梯度提升在测量特征上训练分类器",
        "map_barcodes": "将测序条形码映射到筛选数据",
        "regression": "对筛选评分进行回归分析",
        "align": "配准并拼接图像块，生成增量写入且内存使用受限的镶嵌图",
        "convert": "将 ND2、CZI、LIF 和 OME-TIFF 图像转换为 Yokogawa TIFF 布局，并记录源文件映射",
        "foreign": '将图像、掩膜和测量表导入 spaCR 项目——转换显微镜格式、映射源列，或采用在其他工具中生成的掩膜',
        "external_masks": "将图像和外部生成的标签掩膜导入为已测量且可供标注的 spaCR 项目",
        "queue": "在多个板上执行同一处理流程",
        "batch": "将模块、板和设置加入队列，以便无人值守地顺序执行",
        "distributed_jobs": "通过 SSH 工作站、Slurm 或云端/HPC 命令提交并监控 spaCR 运行",
        "db_browser": "浏览、筛选并导出 measurements.db 中的表",
        "model_compare": "在相同视野上比较两个 Cellpose 模型的并排掩膜、对象计数差异和调整 Rand 指数（ARI）",
        "model_zoo": "浏览、验证并下载 Cellpose 和分类器模型，并在选定视野上进行基准测试",
        "plate_view": "将测量结果可视化为板热图，并检测边缘效应",
        "agreement": "计算各标注列的 Cohen 或 Fleiss κ，并复核标注不一致的图像裁剪",
        "umap": "使用图像缩略图可视化 UMAP 嵌入",
        "activation": "为图像分类器的预测生成类激活图",
        "train_compare": "比较多个训练运行的曲线和设置",
        "classifier_evaluation": "评估留出集预测、嵌套交叉验证、校准、数据泄漏和每板指标",
        "run_history": "搜索运行设置、输出、警告、失败记录和性能指标",
        "report": "生成包含质量控制结果、图表、统计、设置和软件版本的可共享 HTML 或 PDF 报告",
        "analyze_plaques": "定量分析空斑实验测量结果",
        "recruitment": "定量分析分子募集测量结果",
        "invasion": "使用双色差异染色定量附着和已侵入的寄生虫，并计算每孔侵入效率",
        "replication": "定量每个空泡中的寄生虫数量，并计算各条件的复制率",
    },
    "hi": {
        "mask": "Cellpose और समर्थित वैकल्पिक विधियों से माइक्रोस्कोपी छवियों में कोशिकाओं, नाभिकों, रोगजनकों और अंगकों के विभाजन मास्क बनाएँ",
        "timelapse": "समय-श्रृंखला के सभी फ़्रेमों में ऑब्जेक्ट का खंडन और ट्रैकिंग करें",
        "motility": "ट्रैक वेग और सीधापन परिमाणित करें तथा संक्रमण स्थिति के अनुसार परिणामों को स्तरीकृत करें",
        "measure": "प्रत्येक ऑब्जेक्ट की तीव्रता और आकारिकी विशेषताओं का परिमाण निर्धारित करें",
        "annotate": "एकल-ऑब्जेक्ट छवियों को एनोटेशन दें और उन्हें परियोजना डेटाबेस में सहेजें",
        "classify_merged": "छवि क्रॉप पर PyTorch या मापी गई विशेषताओं पर ग्रेडिएंट बूस्टिंग से वर्गीकारक प्रशिक्षित करें",
        "map_barcodes": "अनुक्रमण बारकोड को स्क्रीनिंग डेटा से मैप करें",
        "regression": "स्क्रीनिंग स्कोर का प्रतिगमन विश्लेषण करें",
        "align": "सीमित मेमोरी उपयोग के साथ क्रमिक रूप से लिखे जाने वाले मोज़ेक में छवि टाइलों को पंजीकृत और संयोजित करें",
        "convert": "ND2, CZI, LIF और OME-TIFF छवियों को Yokogawa TIFF विन्यास में बदलें और स्रोत फ़ाइल मैपिंग दर्ज करें",
        "foreign": 'छवियों, मास्क और माप तालिकाओं को spaCR प्रोजेक्ट में लाएँ — माइक्रोस्कोप प्रारूप बदलकर, स्रोत कॉलम मैप करके, या कहीं और बनाए गए मास्क अपनाकर',
        "external_masks": "छवियों और बाहरी लेबल मास्क को मापी हुई, एनोटेशन के लिए तैयार spaCR परियोजना के रूप में आयात करें",
        "queue": "एक ही प्रसंस्करण पाइपलाइन को अनेक प्लेटों पर चलाएँ",
        "batch": "मॉड्यूल, प्लेट और सेटिंग को बिना निगरानी क्रमिक निष्पादन के लिए कतार में लगाएँ",
        "distributed_jobs": "SSH वर्कस्टेशन, Slurm या cloud/HPC कमांड पर spaCR रन जमा करें और उनकी निगरानी करें",
        "db_browser": "measurements.db की तालिकाएँ ब्राउज़, फ़िल्टर और निर्यात करें",
        "model_compare": "समान फ़ील्ड पर दो Cellpose मॉडलों की साथ-साथ मास्क, ऑब्जेक्ट-संख्या के अंतर और समायोजित Rand सूचकांक (ARI) के आधार पर तुलना करें",
        "model_zoo": "Cellpose और classifier मॉडल ब्राउज़, सत्यापित और डाउनलोड करें तथा चयनित फ़ील्ड पर उनका बेंचमार्क करें",
        "plate_view": "मापों को प्लेट हीटमैप के रूप में प्रदर्शित करें और किनारी प्रभाव पहचानें",
        "agreement": "एनोटेशन कॉलमों में Cohen या Fleiss κ की गणना करें और असंगत एनोटेशन वाले क्रॉप की समीक्षा करें",
        "umap": "छवि ग्लिफ़ के साथ UMAP एम्बेडिंग प्रदर्शित करें",
        "activation": "छवि वर्गीकारक के पूर्वानुमानों के लिए क्लास सक्रियण मानचित्र बनाएँ",
        "train_compare": "अनेक प्रशिक्षण रन के वक्र और सेटिंग की तुलना करें",
        "classifier_evaluation": "held-out prediction, nested CV, calibration, leakage और प्रति-प्लेट मेट्रिक्स का मूल्यांकन करें",
        "run_history": "रन सेटिंग, आउटपुट, चेतावनी, विफलता और प्रदर्शन मेट्रिक खोजें",
        "report": "QC परिणाम, आकृतियाँ, सांख्यिकी, सेटिंग और सॉफ़्टवेयर संस्करण वाली साझा करने योग्य HTML या PDF रिपोर्ट बनाएँ",
        "analyze_plaques": "प्लाक परीक्षण के मापों का परिमाण निर्धारित करें",
        "recruitment": "आणविक रिक्रूटमेंट के मापों का परिमाण निर्धारित करें",
        "invasion": "दो-रंगी विभेदी अभिरंजन से संलग्न और आक्रमण कर चुके परजीवियों का परिमाण निर्धारित करें तथा प्रति वेल आक्रमण दक्षता की गणना करें",
        "replication": "प्रति वैक्यूल परजीवियों का परिमाण निर्धारित करें और प्रत्येक परिस्थिति के लिए प्रतिकृति दर की गणना करें",
    },
    "ko": {
        "mask": "Cellpose 및 지원되는 대체 방법을 사용하여 현미경 이미지에서 세포, 핵, 병원체 및 소기관의 분할 마스크를 생성합니다",
        "timelapse": "시계열의 각 프레임에서 객체를 분할하고 추적합니다",
        "motility": "트랙 속도와 직진성을 정량화하고 감염 상태별로 결과를 층화합니다",
        "measure": "객체별 강도 및 형태학적 특징을 정량화합니다",
        "annotate": "개별 객체 이미지에 주석을 할당하고 프로젝트 데이터베이스에 저장합니다",
        "classify_merged": "이미지 크롭에는 PyTorch를, 측정된 특징에는 그래디언트 부스팅을 사용하여 분류기를 학습합니다",
        "map_barcodes": "시퀀싱 바코드를 스크리닝 데이터에 매핑합니다",
        "regression": "스크리닝 점수를 회귀 분석합니다",
        "align": "메모리 사용량을 제한하면서 점진적으로 기록되는 모자이크로 이미지 타일을 정합하고 스티칭합니다",
        "convert": "ND2, CZI, LIF 및 OME-TIFF 이미지를 Yokogawa TIFF 레이아웃으로 변환하고 원본 파일 매핑을 기록합니다",
        "foreign": '이미지, 마스크, 측정 표를 spaCR 프로젝트로 가져옵니다 — 현미경 형식을 변환하거나, 원본 열을 매핑하거나, 다른 곳에서 만든 마스크를 채택합니다',
        "external_masks": "이미지와 외부 레이블 마스크를 측정 및 주석 처리가 가능한 spaCR 프로젝트로 가져옵니다",
        "queue": "동일한 처리 파이프라인을 여러 플레이트에서 실행합니다",
        "batch": "모듈, 플레이트 및 설정을 무인 순차 실행을 위해 대기열에 추가합니다",
        "distributed_jobs": "SSH 워크스테이션, Slurm 또는 cloud/HPC 명령으로 spaCR 실행을 제출하고 모니터링합니다",
        "db_browser": "measurements.db 테이블을 탐색, 필터링 및 내보냅니다",
        "model_compare": "동일한 시야에서 두 Cellpose 모델을 나란히 표시한 마스크, 객체 수 차이 및 조정 Rand 지수(ARI)로 비교합니다",
        "model_zoo": "Cellpose 및 분류기 모델을 탐색, 검증 및 다운로드하고 선택한 시야에서 벤치마크합니다",
        "plate_view": "측정값을 플레이트 히트맵으로 시각화하고 가장자리 효과를 감지합니다",
        "agreement": "주석 열에 대해 Cohen 또는 Fleiss κ를 계산하고 주석이 일치하지 않는 크롭을 검토합니다",
        "umap": "이미지 글리프를 사용하여 UMAP 임베딩을 시각화합니다",
        "activation": "이미지 분류기 예측을 위한 클래스 활성화 맵을 생성합니다",
        "train_compare": "여러 학습 실행의 곡선과 설정을 비교합니다",
        "classifier_evaluation": "홀드아웃 예측, nested CV, 보정, 데이터 누출 및 플레이트별 지표를 평가합니다",
        "run_history": "실행 설정, 출력, 경고, 실패 및 성능 지표를 검색합니다",
        "report": "QC 결과, 그림, 통계, 설정 및 소프트웨어 버전을 포함하는 공유 가능한 HTML 또는 PDF 보고서를 생성합니다",
        "analyze_plaques": "플라크 분석 측정값을 정량화합니다",
        "recruitment": "분자 리크루트먼트 측정값을 정량화합니다",
        "invasion": "2색 감별 염색으로 부착된 기생충과 침입한 기생충을 정량화하고 웰별 침입 효율을 계산합니다",
        "replication": "액포별 기생충 수를 정량화하고 조건별 복제율을 계산합니다",
    },
}


_EXPECTED_LANGUAGE_CODES = {"zh_CN", "hi", "ko"}
_APP_KEY_SETS = {frozenset(summaries) for summaries in MODULE_SUMMARIES_ASIA.values()}

assert set(MODULE_SUMMARIES_ASIA) == _EXPECTED_LANGUAGE_CODES
assert len(_APP_KEY_SETS) == 1, "module-summary catalogs must contain identical app keys"
assert all(
    len(summaries) == 30 for summaries in MODULE_SUMMARIES_ASIA.values()
), "each reviewed module-summary catalog must contain exactly 30 built-in apps"

del _APP_KEY_SETS, _EXPECTED_LANGUAGE_CODES
