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
        "mask": "使用 Cellpose 为细胞、细胞核和病原体生成掩膜",
        "timelapse": "对时间序列各帧中的对象进行分割和跟踪",
        "motility": "自动化运动性实验：跟踪速度并进行感染质量控制",
        "measure": "测量单个对象的强度和形态学特征",
        "annotate": "在网格中标注单个对象图像，并保存到数据库",
        "classify": "训练 Torch CNN/Transformer，对单个对象进行分类",
        "ml_analyze": "使用经典机器学习（XGBoost、随机森林等）分析筛选特征",
        "map_barcodes": "将测序条形码映射到筛选数据",
        "regression": "对筛选评分进行回归分析",
        "align": "将图像块配准并拼接成一个画布；采用增量写入，因此无需将 20000×20000 的拼接图完整载入 RAM",
        "convert": "将 ND2/CZI/LIF/OME-TIFF 转换为 Yokogawa TIFF：先预览映射，再生成指向原始文件的映射文件",
        "foreign": "将第三方的图像、掩膜和测量表导入 spaCR 项目，并把其列映射到 spaCR 的列",
        "external_masks": "将图像和外部生成的标签掩膜转换为已完成测量、可供标注的 spaCR 项目",
        "queue": "让多个板依次通过同一处理流程",
        "batch": "将任意模块、板和设置加入队列，并在夜间自动运行",
        "distributed_jobs": "通过 SSH 工作站、Slurm 或云端/HPC 命令提交并监控 spaCR 运行",
        "db_browser": "无需 sqlite3 CLI 即可浏览和导出 measurements.db",
        "make_masks": "针对您的数据集微调 Cellpose 模型",
        "train_cellpose": "训练自定义 Cellpose 模型",
        "cellpose_masks": "生成 Cellpose 掩膜",
        "model_compare": "在相同视野上运行两个 Cellpose 模型：并排比较掩膜，以及对象计数与 ARI 的差异",
        "model_zoo": "浏览、验证并下载 Cellpose 和分类器模型，并在您的三个视野上进行基准测试",
        "plate_view": "将任意测量结果显示为板热图，并检测边缘效应",
        "agreement": "计算标注列之间的 Cohen/Fleiss κ，并复核不一致项",
        "umap": "生成以图像缩略图显示的 UMAP 嵌入",
        "activation": "生成激活图",
        "train_compare": "叠加多个训练运行的曲线，并在旁边对比其设置差异",
        "classifier_evaluation": "评估留出集预测、嵌套交叉验证、校准、数据泄漏和每板指标",
        "run_history": "搜索每项作业的设置、文件、警告、失败记录和性能",
        "report": "一键生成可共享的 HTML/PDF：包括质量控制结论、图表、统计、设置和版本信息",
        "analyze_plaques": "分析空斑实验数据",
        "recruitment": "分析募集实验数据",
        "invasion": "双色细胞外/细胞内染色：区分附着与已侵入的寄生虫，并计算每孔侵入效率",
        "replication": "内出芽生殖：按空泡统计寄生虫数量，并计算各条件的复制率",
    },
    "hi": {
        "mask": "कोशिकाओं, नाभिकों और रोगजनकों के लिए Cellpose मास्क बनाएँ",
        "timelapse": "समय-श्रृंखला के सभी फ़्रेमों में ऑब्जेक्ट का खंडन और ट्रैकिंग करें",
        "motility": "स्वचालित गतिशीलता परीक्षण: वेग ट्रैक करें और संक्रमण का गुणवत्ता नियंत्रण करें",
        "measure": "एकल ऑब्जेक्ट की तीव्रता और आकारिकी विशेषताएँ मापें",
        "annotate": "ग्रिड पर एकल-ऑब्जेक्ट छवियों को एनोटेट करें और डेटाबेस में सहेजें",
        "classify": "एकल ऑब्जेक्ट को वर्गीकृत करने के लिए Torch CNN/Transformer प्रशिक्षित करें",
        "ml_analyze": "स्क्रीनिंग फीचर्स पर पारंपरिक ML (XGBoost / random forest / …) लागू करें",
        "map_barcodes": "अनुक्रमण बारकोड को स्क्रीनिंग डेटा से मैप करें",
        "regression": "स्क्रीनिंग स्कोर का प्रतिगमन विश्लेषण करें",
        "align": "टाइलों को पंजीकृत करके एक सिले हुए कैनवास में जोड़ें; क्रमिक लेखन के कारण 20000×20000 मोज़ेक को कभी भी पूरी तरह RAM में रखने की आवश्यकता नहीं होती",
        "convert": "ND2/CZI/LIF/OME-TIFF को Yokogawa TIFF में बदलें: पहले मैपिंग का पूर्वावलोकन करें, फिर मूल फ़ाइलों से संबद्ध मैप फ़ाइल बनाएँ",
        "foreign": "दूसरे स्रोत की छवियों, मास्क और मापन तालिका को spaCR परियोजना में आयात करें तथा उनके कॉलमों को spaCR के कॉलमों से मैप करें",
        "external_masks": "छवियों और बाहरी रूप से बनाए गए लेबल मास्क को मापी हुई spaCR परियोजना में बदलें, जो एनोटेशन के लिए तैयार हो",
        "queue": "एकाधिक प्लेटों को एक ही पाइपलाइन में क्रम से चलाएँ",
        "batch": "किसी भी मॉड्यूल, प्लेट और सेटिंग को कतार में लगाकर रात भर चलाएँ",
        "distributed_jobs": "SSH वर्कस्टेशन, Slurm या cloud/HPC कमांड पर spaCR रन जमा करें और उनकी निगरानी करें",
        "db_browser": "sqlite3 CLI के बिना measurements.db ब्राउज़ और निर्यात करें",
        "make_masks": "अपने डेटासेट के लिए Cellpose मॉडल को फ़ाइन-ट्यून करें",
        "train_cellpose": "कस्टम Cellpose मॉडल प्रशिक्षित करें",
        "cellpose_masks": "Cellpose मास्क बनाएँ",
        "model_compare": "एक ही फ़ील्ड पर दो Cellpose मॉडल चलाएँ: मास्क को साथ-साथ तथा ऑब्जेक्ट-संख्या और ARI के अंतर को देखें",
        "model_zoo": "Cellpose और classifier मॉडल ब्राउज़, सत्यापित और डाउनलोड करें तथा अपने तीन फ़ील्ड पर उनका बेंचमार्क करें",
        "plate_view": "किसी भी मापन को प्लेट हीटमैप के रूप में देखें और किनारी प्रभाव पहचानें",
        "agreement": "एनोटेशन कॉलमों के बीच Cohen/Fleiss κ की गणना करें और असहमतियों की समीक्षा करें",
        "umap": "छवि ग्लिफ़ के साथ UMAP embedding बनाएँ",
        "activation": "activation map बनाएँ",
        "train_compare": "कई training run के कर्व एक-दूसरे पर चढ़ाकर दिखाएँ और उनकी सेटिंग के अंतर साथ-साथ देखें",
        "classifier_evaluation": "held-out prediction, nested CV, calibration, leakage और प्रति-प्लेट मेट्रिक्स का मूल्यांकन करें",
        "run_history": "हर जॉब की सेटिंग, फ़ाइल, चेतावनी, विफलता और प्रदर्शन खोजें",
        "report": "एक क्लिक में साझा करने योग्य HTML/PDF बनाएँ: QC निष्कर्ष, आकृतियाँ, सांख्यिकी, सेटिंग और संस्करण",
        "analyze_plaques": "प्लाक परीक्षण डेटा का विश्लेषण करें",
        "recruitment": "रिक्रूटमेंट डेटा का विश्लेषण करें",
        "invasion": "दो-रंगी बाहरी/आंतरिक अभिरंजन: संलग्न और आक्रमण कर चुके परजीवियों में अंतर करें तथा प्रति वेल आक्रमण दक्षता मापें",
        "replication": "Endodyogeny: प्रति वैक्यूल पर परजीवियों की संख्या गिनें और प्रत्येक परिस्थिति के लिए प्रतिकृति दर निर्धारित करें",
    },
    "ko": {
        "mask": "세포, 핵 및 병원체의 Cellpose 마스크를 생성합니다",
        "timelapse": "시계열의 각 프레임에서 객체를 분할하고 추적합니다",
        "motility": "자동 운동성 분석: 속도를 추적하고 감염 품질을 관리합니다",
        "measure": "개별 객체의 강도 및 형태학적 특징을 측정합니다",
        "annotate": "그리드에서 개별 객체 이미지를 주석 처리하고 데이터베이스에 저장합니다",
        "classify": "개별 객체 분류를 위한 Torch CNN/Transformer를 학습합니다",
        "ml_analyze": "스크리닝 특징에 고전적 ML(XGBoost, random forest 등)을 적용합니다",
        "map_barcodes": "시퀀싱 바코드를 스크리닝 데이터에 매핑합니다",
        "regression": "스크리닝 점수를 회귀 분석합니다",
        "align": "타일을 하나의 캔버스로 정합하고 스티칭합니다. 점진적으로 기록하므로 20000×20000 모자이크 전체를 RAM에 올릴 필요가 없습니다",
        "convert": "ND2/CZI/LIF/OME-TIFF를 Yokogawa TIFF로 변환합니다. 먼저 매핑을 미리 본 뒤 원본 파일로 연결되는 맵 파일을 생성합니다",
        "foreign": "외부의 이미지, 마스크 및 측정 테이블을 spaCR 프로젝트로 가져오고 해당 열을 spaCR 열에 매핑합니다",
        "external_masks": "이미지와 외부에서 생성한 레이블 마스크를 측정이 완료되어 주석 처리할 수 있는 spaCR 프로젝트로 변환합니다",
        "queue": "여러 플레이트를 동일한 파이프라인에서 연속으로 처리합니다",
        "batch": "원하는 모듈, 플레이트 및 설정을 대기열에 넣어 밤새 실행합니다",
        "distributed_jobs": "SSH 워크스테이션, Slurm 또는 cloud/HPC 명령으로 spaCR 실행을 제출하고 모니터링합니다",
        "db_browser": "sqlite3 CLI 없이 measurements.db를 탐색하고 내보냅니다",
        "make_masks": "데이터 세트에 맞게 Cellpose 모델을 미세 조정합니다",
        "train_cellpose": "사용자 지정 Cellpose 모델을 학습합니다",
        "cellpose_masks": "Cellpose 마스크를 생성합니다",
        "model_compare": "동일한 시야에서 두 Cellpose 모델을 실행하여 마스크를 나란히 보고 객체 수와 ARI 차이를 비교합니다",
        "model_zoo": "Cellpose 및 분류기 모델을 탐색, 검증 및 다운로드하고 사용자 시야 세 개에서 벤치마크합니다",
        "plate_view": "모든 측정값을 플레이트 히트맵으로 표시하고 가장자리 효과를 감지합니다",
        "agreement": "주석 열 간의 Cohen/Fleiss κ를 계산하고 불일치 항목을 검토합니다",
        "umap": "이미지 글리프가 포함된 UMAP 임베딩을 생성합니다",
        "activation": "활성화 맵을 생성합니다",
        "train_compare": "여러 학습 실행의 곡선을 겹쳐 표시하고 설정 차이를 나란히 비교합니다",
        "classifier_evaluation": "홀드아웃 예측, nested CV, 보정, 데이터 누출 및 플레이트별 지표를 평가합니다",
        "run_history": "모든 작업의 설정, 파일, 경고, 실패 및 성능을 검색합니다",
        "report": "한 번의 클릭으로 공유 가능한 HTML/PDF를 생성합니다. QC 판정, 그림, 통계, 설정 및 버전이 포함됩니다",
        "analyze_plaques": "플라크 분석 데이터를 분석합니다",
        "recruitment": "리크루트먼트 데이터를 분석합니다",
        "invasion": "2색 세포외/세포내 염색으로 부착된 기생충과 침입한 기생충을 구분하고 웰별 침입 효율을 산출합니다",
        "replication": "내출아법(Endodyogeny): 액포별 기생충 수를 측정하고 조건별 복제율로 산출합니다",
    },
}


_EXPECTED_LANGUAGE_CODES = {"zh_CN", "hi", "ko"}
_APP_KEY_SETS = {frozenset(summaries) for summaries in MODULE_SUMMARIES_ASIA.values()}

assert set(MODULE_SUMMARIES_ASIA) == _EXPECTED_LANGUAGE_CODES
assert len(_APP_KEY_SETS) == 1, "module-summary catalogs must contain identical app keys"
assert all(
    len(summaries) == 34 for summaries in MODULE_SUMMARIES_ASIA.values()
), "each module-summary catalog must contain exactly 34 built-in apps"

del _APP_KEY_SETS, _EXPECTED_LANGUAGE_CODES
