### 핵심 워크플로

기본 워크플로는 6개 모듈로 구성됩니다:

- <strong>Mask</strong> Cellpose로 세포, 핵, 병원체 및 세포소기관을 분할합니다.
- <strong>Measure</strong> 형태, 강도, 텍스처, 공간 및 공위치 특성과 객체 크롭을 SQLite에 저장합니다.
- <strong>Annotate</strong> 키보드로 조작하는 격자에서 크롭에 라벨을 지정하고 능동 학습 대기열을 지원합니다.
- <strong>Classify</strong> 이미지 또는 측정값 기반 모델을 학습하고 각 체크포인트에 홀드아웃 데이터 성능을 기록합니다.
- <strong>Map Barcodes</strong> FASTQ 리드를 웰과 gRNA에 매핑하고 풍부도, 충돌 및 커버리지 QC를 제공합니다.
- <strong>Regression</strong> 연속형, 비율형 및 계수형 반응에 적합한 모델 계열로 가이드, 유전자, 조건 및 대조군 효과를 추정합니다.
