# JOBSIM — 직무 시뮬레이션 기반 적성 탐색 웹앱

KNOW(한국직업정보) 데이터를 기반으로 직무를 군집화하고,
사용자가 실무 미션을 수행해 5개의 사고방식 축 점수로 적성을 탐색하는 단일 HTML 웹앱입니다.

---

## 프로젝트 구조

```
job_sim/
├── data/
│   ├── raw/              # XML 원본 (K000000821~K000007583, 537개 직업 폴더)
│   └── processed/        # 파이프라인 산출물
│       ├── activities.csv         (537직업 × 41활동)
│       ├── abilities.csv          (537직업 × 44능력)
│       ├── personalities.csv      (537직업 × 16성격)
│       ├── clusters.csv           (직업별 군집 레이블)
│       ├── axis_mapping.json      (5축 ↔ 활동 항목 매핑)
│       └── cluster_weights.json   (5개 군집별 5축 가중치)
├── pipeline/
│   ├── 01_parse_xml.py   # XML → CSV 3종
│   ├── 02_clustering.py  # k-means(k=8) + 시뮬레이션 대상 선정
│   ├── 03_factor_analysis.py  # 요인분석 + 5축 매핑
│   └── 04_weights.py     # 군집별 5축 가중치 (Softmax)
├── missions/
│   └── all_missions.json  # 15개 미션 (5직업 × 3개)
├── app/
│   └── index.html        # 단일 파일 웹앱 (빌드 불필요)
└── README.md
```

---

## 의존성 (Python)

```
pip install pandas numpy scikit-learn scipy factor_analyzer
```

> Python 3.10 이상 권장. factor_analyzer 는 선택 사항(요인분석 시각화용)으로,
> 미설치 시에도 하드코딩된 축 매핑으로 동작합니다.

---

## 파이프라인 실행

> **필요 전제:** `data/raw/` 아래에 각 직업 코드별 서브 디렉토리(K000000821/ …)가
> 존재하고, 각 디렉토리에 `dtlGb_5.xml`, `dtlGb_6.xml`, `dtlGb_7.xml` 이 있어야 합니다.

```bash
cd job_sim

# 1단계: XML 파싱 → CSV 3종 생성
python pipeline/01_parse_xml.py

# 2단계: k-means 군집화 → clusters.csv
python pipeline/02_clustering.py

# 3단계: 요인분석 + 5축 매핑 → axis_mapping.json
python pipeline/03_factor_analysis.py

# 4단계: 군집별 5축 가중치 → cluster_weights.json
python pipeline/04_weights.py
```

각 단계가 완료되면 `data/processed/` 에 파일이 생성됩니다.

---

## 웹앱 실행

`app/index.html` 은 외부 라이브러리 없이 브라우저에서 바로 열 수 있습니다.

### 방법 1: 파일 직접 열기
```bash
open app/index.html        # macOS
xdg-open app/index.html    # Linux
start app/index.html       # Windows
```

### 방법 2: 로컬 HTTP 서버 (권장)
```bash
cd job_sim/app
python -m http.server 8080
# → http://localhost:8080 에서 접속
```

> 데이터(`cluster_weights.json` 등)는 `app/index.html` 에 직접 임베드되어 있으므로
> 별도 서버 없이도 동작합니다.

---

## 채점 방식

| 단계 | 내용 |
|------|------|
| 키워드 매칭 | 각 미션의 `rubric` 에 정의된 축별 키워드를 사용자 답변에서 카운트 |
| 정규화 | Sigmoid 함수로 0~1 범위 변환 |
| 가중치 적용 | 미션의 `axis_signals` 가중치를 곱해 축별 기여 점수 산출 |
| 평균 | 동일 직업의 미션 점수들을 평균 → 사용자 5축 프로파일 |
| 적합도 | 사용자 프로파일 벡터 ↔ 군집 가중치 벡터 코사인 유사도 → 0~100점 |

---

## 5개 사고방식 축

| 축 | 이름 | 핵심 활동 |
|----|------|----------|
| AX1 | 정보분석·논리 | 정보수집, 자료분석, 정보처리, 컴퓨터업무, 정보평가 |
| AX2 | 관찰·탐색 | 절차관찰, 사물파악, 새로운지식습득, 장비검사 |
| AX3 | 전략·판단 | 의사결정, 목표전략, 업무계획, 창조적생각 |
| AX4 | 리더십·조직 | 부하지시, 팀구성, 조직편성, 인사업무 |
| AX5 | 대인서비스 | 대인관계, 직접응대, 사람배려, 조언상담 |

---

## 시뮬레이션 대상 5개 군집

| 군집 ID | 군집명 | 대표 직업 | 주요 강점 축 |
|---------|-------|---------|------------|
| C1 | 리더십·인사관리 | 전략 컨설턴트 | AX4, AX5 |
| C2 | 데이터·IT | 데이터 분석가 | AX1 |
| C4 | 정보·행정서비스 | 리서치 애널리스트 | AX1, AX3 |
| C5 | 기술·현장분석 | 품질관리 엔지니어 | AX2 |
| C6 | 대인서비스 | 고객경험 매니저 | AX5 |

---

## 파이프라인 주요 지표

- 직업 수: 537개
- Hopkins 통계량: 0.74 (클러스터 경향 있음)
- Silhouette Score (k=8): 0.13
- 요인분석 Kaiser Rule 요인 수: 5개

---

## 라이선스

교육/비상업적 목적으로 활용 가능합니다.
KNOW 데이터 원본은 한국고용정보원의 이용 약관을 따릅니다.
