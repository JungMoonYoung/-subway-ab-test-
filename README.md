# 지하철 이용자 행동 기반 경로 추천 A/B 테스트 시스템

## 프로젝트 개요

본 프로젝트는 지하철 이용자의 경로 선택 행동을 분석하기 위한 **A/B 테스트 기반 시뮬레이션 시스템**입니다.

- **목적**: UI 디자인(빠름 중심 vs 편안함 중심)이 사용자의 경로 선택에 미치는 영향 분석
- **데이터**: 100,000명 × 5회 반복 = 500,000 rows의 합성 데이터
- **방법론**: 로지스틱 회귀 모델, 반복 측정 통계 분석

## 주요 기능

- 사용자 특성 기반 경로 선택 시뮬레이션
- 학습 효과(Learning Effect) 모델링
- 통계 검정 (Z-test, Chi-square, Mixed-Effects Model, GEE)
- 시각화 및 자동 리포트 생성
- (선택) Streamlit 대시보드

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd subway-ab-test
```

### 2. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

## 실행 방법

### DAY 1: 사용자 데이터 생성

```bash
python data/generate_users.py
```

**산출물**: `data/users_base.csv` (100,000명 사용자 정보)

### DAY 2: 반복 측정 시뮬레이션 (예정)

```bash
python data/simulate_trials.py
```

### DAY 3: 선택 행동 모델링 (예정)

데이터 생성 완료 후 실행

### DAY 4-6: 분석 및 리포트 (예정)

통계 분석, 시각화, 대시보드 구현

## 프로젝트 구조

```
subway-ab-test/
├── data/                  # 데이터 생성 모듈
│   ├── generate_users.py  # 사용자 기본 정보 생성
│   └── users_base.csv     # 생성된 사용자 데이터
├── analysis/              # 통계 분석 모듈
├── notebooks/             # Jupyter Notebooks
├── dashboard/             # Streamlit 대시보드
├── figures/               # 시각화 결과
├── config.py              # 파라미터 설정
├── requirements.txt       # 의존성 패키지
├── SRS.MD                 # 요구사항 명세서
├── PLAN.MD                # 구현 계획서
└── README.md              # 본 문서
```

## 주요 파라미터 (config.py)

- **NUM_USERS**: 100,000 (사용자 수)
- **NUM_TRIALS**: 5 (반복 측정 횟수)
- **RANDOM_SEED**: 42 (재현성 보장)
- **로지스틱 회귀 계수**:
  - BETA_0_A = 0.3 (A그룹 절편)
  - BETA_TIME_PRESSURE = 1.2
  - BETA_PREVIOUS_CHOICE = -0.4 (학습 효과)

## 현재 진행 상황

- [x] DAY 1: 프로젝트 세팅 및 사용자 데이터 생성 (완료)
- [ ] DAY 2: 반복 측정 시뮬레이션
- [ ] DAY 3: 선택 행동 모델링
- [ ] DAY 4: 통계 분석
- [ ] DAY 5: 시각화 및 리포트
- [ ] DAY 6: 최종 정리 및 대시보드

## 기술 스택

- **언어**: Python 3.8+
- **데이터 처리**: Pandas, NumPy
- **통계 분석**: SciPy, Statsmodels, Scikit-learn
- **시각화**: Matplotlib, Seaborn, Plotly
- **대시보드**: Streamlit (선택)

## 문서

- **SRS.MD**: 요구사항 명세서 (상세 설계 문서)
- **PLAN.MD**: 일자별 구현 계획서 (DAY 1~6)

## 라이선스

MIT License

## 문의

프로젝트 관련 문의사항은 이슈로 남겨주세요.

초반에 진행했을때 FAST의 비율이 98프로가 넘는 형태로 진행되어서 무의미 하다고 판단하여 파라미터 값을 조정하여 학습의 의미가 있도록 진행
