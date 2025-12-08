# PLAN — 지하철 A/B 테스트 시스템 구현 계획서

**버전:** 2.0 (Day-based)
**프로젝트 시작일:** 2025-10-01
**작성일:** 2025-10-01
**문서 유형:** Implementation Plan

---

## 1. 프로젝트 개요

본 문서는 SRS.MD에 명시된 요구사항을 기반으로, **지하철 이용자 행동 기반 경로 추천 A/B 테스트 시스템**의 구현 계획을 일자별로 정리한 실행 계획서이다.

**프로젝트 기간:** 2025년 10월부터 진행

### 1.1 목표

- 100,000명 사용자 × 5회 반복 = 500,000 rows의 합성 데이터 생성
- 로지스틱 회귀 기반 선택 행동 시뮬레이션
- 반복 측정 데이터에 적합한 통계 분석 수행
- 결과 시각화 및 인사이트 도출
- (선택) Streamlit 대시보드 구현

### 1.2 개발 방향

- **모듈화**: 데이터 생성, 분석, 시각화를 독립적인 모듈로 분리
- **재현 가능성**: seed 기반 랜덤 제어
- **확장 가능성**: 사용자 수, 회차 수 조정 용이
- **파라미터 중앙 관리**: config.py를 통한 모든 계수 관리

---

## 2. 프로젝트 구조

```
subway-ab-test/
│
├── data/                          # 데이터 생성 모듈
│   ├── generate_users.py          # 사용자 기본 정보 생성
│   ├── simulate_trials.py         # 5회 반복 시뮬레이션
│   └── synthetic_data.parquet     # 최종 생성 데이터
│
├── analysis/                      # 통계 분석 모듈
│   ├── basic_tests.py             # Z-test, Chi-square, Effect size
│   ├── mixed_models.py            # Mixed-Effects, GEE
│   ├── visualization.py           # 차트 생성
│   └── report_generator.py        # 분석 결과 리포트 생성
│
├── notebooks/                     # Jupyter Notebooks
│   ├── 01_data_generation.ipynb   # 데이터 생성 실험
│   ├── 02_exploratory_analysis.ipynb  # EDA
│   ├── 03_statistical_tests.ipynb # 통계 검정
│   └── 04_final_report.ipynb      # 최종 결과 정리
│
├── dashboard/                     # Streamlit 대시보드 (선택)
│   └── streamlit_app.py
│
├── config.py                      # 파라미터 설정 (β 계수, seed 등)
├── requirements.txt               # 의존성 패키지
├── SRS.MD                         # 요구사항 명세서
├── PLAN.MD                        # 본 문서 (구현 계획서)
└── README.md                      # 프로젝트 설명
```

---

## 3. 일자별 개발 계획

### DAY 1: 프로젝트 세팅 및 사용자 데이터 생성

**목표**: 개발 환경 구축 및 100,000명 사용자 기본 정보 생성

#### 오전 (3시간)

**1. 프로젝트 환경 설정**
- 프로젝트 디렉토리 구조 생성
- `requirements.txt` 작성
  - 필수 패키지: pandas, numpy, scipy, statsmodels, scikit-learn
  - 시각화: matplotlib, seaborn, plotly
  - 선택: streamlit
- 가상환경 생성 및 패키지 설치
- Git 초기화 (선택)

**2. config.py 작성**
- 시뮬레이션 기본 설정
  - NUM_USERS = 100000
  - NUM_TRIALS = 5
  - RANDOM_SEED = 42
- 로지스틱 회귀 계수 정의
  - BETA_0_A = 0.3, BETA_0_B = -0.2
  - BETA_TIME_PRESSURE = 1.2
  - BETA_PERSONALITY = 0.9
  - BETA_TIME_DIFF = 0.45
  - BETA_PREVIOUS_CHOICE = -0.4
  - NOISE_STD = 0.4
- 경로 시간/혼잡도 분포 파라미터
- 사용자 특성 비율 정의

#### 오후 (4시간)

**3. generate_users.py 구현**
- 사용자 ID 생성 (1~100,000)
- A/B 그룹 랜덤 배정 (Bernoulli 0.5)
- 사용자 특성 할당
  - `personality_type`: efficiency(55%), comfort(35%), neutral(10%)
  - `travel_frequency`: daily(60%), weekly(30%), rarely(10%)
- DataFrame 생성 및 검증

**4. 데이터 검증 및 시각화**
- 결측값, 중복 확인
- 그룹 배정 균형 확인 (A/B 각 50,000명)
- 사용자 특성 분포 히스토그램 생성
- `users_base.csv` 저장

**5. README.md 초안 작성**
- 프로젝트 개요
- 설치 방법
- 실행 방법

#### 산출물
- [ ] 프로젝트 디렉토리 구조
- [ ] `config.py`
- [ ] `requirements.txt`
- [ ] `data/generate_users.py`
- [ ] `users_base.csv` (100,000 rows)
- [ ] 사용자 특성 분포 차트
- [ ] `README.md` 초안

---

### DAY 2: 반복 측정 시뮬레이션 구현

**목표**: 각 사용자당 5회 반복 측정 데이터 생성 (선택 행동 제외)

#### 오전 (4시간)

**1. simulate_trials.py 기본 구조 작성**
- `users_base.csv` 로드
- 사용자별 5회 반복 루프 구현
- 각 trial마다 기본 정보 생성
  - `trial_number` (1~5)
  - `days_since_first` (0~4)
  - `created_at` (날짜 시뮬레이션)

**2. time_pressure 생성 로직**
- 개인별 baseline tendency 할당
  - 정규분포 기반, 평균=1, 표준편차=0.5
- 회차별 랜덤 변동 추가
- 최종값: 0/1/2 중 하나로 변환
- 검증: 전체 분포 확인 (0/1/2 비율)

#### 오후 (3시간)

**3. 경로 시간 및 혼잡도 샘플링**
- 각 trial마다 확률적으로 생성
  - `route_time_fast` ~ N(25, 2²)
  - `route_time_relax` ~ N(36, 3²)
  - `congestion_fast` ~ N(145, 15²)
  - `congestion_relax` ~ N(75, 10²)
- 시간-혼잡도 상관관계 반영
  - `actual_time = base_time + (congestion - 100) × 0.08`
- 범위 검증 (시간 > 0, 혼잡도 > 0)

**4. 중간 데이터 검증**
- 샘플 사용자 5명의 5회 데이터 출력
- 결측값 확인
- 분포 시각화 (경로 시간, 혼잡도)
- `trials_data_partial.csv` 저장 (선택 행동 제외)

#### 산출물
- [ ] `data/simulate_trials.py` (부분 완성)
- [ ] `trials_data_partial.csv` (500,000 rows, 선택 컬럼 제외)
- [ ] time_pressure, 경로 시간, 혼잡도 분포 차트

---

### DAY 3: 선택 행동 모델링 및 완전한 데이터 생성

**목표**: 로지스틱 회귀 기반 경로 선택 및 학습 효과 구현

#### 오전 (4시간)

**1. 로지스틱 선택 모델 구현**
- 로지스틱 함수 작성
  ```
  logit = β0[group] + β1·time_pressure + β2·personality
          + β3·time_difference + β4·previous_choice + ε
  prob_fast = 1 / (1 + exp(-logit))
  ```
- config.py에서 계수 불러오기
- personality_type 인코딩 (efficiency=1, comfort=-1, neutral=0)
- time_difference 계산 (절대값)

**2. 학습 효과 구현**
- trial_number=1: `previous_choice = None`, β4 항 = 0
- trial_number≥2: 이전 회차 선택 참조
- 사용자별 순차 처리 구현
- Fast/Relax 선택 결과 생성

#### 오후 (3시간)

**3. 만족도 및 의사결정 시간 생성**
- `satisfaction_score` 로직
  - 기본 점수: 3.0
  - 매칭 보너스: 성향과 선택 일치 시 +1~2
  - 압박 패널티: 급한데 Relax 선택 시 -1
  - 랜덤 노이즈 ~ N(0, 0.5²)
  - 최종 클리핑: 0~5
- `decision_time` 생성
  - 기본: 3~8초 (정규분포)
  - time_pressure 영향 반영
  - 랜덤 변동 추가

**4. 최종 데이터 생성 및 검증**
- 모든 컬럼 포함 DataFrame 생성
- 재현성 테스트 (동일 seed로 2회 실행)
- `synthetic_data.parquet` 저장
- `synthetic_data.csv` 저장 (호환성)
- 데이터 무결성 검사
  - 결측값 없음
  - 범위 검증 (satisfaction 0~5, decision_time > 0)
  - 논리적 일관성 (trial_number=1일 때 previous_choice=None)

**5. 01_data_generation.ipynb 작성**
- DAY 1~3의 데이터 생성 과정 정리
- 중간 결과 시각화
- 샘플 데이터 출력 및 해석

#### 산출물
- [ ] `data/simulate_trials.py` (완성)
- [ ] `synthetic_data.parquet` (500,000 rows)
- [ ] `synthetic_data.csv`
- [ ] 재현성 테스트 결과
- [ ] `notebooks/01_data_generation.ipynb`

---

### DAY 4: 통계 분석 구현 (기초 검정)

**목표**: 기초 통계 검정 및 반복 측정 모델링 구현

#### 오전 (4시간)

**1. basic_tests.py 구현**
- Two-Proportion Z-Test
  - A그룹 vs B그룹 Fast Route 선택 비율 비교
  - p-value, z-statistic 계산
- Chi-square Test
  - 그룹(A/B) × 선택(Fast/Relax) 교차표
  - 독립성 검정
- Effect Size (Cohen's h)
  - 비율 차이의 실질적 크기
  - small/medium/large 해석
- Confidence Interval
  - A/B 그룹별 Fast 비율 95% CI

**2. 결과 검증 및 정리**
- 검정 결과를 딕셔너리 또는 DataFrame으로 구조화
- 결과 해석 및 의미 분석
- 주요 지표 추출 (선택률 차이, p-value, effect size)

#### 오후 (3시간)

**3. mixed_models.py 구현**
- Mixed-Effects Logistic Regression
  - 패키지: statsmodels.regression.mixed_linear_model
  - 모델 구조: `선택 ~ group + time_pressure + personality + trial_number + (1 | user_id)`
  - 고정효과 및 랜덤효과 추정
- GEE (Generalized Estimating Equations)
  - 패키지: statsmodels.genmod.GEE
  - AR(1) correlation structure 적용
  - 모델 적합 및 계수 해석

**4. 다중 비교 보정**
- FDR (Benjamini-Hochberg) 적용
- 패키지: statsmodels.stats.multitest.multipletests
- 보정 전/후 p-value 비교 테이블 생성

**5. 02_exploratory_analysis.ipynb 작성**
- EDA (Exploratory Data Analysis)
  - 변수별 분포 확인
  - 그룹별 기술통계
  - 상관관계 분석
- 초기 패턴 발견 및 기록

#### 산출물
- [ ] `analysis/basic_tests.py`
- [ ] `analysis/mixed_models.py`
- [ ] 통계 검정 결과 DataFrame/딕셔너리
- [ ] FDR 보정 결과 테이블
- [ ] `notebooks/02_exploratory_analysis.ipynb`

---

### DAY 5: 시각화 및 분석 리포트 생성

**목표**: 결과 시각화 및 종합 분석 리포트 작성

#### 오전 (4시간)

**1. visualization.py 구현**
- A vs B 선택 비율 비교 (Bar Plot)
  - X축: 그룹(A/B)
  - Y축: Fast Route 선택 비율(%)
  - 에러바: 95% CI
- 사용자 특성별 Breakdown
  - Facet plot: personality_type별 선택 비율
  - Heatmap: time_pressure × personality 조합별 선택 비율
- 회차별 행동 변화 (시계열)
  - X축: trial_number (1~5)
  - Y축: Fast Route 선택 비율
  - 그룹별 라인 (A/B)
  - 목적: 학습 효과 시각화

**2. 추가 시각화**
- 로지스틱 회귀 계수 플롯
  - X축: 변수명
  - Y축: 계수 크기 (β)
  - 에러바: 95% CI
- 혼잡도-시간 관계 산점도
  - X축: congestion_rate
  - Y축: actual_time
  - 회귀선 표시
- 모든 차트를 `figures/` 디렉토리에 PNG/SVG 저장

#### 오후 (3시간)

**3. report_generator.py 구현**
- 모든 검정 결과를 Markdown 형식으로 정리
- 주요 섹션:
  - 요약 (Executive Summary)
  - 연구 질문별 답변 (RQ1, RQ2, RQ3)
  - 통계 검정 결과 (표 포함)
  - 시각화 차트 임베드
  - 주요 인사이트 자동 추출
    - "A그룹이 B그룹보다 Fast 선택률이 X%p 높음 (p<0.05)"
    - "time_pressure가 1 증가할 때 Fast 선택 확률 Y% 증가"
  - 제언 및 결론
- `statistical_report.md` 생성

**4. 03_statistical_tests.ipynb 작성**
- DAY 4의 모든 통계 검정 실행
- 결과 해석 및 시각화
- 가설 검정 결과 정리

#### 산출물
- [ ] `analysis/visualization.py`
- [ ] `figures/` 디렉토리 (최소 5개 차트)
- [ ] `analysis/report_generator.py`
- [ ] `statistical_report.md`
- [ ] `notebooks/03_statistical_tests.ipynb`

---

### DAY 6: 최종 정리 및 대시보드 (선택)

**목표**: 프로젝트 최종 정리 및 Streamlit 대시보드 구현 (선택)

#### 오전 (3시간)

**1. 04_final_report.ipynb 작성**
- 전체 프로젝트 요약
  - 프로젝트 목표 및 배경
  - 데이터 생성 과정 요약
  - 통계 분석 결과 종합
- 연구 질문 답변
  - RQ1: UI 효과 검증
  - RQ2: 사용자 속성 영향
  - RQ3: 학습 효과 존재 여부
- 주요 인사이트 및 제언
  - 단순 결론
  - 전략적 결론
  - 제품 적용 제안
- 한계점 및 향후 연구 방향

**2. README.md 완성**
- 프로젝트 개요 상세 작성
- 설치 및 실행 방법
- 파일 구조 설명
- 주요 결과 요약
- 라이선스 및 기여 방법 (선택)

#### 오후 (4시간) - 선택 사항

**3. Streamlit 대시보드 구현 (선택)**
- `dashboard/streamlit_app.py` 작성
- 필수 기능
  - 사이드바: 그룹 선택(A/B/All), 필터링 옵션
  - KPI 카드
    - A그룹 Fast 선택률
    - B그룹 Fast 선택률
    - p-value
    - Effect size
  - 비교 차트
    - A vs B bar plot
    - 회차별 line plot
  - 사용자 특성별 breakdown table
- 선택 기능 (시간 허용 시)
  - 파라미터 슬라이더 (β 계수 조정)
  - "재시뮬레이션" 버튼
  - 인터랙티브 차트 (Plotly)

**4. 통합 테스트**
- 전체 워크플로우 재실행
  - 데이터 생성 → 분석 → 시각화 → 리포트
- 재현성 재확인 (동일 seed)
- 모든 Jupyter Notebook 실행 확인
- (선택) Streamlit 앱 테스트

#### 산출물
- [ ] `notebooks/04_final_report.ipynb`
- [ ] `README.md` (완성)
- [ ] (선택) `dashboard/streamlit_app.py`
- [ ] 통합 테스트 결과

---

## 4. 일정 요약

| Day | 주요 작업                          | 핵심 산출물                                    | 예상 시간 |
|-----|------------------------------------|------------------------------------------------|-----------|
| 1   | 환경 설정 및 사용자 데이터 생성    | config.py, generate_users.py, users_base.csv   | 7시간     |
| 2   | 반복 측정 시뮬레이션               | simulate_trials.py (부분), trials_data_partial.csv | 7시간     |
| 3   | 선택 행동 모델링 및 데이터 완성    | synthetic_data.parquet, 01_data_generation.ipynb | 7시간     |
| 4   | 통계 분석 구현                     | basic_tests.py, mixed_models.py, 02_exploratory_analysis.ipynb | 7시간     |
| 5   | 시각화 및 리포트 생성              | visualization.py, statistical_report.md, 03_statistical_tests.ipynb | 7시간     |
| 6   | 최종 정리 및 대시보드 (선택)       | 04_final_report.ipynb, README.md, streamlit_app.py | 7시간     |
| **총계** |                                | **프로젝트 완료**                              | **42시간** |

**참고**:
- 각 Day는 순수 개발 시간 기준 7시간 (오전 3-4시간, 오후 3-4시간)
- 실제 소요 시간은 디버깅, 학습, 휴식 포함 시 1.5배 증가 가능 (약 10시간/일)
- DAY 6의 Streamlit 대시보드는 선택 사항 (시간 부족 시 생략 가능)

---

## 5. 핵심 파라미터 정리 (config.py 구조)

```
config.py 구조 (코드 없이 개념만)

[시뮬레이션 기본 설정]
- NUM_USERS = 100000
- NUM_TRIALS = 5
- RANDOM_SEED = 42

[로지스틱 회귀 계수]
- BETA_0_A = 0.3   (A그룹 절편)
- BETA_0_B = -0.2  (B그룹 절편)
- BETA_TIME_PRESSURE = 1.2
- BETA_PERSONALITY = 0.9
- BETA_TIME_DIFF = 0.45
- BETA_PREVIOUS_CHOICE = -0.4
- NOISE_STD = 0.4

[경로 시간 분포]
- FAST_TIME_MEAN = 25분
- FAST_TIME_STD = 2분
- RELAX_TIME_MEAN = 36분
- RELAX_TIME_STD = 3분

[혼잡도 분포]
- FAST_CONGESTION_MEAN = 145%
- FAST_CONGESTION_STD = 15%
- RELAX_CONGESTION_MEAN = 75%
- RELAX_CONGESTION_STD = 10%
- DELAY_FACTOR = 0.08

[사용자 특성 비율]
- PERSONALITY_EFFICIENCY_RATIO = 0.55
- PERSONALITY_COMFORT_RATIO = 0.35
- PERSONALITY_NEUTRAL_RATIO = 0.10
- TRAVEL_DAILY_RATIO = 0.60
- TRAVEL_WEEKLY_RATIO = 0.30
- TRAVEL_RARELY_RATIO = 0.10

[통계 검정 설정]
- ALPHA = 0.05
- POWER_TARGET = 0.80
- FDR_METHOD = "fdr_bh"  (Benjamini-Hochberg)
```

---

## 6. 테스트 및 검증 계획

### 6.1 데이터 검증 (매일 수행)

각 Day 완료 시 다음 항목 확인:

**데이터 무결성**
- 결측값 없음
- 범위 검증 (예: satisfaction_score 0~5)
- 중복 레코드 없음

**분포 적합성**
- 히스토그램으로 시각 확인
- Kolmogorov-Smirnov 검정 (정규성)

**논리적 일관성**
- `trial_number=1`일 때 `previous_choice=None`
- `days_since_first = trial_number - 1`
- A그룹 사용자는 모든 trial에서 A그룹 유지

### 6.2 모델 검증 (DAY 3)

**선택 확률 범위**
- 0 ≤ P(Fast) ≤ 1 확인
- 극단적 케이스 테스트

**학습 효과 확인**
- 샘플 사용자 5명의 5회 선택 패턴 수동 검토
- β4가 음수이므로, Fast 선택 후 Relax 확률 증가 확인

### 6.3 통계 검정 결과 해석 (DAY 4-5)

- p-value가 예상 범위 내인지 (너무 작거나 크지 않은지)
- Effect size가 설정한 Power Analysis와 일치하는지

### 6.4 재현성 테스트 (DAY 3, DAY 6)

- 동일 seed로 2회 실행 시 완전히 동일한 데이터 생성 확인
- seed 변경 시 다른 데이터 생성 확인

---

## 7. 리스크 및 대응 방안

| 리스크                              | 영향도 | 대응 방안                                  | 발생 시점 |
|-------------------------------------|--------|--------------------------------------------|-----------|
| 500,000 rows 처리 시 메모리 부족    | 중     | Parquet 포맷 사용, 청크 단위 처리          | DAY 2-3   |
| Mixed-Effects Model 수렴 실패       | 중     | 초기값 조정, GEE로 대체                    | DAY 4     |
| 통계적 유의성 미달성 (p > 0.05)     | 중     | Effect size 분석으로 보완, 탐색적 분석 강조| DAY 4-5   |
| 학습 효과 모델링 복잡도 증가        | 중     | 단순 선형 모델 유지, 향후 확장 계획 명시   | DAY 3     |
| Streamlit 대시보드 개발 시간 부족   | 하     | DAY 6를 선택 사항으로 처리                 | DAY 6     |
| 일정 지연                           | 중     | DAY 6(대시보드) 생략, 핵심 기능에 집중     | DAY 1-5   |

---

## 8. 성공 기준 (Acceptance Criteria)

프로젝트 완료 시 다음 조건을 모두 만족해야 한다:

**필수 항목**:
- [ ] 500,000 rows 합성 데이터 생성 완료
- [ ] Parquet 및 CSV 포맷으로 저장
- [ ] 로지스틱 회귀 모델 정상 작동 (학습 효과 포함)
- [ ] Two-Proportion Z-Test, Chi-square, Mixed-Effects Model 결과 산출
- [ ] 주요 가설(H1, H2) 검정 완료 (p-value 계산)
- [ ] 최소 5개 이상 시각화 차트 생성
- [ ] 분석 리포트 (Markdown) 작성
- [ ] Jupyter Notebook 4개 작성 (data, EDA, tests, report)
- [ ] 동일 seed로 재현 가능 확인
- [ ] README.md 완성

**선택 항목**:
- [ ] Streamlit 대시보드 기본 기능 구현
- [ ] PDF 형식 리포트 추가 생성
- [ ] 인터랙티브 차트 (Plotly) 추가

---

## 9. 향후 확장 계획

본 프로젝트 완료 후, 다음 확장이 가능하다:

### 9.1 모델 고도화
- **Non-linear learning effect**: 회차가 증가할수록 학습 속도 감소 모델링
- **교호작용 항 추가**: 그룹 × time_pressure 교호작용
- **베이지안 모델링**: 불확실성 정량화

### 9.2 데이터 확장
- **3개 이상 경로**: 다중 선택지 모델 (Multinomial Logit)
- **실시간 정보 업데이트**: 중간에 정보가 변하는 시나리오
- **외부 변수**: 날씨, 요일, 시간대 효과

### 9.3 실제 적용
- **실제 사용자 A/B 테스트 설계**: 본 시뮬레이션을 기반으로 실제 실험 설계
- **Dynamic UI**: 시간대별 UI 자동 전환 시스템
- **개인화 추천**: 사용자 이력 기반 경로 추천

---

## 10. 참고 자료

### 10.1 기술 문서
- Statsmodels Documentation (Mixed-Effects, GEE)
- Pandas Performance Optimization
- Parquet File Format Specification

### 10.2 통계 방법론
- "Repeated Measures Design" (Keppel & Wickens)
- "Applied Longitudinal Analysis" (Fitzmaurice et al.)
- "A/B Testing: The Most Powerful Way to Turn Clicks Into Customers" (Siroker & Koomen)

### 10.3 행동경제학
- Kahneman, D. (2011). "Thinking, Fast and Slow"
- Ben-Akiva & Lerman (1985). "Discrete Choice Analysis"

---

## 11. 문서 이력

| 버전 | 날짜       | 변경 내용                     | 작성자 |
|------|------------|-------------------------------|--------|
| 1.0  | 2025-10-01 | 최초 작성 (Phase 기반)        | -      |
| 2.0  | 2025-10-15 | Day 기반으로 재구성           | -      |

---

## 12. 부록: Day별 체크리스트

### DAY 1 체크리스트
- [ ] 프로젝트 디렉토리 구조 생성
- [ ] requirements.txt 작성 및 패키지 설치
- [ ] config.py 작성 (모든 파라미터 포함)
- [ ] generate_users.py 구현
- [ ] users_base.csv 생성 (100,000 rows)
- [ ] 사용자 특성 분포 확인 (히스토그램)
- [ ] 그룹 배정 균형 검증 (A/B 각 50,000명)
- [ ] README.md 초안 작성

### DAY 2 체크리스트
- [ ] simulate_trials.py 기본 구조 작성
- [ ] 5회 반복 루프 구현
- [ ] time_pressure 생성 로직 구현 및 검증
- [ ] 경로 시간 샘플링 (Fast/Relax)
- [ ] 혼잡도 샘플링 (Fast/Relax)
- [ ] 시간-혼잡도 상관관계 반영
- [ ] trials_data_partial.csv 생성 (선택 행동 제외)
- [ ] 샘플 사용자 5명 데이터 출력 확인
- [ ] 분포 시각화 (시간, 혼잡도)

### DAY 3 체크리스트
- [ ] 로지스틱 함수 구현
- [ ] β 계수 적용 (config.py에서 로드)
- [ ] personality_type 인코딩
- [ ] time_difference 계산
- [ ] previous_choice 처리 (첫 trial None)
- [ ] 학습 효과 반영 (β4)
- [ ] 선택 결과 생성 (Fast/Relax)
- [ ] satisfaction_score 생성 로직
- [ ] decision_time 생성 로직
- [ ] synthetic_data.parquet 생성
- [ ] synthetic_data.csv 생성
- [ ] 재현성 테스트 (동일 seed, 2회 실행)
- [ ] 데이터 무결성 검사 (결측값, 범위, 논리)
- [ ] 01_data_generation.ipynb 작성

### DAY 4 체크리스트
- [ ] basic_tests.py 구현
  - [ ] Two-Proportion Z-Test
  - [ ] Chi-square Test
  - [ ] Effect Size (Cohen's h)
  - [ ] Confidence Interval
- [ ] 검정 결과 구조화 (DataFrame/dict)
- [ ] mixed_models.py 구현
  - [ ] Mixed-Effects Logistic Regression
  - [ ] GEE with AR(1)
  - [ ] 모델 요약 결과 생성
- [ ] 다중 비교 보정 (FDR)
- [ ] 보정 전/후 p-value 비교 테이블
- [ ] 02_exploratory_analysis.ipynb 작성
  - [ ] EDA 수행
  - [ ] 변수별 분포 확인
  - [ ] 상관관계 분석

### DAY 5 체크리스트
- [ ] visualization.py 구현
  - [ ] A vs B bar plot
  - [ ] 사용자 특성별 breakdown (facet plot, heatmap)
  - [ ] 회차별 시계열 line plot
  - [ ] 로지스틱 계수 plot
  - [ ] 혼잡도-시간 산점도
- [ ] 모든 차트 figures/ 디렉토리 저장
- [ ] report_generator.py 구현
  - [ ] Markdown 형식 리포트 구조 작성
  - [ ] 통계 검정 결과 삽입
  - [ ] 시각화 차트 임베드
  - [ ] 인사이트 자동 추출
- [ ] statistical_report.md 생성
- [ ] 03_statistical_tests.ipynb 작성
  - [ ] 모든 통계 검정 실행
  - [ ] 결과 해석 및 시각화

### DAY 6 체크리스트
- [ ] 04_final_report.ipynb 작성
  - [ ] 전체 프로젝트 요약
  - [ ] 연구 질문 답변 (RQ1, RQ2, RQ3)
  - [ ] 주요 인사이트 및 제언
  - [ ] 한계점 및 향후 연구 방향
- [ ] README.md 완성
  - [ ] 프로젝트 개요
  - [ ] 설치 및 실행 방법
  - [ ] 파일 구조 설명
  - [ ] 주요 결과 요약
- [ ] (선택) streamlit_app.py 구현
  - [ ] 기본 KPI 대시보드
  - [ ] 비교 차트
  - [ ] 필터링 기능
  - [ ] (선택) 파라미터 조정 및 재시뮬레이션
- [ ] 통합 테스트
  - [ ] 전체 워크플로우 재실행
  - [ ] 재현성 재확인
  - [ ] 모든 Jupyter Notebook 실행 확인
  - [ ] (선택) Streamlit 앱 테스트

---

**문서 종료**
