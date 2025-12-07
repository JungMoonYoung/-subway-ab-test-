# SRS — 지하철 이용자 행동 기반 경로 추천 A/B 테스트 시스템

**버전:** 1.0 Final
**작성일:** 2025-12-04
**문서 유형:** Software Requirements Specification

---

## 윤리적 고지

본 실험은 실제 사용자 데이터를 포함하지 않으며, 모든 데이터는 시뮬레이션 기반 **합성 데이터(Synthetic Data)**입니다.
실제 사용자 실험이 없으므로 IRB 승인이 필요하지 않습니다.

---

## 1. 프로젝트 개요

### 1.1 프로젝트 목적

본 프로젝트는 지하철 이용자의 이동 목적 및 상황에 따라 경로 선택 행동이 어떻게 달라지는지 검증하기 위해 설계된 **A/B 테스트 기반 시뮬레이션 시스템**이다.

사용자는 A안(더 빠르지만 혼잡) 또는 B안(느리지만 여유 있음) UI를 제공받으며, 본 연구는 각 UI가 사용자 행동 데이터(경로 선택, 선택 확률, 상황 변수 영향)에 미치는 효과를 분석하는 것이 목표이다.

### 1.2 문제 정의

현실의 지하철 이용 환경에서 이용자는 다음과 같은 선택 상황에 직면한다:

- "혼잡하지만 빠른 길을 갈 것인가?"
- "느리지만 여유롭고 덜 붐비는 길을 갈 것인가?"

본 프로젝트는 **UI 제공 방식, 정보 설계, 의사결정 압박**이 행동에 미치는 영향을 실험 설계 기반 데이터 분석 형태로 검증한다.

---

## 2. 목표 및 범위

### 2.1 연구 질문 (Research Questions)

| ID  | 연구 질문                                                                 |
|-----|---------------------------------------------------------------------------|
| RQ1 | UI 추천 방식(A vs B)이 사용자 경로 선택 행동에 의미 있는 영향을 주는가?  |
| RQ2 | 사용자 속성(시간 여유도, 이용 목적, 성향 등)이 선택 행동의 주요 예측 요인인가? |
| RQ3 | 반복 실험 환경에서 학습 효과가 존재하는가?                                |

### 2.2 가설 설정

| ID  | 가설(Hypothesis)                                                          | 방향성              |
|-----|---------------------------------------------------------------------------|---------------------|
| H1  | 빠른 경로(A안 UI)를 제공받은 사용자 그룹은 B 그룹보다 빠른 경로 선택률이 높을 것이다. | Positive            |
| H2  | 시간 여유가 없을수록 빠른 경로 선택 확률이 증가할 것이다.                 | Positive            |
| H3  | 이전 선택 경험이 다음 선택에 영향을 미친다 (학습 효과).                   | Exploratory         |

---

## 3. 실험 설계

### 3.1 사용자 배정 방식

- **사용자 총 100,000명 시뮬레이션**
- **랜덤 배정**
  - A그룹: 50,000명 → "빠름 중심" UI
  - B그룹: 50,000명 → "편안함 중심" UI
- **배정 방식**: Bernoulli Randomization (0.5 비율)

### 3.2 반복 측정 설계 (Longitudinal Design)

본 실험은 **반복 측정(Repeated Measures)** 기반으로 설계되었다.

| 항목            | 값                     |
|-----------------|------------------------|
| 측정 주기       | 1일 1회 선택(가정)     |
| 총 반복 횟수    | 5회                    |
| 데이터 유형     | 시계열(Time-series)    |
| 최종 데이터 크기 | 500,000 rows (100K users × 5 trials) |

**각 사용자는 동일한 그룹(A 또는 B) UI를 5회 반복 경험하며**, 매 회차마다 선택 행동이 기록된다.

### 3.3 KPI 정의

#### Primary KPI

| KPI 종류       | 지표                      | 설명                                   |
|----------------|---------------------------|----------------------------------------|
| Primary KPI    | Route Conversion Rate (RCR) | Fast Route 선택 비율 (Trial-level / User-level) |

- **Trial-level RCR**: 각 trial에서 Fast Route 선택 비율
- **User-level RCR**: 사용자별 5회 중 Fast Route 선택 비율(%)

#### Secondary KPI

| KPI 종류       | 지표                  | 설명                          |
|----------------|-----------------------|-------------------------------|
| Secondary KPI  | Satisfaction Score    | 선택 이후 기대 만족도(0~5)    |
| Supporting KPI | Decision Time(sec)    | 선택에 걸린 시간              |
| Exploratory KPI| Learning Effect       | 반복 실험 시 행동 변화 패턴   |

---

## 4. 경로 정의 및 모델링

### 4.1 경로 특성

| 변수           | Fast Route (A Group UI)      | Relax Route (B Group UI)     |
|----------------|------------------------------|------------------------------|
| 평균 소요시간  | 22~28분 (정규분포)           | 32~40분 (정규분포)           |
| 기본 혼잡도    | 120–170% (정규분포)          | 60–90% (정규분포)            |

**시간-혼잡도 상관관계 모델:**

```
actual_time = base_time + (congestion_rate - 100%) × 0.08
```

- `delay_factor = 0.08`
- 혼잡도가 150%일 경우: (150 - 100) × 0.08 = 4분 추가 지연

**예시:**
- Fast Route 기본 25분 + 혼잡도 160% → 실제 약 29.8분
- Relax Route 기본 36분 + 혼잡도 75% → 실제 약 34분

### 4.2 UI 표시 방식

**A안 (효율 중심 UI) 예시:**
```
🚄 24분 | 혼잡도 🔴 매우 붐빔 (160%)
```

**B안 (편안함 중심 UI) 예시:**
```
😌 36분 | 혼잡도 🟢 여유로움 (75%)
```

### 4.3 UI 차이점 상세

| 항목           | A안 (효율 중심 UI)             | B안 (편안함 중심 UI)           |
|----------------|--------------------------------|--------------------------------|
| 메인 강조 정보 | ⏱ 시간(큰 폰트)                | 😊 쾌적도(퍼센트/아이콘)       |
| 색상           | 파란색·주황색 대비             | 연녹색 계열                    |
| 경로 순서      | Fast Route → Relax Route       | Relax Route → Fast Route       |
| CTA 버튼       | "빠르게 가기"                  | "편하게 가기"                  |

---

## 5. 데이터 모델링

### 5.1 데이터 구조 (Schema)

| 컬럼명              | 타입         | 설명                                      |
|---------------------|--------------|-------------------------------------------|
| user_id             | INT          | 사용자 식별값                             |
| assigned_group      | STR          | A 또는 B                                  |
| personality_type    | STR          | efficiency-oriented / comfort-oriented / neutral |
| travel_frequency    | STR          | daily, weekly, rarely                     |
| **trial_number**    | INT (1~5)    | 반복 측정 회차                            |
| time_pressure       | INT (0–2)    | 0=급함, 1=보통, 2=여유 (회차별 변동)     |
| **previous_choice** | STR          | Fast / Relax / None (첫 회차)             |
| **days_since_first**| INT (0~4)    | 첫 측정 이후 경과일                       |
| route_time_fast     | FLOAT        | 해당 회차 Fast Route 소요시간             |
| route_time_relax    | FLOAT        | 해당 회차 Relax Route 소요시간            |
| congestion_fast     | FLOAT        | Fast Route 혼잡도(%)                      |
| congestion_relax    | FLOAT        | Relax Route 혼잡도(%)                     |
| selected_route      | STR          | Fast or Relax                             |
| decision_time       | FLOAT        | 행동 시간(초)                             |
| satisfaction_score  | INT (0–5)    | 선택 직후 기대 만족도                     |
| created_at          | DATETIME     | 기록 시점                                 |

### 5.2 변수 특성

#### 고정 변수 (사용자당 1번 할당, 5회 동일)
- `personality_type`
- `assigned_group`
- `travel_frequency`

#### 변동 변수 (회차별로 변화)
- `time_pressure` (일부 개인 경향성 + 랜덤 요소)
- `route_time_fast`, `route_time_relax` (확률 샘플링)
- `congestion_fast`, `congestion_relax`
- `decision_time`, `satisfaction_score`

### 5.3 선택 행동 확률 모델 (Logistic Regression)

사용자의 Fast Route 선택 확률은 다음 로지스틱 회귀 모델로 결정된다:

```
P(Fast Route | X) = sigmoid(β0 + β1·time_pressure + β2·personality
                            + β3·time_difference + β4·previous_choice + ε)
```

**파라미터 정의:**

| 변수                | 계수 (β) | 설명                                        |
|---------------------|----------|---------------------------------------------|
| β0 (Intercept)      | 그룹별 상이 | A그룹: +0.3 / B그룹: -0.2                  |
| β1 (time_pressure)  | +1.2     | 급할수록 Fast 선택 증가                     |
| β2 (personality)    | +0.9     | efficiency-oriented = 1, comfort = -1, neutral = 0 |
| β3 (time_difference)| +0.45    | 시간 절약 폭이 클수록 선택 증가             |
| β4 (previous_choice)| -0.4     | 학습/회피 효과 (이전 Fast 선택 → 다음 Relax 증가) |
| ε (noise)           | N(0, 0.4²) | 개인별 무작위 변동                         |

**time_difference 정의:**
```
time_difference = abs(route_time_fast - route_time_relax)
```

### 5.4 만족도(Satisfaction Score) 생성 로직

선택 직후 기대 만족도는 다음 요인으로 결정된다:

```
satisfaction = base_score
               + match_bonus (성향과 선택이 일치하면 +1~2점)
               + pressure_penalty (급한데 Relax 선택 시 -1점)
               + random_noise ~ N(0, 0.5)
```

최종 점수는 0~5 범위로 클리핑된다.

### 5.5 학습 효과 (Learning Effect)

- **β4 = -0.4**: 이전에 Fast Route를 선택했다면, 다음 회차에서 Relax Route 선택 확률이 증가한다.
- **첫 번째 trial (trial_number=1)**에서는 `previous_choice = None`이며, β4 항이 0으로 처리된다.
- 2회차부터 이전 선택이 반영되어, 사용자는 경험을 통해 행동을 조정한다.

---

## 6. 통계 분석 방법

### 6.1 기초 통계 검정

| 분석 방법              | 목적                              |
|------------------------|-----------------------------------|
| Two-Proportion Z-Test  | A vs B 그룹 간 선택 비율 차이 검정 |
| Chi-square Test        | 범주형 변수 간 독립성 검정        |
| Effect Size (Cohen's h)| 실질적 효과 크기 측정             |
| Confidence Interval    | 추정값의 신뢰 구간                |

### 6.2 반복 측정 전용 분석 (핵심)

반복 측정 데이터의 특성상, 단순 검정으로는 부족하므로 다음 기법을 적용한다:

| 분석 방법                        | 목적                                       |
|----------------------------------|--------------------------------------------|
| Mixed-Effects Logistic Regression| 사용자별 랜덤 효과(Random Intercept) 고려  |
| GEE (Generalized Estimating Equations) | 반복 측정 독립 가정 완화, AR(1) correlation |
| Repeated Measures ANOVA          | 보조 비교용 (연속형 변수 분석)            |

**Mixed-Effects Model 구조:**
```
선택 ~ 고정효과(group, time_pressure, personality, trial_number)
       + (1 | user_id)  ← 사용자별 랜덤 절편
```

**GEE Correlation Structure:**
- AR(1) (Autoregressive) 상관 구조 사용
- 인접한 trial 간 상관관계 반영

### 6.3 다중 비교 보정 (Multiple Testing Correction)

RQ1, RQ2, RQ3를 모두 검정하므로, **FDR (False Discovery Rate)** 보정을 적용한다.

- **방법**: Benjamini-Hochberg Correction
- **목표**: Family-wise error rate 제어

### 6.4 통계적 유의성 기준

| 기준          | 값       |
|---------------|----------|
| 유의수준 (α)  | 0.05     |
| 검정력 (Power)| ≥ 0.80   |
| 효과 크기     | Cohen's h ≥ 0.05 (작은 효과) |

### 6.5 Power Analysis 근거

**표본 수 100,000명**은 다음 기준으로 산출되었다:

- **목표 검정력**: 0.80 이상
- **예상 효과 크기**: Cohen's h ≈ 0.05~0.1 (작은 효과 가정)
- **계산 도구**: `statsmodels.stats.power.NormalIndPower`
- **결과**: 필요 최소 샘플 수 ≈ 84,500명
- **최종 설정**: 100,000명 (검정력 0.90 확보)

---

## 7. 기능 요구사항

### 7.1 데이터 생성 모듈

| ID     | 요구사항                                | 상태 |
|--------|-----------------------------------------|------|
| FR-01  | 사용자 100,000명 랜덤 생성              | 필수 |
| FR-02  | 그룹별 UI 할당 알고리즘 구현            | 필수 |
| FR-03  | 5회 반복 측정 시뮬레이션                | 필수 |
| FR-04  | 선택 행동 확률 기반 생성 (로지스틱)     | 필수 |
| FR-05  | 학습 효과 반영 (previous_choice)        | 필수 |
| FR-06  | CSV/Parquet export 기능                 | 필수 |
| FR-07  | Seed 기반 재현 가능성 보장              | 필수 |

### 7.2 분석 모듈

| 기능                  | 세부 내용                                    |
|-----------------------|----------------------------------------------|
| 기초 통계 검정        | Two-Proportion Z-Test, Chi-square            |
| 효과 검정             | Effect Size (Cohen's h), Confidence Interval |
| 반복 측정 모델링      | Mixed-Effects Model, GEE                     |
| 다중 비교 보정        | FDR (Benjamini-Hochberg)                     |
| 로지스틱 회귀         | 변수별 영향력 분석                           |

### 7.3 시각화

| 시각화 요소                  | 예시                              |
|------------------------------|-----------------------------------|
| 비교 bar plot                | A vs B 선택 비율                  |
| 사용자 특성별 breakdown      | personality, time_pressure별 분석 |
| 회차별 행동 변화 (시계열)    | Trial 1~5 선택 패턴 변화          |
| 로지스틱 회귀 계수 plot      | 행동 영향 요인 시각화             |
| 혼잡도 히트맵                | 변수별 선택 패턴                  |

### 7.4 Streamlit 대시보드 (Optional)

#### 필수 기능
- A/B 선택 결과 비교 시각화
- 사용자 특성별 breakdown 분석
- KPI Dashboard

#### 선택 기능
- 변수(weight) 조정하여 재시뮬레이션
- 회차별 행동 변화 애니메이션
- 인사이트 자동 텍스트 출력

---

## 8. 비기능 요구사항

| 항목             | 기준                                  |
|------------------|---------------------------------------|
| 데이터 처리 성능 | 500,000 rows 생성 within <5 sec       |
| 재현 가능성      | seed 기반 동일 결과 생성              |
| 확장성           | 사용자 수 증가(최대 1M) 가능          |
| 데이터 포맷      | Parquet (효율성) / CSV (호환성)       |
| 코드 가독성      | 모듈화, config.py로 파라미터 중앙 관리|

---

## 9. 기술 스택

| 용도                  | 기술                                |
|-----------------------|-------------------------------------|
| 데이터 생성           | Python, NumPy, Faker, Pandas        |
| 통계 분석             | SciPy, Statsmodels, Scikit-learn    |
| 시각화                | Seaborn, Plotly, Matplotlib         |
| 결과 대시보드 (선택)  | Streamlit                           |
| 데이터 저장           | Parquet (Apache Arrow), CSV         |

---

## 10. 결과물 (Deliverables)

| 산출물                          | 형식               |
|---------------------------------|--------------------|
| Synthetic Dataset               | CSV / Parquet      |
| Analysis Notebook               | .ipynb             |
| Statistical Report              | PDF / Markdown     |
| Parameter Configuration         | config.py          |
| Optional: Streamlit Dashboard   | Web App            |

---

## 11. 성공 지표 (Success Criteria)

| 기준                      | 조건                                    |
|---------------------------|-----------------------------------------|
| p-value 검정 적용         | 100% (모든 가설 검정)                   |
| 분석 결과 해석 가능       | 테스트 가설 검증 포함                   |
| 인사이트 응용             | 정책/UX 개선 방향 제시                  |
| 재현성 보장               | 동일 seed로 동일 결과 생성              |
| 통계적 유의성 달성        | 주요 가설(H1, H2)에서 p < 0.05          |

### 인사이트 수준 예시

| 결과 유형        | 예시 문장                                                                 |
|------------------|---------------------------------------------------------------------------|
| 단순 결론        | "A UI에서 Fast 선택률이 더 높았다."                                      |
| 전략적 결론      | "시간 압박이 높은 사용자 세그먼트에서는 Efficiency-oriented UI가 효과적이다." |
| 제품 적용 제안   | "Rush Hour에는 A UI, Off-Peak 시간에는 B UI 적용하는 Dynamic UI 전략이 유효하다." |

---

## 12. 프로젝트 제약사항

### 12.1 범위 제한
- 본 프로젝트는 **시뮬레이션 연구**이며, 실제 지하철 시스템과 연동하지 않는다.
- 실제 사용자 테스트는 포함되지 않는다.

### 12.2 가정 사항
- 사용자는 제공된 정보(시간, 혼잡도)를 이해하고 합리적으로 판단한다.
- 경로 시간 및 혼잡도는 실제 데이터를 근사한 확률 분포로 생성된다.
- 학습 효과는 단순 선형 모델로 근사 가능하다.

### 12.3 향후 확장 가능성
- 더 복잡한 경로 네트워크 (3개 이상 경로 선택)
- 실시간 정보 업데이트 효과 분석
- 실제 사용자 대상 A/B 테스트 설계 및 적용

---

## 13. 참고 문헌 및 이론적 배경

### 13.1 행동경제학 이론
- **Prospect Theory (Kahneman & Tversky)**: 시간 절약과 혼잡도 회피 간의 트레이드오프
- **Bounded Rationality (Simon)**: 제한된 정보 하에서의 의사결정

### 13.2 교통수단 선택 모델
- Discrete Choice Models (Logit, Nested Logit)
- 시간 가치(Value of Time) 연구

### 13.3 A/B Testing 방법론
- Controlled Experiments (Kohavi et al.)
- Mixed-Effects Models for Repeated Measures

---

## 문서 이력

| 버전 | 날짜       | 변경 내용                     | 작성자 |
|------|------------|-------------------------------|--------|
| 1.0  | 2025-12-04 | 최초 작성 (최종 확정 버전)    | -      |

---

**문서 종료**
