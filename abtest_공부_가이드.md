# 지하철 A/B Test 프로젝트 완벽 가이드
## 데이터 분석가 취업 준비용 학습 자료

---

# 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [발견한 문제점과 해결 방법](#2-발견한-문제점과-해결-방법)
3. [통계 개념 완벽 정리](#3-통계-개념-완벽-정리)
4. [코드별 상세 설명](#4-코드별-상세-설명)
5. [면접 대비 Q&A](#5-면접-대비-qa)
6. [추가 학습 자료](#6-추가-학습-자료)

---

# 1. 프로젝트 개요

## 1.1 프로젝트 목표

**질문**: "지하철 앱 UI를 바꾸면 사용자 행동이 변할까?"

- **A그룹 (실험군)**: "빠른 경로"를 강조하는 UI
- **B그룹 (대조군)**: "여유로운 경로"를 강조하는 UI

**측정 지표**:
- Fast Route 선택률 차이
- 사용자 만족도
- 혼잡도 분산 효과

## 1.2 데이터 규모

```
참가자: 100,000명
Trial: 각 사용자당 5회 반복
총 데이터: 100,000 × 5 = 500,000 rows
```

**왜 반복 측정?**
- 한 번만 측정하면 우연일 수 있음
- 5번 반복하면 학습 효과(Learning Effect) 관찰 가능
- "처음엔 빨리 가려다가 혼잡하니 나중엔 여유롭게 가기로 결정" 같은 변화 포착

## 1.3 핵심 기능: 동적 혼잡도 피드백

**일반적인 시뮬레이션**:
```
Trial 1 혼잡도: 고정값 (예: Fast 85%, Relax 45%)
Trial 2 혼잡도: 고정값 (똑같음)
Trial 3 혼잡도: 고정값 (똑같음)
```

**우리 프로젝트의 차별점**:
```python
# Trial 1
Fast 선택률: 92.16% → 다음 Trial에서 Fast가 더 혼잡해짐

# Trial 2
Fast 혼잡도: 85% → 195.6% (이전 선택률 반영!)
Fast 선택률: 74.82% (사람들이 학습함)

# Trial 3
Fast 혼잡도: 174.8% (다시 조정됨)
Fast 선택률: 57.13% (계속 감소)
```

**현실성**:
- 실제로 특정 경로를 많이 선택하면 그 경로가 혼잡해짐
- 혼잡을 경험한 사람들이 다음번에 다른 선택을 함
- 이것이 "동적 피드백 시스템"

---

# 2. 발견한 문제점과 해결 방법

## 2.1 치명적 문제 1: 통계 해석 오류 🔴

### 문제 상황

**잘못된 README.md (수정 전)**:
```markdown
| 변수 | 계수 | 해석 |
|------|------|------|
| UI 그룹 | +0.330 | Fast 선택 확률 33.0% 증가 ❌ |
| 시간 압박 | +0.936 | Fast 선택 확률 93.6% 증가 ❌ |
```

### 왜 틀렸는가?

**로지스틱 회귀의 계수는 "로그 오즈비"입니다!**

```
일반 선형 회귀:    y = β₀ + β₁x
                   β₁ = x가 1 증가할 때 y의 증가량 ✓

로지스틱 회귀:     log(odds) = β₀ + β₁x
                   β₁ ≠ 확률 증가량 ❌
                   β₁ = 로그 오즈의 증가량 ✓
```

### 올바른 해석

**Step 1**: 계수에서 오즈비 계산
```python
β = 0.330
오즈비(OR) = exp(0.330) = 1.39
```

**Step 2**: 해석
```
"A그룹의 Fast 선택 오즈가 B그룹 대비 1.39배"
또는
"A그룹의 Fast 선택 오즈가 39% 증가"
```

### 오즈(Odds)란?

**확률 vs 오즈**:
```
확률(Probability): 성공 / 전체
오즈(Odds):        성공 / 실패

예시:
- 100명 중 70명 성공
- 확률 = 70/100 = 0.7 (70%)
- 오즈 = 70/30 = 2.33 ("성공이 실패보다 2.33배 많음")
```

**왜 오즈를 쓰는가?**
```
확률은 0~1 범위로 제한됨 (대칭적이지 않음)
  확률 0.1 → 0.5: +0.4
  확률 0.5 → 0.9: +0.4 (같은 변화처럼 보이지만 의미가 다름)

오즈는 0~∞ 범위 (로그를 취하면 -∞~∞로 대칭적)
  오즈 0.11 → 1.00: 9배 증가
  오즈 1.00 → 9.00: 9배 증가 (대칭적!)
```

### 수정된 코드

**mixed_models.py**:
```python
# Before
print(f"{var}: {coef:.4f}")

# After
odds_ratio = np.exp(coef)
print(f"{var}: β={coef:.4f}, OR={odds_ratio:.3f}")
```

**README.md**:
```markdown
| 변수 | 계수 (β) | 오즈비 (OR) | 해석 |
|------|---------|------------|------|
| UI 그룹 | +0.330 | 1.39 | A그룹의 Fast 선택 오즈가 1.39배 ✅ |
```

---

## 2.2 치명적 문제 2: 데이터 규모 불일치 🔴

### 문제 상황

```python
# README.md
"100,000명 × 5회 = 500,000 rows"

# app.py (line 109)
"10,000명 × 5회 = 50,000 rows"  ❌
```

**왜 문제인가?**
- 면접관: "어느 게 맞나요?"
- 당신: "아... 그게..."
- 면접관: "데이터 규모도 모르고 분석했나요?" 😱

### 해결 방법

**1단계: 실제 데이터 확인**
```bash
python -c "import pandas as pd; \
df = pd.read_parquet('data/synthetic_data_dynamic.parquet'); \
print(f'Total rows: {len(df):,}')"

# 출력: Total rows: 500,000
```

**2단계: 모든 문서 수정**
```python
# app.py
st.markdown("""
- **참가자**: 100,000명  ✅
- **총 데이터**: 500,000 rows  ✅
""")
```

---

## 2.3 중요 문제 3: 하드코딩된 통계값

### 문제 상황

**app.py (수정 전)**:
```python
st.markdown("""
- **Two-Proportion Z-Test**: p < 0.001  # 하드코딩! ❌
- **Cohen's h**: 0.126                   # 하드코딩! ❌
""")
```

**왜 문제인가?**
- 데이터를 다시 생성하면 값이 바뀌는데 대시보드는 그대로
- 실제 분석 결과와 표시된 값이 다를 수 있음

### 해결 방법

**1단계: 분석 결과 저장**
```python
# analysis/basic_tests.py
results = {
    'z_statistic': z_stat,
    'p_value': p_val,
    'cohen_h': cohen_h
}
pd.DataFrame([results]).to_csv('analysis/basic_tests_results.csv')
```

**2단계: 대시보드에서 로드**
```python
# app.py
df_test = pd.read_csv('analysis/basic_tests_results.csv')
z_stat = df_test.loc[0, 'z_stat']
p_val = df_test.loc[0, 'p_value']
cohen_h = df_test.loc[2, 'cohens_h']

st.markdown(f"""
- **Z-Test**: z = {z_stat:.2f}, p = {p_val:.3f}  ✅
- **Cohen's h**: {cohen_h:.3f}                    ✅
""")
```

**장점**:
- 데이터 재생성 시 자동 업데이트
- 실제 분석 결과와 항상 일치
- 재현성(Reproducibility) 확보

---

# 3. 통계 개념 완벽 정리

## 3.1 A/B 테스트란?

**기본 아이디어**:
```
1. 두 그룹을 랜덤하게 나눔 (A, B)
2. A그룹에게는 변화를 줌 (예: 새 UI)
3. B그룹은 그대로 유지 (기존 UI)
4. 결과를 비교
```

**왜 랜덤 배정이 중요한가?**

**나쁜 예**:
```
A그룹: 서울 사용자 (새 UI)
B그룹: 부산 사용자 (기존 UI)

결과: A그룹 선택률 높음
→ 이게 UI 때문인가? 지역 차이 때문인가? 알 수 없음! ❌
```

**좋은 예**:
```python
# 랜덤 배정
df['assigned_group'] = np.where(
    np.random.random(size=len(df)) < 0.5,
    'A', 'B'
)

# A, B 그룹의 다른 특성들이 평균적으로 같아짐
# → 차이가 있다면 UI 때문!
```

## 3.2 통계적 유의성 (p-value)

**질문**: "A그룹 74%, B그룹 68%... 이 차이가 진짜인가, 우연인가?"

### p-value의 의미

```
p-value = 0.001 의미:

"만약 진짜로는 차이가 없는데,
 우연히 이런 차이가 관찰될 확률이 0.1%"

→ 우연일 가능성이 매우 낮으므로
→ "진짜 차이가 있다"고 결론
```

### 유의수준 (α)

```
α = 0.05 (일반적 기준)

p < 0.05: 통계적으로 유의미 ✓
p ≥ 0.05: 유의미하지 않음 (차이 없다고 할 수 없음)
```

### Two-Proportion Z-Test

**무엇을 테스트하는가?**
```
H₀ (귀무가설): p_A = p_B (두 그룹 비율이 같음)
H₁ (대립가설): p_A ≠ p_B (두 그룹 비율이 다름)
```

**계산 과정**:
```python
# 1. 풀링된 비율 (pooled proportion)
p_pool = (n_A * p_A + n_B * p_B) / (n_A + n_B)

# 2. 표준오차
SE = sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))

# 3. Z 통계량
z = (p_A - p_B) / SE

# 4. p-value (양측 검정)
p_value = 2 * (1 - norm.cdf(abs(z)))
```

**예시**:
```
A그룹: 184,851 / 249,670 = 0.7404
B그룹: 171,038 / 250,330 = 0.6833
차이: 0.0571 (5.71%p)

z = 44.60
p < 0.001

→ "우연히 이런 차이가 날 확률 < 0.1%"
→ "통계적으로 유의미한 차이!"
```

## 3.3 효과 크기 (Effect Size)

### 왜 필요한가?

```
p-value의 문제점:
- 표본 크기가 크면 작은 차이도 유의미하게 나옴
- "통계적으로 유의미" ≠ "실용적으로 의미있음"

예시:
n = 1,000,000
A그룹: 50.1%, B그룹: 50.0%
→ p < 0.05 (유의미!) 하지만 0.1%p 차이가 실용적인가?
```

### Cohen's h

**비율 차이의 효과 크기**:
```python
def cohens_h(p1, p2):
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2

h = cohens_h(0.7404, 0.6833)  # 0.126
```

**해석 기준**:
```
|h| < 0.2:  small (작은 효과)
|h| < 0.5:  medium (중간 효과)
|h| ≥ 0.5:  large (큰 효과)

우리: h = 0.126 → small but meaningful
```

**종합 판단**:
```
통계적 유의성: p < 0.001 ✓ (진짜 차이가 있음)
효과 크기: h = 0.126 (작지만 의미있음)
실용적 의미: 5.71%p → 800만명 × 5.71% = 45만명 유도 가능 ✓

→ 결론: 실무에 적용할 가치가 있음!
```

## 3.4 로지스틱 회귀 (Logistic Regression)

### 왜 로지스틱 회귀를 쓰는가?

**우리의 종속변수**:
```
선택 = {Fast, Relax}  → 이진 변수 (0 또는 1)
```

**일반 선형 회귀의 문제**:
```python
# 일반 회귀
y = β₀ + β₁x

# 문제 1: y가 0~1 범위를 벗어날 수 있음
# x = 10일 때: y = 0.5 + 0.2*10 = 2.5 (125%??) ❌

# 문제 2: 확률의 비선형성을 표현 못함
# 0% → 10%: 쉬움
# 90% → 100%: 매우 어려움
```

**로지스틱 회귀의 해결책**:
```python
# 로지스틱 회귀
log(p / (1-p)) = β₀ + β₁x

# logit 변환으로 확률을 -∞~∞ 범위로 확장
# 그 후 선형 모델 적용
# 예측할 때는 역변환(시그모이드)으로 다시 0~1로
```

### Sigmoid 함수

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# logit = β₀ + β₁x
# p = sigmoid(logit)

# 예시
logit = -2:  p = 0.12 (12%)
logit =  0:  p = 0.50 (50%)
logit = +2:  p = 0.88 (88%)
```

**시각화**:
```
p(x)
1.0 |           -------
    |         /
0.5 |       /
    |     /
0.0 |----
    +---+---+---+---+---+> x
      -2  -1   0  +1  +2

S자 곡선 → "Sigmoid"
```

### 계수 해석

**우리 모델**:
```python
log(odds) = β₀ + β₁(group) + β₂(time_pressure) + ...
```

**각 계수의 의미**:

**β₁ = 0.330 (group_numeric)**:
```
A그룹(1) vs B그룹(0) 비교 시
log(odds_A) - log(odds_B) = 0.330
log(odds_A / odds_B) = 0.330
odds_A / odds_B = exp(0.330) = 1.39

→ "A그룹의 오즈가 B그룹보다 1.39배"
```

**β₂ = 0.936 (time_pressure)**:
```
급함(0) → 보통(1) 변화 시
log(odds) 증가량 = 0.936
odds 증가 비율 = exp(0.936) = 2.55

→ "압박 1단계 증가 시 Fast 선택 오즈가 2.55배"
```

**β₄ = -0.404 (trial_index)**:
```
Trial 1 → Trial 2 변화 시
log(odds) 증가량 = -0.404 (감소!)
odds 변화 비율 = exp(-0.404) = 0.67

→ "Trial 1회 증가 시 Fast 선택 오즈가 0.67배 (33% 감소)"
→ 학습 효과!
```

## 3.5 GEE (Generalized Estimating Equations)

### 왜 GEE가 필요한가?

**문제 상황**:
```
사용자 A: Trial 1, 2, 3, 4, 5 (5개 관측값)
사용자 B: Trial 1, 2, 3, 4, 5 (5개 관측값)
...

일반 로지스틱 회귀 가정:
"모든 관측값이 독립적" ❌

현실:
사용자 A의 Trial 1과 Trial 2는 상관관계 있음!
→ 같은 사람이니까
→ Trial 1에서 Fast 선택 → Trial 2에서도 Fast 선택 경향
```

**GEE의 해결책**:
```
1. 개인 내 상관관계(within-subject correlation)를 고려
2. 상관 구조(correlation structure) 지정
   - AR(1): 인접한 Trial끼리 더 강한 상관
   - Exchangeable: 모든 Trial끼리 같은 상관
```

### AR(1) 상관 구조

```
Corr(Trial 1, Trial 2) = ρ
Corr(Trial 1, Trial 3) = ρ²
Corr(Trial 1, Trial 4) = ρ³
...

예: ρ = 0.5
Trial 1-2: 0.50 (50% 상관)
Trial 1-3: 0.25 (25% 상관)
Trial 1-4: 0.13 (13% 상관)

→ 시간이 지날수록 상관 약해짐 (현실적!)
```

### GEE 코드

```python
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Autoregressive

model = GEE(
    endog=df['choice_binary'],           # 종속변수 (0/1)
    exog=df[['group', 'pressure', ...]],  # 독립변수들
    groups=df['user_id'],                 # 그룹핑 변수
    family=Binomial(),                    # 이진 분포
    cov_struct=Autoregressive()           # AR(1) 구조
)

result = model.fit()
```

## 3.6 FDR (False Discovery Rate) 보정

### 다중 검정 문제

**시나리오**:
```
변수 6개를 동시에 테스트
각각 α = 0.05 기준

문제:
- 각 테스트에서 false positive 확률 = 5%
- 6개 중 최소 1개 false positive 확률 = 1 - (0.95)⁶ = 26.5%!

→ "6개 중 하나는 우연히 유의미하게 나올 수 있음"
```

**예시**:
```
변수 1: p = 0.001 → 유의미 ✓
변수 2: p = 0.003 → 유의미 ✓
변수 3: p = 0.010 → 유의미 ✓
변수 4: p = 0.030 → 유의미? (6개 중 4번째인데...)
변수 5: p = 0.045 → 유의미? (우연일 수도...)
변수 6: p = 0.200 → 유의미하지 않음 ✗

→ 변수 5가 false positive일 가능성?
```

### Benjamini-Hochberg 방법

**절차**:
```python
# 1. p-value를 오름차순 정렬
p_values = [0.001, 0.003, 0.010, 0.030, 0.045, 0.200]

# 2. 각 p-value에 순위 부여
ranks = [1, 2, 3, 4, 5, 6]

# 3. 임계값 계산
α = 0.05
m = 6  # 총 테스트 수
thresholds = [α * r / m for r in ranks]
# [0.0083, 0.0167, 0.0250, 0.0333, 0.0417, 0.0500]

# 4. p-value < threshold인 것만 유의미
결과:
  변수 1: 0.001 < 0.0083 → 유의미 ✓
  변수 2: 0.003 < 0.0167 → 유의미 ✓
  변수 3: 0.010 < 0.0250 → 유의미 ✓
  변수 4: 0.030 < 0.0333 → 유의미 ✓
  변수 5: 0.045 > 0.0417 → 유의미 아님 ✗
  변수 6: 0.200 > 0.0500 → 유의미 아님 ✗
```

**우리 프로젝트 결과**:
```
6개 변수 모두 p < 0.001
→ FDR 보정 후에도 모두 유의미 ✓
→ 매우 강력한 결과!
```

---

# 4. 핵심 코드 설명

## 4.1 동적 혼잡도 계산

```python
def calculate_dynamic_congestion(trial_number, previous_trial_data):
    """
    Trial별 혼잡도를 이전 선택 결과에 따라 동적 계산

    핵심 아이디어:
    - Fast를 많이 선택 → Fast가 더 혼잡해짐
    - 혼잡해진 Fast → 다음 Trial에서 선택률 감소
    """
    if trial_number == 1:
        # Trial 1: 기본값
        return (85.0, 45.0)
    else:
        # Trial 2+: 이전 선택 비율 반영
        prev_fast_ratio = (previous_trial_data['selected_route'] == 'Fast').mean()

        # 혼잡도 = 기본값 + (선택 비율 × 배수)
        congestion_fast = 85.0 + (prev_fast_ratio * 120)
        congestion_relax = 45.0 + ((1 - prev_fast_ratio) * 120)

        return (congestion_fast, congestion_relax)
```

**실제 예시**:
```
Trial 1: Fast 92.16% 선택

Trial 2 혼잡도:
Fast = 85 + (0.9216 × 120) = 195.6%
Relax = 45 + (0.0784 × 120) = 54.4%

→ Fast가 2배 이상 혼잡!
→ Trial 2에서 Fast 선택률 74.82%로 감소
```

## 4.2 선택 확률 계산 (로지스틱 회귀)

```python
def calculate_choice_probability(trial_data, previous_trial_data):
    """
    로지스틱 회귀 모델로 Fast 선택 확률 계산

    logit(P) = β₀ + β₁·group + β₂·pressure + β₃·personality +
               β₄·previous + β₅·congestion
    """
    # β₀: 그룹별 기본 편향
    beta_0 = np.where(trial_data['assigned_group'] == 'A', 0.3, -0.3)

    # β₁: 시간 압박 (급할수록 Fast)
    beta_1_term = 1.2 * trial_data['time_pressure']

    # β₂: 성향 (효율지향 → Fast)
    personality = encode_personality(trial_data['personality_type'])
    beta_2_term = 0.6 * personality

    # β₃: 이전 선택 (학습 효과)
    if previous_trial_data is not None:
        previous_fast = ...  # 이전에 Fast 선택했는지
        beta_3_term = -0.4 * previous_fast  # 음수! (학습)

    # β₄: 혼잡도 경험
    if previous_trial_data is not None:
        congestion_experienced = ...  # 경험한 혼잡도
        beta_4_term = -0.01 * congestion_experienced

    # Logit 계산
    logit = beta_0 + beta_1_term + beta_2_term + beta_3_term + beta_4_term

    # Sigmoid로 확률 변환
    prob_fast = 1 / (1 + np.exp(-logit))

    return prob_fast
```

## 4.3 GEE 분석

```python
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Autoregressive

def gee_analysis(df):
    # 시간 순서로 정렬 (중요!)
    df_sorted = df.sort_values(['user_id', 'trial_number'])

    # 변수 준비
    exog_vars = [
        'group_numeric',         # A=1, B=0
        'time_pressure',         # 0, 1, 2
        'personality_numeric',   # -1, 0, 1
        'trial_index',           # 0~4
        'time_diff',
        'congestion_diff'
    ]

    # GEE 모델
    model = GEE(
        endog=df_sorted['choice_binary'],  # Fast=1, Relax=0
        exog=df_sorted[exog_vars],
        groups=df_sorted['user_id'],       # 그룹핑!
        family=Binomial(),
        cov_struct=Autoregressive()        # AR(1)
    )

    result = model.fit()

    # 오즈비 계산
    for var, coef in zip(exog_vars, result.params):
        odds_ratio = np.exp(coef)
        print(f"{var}: β={coef:.4f}, OR={odds_ratio:.3f}")

    return result
```

---

# 5. 면접 대비 핵심 Q&A

## Q1: "로지스틱 회귀 계수를 어떻게 해석하나요?"

**완벽한 답변**:
```
"로지스틱 회귀의 계수는 로그 오즈비입니다.

예를 들어 group_numeric 계수가 0.330이면:
1. exp(0.330) = 1.39 (오즈비 계산)
2. 'A그룹의 Fast 선택 오즈가 B그룹 대비 1.39배'

확률로 직접 해석할 수 없는 이유는
확률 증가량이 베이스라인에 따라 다르기 때문입니다.

예: 오즈 1.39배 증가 시
- 10% → 14% (+4%p)
- 50% → 58% (+8%p)
- 90% → 93% (+3%p)

모두 같은 오즈비(1.39)지만 확률 증가량은 다릅니다."
```

## Q2: "왜 GEE를 사용했나요?"

**완벽한 답변**:
```
"반복 측정 데이터이기 때문입니다.

각 사용자가 5회 반복 측정되므로
같은 사용자의 관측값들은 독립적이지 않습니다.

일반 로지스틱 회귀는 독립성을 가정하므로
표준오차를 과소추정하고 p-value가 실제보다 작게 나옵니다.

GEE는 AR(1) 상관 구조로
개인 내 상관관계를 고려하여
올바른 표준오차를 계산합니다."
```

## Q3: "동적 혼잡도 피드백이 뭔가요?"

**완벽한 답변**:
```
"Trial별로 이전 선택 결과를 반영하여
혼잡도를 동적으로 업데이트하는 시스템입니다.

예시:
- Trial 1: Fast 92% 선택
- Trial 2 혼잡도: Fast 195% (↑), Relax 54%
- Trial 2: Fast 75% 선택 (감소!)
- Trial 3 혼잡도: Fast 175%, Relax 75%
- Trial 3: Fast 57% 선택 (더 감소)

이를 통해 현실적인 학습 효과를 구현했고,
GEE 분석에서 trial_index 계수 -0.404 (p<0.001)로
통계적으로 검증했습니다."
```

## Q4: "프로젝트의 가장 큰 기여는?"

**완벽한 답변**:
```
"두 가지입니다:

1. 기술적 차별점:
   동적 피드백 시스템으로 정적 A/B 테스트의 한계 극복

2. 통계적 엄밀성:
   - GEE로 반복 측정 데이터 올바르게 분석
   - FDR 보정으로 다중 검정 문제 해결
   - 6개 변수 모두 p<0.001로 매우 강력한 결과

실무 적용:
- 5.71%p 차이 × 800만명 = 45만명 유도 가능
- 개인화 전략 (압박↑ → Fast 추천)
- 혼잡도 분산 효과"
```

---

# 6. 최종 체크리스트

## 면접 전 반드시 확인

### 통계 개념
- [ ] 오즈(Odds) = P/(1-P)
- [ ] 오즈비(OR) = exp(β)
- [ ] 로지스틱 회귀 계수 ≠ 확률 증가량
- [ ] p-value 의미 정확히 이해
- [ ] GEE가 왜 필요한지 설명 가능
- [ ] FDR 보정 이유 이해

### 핵심 수치 암기
- [ ] 데이터: 100,000명 × 5회 = 500,000 rows
- [ ] 차이: 5.71%p (A 74.04%, B 68.33%)
- [ ] z통계량: 44.60, p < 0.001
- [ ] Cohen's h: 0.126 (small)
- [ ] OR: group 1.39, pressure 2.55, trial 0.67

### 프로젝트 내용
- [ ] 동적 혼잡도 피드백 메커니즘 설명 가능
- [ ] 학습 효과 (92.16% → 65.72%)
- [ ] 파라미터 튜닝 과정

### 개선 사항
- [ ] 통계 해석 오류 발견 및 수정
- [ ] 하드코딩 → 동적 로드
- [ ] 재현성 확보 (random_state)

---

**공부 완료하셨으면 자신감 가지세요!**
**이 프로젝트는 통계, 코딩, 도메인 지식을 모두 보여줄 수 있는 훌륭한 포트폴리오입니다.**

---

*작성일: 2025-12-09*
*프로젝트 기간: 2025년 7월 ~ 2025년 8월*
*버전: 1.0*
