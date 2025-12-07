# DAY 3 비판적 코드 리뷰

**날짜**: 2025-12-04
**파일**: `data/add_choice_behavior.py`
**리뷰 대상**: 선택 행동 모델링 및 학습 효과 구현
**리뷰 방식**: 비판적이고 확실한 관점

---

## 🔴 CRITICAL ISSUES (치명적 - 즉시 수정 필요)

### ❌ Issue #1: 로직 순서 오류 (process_all_trials)

**심각도**: 🔴 CRITICAL - 프로그램 실행 불가
**위치**: `add_choice_behavior.py:256-265`

**문제점**:
```python
for idx in user_data.index:
    trial_num = user_data.loc[idx, 'trial_number']
    if trial_num > 1:
        prev_idx = user_data[user_data['trial_number'] == trial_num - 1].index[0]
        user_data.loc[idx, 'previous_choice'] = user_data.loc[prev_idx, 'selected_route']  # ❌

# 선택 생성
user_data['selected_route'] = generate_route_choice(user_data)  # 여기서 생성!
```

**분석**:
- Line 262에서 `selected_route`를 참조하지만, 이 컬럼은 **아직 존재하지 않음**
- `selected_route`는 Line 265에서 생성됨
- KeyError 발생으로 프로그램 실행 불가

**에러 메시지**:
```
KeyError: 'selected_route'
```

**올바른 로직**:
1. Trial 1 → previous_choice=None, selected_route 생성
2. Trial 2 → previous_choice=Trial1의 selected_route, selected_route 생성
3. Trial 3 → previous_choice=Trial2의 selected_route, selected_route 생성
...

**수정 방안**:
Trial별로 **순차적으로** 선택을 생성하면서 previous_choice를 업데이트해야 함.

---

### ❌ Issue #2: Random Seed 위치

**심각도**: 🔴 CRITICAL - 재현성 보장 안됨
**위치**: `add_choice_behavior.py:15-17`

**현재 코드**:
```python
# Random seed 설정 (재현성)
np.random.seed(config.RANDOM_SEED)
print(f"[SEED] Random seed 설정: {config.RANDOM_SEED}")
```

**문제점**:
- DAY 2 Critical Review에서 동일한 이슈 지적됨
- Seed가 모듈 최상단이 아닌 import 아래 위치
- 주석만 있고 섹션 구분이 불명확

**수정 방안**:
```python
# config.py 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ===== CRITICAL FIX #2: Random Seed를 모듈 최상단에 배치 =====
np.random.seed(config.RANDOM_SEED)
print(f"[SEED] Random seed 설정: {config.RANDOM_SEED}")
```

---

### ❌ Issue #3: 학습 효과 검증 로직 오류

**심각도**: 🔴 CRITICAL - 결과 해석 불가
**위치**: `add_choice_behavior.py:329-340`

**현재 코드**:
```python
fast_after_fast = (prev_fast['selected_route'] == 'Fast').mean()
fast_after_relax = (prev_relax['selected_route'] == 'Fast').mean()

print(f"  이전 Fast 후 Fast 선택률: {fast_after_fast:.2%}")
print(f"  이전 Relax 후 Fast 선택률: {fast_after_relax:.2%}")
print(f"  차이: {fast_after_relax - fast_after_fast:.2%}p (양수면 학습 효과 확인)")  # ❌
```

**문제점**:
- β4 = -0.4 (음수) → 이전 Fast 선택 시 **다음 Fast 확률 감소**
- 즉, `fast_after_fast < fast_after_relax` 여야 정상
- 하지만 출력 메시지는 "양수면 학습 효과 확인"이라고 잘못 설명
- **수식적 해석과 정반대**

**올바른 해석**:
- β4 = -0.4 → P(Fast | previous=Fast) **감소**
- 학습 효과 = 경험 후 선택 변경 성향
- `fast_after_relax - fast_after_fast > 0` 여야 정상

**수정 방안**:
```python
difference = fast_after_relax - fast_after_fast
print(f"  차이: {difference:.2%}p (양수면 학습 효과 정상)")
print(f"  β4={config.BETA_PREVIOUS_CHOICE} → 이전 Fast 후 Fast 확률 감소 예상")
```

---

## 🟠 MAJOR ISSUES (중요 - 빠른 수정 권장)

### ⚠️ Issue #4: 비효율적 사용자별 순차 처리

**심각도**: 🟠 MAJOR - 성능 문제
**위치**: `add_choice_behavior.py:252-267`

**현재 코드**:
```python
for user_id in df['user_id'].unique():  # 100,000회 반복
    user_data = df[df['user_id'] == user_id].copy()  # 500,000 rows 필터링

    for idx in user_data.index:  # 5회 반복
        trial_num = user_data.loc[idx, 'trial_number']
        if trial_num > 1:
            prev_idx = user_data[user_data['trial_number'] == trial_num - 1].index[0]
            user_data.loc[idx, 'previous_choice'] = ...

    all_results.append(user_data)
```

**문제점**:
1. **DataFrame 필터링 100,000번** (`df[df['user_id'] == user_id]`)
2. **이중 루프**: 외부 100,000 × 내부 5 = 500,000회 반복
3. `.loc[]` 사용으로 인한 오버헤드
4. **예상 실행 시간**: 5-10분 (DAY 2는 ~3초)

**벤치마크 비교**:
- DAY 1 (100,000 users 생성): ~1초
- DAY 2 (500,000 rows 생성): ~3초
- DAY 3 (현재 로직): 예상 5-10분 (**100-200배 느림**)

**수정 방안**:
- **벡터화된 shift() 사용**: `df.groupby('user_id')['selected_route'].shift(1)`
- Trial별 순차 생성으로 로직 재구성

---

### ⚠️ Issue #5: Magic Number 다수 발견

**심각도**: 🟠 MAJOR - 유지보수성 저하
**위치**: 여러 곳

**발견된 Magic Numbers**:
```python
# generate_satisfaction_score()
base_score = 3.0                              # Line 164
match_bonus[mask_efficiency_fast] = 2.0       # Line 171
match_bonus[mask_comfort_relax] = 2.0         # Line 175
match_bonus[mask_neutral] = 1.0               # Line 179
pressure_penalty[mask_urgent_relax] = -1.0    # Line 184
noise = np.random.normal(0, 0.5, size=len(df))  # Line 187

# generate_decision_time()
base_time = np.random.normal(5.5, 1.5, size=len(df))  # Line 212
pressure_effect = (df['time_pressure'] - 1) * 1.5     # Line 218
decision_time = np.maximum(decision_time, 1.0)        # Line 223
```

**문제점**:
- DAY 2에서 모든 magic number를 config.py로 이동했는데 DAY 3에서 다시 발생
- **9개 이상의 하드코딩된 상수**
- 튜닝 시 코드 전체 검색 필요

**수정 방안**:
config.py에 추가:
```python
# ============================================
# 만족도 생성 파라미터
# ============================================
SATISFACTION_BASE = 3.0
SATISFACTION_MATCH_BONUS_STRONG = 2.0
SATISFACTION_MATCH_BONUS_NEUTRAL = 1.0
SATISFACTION_PRESSURE_PENALTY = -1.0
SATISFACTION_NOISE_STD = 0.5

# ============================================
# 의사결정 시간 파라미터
# ============================================
DECISION_TIME_MEAN = 5.5
DECISION_TIME_STD = 1.5
DECISION_TIME_PRESSURE_EFFECT = 1.5
DECISION_TIME_MIN = 1.0
```

---

### ⚠️ Issue #6: 에러 처리 부족

**심각도**: 🟠 MAJOR - 견고성 부족
**위치**: `add_choice_behavior.py:34-43`

**현재 코드**:
```python
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except FileNotFoundError:
    raise FileNotFoundError(f"Trial 데이터 파일을 찾을 수 없습니다: {file_path}")

if len(df) == 0:
    raise ValueError(f"빈 데이터 파일입니다: {file_path}")

print(f"[OK] Trial 데이터 로드: {len(df):,} rows")
return df
```

**문제점**:
- **필수 컬럼 검증 없음** (DAY 2와 달리)
- 선택 생성에 필요한 컬럼: `['trial_number', 'time_pressure', 'personality_type', 'route_time_fast', 'route_time_relax', 'assigned_group']`
- 컬럼 누락 시 런타임 에러 발생

**수정 방안**:
```python
# 필수 컬럼 검증
required_cols = [
    'user_id', 'trial_number', 'assigned_group', 'personality_type',
    'time_pressure', 'route_time_fast', 'route_time_relax'
]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"필수 컬럼 누락: {missing_cols}")
```

---

### ⚠️ Issue #7: 확률 계산 시 Overflow 위험

**심각도**: 🟠 MAJOR - 수치 안정성
**위치**: `add_choice_behavior.py:106`

**현재 코드**:
```python
# Sigmoid 함수
prob_fast = 1 / (1 + np.exp(-logit))
```

**문제점**:
- `logit`이 매우 큰 양수일 때 `np.exp(-logit)` → 0 → `prob_fast = 1.0` (정상)
- `logit`이 매우 큰 음수일 때 `np.exp(-logit)` → ∞ → **Overflow 발생**

**예시**:
```python
logit = -1000
np.exp(-logit)  # = np.exp(1000) = inf
1 / (1 + inf)   # = 0 (하지만 경고 발생)
```

**수정 방안**:
수치적으로 안정적인 sigmoid 구현:
```python
# 수치적으로 안정적인 Sigmoid
def stable_sigmoid(x):
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

prob_fast = stable_sigmoid(logit)
```

또는 scipy 사용:
```python
from scipy.special import expit
prob_fast = expit(logit)
```

---

## 🟡 MINOR ISSUES (경미 - 개선 권장)

### Issue #8: Docstring 불완전

**심각도**: 🟡 MINOR
**위치**: 여러 함수

**문제점**:
```python
def encode_personality(personality_type):
    """
    Personality type을 숫자로 인코딩

    Args:
        personality_type: personality_type 컬럼 (Series 또는 str)

    Returns:
        int or np.ndarray: efficiency=1, comfort=-1, neutral=0
    """
```

- `Args` 타입이 불명확 ("Series 또는 str" → `pd.Series | str`)
- `Returns` 타입이 불명확 ("int or np.ndarray" → 어느 경우에 어떤 타입?)

**수정 방안**:
```python
def encode_personality(personality_type: pd.Series | str) -> np.ndarray | int:
    """
    Personality type을 숫자로 인코딩

    Args:
        personality_type: personality_type 컬럼
            - pd.Series: 전체 컬럼 (벡터화 처리)
            - str: 단일 값

    Returns:
        - pd.Series 입력 시: np.ndarray (same length)
        - str 입력 시: int

        인코딩 규칙:
        - 'efficiency-oriented' → 1
        - 'comfort-oriented' → -1
        - 'neutral' → 0
    """
```

---

### Issue #9: 검증 로직 비일관성

**심각도**: 🟡 MINOR
**위치**: `add_choice_behavior.py:289-342`

**현재 상태**:
```python
def validate_complete_data(df):
    # 1. 결측값 확인
    # 2. 선택 분포 확인
    # 3. 그룹별 선택 분포
    # 4. 만족도 통계
    # 5. decision_time 통계
    # 6. 학습 효과 확인
```

**문제점**:
- DAY 2 검증은 **범위 검증** 포함 (예: `route_time_fast >= MIN_ROUTE_TIME_FAST`)
- DAY 3 검증은 범위 검증 없음
- 예: `satisfaction_score`가 0~5 범위인지 확인 안함

**수정 방안**:
```python
# 7. 범위 검증
print(f"\n범위 검증:")
satisfaction_valid = ((df['satisfaction_score'] >= 0) & (df['satisfaction_score'] <= 5)).all()
decision_valid = (df['decision_time'] >= config.DECISION_TIME_MIN).all()
print(f"  satisfaction_score [0, 5]: {satisfaction_valid}")
print(f"  decision_time >= {config.DECISION_TIME_MIN}: {decision_valid}")
```

---

### Issue #10: 불필요한 중복 코드

**심각도**: 🟡 MINOR
**위치**: `add_choice_behavior.py:244-245`

**현재 코드**:
```python
# 1단계: previous_choice 초기화 (첫 trial은 None)
df['previous_choice'] = None
df.loc[df['trial_number'] == 1, 'previous_choice'] = None
```

**문제점**:
- Line 244에서 이미 전체를 None으로 설정
- Line 245는 불필요한 중복 작업

**수정 방안**:
```python
# 1단계: previous_choice 초기화 (첫 trial은 None)
df['previous_choice'] = None
```

---

## 🔵 SUGGESTIONS (제안 - 선택 사항)

### Suggestion #1: 진행률 표시

**위치**: `process_all_trials()` 함수

**제안**:
```python
from tqdm import tqdm

for user_id in tqdm(df['user_id'].unique(), desc="선택 행동 모델링"):
    user_data = df[df['user_id'] == user_id].copy()
    ...
```

**효과**: 100,000명 처리 시 진행 상황 가시화

---

### Suggestion #2: 로깅 시스템

**제안**:
```python
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

logger.info("Trial 데이터 로드 시작...")
```

---

## 📊 코드 품질 평가

### 현재 점수: **45/100** (F)

| 항목 | 점수 | 평가 |
|------|------|------|
| 기능 동작 | 0/10 | ❌ 실행 불가 (KeyError) |
| 로직 정확성 | 3/10 | 🔴 순서 오류, 학습 효과 검증 오류 |
| 견고성 | 4/10 | 🟠 에러 처리 부족, overflow 위험 |
| 성능 | 3/10 | 🔴 비효율적 순차 처리 (100-200배 느림) |
| 유지보수성 | 5/10 | 🟠 Magic number 다수 |
| 재현성 | 8/10 | ✅ Seed 설정 있음 (위치 개선 필요) |
| 문서화 | 7/10 | ⚠️ Docstring 불완전 |
| 코드 스타일 | 8/10 | ✅ 일관성 유지 |

**등급**: F (프로덕션 부적합)

---

## 🎯 수정 우선순위

### Priority 1 (즉시)
1. ✅ Issue #1: 로직 순서 수정 (실행 불가)
2. ✅ Issue #3: 학습 효과 검증 로직 수정

### Priority 2 (중요)
3. ✅ Issue #4: 성능 최적화 (벡터화)
4. ✅ Issue #5: Magic number → config.py
5. ✅ Issue #6: 에러 처리 강화
6. ✅ Issue #7: Sigmoid 수치 안정성

### Priority 3 (개선)
7. ⚠️ Issue #9: 범위 검증 추가
8. ⚠️ Issue #10: 중복 코드 제거

---

## 📝 결론

**DAY 3 코드는 현재 프로덕션 배포 불가능 상태입니다.**

**주요 문제점**:
1. 🔴 **프로그램 실행 불가** (KeyError)
2. 🔴 **로직 오류** (학습 효과 검증 해석 반대)
3. 🟠 **성능 문제** (DAY 2 대비 100배 이상 느림)
4. 🟠 **유지보수성 저하** (magic number 재발생)

**개선 후 목표**:
- ✅ 정상 실행
- ✅ 학습 효과 올바른 검증
- ✅ 실행 시간 < 30초 (현재 예상 5-10분)
- ✅ 모든 파라미터 config.py 관리
- ✅ 코드 품질 점수 85+ (B+)

---

**리뷰 완료일**: 2025-12-04
**리뷰어**: Claude (Critical Mode)
**최종 판정**: ❌ 수정 필수 (3 Critical, 4 Major issues)
