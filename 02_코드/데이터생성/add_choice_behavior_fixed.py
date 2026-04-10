"""
DAY 3: 선택 행동 모델링 및 학습 효과 구현
로지스틱 회귀 기반 경로 선택 + 만족도 + decision_time 생성

FIXED VERSION: 모든 Critical 및 Major 이슈 수정
"""

import numpy as np
import pandas as pd
import sys
import os
from scipy.special import expit  # 수치적으로 안정적인 sigmoid

# config.py 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ===== CRITICAL FIX #2: Random Seed를 모듈 최상단에 배치 =====
np.random.seed(config.RANDOM_SEED)
print(f"[SEED] Random seed 설정: {config.RANDOM_SEED}")


def load_trials_data(file_path='data/trials_data_partial.csv'):
    """
    DAY 2에서 생성한 trial 데이터 로드

    Args:
        file_path: trial 데이터 파일 경로

    Returns:
        pd.DataFrame: trial 데이터
    """
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, file_path)

    # ===== MAJOR FIX #6: 에러 처리 강화 =====
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except FileNotFoundError:
        raise FileNotFoundError(f"Trial 데이터 파일을 찾을 수 없습니다: {file_path}")

    if len(df) == 0:
        raise ValueError(f"빈 데이터 파일입니다: {file_path}")

    # 필수 컬럼 검증
    required_cols = [
        'user_id', 'trial_number', 'assigned_group', 'personality_type',
        'time_pressure', 'route_time_fast', 'route_time_relax'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")

    print(f"[OK] Trial 데이터 로드: {len(df):,} rows")
    return df


def encode_personality(personality_type):
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
    mapping = {
        'efficiency-oriented': 1,
        'comfort-oriented': -1,
        'neutral': 0
    }

    if isinstance(personality_type, pd.Series):
        return personality_type.map(mapping)
    else:
        return mapping.get(personality_type, 0)


def calculate_choice_probability(df):
    """
    로지스틱 회귀 기반 Fast Route 선택 확률 계산

    P(Fast) = sigmoid(β0 + β1·time_pressure + β2·personality
                      + β3·time_difference + β4·previous_choice + ε)

    Args:
        df: trial DataFrame (personality 인코딩 및 previous_choice 포함)

    Returns:
        np.ndarray: Fast Route 선택 확률 (0~1)
    """
    # β0: 그룹별 절편
    beta_0 = np.where(df['assigned_group'] == 'A', config.BETA_0_A, config.BETA_0_B)

    # β1: time_pressure
    beta_1_term = config.BETA_TIME_PRESSURE * df['time_pressure']

    # β2: personality (인코딩 필요)
    personality_encoded = encode_personality(df['personality_type'])
    beta_2_term = config.BETA_PERSONALITY * personality_encoded

    # β3: time_difference (절대값)
    time_difference = np.abs(df['route_time_fast'] - df['route_time_relax'])
    beta_3_term = config.BETA_TIME_DIFF * time_difference

    # β4: previous_choice (Fast=1, Relax=0, None=0)
    previous_choice_encoded = np.where(df['previous_choice'] == 'Fast', 1, 0)
    beta_4_term = config.BETA_PREVIOUS_CHOICE * previous_choice_encoded

    # ε: 노이즈
    noise = np.random.normal(0, config.NOISE_STD, size=len(df))

    # Logit 계산
    logit = beta_0 + beta_1_term + beta_2_term + beta_3_term + beta_4_term + noise

    # ===== MAJOR FIX #7: 수치적으로 안정적인 Sigmoid =====
    prob_fast = expit(logit)  # scipy.special.expit (안정적인 sigmoid)

    return prob_fast


def generate_route_choice(df):
    """
    확률 기반 경로 선택 생성

    Args:
        df: trial DataFrame (확률 계산 완료)

    Returns:
        pd.Series: 'Fast' 또는 'Relax'
    """
    prob_fast = calculate_choice_probability(df)

    # 확률 기반 선택
    random_values = np.random.random(size=len(df))
    selected_route = np.where(random_values < prob_fast, 'Fast', 'Relax')

    return pd.Series(selected_route, index=df.index)


def generate_satisfaction_score(df):
    """
    선택 직후 기대 만족도 생성 (0~5)

    만족도 = 기본 점수 + 매칭 보너스 + 압박 패널티 + 노이즈

    Args:
        df: trial DataFrame (selected_route, personality_type, time_pressure 포함)

    Returns:
        np.ndarray: satisfaction_score (0~5)
    """
    # ===== MAJOR FIX #5: Magic Number 제거 =====
    base_score = config.SATISFACTION_BASE

    # 매칭 보너스: 성향과 선택이 일치하면 +1~2
    match_bonus = np.zeros(len(df))

    # efficiency-oriented가 Fast 선택 → +2
    mask_efficiency_fast = (df['personality_type'] == 'efficiency-oriented') & (df['selected_route'] == 'Fast')
    match_bonus[mask_efficiency_fast] = config.SATISFACTION_MATCH_BONUS_STRONG

    # comfort-oriented가 Relax 선택 → +2
    mask_comfort_relax = (df['personality_type'] == 'comfort-oriented') & (df['selected_route'] == 'Relax')
    match_bonus[mask_comfort_relax] = config.SATISFACTION_MATCH_BONUS_STRONG

    # neutral은 어느 쪽이든 +1
    mask_neutral = df['personality_type'] == 'neutral'
    match_bonus[mask_neutral] = config.SATISFACTION_MATCH_BONUS_NEUTRAL

    # 압박 패널티: 급한데 Relax 선택 시 -1
    pressure_penalty = np.zeros(len(df))
    mask_urgent_relax = (df['time_pressure'] == 0) & (df['selected_route'] == 'Relax')
    pressure_penalty[mask_urgent_relax] = config.SATISFACTION_PRESSURE_PENALTY

    # 랜덤 노이즈
    noise = np.random.normal(0, config.SATISFACTION_NOISE_STD, size=len(df))

    # 최종 만족도
    satisfaction = base_score + match_bonus + pressure_penalty + noise

    # 0~5 범위로 클리핑
    satisfaction = np.clip(satisfaction, 0, 5)

    return satisfaction


def generate_decision_time(df):
    """
    의사결정 시간 생성 (초)

    기본: 3~8초 (정규분포)
    time_pressure 영향: 급할수록 짧게

    Args:
        df: trial DataFrame (time_pressure 포함)

    Returns:
        np.ndarray: decision_time (초)
    """
    # ===== MAJOR FIX #5: Magic Number 제거 =====
    base_time = np.random.normal(config.DECISION_TIME_MEAN, config.DECISION_TIME_STD, size=len(df))

    # time_pressure 영향
    # 0 (급함): -1.5초
    # 1 (보통): 0초
    # 2 (여유): +1.5초
    pressure_effect = (df['time_pressure'] - 1) * config.DECISION_TIME_PRESSURE_EFFECT

    decision_time = base_time + pressure_effect

    # 최소 1초
    decision_time = np.maximum(decision_time, config.DECISION_TIME_MIN)

    return decision_time


def process_all_trials(df):
    """
    모든 trial에 대해 선택 행동 및 관련 데이터 생성

    ===== CRITICAL FIX #1: 로직 순서 완전 재구성 =====
    Trial별 순차 생성으로 학습 효과 올바르게 반영

    Args:
        df: trial DataFrame (DAY 2 출력)

    Returns:
        pd.DataFrame: 선택 행동이 추가된 완전한 데이터
    """
    print("\n선택 행동 모델링 시작...")

    # 정렬 (user_id, trial_number 순)
    df = df.sort_values(['user_id', 'trial_number']).reset_index(drop=True)

    # ===== MAJOR FIX #4: 벡터화된 처리로 성능 최적화 =====
    print("  Trial별 순차 생성 중...")

    # Trial 1: previous_choice 없음
    print("    Trial 1/5 처리 중...")
    trial_1 = df[df['trial_number'] == 1].copy()
    trial_1['previous_choice'] = None
    trial_1['selected_route'] = generate_route_choice(trial_1)

    all_trials = [trial_1]

    # Trial 2~5: 이전 trial 결과 사용
    for trial_num in range(2, config.NUM_TRIALS + 1):
        print(f"    Trial {trial_num}/{config.NUM_TRIALS} 처리 중...")

        current_trial = df[df['trial_number'] == trial_num].copy()
        previous_trial = all_trials[trial_num - 2]  # 바로 이전 trial

        # previous_choice 설정 (user_id 기준 매칭)
        previous_choices = previous_trial.set_index('user_id')['selected_route']
        current_trial['previous_choice'] = current_trial['user_id'].map(previous_choices).values

        # 선택 생성
        current_trial['selected_route'] = generate_route_choice(current_trial)

        all_trials.append(current_trial)

    # 결합
    df_complete = pd.concat(all_trials, ignore_index=True)

    print("  [OK] 선택 행동 생성 완료")

    # 3단계: 만족도 생성
    print("  만족도 생성 중...")
    df_complete['satisfaction_score'] = generate_satisfaction_score(df_complete)
    print("  [OK] 만족도 생성 완료")

    # 4단계: decision_time 생성
    print("  의사결정 시간 생성 중...")
    df_complete['decision_time'] = generate_decision_time(df_complete)
    print("  [OK] 의사결정 시간 생성 완료")

    print("[OK] 모든 선택 행동 모델링 완료")

    return df_complete


def validate_complete_data(df):
    """
    완성된 데이터 검증

    Args:
        df: 완성된 DataFrame
    """
    print("\n=== 완성 데이터 검증 ===")

    # 1. 결측값 확인
    missing = df.isnull().sum()
    print(f"결측값 (previous_choice 제외):")
    missing_without_prev = missing.drop('previous_choice') if 'previous_choice' in missing.index else missing
    print(f"  총 {missing_without_prev.sum()}개 (예상: 0)")

    # 2. 선택 분포 확인
    choice_counts = df['selected_route'].value_counts(normalize=True)
    print(f"\n선택 분포:")
    for choice, ratio in choice_counts.items():
        print(f"  {choice}: {ratio:.2%}")

    # 3. 그룹별 선택 분포
    print(f"\n그룹별 Fast Route 선택률:")
    for group in ['A', 'B']:
        group_data = df[df['assigned_group'] == group]
        fast_ratio = (group_data['selected_route'] == 'Fast').mean()
        print(f"  {group}그룹: {fast_ratio:.2%}")

    # 4. 만족도 통계
    print(f"\n만족도 통계:")
    print(f"  평균: {df['satisfaction_score'].mean():.2f}")
    print(f"  표준편차: {df['satisfaction_score'].std():.2f}")
    print(f"  범위: [{df['satisfaction_score'].min():.2f}, {df['satisfaction_score'].max():.2f}]")

    # 5. decision_time 통계
    print(f"\ndecision_time 통계 (초):")
    print(f"  평균: {df['decision_time'].mean():.2f}")
    print(f"  표준편차: {df['decision_time'].std():.2f}")
    print(f"  범위: [{df['decision_time'].min():.2f}, {df['decision_time'].max():.2f}]")

    # ===== CRITICAL FIX #3: 학습 효과 검증 로직 수정 =====
    print(f"\n학습 효과 검증:")
    trial_2plus = df[df['trial_number'] > 1]
    prev_fast = trial_2plus[trial_2plus['previous_choice'] == 'Fast']
    prev_relax = trial_2plus[trial_2plus['previous_choice'] == 'Relax']

    fast_after_fast = (prev_fast['selected_route'] == 'Fast').mean()
    fast_after_relax = (prev_relax['selected_route'] == 'Fast').mean()

    difference = fast_after_relax - fast_after_fast

    print(f"  이전 Fast 후 Fast 선택률: {fast_after_fast:.2%}")
    print(f"  이전 Relax 후 Fast 선택률: {fast_after_relax:.2%}")
    print(f"  차이: {difference:.2%}p")
    print(f"  β4={config.BETA_PREVIOUS_CHOICE:.2f} → 이전 Fast 후 Fast 확률 감소 예상")

    if difference > 0:
        print(f"  [OK] 학습 효과 정상 (이전 Fast -> 다음 Fast 확률 감소)")
    else:
        print(f"  [WARNING] 학습 효과 이상 (검토 필요)")

    # ===== MINOR FIX #9: 범위 검증 추가 =====
    print(f"\n범위 검증:")
    satisfaction_valid = ((df['satisfaction_score'] >= 0) & (df['satisfaction_score'] <= 5)).all()
    decision_valid = (df['decision_time'] >= config.DECISION_TIME_MIN).all()
    print(f"  satisfaction_score [0, 5]: {satisfaction_valid}")
    print(f"  decision_time >= {config.DECISION_TIME_MIN}: {decision_valid}")

    print("\n[OK] 검증 완료")


def save_complete_data(df, output_base='data/synthetic_data'):
    """
    완성된 데이터를 Parquet 및 CSV로 저장

    Args:
        df: 완성된 DataFrame
        output_base: 출력 파일 기본 경로 (확장자 제외)
    """
    if not os.path.isabs(output_base):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_base = os.path.join(base_dir, output_base)

    # Parquet 저장
    parquet_path = output_base + '.parquet'
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    parquet_size = os.path.getsize(parquet_path) / 1024 / 1024
    print(f"\n[OK] Parquet 저장: {parquet_path}")
    print(f"  크기: {parquet_size:.1f} MB")

    # CSV 저장 (호환성)
    csv_path = output_base + '.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    csv_size = os.path.getsize(csv_path) / 1024 / 1024
    print(f"[OK] CSV 저장: {csv_path}")
    print(f"  크기: {csv_size:.1f} MB")

    print(f"\n전체 데이터: {len(df):,} rows x {len(df.columns)} columns")


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("DAY 3: 선택 행동 모델링 및 학습 효과 구현")
    print("=" * 50)

    # 1. DAY 2 데이터 로드
    df_trials = load_trials_data()

    # 2. 선택 행동 모델링
    df_complete = process_all_trials(df_trials)

    # 3. 검증
    validate_complete_data(df_complete)

    # 4. 저장
    save_complete_data(df_complete)

    print("\n" + "=" * 50)
    print("DAY 3 작업 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
