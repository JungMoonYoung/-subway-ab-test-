"""
통합 데이터 생성 스크립트: 동적 혼잡도 피드백 시스템

DAY 1 (사용자) + DAY 2 (trial) + DAY 3 (선택) 통합
Trial별 순차 처리로 혼잡도 피드백 루프 구현
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from scipy.special import expit

# config.py 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Random Seed 설정
np.random.seed(config.RANDOM_SEED)
print(f"[SEED] Random seed 설정: {config.RANDOM_SEED}")


def load_users(file_path='data/users_base.csv'):
    """사용자 데이터 로드"""
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, file_path)

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except FileNotFoundError:
        raise FileNotFoundError(f"사용자 데이터 파일을 찾을 수 없습니다: {file_path}")

    if len(df) == 0:
        raise ValueError(f"빈 데이터 파일입니다: {file_path}")

    required_cols = ['user_id', 'assigned_group', 'personality_type', 'travel_frequency']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")

    print(f"[OK] 사용자 데이터 로드: {len(df):,}명")
    return df


def encode_personality(personality_type):
    """Personality type을 숫자로 인코딩"""
    mapping = {
        'efficiency-oriented': 1,
        'comfort-oriented': -1,
        'neutral': 0
    }
    if isinstance(personality_type, pd.Series):
        return personality_type.map(mapping)
    else:
        return mapping.get(personality_type, 0)


def generate_trial_base_data(df_users, trial_number):
    """
    특정 trial의 기본 데이터 생성 (혼잡도 제외)

    Returns:
        pd.DataFrame: trial 기본 데이터 (혼잡도는 나중에 추가)
    """
    num_users = len(df_users)

    trial_data = df_users.copy()
    trial_data['trial_number'] = trial_number

    # 날짜
    base_date = datetime.strptime(config.BASE_DATE, "%Y-%m-%d")
    trial_data['created_at'] = base_date + timedelta(
        days=(trial_number - 1) * config.TRIAL_INTERVAL_DAYS
    )
    trial_data['days_since_first'] = (trial_number - 1) * config.TRIAL_INTERVAL_DAYS

    # time_pressure (사용자별 baseline + 랜덤 변동)
    if trial_number == 1:
        # Trial 1: baseline 생성
        baseline = np.random.normal(
            loc=config.TIME_PRESSURE_BASELINE_MEAN,
            scale=config.TIME_PRESSURE_BASELINE_STD,
            size=num_users
        )
        trial_data['time_pressure_baseline'] = baseline
    # Note: Trial 2+는 이전 trial에서 복사

    # 랜덤 변동
    random_noise = np.random.normal(
        loc=0,
        scale=config.TIME_PRESSURE_NOISE_STD,
        size=num_users
    )
    time_pressure_float = trial_data.get('time_pressure_baseline', 1.0) + random_noise
    time_pressure_float = np.clip(time_pressure_float, 0, 2)
    trial_data['time_pressure'] = np.round(time_pressure_float).astype(int)

    # 경로 시간
    trial_data['route_time_fast'] = np.maximum(
        np.random.normal(config.FAST_TIME_MEAN, config.FAST_TIME_STD, size=num_users),
        config.MIN_ROUTE_TIME_FAST
    )
    trial_data['route_time_relax'] = np.maximum(
        np.random.normal(config.RELAX_TIME_MEAN, config.RELAX_TIME_STD, size=num_users),
        config.MIN_ROUTE_TIME_RELAX
    )

    return trial_data


def calculate_dynamic_congestion(trial_number, previous_trial_data=None):
    """
    동적 혼잡도 계산: 이전 trial 선택 비율에 따라 결정

    Args:
        trial_number: 현재 trial 번호
        previous_trial_data: 이전 trial DataFrame (selected_route 포함)

    Returns:
        tuple: (congestion_fast, congestion_relax) - 각각 scalar 값
    """
    if trial_number == 1 or previous_trial_data is None:
        # Trial 1: 기본 혼잡도
        congestion_fast = config.BASE_CONGESTION_FAST
        congestion_relax = config.BASE_CONGESTION_RELAX
    else:
        # Trial 2+: 이전 선택 비율에 따라 동적 계산
        prev_fast_ratio = (previous_trial_data['selected_route'] == 'Fast').mean()
        prev_relax_ratio = 1 - prev_fast_ratio

        # 혼잡도 = 기본값 + (선택 비율 × 배수)
        congestion_fast = config.BASE_CONGESTION_FAST + (prev_fast_ratio * config.CONGESTION_MULTIPLIER)
        congestion_relax = config.BASE_CONGESTION_RELAX + (prev_relax_ratio * config.CONGESTION_MULTIPLIER)

        # 범위 제한
        congestion_fast = np.clip(congestion_fast, config.MIN_CONGESTION, config.MAX_CONGESTION)
        congestion_relax = np.clip(congestion_relax, config.MIN_CONGESTION, config.MAX_CONGESTION)

    return congestion_fast, congestion_relax


def add_congestion_to_trial(trial_data, congestion_fast_mean, congestion_relax_mean):
    """
    Trial 데이터에 혼잡도 추가 (개인별 랜덤 노이즈 포함)

    Args:
        trial_data: Trial DataFrame
        congestion_fast_mean: Fast 평균 혼잡도
        congestion_relax_mean: Relax 평균 혼잡도
    """
    num_users = len(trial_data)

    # 개인별 랜덤 노이즈 추가 (더 현실적)
    trial_data['congestion_fast'] = np.maximum(
        np.random.normal(congestion_fast_mean, config.CONGESTION_NOISE_STD, size=num_users),
        config.MIN_CONGESTION
    )
    trial_data['congestion_relax'] = np.maximum(
        np.random.normal(congestion_relax_mean, config.CONGESTION_NOISE_STD, size=num_users),
        config.MIN_CONGESTION
    )

    # 혼잡도에 따른 시간 지연
    trial_data['delay_fast'] = trial_data['congestion_fast'] * config.DELAY_FACTOR
    trial_data['delay_relax'] = trial_data['congestion_relax'] * config.DELAY_FACTOR

    return trial_data


def calculate_choice_probability(trial_data, previous_trial_data=None):
    """
    로지스틱 회귀 기반 Fast Route 선택 확률 계산

    β5 (혼잡도 경험) 추가!
    """
    # β0: 그룹별 절편
    beta_0 = np.where(trial_data['assigned_group'] == 'A', config.BETA_0_A, config.BETA_0_B)

    # β1: time_pressure
    beta_1_term = config.BETA_TIME_PRESSURE * trial_data['time_pressure']

    # β2: personality
    personality_encoded = encode_personality(trial_data['personality_type'])
    beta_2_term = config.BETA_PERSONALITY * personality_encoded

    # β3: time_difference
    time_difference = np.abs(trial_data['route_time_fast'] - trial_data['route_time_relax'])
    beta_3_term = config.BETA_TIME_DIFF * time_difference

    # β4: previous_choice
    if previous_trial_data is not None:
        previous_choice = trial_data['user_id'].map(
            previous_trial_data.set_index('user_id')['selected_route']
        ).fillna('None')
        previous_choice_encoded = np.where(previous_choice == 'Fast', 1, 0)
    else:
        previous_choice_encoded = np.zeros(len(trial_data))

    beta_4_term = config.BETA_PREVIOUS_CHOICE * previous_choice_encoded

    # β5: congestion_experience (NEW!)
    # 이전 trial에서 경험한 혼잡도
    if previous_trial_data is not None:
        # 이전에 선택한 경로의 혼잡도
        prev_data_indexed = previous_trial_data.set_index('user_id')
        previous_choice_series = trial_data['user_id'].map(prev_data_indexed['selected_route']).fillna('None')

        # Fast 선택했으면 Fast 혼잡도, Relax 선택했으면 Relax 혼잡도
        prev_congestion_fast = trial_data['user_id'].map(prev_data_indexed['congestion_fast']).fillna(0)
        prev_congestion_relax = trial_data['user_id'].map(prev_data_indexed['congestion_relax']).fillna(0)

        experienced_congestion = np.where(
            previous_choice_series == 'Fast',
            prev_congestion_fast,
            np.where(previous_choice_series == 'Relax', prev_congestion_relax, 0)
        )
    else:
        experienced_congestion = np.zeros(len(trial_data))

    beta_5_term = config.BETA_CONGESTION * experienced_congestion

    # 노이즈
    noise = np.random.normal(0, config.NOISE_STD, size=len(trial_data))

    # Logit 계산
    logit = beta_0 + beta_1_term + beta_2_term + beta_3_term + beta_4_term + beta_5_term + noise

    # Sigmoid
    prob_fast = expit(logit)

    return prob_fast


def generate_route_choice(trial_data, previous_trial_data=None):
    """확률 기반 경로 선택"""
    prob_fast = calculate_choice_probability(trial_data, previous_trial_data)

    random_values = np.random.random(size=len(trial_data))
    selected_route = np.where(random_values < prob_fast, 'Fast', 'Relax')

    return pd.Series(selected_route, index=trial_data.index)


def generate_satisfaction_score(trial_data):
    """만족도 생성 (혼잡도 패널티 포함)"""
    base_score = config.SATISFACTION_BASE

    # 매칭 보너스
    match_bonus = np.zeros(len(trial_data))

    mask_efficiency_fast = (trial_data['personality_type'] == 'efficiency-oriented') & (trial_data['selected_route'] == 'Fast')
    match_bonus[mask_efficiency_fast] = config.SATISFACTION_MATCH_BONUS_STRONG

    mask_comfort_relax = (trial_data['personality_type'] == 'comfort-oriented') & (trial_data['selected_route'] == 'Relax')
    match_bonus[mask_comfort_relax] = config.SATISFACTION_MATCH_BONUS_STRONG

    mask_neutral = trial_data['personality_type'] == 'neutral'
    match_bonus[mask_neutral] = config.SATISFACTION_MATCH_BONUS_NEUTRAL

    # 압박 패널티
    pressure_penalty = np.zeros(len(trial_data))
    mask_urgent_relax = (trial_data['time_pressure'] == 0) & (trial_data['selected_route'] == 'Relax')
    pressure_penalty[mask_urgent_relax] = config.SATISFACTION_PRESSURE_PENALTY

    # 혼잡도 패널티 (NEW!)
    congestion_penalty = np.zeros(len(trial_data))

    # Fast 선택 시: 혼잡도가 임계값 초과하면 패널티
    mask_fast = trial_data['selected_route'] == 'Fast'
    congestion_excess_fast = np.maximum(
        trial_data.loc[mask_fast, 'congestion_fast'] - config.SATISFACTION_CONGESTION_THRESHOLD_FAST,
        0
    )
    congestion_penalty[mask_fast] = congestion_excess_fast * config.SATISFACTION_CONGESTION_PENALTY_FACTOR

    # Relax 선택 시: 혼잡도가 임계값 초과하면 패널티
    mask_relax = trial_data['selected_route'] == 'Relax'
    congestion_excess_relax = np.maximum(
        trial_data.loc[mask_relax, 'congestion_relax'] - config.SATISFACTION_CONGESTION_THRESHOLD_RELAX,
        0
    )
    congestion_penalty[mask_relax] = congestion_excess_relax * config.SATISFACTION_CONGESTION_PENALTY_FACTOR

    # 랜덤 노이즈
    noise = np.random.normal(0, config.SATISFACTION_NOISE_STD, size=len(trial_data))

    # 최종 만족도
    satisfaction = base_score + match_bonus + pressure_penalty + congestion_penalty + noise
    satisfaction = np.clip(satisfaction, 0, 5)

    return satisfaction


def generate_decision_time(trial_data):
    """의사결정 시간 생성"""
    base_time = np.random.normal(config.DECISION_TIME_MEAN, config.DECISION_TIME_STD, size=len(trial_data))
    pressure_effect = (trial_data['time_pressure'] - 1) * config.DECISION_TIME_PRESSURE_EFFECT
    decision_time = base_time + pressure_effect
    decision_time = np.maximum(decision_time, config.DECISION_TIME_MIN)

    return decision_time


def generate_all_trials(df_users):
    """
    전체 Trial 생성 (동적 혼잡도 피드백 포함)

    Trial 1 → 선택 → 혼잡도 계산 → Trial 2 → 선택 → ...
    """
    print("\n=== 동적 혼잡도 시스템으로 데이터 생성 ===")

    all_trials = []
    previous_trial = None

    for trial_num in range(1, config.NUM_TRIALS + 1):
        print(f"\nTrial {trial_num}/{config.NUM_TRIALS} 처리 중...")

        # 1. 기본 데이터 생성
        trial_data = generate_trial_base_data(df_users, trial_num)

        # time_pressure_baseline 복사 (Trial 2+)
        if trial_num > 1 and previous_trial is not None:
            baseline_map = previous_trial.set_index('user_id')['time_pressure_baseline']
            trial_data['time_pressure_baseline'] = trial_data['user_id'].map(baseline_map)

        # 2. 동적 혼잡도 계산
        congestion_fast_mean, congestion_relax_mean = calculate_dynamic_congestion(
            trial_num,
            previous_trial
        )

        print(f"  혼잡도: Fast {congestion_fast_mean:.1f}%, Relax {congestion_relax_mean:.1f}%")

        # 3. 혼잡도 추가
        trial_data = add_congestion_to_trial(trial_data, congestion_fast_mean, congestion_relax_mean)

        # 4. 경로 선택 생성
        trial_data['selected_route'] = generate_route_choice(trial_data, previous_trial)

        # 선택 분포 출력
        fast_ratio = (trial_data['selected_route'] == 'Fast').mean()
        print(f"  선택: Fast {fast_ratio*100:.2f}%, Relax {(1-fast_ratio)*100:.2f}%")

        # 5. previous_choice 설정
        if previous_trial is not None:
            prev_choice_map = previous_trial.set_index('user_id')['selected_route']
            trial_data['previous_choice'] = trial_data['user_id'].map(prev_choice_map)
        else:
            trial_data['previous_choice'] = None

        # 6. 만족도 생성
        trial_data['satisfaction_score'] = generate_satisfaction_score(trial_data)

        # 7. 의사결정 시간 생성
        trial_data['decision_time'] = generate_decision_time(trial_data)

        all_trials.append(trial_data)
        previous_trial = trial_data

        print(f"  [OK] Trial {trial_num} 완료")

    # 결합
    df_complete = pd.concat(all_trials, ignore_index=True)
    print(f"\n[OK] 전체 데이터 생성 완료: {len(df_complete):,} rows")

    return df_complete


def validate_data(df):
    """데이터 검증"""
    print("\n=== 데이터 검증 ===")

    # Trial별 선택 분포
    print("\nTrial별 Fast Route 선택률:")
    for trial_num in range(1, config.NUM_TRIALS + 1):
        trial_data = df[df['trial_number'] == trial_num]
        fast_ratio = (trial_data['selected_route'] == 'Fast').mean()
        avg_congestion_fast = trial_data['congestion_fast'].mean()
        avg_congestion_relax = trial_data['congestion_relax'].mean()
        print(f"  Trial {trial_num}: Fast {fast_ratio*100:.2f}% "
              f"(혼잡도 Fast={avg_congestion_fast:.1f}%, Relax={avg_congestion_relax:.1f}%)")

    # 전체 선택 분포
    print(f"\n전체 선택 분포:")
    choice_counts = df['selected_route'].value_counts(normalize=True)
    for choice, ratio in choice_counts.items():
        print(f"  {choice}: {ratio*100:.2f}%")

    # 만족도 통계
    print(f"\n만족도 통계:")
    print(f"  평균: {df['satisfaction_score'].mean():.2f}")
    print(f"  표준편차: {df['satisfaction_score'].std():.2f}")

    # 학습 효과 검증
    print(f"\n학습 효과 검증 (Trial 2-5):")
    trial_2plus = df[df['trial_number'] > 1]
    prev_fast = trial_2plus[trial_2plus['previous_choice'] == 'Fast']
    prev_relax = trial_2plus[trial_2plus['previous_choice'] == 'Relax']

    if len(prev_fast) > 0 and len(prev_relax) > 0:
        fast_after_fast = (prev_fast['selected_route'] == 'Fast').mean()
        fast_after_relax = (prev_relax['selected_route'] == 'Fast').mean()
        print(f"  이전 Fast 후 Fast: {fast_after_fast*100:.2f}%")
        print(f"  이전 Relax 후 Fast: {fast_after_relax*100:.2f}%")
        print(f"  차이: {(fast_after_relax - fast_after_fast)*100:.2f}%p")

    print("\n[OK] 검증 완료")


def save_data(df, output_base='data/synthetic_data_dynamic'):
    """데이터 저장"""
    if not os.path.isabs(output_base):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_base = os.path.join(base_dir, output_base)

    # Parquet
    parquet_path = output_base + '.parquet'
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    parquet_size = os.path.getsize(parquet_path) / 1024 / 1024
    print(f"\n[OK] Parquet 저장: {parquet_path}")
    print(f"  크기: {parquet_size:.1f} MB")

    # CSV
    csv_path = output_base + '.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    csv_size = os.path.getsize(csv_path) / 1024 / 1024
    print(f"[OK] CSV 저장: {csv_path}")
    print(f"  크기: {csv_size:.1f} MB")

    print(f"\n전체 데이터: {len(df):,} rows x {len(df.columns)} columns")


def main():
    """메인 실행"""
    print("=" * 60)
    print("통합 데이터 생성: 동적 혼잡도 피드백 시스템")
    print("=" * 60)

    # 1. 사용자 로드
    df_users = load_users()

    # 2. 전체 Trial 생성 (동적 혼잡도)
    df_complete = generate_all_trials(df_users)

    # 3. 검증
    validate_data(df_complete)

    # 4. 저장
    save_data(df_complete)

    print("\n" + "=" * 60)
    print("동적 혼잡도 데이터 생성 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
