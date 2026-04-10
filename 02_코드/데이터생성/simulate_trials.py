"""
반복 측정 시뮬레이션 모듈 (수정 버전)
각 사용자당 5회 반복 측정 데이터 생성 (선택 행동 제외)

수정 사항:
- Random seed를 모듈 최상단으로 이동 (재현성 보장)
- Magic numbers를 config.py로 이동
- 에러 처리 추가
- 크기 검증 assert 추가
- 날짜 로직 config.py 사용
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# config.py 임포트를 위한 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ===== CRITICAL FIX #2: Random Seed를 모듈 최상단에 배치 =====
np.random.seed(config.RANDOM_SEED)
print(f"[SEED] Random seed 설정: {config.RANDOM_SEED}")


def load_users(file_path='data/users_base.csv'):
    """
    DAY 1에서 생성한 사용자 데이터 로드

    Args:
        file_path: 사용자 데이터 파일 경로

    Returns:
        pd.DataFrame: 사용자 정보
            필수 컬럼: user_id, assigned_group, personality_type, travel_frequency

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        ValueError: 빈 파일이거나 필수 컬럼이 누락된 경우
    """
    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, file_path)

    # ===== MAJOR FIX #6: 에러 처리 추가 =====
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except FileNotFoundError:
        raise FileNotFoundError(f"사용자 데이터 파일을 찾을 수 없습니다: {file_path}")

    # 빈 파일 검증
    if len(df) == 0:
        raise ValueError(f"빈 데이터 파일입니다: {file_path}")

    # 필수 컬럼 검증
    required_cols = ['user_id', 'assigned_group', 'personality_type', 'travel_frequency']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")

    print(f"[OK] 사용자 데이터 로드: {len(df):,}명")
    return df


def generate_time_pressure_baseline(num_users):
    """
    사용자별 time_pressure baseline tendency 생성
    이 baseline은 개인의 평소 급박함 정도를 나타냄

    Args:
        num_users: 사용자 수 (int)

    Returns:
        np.ndarray: 사용자별 baseline, shape (num_users,), 범위 [0, 2]
    """
    # ===== MAJOR FIX #5: Magic Number를 config.py에서 가져옴 =====
    baseline = np.random.normal(
        loc=config.TIME_PRESSURE_BASELINE_MEAN,
        scale=config.TIME_PRESSURE_BASELINE_STD,
        size=num_users
    )
    # 0~2 범위로 클리핑
    baseline = np.clip(baseline, 0, 2)
    return baseline


def generate_trial_data(df_users, trial_number, time_pressure_baselines):
    """
    특정 trial에 대한 데이터 생성

    Args:
        df_users: 사용자 DataFrame
            필수 컬럼: user_id, assigned_group, personality_type, travel_frequency
        trial_number: 현재 trial 번호 (1~5)
        time_pressure_baselines: 사용자별 time_pressure baseline
            shape (num_users,), np.ndarray

    Returns:
        pd.DataFrame: 해당 trial의 데이터 (num_users rows)
            추가 컬럼: trial_number, days_since_first, created_at, time_pressure,
                      route_time_fast, route_time_relax, congestion_fast,
                      congestion_relax, actual_time_fast, actual_time_relax

    Raises:
        AssertionError: baseline 크기가 df_users와 불일치 시
    """
    num_users = len(df_users)

    # ===== CRITICAL FIX #3: 크기 검증 =====
    assert len(time_pressure_baselines) == num_users, \
        f"Baseline 크기 불일치: {len(time_pressure_baselines)} != {num_users}"

    # 1. Trial 기본 정보
    trial_data = df_users.copy()
    trial_data['trial_number'] = trial_number
    trial_data['days_since_first'] = trial_number - 1

    # ===== MAJOR FIX #4: 날짜 로직을 config.py에서 가져옴 =====
    base_date = datetime.strptime(config.BASE_DATE, "%Y-%m-%d")
    trial_data['created_at'] = base_date + timedelta(
        days=(trial_number - 1) * config.TRIAL_INTERVAL_DAYS
    )

    # 2. time_pressure 생성 (개인 baseline + 랜덤 변동)
    random_noise = np.random.normal(
        loc=0,
        scale=config.TIME_PRESSURE_NOISE_STD,
        size=num_users
    )
    time_pressure_float = time_pressure_baselines + random_noise
    # 0, 1, 2 중 하나로 반올림
    trial_data['time_pressure'] = np.clip(np.round(time_pressure_float), 0, 2).astype(int)

    # 3. Fast Route 시간 샘플링
    trial_data['route_time_fast'] = np.random.normal(
        loc=config.FAST_TIME_MEAN,
        scale=config.FAST_TIME_STD,
        size=num_users
    )
    trial_data['route_time_fast'] = np.maximum(
        trial_data['route_time_fast'],
        config.MIN_ROUTE_TIME_FAST
    )

    # 4. Relax Route 시간 샘플링
    trial_data['route_time_relax'] = np.random.normal(
        loc=config.RELAX_TIME_MEAN,
        scale=config.RELAX_TIME_STD,
        size=num_users
    )
    trial_data['route_time_relax'] = np.maximum(
        trial_data['route_time_relax'],
        config.MIN_ROUTE_TIME_RELAX
    )

    # 5. Fast Route 혼잡도 샘플링
    trial_data['congestion_fast'] = np.random.normal(
        loc=config.FAST_CONGESTION_MEAN,
        scale=config.FAST_CONGESTION_STD,
        size=num_users
    )
    trial_data['congestion_fast'] = np.maximum(
        trial_data['congestion_fast'],
        config.MIN_CONGESTION_FAST
    )

    # 6. Relax Route 혼잡도 샘플링
    trial_data['congestion_relax'] = np.random.normal(
        loc=config.RELAX_CONGESTION_MEAN,
        scale=config.RELAX_CONGESTION_STD,
        size=num_users
    )
    trial_data['congestion_relax'] = np.maximum(
        trial_data['congestion_relax'],
        config.MIN_CONGESTION_RELAX
    )

    # 7. 시간-혼잡도 상관관계 반영 (actual_time)
    # actual_time = base_time + (congestion - 100) * DELAY_FACTOR
    trial_data['actual_time_fast'] = trial_data['route_time_fast'] + \
        (trial_data['congestion_fast'] - 100) * config.DELAY_FACTOR

    trial_data['actual_time_relax'] = trial_data['route_time_relax'] + \
        (trial_data['congestion_relax'] - 100) * config.DELAY_FACTOR

    return trial_data


def simulate_all_trials(df_users):
    """
    모든 사용자의 5회 반복 측정 시뮬레이션

    Args:
        df_users: 사용자 DataFrame

    Returns:
        pd.DataFrame: 전체 trial 데이터 (num_users * NUM_TRIALS rows)
    """
    print(f"\n5회 반복 측정 시뮬레이션 시작...")
    print(f"예상 총 rows: {len(df_users) * config.NUM_TRIALS:,}")

    # 사용자별 time_pressure baseline 생성 (한 번만)
    time_pressure_baselines = generate_time_pressure_baseline(len(df_users))

    all_trials = []

    for trial_num in range(1, config.NUM_TRIALS + 1):
        print(f"  Trial {trial_num}/{config.NUM_TRIALS} 생성 중...")
        trial_data = generate_trial_data(df_users, trial_num, time_pressure_baselines)
        all_trials.append(trial_data)

    # 모든 trial 데이터 결합
    df_all = pd.concat(all_trials, ignore_index=True)

    print(f"[OK] 시뮬레이션 완료: {len(df_all):,} rows 생성")
    return df_all


def validate_trials(df):
    """
    생성된 trial 데이터 검증

    Args:
        df: trial DataFrame

    Raises:
        AssertionError: 검증 실패 시
    """
    print("\n=== 데이터 검증 ===")

    # 1. 결측값 확인
    missing = df.isnull().sum()
    print(f"결측값: {missing.sum()} (예상: 0)")
    assert missing.sum() == 0, f"결측값이 존재합니다!\n{missing[missing > 0]}"

    # 2. 데이터 크기 확인
    expected_rows = config.NUM_USERS * config.NUM_TRIALS
    actual_rows = len(df)
    print(f"데이터 크기: {actual_rows:,} rows (예상: {expected_rows:,})")
    assert actual_rows == expected_rows, f"데이터 크기 불일치!"

    # 3. trial_number 범위 확인
    trial_min = df['trial_number'].min()
    trial_max = df['trial_number'].max()
    print(f"trial_number 범위: {trial_min}~{trial_max} (예상: 1~{config.NUM_TRIALS})")
    assert trial_min == 1 and trial_max == config.NUM_TRIALS, "trial_number 범위 오류!"

    # 4. time_pressure 분포 확인
    tp_counts = df['time_pressure'].value_counts(normalize=True).sort_index()
    print(f"\ntime_pressure 분포:")
    for tp, ratio in tp_counts.items():
        print(f"  {tp}: {ratio:.2%}")

    # 5. 경로 시간 통계
    print(f"\n경로 시간 통계 (분):")
    print(f"  Fast Route: 평균 {df['route_time_fast'].mean():.2f}, "
          f"표준편차 {df['route_time_fast'].std():.2f}")
    print(f"  Relax Route: 평균 {df['route_time_relax'].mean():.2f}, "
          f"표준편차 {df['route_time_relax'].std():.2f}")

    # 6. 혼잡도 통계
    print(f"\n혼잡도 통계 (%):")
    print(f"  Fast Route: 평균 {df['congestion_fast'].mean():.1f}%, "
          f"표준편차 {df['congestion_fast'].std():.1f}%")
    print(f"  Relax Route: 평균 {df['congestion_relax'].mean():.1f}%, "
          f"표준편차 {df['congestion_relax'].std():.1f}%")

    # 7. 범위 검증
    print(f"\n범위 검증:")
    print(f"  route_time_fast >= {config.MIN_ROUTE_TIME_FAST}: "
          f"{(df['route_time_fast'] >= config.MIN_ROUTE_TIME_FAST).all()}")
    print(f"  route_time_relax >= {config.MIN_ROUTE_TIME_RELAX}: "
          f"{(df['route_time_relax'] >= config.MIN_ROUTE_TIME_RELAX).all()}")
    print(f"  congestion_fast >= {config.MIN_CONGESTION_FAST}: "
          f"{(df['congestion_fast'] >= config.MIN_CONGESTION_FAST).all()}")
    print(f"  congestion_relax >= {config.MIN_CONGESTION_RELAX}: "
          f"{(df['congestion_relax'] >= config.MIN_CONGESTION_RELAX).all()}")

    print("\n[OK] 검증 완료: 모든 테스트 통과")


def display_sample_users(df, num_samples=3):
    """
    샘플 사용자의 5회 데이터 출력

    Args:
        df: trial DataFrame
        num_samples: 출력할 사용자 수
    """
    print(f"\n=== 샘플 사용자 {num_samples}명의 {config.NUM_TRIALS}회 데이터 ===")

    sample_user_ids = df['user_id'].unique()[:num_samples]

    for user_id in sample_user_ids:
        user_data = df[df['user_id'] == user_id].sort_values('trial_number')
        print(f"\n[User {user_id}] - {user_data.iloc[0]['assigned_group']}그룹, "
              f"{user_data.iloc[0]['personality_type']}")
        print(user_data[['trial_number', 'time_pressure', 'route_time_fast',
                         'route_time_relax', 'congestion_fast', 'congestion_relax']].to_string(index=False))


def save_trials(df, output_path='data/trials_data_partial.csv'):
    """
    trial 데이터를 CSV 파일로 저장

    Args:
        df: trial DataFrame
        output_path: 저장 경로
    """
    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(output_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, output_path)

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 데이터 저장 완료: {output_path}")
    print(f"  크기: {len(df):,} rows x {len(df.columns)} columns")
    print(f"  파일 크기: 약 {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("DAY 2: 반복 측정 시뮬레이션 시작 (수정 버전)")
    print("=" * 50)

    # 1. 사용자 데이터 로드
    df_users = load_users()

    # 2. 5회 반복 측정 시뮬레이션
    df_trials = simulate_all_trials(df_users)

    # 3. 데이터 검증
    validate_trials(df_trials)

    # 4. 샘플 사용자 출력
    display_sample_users(df_trials, num_samples=3)

    # 5. 저장
    save_trials(df_trials)

    print("\n" + "=" * 50)
    print("DAY 2 작업 완료! (모든 Critical/Major 이슈 수정)")
    print("=" * 50)
    print("\n다음 단계: DAY 3에서 선택 행동 모델링 추가")


if __name__ == "__main__":
    main()
