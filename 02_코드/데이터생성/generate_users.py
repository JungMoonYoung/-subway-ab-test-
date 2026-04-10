"""
사용자 기본 정보 생성 모듈
100,000명의 사용자를 생성하고 A/B 그룹 배정 및 특성 할당
"""

import numpy as np
import pandas as pd
import sys
import os

# config.py 임포트를 위한 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_users():
    """
    사용자 기본 정보 생성

    Returns:
        pd.DataFrame: 사용자 정보가 담긴 DataFrame
    """
    # Random seed 설정 (재현성)
    np.random.seed(config.RANDOM_SEED)

    print(f"사용자 {config.NUM_USERS:,}명 생성 중...")

    # 사용자 ID 생성
    user_ids = np.arange(1, config.NUM_USERS + 1)

    # A/B 그룹 랜덤 배정 (Bernoulli 0.5)
    assigned_groups = np.random.choice(['A', 'B'], size=config.NUM_USERS, p=[0.5, 0.5])

    # Personality type 할당
    personality_types = np.random.choice(
        ['efficiency-oriented', 'comfort-oriented', 'neutral'],
        size=config.NUM_USERS,
        p=[
            config.PERSONALITY_EFFICIENCY_RATIO,
            config.PERSONALITY_COMFORT_RATIO,
            config.PERSONALITY_NEUTRAL_RATIO
        ]
    )

    # Travel frequency 할당
    travel_frequencies = np.random.choice(
        ['daily', 'weekly', 'rarely'],
        size=config.NUM_USERS,
        p=[
            config.TRAVEL_DAILY_RATIO,
            config.TRAVEL_WEEKLY_RATIO,
            config.TRAVEL_RARELY_RATIO
        ]
    )

    # DataFrame 생성
    df = pd.DataFrame({
        'user_id': user_ids,
        'assigned_group': assigned_groups,
        'personality_type': personality_types,
        'travel_frequency': travel_frequencies
    })

    print("[OK] 사용자 생성 완료")
    return df


def validate_users(df):
    """
    생성된 사용자 데이터 검증

    Args:
        df: 사용자 DataFrame
    """
    print("\n=== 데이터 검증 ===")

    # 1. 결측값 확인
    missing = df.isnull().sum()
    print(f"결측값: {missing.sum()} (예상: 0)")
    assert missing.sum() == 0, "결측값이 존재합니다!"

    # 2. 중복 확인
    duplicates = df.duplicated(subset=['user_id']).sum()
    print(f"중복 user_id: {duplicates} (예상: 0)")
    assert duplicates == 0, "중복된 user_id가 존재합니다!"

    # 3. 그룹 배정 균형 확인
    group_counts = df['assigned_group'].value_counts()
    print(f"\n그룹 배정:")
    print(f"  A그룹: {group_counts['A']:,}명")
    print(f"  B그룹: {group_counts['B']:,}명")
    print(f"  차이: {abs(group_counts['A'] - group_counts['B']):,}명")

    # 4. Personality type 분포 확인
    personality_counts = df['personality_type'].value_counts(normalize=True)
    print(f"\nPersonality Type 분포:")
    for ptype, ratio in personality_counts.items():
        print(f"  {ptype}: {ratio:.2%}")

    # 5. Travel frequency 분포 확인
    travel_counts = df['travel_frequency'].value_counts(normalize=True)
    print(f"\nTravel Frequency 분포:")
    for freq, ratio in travel_counts.items():
        print(f"  {freq}: {ratio:.2%}")

    print("\n[OK] 검증 완료: 모든 테스트 통과")


def save_users(df, output_path='data/users_base.csv'):
    """
    사용자 데이터를 CSV 파일로 저장

    Args:
        df: 사용자 DataFrame
        output_path: 저장 경로
    """
    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(output_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, output_path)

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 데이터 저장 완료: {output_path}")
    print(f"  크기: {len(df):,} rows x {len(df.columns)} columns")


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("사용자 기본 정보 생성 시작")
    print("=" * 50)

    # 1. 사용자 생성
    df_users = generate_users()

    # 2. 데이터 검증
    validate_users(df_users)

    # 3. 저장
    save_users(df_users)

    # 4. 샘플 데이터 출력
    print("\n=== 샘플 데이터 (처음 10명) ===")
    print(df_users.head(10))

    print("\n" + "=" * 50)
    print("DAY 1 작업 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
