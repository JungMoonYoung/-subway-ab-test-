"""
DAY 4: 기초 통계 검정 구현

Two-Proportion Z-Test, Chi-square Test, Effect Size 계산
"""

import numpy as np
import pandas as pd
import sys
import os
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# config.py 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Random Seed 설정
np.random.seed(config.RANDOM_SEED)
print(f"[SEED] Random seed 설정: {config.RANDOM_SEED}")


def load_data(file_path='data/synthetic_data_dynamic.parquet'):
    """
    생성된 데이터 로드

    Args:
        file_path: 데이터 파일 경로

    Returns:
        pd.DataFrame: 전체 데이터
    """
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, file_path)

    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
    except FileNotFoundError:
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

    if len(df) == 0:
        raise ValueError(f"빈 데이터 파일입니다: {file_path}")

    print(f"[OK] 데이터 로드: {len(df):,} rows x {len(df.columns)} columns")
    return df


def two_proportion_ztest(df):
    """
    Two-Proportion Z-Test: A그룹 vs B그룹 Fast Route 선택 비율 비교

    H0: p_A = p_B (두 그룹의 Fast 선택 비율이 같다)
    H1: p_A ≠ p_B (두 그룹의 Fast 선택 비율이 다르다)

    Args:
        df: 전체 데이터 DataFrame

    Returns:
        dict: 검정 결과
            - p_A: A그룹 Fast 선택 비율
            - p_B: B그룹 Fast 선택 비율
            - diff: p_A - p_B
            - z_stat: z-통계량
            - p_value: p-value (양측 검정)
            - significant: 유의미 여부 (alpha=0.05)
    """
    print("\n=== Two-Proportion Z-Test ===")

    # A/B 그룹별 Fast 선택 집계
    group_a = df[df['assigned_group'] == 'A']
    group_b = df[df['assigned_group'] == 'B']

    count_a = len(group_a)
    count_b = len(group_b)

    fast_a = (group_a['selected_route'] == 'Fast').sum()
    fast_b = (group_b['selected_route'] == 'Fast').sum()

    p_a = fast_a / count_a
    p_b = fast_b / count_b

    print(f"A그룹: {fast_a:,} / {count_a:,} = {p_a:.4f}")
    print(f"B그룹: {fast_b:,} / {count_b:,} = {p_b:.4f}")
    print(f"차이: {p_a - p_b:.4f}")

    # Z-Test 수행
    counts = np.array([fast_a, fast_b])
    nobs = np.array([count_a, count_b])

    z_stat, p_value = proportions_ztest(counts, nobs, alternative='two-sided')

    significant = p_value < config.ALPHA

    print(f"\nz-통계량: {z_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"유의수준: {config.ALPHA}")
    print(f"결과: {'유의미함' if significant else '유의미하지 않음'}")

    return {
        'test': 'Two-Proportion Z-Test',
        'p_A': p_a,
        'p_B': p_b,
        'diff': p_a - p_b,
        'z_stat': z_stat,
        'p_value': p_value,
        'alpha': config.ALPHA,
        'significant': significant
    }


def chi_square_test(df):
    """
    Chi-square Test of Independence: 그룹(A/B) × 선택(Fast/Relax) 독립성 검정

    H0: 그룹과 선택이 독립이다
    H1: 그룹과 선택이 독립이 아니다

    Args:
        df: 전체 데이터 DataFrame

    Returns:
        dict: 검정 결과
            - chi2: 카이제곱 통계량
            - p_value: p-value
            - dof: 자유도
            - expected: 기대 빈도
            - observed: 관측 빈도
            - significant: 유의미 여부
    """
    print("\n=== Chi-square Test of Independence ===")

    # 교차표 생성
    contingency_table = pd.crosstab(
        df['assigned_group'],
        df['selected_route'],
        margins=False
    )

    print(f"\n관측 빈도:")
    print(contingency_table)

    # Chi-square 검정
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"\n기대 빈도:")
    print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

    significant = p_value < config.ALPHA

    print(f"\nχ² 통계량: {chi2:.4f}")
    print(f"자유도: {dof}")
    print(f"p-value: {p_value:.6f}")
    print(f"유의수준: {config.ALPHA}")
    print(f"결과: {'유의미함' if significant else '유의미하지 않음'}")

    return {
        'test': 'Chi-square Test',
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'expected': expected,
        'observed': contingency_table.values,
        'alpha': config.ALPHA,
        'significant': significant
    }


def cohens_h(p1, p2):
    """
    Cohen's h: 두 비율 간 효과 크기

    h = 2 * (arcsin(√p1) - arcsin(√p2))

    해석:
    - |h| < 0.2: small
    - 0.2 ≤ |h| < 0.5: medium
    - |h| ≥ 0.5: large

    Args:
        p1: 첫 번째 비율
        p2: 두 번째 비율

    Returns:
        float: Cohen's h 값
    """
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def effect_size_analysis(df):
    """
    Effect Size 계산 및 해석

    Args:
        df: 전체 데이터 DataFrame

    Returns:
        dict: 효과 크기 분석 결과
    """
    print("\n=== Effect Size (Cohen's h) ===")

    # A/B 그룹별 비율
    group_a = df[df['assigned_group'] == 'A']
    group_b = df[df['assigned_group'] == 'B']

    p_a = (group_a['selected_route'] == 'Fast').mean()
    p_b = (group_b['selected_route'] == 'Fast').mean()

    h = cohens_h(p_a, p_b)

    # 해석 (Cohen, 1988 기준)
    if abs(h) < config.COHENS_H_SMALL:
        interpretation = 'small'
    elif abs(h) < config.COHENS_H_MEDIUM:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    print(f"A그룹 Fast 비율: {p_a:.4f}")
    print(f"B그룹 Fast 비율: {p_b:.4f}")
    print(f"Cohen's h: {h:.4f}")
    print(f"해석: {interpretation} effect size")

    return {
        'test': 'Effect Size',
        'p_A': p_a,
        'p_B': p_b,
        'cohens_h': h,
        'interpretation': interpretation
    }


def confidence_intervals(df):
    """
    신뢰구간 계산: A/B 그룹별 Fast 선택 비율 95% CI

    Args:
        df: 전체 데이터 DataFrame

    Returns:
        dict: 신뢰구간 결과
    """
    print("\n=== Confidence Intervals (95%) ===")

    results = {}

    for group in ['A', 'B']:
        group_data = df[df['assigned_group'] == group]
        n = len(group_data)
        count = (group_data['selected_route'] == 'Fast').sum()
        p = count / n

        # Wilson score interval (더 정확)
        ci_low, ci_high = proportion_confint(count, n, alpha=1-0.95, method='wilson')

        print(f"\n{group}그룹:")
        print(f"  비율: {p:.4f}")
        print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  폭: {ci_high - ci_low:.4f}")

        results[f'group_{group}'] = {
            'n': n,
            'count': count,
            'proportion': p,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'ci_width': ci_high - ci_low
        }

    return results


def trial_level_analysis(df):
    """
    Trial별 선택 비율 분석 (동적 변화 확인)

    Args:
        df: 전체 데이터 DataFrame

    Returns:
        pd.DataFrame: Trial별 통계
    """
    print("\n=== Trial별 분석 ===")

    trial_stats = []

    for trial_num in range(1, config.NUM_TRIALS + 1):
        trial_data = df[df['trial_number'] == trial_num]

        # 전체 비율
        overall_fast = (trial_data['selected_route'] == 'Fast').mean()

        # 그룹별 비율
        group_a = trial_data[trial_data['assigned_group'] == 'A']
        group_b = trial_data[trial_data['assigned_group'] == 'B']

        fast_a = (group_a['selected_route'] == 'Fast').mean()
        fast_b = (group_b['selected_route'] == 'Fast').mean()

        # 평균 혼잡도
        avg_congestion_fast = trial_data['congestion_fast'].mean()
        avg_congestion_relax = trial_data['congestion_relax'].mean()

        trial_stats.append({
            'trial': trial_num,
            'overall_fast_rate': overall_fast,
            'group_A_fast_rate': fast_a,
            'group_B_fast_rate': fast_b,
            'diff_A_B': fast_a - fast_b,
            'avg_congestion_fast': avg_congestion_fast,
            'avg_congestion_relax': avg_congestion_relax
        })

    df_trial_stats = pd.DataFrame(trial_stats)

    print(df_trial_stats.to_string(index=False, float_format='%.4f'))

    return df_trial_stats


def personality_analysis(df):
    """
    Personality type별 선택 분석

    Args:
        df: 전체 데이터 DataFrame

    Returns:
        pd.DataFrame: Personality별 통계
    """
    print("\n=== Personality Type별 분석 ===")

    personality_stats = []

    for personality in ['efficiency-oriented', 'comfort-oriented', 'neutral']:
        pers_data = df[df['personality_type'] == personality]

        if len(pers_data) == 0:
            continue

        fast_rate = (pers_data['selected_route'] == 'Fast').mean()
        n = len(pers_data)

        # 그룹별
        group_a = pers_data[pers_data['assigned_group'] == 'A']
        group_b = pers_data[pers_data['assigned_group'] == 'B']

        fast_a = (group_a['selected_route'] == 'Fast').mean() if len(group_a) > 0 else np.nan
        fast_b = (group_b['selected_route'] == 'Fast').mean() if len(group_b) > 0 else np.nan

        personality_stats.append({
            'personality': personality,
            'n': n,
            'fast_rate': fast_rate,
            'group_A_fast_rate': fast_a,
            'group_B_fast_rate': fast_b
        })

    df_pers_stats = pd.DataFrame(personality_stats)

    print(df_pers_stats.to_string(index=False, float_format='%.4f'))

    return df_pers_stats


def save_results(results, output_path='analysis/basic_tests_results.csv'):
    """
    검정 결과 저장

    Args:
        results: 결과 딕셔너리 리스트
        output_path: 출력 파일 경로
    """
    if not os.path.isabs(output_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, output_path)

    # 딕셔너리 결과를 DataFrame으로 변환
    results_list = []

    for result in results:
        if isinstance(result, dict) and 'test' in result:
            results_list.append(result)

    if results_list:
        df_results = pd.DataFrame(results_list)
        df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[OK] 결과 저장: {output_path}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("DAY 4: 기초 통계 검정")
    print("=" * 60)

    # 1. 데이터 로드
    df = load_data()

    # 2. Two-Proportion Z-Test
    result_ztest = two_proportion_ztest(df)

    # 3. Chi-square Test
    result_chi2 = chi_square_test(df)

    # 4. Effect Size
    result_effect = effect_size_analysis(df)

    # 5. Confidence Intervals
    result_ci = confidence_intervals(df)

    # 6. Trial별 분석
    df_trial_stats = trial_level_analysis(df)

    # 7. Personality별 분석
    df_pers_stats = personality_analysis(df)

    # 8. 결과 저장
    results = [result_ztest, result_chi2, result_effect]
    save_results(results)

    # Trial별 결과 저장
    trial_output = 'analysis/trial_level_stats.csv'
    if not os.path.isabs(trial_output):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        trial_output = os.path.join(base_dir, trial_output)
    df_trial_stats.to_csv(trial_output, index=False, encoding='utf-8-sig')
    print(f"[OK] Trial별 통계 저장: {trial_output}")

    # Personality별 결과 저장
    pers_output = 'analysis/personality_stats.csv'
    if not os.path.isabs(pers_output):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pers_output = os.path.join(base_dir, pers_output)
    df_pers_stats.to_csv(pers_output, index=False, encoding='utf-8-sig')
    print(f"[OK] Personality별 통계 저장: {pers_output}")

    print("\n" + "=" * 60)
    print("기초 통계 검정 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
