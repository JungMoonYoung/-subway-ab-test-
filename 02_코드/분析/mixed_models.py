"""
DAY 4: Mixed-Effects Model 및 GEE 구현

반복 측정 데이터 분석
"""

import numpy as np
import pandas as pd
import sys
import os
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Autoregressive, Exchangeable
from statsmodels.stats.multitest import multipletests

# config.py 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Random Seed 설정
np.random.seed(config.RANDOM_SEED)
print(f"[SEED] Random seed 설정: {config.RANDOM_SEED}")


def load_data(file_path='data/synthetic_data_dynamic.parquet'):
    """데이터 로드"""
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

    print(f"[OK] 데이터 로드: {len(df):,} rows")
    return df


def prepare_data_for_modeling(df):
    """
    모델링을 위한 데이터 전처리

    Args:
        df: 원본 데이터

    Returns:
        pd.DataFrame: 전처리된 데이터
    """
    print("\n=== 데이터 전처리 ===")

    df_model = df.copy()

    # 종속변수: Fast 선택 = 1, Relax 선택 = 0
    df_model['choice_binary'] = (df_model['selected_route'] == 'Fast').astype(int)

    # 독립변수 인코딩
    # assigned_group: A=1, B=0
    df_model['group_numeric'] = (df_model['assigned_group'] == 'A').astype(int)

    # personality_type: efficiency=1, neutral=0, comfort=-1
    personality_map = {
        'efficiency-oriented': 1,
        'neutral': 0,
        'comfort-oriented': -1
    }
    df_model['personality_numeric'] = df_model['personality_type'].map(personality_map)

    # travel_frequency: daily=2, weekly=1, rarely=0
    travel_map = {
        'daily': 2,
        'weekly': 1,
        'rarely': 0
    }
    df_model['travel_numeric'] = df_model['travel_frequency'].map(travel_map)

    # 시간 차이
    df_model['time_diff'] = df_model['route_time_relax'] - df_model['route_time_fast']

    # 혼잡도 차이
    df_model['congestion_diff'] = df_model['congestion_fast'] - df_model['congestion_relax']

    # Trial number를 0부터 시작 (해석 용이)
    df_model['trial_index'] = df_model['trial_number'] - 1

    # 결측값 확인
    print(f"결측값 확인:")
    missing = df_model[['choice_binary', 'group_numeric', 'personality_numeric',
                        'time_pressure', 'trial_index']].isnull().sum()
    print(missing[missing > 0] if (missing > 0).any() else "  없음")

    print(f"[OK] 전처리 완료: {len(df_model):,} rows")

    return df_model


def mixed_effects_logistic_regression(df):
    """
    Mixed-Effects Logistic Regression (근사적)

    Note: statsmodels의 MixedLM은 선형 모델이므로,
    로지스틱 회귀의 경우 GEE를 사용하는 것이 더 적합합니다.
    여기서는 선형 확률 모델로 근사합니다.

    Model: choice ~ group + time_pressure + personality + trial + (1 | user_id)

    Args:
        df: 전처리된 데이터

    Returns:
        MixedLMResults: 모델 결과
    """
    print("\n=== Mixed-Effects Linear Model ===")
    print("(선형 확률 모델로 근사)")

    # 모델 공식
    formula = 'choice_binary ~ group_numeric + time_pressure + personality_numeric + trial_index + time_diff'

    try:
        # MixedLM 적합
        model = MixedLM.from_formula(
            formula,
            data=df,
            groups=df['user_id'],
            re_formula='1'  # Random intercept
        )

        result = model.fit(method='lbfgs', maxiter=100)

        print(result.summary())

        return result

    except Exception as e:
        print(f"[WARNING] Mixed-Effects 모델 적합 실패: {e}")
        print("  데이터 크기가 너무 크거나 수렴하지 않을 수 있습니다.")
        return None


def gee_analysis(df, cov_struct='ar1'):
    """
    GEE (Generalized Estimating Equations) 분석

    Model: choice ~ group + time_pressure + personality + trial + congestion_diff

    Args:
        df: 전처리된 데이터
        cov_struct: 'ar1' (AR(1)) 또는 'exchangeable'

    Returns:
        GEEResults: 모델 결과
    """
    print(f"\n=== GEE Analysis ({cov_struct.upper()}) ===")

    # 정렬 (GEE는 시간 순서가 중요)
    df_sorted = df.sort_values(['user_id', 'trial_number']).reset_index(drop=True)

    # Covariance structure 선택
    if cov_struct == 'ar1':
        cov_structure = Autoregressive()
    elif cov_struct == 'exchangeable':
        cov_structure = Exchangeable()
    else:
        raise ValueError(f"Unknown covariance structure: {cov_struct}")

    # 독립변수 선택
    exog_vars = ['group_numeric', 'time_pressure', 'personality_numeric',
                 'trial_index', 'time_diff', 'congestion_diff']

    # 결측값 제거
    df_clean = df_sorted[['choice_binary', 'user_id'] + exog_vars].dropna()

    print(f"분석 데이터: {len(df_clean):,} rows, {df_clean['user_id'].nunique():,} users")

    try:
        # GEE 모델 적합
        model = GEE(
            endog=df_clean['choice_binary'],
            exog=df_clean[exog_vars],
            groups=df_clean['user_id'],
            family=Binomial(),
            cov_struct=cov_structure
        )

        result = model.fit(maxiter=100)

        print(result.summary())

        # 계수 해석 (오즈비 추가)
        print("\n계수 해석 (로지스틱 회귀):")
        params = result.params
        pvalues = result.pvalues

        for var, coef, pval in zip(exog_vars, params, pvalues):
            odds_ratio = np.exp(coef)
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            print(f"  {var:25s}: β={coef:7.4f}, OR={odds_ratio:6.3f}  (p={pval:.6f}) {sig}")

        return result

    except Exception as e:
        print(f"[WARNING] GEE 모델 적합 실패: {e}")
        return None


def gee_with_interactions(df):
    """
    GEE with Interaction Terms

    Model: choice ~ group * trial + group * personality + ...

    Args:
        df: 전처리된 데이터

    Returns:
        GEEResults: 모델 결과
    """
    print("\n=== GEE with Interactions ===")

    df_sorted = df.sort_values(['user_id', 'trial_number']).reset_index(drop=True)

    # 교호작용 항 생성
    df_sorted['group_x_trial'] = df_sorted['group_numeric'] * df_sorted['trial_index']
    df_sorted['group_x_personality'] = df_sorted['group_numeric'] * df_sorted['personality_numeric']
    df_sorted['trial_x_congestion'] = df_sorted['trial_index'] * df_sorted['congestion_diff']

    exog_vars = [
        'group_numeric', 'time_pressure', 'personality_numeric', 'trial_index',
        'time_diff', 'congestion_diff',
        'group_x_trial', 'group_x_personality', 'trial_x_congestion'
    ]

    df_clean = df_sorted[['choice_binary', 'user_id'] + exog_vars].dropna()

    print(f"분석 데이터: {len(df_clean):,} rows")

    try:
        model = GEE(
            endog=df_clean['choice_binary'],
            exog=df_clean[exog_vars],
            groups=df_clean['user_id'],
            family=Binomial(),
            cov_struct=Autoregressive()
        )

        result = model.fit(maxiter=100)

        print(result.summary())

        # 유의미한 교호작용 확인
        print("\n교호작용 항:")
        interaction_vars = ['group_x_trial', 'group_x_personality', 'trial_x_congestion']
        for var in interaction_vars:
            idx = exog_vars.index(var)
            coef = result.params.iloc[idx]  # .iloc[] 사용 (FutureWarning 수정)
            pval = result.pvalues.iloc[idx]
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            print(f"  {var:25s}: {coef:8.4f}  (p={pval:.6f}) {sig}")

        return result

    except Exception as e:
        print(f"[WARNING] Interaction GEE 모델 적합 실패: {e}")
        return None


def fdr_correction(results_list):
    """
    FDR (False Discovery Rate) 보정

    Benjamini-Hochberg 방법 적용

    Args:
        results_list: [(variable_name, p_value), ...] 리스트

    Returns:
        pd.DataFrame: 보정 결과
    """
    print("\n=== FDR Correction (Benjamini-Hochberg) ===")

    if not results_list:
        print("[WARNING] 보정할 결과가 없습니다.")
        return None

    variables, pvalues = zip(*results_list)

    # FDR 보정
    reject, pvals_corrected, _, _ = multipletests(
        pvalues,
        alpha=config.ALPHA,
        method=config.FDR_METHOD
    )

    # 결과 DataFrame
    df_fdr = pd.DataFrame({
        'variable': variables,
        'p_value': pvalues,
        'p_value_fdr': pvals_corrected,
        'reject_null': reject
    })

    df_fdr = df_fdr.sort_values('p_value')

    print(df_fdr.to_string(index=False, float_format='%.6f'))

    # 유의미한 변수
    significant = df_fdr[df_fdr['reject_null'] == True]
    print(f"\n유의미한 변수 (FDR < {config.ALPHA}): {len(significant)}/{len(df_fdr)}")

    return df_fdr


def compare_models(results_dict):
    """
    여러 모델 결과 비교

    Args:
        results_dict: {model_name: result, ...}

    Returns:
        pd.DataFrame: 비교 테이블
    """
    print("\n=== Model Comparison ===")

    comparison_data = []

    for model_name, result in results_dict.items():
        if result is None:
            continue

        # AIC, BIC (가능한 경우)
        aic = getattr(result, 'aic', np.nan)
        bic = getattr(result, 'bic', np.nan)

        # Log-likelihood
        llf = getattr(result, 'llf', np.nan)

        # 파라미터 수
        nparams = len(result.params)

        comparison_data.append({
            'model': model_name,
            'log_likelihood': llf,
            'aic': aic,
            'bic': bic,
            'n_params': nparams
        })

    df_comparison = pd.DataFrame(comparison_data)

    print(df_comparison.to_string(index=False, float_format='%.2f'))

    return df_comparison


def save_results(results_dict, output_dir='analysis'):
    """
    모델 결과 저장

    Args:
        results_dict: {model_name: result, ...}
        output_dir: 출력 디렉토리
    """
    if not os.path.isabs(output_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, output_dir)

    for model_name, result in results_dict.items():
        if result is None:
            continue

        # 계수 테이블 저장
        try:
            if hasattr(result, 'params'):
                df_params = pd.DataFrame({
                    'coefficient': result.params,
                    'odds_ratio': np.exp(result.params),  # 오즈비 추가
                    'std_err': result.bse if hasattr(result, 'bse') else np.nan,
                    'z_value': result.tvalues if hasattr(result, 'tvalues') else np.nan,
                    'p_value': result.pvalues if hasattr(result, 'pvalues') else np.nan
                })

                output_path = os.path.join(output_dir, f'{model_name}_results.csv')
                df_params.to_csv(output_path, encoding='utf-8-sig')
                print(f"[OK] {model_name} 결과 저장: {output_path}")

        except Exception as e:
            print(f"[WARNING] {model_name} 저장 실패: {e}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("DAY 4: Mixed-Effects Models & GEE")
    print("=" * 60)

    # 1. 데이터 로드 및 전처리
    df = load_data()
    df_model = prepare_data_for_modeling(df)

    results_dict = {}

    # 2. GEE with AR(1)
    print("\n" + "=" * 60)
    result_gee_ar1 = gee_analysis(df_model, cov_struct='ar1')
    if result_gee_ar1:
        results_dict['gee_ar1'] = result_gee_ar1

    # 3. GEE with Exchangeable
    print("\n" + "=" * 60)
    result_gee_exch = gee_analysis(df_model, cov_struct='exchangeable')
    if result_gee_exch:
        results_dict['gee_exchangeable'] = result_gee_exch

    # 4. GEE with Interactions
    print("\n" + "=" * 60)
    result_gee_interact = gee_with_interactions(df_model)
    if result_gee_interact:
        results_dict['gee_interactions'] = result_gee_interact

    # 5. FDR Correction (주요 모델 기준)
    if result_gee_ar1:
        exog_vars = ['group_numeric', 'time_pressure', 'personality_numeric',
                     'trial_index', 'time_diff', 'congestion_diff']
        pvalue_list = [(var, pval) for var, pval in zip(exog_vars, result_gee_ar1.pvalues)]
        df_fdr = fdr_correction(pvalue_list)

        if df_fdr is not None:
            fdr_output = os.path.join('analysis', 'fdr_correction.csv')
            if not os.path.isabs(fdr_output):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                fdr_output = os.path.join(base_dir, fdr_output)
            df_fdr.to_csv(fdr_output, index=False, encoding='utf-8-sig')
            print(f"[OK] FDR 보정 결과 저장: {fdr_output}")

    # 6. Model Comparison
    if results_dict:
        df_comparison = compare_models(results_dict)

        comp_output = os.path.join('analysis', 'model_comparison.csv')
        if not os.path.isabs(comp_output):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            comp_output = os.path.join(base_dir, comp_output)
        df_comparison.to_csv(comp_output, index=False, encoding='utf-8-sig')
        print(f"[OK] 모델 비교 저장: {comp_output}")

    # 7. 결과 저장
    save_results(results_dict)

    print("\n" + "=" * 60)
    print("Mixed-Effects Models & GEE 분석 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
