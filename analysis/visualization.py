"""
DAY 5: 시각화 구현

A/B Test 결과 시각화 및 차트 생성
"""

# 한글 폰트 glyph 경고 억제 (맨 먼저 설정)
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows)
# 사용 가능한 한글 폰트 찾기
available_fonts = [f.name for f in fm.fontManager.ttflist]
korean_fonts = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Noto Sans CJK KR']

font_name = None
for font in korean_fonts:
    if font in available_fonts:
        font_name = font
        break

if font_name:
    plt.rcParams['font.family'] = font_name
    print(f"[OK] 한글 폰트 설정: {font_name}")
else:
    print(f"[WARNING] 한글 폰트 없음, 기본 폰트 사용")

plt.rcParams['axes.unicode_minus'] = False

# config.py 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Random Seed 설정
np.random.seed(config.RANDOM_SEED)
print(f"[SEED] Random seed 설정: {config.RANDOM_SEED}")

# 스타일 설정
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)


def load_data(file_path='data/synthetic_data_dynamic.parquet'):
    """데이터 로드"""
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, file_path)

    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, encoding='utf-8-sig')

    print(f"[OK] 데이터 로드: {len(df):,} rows")
    return df


def ensure_output_dir(output_dir='figures'):
    """출력 디렉토리 생성"""
    if not os.path.isabs(output_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_ab_comparison(df, output_dir):
    """
    A vs B 그룹 Fast Route 선택 비율 비교 (Bar Plot with 95% CI)
    """
    print("\n[1/8] A vs B 비교 차트 생성 중...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # 그룹별 선택 비율 계산
    group_stats = []
    for group in ['A', 'B']:
        group_data = df[df['assigned_group'] == group]
        n = len(group_data)
        fast_count = (group_data['selected_route'] == 'Fast').sum()
        fast_rate = fast_count / n

        # 95% CI (Wilson score interval)
        from statsmodels.stats.proportion import proportion_confint
        ci_low, ci_high = proportion_confint(fast_count, n, alpha=0.05, method='wilson')

        group_stats.append({
            'group': group,
            'fast_rate': fast_rate,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'error': fast_rate - ci_low
        })

    df_stats = pd.DataFrame(group_stats)

    # Bar plot
    colors = ['#2E86AB', '#A23B72']
    bars = ax.bar(df_stats['group'], df_stats['fast_rate'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Error bars
    ax.errorbar(df_stats['group'], df_stats['fast_rate'],
                yerr=[df_stats['error'].values,
                      (df_stats['ci_high'] - df_stats['fast_rate']).values],
                fmt='none', color='black', capsize=10, capthick=2, linewidth=2)

    # 값 표시
    for i, (idx, row) in enumerate(df_stats.iterrows()):
        ax.text(i, row['fast_rate'] + 0.05, f"{row['fast_rate']*100:.2f}%",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_xlabel('그룹', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fast Route 선택 비율', fontsize=14, fontweight='bold')
    ax.set_title('A/B 그룹별 Fast Route 선택 비율 비교',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    plt.tight_layout()
    output_path = os.path.join(output_dir, '01_ab_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 저장: {output_path}")


def plot_personality_breakdown(df, output_dir):
    """
    Personality type별 선택 비율 (Facet Plot)
    """
    print("\n[2/8] Personality 분석 차트 생성 중...")

    # 데이터 준비
    personality_stats = []
    for personality in ['efficiency-oriented', 'comfort-oriented', 'neutral']:
        for group in ['A', 'B']:
            subset = df[(df['personality_type'] == personality) &
                        (df['assigned_group'] == group)]
            if len(subset) > 0:
                fast_rate = (subset['selected_route'] == 'Fast').mean()
                personality_stats.append({
                    'personality': personality,
                    'group': group,
                    'fast_rate': fast_rate
                })

    df_pers = pd.DataFrame(personality_stats)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    personalities = ['efficiency-oriented', 'comfort-oriented', 'neutral']
    titles = ['효율 지향', '편안함 지향', '중립']
    colors = {'A': '#2E86AB', 'B': '#A23B72'}

    for i, (personality, title) in enumerate(zip(personalities, titles)):
        ax = axes[i]
        subset = df_pers[df_pers['personality'] == personality]

        x_pos = np.arange(len(subset))
        bars = ax.bar(x_pos, subset['fast_rate'],
                      color=[colors[g] for g in subset['group']],
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        # 값 표시
        for j, (idx, row) in enumerate(subset.iterrows()):
            ax.text(j, row['fast_rate'] + 0.03, f"{row['fast_rate']*100:.1f}%",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(subset['group'], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

        if i == 0:
            ax.set_ylabel('Fast Route 선택 비율', fontsize=13, fontweight='bold')

    fig.suptitle('Personality Type별 Fast Route 선택 비율',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, '02_personality_breakdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 저장: {output_path}")


def plot_trial_trends(df, output_dir):
    """
    Trial별 선택 비율 변화 (시계열)
    """
    print("\n[3/8] Trial별 추이 차트 생성 중...")

    # Trial별 통계
    trial_stats = []
    for trial in range(1, config.NUM_TRIALS + 1):
        for group in ['A', 'B']:
            subset = df[(df['trial_number'] == trial) &
                        (df['assigned_group'] == group)]
            fast_rate = (subset['selected_route'] == 'Fast').mean()
            trial_stats.append({
                'trial': trial,
                'group': group,
                'fast_rate': fast_rate
            })

    df_trial = pd.DataFrame(trial_stats)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for group, color in [('A', '#2E86AB'), ('B', '#A23B72')]:
        subset = df_trial[df_trial['group'] == group]
        ax.plot(subset['trial'], subset['fast_rate'],
                'o-', color=color, linewidth=3, markersize=10,
                label=f'그룹 {group}', alpha=0.8)

    ax.set_xlabel('Trial 번호', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fast Route 선택 비율', fontsize=14, fontweight='bold')
    ax.set_title('Trial별 Fast Route 선택 비율 변화 (학습 효과)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(1, config.NUM_TRIALS + 1))
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = os.path.join(output_dir, '03_trial_trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 저장: {output_path}")


def plot_congestion_heatmap(df, output_dir):
    """
    Time Pressure × Personality 조합별 선택 비율 (Heatmap)
    """
    print("\n[4/8] Heatmap 차트 생성 중...")

    # 데이터 준비
    heatmap_data = df.groupby(['time_pressure', 'personality_type'])['selected_route'].apply(
        lambda x: (x == 'Fast').mean()
    ).reset_index()

    # Pivot
    heatmap_pivot = heatmap_data.pivot(
        index='personality_type',
        columns='time_pressure',
        values='selected_route'
    )

    # 순서 조정
    personality_order = ['efficiency-oriented', 'neutral', 'comfort-oriented']
    heatmap_pivot = heatmap_pivot.reindex(personality_order)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(heatmap_pivot, annot=True, fmt='.2%', cmap='RdYlGn',
                cbar_kws={'label': 'Fast Route 선택 비율'},
                linewidths=1, linecolor='white',
                vmin=0, vmax=1, ax=ax)

    ax.set_xlabel('Time Pressure', fontsize=13, fontweight='bold')
    # X축 레이블 명시화
    ax.set_xticklabels(['급함(0)', '보통(1)', '여유(2)'])
    ax.set_ylabel('Personality Type', fontsize=13, fontweight='bold')
    ax.set_title('Time Pressure × Personality별 Fast Route 선택 비율',
                 fontsize=15, fontweight='bold', pad=20)

    # Y축 레이블 변경
    ax.set_yticklabels(['효율 지향', '중립', '편안함 지향'], rotation=0)

    plt.tight_layout()
    output_path = os.path.join(output_dir, '04_pressure_personality_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 저장: {output_path}")


def plot_gee_coefficients(output_dir):
    """
    GEE 모델 계수 플롯 (95% CI 포함)
    """
    print("\n[5/8] GEE 계수 차트 생성 중...")

    # GEE 결과 로드
    gee_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'analysis', 'gee_ar1_results.csv')

    if not os.path.exists(gee_path):
        print(f"  [WARNING] GEE 결과 파일 없음: {gee_path}")
        return

    df_gee = pd.read_csv(gee_path, index_col=0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    variables = df_gee.index.tolist()
    coefs = df_gee['coefficient'].values
    ci_low = df_gee['coefficient'] - 1.96 * df_gee['std_err']
    ci_high = df_gee['coefficient'] + 1.96 * df_gee['std_err']

    colors = ['red' if c < 0 else 'blue' for c in coefs]

    y_pos = np.arange(len(variables))
    ax.barh(y_pos, coefs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Error bars
    ax.errorbar(coefs, y_pos, xerr=[coefs - ci_low, ci_high - coefs],
                fmt='none', color='black', capsize=5, capthick=1.5, linewidth=1.5)

    # 0 참조선
    ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables, fontsize=11)
    ax.set_xlabel('계수 (β)', fontsize=13, fontweight='bold')
    ax.set_title('GEE 모델 계수 및 95% 신뢰구간', fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = os.path.join(output_dir, '05_gee_coefficients.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 저장: {output_path}")


def plot_satisfaction_distribution(df, output_dir):
    """
    만족도 분포 (그룹별 비교)
    """
    print("\n[6/8] 만족도 분포 차트 생성 중...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    for group, color in [('A', '#2E86AB'), ('B', '#A23B72')]:
        subset = df[df['assigned_group'] == group]
        axes[0].hist(subset['satisfaction_score'], bins=30, alpha=0.6,
                     label=f'그룹 {group}', color=color, edgecolor='black')

    axes[0].set_xlabel('만족도 점수', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('빈도', fontsize=13, fontweight='bold')
    axes[0].set_title('만족도 점수 분포', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3, linestyle='--')

    # Boxplot
    data_to_plot = [df[df['assigned_group'] == 'A']['satisfaction_score'],
                    df[df['assigned_group'] == 'B']['satisfaction_score']]
    bp = axes[1].boxplot(data_to_plot, tick_labels=['그룹 A', '그룹 B'],
                         patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], ['#2E86AB', '#A23B72']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_ylabel('만족도 점수', fontsize=13, fontweight='bold')
    axes[1].set_title('만족도 점수 분포 (Boxplot)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = os.path.join(output_dir, '06_satisfaction_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 저장: {output_path}")


def plot_congestion_dynamics(df, output_dir):
    """
    Trial별 혼잡도 변화 (동적 피드백 시각화)
    """
    print("\n[7/8] 혼잡도 동적 변화 차트 생성 중...")

    # Trial별 평균 혼잡도
    trial_congestion = df.groupby('trial_number').agg({
        'congestion_fast': 'mean',
        'congestion_relax': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(trial_congestion['trial_number'], trial_congestion['congestion_fast'],
            'o-', color='#E63946', linewidth=3, markersize=10,
            label='Fast Route 혼잡도', alpha=0.8)
    ax.plot(trial_congestion['trial_number'], trial_congestion['congestion_relax'],
            's-', color='#06A77D', linewidth=3, markersize=10,
            label='Relax Route 혼잡도', alpha=0.8)

    ax.set_xlabel('Trial 번호', fontsize=14, fontweight='bold')
    ax.set_ylabel('평균 혼잡도 (%)', fontsize=14, fontweight='bold')
    ax.set_title('Trial별 경로 혼잡도 변화 (동적 피드백)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(1, config.NUM_TRIALS + 1))
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = os.path.join(output_dir, '07_congestion_dynamics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 저장: {output_path}")


def plot_learning_effect(df, output_dir):
    """
    학습 효과 시각화 (이전 선택에 따른 다음 선택)
    """
    print("\n[8/8] 학습 효과 차트 생성 중...")

    # Trial 2+ 데이터
    df_learning = df[df['trial_number'] > 1].copy()

    # 이전 선택별 다음 Fast 선택률
    learning_stats = df_learning.groupby('previous_choice')['selected_route'].apply(
        lambda x: (x == 'Fast').mean()
    ).reset_index()
    learning_stats.columns = ['previous_choice', 'next_fast_rate']

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Fast': '#E63946', 'Relax': '#06A77D'}
    bars = ax.bar(range(len(learning_stats)), learning_stats['next_fast_rate'],
                  color=[colors[choice] for choice in learning_stats['previous_choice']],
                  alpha=0.8, edgecolor='black', linewidth=2)

    # 값 표시
    for i, row in learning_stats.iterrows():
        ax.text(i, row['next_fast_rate'] + 0.03, f"{row['next_fast_rate']*100:.2f}%",
                ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_xticks(range(len(learning_stats)))
    ax.set_xticklabels([f'이전: {choice}' for choice in learning_stats['previous_choice']],
                       fontsize=12)
    ax.set_ylabel('다음 Trial Fast 선택 비율', fontsize=13, fontweight='bold')
    ax.set_title('학습 효과: 이전 선택에 따른 다음 선택 변화',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 차이 표시
    if len(learning_stats) == 2:
        diff = learning_stats.iloc[1]['next_fast_rate'] - learning_stats.iloc[0]['next_fast_rate']
        ax.text(0.5, 0.95, f'차이: {diff*100:.2f}%p',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    output_path = os.path.join(output_dir, '08_learning_effect.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 저장: {output_path}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("DAY 5: 시각화 생성")
    print("=" * 60)

    # 1. 데이터 로드
    df = load_data()

    # 2. 출력 디렉토리 생성
    output_dir = ensure_output_dir()
    print(f"\n출력 디렉토리: {output_dir}")

    # 3. 모든 차트 생성
    plot_ab_comparison(df, output_dir)
    plot_personality_breakdown(df, output_dir)
    plot_trial_trends(df, output_dir)
    plot_congestion_heatmap(df, output_dir)
    plot_gee_coefficients(output_dir)
    plot_satisfaction_distribution(df, output_dir)
    plot_congestion_dynamics(df, output_dir)
    plot_learning_effect(df, output_dir)

    print("\n" + "=" * 60)
    print("모든 시각화 생성 완료!")
    print(f"총 8개 차트 저장: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
