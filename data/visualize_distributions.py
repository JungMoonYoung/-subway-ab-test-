"""
DAY 2 데이터 분포 시각화
time_pressure, 경로 시간, 혼잡도 분포를 히스토그램으로 저장
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# config.py 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 스타일 설정
sns.set_style("whitegrid")


def load_data(file_path='data/trials_data_partial.csv'):
    """데이터 로드"""
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, file_path)

    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"[OK] 데이터 로드: {len(df):,} rows")
    return df


def plot_distributions(df):
    """
    모든 분포 시각화 및 저장
    """
    # Figure 생성 (3x2 그리드)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('DAY 2 데이터 분포 분석', fontsize=16, y=0.995)

    # 1. time_pressure 분포 (막대 그래프)
    ax1 = axes[0, 0]
    tp_counts = df['time_pressure'].value_counts(normalize=True).sort_index()
    ax1.bar(tp_counts.index, tp_counts.values, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Time Pressure (0=급함, 1=보통, 2=여유)')
    ax1.set_ylabel('비율')
    ax1.set_title('Time Pressure 분포')
    ax1.set_xticks([0, 1, 2])
    for i, v in enumerate(tp_counts.values):
        ax1.text(i, v + 0.01, f'{v:.1%}', ha='center')

    # 2. Fast Route 시간 분포
    ax2 = axes[0, 1]
    ax2.hist(df['route_time_fast'], bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(df['route_time_fast'].mean(), color='red', linestyle='--',
                label=f'평균: {df["route_time_fast"].mean():.1f}분')
    ax2.set_xlabel('시간 (분)')
    ax2.set_ylabel('빈도')
    ax2.set_title('Fast Route 시간 분포')
    ax2.legend()

    # 3. Relax Route 시간 분포
    ax3 = axes[1, 0]
    ax3.hist(df['route_time_relax'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.axvline(df['route_time_relax'].mean(), color='darkgreen', linestyle='--',
                label=f'평균: {df["route_time_relax"].mean():.1f}분')
    ax3.set_xlabel('시간 (분)')
    ax3.set_ylabel('빈도')
    ax3.set_title('Relax Route 시간 분포')
    ax3.legend()

    # 4. Fast Route 혼잡도 분포
    ax4 = axes[1, 1]
    ax4.hist(df['congestion_fast'], bins=50, color='orangered', alpha=0.7, edgecolor='black')
    ax4.axvline(df['congestion_fast'].mean(), color='darkred', linestyle='--',
                label=f'평균: {df["congestion_fast"].mean():.1f}%')
    ax4.set_xlabel('혼잡도 (%)')
    ax4.set_ylabel('빈도')
    ax4.set_title('Fast Route 혼잡도 분포')
    ax4.legend()

    # 5. Relax Route 혼잡도 분포
    ax5 = axes[2, 0]
    ax5.hist(df['congestion_relax'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax5.axvline(df['congestion_relax'].mean(), color='darkblue', linestyle='--',
                label=f'평균: {df["congestion_relax"].mean():.1f}%')
    ax5.set_xlabel('혼잡도 (%)')
    ax5.set_ylabel('빈도')
    ax5.set_title('Relax Route 혼잡도 분포')
    ax5.legend()

    # 6. 시간-혼잡도 관계 (산점도 샘플링)
    ax6 = axes[2, 1]
    # 전체 데이터는 너무 많으므로 1%만 샘플링
    sample_df = df.sample(n=5000, random_state=42)
    ax6.scatter(sample_df['congestion_fast'], sample_df['actual_time_fast'],
                alpha=0.3, s=10, label='Fast Route', color='coral')
    ax6.scatter(sample_df['congestion_relax'], sample_df['actual_time_relax'],
                alpha=0.3, s=10, label='Relax Route', color='skyblue')
    ax6.set_xlabel('혼잡도 (%)')
    ax6.set_ylabel('실제 소요시간 (분)')
    ax6.set_title('혼잡도-시간 관계 (샘플 5,000개)')
    ax6.legend()

    plt.tight_layout()

    # 저장
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'day2_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 시각화 저장: {output_path}")

    # 화면 출력하지 않음 (서버 환경 고려)
    plt.close()


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("DAY 2 데이터 분포 시각화")
    print("=" * 50)

    # 1. 데이터 로드
    df = load_data()

    # 2. 시각화
    plot_distributions(df)

    print("\n[OK] 시각화 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
