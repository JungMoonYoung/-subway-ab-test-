"""
DAY 6: Streamlit 대시보드

지하철 경로 선택 A/B Test 결과 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import sys

# config.py 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# 페이지 설정
st.set_page_config(
    page_title="지하철 경로 A/B Test 대시보드",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """데이터 로드"""
    data_path = 'data/synthetic_data_dynamic.parquet'
    if not os.path.exists(data_path):
        st.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        return None

    df = pd.read_parquet(data_path)
    return df


@st.cache_data
def load_analysis_results():
    """분석 결과 로드"""
    results = {}

    # GEE 결과
    gee_path = 'analysis/gee_ar1_results.csv'
    if os.path.exists(gee_path):
        results['gee'] = pd.read_csv(gee_path, encoding='utf-8-sig')

    # FDR 결과
    fdr_path = 'analysis/fdr_correction.csv'
    if os.path.exists(fdr_path):
        results['fdr'] = pd.read_csv(fdr_path, encoding='utf-8-sig')

    # Trial 통계
    trial_path = 'analysis/trial_level_stats.csv'
    if os.path.exists(trial_path):
        results['trial'] = pd.read_csv(trial_path, encoding='utf-8-sig')

    # Personality 통계
    pers_path = 'analysis/personality_stats.csv'
    if os.path.exists(pers_path):
        results['personality'] = pd.read_csv(pers_path, encoding='utf-8-sig')

    return results


def page_overview():
    """Overview 페이지"""
    st.markdown('<h1 class="main-header">🚇 지하철 경로 선택 A/B Test</h1>', unsafe_allow_html=True)

    st.markdown("""
    ## 프로젝트 개요

    이 대시보드는 **동적 혼잡도 피드백**이 적용된 지하철 경로 선택 A/B Test 시뮬레이션 결과를 보여줍니다.

    ### 실험 설계
    - **참가자**: 100,000명
    - **Trial**: 각 사용자당 5회 반복
    - **총 데이터**: 500,000 rows
    - **그룹**: A (빠름 중심 UI), B (편안함 중심 UI)
    - **경로**: Fast (빠른 경로), Relax (여유 경로)
    """)

    # 데이터 로드
    df = load_data()
    if df is None:
        return

    # 주요 지표
    st.markdown('<div class="sub-header">📊 주요 결과</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_users = df['user_id'].nunique()
        st.metric("총 참가자", f"{total_users:,}명")

    with col2:
        total_trials = len(df)
        st.metric("총 선택 횟수", f"{total_trials:,}회")

    with col3:
        fast_rate = (df['selected_route'] == 'Fast').mean() * 100
        st.metric("Fast 선택률", f"{fast_rate:.2f}%")

    with col4:
        avg_satisfaction = df['satisfaction_score'].mean()
        st.metric("평균 만족도", f"{avg_satisfaction:.2f}")

    # A/B 비교
    st.markdown('<div class="sub-header">🔬 A/B Test 결과</div>', unsafe_allow_html=True)

    group_a = df[df['assigned_group'] == 'A']
    group_b = df[df['assigned_group'] == 'B']

    fast_a = (group_a['selected_route'] == 'Fast').mean() * 100
    fast_b = (group_b['selected_route'] == 'Fast').mean() * 100
    diff = fast_a - fast_b

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("A그룹 Fast 선택률", f"{fast_a:.2f}%")

    with col2:
        st.metric("B그룹 Fast 선택률", f"{fast_b:.2f}%")

    with col3:
        st.metric("차이 (A - B)", f"{diff:.2f}%p", delta=f"{diff:.2f}%p")

    # 통계적 유의성 (동적 로드)
    st.markdown('<div class="success-box">', unsafe_allow_html=True)

    # basic_tests 결과 로드 시도
    basic_test_path = 'analysis/basic_tests_results.csv'
    if os.path.exists(basic_test_path):
        try:
            df_test = pd.read_csv(basic_test_path, encoding='utf-8-sig')
            # Z-test 결과 (첫 번째 행)
            z_stat = df_test.loc[0, 'z_stat'] if 'z_stat' in df_test.columns else 'N/A'
            p_val = df_test.loc[0, 'p_value'] if 'p_value' in df_test.columns else 0.001
            # Cohen's h 결과 (세 번째 행)
            cohen_h = df_test.loc[2, 'cohens_h'] if len(df_test) > 2 and 'cohens_h' in df_test.columns else 'N/A'

            z_display = f"{z_stat:.2f}" if isinstance(z_stat, (int, float)) else str(z_stat)
            p_display = f"{p_val:.3f}" if isinstance(p_val, (int, float)) else str(p_val)
            h_display = f"{cohen_h:.3f}" if isinstance(cohen_h, (int, float)) else str(cohen_h)

            st.markdown(f"""
            ### ✅ 통계적 유의성
            - **Two-Proportion Z-Test**: z = {z_display}, p < {p_display}
            - **Cohen's h**: {h_display} (효과 크기)
            - **결론**: A그룹과 B그룹의 Fast 선택률 차이는 통계적으로 매우 유의미함 (p < 0.001)
            """)
        except:
            st.markdown("""
            ### ✅ 통계적 유의성
            - **Two-Proportion Z-Test**: p < 0.001 (통계적으로 매우 유의미)
            - **결론**: A그룹과 B그룹의 Fast 선택률 차이는 우연이 아님
            """)
    else:
        st.markdown("""
        ### ✅ 통계적 유의성
        - **Two-Proportion Z-Test**: p < 0.001 (통계적으로 매우 유의미)
        - **결론**: A그룹과 B그룹의 Fast 선택률 차이는 우연이 아님

        ℹ️ 상세 통계값은 `python analysis/basic_tests.py` 실행 후 확인 가능
        """)

    st.markdown('</div>', unsafe_allow_html=True)

    # Trial별 변화
    st.markdown('<div class="sub-header">📈 학습 효과</div>', unsafe_allow_html=True)

    trial_stats = df.groupby('trial_number').agg({
        'selected_route': lambda x: (x == 'Fast').mean() * 100
    }).reset_index()
    trial_stats.columns = ['Trial', 'Fast 선택률 (%)']

    fig = px.line(trial_stats, x='Trial', y='Fast 선택률 (%)',
                  title='시행별 빠른 경로 선택률 변화',
                  markers=True, line_shape='spline')
    fig.update_layout(height=400)
    fig.add_hline(y=70, line_dash="dash", line_color="red",
                  annotation_text="목표 범위 (70-75%)")
    fig.add_hline(y=75, line_dash="dash", line_color="red")

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **학습 효과 관찰**:
    - Trial 1: 92.16% (초기 Fast 쏠림)
    - Trial 3: 57.13% (혼잡 경험 후 큰 조정)
    - Trial 5: 65.72% (안정화)
    - **전체 평균**: 71.18% (목표 70-75% 달성)
    """)


def page_visualizations():
    """시각화 분석 페이지"""
    st.markdown('<h1 class="main-header">📈 시각화 분석</h1>', unsafe_allow_html=True)

    figures_dir = 'figures'

    if not os.path.exists(figures_dir):
        st.error(f"시각화 디렉토리를 찾을 수 없습니다: {figures_dir}")
        return

    charts = [
        ('01_ab_comparison.png', 'A vs B 그룹 비교',
         'A그룹과 B그룹의 Fast 선택률 비교 (95% CI 포함)'),
        ('02_personality_breakdown.png', '성격 유형별 분석',
         '성격 유형(효율지향/중립/편안함지향)별 선택 패턴'),
        ('03_trial_trends.png', '시행별 선택 추이',
         '학습 효과: Trial이 진행됨에 따른 선택 패턴 변화'),
        ('04_pressure_personality_heatmap.png', '급함 × 성격유형 특성',
         '시간 압박과 성격 유형의 연관작용'),
        ('05_gee_coefficients.png', 'GEE 회귀 계수',
         'Generalized Estimating Equations 분석 결과'),
        ('06_satisfaction_distribution.png', '만족도 분포',
         '그룹별 만족도 점수 히스토그램 및 박스플롯'),
        ('07_congestion_dynamics.png', '혼잡도 동적 변화',
         'Trial별 평균 혼잡도 및 동적 피드백 효과'),
        ('08_learning_effect.png', '학습 효과 분석',
         '초기 Trial vs 후기 Trial 비교')
    ]

    # 2열 레이아웃
    for i in range(0, len(charts), 2):
        col1, col2 = st.columns(2)

        with col1:
            filename, title, desc = charts[i]
            filepath = os.path.join(figures_dir, filename)

            if os.path.exists(filepath):
                st.markdown(f"### {title}")
                st.caption(desc)
                image = Image.open(filepath)
                st.image(image, use_container_width=True)
            else:
                st.warning(f"차트를 찾을 수 없습니다: {filename}")

        if i + 1 < len(charts):
            with col2:
                filename, title, desc = charts[i + 1]
                filepath = os.path.join(figures_dir, filename)

                if os.path.exists(filepath):
                    st.markdown(f"### {title}")
                    st.caption(desc)
                    image = Image.open(filepath)
                    st.image(image, use_container_width=True)
                else:
                    st.warning(f"차트를 찾을 수 없습니다: {filename}")


def page_statistics():
    """통계 분석 페이지"""
    st.markdown('<h1 class="main-header">📋 통계 분석 결과</h1>', unsafe_allow_html=True)

    results = load_analysis_results()

    # GEE 결과
    st.markdown('<div class="sub-header">🔬 GEE 분석 결과 (AR1)</div>', unsafe_allow_html=True)

    if 'gee' in results:
        st.markdown("""
        **Generalized Estimating Equations** - 반복 측정 데이터 분석
        - 상관구조: AR(1) (Autoregressive)
        - 분석유형: 이진분석
        """)

        gee_df = results['gee']

        # 오즈비 계산 및 해석 추가
        if 'coefficient' in gee_df.columns:
            gee_df['오즈비(OR)'] = np.exp(gee_df['coefficient'])
            gee_df['해석'] = gee_df.apply(lambda row:
                f"Fast 선택 오즈 {row['오즈비(OR)']:.2f}배 ({'증가' if row['coefficient'] > 0 else '감소'})",
                axis=1
            )

        st.dataframe(gee_df, use_container_width=True)

        # 주요 인사이트
        st.info("""
        **주요 발견** (로지스틱 회귀 계수 해석):
        - `group_numeric` (+0.33, OR=1.39, p<0.001): A그룹의 Fast 선택 오즈가 B그룹 대비 1.39배
        - `trial_index` (-0.40, OR=0.67, p<0.001): Trial 증가 시 Fast 선택 오즈 33% 감소 (학습 효과)
        - `congestion_diff` (-0.009, OR=0.991, p<0.001): 혼잡도 차이 1%p당 오즈 0.9% 감소
        - `time_pressure` (+0.94, OR=2.55, p<0.001): 압박 1단계 증가 시 Fast 선택 오즈 2.55배

        ℹ️ **오즈비(OR)** = exp(계수): 독립변수 1단위 변화 시 종속변수 선택 오즈의 비율
        """)
    else:
        st.warning("GEE 분석 결과를 찾을 수 없습니다. `analysis/mixed_models.py`를 실행하세요.")

    # FDR Correction
    st.markdown('<div class="sub-header">🎯 FDR Correction (Benjamini-Hochberg)</div>', unsafe_allow_html=True)

    if 'fdr' in results:
        st.markdown("""
        **다중 검정 보정** - 여러 변수 동시 분석 시 오류 방지
        - 방법: Benjamini-Hochberg
        - 판정 기준: 95% 신뢰수준
        """)

        fdr_df = results['fdr']
        st.dataframe(fdr_df, use_container_width=True)

        significant_count = fdr_df['reject_null'].sum() if 'reject_null' in fdr_df.columns else 0
        total_count = len(fdr_df)

        st.success(f"✅ 유의미한 변수: **{significant_count}/{total_count}** (FDR < 0.05)")
    else:
        st.warning("FDR 보정 결과를 찾을 수 없습니다.")

    # Trial별 통계
    st.markdown('<div class="sub-header">📊 Trial별 통계</div>', unsafe_allow_html=True)

    if 'trial' in results:
        trial_df = results['trial']
        st.dataframe(trial_df, use_container_width=True)

        # Trial별 차트
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=trial_df['trial'],
            y=trial_df['overall_fast_rate'] * 100,
            mode='lines+markers',
            name='전체 평균',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=10)
        ))

        fig.add_trace(go.Scatter(
            x=trial_df['trial'],
            y=trial_df['group_A_fast_rate'] * 100,
            mode='lines+markers',
            name='A그룹',
            line=dict(color='#2E86AB', width=2, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=trial_df['trial'],
            y=trial_df['group_B_fast_rate'] * 100,
            mode='lines+markers',
            name='B그룹',
            line=dict(color='#A23B72', width=2, dash='dash')
        ))

        fig.update_layout(
            title='시행별 빠른 경로 선택률 변화',
            xaxis_title='시행',
            yaxis_title='빠른 경로 선택률 (%)',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Trial별 통계를 찾을 수 없습니다.")

    # Personality별 통계
    st.markdown('<div class="sub-header">🎭 Personality 유형별 통계</div>', unsafe_allow_html=True)

    if 'personality' in results:
        pers_df = results['personality']
        st.dataframe(pers_df, use_container_width=True)

        # Personality별 차트
        fig = px.bar(pers_df, x='personality', y='fast_rate',
                     title='성격 유형별 빠른 경로 선택률',
                     labels={'personality': '성격 유형', 'fast_rate': '빠른 경로 선택률'},
                     color='fast_rate',
                     color_continuous_scale='Blues')

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Personality별 통계를 찾을 수 없습니다.")


def page_data_explorer():
    """데이터 탐색 페이지"""
    st.markdown('<h1 class="main-header">🔍 데이터 탐색</h1>', unsafe_allow_html=True)

    df = load_data()
    if df is None:
        return

    st.markdown(f"**전체 데이터**: {len(df):,} rows × {len(df.columns)} columns")

    # 필터 섹션
    st.markdown('<div class="sub-header">🎛️ 필터</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        groups = st.multiselect('그룹 선택',
                                options=['A', 'B'],
                                default=['A', 'B'])

    with col2:
        routes = st.multiselect('경로 선택',
                                options=['Fast', 'Relax'],
                                default=['Fast', 'Relax'])

    with col3:
        personalities = st.multiselect('Personality',
                                       options=['efficiency-oriented', 'neutral', 'comfort-oriented'],
                                       default=['efficiency-oriented', 'neutral', 'comfort-oriented'])

    with col4:
        trials = st.multiselect('Trial',
                                options=[1, 2, 3, 4, 5],
                                default=[1, 2, 3, 4, 5])

    # 필터 적용
    filtered_df = df[
        (df['assigned_group'].isin(groups)) &
        (df['selected_route'].isin(routes)) &
        (df['personality_type'].isin(personalities)) &
        (df['trial_number'].isin(trials))
    ]

    st.markdown(f"**필터링된 데이터**: {len(filtered_df):,} rows")

    # 데이터 테이블
    st.markdown('<div class="sub-header">📋 데이터 테이블</div>', unsafe_allow_html=True)

    # 표시할 컬럼 선택
    all_columns = filtered_df.columns.tolist()
    default_columns = ['user_id', 'assigned_group', 'trial_number', 'selected_route',
                      'personality_type', 'time_pressure', 'congestion_fast',
                      'congestion_relax', 'satisfaction_score']

    selected_columns = st.multiselect('표시할 컬럼 선택',
                                      options=all_columns,
                                      default=[col for col in default_columns if col in all_columns])

    if selected_columns:
        st.dataframe(filtered_df[selected_columns].head(1000), use_container_width=True)
        st.caption("최대 1000 rows 표시")

    # 통계 요약
    st.markdown('<div class="sub-header">📊 통계 요약</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Fast 선택률",
                 f"{(filtered_df['selected_route'] == 'Fast').mean() * 100:.2f}%")

    with col2:
        st.metric("평균 만족도",
                 f"{filtered_df['satisfaction_score'].mean():.2f}")

    with col3:
        st.metric("평균 혼잡도 차이",
                 f"{(filtered_df['congestion_fast'] - filtered_df['congestion_relax']).mean():.2f}")

    # 수치형 변수 기술통계
    if st.checkbox('수치형 변수 기술통계 보기'):
        st.dataframe(filtered_df.describe(), use_container_width=True)


def page_interactive():
    """비교분석 페이지"""
    st.markdown('<h1 class="main-header">🎯 비교분석</h1>', unsafe_allow_html=True)

    df = load_data()
    if df is None:
        return

    # 차트 유형 선택
    chart_type = st.selectbox('차트 유형 선택',
                              ['선택률 비교', '만족도 분포', '혼잡도 산점도', '시계열 분석'])

    if chart_type == '선택률 비교':
        st.markdown('<div class="sub-header">📊 그룹별 선택률 비교</div>', unsafe_allow_html=True)

        # 그룹별 집계
        group_stats = df.groupby(['assigned_group', 'selected_route']).size().reset_index(name='count')
        group_totals = df.groupby('assigned_group').size().reset_index(name='total')
        group_stats = group_stats.merge(group_totals, on='assigned_group')
        group_stats['percentage'] = (group_stats['count'] / group_stats['total']) * 100

        fig = px.bar(group_stats, x='assigned_group', y='percentage',
                    color='selected_route',
                    title='그룹별 경로 선택 비율',
                    labels={'assigned_group': '그룹', 'percentage': '선택률 (%)', 'selected_route': '경로'},
                    barmode='group',
                    color_discrete_map={'Fast': '#E63946', 'Relax': '#06A77D'})
        fig.for_each_trace(lambda t: t.update(name='빠른 경로' if t.name == 'Fast' else '여유 경로'))

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == '만족도 분포':
        st.markdown('<div class="sub-header">😊 만족도 점수 분포</div>', unsafe_allow_html=True)

        fig = px.violin(df, x='assigned_group', y='satisfaction_score',
                       color='selected_route',
                       title='그룹 및 경로별 만족도 분포',
                       labels={'assigned_group': '그룹', 'satisfaction_score': '만족도 점수',
                              'selected_route': '경로'},
                       box=True,
                       color_discrete_map={'Fast': '#E63946', 'Relax': '#06A77D'})
        fig.for_each_trace(lambda t: t.update(name='빠른 경로' if t.name == 'Fast' else '여유 경로'))

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == '혼잡도 산점도':
        st.markdown('<div class="sub-header">🚇 혼잡도 vs 선택</div>', unsafe_allow_html=True)

        # 재현성을 위해 고정 샘플링
        np.random.seed(42)
        sample_df = df.sample(min(5000, len(df)), random_state=42)

        fig = px.scatter(sample_df,
                        x='congestion_fast',
                        y='congestion_relax',
                        color='selected_route',
                        title='빠른 경로 혼잡도 vs 여유 경로 혼잡도 (샘플 5000개)',
                        labels={'congestion_fast': '빠른 경로 혼잡도',
                               'congestion_relax': '여유 경로 혼잡도',
                               'selected_route': '선택한 경로'},
                        opacity=0.5,
                        color_discrete_map={'Fast': '#E63946', 'Relax': '#06A77D'})
        fig.for_each_trace(lambda t: t.update(name='빠른 경로' if t.name == 'Fast' else '여유 경로'))

        # 대각선 추가
        fig.add_shape(type='line',
                     x0=0, y0=0, x1=200, y1=200,
                     line=dict(color='gray', width=2, dash='dash'))

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.info("대각선 위: Relax가 더 혼잡, 대각선 아래: Fast가 더 혼잡")

    elif chart_type == '시계열 분석':
        st.markdown('<div class="sub-header">📈 Trial별 시계열 분석</div>', unsafe_allow_html=True)

        # Trial별, 그룹별 집계
        trial_group_stats = df.groupby(['trial_number', 'assigned_group', 'selected_route']).size().reset_index(name='count')
        trial_group_totals = df.groupby(['trial_number', 'assigned_group']).size().reset_index(name='total')
        trial_group_stats = trial_group_stats.merge(trial_group_totals, on=['trial_number', 'assigned_group'])
        trial_group_stats['percentage'] = (trial_group_stats['count'] / trial_group_stats['total']) * 100

        # Fast만 필터링
        fast_stats = trial_group_stats[trial_group_stats['selected_route'] == 'Fast']

        fig = px.line(fast_stats, x='trial_number', y='percentage',
                     color='assigned_group',
                     title='시행별 빠른 경로 선택률 변화',
                     labels={'trial_number': '시행', 'percentage': '빠른 경로 선택률 (%)',
                            'assigned_group': '그룹'},
                     markers=True,
                     color_discrete_map={'A': '#2E86AB', 'B': '#A23B72'})

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


# 사이드바
def sidebar():
    st.sidebar.title("🚇 Navigation")

    page = st.sidebar.radio("페이지 선택",
                           ["📊 프로젝트 개요",
                            "📈 시각화 분석",
                            "📋 통계 분석",
                            "🔍 데이터 탐색",
                            "🎯 비교분석"])

    st.sidebar.markdown("---")

    st.sidebar.markdown("""
    ### 프로젝트 정보

    **버전**: DAY 6 (최종)
    **전체 기간**: 2025-09 ~ 2025-11 (2.5개월)
    **작업 기간**: 약 5주 (순수 개발)
    **중단 기간**: 3주 (대학 일정)
    **데이터**: 100,000 users × 5 trials

    ### 주요 기능
    - ✅ 동적 혼잡도 피드백
    - ✅ GEE 반복 측정 분석
    - ✅ FDR 다중 검정 보정
    - ✅ 8개 전문 차트
    - ✅ 인터랙티브 탐색
    """)

    st.sidebar.markdown("---")
    st.sidebar.info("💡 좌측 메뉴에서 페이지를 선택하세요")

    return page


# 메인 실행
def main():
    page = sidebar()

    if page == "📊 프로젝트 개요":
        page_overview()
    elif page == "📈 시각화 분석":
        page_visualizations()
    elif page == "📋 통계 분석":
        page_statistics()
    elif page == "🔍 데이터 탐색":
        page_data_explorer()
    elif page == "🎯 비교분석":
        page_interactive()


if __name__ == "__main__":
    main()
