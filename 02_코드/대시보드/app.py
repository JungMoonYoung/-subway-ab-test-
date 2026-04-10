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


def page_insight():
    """📌 SUMMARY 브리핑 (대시보드 진입 첫 화면)

    채용담당자/현업 사용자에게 '왜 이 실험을 했고, 어떻게 풀었으며,
    무엇을 발견했고, 어떻게 활용할 수 있는지'를 한 페이지에 정리한다.
    """

    # ---------- 컴팩트 히어로 ----------
    st.markdown("""
    <div style="padding:20px 26px;border-radius:14px;
                background:linear-gradient(135deg,#1a2a4a 0%,#3a1b4e 100%);
                border:1px solid rgba(255,255,255,0.08);margin-bottom:18px;">
        <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
            <div style="font-size:24px;font-weight:800;color:#fff;">
                🚇 지하철 이용행동 A/B Test
            </div>
            <div style="font-size:14px;color:#9ec5ff;letter-spacing:0.5px;">
                통제 불가능한 변수 속에서 실험을 설계하고, 통계적으로 검증할 수 있습니다
            </div>
        </div>
        <div style="font-size:14px;color:#c9d1e8;margin-top:8px;line-height:1.6;">
            <b style="color:#fff;">100,000명 × 5 Trial = 500,000 rows</b> 시뮬레이션 +
            <b style="color:#9ec5ff;">GEE(AR1) 반복측정 분석</b>으로
            UI가 행동에 미치는 진짜 영향을 분리합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- 4×4 매트릭스 ----------
    C_DESIGN = "#2E86AB"   # Navy
    C_METHOD = "#A23B72"   # Magenta
    C_FIND   = "#F18F01"   # Orange

    td_base = (
        "padding:18px 20px;vertical-align:top;"
        "border:1px solid rgba(255,255,255,0.08);"
        "background:rgba(255,255,255,0.03);"
    )
    td_label = (
        "padding:18px 20px;vertical-align:top;"
        "border:1px solid rgba(255,255,255,0.08);"
        "background:rgba(158,197,255,0.08);"
        "width:14%;"
    )

    def _header_cell(icon, title, color, question):
        return (
            f'<td style="{td_base}border-top:3px solid {color};width:28.6%;">'
            f'<div style="font-size:32px;line-height:1;">{icon}</div>'
            f'<div style="font-size:22px;font-weight:800;color:{color};margin:10px 0 6px 0;letter-spacing:-0.5px;">{title}</div>'
            f'<div style="color:#c9d1e8;font-size:13.5px;font-style:italic;line-height:1.5;">{question}</div>'
            f'</td>'
        )

    def _label_cell(icon, text):
        return (
            f'<td style="{td_label}">'
            f'<div style="font-size:22px;">{icon}</div>'
            f'<div style="color:#fff;font-size:16px;font-weight:800;margin-top:6px;line-height:1.3;">{text}</div>'
            f'</td>'
        )

    def _content_cell(html_content, color=None, bold=False):
        weight = "600" if bold else "400"
        col = color if color else "#c9d1e8"
        return (
            f'<td style="{td_base}">'
            f'<div style="color:{col};font-size:14px;line-height:1.8;font-weight:{weight};">{html_content}</div>'
            f'</td>'
        )

    table_html = (
        '<table style="width:100%;border-collapse:separate;border-spacing:0;border-radius:14px;overflow:hidden;">'
        # 헤더 행
        '<tr>'
        f'{_label_cell("🗂️", "구분")}'
        f'{_header_cell("🧪", "실험 설계", C_DESIGN, "100만 명 현장 실험은 불가능하다. 그렇다면 어떻게 검증할까?")}'
        f'{_header_cell("📊", "분석 방법론", C_METHOD, "같은 사람이 5번 선택한 데이터, 독립성 가정이 무너진다")}'
        f'{_header_cell("💡", "핵심 발견", C_FIND, "UI · 상황 · 성향 중 무엇이 선택을 가장 좌우하는가?")}'
        '</tr>'
        # 행 1: 문제 정의
        '<tr>'
        f'{_label_cell("🎯", "문제 정의")}'
        f'{_content_cell("현실의 불확실성을 어떻게<br>시뮬레이션에 반영할 것인가")}'
        f'{_content_cell("반복측정 데이터에서<br>진짜 인과 효과를 분리하기")}'
        f'{_content_cell("수치 나열이 아닌<br>의미 있는 패턴 도출")}'
        '</tr>'
        # 행 2: 접근 방법
        '<tr>'
        f'{_label_cell("🔬", "접근 방법")}'
        f'{_content_cell("• 성격 3유형 분포 생성<br>• <b>동적 혼잡도 피드백 루프</b><br>• Bernoulli 확률적 선택")}'
        f'{_content_cell("• Two-Proportion Z-Test<br>• <b>GEE (AR1) 반복측정</b><br>• <b>FDR 다중검정 보정</b>")}'
        f'{_content_cell("• 6개 변수 OR 순위 산출<br>• Trial별 학습 효과 추적<br>• Personality × 시간압박 교차")}'
        '</tr>'
        # 행 3: 왜 이 방법인가
        '<tr>'
        f'{_label_cell("📌", "왜 이 방법인가")}'
        f'{_content_cell("정적 시뮬레이션은 Fast 98% 쏠림.<br><b>동적 피드백</b>으로 71% 안정화")}'
        f'{_content_cell("단순 t-test는 같은 사람의<br>반복 관측 무시.<br><b>GEE가 상관구조 통제</b>")}'
        f'{_content_cell('수치 나열이 아닌<br><b>"UI보다 상황·성향이 핵심"</b><br>으로 해석')}'
        '</tr>'
        # 행 4: 현업 활용
        '<tr>'
        f'{_label_cell("💼", "현업 활용")}'
        f'{_content_cell("→ 동적 피드백 UI로<br><b>혼잡 노선 이용객 자율 분산</b>", color=C_DESIGN, bold=True)}'
        f'{_content_cell("→ 시간압박·성향 조합으로<br><b>선택 확률 85~94% 예측</b>", color=C_METHOD, bold=True)}'
        f'{_content_cell("→ 규제·요금 없이 UI만으로<br><b>3일차부터 행동 안정화</b>", color=C_FIND, bold=True)}'
        '</tr>'
        '</table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("")

    # ---------- 영향 요인 OR 순위 ----------
    st.markdown('<h3 style="margin-top:18px;">🏆 영향 요인 OR 순위 (GEE 회귀 결과)</h3>', unsafe_allow_html=True)
    st.caption("같은 사람의 5회 반복 선택을 GEE(AR1) 모델로 분석한 결과입니다.")

    or_df = pd.DataFrame({
        "순위": ["1", "2", "3", "4", "5", "6"],
        "요인": ["시간 압박", "효율지향 성향", "UI (A그룹)", "시간 차이", "Trial 증가", "혼잡도 차이"],
        "OR": ["2.55", "1.81", "1.39", "1.14", "0.67", "0.99"],
        "해석": [
            "빠른 경로 선택 확률 155% 증가",
            "개인 성향에 따라 81% 증가",
            "시각적 효과로 39% 증가",
            "14% 차이만",
            "반복 경험으로 33% 감소 (학습 효과)",
            "미약하나 유의 (p<0.001)",
        ],
    })
    st.dataframe(or_df, use_container_width=True, hide_index=True)

    st.success(
        "💡 **핵심 발견**: 사용자 선택 행동은 **UI보다 상황(시간 압박)·성향(효율지향) 변수의 영향을 더 크게 받는다.**"
    )

    st.markdown("---")

    # ---------- 상세 탭 ----------
    itab1, itab2, itab3 = st.tabs([
        "🧭 방법론 선택의 근거",
        "🛠️ 문제 해결 경험",
        "💼 비즈니스 활용 시나리오",
    ])

    with itab1:
        st.caption("\"통계를 돌린 것\"과 \"통계를 고른 것\"은 다릅니다. 이 프로젝트에서 각 방법을 선택한 이유입니다.")
        method_df = pd.DataFrame({
            "분석 단계": ["데이터 확보", "현실성 확보", "기본 검정", "반복측정", "다중 검정"],
            "흔한 접근": [
                "실제 사용자 모집·실험",
                "정적 무작위 생성",
                "t-test",
                "시점별 t-test 반복",
                "그냥 p<0.05",
            ],
            "이 프로젝트 선택": [
                "현실 분포 기반 시뮬레이션",
                "동적 혼잡도 피드백 루프",
                "Z-Test + Cohen's h",
                "GEE (AR1)",
                "FDR (Benjamini-Hochberg)",
            ],
            "선택 이유": [
                "100만 명 현장 실험은 윤리·비용 불가능",
                "정적 생성은 Fast 98% 쏠림 → 의미 없음",
                "이진 선택 + 효과 크기까지 측정",
                "같은 사람 5회 관측 = 상관 존재. 무시 시 p값 왜곡",
                "Type I error 폭증 방지, Bonferroni보다 검정력 유지",
            ],
        })
        st.dataframe(method_df, use_container_width=True, hide_index=True)

    with itab2:
        st.caption("실험 설계 과정에서 마주친 두 가지 핵심 문제와 해결 과정입니다.")

        st.markdown("#### 1️⃣ 정적 환경의 한계를 극복한 동적 A/B 시뮬레이션")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.error("**🚨 문제**\n\n"
                     "초기 시뮬레이션에서 Fast 선택률 **98% 쏠림** → "
                     "A/B Test 자체가 무의미. 계수 조정·혼잡도 차이 모두 실패")
        with c2:
            st.info("**🔧 해결**\n\n"
                    "\"사용자 선택 → 혼잡도 변화 → 다음 선택에 영향\" "
                    "**동적 피드백 루프** 설계 → "
                    "현실적 분포 **71.18% 안정화**")
        with c3:
            st.success("**🎓 배운 점**\n\n"
                       "정적 테스트는 현실을 반영하지 못한다. "
                       "**현실을 제대로 반영하려면 동적 관계를 가진 테스트 구성이 필수.**")

        st.markdown("")
        st.markdown("#### 2️⃣ 동일 사용자의 반복 선택 → 독립성 가정 위배")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.error("**🚨 문제**\n\n"
                     "사용자의 5회 선택은 독립이 아닌데, "
                     "시간 상관관계를 무시하고 분석하면 **p값이 왜곡**됨")
        with c2:
            st.info("**🔧 해결**\n\n"
                    "**GEE 모델 + AR(1)** 적용으로 "
                    "반복 측정 효과 통제 + "
                    "**시간순 선택 간 상관관계 반영**")
        with c3:
            st.success("**🎓 배운 점**\n\n"
                       "결론보다 **데이터 구조에 맞는 모델 선택**이 "
                       "분석의 핵심. 통계를 '돌리는' 게 아니라 '고르는' 능력.")

    with itab3:
        st.caption("이 실험 결과를 실제 교통·플랫폼 운영에 적용할 수 있는 3가지 시나리오입니다.")

        biz_card = (
            "padding:18px;border-radius:12px;height:100%;"
            "background:rgba(255,255,255,0.03);"
            "border:1px solid rgba(158,197,255,0.2);"
        )

        # 1. 러시아워 혼잡도 분산
        st.markdown(f"#### 🚇 1. 러시아워 혼잡도 분산 전략")
        st.markdown(
            f'<div style="{biz_card}">'
            f'<div style="color:#c9d1e8;font-size:14px;line-height:1.7;">'
            f'러시아워에 <b style="color:#fff;">여유 경로 UI를 노출</b>해 혼잡 노선 이용객을 분산. '
            f'2호선 강남~잠실 등에 <b style="color:{C_DESIGN};">"쾌적·앉아서 이용"</b> 문구만 적용해도 효과 가능.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        # 2. 개인화 경로 추천
        st.markdown(f"#### 🎯 2. 개인화 경로 추천")
        st.markdown(
            f'<div style="{biz_card}">'
            f'<div style="color:#c9d1e8;font-size:14px;line-height:1.7;">'
            f'시간 압박+성향 조합으로 선택 확률 <b style="color:{C_METHOD};">85~94% 예측 가능</b>. '
            f'출퇴근·효율형엔 시간 강조, 여유·편안형엔 쾌적성 강조 → <b style="color:#fff;">동일 노선에서도 사용자별 최적 경로 제안</b>.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        # 3. 동적 피드백의 학습 효과
        st.markdown(f"#### 🔁 3. 동적 피드백의 학습 효과")
        st.markdown(
            f'<div style="{biz_card}">'
            f'<div style="color:#c9d1e8;font-size:14px;line-height:1.7;">'
            f'혼잡도 지수만 UI로 노출해도 사용자가 스스로 여유 경로를 선택. '
            f'<b style="color:{C_FIND};">규제·요금 없이 UI만으로 행동 변화 유도</b>, '
            f'추가 비용 없이 <b style="color:#fff;">3일차부터 분산 행동 안정화</b>.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### 🎯 공통 지향점")
        st.markdown('> **"A/B Test는 \'어느 UI가 좋은가\'가 아니라, \'어떤 조건에서 사람이 무엇을 선택하는가\'를 밝히는 도구다."**')
        g1, g2, g3 = st.columns(3)
        with g1:
            st.success("🔓 **탈직관**\n\n'감'이 아닌\nGEE·FDR 기반 엄밀 검증")
        with g2:
            st.success("🔁 **재현성**\n\n가상 데이터 → 분석 → 리포트까지\n자동화")
        with g3:
            st.success("🧠 **예측화**\n\n과거 행동 → 미래 선택 패턴\n예측 (85~94%)")


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
                           ["📌 SUMMARY",
                            "📊 프로젝트 개요",
                            "📈 시각화 분석",
                            "📋 통계 분석",
                            "🔍 데이터 탐색",
                            "🎯 비교분석"])

    st.sidebar.markdown("---")

    st.sidebar.markdown("""
    ### 주요 기능
    - ✅ 동적 혼잡도 피드백
    - ✅ GEE 반복 측정 분석
    - ✅ FDR 다중 검정 보정
    - ✅ 8개 전문 차트
    - ✅ 인터랙티브 탐색
    """)

    st.sidebar.markdown("---")

    return page


# 메인 실행
def main():
    page = sidebar()

    if page == "📌 SUMMARY":
        page_insight()
    elif page == "📊 프로젝트 개요":
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
