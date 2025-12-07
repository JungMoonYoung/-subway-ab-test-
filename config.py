"""
Configuration file for Subway A/B Test Simulation
모든 시뮬레이션 파라미터를 중앙에서 관리
"""

# ============================================
# 시뮬레이션 기본 설정
# ============================================
NUM_USERS = 100000
NUM_TRIALS = 5
RANDOM_SEED = 42

# ============================================
# 로지스틱 회귀 계수
# ============================================
# 그룹별 절편 (A: 빠른 경로 중심, B: 편안한 경로 중심)
BETA_0_A = 0.3
BETA_0_B = -0.2

# 변수별 계수
BETA_TIME_PRESSURE = 1.2      # 시간 압박이 높을수록 빠른 경로 선호
BETA_PERSONALITY = 0.9        # 효율 지향 성향일수록 빠른 경로 선호
BETA_TIME_DIFF = 0.16         # 시간 차이가 클수록 빠른 경로 선호 (0.18 → 0.16로 미세조정)
BETA_PREVIOUS_CHOICE = -0.7   # 학습 효과: 이전 Fast 선택 시 다음엔 Relax 확률 증가 (-0.6 → -0.7)
BETA_CONGESTION = -0.014      # 경험한 혼잡도의 영향 (혼잡도 10% 증가당 Fast 확률 -0.14, -0.012 → -0.014)
NOISE_STD = 0.5               # 로지스틱 모델 노이즈 표준편차 (0.4 → 0.5로 증가, 더 현실적)

# ============================================
# 경로 시간 분포 (단위: 분)
# ============================================
FAST_TIME_MEAN = 25
FAST_TIME_STD = 2

RELAX_TIME_MEAN = 36
RELAX_TIME_STD = 3

# ============================================
# 혼잡도 분포 (단위: %)
# ============================================
# 동적 혼잡도 시스템 (이전 trial 선택에 따라 변동)
BASE_CONGESTION_FAST = 85           # Fast 기본 혼잡도 (%)
BASE_CONGESTION_RELAX = 45          # Relax 기본 혼잡도 (%)
CONGESTION_MULTIPLIER = 120         # 선택 비율 → 혼잡도 변환 계수
                                     # 예: 80% 선택 → +96% 혼잡도
CONGESTION_NOISE_STD = 8            # 혼잡도 랜덤 노이즈 표준편차

# 혼잡도-시간 지연 계수
DELAY_FACTOR = 0.08  # 혼잡도 1% 증가당 0.08분 지연

# 혼잡도 범위
MIN_CONGESTION = 30   # 최소 혼잡도
MAX_CONGESTION = 200  # 최대 혼잡도

# ============================================
# time_pressure 생성 파라미터
# ============================================
TIME_PRESSURE_BASELINE_MEAN = 1.0      # 평균 (0=급함, 1=보통, 2=여유)
TIME_PRESSURE_BASELINE_STD = 0.5       # 개인별 baseline 표준편차
TIME_PRESSURE_NOISE_STD = 0.3          # 회차별 랜덤 변동 표준편차

# ============================================
# 경로 시간/혼잡도 최소값
# ============================================
MIN_ROUTE_TIME_FAST = 10       # Fast Route 최소 시간 (분)
MIN_ROUTE_TIME_RELAX = 15      # Relax Route 최소 시간 (분)
MIN_CONGESTION_FAST = 50       # Fast Route 최소 혼잡도 (%)
MIN_CONGESTION_RELAX = 30      # Relax Route 최소 혼잡도 (%)

# ============================================
# 날짜 설정
# ============================================
BASE_DATE = "2025-01-06"       # 첫 측정일 (월요일)
TRIAL_INTERVAL_DAYS = 1        # trial 간격 (일)

# ============================================
# 사용자 특성 비율
# ============================================
# Personality type 분포
PERSONALITY_EFFICIENCY_RATIO = 0.55
PERSONALITY_COMFORT_RATIO = 0.35
PERSONALITY_NEUTRAL_RATIO = 0.10

# Travel frequency 분포
TRAVEL_DAILY_RATIO = 0.60
TRAVEL_WEEKLY_RATIO = 0.30
TRAVEL_RARELY_RATIO = 0.10

# ============================================
# 만족도 생성 파라미터
# ============================================
SATISFACTION_BASE = 3.0                      # 기본 만족도 점수
SATISFACTION_MATCH_BONUS_STRONG = 2.0        # 성향-선택 강한 매칭 보너스
SATISFACTION_MATCH_BONUS_NEUTRAL = 1.0       # 중립 성향 보너스
SATISFACTION_PRESSURE_PENALTY = -1.0         # 급한데 Relax 선택 시 패널티
SATISFACTION_NOISE_STD = 0.5                 # 만족도 노이즈 표준편차

# 혼잡도 기반 만족도 패널티
SATISFACTION_CONGESTION_THRESHOLD_FAST = 140   # Fast Route 혼잡도 임계값
SATISFACTION_CONGESTION_THRESHOLD_RELAX = 80   # Relax Route 혼잡도 임계값
SATISFACTION_CONGESTION_PENALTY_FACTOR = -0.015  # 혼잡도 1% 초과당 만족도 -0.015점

# ============================================
# 의사결정 시간 파라미터
# ============================================
DECISION_TIME_MEAN = 5.5                     # 평균 의사결정 시간 (초)
DECISION_TIME_STD = 1.5                      # 의사결정 시간 표준편차
DECISION_TIME_PRESSURE_EFFECT = 1.5          # time_pressure 1단계당 시간 변화 (초)
DECISION_TIME_MIN = 1.0                      # 최소 의사결정 시간 (초)

# ============================================
# 통계 검정 설정
# ============================================
ALPHA = 0.05              # 유의수준
POWER_TARGET = 0.80       # 목표 검정력
FDR_METHOD = "fdr_bh"     # Benjamini-Hochberg 보정

# Effect Size 임계값 (Cohen, 1988)
COHENS_H_SMALL = 0.2      # Small effect size
COHENS_H_MEDIUM = 0.5     # Medium effect size
