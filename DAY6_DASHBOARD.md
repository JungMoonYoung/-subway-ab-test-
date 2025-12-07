# DAY 6: Streamlit 대시보드

**날짜**: 2025-12-04
**구현 완료**: ✅ 인터랙티브 웹 대시보드
**실행 상태**: ✅ 정상 실행 (http://localhost:8501)

---

## 📋 대시보드 개요

**목적**: 생성된 데이터와 분석 결과를 인터랙티브하게 탐색할 수 있는 웹 대시보드

**기술 스택**:
- Streamlit 1.52.0
- Plotly 6.5.0
- Pandas 2.3.3
- PIL (이미지 표시)

---

## 🎯 구현된 기능

### 1. 📊 Overview 페이지

**주요 내용**:
- 프로젝트 개요 및 실험 설계 설명
- 주요 지표 4개 표시 (참가자, 선택 횟수, Fast 선택률, 평균 만족도)
- A/B Test 결과 비교 (A그룹 vs B그룹)
- 통계적 유의성 표시 (p < 0.001)
- Trial별 Fast 선택률 변화 인터랙티브 차트 (Plotly)
- 학습 효과 요약

**특징**:
- Plotly 라인 차트로 동적 시각화
- 목표 범위(70-75%) 표시선 포함
- 실시간 데이터 로드 (@st.cache_data)

---

### 2. 📈 시각화 갤러리 페이지

**주요 내용**:
- 8개 생성된 PNG 차트 표시
- 2열 레이아웃으로 깔끔한 배치
- 각 차트별 제목 및 설명 추가

**표시되는 차트**:
1. A vs B 그룹 비교 (95% CI)
2. Personality 유형별 분석
3. Trial별 선택 추이
4. 급함 × Personality 히트맵
5. GEE 회귀 계수
6. 만족도 분포
7. 혼잡도 동적 변화
8. 학습 효과 분석

**특징**:
- PIL을 사용한 고품질 이미지 표시
- 파일 존재 여부 확인 및 에러 처리
- 반응형 레이아웃 (use_container_width=True)

---

### 3. 📋 통계 분석 페이지

**주요 내용**:
- GEE Analysis 결과 테이블
- 계수 해석 컬럼 추가 (양/음)
- FDR Correction 결과 테이블
- 유의미한 변수 개수 표시
- Trial별 통계 테이블 및 차트
- Personality별 통계 테이블 및 바 차트

**GEE 결과 표시**:
```
group_numeric: +0.33 (p<0.001) - A그룹이 Fast 선택 확률 높음
trial_index: -0.40 (p<0.001) - Trial 증가 시 Fast 선택 감소
congestion_diff: -0.009 (p<0.001) - 혼잡도 차이가 클수록 Fast 선택 감소
time_pressure: +0.94 (p<0.001) - 급할수록 Fast 선택 증가
```

**특징**:
- 인터랙티브 Plotly 차트 (Go.Figure)
- 3개 라인 비교 (전체 평균, A그룹, B그룹)
- Personality별 바 차트 (색상 그라데이션)

---

### 4. 🔍 데이터 탐색 페이지

**주요 내용**:
- 다중 필터 기능:
  - 그룹 선택 (A, B)
  - 경로 선택 (Fast, Relax)
  - Personality 선택 (효율지향, 중립, 편안함지향)
  - Trial 선택 (1-5)
- 표시할 컬럼 선택 가능
- 필터링된 데이터 테이블 표시 (최대 1000 rows)
- 통계 요약 (Fast 선택률, 평균 만족도, 평균 혼잡도 차이)
- 수치형 변수 기술통계 (선택 시)

**특징**:
- 실시간 필터링 (st.multiselect)
- 유연한 컬럼 선택
- 동적 통계 계산

---

### 5. 🎯 인터랙티브 차트 페이지

**주요 내용**:
4가지 차트 유형 제공:

#### a) 선택률 비교
- 그룹별 경로 선택 비율 (Bar Chart)
- Fast/Relax 색상 구분
- Group by 차트

#### b) 만족도 분포
- Violin Plot (박스플롯 포함)
- 그룹 및 경로별 분포 비교
- 분포 형태 시각화

#### c) 혼잡도 산점도
- Fast 혼잡도 vs Relax 혼잡도
- 5000개 샘플링 (성능 최적화)
- 대각선 기준선 표시
- 선택한 경로별 색상 구분

#### d) 시계열 분석
- Trial별 Fast 선택률 변화
- A그룹 vs B그룹 비교
- 라인 차트 + 마커

**특징**:
- 완전 인터랙티브 (Plotly)
- 줌, 팬, 호버 기능
- 동적 범례

---

## 🎨 UI/UX 디자인

### 스타일링
```css
- 메인 헤더: 3rem, #2E86AB (파란색)
- 서브 헤더: 1.5rem, #A23B72 (보라색)
- Metric 카드: 회색 배경, 파란색 좌측 테두리
- Success Box: 초록색 배경, 초록색 좌측 테두리
```

### 레이아웃
- **Wide layout**: 전체 화면 활용
- **Sidebar**: 왼쪽 네비게이션 메뉴
- **Column 레이아웃**: 2-4열 반응형
- **Container width**: use_container_width=True (반응형)

### 색상 팔레트
- 그룹 A: `#2E86AB` (파란색)
- 그룹 B: `#A23B72` (보라색)
- Fast: `#E63946` (빨간색)
- Relax: `#06A77D` (초록색)

---

## 🚀 실행 방법

### 1. 대시보드 실행
```bash
cd C:\claude\지하철ABTEST
streamlit run app.py
```

### 2. 포트 지정 실행
```bash
streamlit run app.py --server.port=8501
```

### 3. 헤드리스 모드 (서버)
```bash
streamlit run app.py --server.headless=true --server.port=8501
```

### 4. 브라우저에서 접속
```
Local URL: http://localhost:8501
Network URL: http://172.30.1.12:8501
```

---

## 📊 데이터 캐싱

**성능 최적화**:
```python
@st.cache_data
def load_data():
    """데이터 로드 (캐싱)"""
    df = pd.read_parquet('data/synthetic_data_dynamic.parquet')
    return df

@st.cache_data
def load_analysis_results():
    """분석 결과 로드 (캐싱)"""
    # GEE, FDR, Trial, Personality 결과 로드
    return results
```

**효과**:
- 초기 로드 1회만 수행
- 페이지 전환 시 즉시 표시
- 50,000 rows 데이터 빠른 처리

---

## 🔧 의존성 파일

**requirements.txt** (생성 권장):
```txt
streamlit==1.52.0
plotly==6.5.0
pandas==2.3.3
numpy==2.3.5
pillow==12.0.0
pyarrow==22.0.0
```

---

## 📁 파일 구조

```
지하철ABTEST/
├── app.py                          # 대시보드 메인 파일
├── config.py                       # 설정 파일
├── data/
│   └── synthetic_data_dynamic.parquet  # 원본 데이터
├── analysis/
│   ├── gee_ar1_results.csv        # GEE 결과
│   ├── fdr_correction.csv         # FDR 결과
│   ├── trial_level_stats.csv      # Trial 통계
│   └── personality_stats.csv      # Personality 통계
├── figures/
│   ├── 01_ab_comparison.png       # 차트 1
│   ├── 02_personality_breakdown.png
│   ├── ...
│   └── 08_learning_effect.png     # 차트 8
└── DAY6_DASHBOARD.md              # 이 문서
```

---

## ✅ 기능 체크리스트

### 데이터 표시
- [x] 원본 데이터 로드 및 표시
- [x] 필터링 기능
- [x] 컬럼 선택 기능
- [x] 통계 요약

### 시각화
- [x] 8개 PNG 차트 표시
- [x] 인터랙티브 Plotly 차트 5개
- [x] 반응형 레이아웃

### 통계 분석
- [x] GEE 결과 테이블
- [x] FDR 결과 테이블
- [x] Trial별 통계
- [x] Personality별 통계

### UI/UX
- [x] 사이드바 네비게이션
- [x] 5개 페이지 구성
- [x] 커스텀 CSS 스타일링
- [x] 아이콘 사용
- [x] 반응형 디자인

### 성능
- [x] 데이터 캐싱 (@st.cache_data)
- [x] 샘플링 (산점도 5000개)
- [x] 효율적인 데이터 로드

---

## 🎯 주요 인사이트 (대시보드에서 확인 가능)

### 1. A/B Test 결과
- A그룹 Fast 선택률: **74.04%**
- B그룹 Fast 선택률: **68.33%**
- 차이: **5.71%p** (p < 0.001, 매우 유의미)

### 2. 학습 효과
- Trial 1: 92.16% (초기 쏠림)
- Trial 3: 57.13% (큰 조정)
- Trial 5: 65.72% (안정화)
- **전체 평균: 71.18%** (목표 달성)

### 3. 동적 혼잡도 효과
- `congestion_diff` 계수: **-0.009 (p<0.001)**
- 혼잡도가 높을수록 Fast 선택 감소
- 피드백 루프 작동 확인

### 4. Personality 효과
- 효율지향: Fast 선택률 높음
- 편안함지향: Relax 선택률 높음
- `personality_numeric` 계수: **+0.60 (p<0.001)**

---

## 🚀 향후 개선 가능 사항

### 추가 기능 (선택)
1. **PDF 리포트 생성**: 차트 및 통계를 PDF로 다운로드
2. **실시간 시뮬레이션**: 파라미터 조정 후 즉시 재실행
3. **비교 모드**: 여러 시뮬레이션 결과 비교
4. **통계 검정 도구**: 직접 검정 수행 가능
5. **머신러닝 예측**: 사용자 선택 예측 모델

### UI 개선 (선택)
1. 다크 모드 지원
2. 차트 다운로드 버튼
3. 데이터 다운로드 (CSV, Excel)
4. 커스텀 테마
5. 애니메이션 효과

---

## 📊 성능 지표

**데이터 크기**:
- 원본 데이터: 50,000 rows × 18 columns
- 로드 시간: < 1초 (캐싱 적용)
- 메모리 사용: ~50MB

**페이지 로드 속도**:
- Overview: < 1초
- 시각화 갤러리: < 2초 (이미지 로드)
- 통계 분석: < 1초
- 데이터 탐색: < 1초
- 인터랙티브 차트: < 2초

---

## 🎉 최종 결과

**DAY 6 완료!**

- ✅ Streamlit 대시보드 구현 완료
- ✅ 5개 페이지, 20+ 차트/테이블
- ✅ 완전 인터랙티브
- ✅ 고성능 (캐싱 최적화)
- ✅ 전문적인 UI/UX
- ✅ 반응형 디자인
- ✅ 정상 실행 (http://localhost:8501)

**총 구현 시간**: ~30분
**코드 라인 수**: ~580 lines

**대시보드 URL**:
```
http://localhost:8501
```

---

## 📝 사용 가이드

### 1. 대시보드 시작
```bash
# 터미널에서 실행
cd C:\claude\지하철ABTEST
streamlit run app.py
```

### 2. 페이지 탐색
- 좌측 사이드바에서 페이지 선택
- 5개 페이지 순서대로 탐색 권장

### 3. 필터 사용 (데이터 탐색 페이지)
- 그룹, 경로, Personality, Trial 선택
- 실시간 필터링 적용
- 통계 자동 업데이트

### 4. 차트 인터랙션
- 마우스 호버: 상세 정보 표시
- 드래그: 줌 인
- 더블 클릭: 줌 아웃
- 범례 클릭: 데이터 시리즈 토글

### 5. 대시보드 종료
- 터미널에서 `Ctrl + C`

---

**구현 완료일**: 2025-12-04
**작성자**: Claude
**최종 상태**: ✅ 프로덕션 배포 가능
