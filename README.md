# 🛠️ 반도체 공정 데이터 클래스 불균형 해결 및 Feature Selection 기반 수율(Yield) 예측 모델링

## 1. Project Overview
- **Goal**: 실제 반도체 팹 데이터와 유사한 구조를 가진 Kaggle SECOM 데이터셋을 활용해, 극단적인 클래스 불균형(정상 93% : 불량 7%)과 590개 고차원 센서 변수 속에서 수율 저하를 유발하는 핵심 변수를 선별하고 예측 모델을 구축
- **Target JD Alignment**: SK하이닉스 / 삼성전자 양산기술(장비) 직무 — 설비 데이터 기반 이상 탐지 및 수율 개선 역량 연계
- **Related Work**: 이전 사출성형 공정 불량 예측 프로젝트에서 "다중 클래스·소수 샘플 문제를 무작정 여러 알고리즘으로 풀기보다, 문제 정의(대상 범위) 자체를 재구성해야 성능이 개선된다"는 인사이트를 얻은 바 있음. 이번 SECOM 프로젝트에서도 동일한 원칙을 적용해, 알고리즘 교체 대신 변수 구성(노이즈 제거)을 통해 문제를 재정의하는 방향으로 접근함

## 2. Technical Stack
- **Language & Environment**: Python 3.12, Google Colab
- **Libraries**: Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE, SMOTEENN), XGBoost

## 3. Dataset
- Kaggle SECOM (UCI Machine Learning Repository)
- 1,567개 샘플 × 590개 센서 변수, Pass/Fail 라벨 (정상 1463 / 불량 104)

## 4. Key Troubleshooting Steps

### (1) 극단적 클래스 불균형 대응
정상:불량 = 93.4% : 6.6%의 심한 불균형 상태에서, 단순히 Recall 또는 F1만 보고 파라미터를 최적화하면 모델이 한쪽으로 완전히 쏠리는 문제를 확인함 (전 클래스를 불량으로 예측하거나, 반대로 전부 정상으로 예측). SMOTE 오버샘플링과 `scale_pos_weight` 값을 함께 조정하며 균형점을 탐색.

### (2) 고차원 변수 속 노이즈 제거 (Feature Selection)
590개 변수 중 결측 비율 40% 이상 열 제거(558개) → 분산 임계치(Variance Threshold 0.05) 필터링(267개) → XGBoost Feature Importance 기반 상위 변수 선별. 267개 상태에서는 모델이 불안정했으며, 변수 개수별 비교 실험을 통해 **30개가 최적 지점**임을 검증함.

### (3) 오버샘플링 기법 비교 (SMOTE vs SMOTEENN)
SMOTEENN(SMOTE + 이웃 정리)이 이론적으로는 더 정교한 기법이지만, 실제 적용 결과 다수 클래스(정상) 샘플이 과도하게 제거되며 오히려 모든 예측이 붕괴(Precision/Recall 0.00)하는 것을 확인. 단순 SMOTE 대비 이점이 없어 최종 모델에서는 제외.

## 5. Experiment Results

| 실험 | 변수 수 | 리샘플링 | scale_pos_weight | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 1 | 267 | SMOTE | 5 | 0.07 | 1.00 | 0.13 |
| 2 | 267 | SMOTE | 1 | 0.00 | 0.00 | 0.00 |
| 3 | 267 | SMOTE | 2 | 0.10 | 0.05 | 0.06 |
| **4 (최종 채택)** | **30** | **SMOTE** | **2** | **0.23** | **0.29** | **0.26** |
| 5 | 50 | SMOTE | 2 | 0.11 | 0.10 | 0.10 |
| 6 | 30 | SMOTEENN | 1 | 0.00 | 0.00 | 0.00 |
| 7 | 15 | SMOTE | 2 | 0.15 | 0.29 | 0.20 |

**최종 모델**: XGBoost / 변수 30개 / SMOTE / `max_depth=5, learning_rate=0.1, n_estimators=100, scale_pos_weight=2`

## 6. Lessons Learned
- 변수를 줄일수록 무조건 좋아지는 것이 아니라(15개 실험에서 Precision 하락), '노이즈 제거'와 '정보 손실' 사이에 최적 지점이 존재함을 직접 실험으로 확인
- Accuracy는 이 데이터에서 무의미한 지표이며(정상만 예측해도 93%), Precision·Recall·F1을 함께 봐야 함을 확인
- 모델의 실제 신뢰도는 base rate(6.6%) 대비 개선 여부로 판단해야 하며, 절대적인 숫자보다 시행착오 과정의 논리가 더 중요함
- **알고리즘보다 문제 정의가 성능을 좌우한다는 원칙 재확인**: 사출성형 프로젝트에서도 다중 클래스(Reason)를 여러 알고리즘으로 직접 풀려 했을 때는 성능이 낮았고, 부품(CN7/RG3)별로 데이터를 나눠 이진분류로 문제를 재정의한 뒤에야 성능이 크게 개선됐던 경험이 있음. SECOM에서도 알고리즘을 바꾸는 대신 변수 구성을 재정의(267→30개)하는 방향을 택한 것은 이 경험에서 얻은 원칙을 재적용한 것

## 7. How to Run
1. Kaggle에서 `uci-secom.csv` 다운로드
2. Google Colab에서 노트북 순서대로 실행 (데이터 로드 → EDA → 전처리 → Feature Selection → SMOTE → GridSearchCV → 평가)
3. 필요 라이브러리: `pip install imbalanced-learn xgboost`
## 🔗 Related Project

## 🔗 Related Project

웨이퍼 불량 패턴 시각화 및 FDC 관점 해석 도구는 별도 레포에서 확인하실 수 있습니다.
👉 [wafer-defect-visualization](https://github.com/hyunsle/wafer-defect-visualization)
