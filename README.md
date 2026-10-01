# Real Estate Investment Advisor
## Predicting Property Profitability & Future Value

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-green.svg)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Zero Leakage](https://img.shields.io/badge/Methodology-Zero_Leakage-brightgreen.svg)](#target-leakage-audit--methodology-realignment)

An enterprise-grade, methodologically audited machine learning and decision-analytics platform designed to evaluate residential real estate opportunities across major Indian metropolitan markets.

---

## 1. Executive Summary & Three-Module Architecture

Following an exhaustive methodological audit to eliminate target leakage, the system operates across three decoupled modules:

```
+---------------------------------------------------------------------------------------------------+
|                                REAL ESTATE INVESTMENT ADVISOR                                      |
+------------------------------------+----------------------------------+---------------------------+
| MODULE A: GENUINE ML VALUATION     | MODULE B: MCDA INVESTMENT SCORE  | MODULE C: SCENARIO GROWTH |
+------------------------------------+----------------------------------+---------------------------+
| - Baseline Listing Price Estimate   | - Multi-Criteria Decision Score  | - 5-Year Scenario Horizon |
| - Target: Price_in_Lakhs           | - Range: 0 to 100 Index          | - Multi-Tier Projections: |
| - Zero Target Leakage:             | - Weightings:                    |     * Conservative (5.0%) |
|     * Excludes Price_per_SqFt      |     * 40% Infrastructure/Utility |     * Moderate (7.5%)     |
|     * Excludes infra_score         |     * 35% Valuation Safety Margin|     * Optimistic (10.0%)  |
|     * Excludes synthetic targets   |     * 25% Structural Freshness   | - Labeled: Illustrative   |
| - Champion: Ridge Baseline         | - Champion: LightGBM (93.6% Acc) |   scenario assumptions    |
+------------------------------------+----------------------------------+---------------------------+
```

---

## 2. Target Leakage Audit & Academic Integrity

In initial experiments, models achieved artificial metrics ($F1 = 0.999$, $R^2 = 1.000$) due to target leakage:
- `Price_per_SqFt` was computed as $\text{Price\_in\_Lakhs} / \text{Size\_in\_SqFt}$ and fed back into regression predicting `Price_in_Lakhs`.
- Synthetic scoring formulas were fed to classifiers as input features.

### Corrective Actions Taken
1. **Permanent Exclusion**: `Price_per_SqFt`, `infra_score`, and `investment_score_raw` were removed from all ML feature matrices.
2. **Target Clarification**: `Price_After_5_Years` was banned as a training target because it was synthetically generated.
3. **Automated Unit Testing**: Tests in `tests/test_leakage.py` strictly assert that no target-derived variables enter the ML pipelines.
4. **Honest Dataset Limitation Disclosure**:
   The listing prices in `india_housing_prices.csv` exhibit near-zero correlation with physical attributes ($|\rho| < 0.005$) and follow a uniform random distribution $U(10, 500)$. With circular leakage removed, the model achieves $R^2 \approx 0.000$ with an MAE of ₹122.3 Lakhs (matching the theoretical mean absolute error of a uniform distribution).
   Rather than hiding this with circular features, this limitation is transparently documented: the raw dataset is methodologically invalid for demonstrating genuine predictive performance on listing prices alone.

---

## 3. Champion Model Benchmarks

All models were evaluated on a holdout test split of 50,000 observations (80/20 train/test split) and logged to MLflow (`sqlite:///mlflow.db`).

### Module A: Property Valuation Regression (`Price_in_Lakhs`)
| Model Candidate | MAE (₹ Lakhs) | RMSE (₹ Lakhs) | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression Baseline** | **122.32** | **141.21** | **-0.0002** | **Champion** |
| Random Forest Regressor | 122.34 | 141.23 | -0.0005 | Evaluated |
| LightGBM Regressor | 122.39 | 141.34 | -0.0021 | Evaluated |

### Module B: Investment Potential Classification (MCDA $\ge 70.0$)
| Model Candidate | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Classifier** | **93.55%** | **90.49%** | **88.62%** | **0.8955** | **0.9861** | **Champion** |
| Logistic Regression | 93.03% | 89.38% | 88.14% | 0.8875 | 0.9806 | Evaluated |
| Random Forest Classifier | 88.78% | 91.50% | 70.57% | 0.7968 | 0.9667 | Evaluated |

---

## 4. Investment Potential Scoring & Scenario Projections

### Module B: Deterministic MCDA Formula
$$\text{Investment Score} = 100 \times \left( 0.40 \times \text{Infra} + 0.35 \times \text{ValueMargin} + 0.25 \times \text{Freshness} \right)$$
* **Infrastructure ($40\%$)**: Transit accessibility ($20\%$), amenity count ($20\%$), schools ($15\%$), hospitals ($15\%$), security ($15\%$), parking ($15\%$).
* **Valuation Safety Margin ($35\%$)**: Normalized inverse of `Price_per_SqFt`, rewarding entry at a discount.
* **Asset Freshness ($25\%$)**: Normalized inverse of `Age_of_Property`, rewarding newer structures.

### Module C: 5-Year Capital Growth Scenarios
$$V_5 = V_0 \times (1 + r)^5$$
* **Conservative ($5.0\%$ p.a.)**: $+27.63\%$ cumulative capital growth over 5 years.
* **Moderate ($7.5\%$ p.a.)**: $+43.56\%$ cumulative capital growth over 5 years.
* **Optimistic ($10.0\%$ p.a.)**: $+61.05\%$ cumulative capital growth over 5 years.

> **Methodological Disclosure**: Rates are labeled as illustrative scenario assumptions for financial planning and stress testing, not as guaranteed forecasts or official RBI benchmark returns.

---

## 5. Repository Structure

```
Real Estate Investment Advisor/
├── app.py                     # Interactive Streamlit application
├── requirements.txt           # Project dependencies
├── mlflow.db                  # SQLite database tracking MLflow experiment runs
├── india_housing_prices.csv   # Raw source dataset (250,000 rows)
├── data/
│   └── processed/             # Cleaned datasets (Parquet / CSV)
├── docs/
│   ├── PRD.md                 # Product Requirements Document
│   ├── ARCHITECTURE.md        # Technical architecture specification
│   ├── DECISIONS.md           # Architecture Decision Records (ADRs)
│   ├── MEMORY.md              # Project memory and operational state
│   ├── DESIGN.md              # UI/UX design specifications
│   ├── RULES.md               # Engineering and anti-leakage rules
│   └── TASKS.md               # Task completion tracking
├── models/
│   ├── classification_model.joblib # Serialized champion LightGBM classifier
│   └── regression_model.joblib     # Serialized champion Ridge regressor
├── reports/
│   ├── MODEL_BENCHMARKS.md    # Model performance and feature importance report
│   ├── eda_summary.json       # Numerical insights across all 20 business questions
│   └── figures/               # 20 high-resolution EDA visualization charts
├── src/
│   ├── config.py              # Central configuration and feature schemas
│   ├── data_loader.py         # Data ingestion and schema validation
│   ├── feature_engineering.py # Feature creation, MCDA scoring, scenario projections
│   ├── preprocessor.py        # Scikit-learn ColumnTransformer pipeline
│   ├── mlflow_tracker.py      # MLflow logging and experiment management
│   ├── train_classifier.py    # Classification model training suite
│   ├── train_regressor.py     # Valuation regression training suite
│   ├── evaluate.py            # Benchmark generation and report export
│   └── predict.py             # Unified inference engine
└── tests/
    ├── test_leakage.py        # Automated target leakage assertion tests
    └── test_prediction.py     # End-to-end prediction integration tests
```

---

## 6. Getting Started

### Installation
```bash
# 1. Clone repository & install dependencies
pip install -r requirements.txt
```

### Running the Test Suite
```bash
# Run unit and target-leakage tests
python -m unittest discover tests
```

### Launching the Streamlit Application
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`.

### Viewing MLflow Experiments
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Access the MLflow tracking dashboard at `http://localhost:5000`.
