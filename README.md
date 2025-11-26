# Fraud Detection System - Machine Learning Pipeline

## 📋 Project Overview

A comprehensive end-to-end machine learning pipeline for credit card fraud detection, built with real-world data engineering and MLOps best practices. This system processes raw transaction data through feature engineering, model selection, and hyperparameter tuning to create a high-performance fraud detection model.

**Key Achievement**: Achieved **0.9916 AUC** and **89% fraud recall** on highly imbalanced transaction data (3.5% fraud rate).

## 🎯 Business Problem

Credit card fraud causes billions in losses annually. This project addresses the challenge of detecting fraudulent transactions in real-time while handling:
- **Severe class imbalance** (96.5% legitimate vs 3.5% fraudulent)
- **High-dimensional data** (434 raw features)
- **Sparse identity information** (only 24% coverage)
- **Complex temporal patterns**

## 📊 Dataset

### Source
- **Transactions**: 590,540 training samples, 506,691 test samples
- **Features**: 434 raw features including transaction details, card information, device data, and identity features
- **Identity Data**: Supplemental information available for 24.42% of transactions

### Key Data Challenges
- **Missing Data**: 74 features with >80% missing values, 214 features with >50% missing
- **Class Imbalance**: 27.6:1 ratio (non-fraud:fraud)
- **High Cardinality**: Multiple categorical features with 50+ unique values

## 🏗️ System Architecture

### 1. Data Preparation (`data_prep.py`)
**Objective**: Raw data cleaning, merging, and exploratory analysis

**Key Features:**
- Merged transaction and identity datasets
- Comprehensive missing value analysis
- Outlier detection and statistical profiling
- Target variable analysis revealing fraud patterns

**Insights Discovered:**
- Fraud transactions have **8.1% fewer missing values**
- Transaction amounts are highly skewed (skewness=14.37)
- **11.26%** of transactions are outliers with **5.07%** fraud rate vs **3.30%** in normal range

### 2. Feature Engineering (`feature_engineering.py`)
**Objective**: Transform raw features into predictive signals using domain insights

**Feature Categories Created:**

#### A. Transaction Features
- `TransactionAmt_log`: Log transformation for skewed amounts
- `is_amount_outlier`: IQR-based outlier detection
- `amount_zscore`: Statistical normalization

#### B. Temporal Features
- Cyclical encoding: `hour_sin`, `hour_cos`, `day_sin`, `day_cos`
- Fraud peak hours identification
- Weekend/night risk flags

#### C. Risk-Based Features
- **Product Risk Scores**: C=3 (11.5% fraud), H=2 (5.9% fraud), W=0 (2.2% fraud)
- **Card Behavior**: Single-use card detection, frequency encoding
- **Email Risk**: Protonmail domains (40-95% fraud rate), high-risk domain flags

#### D. Advanced Features
- **Composite Fraud Risk Score**: Weighted combination of risk factors
- **Behavioral Clustering**: K-means clustering on transaction patterns
- **Missingness Indicators**: Group-wise missing data patterns

**Real Estate Patterns Implemented:**
- Frequency encoding for high-cardinality features
- Text extraction (email providers, device OS)
- Weighted scoring systems
- Feature interactions

### 3. Feature Selection (`feature_selection.py`)
**Objective**: Strategic feature reduction while preserving predictive power

**Selection Strategy:**
- **Initial**: 434 → 194 features (55% reduction)
- **Correlation-based**: Top 150 V-features by fraud correlation
- **Domain-driven**: Core business features (cards, products, temporal)
- **Final**: 110 highly predictive features

### 4. Model Selection & Tuning (`model_selection.py`)
**Objective**: Comprehensive model evaluation and hyperparameter optimization

**Two-Phase Evaluation:**
- **Phase A**: Quick screening with 3-fold CV
- **Phase B**: Detailed evaluation of top models with 5-fold CV

## 🤖 Models Evaluated

| Model | CV AUC | Validation AUC | Key Parameters |
|-------|--------|----------------|----------------|
| **XGBoost** | 0.9586 | 0.9648 | scale_pos_weight=10, n_estimators=300 |
| LightGBM | 0.9472 | 0.9504 | is_unbalance=True |
| Random Forest | 0.9163 | 0.9187 | class_weight='balanced' |
| Logistic Regression | 0.8465 | 0.8507 | L2 regularization |

## 🏆 Final Model Performance

### Hyperparameter Tuning Results
- **Best Model**: XGBoost
- **Best Parameters**: 
  - `n_estimators`: 300
  - `max_depth`: 8
  - `learning_rate`: 0.1
  - `min_child_weight`: 5

### Validation Performance
```
AUC: 0.9916
Average Precision: 0.9083

Classification Report:
              precision    recall  f1-score   support

Non-Fraud       1.00      0.99      0.99    113,975
Fraud           0.73      0.89      0.80      4,133
```

## 🚀 Technical Highlights

### Data Engineering Excellence
- **Strategic Missing Value Handling**: Used missingness patterns as features
- **Advanced Temporal Encoding**: Cyclical encoding for time features
- **Risk-Based Feature Creation**: Domain-driven feature engineering
- **Real Estate Pattern Implementation**: Professional feature engineering techniques

### Model Development Rigor
- **Class Imbalance Handling**: Multiple strategies (class weights, scale_pos_weight)
- **Stratified Sampling**: Maintained imbalance in all splits
- **Comprehensive Validation**: Both AUC and Average Precision metrics
- **Production-Ready Pipelines**: Full preprocessing integration

### MLOps Best Practices
- **Modular Pipeline**: Separate components for easy maintenance
- **Data Validation**: Comprehensive feature consistency checks
- **Model Serialization**: Full pipeline saving and loading
- **Reproducible Experiments**: Random state management

## 📁 Project Structure

```
fraud-detection/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned and engineered features
├── models/                     # Saved models and metadata
├── notebooks/
│   ├── 01_data_prep.ipynb     # Data exploration and cleaning
│   ├── 02_feature_engineering.ipynb  # Advanced feature creation
│   ├── 03_feature_selection.ipynb    # Strategic feature reduction
│   └── 04_model_selection.ipynb      # Model evaluation and tuning
└── src/
    ├── data_prep.py           # Data processing functions
    ├── feature_engineering.py # Feature creation utilities
    ├── feature_selection.py   # Feature selection algorithms
    └── model_selection.py     # Model training and evaluation
```

## 🛠️ Installation & Usage

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib
```

### Running the Pipeline
1. **Data Preparation**: `python src/data_prep.py`
2. **Feature Engineering**: `python src/feature_engineering.py`
3. **Feature Selection**: `python src/feature_selection.py`
4. **Model Training**: `python src/model_selection.py`

## 📈 Key Insights & Business Impact

### Critical Fraud Patterns Discovered
1. **Email Domains**: Protonmail accounts show 40-95% fraud rates
2. **Device Patterns**: "Other" OS and Android browsers have significantly higher fraud
3. **Temporal Trends**: Fraud peaks during specific hours and weekends
4. **Behavioral Signals**: Single-use cards have 4.24% fraud rate vs 3.5% overall

### Model Strengths
- **High Recall**: 89% of fraud cases detected
- **Excellent Precision**: 73% precision on fraud classification
- **Robust Performance**: Consistent across validation folds
- **Business Interpretable**: Risk scores and feature importance available

## 🔮 Future Enhancements

1. **Ensemble Methods**: Combine top models for improved performance
2. **Anomaly Detection**: Incorporate unsupervised learning approaches
3. **Drift Monitoring**: Implement data and concept drift detection
4. **Explainable AI**: SHAP values for feature importance interpretation

## 👨‍💻 Skills Demonstrated

- **Data Engineering**: Pandas, NumPy, Feature Engineering
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, Model Selection
- **MLOps**: Pipeline Development, Model Serialization, Validation
- **Statistical Analysis**: Correlation Analysis, Outlier Detection, Imbalance Handling
- **Data Visualization**: Matplotlib, Seaborn, Performance Metrics
- **Software Engineering**: Modular Design, Code Organization, Documentation

---

**Developed with 🚀 by a passionate Data Scientist | MLOps Engineer**
