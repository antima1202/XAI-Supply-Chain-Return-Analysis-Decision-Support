# An Explainable AI Framework Integrating Customer Feedback and Transactional Data for Product Return Analysis in Fashion E-Commerce Supply Chains

**MSc Dissertation Project**

---

## Project Overview

This framework analyses confirmed product returns from a fashion e-commerce supply chain to identify operational root causes and generate targeted supply chain recommendations.

Unlike traditional return prediction systems, this framework operates **post-return**. The objective is not to predict whether a customer will return an item, but to classify **why** the return occurred and route actionable recommendations to the responsible operational team.

The system combines transactional data, customer review text, and machine learning to produce explainable, stakeholder-specific decision support outputs.

---

## Dissertation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Data Collection and Integration | ✅ Complete |
| Phase 2 | NLP Customer Feedback Analysis | ✅ Complete |
| Phase 3 | Root Cause Classification (ML) | ✅ Complete |
| Phase 4 | Explainable AI (SHAP) | ✅ Complete |
| Phase 5 | Root Cause Analysis Layer | ✅ Complete |
| Phase 6 | Risk Assessment | ✅ Complete |
| Phase 7 | Recommendation Engine | 🔄 Planned |
| Phase 8 | Operational Optimisation (Linear Programming) | 🔄 Planned |
| Phase 9 | Stakeholder Decision Support Dashboard | 🔄 Planned |
| Phase 10 | Continuous Feedback Loop | 🔄 Planned |

---

## Project Structure

```
files/
├── requirements.txt              ← Install all dependencies from here
├── run_phase2.py                 ← Run Phase 2 NLP pipeline
├── run_phase3.py                 ← Run Phase 3 ML pipeline
├── run_phase4.py                 ← Run Phase 4 SHAP pipeline
│
├── src/
│   ├── __init__.py
│   ├── phase2/
│   │   ├── __init__.py
│   │   ├── utils.py              ← Paths, logger, helpers
│   │   ├── preprocessing.py      ← Text cleaning, tokenisation, lemmatisation
│   │   ├── sentiment_analysis.py ← VADER + TextBlob scoring
│   │   ├── keyword_extraction.py ← TF-IDF keyword extraction
│   │   ├── topic_modelling.py    ← Gensim LDA topic modelling
│   │   ├── complaint_classifier.py ← NLP-driven complaint classification
│   │   ├── visualisations.py     ← All Phase 2 dissertation plots
│   │   └── pipeline.py           ← Phase 2 orchestrator
│   ├── phase3/
│   │   ├── __init__.py
│   │   ├── utils.py              ← Paths, logger, helpers
│   │   ├── feature_engineering.py ← Encoding, SMOTE, train/test split
│   │   ├── models.py             ← LR, Random Forest, XGBoost training
│   │   ├── evaluation.py         ← Metrics, confusion matrix, ROC, importance
│   │   └── pipeline.py           ← Phase 3 orchestrator
│   └── phase4/
│       ├── __init__.py
│       ├── utils.py              ← Paths, logger, helpers
│       ├── explainer.py          ← SHAP TreeExplainer, value computation
│       ├── visualisations.py     ← Global importance, beeswarm, waterfall, heatmap
│       └── pipeline.py           ← Phase 4 orchestrator
│
├── data/
│   ├── raw/
│   │   └── fashion_returns_dataset.csv        ← Phase 1 original dataset
│   └── processed/
│       └── fashion_returns_dataset_nlp.csv    ← Phase 2 enriched dataset
│
├── models/
│   ├── phase2/
│   │   ├── tfidf_vectorizer.joblib
│   │   ├── lda_model.joblib
│   │   ├── lda_dictionary.joblib
│   │   ├── lda_corpus.joblib
│   │   └── topic_label_map.joblib
│   ├── phase3/
│   │   ├── logistic_regression.joblib
│   │   ├── random_forest.joblib
│   │   ├── xgboost.joblib
│   │   ├── best_model.joblib          ← XGBoost (best performer)
│   │   ├── best_model_name.joblib
│   │   ├── label_encoder.joblib
│   │   ├── feature_names.joblib
│   │   ├── scaler.joblib
│   │   ├── X_test.joblib
│   │   ├── y_test.joblib
│   │   └── evaluation_results.joblib
│   └── phase4/
│       ├── shap_explainer.joblib
│       └── shap_values.joblib
│
└── outputs/
    ├── phase2/plots/              ← 9 NLP visualisation plots
    ├── phase3/plots/              ← Confusion matrices, ROC curves, feature importance
    └── phase4/plots/              ← SHAP global, beeswarm, waterfall, heatmap plots
```

---

## Setup Instructions

### Prerequisites

- Python 3.12
- Windows / macOS / Linux

### Step 1 — Create a virtual environment

```bash
py -3.12 -m venv .venv
```

### Step 2 — Activate the virtual environment

**Windows:**
```bash
.\.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download NLTK data (required for Phase 2)

```bash
python -m nltk.downloader punkt_tab stopwords wordnet omw-1.4
```

---

## Running the Pipeline

Phases must be run in order. Each phase depends on the outputs of the previous one.

### Phase 2 — NLP Pipeline

```bash
python run_phase2.py
```

**Input:** `data/raw/fashion_returns_dataset.csv`
**Output:** `data/processed/fashion_returns_dataset_nlp.csv` + NLP model artefacts + 9 plots

Runtime: approximately 3–5 minutes

---

### Phase 3 — Root Cause Classification

```bash
python run_phase3.py
```

**Input:** `data/processed/fashion_returns_dataset_nlp.csv`
**Output:** 3 trained models + evaluation metrics + confusion matrices + ROC curves

Runtime: approximately 2–3 minutes

---

### Phase 4 — SHAP Explainability

```bash
python run_phase4.py
```

**Input:** Phase 3 best model (XGBoost) + test set
**Output:** SHAP explainer + 15 dissertation plots

Runtime: approximately 1–2 minutes

---

## Key Results

### Phase 2 — NLP

| Metric | Value |
|--------|-------|
| Reviews analysed | 50,000 |
| Sentiment scores computed | 50,000 |
| Keywords extracted | 14,000 reviews × 8 keywords |
| LDA topics discovered | 6 |
| LDA coherence score (c_v) | 0.6283 |
| Complaint categories inferred | 11 |

### Phase 3 — Machine Learning

| Model | Accuracy | Macro F1 | Macro ROC-AUC |
|-------|----------|----------|---------------|
| Logistic Regression | 78.9% | 0.767 | 0.951 |
| Random Forest | 95.6% | 0.945 | 0.995 |
| **XGBoost** | **97.1%** | **0.964** | **0.999** |

**Target variable:** `root_cause_category` — 6 operational root cause classes

**Class imbalance handling:** SMOTE applied to training set only

### Phase 4 — SHAP

- Global feature importance across all 6 root cause classes
- Per-class beeswarm plots (6 plots)
- Per-class waterfall plots (6 plots)
- Feature × class importance heatmap

---

## Target Variable

The ML target variable is `root_cause_category` — the six operational root causes of confirmed product returns:

| Root Cause | Training Count | % of Returns |
|------------|---------------|--------------|
| Product Description Mismatch | 3,784 | 27.0% |
| Warehouse / Packaging | 2,840 | 20.3% |
| Customer Preference | 2,802 | 20.0% |
| Manufacturing / Quality Control | 2,284 | 16.3% |
| Supplier Issues | 1,307 | 9.3% |
| Logistics / Delivery | 983 | 7.0% |

---

## Dataset

| Property | Value |
|----------|-------|
| Total rows | 50,000 |
| Total columns | 50 |
| Returned orders (RICH tier) | 14,000 |
| Non-returned orders (LEAN tier) | 36,000 |
| Return rate | 28% |

---

## Dependencies

All dependencies are listed in `requirements.txt`. Key libraries:

| Library | Purpose |
|---------|---------|
| pandas, numpy | Data manipulation |
| scikit-learn | Logistic Regression, Random Forest, preprocessing |
| xgboost | Gradient boosted classification |
| imbalanced-learn | SMOTE oversampling |
| shap | Explainability |
| nltk, vaderSentiment, textblob | NLP and sentiment analysis |
| gensim | LDA topic modelling |
| matplotlib, seaborn, wordcloud | Visualisation |
| joblib | Model persistence |

---

## Notes

- All phases must be run from the `files/` directory
- Never run phases out of order — each phase depends on artefacts from the previous one
- The original dataset `data/raw/fashion_returns_dataset.csv` is never overwritten
- All model artefacts are saved to `models/` and reused in later phases
- All plots are saved to `outputs/` and are dissertation-ready at 150 DPI
