# An Explainable AI Framework Integrating Customer Feedback and Transactional Data for Product Return Prediction in Supply Chains

**MSc Dissertation Project — MSc Applied Artificial Intelligence**  
**WMG, University of Warwick**

---

## Project Overview

This project presents an explainable AI framework for analysing confirmed product returns in a fashion e-commerce supply chain.

The framework operates **post-return**. Rather than predicting whether a customer will return an item, it analyses confirmed returns to classify their likely operational root causes, explain the model's predictions, prioritise causes according to operational risk, and translate the results into stakeholder-specific recommendations.

The framework integrates:

- Transactional data
- Customer review text
- Natural Language Processing (NLP)
- Supervised machine learning
- SHAP-based explainability
- Root-cause analysis
- Risk-weighted prioritisation
- Local Large Language Model (LLM) recommendations
- Scenario-based intervention analysis
- Interactive stakeholder decision support

The final implementation was evaluated using a **50,000-record synthetic dataset** statistically anchored to real e-commerce transaction data and linguistically calibrated using a real Amazon Fashion review corpus. The secondary datasets were used as reference sources rather than being directly combined into the final synthetic dataset.

---

## Dissertation Framework

The final framework consists of the following stages:

| Stage | Component | Status |
|---|---|---|
| 1 | Data Sourcing and Synthetic Dataset Development | Complete |
| 2 | NLP Customer Feedback Analysis | Complete |
| 3 | Feature Engineering | Complete |
| 4 | Root Cause Classification | Complete |
| 5 | Explainable AI (SHAP) | Complete |
| 6 | Root Cause Analysis | Complete |
| 7 | Risk-Weighted Prioritisation | Complete |
| 8 | LLM Recommendation Engine | Complete |
| 9 | Stakeholder Decision Support Dashboard | Complete |
| 10 | Continuous Feedback Loop | Conceptual / Future Work |

The continuous feedback loop is presented as a future extension rather than an implemented component.

> **Development note:** An earlier version of the project included a Linear Programming optimisation stage. This was implemented during development but removed from the final framework after evaluation indicated that the LLM-based recommendation approach provided sufficiently actionable, cost-aware guidance with lower implementation complexity.

---

## Repository Structure

```text
files/
│
├── requirements.txt
├── dashboard.py
│
├── run_phase2.py
├── run_phase3.py
├── run_phase4.py
├── run_phase5.py
├── run_phase6.py
└── run_phase7.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── phase2/
│   │   ├── preprocessing.py
│   │   ├── sentiment_analysis.py
│   │   ├── keyword_extraction.py
│   │   ├── topic_modelling.py
│   │   ├── complaint_classifier.py
│   │   ├── visualisations.py
│   │   └── pipeline.py
│   │
│   ├── phase3/
│   │   ├── feature_engineering.py
│   │   ├── models.py
│   │   ├── evaluation.py
│   │   └── pipeline.py
│   │
│   ├── phase4/
│   │   ├── explainer.py
│   │   ├── visualisations.py
│   │   └── pipeline.py
│   │
│   ├── phase5/
│   │   ├── analyser.py
│   │   ├── visualisations.py
│   │   └── pipeline.py
│   │
│   ├── phase6/
│   │   ├── risk_scorer.py
│   │   ├── visualisations.py
│   │   └── pipeline.py
│   │
│   └── phase7/
│       ├── llm_client.py
│       ├── prompt_builder.py
│       ├── recommendation_engine.py
│       ├── visualisations.py
│       └── pipeline.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/
│   └── phase7/
│
└── outputs/
    ├── phase2/
    ├── phase3/
    ├── phase4/
    ├── phase5/
    ├── phase6/
    └── phase7/

## Installation

### Prerequisites

- Python **3.12**
- Git
- Ollama (for the local LLM recommendation engine)

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>/files
```

### 2. Create a virtual environment

**Windows**

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download required NLP resources

```bash
python -m nltk.downloader punkt_tab stopwords wordnet omw-1.4 vader_lexicon
```

### 5. Install Ollama and download the local model

Install Ollama, then pull the model used in the dissertation:

```bash
ollama pull llama3.2
```

Verify Ollama is running before executing Phase 7.

---

## Running the Framework

Run each phase sequentially from the `files/` directory.

### Phase 2 — NLP Customer Feedback Analysis

Generates sentiment scores, complaint categories, keywords, and LDA topics.

```bash
python run_phase2.py
```

Outputs are saved to `data/processed/phase2/`.

---

### Phase 3 — Feature Engineering and Root Cause Classification

Creates engineered features, applies SMOTE to the training data, trains Logistic Regression, Random Forest, and XGBoost models, and evaluates model performance.

```bash
python run_phase3.py
```

Outputs include trained models, evaluation metrics, and processed feature datasets.

---

### Phase 4 — Explainable AI (SHAP)

Generates SHAP global feature importance, class-level explanations, waterfall plots, and dependence plots.

```bash
python run_phase4.py
```

Outputs are saved to `outputs/phase4/`.

---

### Phase 5 — Root Cause Analysis

Aggregates classified returns into operational insights, temporal trends, regional analysis, and business visualisations.

```bash
python run_phase5.py
```

Outputs are saved to `outputs/phase5/`.

---

### Phase 6 — Risk-Weighted Prioritisation

Calculates composite operational risk scores and assigns priority levels (P1–P6).

```bash
python run_phase6.py
```

Outputs include the risk register and prioritisation tables.

---

### Phase 7 — LLM Recommendation Engine

Uses the local Llama 3.2 model through Ollama to generate operational recommendations for each prioritised root cause.

```bash
python run_phase7.py
```

Outputs are saved to `outputs/phase7/`.

---

## Launching the Dashboard

Once all phases have completed successfully, launch the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser and includes:

- Overview
- Model Performance
- SHAP Explainability
- Root Cause Analysis
- Risk Assessment
- Scenario Simulator
- LLM Recommendations
- Stakeholder Views

## Expected Runtime

Running the complete analytical pipeline (Phases 2–7) on the dissertation dataset takes approximately **10–15 minutes** on a standard laptop with Python 3.12 installed. The dashboard launches after the pipeline has generated the required model artefacts and visualisation outputs.