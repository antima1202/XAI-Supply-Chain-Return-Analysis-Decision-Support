"""
models.py  –  Train and persist the three Phase 3 classifiers.

Models
------
  1. Logistic Regression  — statistical baseline, scaled features
  2. Random Forest        — ensemble baseline, unscaled features
  3. XGBoost              — primary model, unscaled features, handles imbalance

All models are trained on the SMOTE-balanced training set.
Hyperparameters are set to sensible dissertation-appropriate values
with brief justification comments.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .utils import (
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    ROOT_CAUSE_CLASSES,
    get_logger,
    save_artefact,
    timer,
)

logger = get_logger("phase3.models")

N_CLASSES = len(ROOT_CAUSE_CLASSES)


# ---------------------------------------------------------------------------
# Model 1: Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Train a multinomial Logistic Regression classifier.

    Uses:
      solver='lbfgs'         — efficient for multiclass, no regularisation needed
      multi_class='multinomial' — true softmax, not one-vs-rest
      max_iter=1000          — ensures convergence on this feature size
      C=1.0                  — default L2 regularisation
      class_weight='balanced' — secondary guard against imbalance on top of SMOTE
    """
    logger.info("Training Logistic Regression …")
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    with timer("Logistic Regression training"):
        model.fit(X_train, y_train)

    save_artefact(model, LR_MODEL_PATH)
    logger.info("Logistic Regression trained and saved")
    return model


# ---------------------------------------------------------------------------
# Model 2: Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Uses:
      n_estimators=300       — 300 trees: good bias-variance tradeoff
      max_depth=20           — prevents overfitting on this dataset size
      min_samples_split=5    — minimum samples to split a node
      min_samples_leaf=2     — minimum samples per leaf
      class_weight='balanced_subsample' — per-tree balancing
      n_jobs=-1              — parallel training on all cores
    """
    logger.info("Training Random Forest …")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    with timer("Random Forest training"):
        model.fit(X_train, y_train)

    save_artefact(model, RF_MODEL_PATH)
    logger.info("Random Forest trained and saved")
    return model


# ---------------------------------------------------------------------------
# Model 3: XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """
    Train an XGBoost gradient-boosted classifier.

    Uses:
      n_estimators=500       — 500 boosting rounds
      max_depth=6            — standard XGBoost depth; prevents overfitting
      learning_rate=0.05     — lower LR + more trees = better generalisation
      subsample=0.8          — row subsampling per tree (reduces overfitting)
      colsample_bytree=0.8   — feature subsampling per tree
      use_label_encoder=False — suppress deprecation warning
      eval_metric='mlogloss' — multiclass log loss
      objective='multi:softmax' — multiclass classification
      num_class=6            — 6 root cause categories
      tree_method='hist'     — fast histogram-based algorithm
    """
    logger.info("Training XGBoost …")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    with timer("XGBoost training"):
        model.fit(
            X_train, y_train,
            eval_set=None,
            verbose=False,
        )

    save_artefact(model, XGB_MODEL_PATH)
    logger.info("XGBoost trained and saved")
    return model


# ---------------------------------------------------------------------------
# Train all three
# ---------------------------------------------------------------------------

def train_all_models(data: dict) -> dict:
    """
    Train all three models and return them in a dict.

    Parameters
    ----------
    data : dict  — output of feature_engineering.prepare_data()

    Returns
    -------
    dict with keys: 'logistic_regression', 'random_forest', 'xgboost'
    """
    models = {}

    models["Logistic Regression"] = train_logistic_regression(
        data["X_train_scaled"], data["y_train"]
    )
    models["Random Forest"] = train_random_forest(
        data["X_train"], data["y_train"]
    )
    models["XGBoost"] = train_xgboost(
        data["X_train"], data["y_train"]
    )

    logger.info("All three models trained successfully")
    return models
