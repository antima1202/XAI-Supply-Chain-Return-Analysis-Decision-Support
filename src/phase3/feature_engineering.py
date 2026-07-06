"""
feature_engineering.py  –  Feature preparation for Phase 3 ML models.

Steps
-----
1. Load the Phase 2 enriched CSV (RICH rows only — 14,000 confirmed returns)
2. Drop leakage columns, identifiers, and zero-variance features
3. Encode categorical features (one-hot for low-cardinality, label for ordered)
4. Encode NLP categorical features (complaint_category, dominant_topic)
5. Encode target variable (root_cause_category → integer labels)
6. Stratified 80/20 train/test split
7. Apply SMOTE to training set only to handle class imbalance
8. Scale numerical features for Logistic Regression (saved separately)

Feature groups
--------------
  Transactional  : price, discount, delivery_days, return_lag_days, etc.
  Customer       : customer_age, customer_return_rate, customer_order_count, etc.
  Product        : product_subcategory, brand, material, fit_type, colour, etc.
  NLP            : sentiment_score, subjectivity_score, complaint_category,
                   dominant_topic, review_rating, review_word_count

Leakage columns explicitly excluded
------------------------------------
  return_reason       — directly maps 1:1 to root_cause_category
  return_request_date — date of return, not available at prediction time
  size_run            — zero variance (all "True to size")
  profit_margin       — internal metric, not available operationally
  total_amount        — collinear with price_after_discount × quantity
  extracted_keywords  — raw string, already represented by complaint_category
  review_text         — raw text, already represented by NLP features
  review_title        — raw text
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .utils import (
    INPUT_CSV,
    LABEL_ENCODER_PATH,
    FEATURE_NAMES_PATH,
    SCALER_PATH,
    ROOT_CAUSE_CLASSES,
    get_logger,
    save_artefact,
    load_artefact,
)

logger = get_logger("phase3.features")

# ---------------------------------------------------------------------------
# Columns to drop (leakage, identifiers, raw text, zero-variance)
# ---------------------------------------------------------------------------

DROP_COLS = [
    # Identifiers
    "row_id", "order_id", "customer_id", "product_id",
    # Dates (engineered features already exist)
    "order_date", "delivered_date", "return_request_date",
    # Target and LEAN-only flags
    "is_returned",
    # DATA LEAKAGE — encodes target directly
    "return_reason",
    # Zero variance
    "size_run",
    # Collinear / internal
    "profit_margin", "total_amount", "row_source",
    # Raw text (NLP outputs already extracted)
    "review_text", "review_title", "extracted_keywords",
]


# ---------------------------------------------------------------------------
# Step 1: Load and filter RICH rows
# ---------------------------------------------------------------------------

def load_rich_data(path=INPUT_CSV) -> pd.DataFrame:
    """Load the NLP-enriched CSV and return RICH rows only."""
    df = pd.read_csv(path, low_memory=False)
    rich = df[df["is_returned"] == 1].copy()
    logger.info("Loaded %d RICH rows for modelling", len(rich))
    return rich


# ---------------------------------------------------------------------------
# Step 2: Build feature matrix and target
# ---------------------------------------------------------------------------

def build_features(rich: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop excluded columns, encode categoricals, return X and y.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (all numeric after encoding)
    y : pd.Series     — integer-encoded target (root_cause_category)
    """
    df = rich.copy()

    # --- Encode target ---
    le = LabelEncoder()
    le.classes_ = np.array(ROOT_CAUSE_CLASSES)
    y = pd.Series(
        le.transform(df["root_cause_category"]),
        index=df.index,
        name="root_cause_category",
    )
    save_artefact(le, LABEL_ENCODER_PATH)
    logger.info("Target encoded: %s", dict(enumerate(le.classes_)))

    # --- Drop excluded columns and target ---
    drop = [c for c in DROP_COLS if c in df.columns] + ["root_cause_category"]
    df = df.drop(columns=drop)

    # --- One-hot encode low-cardinality categoricals ---
    ohe_cols = [
        "payment_method", "region", "season", "product_subcategory",
        "brand", "colour", "fit_type", "material",
        "customer_gender", "size",
    ]
    ohe_cols = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)

    # --- Label-encode NLP categoricals (ordered by signal strength) ---
    # complaint_category: 11 values
    if "complaint_category" in df.columns:
        cc_le = LabelEncoder()
        df["complaint_category"] = cc_le.fit_transform(df["complaint_category"].astype(str))

    # dominant_topic: 6 values
    if "dominant_topic" in df.columns:
        dt_le = LabelEncoder()
        df["dominant_topic"] = dt_le.fit_transform(df["dominant_topic"].astype(str))

    # --- Ensure all remaining columns are numeric ---
    df = df.select_dtypes(include=[np.number])

    logger.info("Feature matrix: %d rows × %d features", *df.shape)
    save_artefact(list(df.columns), FEATURE_NAMES_PATH)

    return df, y


# ---------------------------------------------------------------------------
# Step 3: Train / test split (stratified)
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Stratified 80/20 split.  Stratification preserves class proportions
    in both train and test sets — important given class imbalance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        "Split: train=%d  test=%d  (stratified)",
        len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 4: SMOTE oversampling (training set only)
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to the training set to balance class distribution.

    SMOTE synthesises new samples for minority classes by interpolating
    between existing samples — it does NOT duplicate rows.

    Applied to training data ONLY.  The test set is never resampled
    to ensure evaluation reflects the real-world class distribution.
    """
    logger.info("Class distribution before SMOTE:")
    for cls, cnt in sorted(y_train.value_counts().items()):
        logger.info("  Class %d: %d samples", cls, cnt)

    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info("Class distribution after SMOTE:")
    for cls, cnt in sorted(pd.Series(y_res).value_counts().items()):
        logger.info("  Class %d: %d samples", cls, cnt)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name="root_cause_category")
    return X_res, y_res


# ---------------------------------------------------------------------------
# Step 5: Scaling (for Logistic Regression only)
# ---------------------------------------------------------------------------

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit StandardScaler on training data and transform both sets.
    Tree-based models (RF, XGBoost) do not use this — only LR.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
    )
    save_artefact(scaler, SCALER_PATH)
    logger.info("Features scaled (StandardScaler) for Logistic Regression")
    return X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# Master feature engineering function
# ---------------------------------------------------------------------------

def prepare_data() -> dict:
    """
    Run the full feature engineering pipeline.

    Returns a dict with keys:
      X_train, X_test, y_train, y_test          — for RF and XGBoost
      X_train_scaled, X_test_scaled             — for Logistic Regression
      X_train_smote, y_train_smote              — SMOTE-balanced (unscaled)
      X_train_smote_scaled, y_train_smote       — SMOTE-balanced + scaled
      feature_names                             — list of feature names
      label_encoder                             — fitted LabelEncoder
    """
    rich = load_rich_data()
    X, y = build_features(rich)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    X_train_scaled, X_test_scaled = scale_features(X_train_smote, X_test)

    return {
        "X_train":              X_train_smote,
        "X_test":               X_test,
        "y_train":              y_train_smote,
        "y_test":               y_test,
        "X_train_scaled":       X_train_scaled,
        "X_test_scaled":        X_test_scaled,
        "feature_names":        list(X.columns),
        "label_encoder":        load_artefact(LABEL_ENCODER_PATH),
    }
