"""
evaluation.py  –  Model evaluation for the Phase 3 classifiers.

Produces per model
------------------
  1. Classification report (precision, recall, F1 per class + weighted avg)
  2. Confusion matrix heatmap
  3. ROC curves (one-vs-rest, per class)
  4. Feature importance plot (RF native + XGBoost native)
  5. Model comparison summary table

All plots saved to outputs/phase3/plots/.
All numeric results saved as a dict for Phase 4 and dissertation reporting.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .utils import (
    PLOTS_DIR,
    ROOT_CAUSE_CLASSES,
    ROOT_CAUSE_SHORT,
    EVAL_RESULTS_PATH,
    get_logger,
    save_artefact,
)

logger = get_logger("phase3.evaluation")

FIGURE_DPI   = 150
ACCENT       = "#2C5F8A"
PALETTE      = sns.color_palette("tab10", n_colors=6)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"font.family": "DejaVu Sans"})

SHORT_LABELS = [ROOT_CAUSE_SHORT[c] for c in ROOT_CAUSE_CLASSES]


def _save(fig, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved → %s", filename)


# ---------------------------------------------------------------------------
# 1. Classification report
# ---------------------------------------------------------------------------

def compute_classification_report(
    y_test, y_pred, model_name: str
) -> dict:
    """Compute and log a full classification report."""
    report = classification_report(
        y_test, y_pred,
        target_names=ROOT_CAUSE_CLASSES,
        output_dict=True,
    )
    report_str = classification_report(
        y_test, y_pred,
        target_names=ROOT_CAUSE_CLASSES,
    )
    logger.info("\n%s Classification Report:\n%s", model_name, report_str)
    return report


# ---------------------------------------------------------------------------
# 2. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_test, y_pred, model_name: str
) -> None:
    """Normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=cm,           # show raw counts
        fmt="d",
        cmap="Blues",
        xticklabels=SHORT_LABELS,
        yticklabels=SHORT_LABELS,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Proportion (row-normalised)"},
    )
    ax.set_title(
        f"Confusion Matrix — {model_name}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Predicted Root Cause", fontsize=11)
    ax.set_ylabel("True Root Cause", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    safe_name = model_name.lower().replace(" ", "_")
    _save(fig, f"confusion_matrix_{safe_name}.png")


# ---------------------------------------------------------------------------
# 3. ROC curves (one-vs-rest)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_test, y_prob, model_name: str
) -> dict:
    """
    Plot ROC curves for each class (one-vs-rest) and compute AUC scores.

    Returns dict of {class_name: auc_score}.
    """
    y_bin = label_binarize(y_test, classes=list(range(len(ROOT_CAUSE_CLASSES))))
    auc_scores = {}

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (cls_name, colour) in enumerate(zip(ROOT_CAUSE_CLASSES, PALETTE)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        auc_scores[cls_name] = round(auc, 4)
        ax.plot(fpr, tpr, color=colour, linewidth=2,
                label=f"{ROOT_CAUSE_SHORT[cls_name]} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    ax.set_title(f"ROC Curves (One-vs-Rest) — {model_name}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    safe_name = model_name.lower().replace(" ", "_")
    _save(fig, f"roc_curves_{safe_name}.png")

    macro_auc = np.mean(list(auc_scores.values()))
    logger.info("%s — Macro ROC-AUC: %.4f", model_name, macro_auc)
    return auc_scores


# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str,
    top_n: int = 25,
) -> None:
    """
    Plot top-N feature importances for RF (native) and XGBoost (native).
    Skips Logistic Regression (handled separately via coefficients).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.info("Skipping feature importance plot for %s", model_name)
        return

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(top_features))
    ax.barh(list(y_pos), top_values[::-1], color=ACCENT, edgecolor="white", height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=11)

    safe_name = model_name.lower().replace(" ", "_")
    _save(fig, f"feature_importance_{safe_name}.png")


# ---------------------------------------------------------------------------
# 5. Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(results: dict) -> None:
    """
    Horizontal bar chart comparing accuracy and macro F1 across models.
    """
    model_names = list(results.keys())
    accuracies  = [results[m]["accuracy"] for m in model_names]
    macro_f1s   = [results[m]["macro_f1"] for m in model_names]
    macro_aucs  = [results[m]["macro_auc"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, accuracies, width, label="Accuracy",
                   color="#2C5F8A", edgecolor="white")
    bars2 = ax.bar(x,          macro_f1s,  width, label="Macro F1",
                   color="#E8563A", edgecolor="white")
    bars3 = ax.bar(x + width,  macro_aucs, width, label="Macro ROC-AUC",
                   color="#4CAF50", edgecolor="white")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Accuracy, Macro F1, Macro ROC-AUC",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    _save(fig, "model_comparison.png")


# ---------------------------------------------------------------------------
# Master evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    feature_names: list[str],
    use_scaled: bool = False,
) -> dict:
    """
    Run full evaluation for a single model.

    Returns a dict with accuracy, macro_f1, macro_auc, per-class metrics.
    """
    y_pred = model.predict(X_test)

    # Probability estimates (needed for ROC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    # Classification report
    report = compute_classification_report(y_test, y_pred, model_name)

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)

    # ROC curves
    auc_scores = {}
    if y_prob is not None:
        auc_scores = plot_roc_curves(y_test, y_prob, model_name)

    # Feature importance
    plot_feature_importance(model, feature_names, model_name)

    # Compile results
    accuracy  = report["accuracy"]
    macro_f1  = report["macro avg"]["f1-score"]
    macro_auc = np.mean(list(auc_scores.values())) if auc_scores else None

    result = {
        "accuracy":    round(accuracy, 4),
        "macro_f1":    round(macro_f1, 4),
        "macro_auc":   round(macro_auc, 4) if macro_auc else None,
        "auc_scores":  auc_scores,
        "report":      report,
    }

    logger.info(
        "%s — Accuracy: %.4f  Macro-F1: %.4f  Macro-AUC: %s",
        model_name, accuracy, macro_f1,
        f"{macro_auc:.4f}" if macro_auc else "N/A",
    )
    return result


def evaluate_all_models(models: dict, data: dict) -> dict:
    """
    Evaluate all three models and save a comparison plot.

    Returns dict of {model_name: evaluation_result}.
    """
    feature_names = data["feature_names"]
    results = {}

    results["Logistic Regression"] = evaluate_model(
        models["Logistic Regression"],
        data["X_test_scaled"],
        data["y_test"],
        "Logistic Regression",
        feature_names,
    )
    results["Random Forest"] = evaluate_model(
        models["Random Forest"],
        data["X_test"],
        data["y_test"],
        "Random Forest",
        feature_names,
    )
    results["XGBoost"] = evaluate_model(
        models["XGBoost"],
        data["X_test"],
        data["y_test"],
        "XGBoost",
        feature_names,
    )

    plot_model_comparison(results)
    save_artefact(results, EVAL_RESULTS_PATH)

    # Identify best model by macro F1
    best_name = max(results, key=lambda m: results[m]["macro_f1"])
    logger.info("Best model by Macro-F1: %s (%.4f)",
                best_name, results[best_name]["macro_f1"])

    return results, best_name
