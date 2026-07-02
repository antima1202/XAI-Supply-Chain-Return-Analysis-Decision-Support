"""
visualisations.py  –  SHAP dissertation plots for Phase 4.

Plots generated
---------------
  1.  shap_global_importance.png     — mean |SHAP| bar chart (top 20 features)
  2.  shap_beeswarm_<class>.png      — beeswarm per root cause class (6 plots)
  3.  shap_beeswarm_summary.png      — combined beeswarm across all classes
  4.  shap_waterfall_<idx>.png       — waterfall for 6 representative predictions
  5.  shap_class_importance.png      — heatmap: feature × class importance
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
import seaborn as sns

from .utils import (
    PLOTS_DIR,
    ROOT_CAUSE_CLASSES,
    ROOT_CAUSE_SHORT,
    get_logger,
)

logger = get_logger("phase4.visualisations")

FIGURE_DPI   = 150
ACCENT       = "#2C5F8A"
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"font.family": "DejaVu Sans"})


def _save(fig, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved → %s", filename)


def _clean_feature_name(name: str) -> str:
    """Make one-hot encoded feature names more readable."""
    return name.replace("_", " ").replace("product subcategory", "subcat").strip()


# ---------------------------------------------------------------------------
# 1. Global feature importance — mean |SHAP| across all classes
# ---------------------------------------------------------------------------

def plot_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    """
    Bar chart of mean absolute SHAP values averaged across all classes.
    Shows which features matter most to the model overall.
    """
    # Mean |SHAP| per feature across samples and classes
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))  # shape: (n_features,)

    indices = np.argsort(mean_abs)[::-1][:top_n]
    top_names  = [_clean_feature_name(feature_names[i]) for i in indices]
    top_values = mean_abs[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(top_names))
    ax.barh(list(y_pos), top_values[::-1], color=ACCENT, edgecolor="white", height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_title(
        f"Global Feature Importance — Mean |SHAP Value| (Top {top_n})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Mean |SHAP Value| (averaged across all root cause classes)", fontsize=10)
    _save(fig, "shap_global_importance.png")


# ---------------------------------------------------------------------------
# 2. Beeswarm per class
# ---------------------------------------------------------------------------

def plot_beeswarm_per_class(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top_n: int = 15,
) -> None:
    """
    Beeswarm plot for each of the 6 root cause classes.
    Each dot = one prediction. Colour = feature value (red=high, blue=low).
    """
    feature_names = list(X_sample.columns)

    for class_idx, class_name in enumerate(ROOT_CAUSE_CLASSES):
        sv_class = shap_values[:, :, class_idx]  # (n_samples, n_features)

        # Select top_n features by mean |SHAP| for this class
        mean_abs = np.abs(sv_class).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:top_n]

        sv_top = sv_class[:, top_idx]
        X_top  = X_sample.iloc[:, top_idx].copy()
        X_top.columns = [_clean_feature_name(feature_names[i]) for i in top_idx]

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            sv_top,
            X_top,
            plot_type="dot",
            show=False,
            max_display=top_n,
            color_bar=True,
        )
        plt.title(
            f"SHAP Beeswarm — {class_name}",
            fontsize=12, fontweight="bold", pad=10,
        )
        safe = class_name.lower().replace(" ", "_").replace("/", "")
        _save(plt.gcf(), f"shap_beeswarm_{safe}.png")
        plt.close("all")


# ---------------------------------------------------------------------------
# 3. Summary beeswarm — all classes combined (predicted class SHAP values)
# ---------------------------------------------------------------------------

def plot_beeswarm_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    y_pred: np.ndarray,
    top_n: int = 20,
) -> None:
    """
    Beeswarm using each sample's predicted class SHAP values.
    Gives a single summary plot across the full model.
    """
    feature_names = list(X_sample.columns)
    n_samples = shap_values.shape[0]

    # For each sample use SHAP values of its predicted class
    sv_pred = np.array([
        shap_values[i, :, y_pred[i]]
        for i in range(n_samples)
    ])

    # Top features by mean |SHAP| across predicted-class values
    mean_abs = np.abs(sv_pred).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    sv_top = sv_pred[:, top_idx]
    X_top  = X_sample.iloc[:, top_idx].copy()
    X_top.columns = [_clean_feature_name(feature_names[i]) for i in top_idx]

    shap.summary_plot(sv_top, X_top, plot_type="dot", show=False,
                      max_display=top_n)
    plt.title("SHAP Beeswarm Summary — All Root Cause Classes (Predicted Class)",
              fontsize=12, fontweight="bold", pad=10)
    _save(plt.gcf(), "shap_beeswarm_summary.png")
    plt.close("all")


# ---------------------------------------------------------------------------
# 4. Waterfall plots — one per root cause class (representative example)
# ---------------------------------------------------------------------------

def plot_waterfall_per_class(
    explainer,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    y_test_sample: np.ndarray,
    top_n: int = 12,
) -> None:
    """
    Waterfall plot for one representative correctly-predicted example
    per root cause class. Shows feature-by-feature contribution to
    the predicted class probability.
    """
    feature_names = [_clean_feature_name(c) for c in X_sample.columns]

    for class_idx, class_name in enumerate(ROOT_CAUSE_CLASSES):
        # Find samples where true label == class and pick first one
        mask = y_test_sample == class_idx
        if mask.sum() == 0:
            logger.warning("No test samples for class %s — skipping waterfall", class_name)
            continue

        sample_idx = np.where(mask)[0][0]
        sv = shap_values[sample_idx, :, class_idx]

        # Build SHAP Explanation object
        base_value = (
            explainer.expected_value[class_idx]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )

        explanation = shap.Explanation(
            values=sv,
            base_values=float(base_value),
            data=X_sample.iloc[sample_idx].values,
            feature_names=feature_names,
        )

        # Select top_n features by |SHAP|
        top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
        explanation_top = shap.Explanation(
            values=sv[top_idx],
            base_values=float(base_value),
            data=X_sample.iloc[sample_idx].values[top_idx],
            feature_names=[feature_names[i] for i in top_idx],
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(explanation_top, show=False, max_display=top_n)
        plt.title(
            f"SHAP Waterfall — {class_name}  (sample #{sample_idx})",
            fontsize=11, fontweight="bold", pad=10,
        )
        safe = class_name.lower().replace(" ", "_").replace("/", "")
        _save(plt.gcf(), f"shap_waterfall_{safe}.png")
        plt.close("all")


# ---------------------------------------------------------------------------
# 5. Class-level feature importance heatmap
# ---------------------------------------------------------------------------

def plot_class_importance_heatmap(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
) -> None:
    """
    Heatmap showing mean |SHAP| per feature per root cause class.
    Rows = features, Columns = root cause classes.
    Reveals which features are most diagnostic for each class.
    """
    # mean |SHAP| per (feature, class)
    mean_abs = np.abs(shap_values).mean(axis=0)  # (n_features, n_classes)

    # Select top_n features by overall importance
    overall = mean_abs.mean(axis=1)
    top_idx = np.argsort(overall)[::-1][:top_n]

    heatmap_data = pd.DataFrame(
        mean_abs[top_idx, :],
        index=[_clean_feature_name(feature_names[i]) for i in top_idx],
        columns=[ROOT_CAUSE_SHORT[c] for c in ROOT_CAUSE_CLASSES],
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        heatmap_data,
        cmap="Blues",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean |SHAP Value|"},
    )
    ax.set_title(
        f"Feature × Root Cause Class Importance — Top {top_n} Features",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Root Cause Class", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    _save(fig, "shap_class_importance_heatmap.png")


# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------

def generate_all_shap_plots(
    explainer,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    y_test_sample: np.ndarray,
    y_pred_sample: np.ndarray,
    feature_names: list[str],
) -> None:
    """Generate and save all Phase 4 SHAP visualisations."""
    logger.info("Generating Phase 4 SHAP visualisations …")

    plot_global_importance(shap_values, feature_names)
    plot_beeswarm_per_class(shap_values, X_sample)
    plot_beeswarm_summary(shap_values, X_sample, y_pred_sample)
    plot_waterfall_per_class(explainer, shap_values, X_sample, y_test_sample)
    plot_class_importance_heatmap(shap_values, feature_names)

    logger.info("All SHAP plots saved to %s", PLOTS_DIR)
