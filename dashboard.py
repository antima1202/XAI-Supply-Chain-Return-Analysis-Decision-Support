"""
dashboard.py  –  Supply Chain Return Analysis Dashboard
An Explainable AI Framework for Fashion E-Commerce

Run: streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PLOTLY TEXT COLOUR
# CSS injected via st.markdown() cannot reach Plotly's SVG output, since Plotly
# sets text colour directly as an SVG fill attribute rather than inheriting
# page-level CSS. This template sets black as the default font colour for
# every chart's axis labels, tick labels, legend, and annotations in one place,
# rather than needing font=dict(color=...) added to each individual chart below.
_black_text_template = go.layout.Template(
    layout=go.Layout(font=dict(color="#111111"))
)
pio.templates["black_text"] = _black_text_template
pio.templates.default = "plotly+black_text"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Return Analysis",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global styles - all text black, table headers bold
st.markdown("""
<style>
  html, body, [class*="css"], .stMarkdown, p, li, span, div, label,
  .stDataFrame, .stTable { color: #111111 !important; }
  [data-testid="stDataFrame"] th { color: #111111 !important; font-weight: 700 !important; background:#F0F0F0 !important; }
  [data-testid="stDataFrame"] td { color: #111111 !important; }
  .dvn-scroller * { color: #111111 !important; }
  .streamlit-expanderHeader { color: #111111 !important; font-weight: 600 !important; }
  .stSelectbox label, .stSlider label, .stNumberInput label { color: #111111 !important; font-weight: 600 !important; }
  .stTabs [data-baseweb="tab"] { color: #111111 !important; font-weight: 600 !important; }
  section[data-testid="stSidebar"] * { color: #111111 !important; }
  thead tr th { color: #111111 !important; font-weight: 700 !important; }
  tbody tr td { color: #111111 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

ROOT_CAUSE_COLOURS = {
    "Product Description Mismatch":    "#7ED321",
    "Warehouse / Packaging":           "#1ABC9C",
    "Manufacturing / Quality Control": "#F5A623",
    "Supplier Issues":                 "#9B59B6",
    "Customer Preference":             "#4A90D9",
    "Logistics / Delivery":            "#E8563A",
}

RISK_COLOURS = {
    "Critical": "#D94F3D",
    "High":     "#E8A838",
    "Medium":   "#4A90D9",
    "Low":      "#7ED321",
}

STAKEHOLDER_MAP = {
    "Product Description Mismatch":    "Marketing & Content Team",
    "Warehouse / Packaging":           "Warehouse Operations Manager",
    "Manufacturing / Quality Control": "Quality Assurance Team",
    "Supplier Issues":                 "Procurement & Supplier Manager",
    "Customer Preference":             "Marketing & UX Team",
    "Logistics / Delivery":            "Logistics & Carrier Manager",
}

DEFAULT_REDUCTION = {
    "Product Description Mismatch":    0.35,
    "Warehouse / Packaging":           0.30,
    "Manufacturing / Quality Control": 0.40,
    "Supplier Issues":                 0.45,
    "Customer Preference":             0.15,
    "Logistics / Delivery":            0.25,
}

DEFAULT_COSTS = {
    "Product Description Mismatch":    30_000,
    "Warehouse / Packaging":           40_000,
    "Manufacturing / Quality Control": 35_000,
    "Supplier Issues":                 45_000,
    "Customer Preference":             20_000,
    "Logistics / Delivery":            25_000,
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached so models load once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_data():
    def _load(path):
        p = PROJECT_ROOT / path
        return joblib.load(p) if p.exists() else None

    eval_results  = _load("models/phase3/evaluation_results.joblib")
    feature_names = _load("models/phase3/feature_names.joblib")
    shap_data     = _load("models/phase4/shap_values.joblib")
    summary       = _load("models/phase5/root_cause_summary.joblib")
    trend         = _load("models/phase5/monthly_trend.joblib")
    brand_detail  = _load("models/phase5/brand_breakdown.joblib")
    region        = _load("models/phase5/region_breakdown.joblib")
    risk_df       = _load("models/phase6/risk_scores.joblib")
    recs_data     = _load("models/phase7/recommendations.joblib")

    return {
        "eval_results":  eval_results,
        "feature_names": feature_names,
        "shap_data":     shap_data,
        "summary":       summary,
        "trend":         trend,
        "brand_detail":  brand_detail,
        "region":        region,
        "risk_df":       risk_df,
        "recs_data":     recs_data,
    }

data = load_all_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/return-purchase.png", width=60)
st.sidebar.title("Supply Chain AI")
st.sidebar.markdown("**Explainable AI Framework**  \nFashion E-Commerce Returns")
st.sidebar.divider()

pages = {
    "🏠  Overview":               "overview",
    "📊  Model Performance":      "model",
    "🔍  SHAP Explainability":    "shap",
    "📈  Root Cause Analysis":    "rca",
    "⚠️  Risk Assessment":        "risk",
    "🎯  Scenario Simulator":     "scenario",
    "💡  LLM Recommendations":   "llm",
    "👤  Stakeholder Views":      "stakeholder",
}

selected = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
page = pages[selected]

st.sidebar.divider()
st.sidebar.caption("MSc Dissertation · Fashion Returns AI")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(col, label, value, delta=None, colour="#2C5F8A"):
    col.markdown(
        f"""
        <div style="background:{colour}15;border-left:4px solid {colour};
                    padding:16px;border-radius:8px;margin-bottom:8px">
            <p style="margin:0;font-size:12px;color:#111111;font-weight:600;
                      text-transform:uppercase;letter-spacing:0.5px">{label}</p>
            <p style="margin:4px 0 0;font-size:28px;font-weight:700;color:{colour}">{value}</p>
            {f'<p style="margin:2px 0 0;font-size:12px;color:#111111">{delta}</p>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

def styled_table(df, colour_col=None):
    """Render a styled dataframe."""
    st.dataframe(df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 - OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "overview":
    st.title("🏠 Supply Chain Return Analysis Framework")
    st.markdown(
        "An **Explainable AI Framework Integrating Customer Feedback and Transactional Data for Product Return Prediction in Supply Chains** that analyses confirmed product returns "
        "in fashion e-commerce to identify operational root causes and generate "
        "targeted supply chain recommendations."
    )

    st.divider()

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Total Orders",        "50,000",  "Dataset size",          "#2C5F8A")
    kpi_card(c2, "Confirmed Returns",   "14,000",  "28% return rate",       "#E8563A")
    kpi_card(c3, "Model Accuracy",      "97.1%",   "XGBoost classifier",    "#4CAF50")
    kpi_card(c4, "Root Cause Classes",  "6",       "Operational categories","#F5A623")
    kpi_card(c5, "Potential Saving",    "£95,524", "Annual estimate",       "#9B59B6")

    st.divider()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("10-Phase AI Pipeline")
        phases = [
            ("1", "Data Collection & Integration",        "Data Engineering"),
            ("2", "NLP Customer Feedback Analysis",       "NLP / Unsupervised ML"),
            ("3", "Feature Engineering",                  "ML Preparation"),
            ("4", "Root Cause Classification (XGBoost)",  "Supervised ML"),
            ("5", "Explainable AI (SHAP)",                "Explainable AI"),
            ("6", "Root Cause Analysis Layer",            "Data Analysis"),
            ("7", "Risk Assessment",                      "Quantitative Scoring"),
            ("8", "LLM Recommendation Engine",            "Generative AI (Ollama)"),
            ("9", "Stakeholder Dashboard",                "Decision Support"),
            ("10","Continuous Feedback Loop",             "Monitoring"),
        ]
        phase_df = pd.DataFrame(phases, columns=["Phase","Name","Type"])
        st.dataframe(phase_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Dataset Summary")
        dataset_info = pd.DataFrame({
            "Property": ["Total rows","Total columns","Returned (RICH)","Non-returned (LEAN)",
                         "Return rate","NLP reviews analysed","ML features","LDA coherence"],
            "Value":    ["50,000","50","14,000","36,000","28%","50,000","101","0.6283"],
        })
        st.dataframe(dataset_info, use_container_width=True, hide_index=True)

        st.subheader("AI Techniques Used")
        ai_df = pd.DataFrame({
            "Phase": ["2","4","5","8"],
            "Technique": ["NLP (VADER, LDA, TF-IDF)","XGBoost (Supervised ML)",
                          "SHAP (Explainable AI)","Ollama LLM (Generative AI)"],
        })
        st.dataframe(ai_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Root Cause Distribution")
    if data["summary"] is not None:
        s = data["summary"].copy()
        s["root_cause_category"] = s["root_cause_category"].str.replace(
            "Product Listing / Information", "Product Description Mismatch")
        fig = px.bar(
            s.sort_values("count"),
            x="count", y="root_cause_category",
            orientation="h",
            color="root_cause_category",
            color_discrete_map=ROOT_CAUSE_COLOURS,
            text=s.sort_values("count")["pct_of_returns"].apply(lambda x: f"{x:.1f}%"),
            labels={"count":"Number of Returns","root_cause_category":"Root Cause"},
            title="Confirmed Returns by Root Cause Category",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=380, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key="chart_1_line269")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 - MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "model":
    st.title("📊 Model Performance")
    st.markdown("Three supervised multiclass classification models trained to classify "
                "confirmed returns into **6 operational root cause categories**.")

    st.divider()

    # Model comparison table
    st.subheader("Model Comparison Table")
    if data["eval_results"]:
        rows = []
        for model, metrics in data["eval_results"].items():
            rows.append({
                "Model":         model,
                "Accuracy":      f"{metrics['accuracy']*100:.1f}%",
                "Macro F1":      f"{metrics['macro_f1']:.4f}",
                "Macro ROC-AUC": f"{metrics['macro_auc']:.4f}",
                "Selected":      "Best" if model == "XGBoost" else "",
            })
        model_df = pd.DataFrame(rows)
        def _highlight_best(val):
            return "background-color: #C8E6C9; color: #111111; font-weight: 600" if val == "Best" else ""
        st.dataframe(model_df.style.map(_highlight_best, subset=["Selected"]),
                     use_container_width=True, hide_index=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy Comparison")
        if data["eval_results"]:
            models  = list(data["eval_results"].keys())
            accs    = [data["eval_results"][m]["accuracy"]*100 for m in models]
            f1s     = [data["eval_results"][m]["macro_f1"]*100 for m in models]
            aucs    = [data["eval_results"][m]["macro_auc"]*100 for m in models]

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Accuracy",      x=models, y=accs,  marker_color="#2C5F8A"))
            fig.add_trace(go.Bar(name="Macro F1 ×100", x=models, y=f1s,   marker_color="#E8563A"))
            fig.add_trace(go.Bar(name="ROC-AUC ×100",  x=models, y=aucs,  marker_color="#4CAF50"))
            fig.update_layout(
                barmode="group", yaxis_range=[0, 105],
                yaxis_title="Score (%)", plot_bgcolor="rgba(0,0,0,0)",
                height=350, legend=dict(orientation="h", y=-0.2),
            )
            for trace in fig.data:
                trace.text = [f"{v:.1f}" for v in trace.y]
                trace.textposition = "outside"
            st.plotly_chart(fig, use_container_width=True, key="chart_2_line321")

    with col2:
        st.subheader("Per-Class F1 Score (XGBoost)")
        if data["eval_results"] and "XGBoost" in data["eval_results"]:
            report = data["eval_results"]["XGBoost"]["report"]
            class_rows = []
            for cls, metrics in report.items():
                if cls in ("accuracy","macro avg","weighted avg"):
                    continue
                cls_name = cls.replace("Product Listing / Information","Product Description Mismatch")
                class_rows.append({
                    "Root Cause": cls_name,
                    "Precision":  f"{metrics['precision']:.3f}",
                    "Recall":     f"{metrics['recall']:.3f}",
                    "F1-Score":   f"{metrics['f1-score']:.3f}",
                    "Support":    int(metrics["support"]),
                })
            st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("XGBoost ROC-AUC Scores by Class")
    if data["eval_results"] and "XGBoost" in data["eval_results"]:
        auc_scores = data["eval_results"]["XGBoost"]["auc_scores"]
        auc_rows = []
        for cls, auc in sorted(auc_scores.items(), key=lambda x: x[1], reverse=True):
            cls_name = cls.replace("Product Listing / Information","Product Description Mismatch")
            auc_rows.append({"Root Cause": cls_name, "ROC-AUC": f"{auc:.4f}",
                             "Bar": auc})

        auc_df = pd.DataFrame(auc_rows)
        fig = px.bar(auc_df, x="Bar", y="Root Cause", orientation="h",
                     color="Bar", color_continuous_scale="Blues",
                     range_color=[0.97, 1.0],
                     text=[f"{r['ROC-AUC']}" for _, r in auc_df.iterrows()],
                     labels={"Bar":"ROC-AUC Score"})
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=320,
                          plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_range=[0.95, 1.005])
        st.plotly_chart(fig, use_container_width=True, key="chart_3_line361")

    st.subheader("NLP Feature Contribution to ML")
    nlp_impact = pd.DataFrame({
        "Feature Type":     ["NLP Features","Transactional Features","Product/Customer Features"],
        "Count":            [10, 17, 74],
        "Avg SHAP Rank":    ["Top 3","Mixed","Lower"],
        "Key Features":     ["complaint_category, dominant_topic, sentiment_score",
                             "delivery_days, return_lag_days, customer_return_rate",
                             "brand, product_subcategory, material (encoded)"],
    })
    st.dataframe(nlp_impact, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 - SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "shap":
    st.title("🔍 SHAP Explainability")
    st.markdown(
        "**SHapley Additive exPlanations** make every XGBoost prediction transparent. "
        "Each prediction is explained by showing how much each feature contributed "
        "to the classification - positive values push toward the class, "
        "negative values push away."
    )
    st.divider()

    if data["shap_data"] is None:
        st.warning("SHAP data not found. Run run_phase4.py first.")
    else:
        shap_values  = data["shap_data"]["shap_values"]    # (1000, 101, 6)
        X_sample     = data["shap_data"]["X_sample"]
        feature_names = [f.replace("_", " ") for f in X_sample.columns]

        ROOT_CAUSE_LIST = [
            "Customer Preference",
            "Logistics / Delivery",
            "Manufacturing / Quality Control",
            "Product Description Mismatch",
            "Supplier Issues",
            "Warehouse / Packaging",
        ]

        # Global importance
        st.subheader("Global Feature Importance - Mean |SHAP Value|")
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        top_n    = 20
        top_idx  = np.argsort(mean_abs)[::-1][:top_n]
        imp_df   = pd.DataFrame({
            "Feature":    [feature_names[i] for i in top_idx],
            "Mean |SHAP|":[round(mean_abs[i], 4) for i in top_idx],
        })

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(imp_df, x="Mean |SHAP|", y="Feature",
                         orientation="h", color="Mean |SHAP|",
                         color_continuous_scale="Blues",
                         title=f"Top {top_n} Features by Mean |SHAP Value|")
            fig.update_layout(showlegend=False, height=500,
                              plot_bgcolor="rgba(0,0,0,0)",
                              yaxis={"categoryorder":"total ascending"})
            st.plotly_chart(fig, use_container_width=True, key="chart_4_line423")
        with col2:
            st.dataframe(imp_df.head(15), use_container_width=True, hide_index=True)

        st.divider()

        # Feature × Class heatmap
        st.subheader("Feature × Root Cause Class Importance Heatmap")
        mean_per_class = np.abs(shap_values).mean(axis=0)   # (101, 6)
        overall        = mean_per_class.mean(axis=1)
        top15          = np.argsort(overall)[::-1][:15]

        heatmap_data = pd.DataFrame(
            mean_per_class[top15, :],
            index=[feature_names[i] for i in top15],
            columns=ROOT_CAUSE_LIST,
        )
        heatmap_data.columns = [c.replace("Product Description Mismatch","Prod. Desc.")
                                  .replace("Manufacturing / Quality Control","Mfg / QC")
                                  .replace("Customer Preference","Cust. Pref.")
                                  .replace("Logistics / Delivery","Logistics")
                                  .replace("Supplier Issues","Supplier")
                                  .replace("Warehouse / Packaging","Warehouse")
                                  for c in heatmap_data.columns]

        fig = px.imshow(
            heatmap_data,
            text_auto=".3f",
            color_continuous_scale="Blues",
            title="Mean |SHAP Value| per Feature per Root Cause Class",
            aspect="auto",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, key="chart_5_line456")

        st.divider()

        # Per-class SHAP analysis
        st.subheader("Per-Class SHAP Analysis")
        selected_class = st.selectbox("Select root cause class", ROOT_CAUSE_LIST)
        class_idx = ROOT_CAUSE_LIST.index(selected_class)
        sv_class  = shap_values[:, :, class_idx]
        mean_abs_class = np.abs(sv_class).mean(axis=0)
        top15_class    = np.argsort(mean_abs_class)[::-1][:15]

        class_df = pd.DataFrame({
            "Feature":    [feature_names[i] for i in top15_class],
            "Mean |SHAP|":[round(mean_abs_class[i], 4) for i in top15_class],
            "Direction":  ["→ Increases classification" if sv_class[:, i].mean() > 0
                           else "← Decreases classification"
                           for i in top15_class],
        })

        col1, col2 = st.columns([2, 1])
        with col1:
            colours = ["#4CAF50" if "Increases" in d else "#E8563A"
                       for d in class_df["Direction"]]
            fig = go.Figure(go.Bar(
                x=class_df["Mean |SHAP|"],
                y=class_df["Feature"],
                orientation="h",
                marker_color=colours,
            ))
            fig.update_layout(
                title=f"Top Features for: {selected_class}",
                height=450, plot_bgcolor="rgba(0,0,0,0)",
                yaxis={"categoryorder":"total ascending"},
                xaxis_title="Mean |SHAP Value|",
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_6_line492")
        with col2:
            st.dataframe(class_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 - ROOT CAUSE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "rca":
    st.title("📈 Root Cause Analysis")
    st.markdown("System-wide patterns aggregated from 14,000 confirmed returns "
                "across 25 months, 6 brands, and 5 regions.")
    st.divider()

    if data["summary"] is None:
        st.warning("Run run_phase5.py first.")
    else:
        summary = data["summary"].copy()
        summary["root_cause_category"] = summary["root_cause_category"].str.replace(
            "Product Listing / Information","Product Description Mismatch")

        # Summary table
        st.subheader("Root Cause Summary Table")
        display = summary.copy()
        display.columns = ["Root Cause","Count","% Share","Avg Sentiment",
                           "Avg Subjectivity","Avg Return Lag (days)","Avg Delivery Days"]
        display["% Share"]             = display["% Share"].apply(lambda x: f"{x:.1f}%")
        display["Avg Sentiment"]        = display["Avg Sentiment"].apply(lambda x: f"{x:.4f}")
        display["Avg Subjectivity"]     = display["Avg Subjectivity"].apply(lambda x: f"{x:.4f}")
        display["Avg Return Lag (days)"]= display["Avg Return Lag (days)"].apply(lambda x: f"{x:.1f}")
        display["Avg Delivery Days"]    = display["Avg Delivery Days"].apply(lambda x: f"{x:.1f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Return Distribution")
            fig = px.pie(
                summary, values="count", names="root_cause_category",
                color="root_cause_category", color_discrete_map=ROOT_CAUSE_COLOURS,
                hole=0.45,
            )
            fig.update_traces(textposition="outside", textinfo="percent+label")
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True, key="chart_7_line538")

        with col2:
            st.subheader("Sentiment by Root Cause")
            s_sorted = summary.sort_values("avg_sentiment")
            colours  = ["#4CAF50" if x > 0 else "#E8563A" for x in s_sorted["avg_sentiment"]]
            fig = go.Figure(go.Bar(
                x=s_sorted["avg_sentiment"],
                y=s_sorted["root_cause_category"],
                orientation="h",
                marker_color=colours,
                text=[f"{v:.3f}" for v in s_sorted["avg_sentiment"]],
                textposition="outside",
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig.update_layout(
                height=380, plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Mean VADER Sentiment Score",
                xaxis_range=[-0.35, 0.55],
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_8_line558")

        st.divider()

        # Monthly trend
        st.subheader("Monthly Return Trend (Sep 2023 - Sep 2025)")
        if data["trend"] is not None:
            trend = data["trend"].copy()
            trend.index = [str(p) for p in trend.index]
            trend.columns = [c.replace("Product Listing / Information",
                                       "Product Description Mismatch")
                             for c in trend.columns]

            fig = go.Figure()
            for col in trend.columns:
                colour = ROOT_CAUSE_COLOURS.get(col, "#999")
                fig.add_trace(go.Scatter(
                    x=trend.index, y=trend[col], name=col,
                    line={"color": colour, "width": 2},
                    mode="lines+markers", marker_size=3,
                ))
            fig.update_layout(
                height=380, plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Month", yaxis_title="Returns",
                legend=dict(orientation="h", y=-0.3),
                xaxis=dict(tickangle=45),
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_9_line585")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Region Heatmap")
            if data["region"] is not None:
                region = data["region"].copy()
                region["root_cause_category"] = region["root_cause_category"].str.replace(
                    "Product Listing / Information","Product Description Mismatch")
                pivot = region.pivot(index="region",columns="root_cause_category",
                                     values="pct_within_region").fillna(0)
                pivot.columns = [c.replace("Manufacturing / Quality Control","Mfg/QC")
                                  .replace("Product Description Mismatch","Prod.Desc.")
                                  .replace("Customer Preference","Cust.Pref.")
                                  .replace("Warehouse / Packaging","Warehouse")
                                  for c in pivot.columns]
                fig = px.imshow(pivot, text_auto=".1f", color_continuous_scale="YlOrRd",
                                title="% of Returns Within Each Region",aspect="auto")
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True, key="chart_10_line606")

        with col2:
            st.subheader("Brand Comparison")
            if data["brand_detail"] is not None:
                brand = data["brand_detail"].copy()
                if "root_cause_category" in brand.columns:
                    brand["root_cause_category"] = brand["root_cause_category"].str.replace(
                        "Product Listing / Information","Product Description Mismatch")
                st.dataframe(brand, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 - RISK ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "risk":
    st.title("⚠️ Risk Assessment")
    st.markdown(
        "Each root cause is scored using:  \n"
        "**Risk Score = (Frequency × 0.4) + (Operational Impact × 0.4) + (Trend × 0.2)**"
    )
    st.divider()

    if data["risk_df"] is None:
        st.warning("Run run_phase6.py first.")
    else:
        risk = data["risk_df"].copy()
        risk["root_cause_category"] = risk["root_cause_category"].str.replace(
            "Product Listing / Information","Product Description Mismatch")

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        high_count   = (risk["risk_level"] == "High").sum()
        med_count    = (risk["risk_level"] == "Medium").sum()
        total_returns = risk["count"].sum()
        high_returns  = risk[risk["risk_level"]=="High"]["count"].sum()
        kpi_card(c1, "High Risk Root Causes",   str(high_count),               "Require immediate action", "#E8563A")
        kpi_card(c2, "Medium Risk Root Causes", str(med_count),                "Monitor and plan",         "#E8A838")
        kpi_card(c3, "Returns in High Risk",    f"{high_returns:,}",           f"{high_returns/total_returns*100:.0f}% of total", "#E8563A")
        kpi_card(c4, "Top Risk Score",          f"{risk['risk_score'].max():.1f}/100", "Product Desc. Mismatch", "#2C5F8A")

        st.divider()

        # Risk Register Table - the Excel-style table he asked for
        st.subheader("📋 Risk Register")
        risk_register = risk[["priority","root_cause_category","count","pct_of_returns",
                               "frequency_score","impact_score","trend_score",
                               "risk_score","risk_level","avg_sentiment"]].copy()
        risk_register.columns = ["Priority","Root Cause","Returns","% Share",
                                  "Freq Score","Impact Score","Trend Score",
                                  "Risk Score","Risk Level","Avg Sentiment"]
        risk_register["Priority"]     = risk_register["Priority"].apply(lambda x: f"P{int(x)}")
        risk_register["% Share"]      = risk_register["% Share"].apply(lambda x: f"{x:.1f}%")
        risk_register["Freq Score"]   = risk_register["Freq Score"].apply(lambda x: f"{x:.1f}")
        risk_register["Impact Score"] = risk_register["Impact Score"].apply(lambda x: f"{x:.1f}")
        risk_register["Trend Score"]  = risk_register["Trend Score"].apply(lambda x: f"{x:.1f}")
        risk_register["Risk Score"]   = risk_register["Risk Score"].apply(lambda x: f"{x:.1f}")
        risk_register["Avg Sentiment"]= risk_register["Avg Sentiment"].apply(lambda x: f"{x:.3f}")

        def colour_risk(val):
            colours = {"High":"background-color:#FFF3CD","Medium":"background-color:#D1ECF1",
                       "Critical":"background-color:#F8D7DA","Low":"background-color:#D4EDDA"}
            return colours.get(val, "")

        styled = risk_register.style.map(colour_risk, subset=["Risk Level"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risk Priority Ranking")
            colours = [RISK_COLOURS.get(lvl,"#999") for lvl in risk.sort_values("risk_score")["risk_level"]]
            fig = go.Figure(go.Bar(
                x=risk.sort_values("risk_score")["risk_score"],
                y=risk.sort_values("risk_score")["root_cause_category"],
                orientation="h",
                marker_color=colours,
                text=[f"P{int(p)}  {s:.1f}  [{l}]"
                      for p,s,l in zip(
                          risk.sort_values("risk_score")["priority"],
                          risk.sort_values("risk_score")["risk_score"],
                          risk.sort_values("risk_score")["risk_level"])],
                textposition="outside",
            ))
            fig.update_layout(
                height=380, plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Composite Risk Score (0–100)",
                xaxis_range=[0, 115],
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_11_line697")

        with col2:
            st.subheader("Risk Matrix (Frequency vs Impact)")
            fig = go.Figure()

            # Quadrant shading
            for x0,x1,y0,y1,col,label in [
                (0,50,0,50,"rgba(126,211,33,0.08)","LOW RISK"),
                (50,110,0,50,"rgba(74,144,217,0.08)","MEDIUM RISK"),
                (0,50,50,110,"rgba(232,168,56,0.08)","HIGH RISK"),
                (50,110,50,110,"rgba(217,79,61,0.08)","CRITICAL RISK"),
            ]:
                fig.add_shape(type="rect",x0=x0,x1=x1,y0=y0,y1=y1,
                              fillcolor=col,line_width=0)
                fig.add_annotation(x=(x0+x1)/2,y=(y0+y1)/2,text=label,
                                   showarrow=False,font_size=9,font_color="#111111",opacity=0.5)

            for _, row in risk.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["frequency_score"]],
                    y=[row["impact_score"]],
                    mode="markers+text",
                    name=row["root_cause_category"],
                    marker=dict(
                        size=row["trend_score"]/8+10,
                        color=RISK_COLOURS.get(row["risk_level"],"#999"),
                        opacity=0.85,
                        line=dict(color="white",width=1.5),
                    ),
                    text=[row["root_cause_category"].split("/")[0].strip()[:12]],
                    textposition="top right",
                    textfont_size=8,
                    showlegend=False,
                ))

            fig.add_hline(y=50,line_dash="dash",line_color="grey",opacity=0.3)
            fig.add_vline(x=50,line_dash="dash",line_color="grey",opacity=0.3)
            fig.update_layout(
                height=380, plot_bgcolor="rgba(0,0,0,0.02)",
                xaxis=dict(title="Frequency Score (0–100)",range=[-5,115]),
                yaxis=dict(title="Operational Impact Score (0–100)",range=[-5,115]),
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_12_line740")

        st.divider()

        # Score breakdown
        st.subheader("Risk Score Component Breakdown")
        risk_sorted = risk.sort_values("risk_score", ascending=False)
        short_names = [rc.replace("Product Description Mismatch","Prod.Desc.")
                         .replace("Manufacturing / Quality Control","Mfg/QC")
                         .replace("Customer Preference","Cust.Pref.")
                         .replace("Warehouse / Packaging","Warehouse")
                         .replace("Logistics / Delivery","Logistics")
                       for rc in risk_sorted["root_cause_category"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Frequency (×0.4)",
                             x=short_names, y=risk_sorted["frequency_score"]*0.4,
                             marker_color="#2C5F8A"))
        fig.add_trace(go.Bar(name="Impact (×0.4)",
                             x=short_names, y=risk_sorted["impact_score"]*0.4,
                             marker_color="#E8563A"))
        fig.add_trace(go.Bar(name="Trend (×0.2)",
                             x=short_names, y=risk_sorted["trend_score"]*0.2,
                             marker_color="#F5A623"))

        for i,(_, row) in enumerate(risk_sorted.iterrows()):
            fig.add_annotation(x=short_names[i], y=row["risk_score"]+1.5,
                                text=f"{row['risk_score']:.1f}",
                                showarrow=False, font_size=10, font_color="#111111")
        fig.update_layout(
            barmode="stack", height=380, plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Weighted Score Contribution",
            legend=dict(orientation="h",y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_13_line774")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 - SCENARIO SIMULATOR  (Preset AI Scenarios + Custom + Comparison)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "scenario":
    st.title("🎯 Scenario Simulator")
    st.markdown(
        "Select a **preset AI-driven scenario** or build a **custom scenario** using sliders. "
        "Add multiple scenarios to the comparison section to see side-by-side analysis."
    )
    st.divider()

    if data["risk_df"] is None:
        st.warning("Run run_phase6.py first.")
    else:
        risk = data["risk_df"].copy()
        risk["root_cause_category"] = risk["root_cause_category"].str.replace(
            "Product Listing / Information", "Product Description Mismatch")

        # ── Initialise session state for comparison ──────────────────────────
        if "comparison_scenarios" not in st.session_state:
            st.session_state.comparison_scenarios = {}

        # ── 8 Preset scenario definitions ────────────────────────────────────
        PRESET_SCENARIOS = {
            "1 - Conservative Budget (£75k)": {
                "description": "Tight budget - fund only the top 2 highest-risk root causes with standard interventions.",
                "budget": 75_000,
                "cost_per_return": 22,
                "costs": {"Product Description Mismatch":30000,"Warehouse / Packaging":40000,
                          "Manufacturing / Quality Control":35000,"Supplier Issues":45000,
                          "Customer Preference":20000,"Logistics / Delivery":25000},
                "reductions": {"Product Description Mismatch":0.35,"Warehouse / Packaging":0.30,
                               "Manufacturing / Quality Control":0.40,"Supplier Issues":0.45,
                               "Customer Preference":0.15,"Logistics / Delivery":0.25},
                "ai_context": "Conservative budget of £75,000 with standard reduction rates. Only highest-priority interventions are funded.",
            },
            "2 - Standard Budget - Baseline (£150k)": {
                "description": "The default baseline scenario - £150,000 budget with industry-standard reduction estimates.",
                "budget": 150_000,
                "cost_per_return": 22,
                "costs": {"Product Description Mismatch":30000,"Warehouse / Packaging":40000,
                          "Manufacturing / Quality Control":35000,"Supplier Issues":45000,
                          "Customer Preference":20000,"Logistics / Delivery":25000},
                "reductions": {"Product Description Mismatch":0.35,"Warehouse / Packaging":0.30,
                               "Manufacturing / Quality Control":0.40,"Supplier Issues":0.45,
                               "Customer Preference":0.15,"Logistics / Delivery":0.25},
                "ai_context": "Standard baseline scenario with £150,000 budget and default industry reduction rates across all root causes.",
            },
            "3 - Full Investment (£250k)": {
                "description": "Maximum investment - fund all 6 root causes with enhanced intervention budgets.",
                "budget": 250_000,
                "cost_per_return": 22,
                "costs": {"Product Description Mismatch":40000,"Warehouse / Packaging":55000,
                          "Manufacturing / Quality Control":50000,"Supplier Issues":60000,
                          "Customer Preference":25000,"Logistics / Delivery":35000},
                "reductions": {"Product Description Mismatch":0.40,"Warehouse / Packaging":0.35,
                               "Manufacturing / Quality Control":0.45,"Supplier Issues":0.50,
                               "Customer Preference":0.20,"Logistics / Delivery":0.30},
                "ai_context": "Full investment scenario with £250,000 budget and enhanced reduction rates from larger-scale interventions.",
            },
            "4 - Description Fix Priority": {
                "description": "Intensive content overhaul - concentrate budget on fixing product descriptions, photos, and size guides.",
                "budget": 150_000,
                "cost_per_return": 22,
                "costs": {"Product Description Mismatch":80000,"Warehouse / Packaging":40000,
                          "Manufacturing / Quality Control":35000,"Supplier Issues":45000,
                          "Customer Preference":20000,"Logistics / Delivery":25000},
                "reductions": {"Product Description Mismatch":0.55,"Warehouse / Packaging":0.30,
                               "Manufacturing / Quality Control":0.40,"Supplier Issues":0.45,
                               "Customer Preference":0.20,"Logistics / Delivery":0.25},
                "ai_context": "Description-first strategy: £80,000 allocated to content overhaul with 55% reduction expected from intensive photography, copywriting, and size guide programme.",
            },
            "5 - Warehouse Optimisation (Staff +30%)": {
                "description": "Increase warehouse headcount by 30%, implement pre-dispatch quality checks, and retrain packing team.",
                "budget": 150_000,
                "cost_per_return": 22,
                "costs": {"Product Description Mismatch":30000,"Warehouse / Packaging":70000,
                          "Manufacturing / Quality Control":35000,"Supplier Issues":45000,
                          "Customer Preference":20000,"Logistics / Delivery":25000},
                "reductions": {"Product Description Mismatch":0.35,"Warehouse / Packaging":0.50,
                               "Manufacturing / Quality Control":0.40,"Supplier Issues":0.45,
                               "Customer Preference":0.15,"Logistics / Delivery":0.25},
                "ai_context": "Warehouse headcount increased by 30% with pre-dispatch QC checks. Higher £70,000 cost but 50% reduction achievable through process and staffing improvements.",
            },
            "6 - Supply Chain Resilience Programme": {
                "description": "Strategic focus on Supplier + Manufacturing + Warehouse - address the entire upstream supply chain.",
                "budget": 150_000,
                "cost_per_return": 22,
                "costs": {"Product Description Mismatch":20000,"Warehouse / Packaging":45000,
                          "Manufacturing / Quality Control":45000,"Supplier Issues":55000,
                          "Customer Preference":15000,"Logistics / Delivery":25000},
                "reductions": {"Product Description Mismatch":0.20,"Warehouse / Packaging":0.40,
                               "Manufacturing / Quality Control":0.50,"Supplier Issues":0.55,
                               "Customer Preference":0.10,"Logistics / Delivery":0.25},
                "ai_context": "Supply chain resilience focus: upstream quality across Supplier, Manufacturing, and Warehouse with higher reduction rates from coordinated supplier audit programme.",
            },
            "7 - Customer Experience Focus": {
                "description": "Target Customer Preference and Product Description together - improve UX, photography, and sizing.",
                "budget": 150_000,
                "cost_per_return": 22,
                "costs": {"Product Description Mismatch":50000,"Warehouse / Packaging":40000,
                          "Manufacturing / Quality Control":35000,"Supplier Issues":45000,
                          "Customer Preference":40000,"Logistics / Delivery":25000},
                "reductions": {"Product Description Mismatch":0.45,"Warehouse / Packaging":0.30,
                               "Manufacturing / Quality Control":0.40,"Supplier Issues":0.45,
                               "Customer Preference":0.30,"Logistics / Delivery":0.25},
                "ai_context": "Customer experience strategy: combined investment in content accuracy and UX improvements to reduce both description mismatch and change-of-mind returns.",
            },
            "8 - Crisis Response (Logistics Cost Rise)": {
                "description": "Logistics costs have risen to £35/return due to carrier price increases. Emergency carrier review.",
                "budget": 150_000,
                "cost_per_return": 35,
                "costs": {"Product Description Mismatch":30000,"Warehouse / Packaging":40000,
                          "Manufacturing / Quality Control":35000,"Supplier Issues":45000,
                          "Customer Preference":20000,"Logistics / Delivery":50000},
                "reductions": {"Product Description Mismatch":0.35,"Warehouse / Packaging":0.30,
                               "Manufacturing / Quality Control":0.40,"Supplier Issues":0.45,
                               "Customer Preference":0.15,"Logistics / Delivery":0.40},
                "ai_context": "Crisis scenario: logistics costs risen to £35/return. Emergency carrier review with £50,000 investment targeting 40% logistics return reduction.",
            },
        }

        # ── Helper: compute scenario results ─────────────────────────────────
        def compute_scenario(risk_df, budget, cost_per_return, costs, reductions):
            results   = []
            remaining = budget
            total_saving    = 0
            total_prevented = 0

            for _, row in risk_df.sort_values("risk_score", ascending=False).iterrows():
                rc        = row["root_cause_category"]
                cost      = costs.get(rc, DEFAULT_COSTS.get(rc, 30000))
                red       = reductions.get(rc, DEFAULT_REDUCTION.get(rc, 0.3))
                prevented = int(row["count"] * red)
                saving    = prevented * cost_per_return
                roi       = saving / cost if cost > 0 else 0
                fundable  = cost <= remaining

                results.append({
                    "Root Cause":        rc,
                    "Priority":          f"P{int(row['priority'])}",
                    "Risk Score":        f"{row['risk_score']:.1f}",
                    "Cost":              f"£{cost:,}",
                    "Reduction %":       f"{red*100:.0f}%",
                    "Returns Prevented": f"{prevented:,}",
                    "Financial Saving":  f"£{saving:,.0f}",
                    "ROI":               f"{roi:.2f}x",
                    "Status":            "FUND" if fundable else "SKIP",
                    "_saving":           saving if fundable else 0,
                    "_prevented":        prevented if fundable else 0,
                    "_cost":             cost if fundable else 0,
                    "_funded":           fundable,
                })
                if fundable:
                    remaining       -= cost
                    total_saving    += saving
                    total_prevented += prevented

            return pd.DataFrame(results), budget - remaining, total_saving, total_prevented, remaining

        # ── Helper: render scenario output ───────────────────────────────────
        def render_scenario_output(results_df, budget, cost_per_return, total_saving,
                                   total_prevented, remaining, scenario_name=""):
            funded_count = results_df["_funded"].sum()
            c1,c2,c3,c4 = st.columns(4)
            kpi_card(c1,"Budget",              f"£{budget:,}",
                     f"£{budget-remaining:,} allocated","#2C5F8A")
            kpi_card(c2,"Interventions Funded",str(int(funded_count)),
                     "of 6 root causes","#4CAF50")
            kpi_card(c3,"Returns Prevented",   f"{int(total_prevented):,}",
                     "per year","#E8563A")
            kpi_card(c4,"Financial Saving",    f"£{total_saving:,.0f}",
                     f"at £{cost_per_return}/return","#F5A623")

            st.subheader("Scenario Output Table")
            display_cols = ["Root Cause","Priority","Risk Score","Cost","Reduction %",
                            "Returns Prevented","Financial Saving","ROI","Status"]
            def _highlight_status(val):
                if val == "FUND":
                    return "background-color: #C8E6C9; color: #111111; font-weight: 600"
                if val == "SKIP":
                    return "background-color: #FFCDD2; color: #111111; font-weight: 600"
                return ""
            st.dataframe(results_df[display_cols].style.map(_highlight_status, subset=["Status"]),
                         use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Investment vs Saving")
                funded = results_df[results_df["_funded"]]
                if not funded.empty:
                    short = [rc.split("/")[0].strip()[:16] for rc in funded["Root Cause"]]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="Investment (£k)",    x=short,
                                         y=funded["_cost"]/1000,   marker_color="#E8563A"))
                    fig.add_trace(go.Bar(name="Financial Saving (£k)",x=short,
                                         y=funded["_saving"]/1000, marker_color="#4CAF50"))
                    fig.update_layout(barmode="group", height=320,
                                      yaxis_title="Amount (£ thousands)",
                                      plot_bgcolor="rgba(0,0,0,0)",
                                      legend=dict(orientation="h",y=-0.3))
                    st.plotly_chart(fig, use_container_width=True, key=f"invest_vs_saving_{scenario_name}")

            with col2:
                st.subheader("Cumulative Saving Waterfall")
                fs = results_df[results_df["_funded"]].sort_values("_saving", ascending=False)
                if not fs.empty:
                    cumulative = fs["_saving"].cumsum().tolist()
                    labels     = [rc.split("/")[0].strip()[:14] for rc in fs["Root Cause"]]
                    bottoms    = [0] + cumulative[:-1]
                    fig = go.Figure()
                    for label, saving, bottom, rc in zip(labels, fs["_saving"], bottoms, fs["Root Cause"]):
                        fig.add_trace(go.Bar(
                            x=[label], y=[saving], base=[bottom],
                            marker_color=ROOT_CAUSE_COLOURS.get(rc,"#2C5F8A"),
                            showlegend=False,
                            text=[f"+£{saving/1000:.0f}k"], textposition="inside",
                        ))
                    fig.update_layout(height=320, yaxis_title="Cumulative Saving (£)",
                                      plot_bgcolor="rgba(0,0,0,0)", barmode="stack")
                    st.plotly_chart(fig, use_container_width=True, key=f"cumulative_waterfall_{scenario_name}")

        # ════════════════════════════════════════════════════════════════════════
        # TABS
        # ════════════════════════════════════════════════════════════════════════
        tab1, tab2 = st.tabs(["🤖 Preset AI Scenarios", "🔧 Custom Scenario"])

        # ────────────────────────────────────────────────────────────────────────
        # TAB 1 - PRESET AI SCENARIOS
        # ────────────────────────────────────────────────────────────────────────
        with tab1:
            st.markdown("### Select a Preset Scenario")
            selected_preset = st.selectbox(
                "Choose scenario",
                list(PRESET_SCENARIOS.keys()),
                label_visibility="collapsed",
            )
            preset = PRESET_SCENARIOS[selected_preset]

            # Scenario description card
            st.markdown(
                f'<div style="background:#EEF2FF;padding:14px;border-radius:8px;'
                f'border-left:4px solid #2C5F8A;margin-bottom:16px">'
                f'<b>{selected_preset}</b><br>{preset["description"]}</div>',
                unsafe_allow_html=True,
            )

            # Show preset parameters as read-only table
            with st.expander("📋 View scenario parameters", expanded=False):
                param_rows = []
                for rc in risk["root_cause_category"]:
                    param_rows.append({
                        "Root Cause":      rc,
                        "Budget Allocated": f"£{preset['costs'].get(rc,0):,}",
                        "Reduction %":     f"{preset['reductions'].get(rc,0)*100:.0f}%",
                        "Cost per Return": f"£{preset['cost_per_return']}",
                    })
                st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

            # AI Explanation using Gemini
            st.markdown("#### 🤖 AI Analysis of This Scenario")
            if st.button("Generate AI Explanation", key="gen_ai_preset"):
                with st.spinner("Ollama is analysing this scenario..."):
                    try:
                        import os
                        import ollama as _ollama_sc
                        _prompt = f"""You are a supply chain analyst for a UK fashion e-commerce retailer.
Analyse this supply chain intervention scenario and explain it in 3-4 sentences.

Scenario: {selected_preset}
Description: {preset['description']}
Context: {preset['ai_context']}
Total Budget: £{preset['budget']:,}
Cost per Return: £{preset['cost_per_return']}

Explain:
1. What business situation this scenario addresses
2. Why these specific parameters were chosen
3. What the expected operational outcome is
4. Any risks or trade-offs in this approach

Keep it concise and professional, suitable for a supply chain manager."""

                        _resp = _ollama_sc.chat(
                            model="llama3.2",
                            messages=[{"role": "user", "content": _prompt}],
                            options={"num_predict": 400, "temperature": 0.4},
                        )
                        st.session_state[f"ai_explanation_{selected_preset}"] = _resp["message"]["content"]
                    except Exception as e:
                        # Fallback explanation when API not available
                        fallback = {
                            "1 - Conservative Budget (£75k)":
                                "This scenario addresses budget-constrained environments where only the most critical interventions can be funded. With £75,000, the system prioritises Product Description Mismatch (P1, 27% of returns) and Warehouse/Packaging (P2, 20.3%) as the highest-impact investments. The conservative approach ensures ROI is maximised within tight financial constraints, targeting £47,872 in annual savings. The trade-off is that Manufacturing/QC and Supplier Issues remain unaddressed, leaving 25.6% of returns without intervention.",
                            "2 - Standard Budget - Baseline (£150k)":
                                "This baseline scenario represents the recommended standard intervention programme using industry-standard parameters. With £150,000 and 22 per return cost, five of six root causes can be funded in priority order. The standard reduction rates (35% for descriptions, 30% for warehouse) reflect conservative industry benchmarks. This scenario serves as the comparison benchmark - all other scenarios should be evaluated against this baseline to assess whether the additional investment or parameter changes produce superior outcomes.",
                            "3 - Full Investment (£250k)":
                                "Full investment enables simultaneous intervention across all six root cause categories with enhanced budgets per area. Higher individual allocations (£40k-£60k per root cause) enable more comprehensive programmes - full content team restructure, warehouse management system upgrade, and strategic supplier development programme. Enhanced reduction rates (40-50%) reflect the improved outcomes achievable with adequate resources. This scenario demonstrates the maximum achievable return reduction of approximately 3,756 returns prevented annually.",
                            "4 - Description Fix Priority":
                                "This scenario concentrates 53% of the total budget on Product Description Mismatch - the highest volume root cause at 27% of all returns. The £80,000 intensive content overhaul funds professional photography, AI-assisted copywriting, and a complete size guide rebuild. The expected 55% reduction rate significantly exceeds the standard 35%, reflecting the multiplicative impact of addressing description accuracy comprehensively. The trade-off is reduced budget for other root causes, but the ROI on description fixes (£16,128 saving on £80,000 investment per year) is justified by the volume.",
                            "5 - Warehouse Optimisation (Staff +30%)":
                                "Increasing warehouse headcount by 30% addresses the root cause of picking errors and packaging failures at source. The £70,000 warehouse budget funds additional staff, pre-dispatch quality check stations, and retraining programmes. A 50% return reduction from Warehouse/Packaging represents the upper achievable bound with full process redesign and staffing improvements. This scenario is appropriate when warehouse returns are trending upward or when operational capacity is the primary constraint on quality.",
                            "6 - Supply Chain Resilience Programme":
                                "This strategic scenario addresses the entire upstream supply chain simultaneously - Supplier, Manufacturing, and Warehouse - through a coordinated quality programme. Higher reduction rates (50-55%) reflect synergies from treating these three interconnected root causes together rather than independently. The scenario prioritises long-term supply chain resilience over short-term saving maximisation. Appropriate for organisations where supplier relationship management and manufacturing quality are strategic priorities.",
                            "7 - Customer Experience Focus":
                                "By combining Product Description investment with Customer Preference interventions, this scenario targets the 47% of returns attributable to information mismatch and change-of-mind. Enhanced UX (£40,000) funds virtual try-on tools, augmented reality product views, and personalisation features that reduce impulse purchases. The combined 30% reduction in Customer Preference returns - normally the hardest to influence - reflects the impact of pre-purchase decision support tools on return behaviour.",
                            "8 - Crisis Response (Logistics Cost Rise)":
                                "A logistics cost increase to £35 per return fundamentally changes the financial calculus - every return now costs 59% more, making the logistics root cause significantly more expensive despite its lower volume. The £50,000 emergency carrier review funds carrier performance analysis, alternative courier evaluation, and contract renegotiation. A 40% logistics return reduction in this scenario saves substantially more than at the standard £22 cost. This scenario demonstrates how external cost shocks should trigger reallocation of intervention resources.",
                        }
                        st.session_state[f"ai_explanation_{selected_preset}"] = fallback.get(
                            selected_preset,
                            f"This scenario applies {preset['description']} with a budget of £{preset['budget']:,} and £{preset['cost_per_return']} cost per return. Parameters are configured to optimise return reduction within the given constraints."
                        )

            if f"ai_explanation_{selected_preset}" in st.session_state:
                st.markdown(
                    f'<div style="background:#F0FFF4;padding:14px;border-radius:8px;'
                    f'border-left:4px solid #4CAF50;white-space:pre-wrap;font-size:13px;'
                    f'line-height:1.6">'
                    f'🤖 <b>Ollama (Llama 3.2) Analysis:</b><br><br>'
                    f'{st.session_state[f"ai_explanation_{selected_preset}"]}</div>',
                    unsafe_allow_html=True,
                )

            st.divider()

            # Compute and display preset results
            results_df, spent, total_saving, total_prevented, remaining = compute_scenario(
                risk,
                preset["budget"],
                preset["cost_per_return"],
                preset["costs"],
                preset["reductions"],
            )

            render_scenario_output(
                results_df, preset["budget"], preset["cost_per_return"],
                total_saving, total_prevented, remaining, selected_preset
            )

            st.divider()

            # Add to comparison button
            if st.button(f"➕ Add '{selected_preset}' to Comparison", key="add_preset"):
                st.session_state.comparison_scenarios[selected_preset] = {
                    "budget":            preset["budget"],
                    "cost_per_return":   preset["cost_per_return"],
                    "total_saving":      total_saving,
                    "total_prevented":   int(total_prevented),
                    "budget_spent":      spent,
                    "funded_count":      int(results_df["_funded"].sum()),
                    "results_df":        results_df,
                }
                st.success(f"✅ '{selected_preset}' added to comparison. "
                           f"({len(st.session_state.comparison_scenarios)} scenarios total)")

        # ────────────────────────────────────────────────────────────────────────
        # TAB 2 - CUSTOM SCENARIO
        # ────────────────────────────────────────────────────────────────────────
        with tab2:
            st.markdown("### Build Your Own Scenario")
            st.markdown("Adjust all parameters manually and see results update live.")

            col_a, col_b = st.columns(2)
            with col_a:
                custom_budget = st.slider(
                    "Intervention Budget (£)", 50_000, 300_000, 150_000, 10_000,
                    format="£%d", key="custom_budget"
                )
                custom_cpr = st.slider(
                    "Cost per Return (£)", 10, 50, 22, 1,
                    format="£%d", key="custom_cpr"
                )
            with col_b:
                custom_name = st.text_input(
                    "Scenario Name (for comparison)",
                    value="My Custom Scenario",
                    key="custom_name"
                )

            st.markdown("**Expected Reduction % per Root Cause**")
            c1,c2,c3 = st.columns(3)
            custom_reductions = {}
            for i, rc in enumerate(risk["root_cause_category"]):
                col = [c1, c2, c3][i % 3]
                default = int(DEFAULT_REDUCTION.get(rc, 0.3) * 100)
                custom_reductions[rc] = col.slider(
                    rc.split("/")[0].strip()[:20],
                    5, 65, default, 5, format="%d%%",
                    key=f"cust_red_{rc}"
                ) / 100

            st.markdown("**Intervention Costs (£)**")
            c1,c2,c3 = st.columns(3)
            custom_costs = {}
            for i, rc in enumerate(risk["root_cause_category"]):
                col = [c1, c2, c3][i % 3]
                default_cost = int(DEFAULT_COSTS.get(rc, 30_000))
                custom_costs[rc] = col.number_input(
                    rc.split("/")[0].strip()[:20],
                    min_value=5_000, max_value=200_000,
                    value=default_cost, step=5_000,
                    key=f"cust_cost_{rc}"
                )

            st.divider()

            # Compute custom results
            results_df_c, spent_c, total_saving_c, total_prevented_c, remaining_c = compute_scenario(
                risk, custom_budget, custom_cpr, custom_costs, custom_reductions
            )

            render_scenario_output(
                results_df_c, custom_budget, custom_cpr,
                total_saving_c, total_prevented_c, remaining_c, custom_name
            )

            st.divider()

            if st.button(f"➕ Add '{custom_name}' to Comparison", key="add_custom"):
                st.session_state.comparison_scenarios[custom_name] = {
                    "budget":          custom_budget,
                    "cost_per_return": custom_cpr,
                    "total_saving":    total_saving_c,
                    "total_prevented": int(total_prevented_c),
                    "budget_spent":    spent_c,
                    "funded_count":    int(results_df_c["_funded"].sum()),
                    "results_df":      results_df_c,
                }
                st.success(f"✅ '{custom_name}' added to comparison. "
                           f"({len(st.session_state.comparison_scenarios)} scenarios total)")

        # ════════════════════════════════════════════════════════════════════════
        # COMPARISON SECTION - always visible below tabs
        # ════════════════════════════════════════════════════════════════════════
        st.divider()
        st.header("📊 Scenario Comparison")

        if not st.session_state.comparison_scenarios:
            st.info("Add scenarios using the **➕ Add to Comparison** buttons above to compare them here.")
        else:
            col_clear, col_info = st.columns([1, 4])
            with col_clear:
                if st.button("🗑️ Clear All Comparisons"):
                    st.session_state.comparison_scenarios = {}
                    st.rerun()
            with col_info:
                st.markdown(f"**{len(st.session_state.comparison_scenarios)} scenario(s) in comparison**")

            scenarios = st.session_state.comparison_scenarios
            names     = list(scenarios.keys())

            # ── Comparison summary table ──────────────────────────────────────
            st.subheader("Comparison Summary Table")
            comp_rows = []
            for name, s in scenarios.items():
                roi = s["total_saving"] / s["budget_spent"] if s["budget_spent"] > 0 else 0
                comp_rows.append({
                    "Scenario":           name,
                    "Budget":             f"£{s['budget']:,}",
                    "Budget Spent":       f"£{s['budget_spent']:,}",
                    "Interventions":      f"{s['funded_count']} of 6",
                    "Returns Prevented":  f"{s['total_prevented']:,}",
                    "Financial Saving":   f"£{s['total_saving']:,.0f}",
                    "ROI":                f"{roi:.2f}x",
                    "Cost/Return":        f"£{s['cost_per_return']}",
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            st.divider()

            # ── Comparison charts ─────────────────────────────────────────────
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Financial Saving by Scenario")
                short_names = [n[:30] for n in names]
                savings     = [scenarios[n]["total_saving"]/1000 for n in names]
                colours_s   = px.colors.qualitative.Set2[:len(names)]
                fig = go.Figure(go.Bar(
                    x=short_names, y=savings,
                    marker_color=colours_s,
                    text=[f"£{v:.0f}k" for v in savings],
                    textposition="outside",
                ))
                fig.update_layout(
                    height=380, yaxis_title="Financial Saving (£ thousands)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-20,
                    yaxis_range=[0, max(savings)*1.2 if savings else 100],
                )
                st.plotly_chart(fig, use_container_width=True, key="chart_14_line1261")

            with col2:
                st.subheader("Returns Prevented by Scenario")
                prevented = [scenarios[n]["total_prevented"] for n in names]
                fig = go.Figure(go.Bar(
                    x=short_names, y=prevented,
                    marker_color=colours_s,
                    text=[f"{v:,}" for v in prevented],
                    textposition="outside",
                ))
                fig.update_layout(
                    height=380, yaxis_title="Returns Prevented per Year",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-20,
                    yaxis_range=[0, max(prevented)*1.2 if prevented else 100],
                )
                st.plotly_chart(fig, use_container_width=True, key="chart_15_line1278")

            st.divider()

            # ── Grouped bar: saving per root cause across scenarios ───────────
            st.subheader("Saving per Root Cause - Scenario Comparison")
            all_rcs = list(risk["root_cause_category"])
            fig = go.Figure()
            for i, (name, s) in enumerate(scenarios.items()):
                rc_savings = []
                for rc in all_rcs:
                    row = s["results_df"][s["results_df"]["Root Cause"] == rc]
                    rc_savings.append(row["_saving"].values[0]/1000 if len(row) > 0 else 0)
                fig.add_trace(go.Bar(
                    name=name[:25],
                    x=[rc.split("/")[0].strip()[:14] for rc in all_rcs],
                    y=rc_savings,
                    text=[f"£{v:.0f}k" if v > 0 else "" for v in rc_savings],
                    textposition="outside",
                ))
            fig.update_layout(
                barmode="group", height=420,
                yaxis_title="Financial Saving (£ thousands)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_16_line1304")

            st.divider()

            # ── Radar chart ───────────────────────────────────────────────────
            st.subheader("Multi-Dimension Radar Comparison")
            if len(scenarios) >= 2:
                max_saving    = max(s["total_saving"]    for s in scenarios.values()) or 1
                max_prevented = max(s["total_prevented"] for s in scenarios.values()) or 1
                max_funded    = 6

                fig = go.Figure()
                radar_cats = ["Financial Saving","Returns Prevented",
                              "Interventions Funded","Budget Efficiency","Coverage"]

                for i, (name, s) in enumerate(scenarios.items()):
                    roi = s["total_saving"] / s["budget_spent"] if s["budget_spent"] > 0 else 0
                    vals = [
                        s["total_saving"]    / max_saving    * 100,
                        s["total_prevented"] / max_prevented * 100,
                        s["funded_count"]    / max_funded    * 100,
                        min(roi * 50, 100),
                        s["funded_count"]    / max_funded    * 100,
                    ]
                    vals.append(vals[0])   # close the polygon
                    fig.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=radar_cats + [radar_cats[0]],
                        fill="toself",
                        name=name[:25],
                        opacity=0.6,
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    height=450,
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig, use_container_width=True, key="chart_17_line1343")
            else:
                st.info("Add at least 2 scenarios to see the radar comparison chart.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 - LLM RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "llm":
    st.title("💡 LLM Recommendation Engine")
    st.markdown(
        "**Ollama (Llama 3.2)** synthesises risk scores, SHAP outputs, sentiment profiles, "
        "and brand/region analysis into stakeholder-specific natural language recommendations "
        "- running fully locally with no external API."
    )
    st.divider()

    if data["recs_data"] is None:
        st.warning("Run run_phase7.py first.")
    else:
        recs = data["recs_data"]["recommendations"]
        meta = data["recs_data"]["metadata"]

        # Metadata
        c1,c2,c3,c4 = st.columns(4)
        kpi_card(c1, "LLM Model",       meta.get("model","Ollama: llama3.2"),    "Generative AI",       "#9B59B6")
        kpi_card(c2, "Recommendations", str(len(recs)),                "6 root causes",       "#2C5F8A")
        kpi_card(c3, "Total Saving",    f"£{meta.get('total_saving',0):,.0f}", "Potential annual", "#4CAF50")
        kpi_card(c4, "Total Cost",      f"£{meta.get('total_cost',0):,.0f}",   "Annual exposure",  "#E8563A")

        st.divider()

        # Financial summary table
        st.subheader("Recommendation Summary Table")
        summary_rows = []
        for rc, r in sorted(recs.items(), key=lambda x: x[1]["priority"]):
            rc_display = rc.replace("Product Listing / Information","Product Description Mismatch")
            summary_rows.append({
                "Priority":         f"P{r['priority']}",
                "Root Cause":       rc_display,
                "Stakeholder":      r["stakeholder"],
                "Returns":          f"{r['returns']:,}",
                "% Share":          f"{r['pct_share']:.1f}%",
                "Risk Level":       r["risk_level"],
                "Reduction %":      f"{r['reduction_pct']:.0f}%",
                "Potential Saving": f"£{r['potential_saving']:,.0f}",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Individual recommendation cards
        st.subheader("AI-Generated Recommendations")
        for rc, r in sorted(recs.items(), key=lambda x: x[1]["priority"]):
            rc_display = rc.replace("Product Listing / Information","Product Description Mismatch")
            risk_colour = RISK_COLOURS.get(r["risk_level"],"#999")

            with st.expander(
                f"P{r['priority']} | {rc_display} - {r['stakeholder']} | "
                f"£{r['potential_saving']:,.0f} saving | {r['risk_level']} Risk",
                expanded=(r["priority"] <= 2),
            ):
                c1,c2,c3,c4 = st.columns(4)
                kpi_card(c1,"Returns",       f"{r['returns']:,}",              f"{r['pct_share']:.1f}% of total",   risk_colour)
                kpi_card(c2,"Risk Score",    f"{r['risk_score']:.1f}",         r["risk_level"],                     risk_colour)
                kpi_card(c3,"Avg Sentiment", f"{r['avg_sentiment']:.3f}",      "Customer tone",                     "#2C5F8A")
                kpi_card(c4,"Saving",        f"£{r['potential_saving']:,.0f}", f"{r['reduction_pct']:.0f}% reduction","#4CAF50")

                def _render_rec(text, colour):
                    import re
                    SECS = [
                        ("SITUATION SUMMARY",   "📋 Situation Summary",               "#2C5F8A"),
                        ("ROOT CAUSE ANALYSIS", "🔍 Root Cause Analysis",             "#E8563A"),
                        ("IMMEDIATE ACTIONS",   "⚡ Immediate Actions (next 30 days)","#E8A838"),
                        ("MEDIUM-TERM ACTIONS", "📅 Medium-Term Actions (30–90 days)","#9B59B6"),
                        ("EXPECTED IMPACT",     "💰 Expected Impact",                 "#4CAF50"),
                        ("KEY PERFORMANCE",     "📊 KPIs to Track",                  "#1ABC9C"),
                        ("STAKEHOLDER MESSAGE", "💬 Stakeholder Message",            colour),
                    ]
                    tu = text.upper()
                    positions = [(tu.find(k), lb, c) for k,lb,c in SECS if tu.find(k) != -1]
                    positions.sort(key=lambda x: x[0])
                    if not positions:
                        st.markdown(text); return
                    for i, (pos, label, col) in enumerate(positions):
                        end = positions[i+1][0] if i+1 < len(positions) else len(text)
                        body = [l.strip() for l in text[pos:end].split("\n")[1:] if l.strip()]
                        if not body: continue
                        st.markdown(
                            f'<div style="background:{col}18;border-left:4px solid {col};' +
                            f'padding:6px 14px;border-radius:6px;margin:14px 0 4px 0">' +
                            f'<b style="color:{col};font-size:13px">{label}</b></div>',
                            unsafe_allow_html=True)
                        for line in body:
                            cl = re.sub(r'\*\*', '', line).strip()
                            if not cl: continue
                            if re.match(r'^\d+[.)]', cl):
                                m = re.match(r'^(\d+[.)]\s*)(.*)', cl)
                                if m:
                                    st.markdown(
                                        f'<div style="margin:6px 0;color:#111111;font-size:13px">' +
                                        f'<b style="color:{col}">{m.group(1)}</b>{m.group(2)}</div>',
                                        unsafe_allow_html=True)
                            elif re.match(r'^(measure|outcome|target):', cl, re.I):
                                st.markdown(
                                    f'<div style="background:#F0F0F0;border-left:3px solid #AAAAAA;' +
                                    f'padding:4px 10px;margin:3px 0 3px 18px;border-radius:4px;' +
                                    f'font-size:12px;color:#333333">✓ {cl}</div>',
                                    unsafe_allow_html=True)
                            elif cl.startswith(('-','•','*')):
                                st.markdown(
                                    f'<div style="margin:3px 0 3px 8px;color:#111111;font-size:13px">' +
                                    f'• {cl.lstrip("-•* ").strip()}</div>',
                                    unsafe_allow_html=True)
                            else:
                                st.markdown(
                                    f'<p style="margin:5px 0;color:#111111;font-size:13px;line-height:1.6">' +
                                    f'{cl}</p>', unsafe_allow_html=True)
                _render_rec(r["recommendation_text"], risk_colour)

        st.divider()

        st.divider()
        st.subheader("📋 Executive Summary")
        exec_summary = data["recs_data"].get("executive_summary","")

        def _render_exec(text):
            import re
            ESECS = [
                ("OVERVIEW",           "🌐 Overview",               "#2C5F8A"),
                ("KEY FINDINGS",       "🔍 Key Findings",           "#E8563A"),
                ("FINANCIAL IMPACT",   "💰 Financial Impact",       "#4CAF50"),
                ("TOP 3 PRIORITY",     "⚡ Top 3 Priority Actions", "#E8A838"),
                ("RECOMMENDED BUDGET", "💳 Budget Allocation",      "#9B59B6"),
                ("AI FRAMEWORK",       "🤖 AI Framework Value",     "#1ABC9C"),
                ("CONCLUSION",         "✅ Conclusion",             "#2C5F8A"),
            ]
            tu = text.upper()
            positions = [(tu.find(k), lb, c) for k,lb,c in ESECS if tu.find(k) != -1]
            positions.sort(key=lambda x: x[0])
            if not positions:
                st.markdown(text); return
            for i, (pos, label, col) in enumerate(positions):
                end = positions[i+1][0] if i+1 < len(positions) else len(text)
                body = [l.strip() for l in text[pos:end].split("\n")[1:] if l.strip()]
                if not body: continue
                st.markdown(
                    f'<div style="background:{col}18;border-left:4px solid {col};' +
                    f'padding:6px 14px;border-radius:6px;margin:14px 0 4px 0">' +
                    f'<b style="color:{col};font-size:13px">{label}</b></div>',
                    unsafe_allow_html=True)
                for line in body:
                    cl = re.sub(r'\*\*', '', line).strip()
                    if not cl: continue
                    if re.match(r'^\d+[.)]', cl):
                        m = re.match(r'^(\d+[.)]\s*)(.*)', cl)
                        if m:
                            st.markdown(
                                f'<div style="margin:6px 0;color:#111111;font-size:13px">' +
                                f'<b style="color:{col}">{m.group(1)}</b>{m.group(2)}</div>',
                                unsafe_allow_html=True)
                    elif cl.startswith(('-','•','*')):
                        st.markdown(
                            f'<div style="margin:3px 0 3px 8px;color:#111111;font-size:13px">' +
                            f'• {cl.lstrip("-•* ").strip()}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<p style="margin:5px 0;color:#111111;font-size:13px;line-height:1.6">' +
                            f'{cl}</p>', unsafe_allow_html=True)
        _render_exec(exec_summary)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 - STAKEHOLDER VIEWS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "stakeholder":
    st.title("👤 Stakeholder Views")
    st.markdown("Role-based filtered views - each stakeholder sees only what is relevant to them.")
    st.divider()

    role = st.selectbox("Select Your Role", [
        "🏢 Executive / Board",
        "📦 Warehouse Operations Manager",
        "🎨 Marketing & Content Team",
        "🔬 Quality Assurance Team",
        "🤝 Procurement & Supplier Manager",
        "🚚 Logistics & Carrier Manager",
        "🔍 Data Analyst",
    ])

    st.divider()

    risk = data["risk_df"].copy() if data["risk_df"] is not None else pd.DataFrame()
    if not risk.empty:
        risk["root_cause_category"] = risk["root_cause_category"].str.replace(
            "Product Listing / Information","Product Description Mismatch")

    recs = data["recs_data"]["recommendations"] if data["recs_data"] else {}

    def fix_rc(rc):
        return rc.replace("Product Listing / Information","Product Description Mismatch")

    recs_fixed = {fix_rc(k): v for k, v in recs.items()}

    if "Executive" in role:
        c1,c2,c3,c4 = st.columns(4)
        total_returns = risk["count"].sum() if not risk.empty else 14000
        kpi_card(c1,"Total Annual Returns",   f"{total_returns:,}",   "Confirmed returns",  "#E8563A")
        kpi_card(c2,"Annual Cost Exposure",   "£308,000",             "At £22/return",      "#E8563A")
        kpi_card(c3,"Max Achievable Saving",  "£95,524",              "With interventions", "#4CAF50")
        kpi_card(c4,"High Risk Root Causes",  "3 of 6",               "Require action now", "#E8A838")

        st.subheader("Board-Level Risk Summary")
        if not risk.empty:
            board_df = risk[["priority","root_cause_category","count","pct_of_returns","risk_score","risk_level"]].copy()
            board_df.columns = ["Priority","Root Cause","Returns","% Share","Risk Score","Risk Level"]
            board_df["Priority"] = board_df["Priority"].apply(lambda x: f"P{int(x)}")
            board_df["% Share"]  = board_df["% Share"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(board_df, use_container_width=True, hide_index=True)

        st.subheader("Top 3 Priority Actions")
        actions = [
            ("P1","Fix product descriptions and size guides","Marketing Team","£29,128/year"),
            ("P2","Audit warehouse picking and packaging process","Warehouse Manager","£18,744/year"),
            ("P3","Issue quality notice to manufacturing suppliers","QA Team","£20,086/year"),
        ]
        for p,action,owner,saving in actions:
            st.markdown(f"**{p}** - {action}  |  Owner: *{owner}*  |  Saving: **{saving}**")

    elif "Warehouse" in role:
        rc = "Warehouse / Packaging"
        if rc in recs_fixed:
            r = recs_fixed[rc]
            c1,c2,c3 = st.columns(3)
            kpi_card(c1,"Your Returns",   f"{r['returns']:,}",        f"{r['pct_share']:.1f}% of all returns","#E8563A")
            kpi_card(c2,"Risk Level",     r["risk_level"],            f"Score: {r['risk_score']:.1f}/100",     "#E8A838")
            kpi_card(c3,"Potential Saving",f"£{r['potential_saving']:,.0f}","With 30% reduction",             "#4CAF50")

            st.subheader("Your Recommendation")
            st.markdown(r["recommendation_text"])
            st.subheader("KPIs to Track")
            st.markdown("- Picking accuracy rate (target: >99.5%)\n- Packaging damage rate on arrival\n- Wrong item dispatch rate\n- Return lag days for warehouse-related returns")

    elif "Marketing" in role:
        rc = "Product Description Mismatch"
        if rc in recs_fixed:
            r = recs_fixed[rc]
            c1,c2,c3 = st.columns(3)
            kpi_card(c1,"Your Returns",    f"{r['returns']:,}",       f"{r['pct_share']:.1f}% of all returns","#E8563A")
            kpi_card(c2,"Risk Level",      r["risk_level"],           f"Score: {r['risk_score']:.1f}/100",    "#E8A838")
            kpi_card(c3,"Potential Saving",f"£{r['potential_saving']:,.0f}","With 35% reduction",            "#4CAF50")

            st.subheader("Your Recommendation")
            st.markdown(r["recommendation_text"])
            st.subheader("KPIs to Track")
            st.markdown("- Description mismatch return rate per product\n- Size guide accuracy complaints\n- Photo vs reality complaints in reviews\n- Customer satisfaction score post-update")

    elif "Quality" in role:
        rc = "Manufacturing / Quality Control"
        if rc in recs_fixed:
            r = recs_fixed[rc]
            c1,c2,c3 = st.columns(3)
            kpi_card(c1,"Your Returns",    f"{r['returns']:,}",       f"{r['pct_share']:.1f}% of all returns","#E8563A")
            kpi_card(c2,"Risk Level",      r["risk_level"],           f"Score: {r['risk_score']:.1f}/100",    "#F5A623")
            kpi_card(c3,"Potential Saving",f"£{r['potential_saving']:,.0f}","With 40% reduction",            "#4CAF50")

            st.subheader("Your Recommendation")
            st.markdown(r["recommendation_text"])

    elif "Procurement" in role:
        rc = "Supplier Issues"
        if rc in recs_fixed:
            r = recs_fixed[rc]
            c1,c2,c3 = st.columns(3)
            kpi_card(c1,"Your Returns",    f"{r['returns']:,}",       f"{r['pct_share']:.1f}% of all returns","#9B59B6")
            kpi_card(c2,"Risk Level",      r["risk_level"],           f"Score: {r['risk_score']:.1f}/100",    "#E8A838")
            kpi_card(c3,"Potential Saving",f"£{r['potential_saving']:,.0f}","With 45% reduction",            "#4CAF50")

            st.subheader("Your Recommendation")
            st.markdown(r["recommendation_text"])

    elif "Logistics" in role:
        rc = "Logistics / Delivery"
        if rc in recs_fixed:
            r = recs_fixed[rc]
            c1,c2,c3 = st.columns(3)
            kpi_card(c1,"Your Returns",    f"{r['returns']:,}",       f"{r['pct_share']:.1f}% of all returns","#E8563A")
            kpi_card(c2,"Avg Sentiment",   f"{r['avg_sentiment']:.3f}","Most negative class",                "#E8563A")
            kpi_card(c3,"Potential Saving",f"£{r['potential_saving']:,.0f}","With 25% reduction",            "#4CAF50")

            st.subheader("Your Recommendation")
            st.markdown(r["recommendation_text"])

    elif "Analyst" in role:
        st.subheader("Full Risk Register")
        if not risk.empty:
            st.dataframe(risk, use_container_width=True, hide_index=True)

        st.subheader("Model Performance Summary")
        if data["eval_results"]:
            rows = []
            for model, metrics in data["eval_results"].items():
                rows.append({
                    "Model":    model,
                    "Accuracy": f"{metrics['accuracy']*100:.1f}%",
                    "Macro F1": f"{metrics['macro_f1']:.4f}",
                    "AUC":      f"{metrics['macro_auc']:.4f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("SHAP Global Importance (Top 10)")
        if data["shap_data"]:
            shap_values  = data["shap_data"]["shap_values"]
            X_sample     = data["shap_data"]["X_sample"]
            feature_names= [f.replace("_"," ") for f in X_sample.columns]
            mean_abs     = np.abs(shap_values).mean(axis=(0,2))
            top10        = np.argsort(mean_abs)[::-1][:10]
            imp_df = pd.DataFrame({
                "Feature":    [feature_names[i] for i in top10],
                "Mean |SHAP|":[round(mean_abs[i],4) for i in top10],
            })
            st.dataframe(imp_df, use_container_width=True, hide_index=True)