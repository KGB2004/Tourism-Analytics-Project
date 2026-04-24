"""
Tourism Analytics Dashboard
============================
A Streamlit app that predicts Tourism Performance Index (TPI)
using Linear Regression and assigns cluster labels using KMeans.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Analytics Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
    <style>
        /* Main background */
        .stApp { background-color: #f0f4f8; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(160deg, #1a3c5e 0%, #2d6a9f 100%);
        }
        [data-testid="stSidebar"] * { color: #e8f4fd !important; }
        [data-testid="stSidebar"] .stSlider label { color: #b8d9f0 !important; }

        /* Header banner */
        .header-banner {
            background: linear-gradient(135deg, #1a3c5e 0%, #2d6a9f 50%, #3a8ec4 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            color: white;
            text-align: center;
            box-shadow: 0 6px 24px rgba(26,60,94,0.25);
        }
        .header-banner h1 { font-size: 2.4rem; margin: 0; font-weight: 800; letter-spacing: -0.5px; }
        .header-banner p  { font-size: 1.05rem; margin: 0.4rem 0 0; opacity: 0.88; }

        /* Result cards */
        .result-card {
            background: white;
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
            border-left: 5px solid #2d6a9f;
        }
        .result-card.high  { border-left-color: #27ae60; }
        .result-card.mid   { border-left-color: #e67e22; }
        .result-card.low   { border-left-color: #e74c3c; }

        .cluster-badge {
            display: inline-block;
            padding: 0.4rem 1.2rem;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1rem;
            margin-top: 0.5rem;
        }
        .badge-high { background: #d4efdf; color: #1e8449; }
        .badge-mid  { background: #fdebd0; color: #ca6f1e; }
        .badge-low  { background: #fadbd8; color: #cb4335; }

        /* Section titles */
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1a3c5e;
            margin: 1.2rem 0 0.6rem;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid #d1e8f7;
        }

        /* Metric overrides */
        [data-testid="metric-container"] {
            background: white;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        }

        /* Divider */
        hr { border-color: #d0e4f0; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str) -> pd.DataFrame:
    """Load and clean the tourism dataset."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    return df


# ─────────────────────────────────────────────
# Feature & Target Setup
# ─────────────────────────────────────────────
FEATURES = [
    "Seasonality_Index",
    "Revenue_Per_Tourist_INR",
    "Foreign_Revenue_Percent",
    "Growth % (Approx.)",
]
TARGET = "TPI"

CLUSTER_LABELS = {
    0: "High Performing State",
    1: "Emerging State",
    2: "Low Performing State",
}
CLUSTER_COLORS = {
    0: ("#27ae60", "high", "badge-high", "🏆"),
    1: ("#e67e22", "mid",  "badge-mid",  "📈"),
    2: ("#e74c3c", "low",  "badge-low",  "🔧"),
}


# ─────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models…")
def train_models(df: pd.DataFrame):
    """
    Train:
    1. LinearRegression to predict TPI
    2. KMeans (k=3) to cluster tourism states

    Returns trained reg model, kmeans model, scaler, and evaluation metrics.
    """
    # Drop rows with missing values in required columns
    required_cols = FEATURES + [TARGET]
    clean_df = df[required_cols].dropna()

    X = clean_df[FEATURES].values
    y = clean_df[TARGET].values

    # ── Linear Regression ──────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # ── KMeans Clustering ──────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Attach cluster labels back to clean_df for later visualisation
    clean_df = clean_df.copy()
    clean_df["Cluster"] = kmeans.labels_

    return reg, kmeans, scaler, r2, mae, clean_df


# ─────────────────────────────────────────────
# Prediction Helpers
# ─────────────────────────────────────────────
def predict_tpi(reg, user_input: dict) -> float:
    """Predict TPI score from user inputs."""
    input_df = pd.DataFrame([user_input])
    return float(reg.predict(input_df[FEATURES])[0])


def predict_cluster(kmeans, scaler, user_input: dict) -> int:
    """Scale user input and assign KMeans cluster."""
    input_df = pd.DataFrame([user_input])
    scaled   = scaler.transform(input_df[FEATURES])
    return int(kmeans.predict(scaled)[0])


def resolve_cluster_label(cluster_id: int) -> tuple[str, str]:
    """Return (friendly label, interpretation sentence) for a cluster id."""
    label = CLUSTER_LABELS.get(cluster_id, "Unknown")

    interpretations = {
        0: (
            "This state is a top-tier tourism destination with strong revenue, "
            "stable seasonality, and consistent visitor growth. "
            "Policy focus: sustain infrastructure and diversify offerings."
        ),
        1: (
            "This state shows moderate tourism performance with clear growth "
            "potential. Targeted investment in marketing and infrastructure "
            "could elevate it to the high-performing category."
        ),
        2: (
            "This state currently lags in tourism metrics. "
            "Strategic interventions — improved connectivity, promotional campaigns, "
            "and foreign-visitor initiatives — are recommended."
        ),
    }
    return label, interpretations.get(cluster_id, "")


# ─────────────────────────────────────────────
# Visualisation Helpers
# ─────────────────────────────────────────────
def gauge_chart(tpi_value: float) -> go.Figure:
    """Render a speedometer gauge for the predicted TPI score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(tpi_value, 4),
        delta={"reference": 0.473, "valueformat": ".4f"},
        title={"text": "Tourism Performance Index (TPI)", "font": {"size": 18}},
        number={"valueformat": ".4f", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 1], "tickformat": ".1f"},
            "bar": {"color": "#2d6a9f"},
            "steps": [
                {"range": [0,    0.35], "color": "#fadbd8"},
                {"range": [0.35, 0.65], "color": "#fdebd0"},
                {"range": [0.65, 1.0],  "color": "#d4efdf"},
            ],
            "threshold": {
                "line": {"color": "#1a3c5e", "width": 3},
                "thickness": 0.82,
                "value": tpi_value,
            },
        },
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#1a3c5e",
    )
    return fig


def cluster_scatter(clean_df: pd.DataFrame, user_input: dict, user_cluster: int) -> go.Figure:
    """Scatter plot of dataset clusters + user's predicted position."""
    color_map = {
        "High Performing State": "#27ae60",
        "Emerging State":        "#e67e22",
        "Low Performing State":  "#e74c3c",
        "Your Input":            "#8e44ad",
    }
    plot_df = clean_df.copy()
    plot_df["Cluster Label"] = plot_df["Cluster"].map(CLUSTER_LABELS)

    fig = px.scatter(
        plot_df,
        x="Seasonality_Index",
        y="TPI",
        color="Cluster Label",
        color_discrete_map=color_map,
        size="Revenue_Per_Tourist_INR",
        size_max=18,
        opacity=0.65,
        hover_data=["Foreign_Revenue_Percent", "Growth % (Approx.)"],
        title="Cluster Distribution — Dataset vs Your Input",
        labels={"Seasonality_Index": "Seasonality Index", "TPI": "Tourism Performance Index"},
    )

    # Add user point
    fig.add_trace(go.Scatter(
        x=[user_input["Seasonality_Index"]],
        y=[predict_tpi(reg, user_input)],
        mode="markers+text",
        marker=dict(color="#8e44ad", size=18, symbol="star", line=dict(color="white", width=2)),
        text=["Your Input"],
        textposition="top center",
        name="Your Input",
    ))

    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,244,248,0.6)",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=20, r=20, t=50, b=60),
        font_color="#1a3c5e",
    )
    return fig


def feature_bar_chart(user_input: dict, clean_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing user input to dataset averages."""
    avg = clean_df[FEATURES].mean()

    categories = [
        "Seasonality Index",
        "Revenue / Tourist (INR)",
        "Foreign Revenue %",
        "Growth %",
    ]
    user_vals = [
        user_input["Seasonality_Index"],
        user_input["Revenue_Per_Tourist_INR"],
        user_input["Foreign_Revenue_Percent"],
        user_input["Growth % (Approx.)"],
    ]
    avg_vals = avg.tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Your Input",
        x=categories,
        y=user_vals,
        marker_color="#2d6a9f",
        opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        name="Dataset Average",
        x=categories,
        y=avg_vals,
        marker_color="#aed6f1",
        opacity=0.85,
    ))
    fig.update_layout(
        barmode="group",
        title="Your Input vs Dataset Average",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,244,248,0.6)",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=20, r=20, t=50, b=60),
        font_color="#1a3c5e",
        yaxis_title="Value",
    )
    return fig


# ─────────────────────────────────────────────
# App Entry Point
# ─────────────────────────────────────────────

# ── Load Data ─────────────────────────────────
DATA_PATH = "ABA Final Project.xlsx"
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"⚠️  Dataset not found at `{DATA_PATH}`. "
        "Please place the Excel file in the same directory as app.py."
    )
    st.stop()

# ── Train Models ──────────────────────────────
reg, kmeans, scaler, r2, mae, clean_df = train_models(df)

# ── Header ────────────────────────────────────
st.markdown("""
    <div class="header-banner">
        <h1>🗺️ Tourism Analytics Dashboard</h1>
        <p>Predict Tourism Performance Index (TPI) &amp; discover which cluster your state belongs to</p>
    </div>
""", unsafe_allow_html=True)

# ── Sidebar — User Inputs ──────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Input Parameters")
    st.markdown("Adjust the sliders to reflect your state's tourism profile.")

    seasonality = st.slider(
        "Seasonality Index",
        min_value=-100.0,
        max_value=252.0,
        value=0.0,
        step=0.5,
        help="Measures how evenly tourists are distributed across months. "
             "Negative = off-season dominant; high positive = strong seasonal spikes.",
    )

    revenue = st.slider(
        "Revenue Per Tourist (INR)",
        min_value=0.0,
        max_value=250.0,
        value=153.0,
        step=0.5,
        help="Average revenue generated per tourist in INR (normalised scale).",
    )

    foreign_rev = st.slider(
        "Foreign Revenue %",
        min_value=0.0,
        max_value=90.0,
        value=19.4,
        step=0.1,
        help="Percentage of total tourism revenue contributed by foreign tourists.",
    )

    growth = st.slider(
        "Growth % (Approx.)",
        min_value=-5.0,
        max_value=15.0,
        value=4.4,
        step=0.1,
        help="Year-on-year approximate tourist arrival growth rate.",
    )

    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("R² Score",  f"{r2:.4f}")
    st.metric("MAE",       f"{mae:.4f}")
    st.markdown("*Trained on the ABA Final Project dataset.*")

# ── Compile user input dict ────────────────────
user_input = {
    "Seasonality_Index":        seasonality,
    "Revenue_Per_Tourist_INR":  revenue,
    "Foreign_Revenue_Percent":  foreign_rev,
    "Growth % (Approx.)":       growth,
}

# ── Run prediction (auto on load, or button click) ─
tpi_score   = predict_tpi(reg, user_input)
cluster_id  = predict_cluster(kmeans, scaler, user_input)
label, interpretation = resolve_cluster_label(cluster_id)
color, card_cls, badge_cls, icon = CLUSTER_COLORS[cluster_id]

# ─────────────────────────────────────────────
# Main Layout — Two Columns
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.05], gap="large")

# ── LEFT: Gauge + Result Cards ─────────────────
with col_left:
    st.markdown('<div class="section-title">📈 Prediction Results</div>', unsafe_allow_html=True)

    # Gauge
    st.plotly_chart(gauge_chart(tpi_score), use_container_width=True, config={"displayModeBar": False})

    # TPI Metric row
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted TPI", f"{tpi_score:.4f}")
    m2.metric("Cluster",       str(cluster_id))
    m3.metric("Percentile",    f"{int(np.mean(clean_df['TPI'] < tpi_score) * 100)}th")

    # Cluster result card
    tpi_pct_label = (
        "High" if tpi_score >= 0.65 else
        "Moderate" if tpi_score >= 0.35 else
        "Low"
    )

    st.markdown(f"""
        <div class="result-card {card_cls}">
            <h3 style="margin:0; color:#1a3c5e;">{icon} Cluster {cluster_id} Assignment</h3>
            <span class="cluster-badge {badge_cls}">{label}</span>
            <p style="margin-top:0.9rem; color:#555; line-height:1.6;">
                {interpretation}
            </p>
            <p style="margin:0; color:#888; font-size:0.88rem;">
                TPI Level: <strong style="color:{color};">{tpi_pct_label}</strong>
                &nbsp;|&nbsp; Score: <strong>{tpi_score:.4f}</strong>
                &nbsp;|&nbsp; Dataset avg: <strong>0.4734</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Interpretation summary
    delta = tpi_score - 0.473
    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    delta_color = "#27ae60" if delta >= 0 else "#e74c3c"

    st.markdown(f"""
        <div style="background:white; border-radius:12px; padding:1rem 1.4rem;
                    box-shadow:0 2px 10px rgba(0,0,0,0.06); margin-top:0.5rem;">
            <p style="margin:0; font-size:0.95rem; color:#333;">
                💡 <strong>Interpretation:</strong> This state falls under the
                <strong style="color:{color};">{label}</strong> category with a TPI of
                <strong>{tpi_score:.4f}</strong> —
                <span style="color:{delta_color}; font-weight:600;">{delta_str}</span>
                vs. the national dataset average (0.4734).
            </p>
        </div>
    """, unsafe_allow_html=True)

# ── RIGHT: Charts ──────────────────────────────
with col_right:
    st.markdown('<div class="section-title">📊 Visual Insights</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Cluster Distribution", "Feature Comparison"])

    with tab1:
        st.plotly_chart(
            cluster_scatter(clean_df, user_input, cluster_id),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Each bubble = one state-month record. Bubble size = Revenue Per Tourist. "
            "Purple star = your input."
        )

    with tab2:
        st.plotly_chart(
            feature_bar_chart(user_input, clean_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption("Your input features (blue) compared to dataset averages (light blue).")

# ─────────────────────────────────────────────
# Bottom Section — Cluster Summary Table
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">🔍 Cluster Profiles (Training Data)</div>', unsafe_allow_html=True)

cluster_summary = (
    clean_df
    .groupby("Cluster")[FEATURES + [TARGET]]
    .mean()
    .round(4)
    .rename(index=CLUSTER_LABELS)
    .rename(columns={"Growth % (Approx.)": "Growth %", "TPI": "Avg TPI"})
)

st.dataframe(
    cluster_summary.style
    .format("{:.4f}")
    .background_gradient(subset=["Avg TPI"], cmap="Blues"),
    use_container_width=True,
)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
    <div style="text-align:center; margin-top:2rem; padding:1rem;
                color:#888; font-size:0.82rem;">
        Tourism Analytics Dashboard &nbsp;|&nbsp;
        Linear Regression + KMeans (k=3) &nbsp;|&nbsp;
        ABA Final Project Dataset &nbsp;|&nbsp;
        Built with Streamlit 🎈
    </div>
""", unsafe_allow_html=True)
