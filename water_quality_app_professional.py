"""
Smart Water Quality & Usage Optimizer
Streamlit dashboard for a hackathon demo.

This version:
- Generates clean hourly demo sensor data.
- Uses deterministic synthetic data so Streamlit reruns are reproducible.
- Trains the quality classifier with a real train/test split.
- Reports actual validation accuracy instead of a hard-coded number.
- Uses Isolation Forest for anomaly detection.
- Detects unusual water-usage days and estimates excess usage.
- Keeps all calculations internally consistent.
- Includes a manual refresh button.
- Avoids unused imports and misleading "real-time" claims.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CONSTANTS
# ============================================================================

FEATURES = [
    "ph",
    "turbidity",
    "temperature",
    "chlorine",
    "bacteria",
    "tds",
]

STATUS_TEXT = {
    0: "SAFE",
    1: "WARNING",
    2: "CONTAMINATED",
}

STATUS_COLOR = {
    0: "green",
    1: "orange",
    2: "red",
}

# Demo thresholds. These are intentionally kept consistent throughout the app.
PH_MIN = 6.5
PH_MAX = 8.5
TURBIDITY_WARNING = 1.5
TURBIDITY_CONTAMINATED = 3.0
BACTERIA_WARNING = 20
BACTERIA_CONTAMINATED = 50
CHLORINE_MIN = 0.2
CHLORINE_WARNING = 0.3
CHLORINE_MAX = 1.0
TDS_WARNING = 500

LEAK_MULTIPLIER = 1.5
COST_PER_LITER = 0.42
CO2_PER_LITER = 0.002


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(37,99,235,.08), transparent 28%),
            radial-gradient(circle at 100% 15%, rgba(14,165,233,.07), transparent 24%),
            #f6f8fc;
    }
    .main .block-container {
        max-width: 1450px;
        padding: 2rem 2.5rem 3rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #172554 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    section[data-testid="stSidebar"] * { color: #f8fafc !important; }
    section[data-testid="stSidebar"] .stCaption { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] [data-testid="stButton"] button {
        background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 10px;
        font-weight: 600;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0369a1 100%);
        border-radius: 22px;
        padding: 30px 34px;
        margin-bottom: 24px;
        box-shadow: 0 18px 45px rgba(15,23,42,.16);
    }
    .hero-eyebrow {
        color: #bfdbfe;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .12em;
        margin-bottom: 8px;
    }
    .hero-title {
        color: white;
        font-size: 2.35rem;
        line-height: 1.15;
        font-weight: 800;
        margin: 0;
    }
    .hero-subtitle { color: #dbeafe; font-size: 1rem; margin-top: 10px; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.92);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px 18px;
        min-height: 112px;
        box-shadow: 0 8px 24px rgba(15,23,42,.06);
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 650 !important; }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 800 !important; }
    .alert-danger, .alert-warning, .alert-success, .demo-note {
        border-radius: 14px !important;
        padding: 16px 18px !important;
        box-shadow: 0 6px 18px rgba(15,23,42,.05);
    }
    .alert-danger { background:#fff1f2; border:1px solid #fecdd3; border-left:5px solid #e11d48; }
    .alert-warning { background:#fffbeb; border:1px solid #fde68a; border-left:5px solid #d97706; }
    .alert-success { background:#f0fdf4; border:1px solid #bbf7d0; border-left:5px solid #16a34a; }
    .demo-note { background:#eff6ff; border:1px solid #bfdbfe; border-left:5px solid #2563eb; }
    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        border: 1px solid #cbd5e1;
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,.88);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        overflow: hidden;
    }
    div[data-testid="stPlotlyChart"] {
        background: rgba(255,255,255,.88);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 8px;
        box-shadow: 0 7px 22px rgba(15,23,42,.05);
    }
    hr { border:none; border-top:1px solid #e2e8f0; margin:1.6rem 0; }
    .footer { text-align:center; color:#64748b; font-size:.82rem; padding:18px 0 5px; }
    @media (max-width:900px) {
        .main .block-container { padding:1.2rem 1rem 2rem; }
        .hero { padding:24px; border-radius:18px; }
        .hero-title { font-size:1.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# DATA GENERATION
# ============================================================================

@st.cache_data
def generate_sensor_data(days: int = 7, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic hourly demo sensor readings."""

    if days < 1:
        raise ValueError("days must be at least 1")

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("h"),
        periods=days * 24,
        freq="h",
    )

    rows = []

    for i, timestamp in enumerate(timestamps):
        day_index = i // 24
        hour = timestamp.hour

        # Normal readings.
        ph = rng.normal(7.2, 0.3)
        turbidity = rng.normal(0.8, 0.3)
        temperature = 22 + 3 * np.sin(hour / 12) + rng.normal(0, 1)
        chlorine = rng.normal(0.6, 0.15)
        bacteria = rng.normal(5, 3)
        tds = rng.normal(420, 50)

        # Simulated contamination event.
        if days >= 3 and day_index == 2 and 13 <= hour <= 16:
            turbidity = rng.uniform(4, 6)
            bacteria = rng.uniform(150, 300)
            chlorine = rng.uniform(0.1, 0.3)

        # Simulated leak event.
        usage = 20.0

        if 8 <= hour <= 9:
            usage += 10

        if 19 <= hour <= 22:
            usage += 15

        if days >= 4 and day_index == 3:
            usage *= 2.5

        rows.append(
            {
                "timestamp": timestamp,
                "ph": float(np.clip(ph, 6.0, 8.5)),
                "turbidity": float(max(0, turbidity)),
                "temperature": float(temperature),
                "chlorine": float(np.clip(chlorine, 0, 2.0)),
                "bacteria": float(max(0, bacteria)),
                "tds": float(max(0, tds)),
                "water_usage": float(usage),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# LABELING + MODEL TRAINING
# ============================================================================

def create_quality_labels(df: pd.DataFrame) -> pd.Series:
    """Create demo labels from the same threshold rules used by the dashboard."""

    labels = np.zeros(len(df), dtype=int)

    contaminated = (
        (df["bacteria"] > BACTERIA_CONTAMINATED)
        | (df["turbidity"] > TURBIDITY_CONTAMINATED)
        | (df["chlorine"] < CHLORINE_MIN)
        | (df["ph"] < PH_MIN)
        | (df["ph"] > PH_MAX)
    )

    warning = (
        (df["bacteria"] > BACTERIA_WARNING)
        | (df["turbidity"] > TURBIDITY_WARNING)
        | (df["chlorine"] < CHLORINE_WARNING)
        | (df["tds"] > TDS_WARNING)
        | (df["ph"] < 6.7)
        | (df["ph"] > 8.2)
    )

    labels[warning] = 1
    labels[contaminated] = 2

    return pd.Series(labels, index=df.index, name="quality")


@st.cache_resource
def train_models(df: pd.DataFrame):
    """Train the demo quality classifier and anomaly detector."""

    X = df[FEATURES].copy()
    y = create_quality_labels(df)

    # A stratified split gives the evaluation a meaningful class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    quality_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    quality_model.fit(X_train, y_train)

    quality_accuracy = accuracy_score(
        y_test,
        quality_model.predict(X_test),
    )

    anomaly_model = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=42,
    )
    anomaly_model.fit(X_train)

    return quality_model, anomaly_model, quality_accuracy


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_line_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    y_title: str,
    thresholds=None,
    height: int = 300,
):
    """Create a consistent Plotly line chart."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[column],
            mode="lines",
            name=title,
            fill="tozeroy",
        )
    )

    if thresholds:
        for value, label, line_color in thresholds:
            fig.add_hline(
                y=value,
                line_dash="dash",
                line_color=line_color,
                annotation_text=label,
            )

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified",
    )

    fig.update_yaxes(title=y_title)

    return fig


def classify_current_reading(latest: pd.Series) -> int:
    """Classify a reading using transparent demo rules."""

    if (
        latest["bacteria"] > BACTERIA_CONTAMINATED
        or latest["turbidity"] > TURBIDITY_CONTAMINATED
        or latest["chlorine"] < CHLORINE_MIN
        or latest["ph"] < PH_MIN
        or latest["ph"] > PH_MAX
    ):
        return 2

    if (
        latest["bacteria"] > BACTERIA_WARNING
        or latest["turbidity"] > TURBIDITY_WARNING
        or latest["chlorine"] < CHLORINE_WARNING
        or latest["tds"] > TDS_WARNING
        or latest["ph"] < 6.7
        or latest["ph"] > 8.2
    ):
        return 1

    return 0


# ============================================================================
# MAIN APP
# ============================================================================

def main():

    # ------------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------------

    st.sidebar.title("⚙️ Dashboard Settings")

    days = st.sidebar.slider(
        "History window",
        min_value=3,
        max_value=14,
        value=7,
        help="Number of days of synthetic hourly sensor data.",
    )

    seed = st.sidebar.number_input(
        "Demo data seed",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        help="Change this to generate a different reproducible dataset.",
    )

    if st.sidebar.button("🔄 Refresh dashboard", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This dashboard uses synthetic data for demonstration. "
        "It is not a substitute for certified laboratory water testing."
    )

    # ------------------------------------------------------------------------
    # Load data + models
    # ------------------------------------------------------------------------

    df = generate_sensor_data(days=days, seed=int(seed))

    try:
        quality_model, anomaly_model, quality_accuracy = train_models(df)
    except ValueError:
        st.error(
            "The generated dataset did not contain enough samples in every "
            "quality class for model training. Try a longer history window "
            "or a different data seed."
        )
        st.stop()

    latest = df.iloc[-1]

    X_latest = latest[FEATURES].to_frame().T

    ml_quality_pred = int(quality_model.predict(X_latest)[0])
    ml_quality_prob = quality_model.predict_proba(X_latest)[0]

    # Use the model's class mapping safely.
    probability_by_class = {
        int(cls): float(prob)
        for cls, prob in zip(quality_model.classes_, ml_quality_prob)
    }

    ml_confidence = float(np.max(ml_quality_prob))

    anomaly_pred = int(anomaly_model.predict(X_latest)[0])

    # Transparent threshold status is shown as the primary status.
    quality_pred = classify_current_reading(latest)

    # ------------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------------

    st.markdown(
        '<div class="hero">'
        '<div class="hero-eyebrow">SMART WATER ANALYTICS • HACKATHON DEMO</div>'
        '<div class="hero-title">💧 Smart Water Quality Monitor</div>'
        '<div class="hero-subtitle">Quality intelligence, anomaly detection and usage insights in one dashboard</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="demo-note">'
        "ℹ️ <b>Hackathon demo:</b> sensor readings are synthetic and generated "
        "locally. ML results demonstrate the workflow rather than certified "
        "real-world prediction accuracy."
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------------

    status = STATUS_TEXT[quality_pred]
    status_color = STATUS_COLOR[quality_pred]

    st.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:1.6rem;
            color:{status_color};
            font-weight:bold;
            padding:10px;
        ">
            {"✅" if quality_pred == 0 else "⚠️" if quality_pred == 1 else "🔴"}
            {status}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Latest reading: {latest['timestamp'].strftime('%d %b %Y, %H:%M')}"
    )

    # ------------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------------

    if quality_pred == 2 or anomaly_pred == -1:
        messages = []

        if quality_pred == 2:
            messages.append("High contamination risk detected from water-quality thresholds.")

        if anomaly_pred == -1:
            messages.append("The ML anomaly detector identified an unusual sensor pattern.")

        st.markdown(
            '<div class="alert-danger"><b>🚨 ALERT DETECTED</b><br>'
            + "<br>".join(messages)
            + "</div>",
            unsafe_allow_html=True,
        )

    elif quality_pred == 1:
        st.markdown(
            '<div class="alert-warning"><b>⚠️ WARNING</b><br>'
            "One or more water-quality readings are outside the normal demo range."
            "</div>",
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            '<div class="alert-success"><b>✅ ALL GOOD</b><br>'
            "Current readings are within the configured demo-safe range."
            "</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------------
    # Current metrics
    # ------------------------------------------------------------------------

    st.markdown("---")
    st.subheader("📊 Current Sensor Readings")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "pH Level",
            f"{latest['ph']:.2f}",
            "Normal" if PH_MIN <= latest["ph"] <= PH_MAX else "Alert",
        )

    with col2:
        st.metric(
            "Turbidity",
            f"{latest['turbidity']:.2f} NTU",
            "Normal" if latest["turbidity"] <= TURBIDITY_WARNING else "High",
        )

    with col3:
        st.metric(
            "Temperature",
            f"{latest['temperature']:.1f} °C",
            "Monitor",
        )

    with col4:
        st.metric(
            "Chlorine",
            f"{latest['chlorine']:.2f} mg/L",
            "Normal"
            if CHLORINE_MIN <= latest["chlorine"] <= CHLORINE_MAX
            else "Alert",
        )

    with col5:
        st.metric(
            "Bacteria",
            f"{latest['bacteria']:.1f} CFU/mL",
            "Normal" if latest["bacteria"] <= BACTERIA_WARNING else "High",
        )

    with col6:
        st.metric(
            "TDS",
            f"{latest['tds']:.0f} mg/L",
            "Normal" if latest["tds"] <= TDS_WARNING else "High",
        )

    # ------------------------------------------------------------------------
    # ML explanation
    # ------------------------------------------------------------------------

    with st.expander("🤖 ML Prediction Details"):
        predicted_name = STATUS_TEXT.get(ml_quality_pred, "UNKNOWN")

        st.write(
            f"**Random Forest prediction:** {predicted_name}"
        )

        st.write(
            f"**Model confidence:** {ml_confidence:.1%}"
        )

        st.write(
            f"**Threshold-based dashboard status:** {STATUS_TEXT[quality_pred]}"
        )

        st.write(
            f"**Anomaly detector:** "
            f"{'Anomaly detected' if anomaly_pred == -1 else 'No anomaly detected'}"
        )

        st.write("**Class probabilities:**")

        probability_table = pd.DataFrame(
            {
                "Status": ["Safe", "Warning", "Contaminated"],
                "Probability": [
                    probability_by_class.get(0, 0.0),
                    probability_by_class.get(1, 0.0),
                    probability_by_class.get(2, 0.0),
                ],
            }
        )

        probability_table["Probability"] = (
            probability_table["Probability"] * 100
        ).round(1)

        st.dataframe(
            probability_table,
            width="stretch",
            hide_index=True,
        )

    # ------------------------------------------------------------------------
    # Trends
    # ------------------------------------------------------------------------

    st.markdown("---")
    st.subheader(f"📈 {days}-Day Water Quality Trend")

    col1, col2 = st.columns(2)

    with col1:
        fig = make_line_chart(
            df,
            "ph",
            "pH Level",
            "pH",
            [
                (PH_MIN, "Minimum safe", "red"),
                (PH_MAX, "Maximum safe", "red"),
            ],
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = make_line_chart(
            df,
            "turbidity",
            "Turbidity",
            "NTU",
            [
                (1.0, "Normal threshold", "green"),
                (TURBIDITY_CONTAMINATED, "Alert threshold", "red"),
            ],
        )
        st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        fig = make_line_chart(
            df,
            "bacteria",
            "Bacteria Count",
            "CFU/mL",
            [
                (BACTERIA_WARNING, "Warning threshold", "orange"),
                (BACTERIA_CONTAMINATED, "Contamination threshold", "red"),
            ],
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = make_line_chart(
            df,
            "chlorine",
            "Chlorine Level",
            "mg/L",
            [
                (CHLORINE_MIN, "Minimum", "red"),
                (CHLORINE_MAX, "Upper demo range", "green"),
            ],
        )
        st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------------------
    # Usage + leak detection
    # ------------------------------------------------------------------------

    st.markdown("---")
    st.subheader("💧 Water Usage & Leak Detection")

    usage_df = df.copy()
    usage_df["date"] = usage_df["timestamp"].dt.date

    daily_usage = (
        usage_df.groupby("date", as_index=False)["water_usage"]
        .sum()
        .rename(columns={"water_usage": "liters"})
    )

    avg_usage = float(daily_usage["liters"].mean())
    leak_threshold = avg_usage * LEAK_MULTIPLIER

    daily_usage["excess_liters"] = np.maximum(
        daily_usage["liters"] - avg_usage,
        0,
    )

    daily_usage["possible_leak"] = (
        daily_usage["liters"] > leak_threshold
    )

    fig_usage = go.Figure()

    fig_usage.add_trace(
        go.Bar(
            x=daily_usage["date"],
            y=daily_usage["liters"],
            name="Daily usage",
            text=[
                f"{value:.0f} L"
                for value in daily_usage["liters"]
            ],
            textposition="outside",
        )
    )

    fig_usage.add_hline(
        y=avg_usage,
        line_dash="dash",
        annotation_text=f"Average: {avg_usage:.0f} L",
    )

    fig_usage.add_hline(
        y=leak_threshold,
        line_dash="dot",
        annotation_text=f"Leak threshold: {leak_threshold:.0f} L",
    )

    fig_usage.update_layout(
        title=f"{days}-Day Water Usage",
        height=400,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_title="Date",
        yaxis_title="Liters",
    )

    st.plotly_chart(fig_usage, width="stretch")

    high_usage_days = daily_usage[
        daily_usage["possible_leak"]
    ].copy()

    excess_usage = float(high_usage_days["excess_liters"].sum())

    if not high_usage_days.empty:
        leak_dates = ", ".join(
            str(date)
            for date in high_usage_days["date"]
        )

        estimated_monthly_savings = excess_usage * COST_PER_LITER

        st.markdown(
            f"""
            <div class="alert-warning">
            <b>💧 Possible Leak / Usage Spike</b><br>
            Unusually high usage detected on: {leak_dates}<br>
            Estimated excess usage in this period: {excess_usage:.0f} L<br>
            Estimated value of excess usage: ₹{estimated_monthly_savings:.0f}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="alert-success"><b>✅ No major usage spike</b><br>'
            "Daily usage is below the configured leak threshold."
            "</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------------

    st.markdown("---")
    st.subheader("💡 Smart Recommendations")

    recommendations = []

    if latest["bacteria"] > BACTERIA_CONTAMINATED:
        recommendations.append(
            (
                "🔴",
                "High bacteria detected",
                "Treat this as a contamination warning and verify the result "
                "with an appropriate water-quality test.",
            )
        )
    elif latest["bacteria"] > BACTERIA_WARNING:
        recommendations.append(
            (
                "🟡",
                "Elevated bacteria",
                "Monitor the reading and consider confirming it with a "
                "validated water-quality test.",
            )
        )

    if latest["turbidity"] > TURBIDITY_CONTAMINATED:
        recommendations.append(
            (
                "🔴",
                "High turbidity",
                "Investigate the source of suspended particles and check "
                "the filtration system.",
            )
        )
    elif latest["turbidity"] > TURBIDITY_WARNING:
        recommendations.append(
            (
                "🟡",
                "Elevated turbidity",
                "Inspect filtration and monitor the trend.",
            )
        )

    if latest["chlorine"] < CHLORINE_MIN:
        recommendations.append(
            (
                "🔴",
                "Very low chlorine",
                "Check the disinfection system and verify the reading before "
                "making treatment changes.",
            )
        )
    elif latest["chlorine"] < CHLORINE_WARNING:
        recommendations.append(
            (
                "🟡",
                "Low chlorine",
                "Monitor the disinfection level and verify the sensor reading.",
            )
        )

    if latest["tds"] > TDS_WARNING:
        recommendations.append(
            (
                "🟡",
                "High TDS",
                "Inspect the water source and consider further water-quality testing.",
            )
        )

    if not high_usage_days.empty:
        recommendations.append(
            (
                "💧",
                "Possible leak",
                "Check faucets, toilets, pipes and other high-use points "
                "for unusual water consumption.",
            )
        )

    if not recommendations:
        recommendations.append(
            (
                "✅",
                "All systems normal",
                "Current readings are within the configured demo range.",
            )
        )

    for icon, title, description in recommendations:
        st.info(f"{icon} **{title}**\n\n{description}")

    # ------------------------------------------------------------------------
    # Impact summary
    # ------------------------------------------------------------------------

    st.markdown("---")
    st.subheader("📊 Impact Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💧 Excess Usage",
            f"{excess_usage:.0f} L",
            "identified above average",
        )

    with col2:
        estimated_value = excess_usage * COST_PER_LITER
        st.metric(
            "💰 Estimated Value",
            f"₹{estimated_value:.0f}",
            "if excess usage is avoided",
        )

    with col3:
        co2_impact = excess_usage * CO2_PER_LITER
        st.metric(
            "🌍 Potential CO₂ Reduction",
            f"{co2_impact:.2f} kg",
            "if excess treatment is avoided",
        )

    with col4:
        st.metric(
            "🔍 Tests / Readings",
            f"{len(df)}",
            f"hourly readings over {days} days",
        )

    # ------------------------------------------------------------------------
    # Model performance
    # ------------------------------------------------------------------------

    st.markdown("---")
    st.subheader("🔬 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Quality Predictor",
            f"{quality_accuracy:.1%}",
            "validation accuracy",
        )

    with col2:
        st.metric(
            "Anomaly Detector",
            "Isolation Forest",
            "unsupervised model",
        )

    with col3:
        st.metric(
            "Leak Detector",
            "Rule-based",
            "usage-spike detection",
        )

    st.caption(
        "The quality accuracy is calculated from a held-out validation split "
        "of the synthetic dataset. It should not be presented as real-world "
        "model accuracy."
    )

    # ------------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------------

    st.markdown("---")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(
        f"""
        <div class="footer">
            Last generated: {now} &nbsp;•&nbsp; Synthetic demo sensor data &nbsp;•&nbsp; 🟢 Dashboard operational
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
