"""
streamlit_app/app.py
─────────────────────
Interactive ER Triage Simulation UI.

Run with:  streamlit run streamlit_app/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from predict import risk_tier
from prioritize import prioritize_queue

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ER Triage AI",
    page_icon="🏥",
    layout="wide",
)

TIER_COLORS = {
    "CRITICAL": "#d62728",
    "HIGH":     "#ff7f0e",
    "MODERATE": "#ffd700",
    "LOW":      "#2ca02c",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Simulation Settings")
    n_patients    = st.slider("Number of patients", 5, 50, 20)
    seed          = st.number_input("Random seed", value=42, step=1)
    show_baseline = st.checkbox("Show FIFO baseline comparison", value=True)
    st.markdown("---")
    st.markdown(
        "**⚠️ Research prototype only.**  "
        "Not validated for clinical use."
    )

# ── Generate synthetic queue ──────────────────────────────────────────────────
@st.cache_data
def generate_queue(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scores = np.concatenate([
        rng.beta(2, 8, int(n * 0.55)),
        rng.beta(4, 6, int(n * 0.25)),
        rng.beta(6, 4, int(n * 0.12)),
        rng.beta(8, 2, max(n - int(n*0.92), 1)),
    ])[:n]
    rng.shuffle(scores)

    ages         = rng.integers(18, 90, size=n)
    wait_minutes = np.round(rng.exponential(40, size=n), 1)

    return pd.DataFrame({
        "patient_id":        [f"PT-{i:04d}" for i in range(1, n + 1)],
        "age":               ages,
        "risk_score":        np.round(scores, 4),
        "risk_tier":         [risk_tier(s) for s in scores],
        "wait_time_minutes": wait_minutes,
    })


raw_queue = generate_queue(n_patients, seed)
ai_queue  = prioritize_queue(raw_queue)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 ER Triage AI — Queue Dashboard")
st.markdown(
    "Patients ranked by predicted short-term deterioration risk. "
    "Higher risk = higher priority."
)

# ── KPI cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
tier_counts = ai_queue["risk_tier"].value_counts()

col1.metric("🔴 CRITICAL", tier_counts.get("CRITICAL", 0))
col2.metric("🟠 HIGH",     tier_counts.get("HIGH", 0))
col3.metric("🟡 MODERATE", tier_counts.get("MODERATE", 0))
col4.metric("🟢 LOW",      tier_counts.get("LOW", 0))

st.markdown("---")

# ── AI queue table ────────────────────────────────────────────────────────────
st.subheader("🤖 AI-Prioritized Queue")

display_cols = ["priority_rank", "patient_id", "age", "risk_score",
                "priority_label", "wait_time_minutes"]
display_cols = [c for c in display_cols if c in ai_queue.columns]

styled = ai_queue[display_cols].rename(columns={
    "priority_rank":    "Rank",
    "patient_id":       "Patient",
    "age":              "Age",
    "risk_score":       "Risk Score",
    "priority_label":   "Status",
    "wait_time_minutes":"Wait (min)",
})


def color_row(row):
    tier_map = {"🔴": "CRITICAL", "🟠": "HIGH", "🟡": "MODERATE", "🟢": "LOW"}
    tier = next(
        (t for prefix, t in tier_map.items() if str(row["Status"]).startswith(prefix)),
        "LOW"
    )
    alpha = {"CRITICAL": "0.20", "HIGH": "0.12", "MODERATE": "0.07", "LOW": "0.03"}
    color = TIER_COLORS.get(tier, "#ffffff")
    return [f"background-color: {color}{alpha.get(tier,'0')}"] * len(row)


st.dataframe(
    styled.style.apply(color_row, axis=1).format({"Risk Score": "{:.3f}"}),
    use_container_width=True,
    height=420,
)

# ── Risk score distribution ───────────────────────────────────────────────────
st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Risk Score Distribution")
    fig = px.histogram(
        ai_queue, x="risk_score", nbins=20,
        color="risk_tier",
        color_discrete_map=TIER_COLORS,
        labels={"risk_score": "Risk Score", "risk_tier": "Tier"},
        template="plotly_white",
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Tier Breakdown")
    tier_df = ai_queue["risk_tier"].value_counts().reset_index()
    tier_df.columns = ["Tier", "Count"]
    fig2 = px.pie(
        tier_df, names="Tier", values="Count",
        color="Tier", color_discrete_map=TIER_COLORS,
        hole=0.45,
        template="plotly_white",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── FIFO vs AI comparison ─────────────────────────────────────────────────────
if show_baseline:
    st.markdown("---")
    st.subheader("📊 AI vs FIFO — Time-to-Treatment for High-Risk Patients")

    fifo_order = raw_queue.copy().reset_index(drop=True)
    fifo_order["queue_position_fifo"] = fifo_order.index + 1
    fifo_order["ttt_fifo"] = fifo_order["queue_position_fifo"] * 15

    ai_order = ai_queue.copy()
    ai_order["ttt_ai"] = ai_order["priority_rank"] * 15

    merged = fifo_order[["patient_id", "risk_tier", "ttt_fifo"]].merge(
        ai_order[["patient_id", "ttt_ai"]], on="patient_id"
    )
    high_risk = merged[merged["risk_tier"].isin(["CRITICAL", "HIGH"])]

    fig3 = go.Figure()
    fig3.add_trace(go.Box(y=high_risk["ttt_fifo"], name="FIFO",
                          marker_color="#7f7f7f", boxmean=True))
    fig3.add_trace(go.Box(y=high_risk["ttt_ai"], name="AI-Assisted",
                          marker_color="#1f77b4", boxmean=True))
    fig3.update_layout(
        yaxis_title="Minutes to Treatment",
        template="plotly_white",
        showlegend=True,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Summary stats
    c1, c2, c3 = st.columns(3)
    median_fifo = high_risk["ttt_fifo"].median()
    median_ai   = high_risk["ttt_ai"].median()
    reduction   = round(100 * (median_fifo - median_ai) / max(median_fifo, 1), 1)
    c1.metric("Median TTT (FIFO)",       f"{median_fifo:.0f} min")
    c2.metric("Median TTT (AI)",         f"{median_ai:.0f} min")
    c3.metric("Reduction for high-risk", f"{reduction}%",
              delta=f"-{median_fifo - median_ai:.0f} min", delta_color="inverse")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "ER Triage AI Prototype · Built with MIMIC-IV · "
    "For research purposes only · Not for clinical use"
)
