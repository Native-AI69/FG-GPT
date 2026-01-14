import os
import glob
import numpy as np
import pandas as pd
import streamlit as st

# Plotting
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="FG-GPT Backtest Viewer", layout="wide")

# -------------------------------
# CONFIG
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = BASE_DIR

# -------------------------------
# DATA LOADING
# -------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")

def filter_day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    day_dt = pd.to_datetime(day).date()
    return df[df["timestamp"].dt.date == day_dt].copy()

def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "p50" in df.columns:
        df["error"] = df["actual"] - df["p50"]
        df["abs_error"] = df["error"].abs()
        df["direction_correct"] = (np.sign(df["actual"]) == np.sign(df["p50"]))
        df["pred_abs"] = df["p50"].abs()
    return df

def available_quantiles(df: pd.DataFrame) -> list[str]:
    # Common quantile column names we support
    candidates = ["p01","p05","p10","p25","p50","p75","p90","p95","p99"]
    return [c for c in candidates if c in df.columns]

def compute_metrics(df: pd.DataFrame) -> dict:
    metrics = {"hours": len(df)}
    if len(df) == 0:
        return metrics

    if "p50" in df.columns:
        err = df["actual"] - df["p50"]
        mae = float(err.abs().mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        metrics.update(
            mae=mae,
            rmse=rmse,
            mae_rmse=(mae / rmse) if rmse != 0 else np.nan,
            mean_error=float(err.mean()),
            direction_accuracy=float((np.sign(df["actual"]) == np.sign(df["p50"])).mean() * 100.0),
        )

    # Interval coverage metrics (if present)
    if {"p05","p95"}.issubset(df.columns):
        metrics["pi90_coverage"] = float(((df["actual"] >= df["p05"]) & (df["actual"] <= df["p95"])).mean() * 100.0)
    if {"p25","p75"}.issubset(df.columns):
        metrics["pi50_coverage"] = float(((df["actual"] >= df["p25"]) & (df["actual"] <= df["p75"])).mean() * 100.0)
    return metrics

# -------------------------------
# PLOTTING HELPERS (PLOTLY)
# -------------------------------
def _transparent_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=30, t=60, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig

def plot_timeseries(df_day: pd.DataFrame, location: str, day: str) -> go.Figure:
    qcols = available_quantiles(df_day)
    fig = go.Figure()

    # Prediction intervals (draw widest first)
    if {"p05","p95"}.issubset(df_day.columns):
        fig.add_trace(go.Scatter(
            x=df_day["timestamp"], y=df_day["p95"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df_day["timestamp"], y=df_day["p05"],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            name="90% PI (p05-p95)",
            hoverinfo="skip",
            opacity=0.20,
        ))

    if {"p25","p75"}.issubset(df_day.columns):
        fig.add_trace(go.Scatter(
            x=df_day["timestamp"], y=df_day["p75"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df_day["timestamp"], y=df_day["p25"],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            name="50% PI (p25-p75)",
            hoverinfo="skip",
            opacity=0.25,
        ))

    # Core series
    fig.add_trace(go.Scatter(
        x=df_day["timestamp"], y=df_day["actual"],
        mode="lines+markers",
        name="Actual",
        line=dict(width=2),
        marker=dict(size=5),
    ))

    if "p50" in df_day.columns:
        fig.add_trace(go.Scatter(
            x=df_day["timestamp"], y=df_day["p50"],
            mode="lines+markers",
            name="Predicted (p50)",
            line=dict(width=2, dash="solid"),
            marker=dict(size=5),
        ))

    fig.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.7)

    # Hover: unified, with a compact "table-like" readout
    hover_cols = ["actual"] + [c for c in ["p05","p25","p50","p75","p95","p99"] if c in df_day.columns]
    # We'll attach these as customdata on the p50 trace (or actual if p50 missing)
    customdata = np.stack([df_day[c].to_numpy() for c in hover_cols], axis=-1)
    hover_header = "<b>%{x|%Y-%m-%d %H:%M}</b><br>"
    hover_lines = []
    for i, c in enumerate(hover_cols):
        label = c.upper()
        hover_lines.append(f"{label}: %{{customdata[{i}]:.2f}}")
    hovertemplate = hover_header + "<br>".join(hover_lines) + "<extra></extra>"

    attach_trace_name = "Predicted (p50)" if "p50" in df_day.columns else "Actual"
    for tr in fig.data:
        if tr.name == attach_trace_name:
            tr.customdata = customdata
            tr.hovertemplate = hovertemplate
        else:
            tr.hovertemplate = "<extra></extra>"

    fig.update_layout(
        title=f"{location} Backtest - {day}",
        xaxis_title="Time",
        yaxis_title="DART ($/MWh)",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return _transparent_layout(fig)

def plot_error_histogram(
    df_day: pd.DataFrame,
    quantiles: list[str],
    bins: int = 30,
) -> go.Figure:
    # Error is (Actual - QuantileForecast)
    if len(quantiles) == 0:
        quantiles = ["p50"] if "p50" in df_day.columns else []

    fig = go.Figure()
    for q in quantiles:
        if q not in df_day.columns:
            continue
        err = (df_day["actual"] - df_day[q]).dropna()
        if err.empty:
            continue
        fig.add_trace(go.Histogram(
            x=err,
            nbinsx=bins,
            name=f"Error (Actual - {q.upper()})",
            opacity=0.55,
        ))

    # Reference lines based on p50 error (if available)
    if "p50" in df_day.columns:
        err50 = (df_day["actual"] - df_day["p50"]).dropna()
        if not err50.empty:
            fig.add_vline(x=0, line_width=2, line_dash="dash", opacity=0.8)
            fig.add_vline(x=float(err50.mean()), line_width=2, line_dash="dash", opacity=0.8)

    fig.update_layout(
        barmode="overlay",
        title="Prediction Error Distribution",
        xaxis_title="Error ($/MWh)",
        yaxis_title="Frequency",
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return _transparent_layout(fig)

def plot_scatter_signal_vs_realization(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="p50",
        y="actual",
        title="Signal vs Realization",
        labels={"p50": "Predicted DART (p50)", "actual": "Actual DART"},
    )
    # Add y=x reference
    minv = float(min(df["p50"].min(), df["actual"].min()))
    maxv = float(max(df["p50"].max(), df["actual"].max()))
    fig.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode="lines", name="y=x", line=dict(dash="dash")))
    fig.add_vline(x=0, line_dash="dash", opacity=0.6)
    fig.add_hline(y=0, line_dash="dash", opacity=0.6)
    fig.update_layout(hovermode="closest")
    return _transparent_layout(fig)

def plot_scatter_strength_vs_risk(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="pred_abs",
        y="abs_error",
        title="Signal Strength vs Risk",
        labels={"pred_abs": "|Predicted DART|", "abs_error": "|Error|"},
    )
    fig.update_layout(hovermode="closest")
    return _transparent_layout(fig)

def metrics_panel(metrics: dict, title: str = "DAILY METRICS") -> None:
    # Right-side compact metrics (Streamlit native, avoids covering plots)
    st.markdown(f"### {title}")
    if not metrics:
        st.write("No metrics available.")
        return
    cols = st.columns(2)
    def _fmt(x, nd=2):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{x:.{nd}f}"
    with cols[0]:
        st.metric("Hours", int(metrics.get("hours", 0)))
        st.metric("MAE", _fmt(metrics.get("mae", np.nan)))
        st.metric("RMSE", _fmt(metrics.get("rmse", np.nan)))
    with cols[1]:
        st.metric("MAE/RMSE", _fmt(metrics.get("mae_rmse", np.nan), 3))
        st.metric("Mean Error", _fmt(metrics.get("mean_error", np.nan)))
        st.metric("Dir Acc (%)", _fmt(metrics.get("direction_accuracy", np.nan)))

    if "pi90_coverage" in metrics or "pi50_coverage" in metrics:
        st.markdown("#### Interval Coverage")
        c2 = st.columns(2)
        with c2[0]:
            if "pi90_coverage" in metrics:
                st.metric("90% PI Coverage (%)", _fmt(metrics["pi90_coverage"]))
        with c2[1]:
            if "pi50_coverage" in metrics:
                st.metric("50% PI Coverage (%)", _fmt(metrics["pi50_coverage"]))

# -------------------------------
# STREAMLIT APP
# -------------------------------
st.title("FG-GPT DART Backtest Viewer")

files = glob.glob(os.path.join(BACKTEST_DIR, "*_backtest_forecasts.csv"))
if not files:
    st.error("No *_backtest_forecasts.csv files found in the repo root.")
    st.stop()

location_map = {os.path.basename(f).replace("_backtest_forecasts.csv", ""): f for f in files}

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    location = st.selectbox("Location", sorted(location_map.keys()))
    df_all = load_csv(location_map[location])

    min_day = df_all["timestamp"].dt.date.min()
    max_day = df_all["timestamp"].dt.date.max()
    selected_day = st.date_input("Day", value=min_day, min_value=min_day, max_value=max_day)

    # Trader filter (used in advanced view)
    threshold_default = 5.0
    threshold = st.slider(
        "Trader filter: only include hours where |p50| > X ($/MWh)",
        min_value=0.0,
        max_value=float(max(1.0, df_all["p50"].abs().max())) if "p50" in df_all.columns else 50.0,
        value=threshold_default,
        step=0.5,
    )

# Day filter
df_day = filter_day(df_all, str(selected_day))
if df_day.empty:
    st.warning("No data for selected day.")
    st.stop()

df_day = add_error_columns(df_day)

tab_core, tab_adv = st.tabs(["Core View", "Advanced / Trader View"])

# -------------------------------
# CORE VIEW
# -------------------------------
with tab_core:
    # Split layout: plots left, metrics right
    left, right = st.columns([3, 1], gap="large")

    with left:
        st.subheader("Curves")
        fig_ts = plot_timeseries(df_day, location, str(selected_day))
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("Error Distribution")
        qcols = available_quantiles(df_day)
        default_q = ["p50"] if "p50" in qcols else (qcols[:1] if qcols else [])
        selected_q = st.multiselect(
            "Choose quantile(s) to plot error distribution for",
            options=qcols,
            default=default_q,
        )
        bins = st.slider("Histogram bins", min_value=10, max_value=120, value=40, step=5)
        fig_hist = plot_error_histogram(df_day, selected_q, bins=bins)
        st.plotly_chart(fig_hist, use_container_width=True)

        with st.expander("Show hourly data"):
            st.dataframe(df_day, use_container_width=True)

    with right:
        metrics_panel(compute_metrics(df_day), title="DAILY METRICS")

# -------------------------------
# ADVANCED / TRADER VIEW
# -------------------------------
with tab_adv:
    if "p50" not in df_day.columns:
        st.error("Advanced view requires a 'p50' column.")
        st.stop()

    df_f = df_day[df_day["pred_abs"] > threshold].copy()
    m_all = compute_metrics(df_day)
    m_f = compute_metrics(df_f) if not df_f.empty else {"hours": 0}

    c1, c2 = st.columns([3, 1], gap="large")
    with c1:
        st.subheader("Scatter: Signal vs Realization")
        st.plotly_chart(plot_scatter_signal_vs_realization(df_day), use_container_width=True)

        st.subheader("Scatter: Signal Strength vs Risk")
        st.plotly_chart(plot_scatter_strength_vs_risk(df_day), use_container_width=True)

        st.subheader("Filtered subset (|p50| > threshold)")
        if df_f.empty:
            st.info("No hours match the current trader filter.")
        else:
            st.plotly_chart(plot_scatter_signal_vs_realization(df_f), use_container_width=True)

    with c2:
        metrics_panel(m_all, title="ALL HOURS")
        st.divider()
        metrics_panel(m_f, title="FILTERED HOURS")
