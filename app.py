#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = BASE_DIR
REQUIRED_COLS = {"timestamp", "actual", "p50"}

# Matplotlib global tweaks for readability
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="FG-GPT Backtest Viewer", layout="wide")
st.title("FG-GPT DART Backtest Viewer")

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.sort_values("timestamp")


def filter_day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    day_dt = pd.to_datetime(day).date()
    return df[df["timestamp"].dt.date == day_dt].copy()


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error"] = df["actual"] - df["p50"]
    df["abs_error"] = df["error"].abs()
    df["direction_correct"] = (np.sign(df["actual"]) == np.sign(df["p50"]))
    df["pred_abs"] = df["p50"].abs()
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    mae = df["abs_error"].mean()
    rmse = np.sqrt((df["error"] ** 2).mean())
    ratio = mae / rmse if rmse != 0 else np.nan

    metrics = {
        "hours": len(df),
        "mae": mae,
        "rmse": rmse,
        "mae_rmse": ratio,
        "mean_error": df["error"].mean(),
        "direction_accuracy": df["direction_correct"].mean() * 100.0,
    }

    if {"p05", "p95"}.issubset(df.columns):
        metrics["pi90_coverage"] = (
            ((df["actual"] >= df["p05"]) & (df["actual"] <= df["p95"])).mean() * 100.0
        )

    if {"p25", "p75"}.issubset(df.columns):
        metrics["pi50_coverage"] = (
            ((df["actual"] >= df["p25"]) & (df["actual"] <= df["p75"])).mean() * 100.0
        )

    return metrics


def _make_transparent(fig, ax):
    # Transparent figure/axes backgrounds, works on dark or white Streamlit themes
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    return fig, ax


# -------------------------------------------------
# CORE PLOTS (SEPARATED)
# -------------------------------------------------
def plot_curves(df: pd.DataFrame, location: str, day: str, show_pi_90: bool, show_pi_50: bool):
    # Smaller, readable figure
    fig, ax = plt.subplots(figsize=(9.5, 4.0))

    # High-contrast colors (work on light/dark)
    c_actual = "#1f77b4"   # blue
    c_pred = "#ff7f0e"     # orange
    c_pi90 = "#1f77b4"
    c_pi50 = "#ff7f0e"

    ax.plot(df["timestamp"], df["actual"], label="Actual", linewidth=2.2, color=c_actual)
    ax.plot(df["timestamp"], df["p50"], label="Predicted (p50)", linewidth=2.2, color=c_pred)

    if show_pi_90 and {"p05", "p95"}.issubset(df.columns):
        ax.fill_between(
            df["timestamp"], df["p05"], df["p95"],
            alpha=0.18, label="90% PI (p5–p95)", color=c_pi90
        )

    if show_pi_50 and {"p25", "p75"}.issubset(df.columns):
        ax.fill_between(
            df["timestamp"], df["p25"], df["p75"],
            alpha=0.18, label="50% PI (p25–p75)", color=c_pi50
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("DART ($/MWh)")
    ax.set_title(f"{location} Backtest - {day}")
    ax.grid(True, alpha=0.25)

    # Legend OUTSIDE, above the plot (won't cover curves)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        framealpha=0.25
    )

    fig.tight_layout()
    _make_transparent(fig, ax)
    return fig


def plot_error_distribution(df: pd.DataFrame, quantile_lines: list[str], bins: int = 25):
    fig, ax = plt.subplots(figsize=(9.5, 3.0))

    ax.hist(df["error"], bins=bins, alpha=0.85)
    ax.axvline(0, linestyle="--", linewidth=1, label="Zero Error")
    ax.axvline(df["error"].mean(), linestyle="--", linewidth=1, label="Mean Error")

    # Quantile lines selection
    q_map = {
        "p05": 0.05,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p95": 0.95,
    }

    if quantile_lines:
        qs = df["error"].quantile([q_map[q] for q in quantile_lines]).to_dict()
        for q_name, q_val in q_map.items():
            if q_name in quantile_lines:
                ax.axvline(qs[q_val], linewidth=1.2, linestyle="-", label=f"{q_name} ({qs[q_val]:.2f})")

    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error (Actual - Predicted) $/MWh")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.25)

    # Legend outside on the right to keep plot clean
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.25)

    fig.tight_layout()
    _make_transparent(fig, ax)
    return fig


# -------------------------------------------------
# ADVANCED / TRADER PLOTS (SEPARATED)
# -------------------------------------------------
def plot_scatter_signal_vs_realization(df: pd.DataFrame, location: str, day: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.8))

    ax.scatter(df["p50"], df["actual"], alpha=0.85)

    mn = min(df["p50"].min(), df["actual"].min())
    mx = max(df["p50"].max(), df["actual"].max())
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_title(f"Signal vs Realization ({location} - {day})")
    ax.set_xlabel("Predicted DART (p50)")
    ax.set_ylabel("Actual DART")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _make_transparent(fig, ax)
    return fig


def plot_scatter_signal_strength_vs_risk(df: pd.DataFrame, location: str, day: str):
    fig, ax = plt.subplots(figsize=(9.5, 3.8))

    ax.scatter(df["pred_abs"], df["abs_error"], alpha=0.85)
    ax.set_title(f"Signal Strength vs Risk ({location} - {day})")
    ax.set_xlabel("|Predicted DART|")
    ax.set_ylabel("|Error|")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    _make_transparent(fig, ax)
    return fig


# -------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------
files = glob.glob(os.path.join(BACKTEST_DIR, "*_backtest_forecasts.csv"))
if not files:
    st.error("No *_backtest_forecasts.csv files found.")
    st.stop()

location_map = {
    os.path.basename(f).replace("_backtest_forecasts.csv", ""): f
    for f in files
}

with st.sidebar:
    st.header("Controls")

    location = st.selectbox("Location", sorted(location_map.keys()))
    df_all = load_csv(location_map[location])

    min_day = df_all["timestamp"].dt.date.min()
    max_day = df_all["timestamp"].dt.date.max()

    selected_day = st.date_input(
        "Day",
        value=min_day,
        min_value=min_day,
        max_value=max_day
    )

    # Trader threshold stays in sidebar (applies to Advanced view)
    max_val = max(1.0, float(df_all["p50"].abs().max()))
    threshold = st.slider(
        "Trader filter: only include hours where |p50| > X ($/MWh)",
        min_value=0.0,
        max_value=max_val,
        value=min(5.0, max_val),
        step=0.5,
    )

df_day = filter_day(df_all, str(selected_day))
if df_day.empty:
    st.warning("No data for selected day.")
    st.stop()

df_day = add_error_columns(df_day)
metrics = compute_metrics(df_day)

tab_core, tab_adv = st.tabs(["Core View", "Advanced / Trader View"])

# ---------------------------
# CORE VIEW
# ---------------------------
with tab_core:
    st.subheader("Curves")
    c1, c2, c3 = st.columns([1.2, 1.2, 2.0])

    with c1:
        show_pi_90 = st.checkbox("Show 90% PI (p5–p95)", value=True)
    with c2:
        show_pi_50 = st.checkbox("Show 50% PI (p25–p75)", value=True)

    # Curves chart
    fig_curves = plot_curves(df_day, location, str(selected_day), show_pi_90, show_pi_50)
    st.pyplot(fig_curves, use_container_width=False)

    # Metrics (not inside the plot)
    with st.expander("Daily Metrics", expanded=True):
        st.write(
            f"- MAE: **${metrics['mae']:.2f}**\n"
            f"- RMSE: **${metrics['rmse']:.2f}**\n"
            f"- MAE/RMSE: **{metrics['mae_rmse']:.3f}**\n"
            f"- Mean Error: **${metrics['mean_error']:.2f}**\n"
            f"- Direction Accuracy: **{metrics['direction_accuracy']:.2f}%**\n"
            f"- Hours: **{metrics['hours']}**"
        )
        if "pi90_coverage" in metrics:
            st.write(f"- 90% PI Coverage: **{metrics['pi90_coverage']:.2f}%**")
        if "pi50_coverage" in metrics:
            st.write(f"- 50% PI Coverage: **{metrics['pi50_coverage']:.2f}%**")

    st.divider()

    st.subheader("Error Distribution")
    q_options = ["All", "p05", "p25", "p50", "p75", "p95"]
    q_choice = st.multiselect(
        "Quantile lines to show",
        options=q_options,
        default=["All"]
    )

    if "All" in q_choice:
        q_lines = ["p05", "p25", "p50", "p75", "p95"]
    else:
        q_lines = [q for q in q_choice if q in {"p05", "p25", "p50", "p75", "p95"}]

    bins = st.slider("Histogram bins", 10, 60, 25, 5)

    fig_hist = plot_error_distribution(df_day, quantile_lines=q_lines, bins=bins)
    st.pyplot(fig_hist, use_container_width=False)

    with st.expander("Show hourly data"):
        st.dataframe(df_day, use_container_width=True)

# ---------------------------
# ADVANCED / TRADER VIEW
# ---------------------------
with tab_adv:
    st.subheader("Trader Filtered View")
    df_f = df_day[df_day["pred_abs"] > threshold].copy()

    # Chart 1: Signal vs Realization
    st.markdown("#### Signal vs Realization")
    fig_a = plot_scatter_signal_vs_realization(df_f if not df_f.empty else df_day, location, str(selected_day))
    st.pyplot(fig_a, use_container_width=False)

    # Chart 2: Signal Strength vs Risk
    st.markdown("#### Signal Strength vs Risk")
    fig_b = plot_scatter_signal_strength_vs_risk(df_f if not df_f.empty else df_day, location, str(selected_day))
    st.pyplot(fig_b, use_container_width=False)

    st.divider()

    # Summary text (kept separate from charts)
    m_all = compute_metrics(df_day)
    if not df_f.empty:
        m_f = compute_metrics(df_f)
    else:
        m_f = None

    with st.expander("Advanced Metrics Summary", expanded=True):
        st.write("**All Hours**")
        st.write(
            f"- MAE: **${m_all['mae']:.2f}**\n"
            f"- RMSE: **${m_all['rmse']:.2f}**\n"
            f"- MAE/RMSE: **{m_all['mae_rmse']:.3f}**\n"
            f"- Direction Accuracy: **{m_all['direction_accuracy']:.2f}%**\n"
            f"- Hours: **{m_all['hours']}**"
        )

        st.write("")
        st.write(f"**Filtered Hours (|p50| > {threshold:.2f})**")
        if m_f:
            st.write(
                f"- MAE: **${m_f['mae']:.2f}**\n"
                f"- RMSE: **${m_f['rmse']:.2f}**\n"
                f"- MAE/RMSE: **{m_f['mae_rmse']:.3f}**\n"
                f"- Direction Accuracy: **{m_f['direction_accuracy']:.2f}%**\n"
                f"- Hours: **{m_f['hours']}**"
            )
        else:
            st.write("- No hours matched the filter.")
