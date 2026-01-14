#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
# This assumes the CSV files are in the SAME folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = BASE_DIR

REQUIRED_COLS = {"timestamp", "actual", "p50"}

# -------------------------------------------------
# DATA LOADING
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
            ((df["actual"] >= df["p05"]) & (df["actual"] <= df["p95"]))
            .mean()
            * 100.0
        )

    if {"p25", "p75"}.issubset(df.columns):
        metrics["pi50_coverage"] = (
            ((df["actual"] >= df["p25"]) & (df["actual"] <= df["p75"]))
            .mean()
            * 100.0
        )

    return metrics

# -------------------------------------------------
# PLOTTING
# -------------------------------------------------
def plot_core(df: pd.DataFrame, location: str, day: str):
    metrics = compute_metrics(df)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.25)

    # --- Time series
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df["timestamp"], df["actual"], label="Actual", linewidth=2)
    ax1.plot(df["timestamp"], df["p50"], label="Predicted (p50)", linewidth=2)

    if {"p05", "p95"}.issubset(df.columns):
        ax1.fill_between(
            df["timestamp"], df["p05"], df["p95"],
            alpha=0.20, label="90% PI (p5–p95)"
        )

    if {"p25", "p75"}.issubset(df.columns):
        ax1.fill_between(
            df["timestamp"], df["p25"], df["p75"],
            alpha=0.25, label="50% PI (p25–p75)"
        )

    ax1.axhline(0, linestyle="--", linewidth=1)
    ax1.set_ylabel("DART ($/MWh)")
    ax1.set_title(f"{location} Backtest – {day}")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower left")

    # --- Metrics box
    lines = [
        "DAILY METRICS",
        "---------------------------",
        f"MAE: ${metrics['mae']:.2f}",
        f"RMSE: ${metrics['rmse']:.2f}",
        f"MAE / RMSE: {metrics['mae_rmse']:.3f}",
        f"Mean Error: ${metrics['mean_error']:.2f}",
        f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%",
    ]

    if "pi90_coverage" in metrics:
        lines.append(f"90% PI Coverage: {metrics['pi90_coverage']:.2f}%")

    if "pi50_coverage" in metrics:
        lines.append(f"50% PI Coverage: {metrics['pi50_coverage']:.2f}%")

    lines.append(f"Hours: {metrics['hours']}")

    ax1.text(
        0.01, 0.02, "\n".join(lines),
        transform=ax1.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.15),
    )

    # --- Error histogram
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(df["error"], bins=30, alpha=0.85)
    ax2.axvline(0, linestyle="--", linewidth=1, label="Zero Error")
    ax2.axvline(metrics["mean_error"], linestyle="--", linewidth=1, label="Mean Error")
    ax2.set_title("Prediction Error Distribution")
    ax2.set_xlabel("Error (Actual − Predicted) $/MWh")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_advanced(df: pd.DataFrame, location: str, day: str, threshold: float):
    df_f = df[df["pred_abs"] > threshold]

    m_all = compute_metrics(df)
    m_f = compute_metrics(df_f) if not df_f.empty else None

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.20)

    # Scatter A
    axA = fig.add_subplot(gs[0, 0])
    axA.scatter(df["p50"], df["actual"], alpha=0.85)
    mn = min(df["p50"].min(), df["actual"].min())
    mx = max(df["p50"].max(), df["actual"].max())
    axA.plot([mn, mx], [mn, mx], linestyle="--")
    axA.axhline(0, linestyle="--")
    axA.axvline(0, linestyle="--")
    axA.set_title("Signal vs Realization")
    axA.set_xlabel("Predicted DART (p50)")
    axA.set_ylabel("Actual DART")
    axA.grid(True, alpha=0.25)

    # Scatter B
    axB = fig.add_subplot(gs[0, 1])
    axB.scatter(df["pred_abs"], df["abs_error"], alpha=0.85)
    axB.set_title("Signal Strength vs Risk")
    axB.set_xlabel("|Predicted DART|")
    axB.set_ylabel("|Error|")
    axB.grid(True, alpha=0.25)

    # Text panel
    axC = fig.add_subplot(gs[1, :])
    axC.axis("off")

    text = [
        f"{location} – Advanced Risk View ({day})",
        "-------------------------------------------",
        f"Filter: |p50| > {threshold:.2f} $/MWh",
        "",
        "All Hours:",
        f"  MAE: ${m_all['mae']:.2f}",
        f"  RMSE: ${m_all['rmse']:.2f}",
        f"  MAE/RMSE: {m_all['mae_rmse']:.3f}",
        f"  Direction Accuracy: {m_all['direction_accuracy']:.2f}%",
        "",
    ]

    if m_f:
        text += [
            "Filtered Hours:",
            f"  Hours: {m_f['hours']}",
            f"  MAE: ${m_f['mae']:.2f}",
            f"  RMSE: ${m_f['rmse']:.2f}",
            f"  MAE/RMSE: {m_f['mae_rmse']:.3f}",
            f"  Direction Accuracy: {m_f['direction_accuracy']:.2f}%",
        ]
    else:
        text.append("Filtered Hours: none")

    axC.text(
        0.01, 0.95, "\n".join(text),
        va="top", ha="left", fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", alpha=0.12),
    )

    plt.tight_layout()
    return fig

# -------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------
st.set_page_config(page_title="FG-GPT Backtest Viewer", layout="wide")
st.title("FG-GPT DART Backtest Viewer")

files = glob.glob(os.path.join(BACKTEST_DIR, "*_backtest_forecasts.csv"))
if not files:
    st.error("No *_backtest_forecasts.csv files found.")
    st.stop()

location_map = {
    os.path.basename(f).replace("_backtest_forecasts.csv", ""): f
    for f in files
}

location = st.selectbox("Select Location", sorted(location_map.keys()))
df = load_csv(location_map[location])

min_day = df["timestamp"].dt.date.min()
max_day = df["timestamp"].dt.date.max()
selected_day = st.date_input("Select Day", value=min_day, min_value=min_day, max_value=max_day)

df_day = filter_day(df, str(selected_day))
if df_day.empty:
    st.warning("No data for selected day.")
    st.stop()

df_day = add_error_columns(df_day)

tab_core, tab_adv = st.tabs(["Core View", "Advanced / Trader View"])

with tab_core:
    fig = plot_core(df_day, location, str(selected_day))
    st.pyplot(fig)
    with st.expander("Show hourly data"):
        st.dataframe(df_day, use_container_width=True)

with tab_adv:
    max_val = max(1.0, float(df_day["pred_abs"].max()))
    threshold = st.slider(
        "Only include hours where |Predicted DART (p50)| > X ($/MWh)",
        min_value=0.0,
        max_value=max_val,
        value=min(5.0, max_val),
        step=0.5,
    )
    fig_adv = plot_advanced(df_day, location, str(selected_day), threshold)
    st.pyplot(fig_adv)

