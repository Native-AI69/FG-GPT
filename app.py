#!/usr/bin/env python
# coding: utf-8

import os
import glob
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# LOGGING (goes to Streamlit Cloud logs)
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="FG-GPT Backtest Viewer", layout="wide")


# -----------------------------
# ACCESS / DISCLAIMER GATE
# -----------------------------
DISCLAIMER = """
**Confidential Demo Notice**

This application, its outputs, and underlying methodology are proprietary to Foresight Grid.

By continuing, you acknowledge and agree that:
- You are not accessing on behalf of a competing organization (including software, analytics, trading, or forecasting competitors).
- You will not share this app, screenshots, data exports, or access credentials with unauthorized parties.
- Access may be logged (timestamp and, when available, authenticated user identity) for security and compliance.
"""

def _get_viewer_identity() -> str:
    """
    Best-effort identity helper. Works if Streamlit viewer auth provides user info.
    Falls back gracefully if not available.
    """
    try:
        user = getattr(st, "experimental_user", None)
        if user:
            if isinstance(user, dict):
                return user.get("email") or user.get("user_name") or "authenticated_user"
            for attr in ("email", "user_name", "name"):
                if hasattr(user, attr) and getattr(user, attr):
                    return str(getattr(user, attr))
    except Exception:
        pass
    return "unknown_user"

def _log_access(event: str) -> None:
    who = _get_viewer_identity()
    ts = datetime.now(timezone.utc).isoformat()
    logging.info("FG-GPT_APP_ACCESS event=%s user=%s ts=%s", event, who, ts)

if "accepted_disclaimer" not in st.session_state:
    st.session_state.accepted_disclaimer = False

with st.expander("Access notice (read and accept)", expanded=not st.session_state.accepted_disclaimer):
    st.markdown(DISCLAIMER)
    accepted = st.checkbox("I accept and wish to continue", value=st.session_state.accepted_disclaimer)
    st.session_state.accepted_disclaimer = accepted

if not st.session_state.accepted_disclaimer:
    st.stop()

_log_access("view")


# -----------------------------
# DATA LOADING
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = BASE_DIR
REQUIRED_COLS = {"timestamp", "actual", "p50"}

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

def filter_day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    day_dt = pd.to_datetime(day).date()
    return df[df["timestamp"].dt.date == day_dt].copy()

def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error"] = df["actual"] - df["p50"]
    df["abs_error"] = df["error"].abs()
    df["direction_correct"] = (np.sign(df["actual"]) == np.sign(df["p50"])).astype(int)
    df["pred_abs"] = df["p50"].abs()
    return df

def compute_metrics(df: pd.DataFrame) -> dict:
    mae = df["abs_error"].mean()
    rmse = np.sqrt((df["error"] ** 2).mean())
    ratio = mae / rmse if rmse != 0 else np.nan

    metrics = {
        "hours": len(df),
        "mae": float(mae),
        "rmse": float(rmse),
        "mae_rmse": float(ratio) if not np.isnan(ratio) else np.nan,
        "mean_error": float(df["error"].mean()),
        "direction_accuracy": float(df["direction_correct"].mean() * 100.0),
    }

    if {"p05", "p95"}.issubset(df.columns):
        metrics["pi90_coverage"] = float(((df["actual"] >= df["p05"]) & (df["actual"] <= df["p95"])).mean() * 100.0)
    if {"p25", "p75"}.issubset(df.columns):
        metrics["pi50_coverage"] = float(((df["actual"] >= df["p25"]) & (df["actual"] <= df["p75"])).mean() * 100.0)

    return metrics


# -----------------------------
# PLOTTING HELPERS (matplotlib)
# -----------------------------
def _style_axes(ax):
    ax.set_facecolor("none")
    ax.grid(True, alpha=0.20)

def _apply_figure_transparency(fig):
    fig.patch.set_alpha(0.0)

def _watermark(fig, text: str):
    fig.text(0.99, 0.01, text, ha="right", va="bottom", fontsize=8, alpha=0.25)

def plot_timeseries(df: pd.DataFrame, location: str, day: str, show_bands: bool = True):
    metrics = compute_metrics(df)

    fig = plt.figure(figsize=(12, 4.2))
    _apply_figure_transparency(fig)
    ax = fig.add_subplot(111)
    _style_axes(ax)

    ax.plot(df["timestamp"], df["actual"], label="Actual", linewidth=2, alpha=0.9)
    ax.plot(df["timestamp"], df["p50"], label="Predicted (p50)", linewidth=2, alpha=0.9)

    if show_bands and {"p05", "p95"}.issubset(df.columns):
        ax.fill_between(df["timestamp"], df["p05"], df["p95"], alpha=0.18, label="90% PI (p05–p95)")
    if show_bands and {"p25", "p75"}.issubset(df.columns):
        ax.fill_between(df["timestamp"], df["p25"], df["p75"], alpha=0.22, label="50% PI (p25–p75)")

    ax.axhline(0, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("DART ($/MWh)")
    ax.set_title(f"{location} Backtest - {day}")

    # legend outside
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    # metrics box outside
    lines = [
        "DAILY METRICS",
        f"MAE: ${metrics['mae']:.2f}",
        f"RMSE: ${metrics['rmse']:.2f}",
        f"Dir Acc: {metrics['direction_accuracy']:.2f}%",
    ]
    if "pi90_coverage" in metrics:
        lines.append(f"90% PI Cov: {metrics['pi90_coverage']:.2f}%")
    if "pi50_coverage" in metrics:
        lines.append(f"50% PI Cov: {metrics['pi50_coverage']:.2f}%")
    lines.append(f"Hours: {metrics['hours']}")

    ax.text(
        1.02, 0.05, "\n".join(lines),
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", alpha=0.12),
        family="monospace"
    )

    fig.tight_layout()
    _watermark(fig, "Foresight Grid | Confidential")
    return fig

def plot_error_histogram(df: pd.DataFrame, error_cols: list[str], bins: int = 30):
    fig = plt.figure(figsize=(12, 3.8))
    _apply_figure_transparency(fig)
    ax = fig.add_subplot(111)
    _style_axes(ax)

    series = []
    labels = []
    for col in error_cols:
        if col not in df.columns:
            continue
        err = (df["actual"] - df[col]).astype(float).dropna()
        series.append(err)
        labels.append(col)

    if not series:
        ax.text(0.5, 0.5, "Selected quantile column(s) not found in data.", ha="center", va="center")
        return fig

    all_err = pd.concat(series, axis=0)
    bin_edges = np.histogram_bin_edges(all_err, bins=bins)

    for err, col in zip(series, labels):
        ax.hist(err, bins=bin_edges, alpha=0.35, label=f"Error vs {col}", edgecolor=None)

    ax.axvline(0, linestyle="--", linewidth=1, alpha=0.8, label="Zero Error")
    mean_err = float((df["actual"] - df["p50"]).mean())
    ax.axvline(mean_err, linestyle="--", linewidth=1, alpha=0.8, label=f"Mean Error (vs p50): {mean_err:.2f}")

    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error (Actual - Selected Quantile) $/MWh")
    ax.set_ylabel("Frequency")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    _watermark(fig, "Foresight Grid | Confidential")
    return fig

def plot_scatter_realization(df: pd.DataFrame):
    fig = plt.figure(figsize=(6.2, 4.6))
    _apply_figure_transparency(fig)
    ax = fig.add_subplot(111)
    _style_axes(ax)

    ax.scatter(df["p50"], df["actual"], alpha=0.75)
    mn = float(min(df["p50"].min(), df["actual"].min()))
    mx = float(max(df["p50"].max(), df["actual"].max()))
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1, alpha=0.7)

    ax.axhline(0, linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(0, linestyle="--", linewidth=1, alpha=0.5)

    ax.set_title("Signal vs Realization")
    ax.set_xlabel("Predicted DART (p50)")
    ax.set_ylabel("Actual DART")
    fig.tight_layout()
    _watermark(fig, "Foresight Grid | Confidential")
    return fig

def plot_scatter_risk(df: pd.DataFrame):
    fig = plt.figure(figsize=(6.2, 4.6))
    _apply_figure_transparency(fig)
    ax = fig.add_subplot(111)
    _style_axes(ax)

    ax.scatter(df["pred_abs"], df["abs_error"], alpha=0.75)
    ax.set_title("Signal Strength vs Risk")
    ax.set_xlabel("|Predicted DART| (p50)")
    ax.set_ylabel("|Error| (vs p50)")
    fig.tight_layout()
    _watermark(fig, "Foresight Grid | Confidential")
    return fig


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("FG-GPT DART Backtest Viewer")

files = glob.glob(os.path.join(BACKTEST_DIR, "*_backtest_forecasts.csv"))
if not files:
    st.error("No *_backtest_forecasts.csv files found next to app.py")
    st.stop()

location_map = {os.path.basename(f).replace("_backtest_forecasts.csv", ""): f for f in files}

with st.sidebar:
    st.header("Controls")

    location = st.selectbox("Location", sorted(location_map.keys()))
    df_all = load_csv(location_map[location])

    min_day = df_all["timestamp"].dt.date.min()
    max_day = df_all["timestamp"].dt.date.max()

    selected_day = st.date_input("Day", value=min_day, min_value=min_day, max_value=max_day)

    st.divider()
    st.subheader("Error distribution")
    available_q = [c for c in ["p05", "p25", "p50", "p75", "p95"] if c in df_all.columns]
    default_q = ["p50"] if "p50" in available_q else available_q[:1]
    selected_q = st.multiselect(
        "Quantile(s) to compare against Actual",
        options=available_q,
        default=default_q,
        help="Error is computed as (Actual - selected quantile). Select one or multiple."
    )
    hist_bins = st.slider("Histogram bins", min_value=10, max_value=100, value=30, step=5)

    st.divider()
    st.subheader("Trader view filter")
    threshold = st.slider(
        "Only include hours where |Predicted DART (p50)| > X ($/MWh)",
        min_value=0.0, max_value=float(max(1.0, df_all["p50"].abs().max())),
        value=5.0, step=0.5
    )

df_day = filter_day(df_all, str(selected_day))
if df_day.empty:
    st.warning("No data for selected day.")
    st.stop()

df_day = add_error_columns(df_day)

tab_core, tab_adv = st.tabs(["Core View", "Advanced / Trader View"])

with tab_core:
    fig_ts = plot_timeseries(df_day, location, str(selected_day))
    st.pyplot(fig_ts, use_container_width=True)

    fig_hist = plot_error_histogram(df_day, selected_q, bins=int(hist_bins))
    st.pyplot(fig_hist, use_container_width=True)

    with st.expander("Show hourly data"):
        st.dataframe(df_day, use_container_width=True)

with tab_adv:
    df_f = df_day[df_day["pred_abs"] > float(threshold)].copy()
    st.caption(f"Filtered hours: {len(df_f)} of {len(df_day)} (|p50| > {threshold:.2f})")

    colA, colB = st.columns(2)
    with colA:
        st.pyplot(plot_scatter_realization(df_f if not df_f.empty else df_day), use_container_width=True)
    with colB:
        st.pyplot(plot_scatter_risk(df_f if not df_f.empty else df_day), use_container_width=True)

    m_all = compute_metrics(df_day)
    m_f = compute_metrics(df_f) if not df_f.empty else None

    st.subheader("Summary metrics (p50-based)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"${m_all['mae']:.2f}")
    c2.metric("RMSE", f"${m_all['rmse']:.2f}")
    c3.metric("Direction Acc", f"{m_all['direction_accuracy']:.2f}%")
    c4.metric("Mean Error", f"${m_all['mean_error']:.2f}")

    if m_f:
        st.markdown("**Filtered (Trader View) metrics**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"${m_f['mae']:.2f}")
        c2.metric("RMSE", f"${m_f['rmse']:.2f}")
        c3.metric("Direction Acc", f"{m_f['direction_accuracy']:.2f}%")
        c4.metric("Mean Error", f"${m_f['mean_error']:.2f}")
