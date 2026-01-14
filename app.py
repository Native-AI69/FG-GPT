import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="FG-GPT Backtest Viewer", layout="wide")

# -------------------------------
# DISCLAIMER (simple deterrent gate)
# -------------------------------
with st.expander("Disclosure and Usage Terms", expanded=True):
    st.markdown(
        """
**Confidential Demo - FG-GPT Backtest Viewer**

By accessing this application, you acknowledge and agree that:

- The content is confidential and intended only for authorized recipients.
- You will not share the app link, credentials, screenshots, outputs, or derived insights with third parties.
- You are not accessing this app on behalf of a competitor, software developer, or reverse-engineering effort.
- Access may be monitored and violations may result in loss of access and potential legal action.

If you do not agree, please close this page.
        """
    )
    agree = st.checkbox("I agree and acknowledge the terms above.")

if not agree:
    st.stop()

# -------------------------------
# CONFIG
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = BASE_DIR

# expected columns
REQUIRED_COLS = {"timestamp", "actual", "p50"}

# -------------------------------
# DATA LOADING
# -------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.sort_values("timestamp")
    return df

def filter_day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    day_dt = pd.to_datetime(day).date()
    return df[df["timestamp"].dt.date == day_dt].copy()

def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error_p50"] = df["actual"] - df["p50"]
    df["abs_error_p50"] = df["error_p50"].abs()
    df["direction_correct_p50"] = (np.sign(df["actual"]) == np.sign(df["p50"]))
    df["pred_abs_p50"] = df["p50"].abs()

    # optional quantiles
    if "p05" in df.columns:
        df["error_p05"] = df["actual"] - df["p05"]
    if "p95" in df.columns:
        df["error_p95"] = df["actual"] - df["p95"]
    if "p25" in df.columns:
        df["error_p25"] = df["actual"] - df["p25"]
    if "p75" in df.columns:
        df["error_p75"] = df["actual"] - df["p75"]

    return df

def compute_metrics(df: pd.DataFrame) -> dict:
    err = df["error_p50"]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    ratio = mae / rmse if rmse != 0 else np.nan

    metrics = {
        "hours": int(len(df)),
        "mae": float(mae),
        "rmse": float(rmse),
        "mae_rmse": float(ratio),
        "mean_error": float(err.mean()),
        "direction_accuracy": float(df["direction_correct_p50"].mean() * 100.0),
    }

    # coverage (optional)
    if {"p05", "p95"}.issubset(df.columns):
        metrics["pi90_coverage"] = float(((df["actual"] >= df["p05"]) & (df["actual"] <= df["p95"])).mean() * 100.0)
    if {"p25", "p75"}.issubset(df.columns):
        metrics["pi50_coverage"] = float(((df["actual"] >= df["p25"]) & (df["actual"] <= df["p75"])).mean() * 100.0)

    return metrics

# -------------------------------
# PLOTTING HELPERS (transparent, legend outside)
# -------------------------------
def _transparent_fig_ax(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    return fig, ax

def plot_timeseries(df: pd.DataFrame, location: str, day: str):
    metrics = compute_metrics(df)

    fig, ax = _transparent_fig_ax(figsize=(10, 4))

    # core series
    ax.plot(df["timestamp"], df["actual"], label="Actual", linewidth=2)
    ax.plot(df["timestamp"], df["p50"], label="Predicted (p50)", linewidth=2)

    # quantile bands (optional)
    if {"p05", "p95"}.issubset(df.columns):
        ax.fill_between(df["timestamp"], df["p05"], df["p95"], alpha=0.18, label="90% PI (p05-p95)")
    if {"p25", "p75"}.issubset(df.columns):
        ax.fill_between(df["timestamp"], df["p25"], df["p75"], alpha=0.18, label="50% PI (p25-p75)")

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_ylabel("DART ($/MWh)")
    ax.set_title(f"{location} Backtest - {day}")
    ax.grid(True, alpha=0.25)

    # legend OUTSIDE plot
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()

    return fig, metrics

def plot_error_histogram(df: pd.DataFrame, quantile_choice: str):
    """
    quantile_choice:
      - "p50"
      - "p05" (if exists)
      - "p95" (if exists)
      - "All available"
    """
    fig, ax = _transparent_fig_ax(figsize=(10, 4))

    series = []
    labels = []

    if quantile_choice == "p50":
        series = [df["error_p50"]]
        labels = ["Error vs p50"]
    elif quantile_choice == "p05" and "error_p05" in df.columns:
        series = [df["error_p05"]]
        labels = ["Error vs p05"]
    elif quantile_choice == "p95" and "error_p95" in df.columns:
        series = [df["error_p95"]]
        labels = ["Error vs p95"]
    elif quantile_choice == "All available":
        # include whichever exists
        series = [df["error_p50"]]
        labels = ["Error vs p50"]
        if "error_p05" in df.columns:
            series.append(df["error_p05"])
            labels.append("Error vs p05")
        if "error_p95" in df.columns:
            series.append(df["error_p95"])
            labels.append("Error vs p95")
    else:
        # fallback
        series = [df["error_p50"]]
        labels = ["Error vs p50"]

    # histogram overlays
    bins = 30
    for s, lab in zip(series, labels):
        ax.hist(s, bins=bins, alpha=0.35, label=lab)

    # vertical reference lines (requested)
    ax.axvline(0, linestyle="--", linewidth=2, color="red", label="Zero Error")
    # mean based on p50, so it stays consistent
    mean_err = float(df["error_p50"].mean())
    ax.axvline(mean_err, linestyle="--", linewidth=2, color="green", label=f"Mean Error (vs p50): {mean_err:.2f}")

    ax.set_title("Prediction Error Distribution")
    ax.set_xlabel("Error (Actual - Selected Quantile) $/MWh")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.25)

    # legend outside
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()

    return fig

def plot_advanced_scatter(df: pd.DataFrame):
    # Scatter A: Pred p50 vs Actual
    fig1, ax1 = _transparent_fig_ax(figsize=(10, 4))
    ax1.scatter(df["p50"], df["actual"], alpha=0.8)
    mn = float(min(df["p50"].min(), df["actual"].min()))
    mx = float(max(df["p50"].max(), df["actual"].max()))
    ax1.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)
    ax1.axvline(0, linestyle="--", linewidth=1)
    ax1.axhline(0, linestyle="--", linewidth=1)
    ax1.set_title("Signal vs Realization")
    ax1.set_xlabel("Predicted DART (p50)")
    ax1.set_ylabel("Actual DART")
    ax1.grid(True, alpha=0.25)
    fig1.tight_layout()

    # Scatter B: |pred| vs |error|
    fig2, ax2 = _transparent_fig_ax(figsize=(10, 4))
    ax2.scatter(df["pred_abs_p50"], df["abs_error_p50"], alpha=0.8)
    ax2.set_title("Signal Strength vs Risk")
    ax2.set_xlabel("|Predicted DART| (p50)")
    ax2.set_ylabel("|Error|")
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()

    return fig1, fig2

# -------------------------------
# STREAMLIT APP
# -------------------------------
st.title("FG-GPT DART Backtest Viewer")

files = glob.glob(os.path.join(BACKTEST_DIR, "*_backtest_forecasts.csv"))
if not files:
    st.error("No *_backtest_forecasts.csv files found next to app.py")
    st.stop()

location_map = {
    os.path.basename(f).replace("_backtest_forecasts.csv", ""): f
    for f in files
}

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    location = st.selectbox("Location", sorted(location_map.keys()))
    df_all = load_csv(location_map[location])

    min_day = df_all["timestamp"].dt.date.min()
    max_day = df_all["timestamp"].dt.date.max()
    selected_day = st.date_input("Day", value=min_day, min_value=min_day, max_value=max_day)

    # trader threshold used in advanced tab
    threshold = st.slider(
        "Trader filter: only include hours where |p50| > X ($/MWh)",
        min_value=0.0,
        max_value=float(max(1.0, df_all["p50"].abs().max())),
        value=5.0,
        step=0.5
    )

df_day = filter_day(df_all, str(selected_day))
if df_day.empty:
    st.warning("No data for selected day.")
    st.stop()

df_day = add_error_columns(df_day)

tab_core, tab_adv = st.tabs(["Core View", "Advanced / Trader View"])

# -------------------------------
# CORE VIEW (separate charts)
# -------------------------------
with tab_core:
    # 1) Time series + metrics (side-by-side, no overlay)
    fig_ts, metrics = plot_timeseries(df_day, location, str(selected_day))

    left, right = st.columns([3, 1], vertical_alignment="top")
    with left:
        st.pyplot(fig_ts, use_container_width=True)
    with right:
        st.markdown("### Daily Metrics")
        st.write(f"**Hours:** {metrics['hours']}")
        st.write(f"**MAE:** {metrics['mae']:.2f}")
        st.write(f"**RMSE:** {metrics['rmse']:.2f}")
        st.write(f"**MAE/RMSE:** {metrics['mae_rmse']:.3f}")
        st.write(f"**Mean Error:** {metrics['mean_error']:.2f}")
        st.write(f"**Direction Acc:** {metrics['direction_accuracy']:.2f}%")
        if "pi90_coverage" in metrics:
            st.write(f"**90% PI Coverage:** {metrics['pi90_coverage']:.2f}%")
        if "pi50_coverage" in metrics:
            st.write(f"**50% PI Coverage:** {metrics['pi50_coverage']:.2f}%")

    st.divider()

    # 2) Error distribution (dropdown)
    choices = ["p50"]
    if "error_p05" in df_day.columns:
        choices.append("p05")
    if "error_p95" in df_day.columns:
        choices.append("p95")
    if len(choices) > 1:
        choices.append("All available")

    q_choice = st.selectbox("Error distribution quantile", choices, index=0)
    fig_hist = plot_error_histogram(df_day, q_choice)
    st.pyplot(fig_hist, use_container_width=True)

    with st.expander("Show hourly data"):
        st.dataframe(df_day, use_container_width=True)

# -------------------------------
# ADVANCED / TRADER VIEW (separate charts)
# -------------------------------
with tab_adv:
    df_f = df_day[df_day["pred_abs_p50"] > float(threshold)].copy()

    c1, c2 = st.columns([1, 1], vertical_alignment="top")
    with c1:
        st.markdown("### Filter summary")
        st.write(f"Threshold: **|p50| > {threshold:.2f} $/MWh**")
        st.write(f"Selected hours: **{len(df_f)} / {len(df_day)}**")

    if df_f.empty:
        st.warning("No hours pass the current threshold. Lower the slider in the sidebar.")
        st.stop()

    fig_sc1, fig_sc2 = plot_advanced_scatter(df_f)

    st.markdown("### Scatter: Signal vs Realization")
    st.pyplot(fig_sc1, use_container_width=True)

    st.markdown("### Scatter: Signal Strength vs Risk")
    st.pyplot(fig_sc2, use_container_width=True)
