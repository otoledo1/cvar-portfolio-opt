"""
Generates the figures listed in the outline's Section 10 priority list:
    1. Cumulative wealth curve -- all 8 strategies
    2. Drawdown curve -- same strategies
    3. Return-vs-CVaR frontier
    4. Regime-stratified performance (heatmap, not grouped bars -- with
       only 5 regimes x 8 strategies, a heatmap reads far better than
       40 skinny bars)
    5. Robustness chart -- the sensitivity sweep, one panel per factor,
       base case highlighted (plus the underlying formatted table)
    6. (Optional) Portfolio weight heatmap over time

Reads only from results/ (already-computed backtest outputs) and
results/additional-metrics-analysis/ (already-computed supplementary
metrics) -- does not re-run any backtest.

Usage:
    python generate_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
AMA_DIR = RESULTS_DIR / "additional-metrics-analysis"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.95  # CVaR confidence level used throughout the backtest

# Shared style: fewer chart-junk defaults (no top/right spine, muted
# grid, consistent font sizes) so figures read like paper exhibits
# rather than default-matplotlib output.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.size": 10,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

STRATEGY_COLORS = {
    "equal_weight": "#9e9e9e",
    "cap_weighted": "#b8860b",
    "min_variance": "#8c564b",
    "mean_variance": "#c2569d",
    "risk_parity": "#6a6a6a",
    "momentum": "#e07b1a",
    "historical_cvar": "#1f6fb2",
    "regime_cvar_rho1.0": "#c0392b",
}
STRATEGY_LABELS = {
    "equal_weight": "Equal weight",
    "cap_weighted": "Cap-weighted",
    "min_variance": "Min variance",
    "mean_variance": "Mean-variance",
    "risk_parity": "Risk parity",
    "momentum": "Momentum",
    "historical_cvar": "Historical CVaR (ρ=0)",
    "regime_cvar_rho1.0": "Regime-Aware CVaR (ρ=1)",
}
STRATEGY_SHORT = {
    "equal_weight": "Equal wt",
    "cap_weighted": "Cap-wtd",
    "min_variance": "Min var",
    "mean_variance": "Mean-var",
    "risk_parity": "Risk parity",
    "momentum": "Momentum",
    "historical_cvar": "Hist. CVaR (ρ=0)",
    "regime_cvar_rho1.0": "Regime CVaR (ρ=1)",
}


def load_all_returns():
    baseline = pd.read_csv(RESULTS_DIR / "baseline_returns.csv", index_col=0, parse_dates=True)
    hist_cvar = pd.read_csv(RESULTS_DIR / "historical_cvar_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
    regime_cvar = pd.read_csv(RESULTS_DIR / "regime_cvar_returns_rho1p0.csv", index_col=0, parse_dates=True).iloc[:, 0]

    all_r = baseline.copy()
    all_r["historical_cvar"] = hist_cvar
    all_r["regime_cvar_rho1.0"] = regime_cvar
    return all_r[list(STRATEGY_COLORS.keys())]


def historical_cvar_at_alpha(returns, alpha=ALPHA):
    losses = -returns
    var_threshold = losses.quantile(alpha)
    tail = losses[losses >= var_threshold]
    return tail.mean()


def ann_return(r):
    return (1 + r).prod() ** (12 / len(r)) - 1


# ---------------------------------------------------------------------
# 1. Cumulative wealth curve
# ---------------------------------------------------------------------
def fig_cumulative_wealth(all_r):
    cum = (1 + all_r).cumprod()

    fig, ax = plt.subplots(figsize=(11, 6))
    for col in all_r.columns:
        lw = 2.6 if col in ("historical_cvar", "regime_cvar_rho1.0") else 1.3
        alpha_line = 1.0 if col in ("historical_cvar", "regime_cvar_rho1.0") else 0.85
        ax.plot(cum.index, cum[col], label=STRATEGY_LABELS[col],
                color=STRATEGY_COLORS[col], linewidth=lw, alpha=alpha_line)

    ax.set_yscale("log")
    ax.set_title("Cumulative wealth, all 8 strategies (log scale)")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    ax.grid(alpha=0.5, which="major")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_cumulative_wealth_curve.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# 2. Drawdown curve
# ---------------------------------------------------------------------
def fig_drawdown(all_r):
    cum = (1 + all_r).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()

    fig, ax = plt.subplots(figsize=(11, 6))
    for col in all_r.columns:
        lw = 2.6 if col in ("historical_cvar", "regime_cvar_rho1.0") else 1.3
        alpha_line = 1.0 if col in ("historical_cvar", "regime_cvar_rho1.0") else 0.85
        ax.plot(drawdown.index, drawdown[col], label=STRATEGY_LABELS[col],
                 color=STRATEGY_COLORS[col], linewidth=lw, alpha=alpha_line)

    ax.set_title("Drawdown, all 8 strategies")
    ax.set_ylabel("Drawdown from running peak")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower left", fontsize=8, ncol=2, frameon=False)
    ax.grid(alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_drawdown_curve.png", dpi=200)
    plt.close(fig)

    max_dd = drawdown.min().rename("max_drawdown")
    max_dd.to_csv(FIG_DIR / "02_drawdown_max_by_strategy.csv")


# ---------------------------------------------------------------------
# 3. Return-vs-CVaR frontier -- direct labels instead of a legend box,
#    dashed line through the non-dominated (Pareto) set.
# ---------------------------------------------------------------------
def fig_frontier(all_r):
    rows = []
    for col in all_r.columns:
        r = all_r[col]
        rows.append({
            "strategy": col,
            "ann_return": ann_return(r),
            "realized_cvar_95": historical_cvar_at_alpha(r),
        })
    df = pd.DataFrame(rows).set_index("strategy")
    df.to_csv(FIG_DIR / "03_return_vs_cvar_frontier_data.csv")

    # Pareto frontier: lower CVaR (less tail risk) and higher return is
    # better. A point is non-dominated if no other point has both lower
    # CVaR AND higher-or-equal return.
    pts = df.sort_values("realized_cvar_95")
    frontier = []
    best_return_so_far = -np.inf
    for strat, row in pts.iterrows():
        if row["ann_return"] > best_return_so_far:
            frontier.append(strat)
            best_return_so_far = row["ann_return"]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    frontier_pts = df.loc[frontier].sort_values("realized_cvar_95")
    ax.plot(frontier_pts["realized_cvar_95"], frontier_pts["ann_return"],
            linestyle="--", color="#aaaaaa", linewidth=1.2, zorder=1)

    for col in all_r.columns:
        is_cvar_model = col in ("historical_cvar", "regime_cvar_rho1.0")
        ax.scatter(df.loc[col, "realized_cvar_95"], df.loc[col, "ann_return"],
                   s=150 if is_cvar_model else 95,
                   color=STRATEGY_COLORS[col], zorder=3,
                   edgecolor="white", linewidth=1.0)

        # Direct label next to each point, nudged so labels don't overlap
        x, y = df.loc[col, "realized_cvar_95"], df.loc[col, "ann_return"]
        ax.annotate(STRATEGY_SHORT[col], (x, y), xytext=(6, 4),
                    textcoords="offset points", fontsize=8.5,
                    color=STRATEGY_COLORS[col],
                    fontweight="bold" if is_cvar_model else "normal")

    ax.set_xlabel("Realized monthly CVaR at 95% (mean loss in worst 5% of months)")
    ax.set_ylabel("Annualized return")
    ax.set_title("Return vs. tail-risk (CVaR) frontier")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_return_vs_cvar_frontier.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# 4. Regime-stratified performance -- annotated heatmap. With only 5
#    regimes x 8 strategies, this is coarse categorical data; a heatmap
#    with printed values reads far better than 40 grouped bars.
# ---------------------------------------------------------------------
def fig_regime_stratified():
    pivot_path = AMA_DIR / "regime_stratified_all_strategies_pivot.csv"
    pivot = pd.read_csv(pivot_path, index_col=0)

    regime_order = ["tranquil", "risk_off", "dollar_strength", "rate_shock", "unclassified"]
    regime_order = [r for r in regime_order if r in pivot.columns]
    pivot = pivot[regime_order] * 100  # to percent

    strat_order = [s for s in STRATEGY_COLORS if s in pivot.index]
    pivot = pivot.loc[strat_order]
    row_labels = [STRATEGY_LABELS[s] for s in strat_order]
    col_labels = [r.replace("_", " ").title() for r in regime_order]

    vmax = np.abs(pivot.values).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Highlight the two CVaR model rows with bold labels
    for i, s in enumerate(strat_order):
        if s in ("historical_cvar", "regime_cvar_rho1.0"):
            ax.get_yticklabels()[i].set_fontweight("bold")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "#222222"
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                    fontsize=8.5, color=text_color)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Regime-stratified mean monthly return, all strategies")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Mean monthly return (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_regime_stratified_performance.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# 5. Robustness: one small panel per swept factor (Sharpe ratio vs.
#    parameter value), base case marked -- replaces a bare table with an
#    actual figure, since 4 factors x 2-4 grid points each is a natural
#    small-multiples layout.
# ---------------------------------------------------------------------
def format_robustness_table():
    df = pd.read_csv(RESULTS_DIR / "sensitivity_analysis_summary.csv", index_col=0)
    df.index.name = "configuration"

    pretty = pd.DataFrame({
        "Ann. return": df["ann_return"].map(lambda x: f"{x:.1%}"),
        "Ann. vol": df["ann_vol"].map(lambda x: f"{x:.1%}"),
        "Sharpe": df["sharpe"].map(lambda x: f"{x:.2f}"),
        "Max drawdown": df["max_drawdown"].map(lambda x: f"{x:.1%}"),
        "Avg turnover": df["avg_turnover"].map(lambda x: f"{x:.1%}"),
    }, index=df.index)

    pretty.to_csv(FIG_DIR / "05_robustness_table_formatted.csv")
    with open(FIG_DIR / "05_robustness_table_formatted.md", "w") as f:
        f.write("# Robustness table (Historical CVaR, rho=0)\n\n")
        f.write("Base case: alpha=0.95, lookback=60mo, txn_cost=5bps, turnover_cap=0.20.\n\n")
        f.write(pretty.to_markdown())
        f.write("\n")
    return df, pretty


def fig_robustness_chart(df):
    factor_groups = {
        "CVaR confidence (α)": {
            "rows": ["alpha=0.9", "alpha=0.95", "alpha=0.99"],
            "x": [0.90, 0.95, 0.99],
            "xlabel": "α",
            "base": "alpha=0.95",
        },
        "Lookback window": {
            "rows": ["lookback=36mo", "lookback=60mo"],
            "x": [36, 60],
            "xlabel": "Months",
            "base": "lookback=60mo",
        },
        "Transaction cost": {
            "rows": ["txn_cost=0bps", "txn_cost=5bps", "txn_cost=10bps", "txn_cost=20bps"],
            "x": [0, 5, 10, 20],
            "xlabel": "bps (one-way)",
            "base": "txn_cost=5bps",
        },
        "Turnover cap": {
            "rows": ["turnover_cap=0.1", "turnover_cap=0.2", "turnover_cap=0.3", "turnover_cap=uncapped"],
            "x": [0.10, 0.20, 0.30, 0.45],  # 0.45 used as a plotting stand-in for "uncapped"
            "xlabel": "Cap (fraction of portfolio/month)",
            "base": "turnover_cap=0.2",
        },
    }

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    base_sharpe = df.loc["alpha=0.95", "sharpe"]

    for ax, (title, spec) in zip(axes, factor_groups.items()):
        y = df.loc[spec["rows"], "sharpe"].values
        x = spec["x"]
        ax.plot(x, y, "-o", color="#1f6fb2", markersize=7, linewidth=1.6, zorder=2)

        base_idx = spec["rows"].index(spec["base"])
        ax.scatter([x[base_idx]], [y[base_idx]], s=160, facecolor="none",
                   edgecolor="#c0392b", linewidth=2, zorder=3, label="Base case")

        if spec["xlabel"].startswith("Cap"):
            labels = ["0.10", "0.20", "0.30", "Uncapped"]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8.5)
        else:
            ax.set_xticks(x)

        ax.axhline(base_sharpe, color="#dddddd", linestyle=":", linewidth=1, zorder=1)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(spec["xlabel"], fontsize=9)
        ax.grid(alpha=0.4)

    axes[0].set_ylabel("Sharpe ratio")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Robustness of Historical CVaR (ρ=0) to design choices", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_robustness_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 6. Portfolio weight heatmap (optional, lower priority)
# ---------------------------------------------------------------------
def fig_weight_heatmap():
    w = pd.read_csv(RESULTS_DIR / "historical_cvar_weights.csv", index_col=0, parse_dates=True)
    w_q = w.resample("QE").mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(w_q.T.values, aspect="auto", cmap="cividis", vmin=0, vmax=0.30)
    ax.set_yticks(range(len(w_q.columns)))
    ax.set_yticklabels(w_q.columns, fontsize=9.5)

    # Clean quarter labels (e.g. "2008-Q3") without the earlier strftime hack
    quarter_labels = [f"{d.year}-Q{(d.month - 1) // 3 + 1}" for d in w_q.index]
    n_ticks = 14
    tick_idx = np.linspace(0, len(w_q.index) - 1, n_ticks).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([quarter_labels[i] for i in tick_idx], rotation=45, ha="right", fontsize=8.5)

    ax.set_title("Historical CVaR (ρ=0) portfolio weights over time (quarterly avg)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
    cbar.set_label("Weight")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_weight_heatmap_optional.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    all_r = load_all_returns()

    print("Building 01_cumulative_wealth_curve.png ...")
    fig_cumulative_wealth(all_r)

    print("Building 02_drawdown_curve.png ...")
    fig_drawdown(all_r)

    print("Building 03_return_vs_cvar_frontier.png ...")
    fig_frontier(all_r)

    print("Building 04_regime_stratified_performance.png ...")
    fig_regime_stratified()

    print("Building 05_robustness_table_formatted.{csv,md} + 05_robustness_chart.png ...")
    sens_df, pretty = format_robustness_table()
    fig_robustness_chart(sens_df)
    print(pretty)

    print("Building 06_weight_heatmap_optional.png ...")
    fig_weight_heatmap()

    print(f"\nAll figures saved to {FIG_DIR}/")
