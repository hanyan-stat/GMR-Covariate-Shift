"""Plot Figure 1, Figure 2, and Figure 3 from the saved result tables."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

XDG_CACHE_HOME = Path.cwd() / ".cache"
XDG_CACHE_HOME.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

MPLCONFIGDIR = Path.cwd() / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PREFERRED_METHOD_ORDER = [
    "OM-oracle",
    "OM-DDW",
    "DRW",
    "Target-labeled",
    "Source-unadjusted",
]

PALETTE = {
    "OM-oracle": "#4E79A7",
    "OM-DDW": "#59A14F",
    "DRW": "#E07B39",
    "Target-labeled": "#C8A13A",
    "Source-unadjusted": "#8E6C9A",
}

MARKERS = {
    "OM-oracle": "o",
    "OM-DDW": "s",
    "DRW": "^",
    "Target-labeled": "D",
    "Source-unadjusted": "P",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.labelsize": 20,
            "axes.titlesize": 18,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "xtick.labelsize": 17,
            "ytick.labelsize": 16,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 16,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def ordered_methods(methods) -> list[str]:
    seen = set(methods)
    preferred = [method for method in PREFERRED_METHOD_ORDER if method in seen]
    extras = sorted(method for method in seen if method not in PREFERRED_METHOD_ORDER)
    return preferred + extras


def method_color(method: str):
    if method in PALETTE:
        return PALETTE[method]
    fallback = plt.get_cmap("tab10")
    return fallback(abs(hash(method)) % 10)


def method_marker(method: str) -> str:
    if method in MARKERS:
        return MARKERS[method]
    fallback_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    return fallback_markers[abs(hash(method)) % len(fallback_markers)]


def darken_color(color: str, factor: float = 0.72):
    rgb = np.array(to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))


def adaptive_limits(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    lower = float(values.min())
    upper = float(values.max())
    spread = max(upper - lower, 1e-6)
    pad = max(0.01, spread * 0.08)
    return lower - pad, upper + pad


def make_figure1(s1_csv: Path, s2_csv: Path, figure_dir: Path) -> None:
    configure_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    s1 = pd.read_csv(s1_csv)
    s2 = pd.read_csv(s2_csv)
    _plot_error_box_s1(s1, figure_dir / "Figure1_left_panel.png")
    _plot_error_box_s2(s2, figure_dir / "Figure1_right_panel.png")


def make_figure2(s1_csv: Path, figure_dir: Path) -> None:
    configure_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    s1 = pd.read_csv(s1_csv)
    _plot_s1_wilks(s1, figure_dir / "Figure2.png")


def make_figure3(s1_csv: Path, figure_dir: Path) -> None:
    configure_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    s1 = pd.read_csv(s1_csv)
    _plot_coverage_s1(s1, figure_dir / "Figure3_left_panel.png")
    _plot_ci_box_s1(s1, figure_dir / "Figure3_right_panel.png")


def _grouped_boxplot(
    ax,
    df: pd.DataFrame,
    x_col: str,
    x_order: list,
    methods: list[str],
    value_col: str,
    y_label: str,
    x_label: str,
    add_zero_line: bool,
) -> None:
    centers = np.arange(len(x_order), dtype=float)
    span = 0.78 if value_col == "theta1_error" else 0.74
    width = min(0.18 if value_col == "theta1_error" else 0.17, span / max(len(methods), 1))
    offsets = np.linspace(-span / 2 + width / 2, span / 2 - width / 2, len(methods))

    for method_idx, method in enumerate(methods):
        color = method_color(method)
        series_list = [
            df.loc[(df[x_col] == x_value) & (df["method"] == method), value_col].to_numpy()
            for x_value in x_order
        ]
        positions = centers + offsets[method_idx]
        bp = ax.boxplot(
            series_list,
            positions=positions,
            widths=width * 0.94,
            patch_artist=True,
            showfliers=True,
            whis=(5, 95),
            manage_ticks=False,
        )

        for box in bp["boxes"]:
            box.set(facecolor=color, edgecolor="#4A4A4A", linewidth=0.8, alpha=0.92)
        for median in bp["medians"]:
            median.set(color="#202020", linewidth=1.25)
        for whisker in bp["whiskers"]:
            whisker.set(color="#6A6A6A", linewidth=0.9)
        for cap in bp["caps"]:
            cap.set(color="#6A6A6A", linewidth=0.9)
        for flier in bp["fliers"]:
            flier.set(
                marker="o",
                markersize=2.7,
                markerfacecolor=darken_color(color),
                markeredgecolor=darken_color(color, factor=0.55),
                markeredgewidth=0.25,
                alpha=0.8,
            )

        means = [np.mean(series) if len(series) else np.nan for series in series_list]
        ax.scatter(
            positions,
            means,
            marker="D",
            s=18,
            facecolor="#F6F3EE",
            edgecolor=darken_color(color, factor=0.75),
            linewidth=0.6,
            alpha=0.9,
            zorder=4,
        )

    if add_zero_line:
        ax.axhline(0, color="#6E6E6E", linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
        lower, upper = adaptive_limits(df[value_col])
        if x_col == "nP":
            upper = min(2.0, upper)
        ax.set_ylim(lower, upper)
    else:
        ax.set_ylim(0.0, float(df[value_col].max()) * 1.08)

    ax.set_xlim(-0.55, len(x_order) - 0.45)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(x_value) for x_value in x_order])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.label.set_size(22)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=method_color(method), edgecolor="#4A4A4A", linewidth=0.8, label=method)
        for method in methods
    ]
    ax.legend(
        handles=legend_handles,
        ncol=len(methods),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1.3,
        prop={"size": 15},
    )


def _plot_error_box_s1(df: pd.DataFrame, out: Path) -> None:
    methods = ordered_methods(df["method"].unique())
    x_order = sorted(pd.unique(df["nP"]).tolist())[:4]
    plot_df = df.loc[df["method"].isin(methods) & df["nP"].isin(x_order)].copy()
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _grouped_boxplot(ax, plot_df, "nP", x_order, methods, "theta1_error", "Estimation error", "Source sample size", True)
    fig.subplots_adjust(top=0.8, left=0.12, bottom=0.14, right=0.98)
    fig.savefig(out)
    plt.close(fig)


def _plot_error_box_s2(df: pd.DataFrame, out: Path) -> None:
    methods = ordered_methods(df["method"].unique())
    x_order = sorted(pd.unique(df["alpha"]).tolist(), key=float)
    plot_df = df.loc[df["method"].isin(methods) & df["alpha"].isin(x_order)].copy()
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _grouped_boxplot(ax, plot_df, "alpha", x_order, methods, "theta1_error", "Estimation error", r"Overlap index $\alpha_r$", True)
    fig.subplots_adjust(top=0.8, left=0.12, bottom=0.14, right=0.98)
    fig.savefig(out)
    plt.close(fig)


def _coverage_summary(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    return (
        df.groupby([x_col, "method"])["theta1_coverage"]
        .agg(["mean"])
        .reset_index()
        .rename(columns={"mean": "coverage_rate"})
        .sort_values([x_col, "method"])
    )


def _draw_coverage_panel(ax, summary: pd.DataFrame, methods: list[str], x_order: list, x_col: str, x_label: str) -> None:
    centers = np.arange(len(x_order), dtype=float)
    offsets = np.linspace(-0.06, 0.06, len(methods)) if len(methods) > 1 else np.array([0.0])
    y_min = float(summary["coverage_rate"].min())
    y_max = float(summary["coverage_rate"].max())
    lower = max(0.0, np.floor((y_min - 0.04) * 20) / 20)
    upper = min(1.01, np.ceil((max(y_max, 0.95) + 0.02) * 50) / 50)
    label_y = min(upper - 0.008, 0.95 + 0.01 * max(upper - lower, 0.05))

    for offset, method in zip(offsets, methods):
        method_df = summary.loc[summary["method"] == method].set_index(x_col).reindex(x_order).reset_index()
        color = method_color(method)
        ax.plot(
            centers + offset,
            method_df["coverage_rate"].to_numpy(),
            color=color,
            linewidth=2.0,
            linestyle="solid",
            marker=method_marker(method),
            markersize=6.8,
            markerfacecolor=color,
            markeredgecolor="#444444",
            markeredgewidth=0.9,
            alpha=0.95,
            zorder=3,
        )

    ax.axhline(0.95, color="#B22222", linewidth=1.6, linestyle=(0, (6, 3)), zorder=1)
    ax.text(len(x_order) - 1.0, label_y + 0.002, "Nominal 95%", ha="left", va="bottom", fontsize=15, color="#9F1D1D")
    ax.set_xlim(-0.15, len(x_order) - 0.85)
    ax.set_ylim(lower, upper)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(x_value) for x_value in x_order])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Empirical coverage rate")
    ax.xaxis.label.set_size(22)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            color=method_color(method),
            linestyle="solid",
            marker=method_marker(method),
            markersize=6.2,
            linewidth=2.0,
            markerfacecolor=method_color(method),
            markeredgecolor="#444444",
            markeredgewidth=0.9,
            label=method,
        )
        for method in methods
    ]
    ax.legend(
        handles=handles,
        ncol=len(methods),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.3,
        prop={"size": 17},
    )


def _plot_coverage_s1(df: pd.DataFrame, out: Path) -> None:
    methods = [method for method in ordered_methods(df["method"].unique()) if method != "Source-unadjusted"]
    x_order = sorted(pd.unique(df["nP"]).tolist())[:4]
    plot_df = df.loc[df["method"].isin(methods) & df["nP"].isin(x_order)].copy()
    summary = _coverage_summary(plot_df, "nP")
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _draw_coverage_panel(ax, summary, methods, x_order, "nP", "Source sample size")
    fig.subplots_adjust(top=0.8, left=0.13, bottom=0.14, right=0.98)
    fig.savefig(out)
    plt.close(fig)


def _plot_ci_box_s1(df: pd.DataFrame, out: Path) -> None:
    methods = [method for method in ordered_methods(df["method"].unique()) if method != "Source-unadjusted"]
    x_order = sorted(pd.unique(df["nP"]).tolist())[:4]
    plot_df = df.loc[df["method"].isin(methods) & df["nP"].isin(x_order)].copy()
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _grouped_boxplot(ax, plot_df, "nP", x_order, methods, "theta1_ci_length", "CI length", "Source sample size", False)
    fig.subplots_adjust(top=0.8, left=0.12, bottom=0.14, right=0.98)
    fig.savefig(out)
    plt.close(fig)


def _chisq2_quantiles(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    probs = (np.arange(1, n + 1) - 0.5) / n
    return -2.0 * np.log1p(-probs)


def _plot_s1_wilks(df: pd.DataFrame, out: Path) -> None:
    methods = ["OM-oracle", "OM-DDW", "DRW"]
    plot_df = df[(df["method"].isin(methods)) & (df["wilks_stat_p2"].notna()) & (df["wilks_stat_p2"] != "")].copy()
    max_n = int(plot_df["nP"].max())
    plot_df = plot_df[plot_df["nP"] == max_n]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharex=False, sharey=False)
    for ax, method in zip(np.atleast_1d(axes), methods):
        values = plot_df.loc[plot_df["method"] == method, "wilks_stat_p2"].astype(float).to_numpy()
        theoretical = _chisq2_quantiles(len(values))
        observed = np.sort(values)
        max_lim = max(
            float(theoretical.max()) if len(theoretical) else 0.0,
            float(observed.max()) if len(observed) else 0.0,
        ) * 1.03 if len(values) else 1.0
        ax.scatter(
            theoretical,
            observed,
            s=18,
            facecolor=method_color(method),
            edgecolor="#3A3A3A",
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )
        ax.plot([0, max_lim], [0, max_lim], color="#B22222", linewidth=1.2, linestyle=(0, (5, 3)), zorder=2)
        ax.set_xlim(0, max_lim)
        ax.set_ylim(0, max_lim)
        ax.set_title(method, fontsize=22, pad=14, loc="center")
        ax.set_xlabel(r"Quantiles of $\chi^2_2$")
        ax.xaxis.label.set_size(22)
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=16)
        ax.grid(color="#D9D9D9", linewidth=0.6, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel(r"Quantiles of $\hat{R}(\theta_0)$")
    fig.subplots_adjust(top=0.9, wspace=0.28, left=0.08, right=0.98, bottom=0.18)
    fig.savefig(out)
    plt.close(fig)
