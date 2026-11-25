"""Draw Figure 1 directly from the unrounded quantile analysis JSON."""


def plot_error_quantiles(result, output):
    """Plot the OEL-DW run's raw T-scale estimates and simultaneous EL sets."""
    import math
    import numpy as np
    import matplotlib as mpl
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    rows = sorted(result["rows"], key=lambda row: row["q"])
    levels = np.array([row["q"] for row in rows])
    # Each row contains the full EL set; the ribbon joins its outer endpoints.
    values = np.array(
        [
            [
                row["e"],
                row["src"],
                row["ref"],
                min(a for a, b in row["bounds"]),
                max(b for a, b in row["bounds"]),
            ]
            for row in rows
        ]
    )
    estimate, source, target, lower, upper = (100 * values).T
    dx = np.min(np.diff(levels))
    blue = (0.217, 0.348, 0.751)
    orange = (0.92, 0.489, 0.074)
    shade = (0.8659, 0.8938, 0.97913)
    style = {
        "font.family": "cmr10",
        "mathtext.fontset": "stix",
        "axes.formatter.use_mathtext": True,
        "text.usetex": False,
        "font.size": 10.9,
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
        "lines.scale_dashes": False,
        "path.simplify": False,
    }
    with mpl.rc_context(style):
        fig = Figure(figsize=(5.655, 2.34), facecolor="white")
        FigureCanvasPdf(fig)
        ax = fig.add_axes([0.090, 0.208, 0.890, 0.734])
        ax.fill_between(levels, lower, upper, facecolor=shade, edgecolor="none")
        (oel_line,) = ax.plot(
            levels,
            estimate,
            color=blue,
            linewidth=1.0,
            marker="o",
            markersize=2.7,
            markerfacecolor=blue,
            markeredgecolor=blue,
            markeredgewidth=0.35,
            label="OEL-DW",
        )
        (target_line,) = ax.plot(
            levels,
            target,
            color=orange,
            linewidth=0.9,
            linestyle=(0, (2.99, 1.99)),
            label="Target reference",
        )
        (source_line,) = ax.plot(
            levels,
            source,
            color="0.30",
            linewidth=0.9,
            linestyle=(0, (2.99, 1.00, 0.90, 1.00)),
            label="Source only",
        )
        ax.set_xlim(levels[0] - 0.4 * dx, levels[-1] + 0.4 * dx)
        ax.set_ylim(0, math.ceil(upper.max()))
        ax.set_xticks(levels[::2], [f"{x:.1f}" for x in levels[::2]])
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.set_xlabel("Quantile level", labelpad=3)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_linewidth(0.4)
        ax.tick_params(direction="out", length=3, width=0.4, pad=3)
        for x, y, marker in [(1, 0, ">"), (0, 1, "^")]:
            ax.plot(
                x,
                y,
                marker=marker,
                markersize=3.2,
                markeredgewidth=0,
                color="black",
                transform=ax.transAxes,
                clip_on=False,
            )
        ribbon = Patch(facecolor=shade, edgecolor="none", label="Simultaneous 95% band")
        ax.legend(
            handles=[oel_line, target_line, source_line, ribbon],
            loc="upper left",
            bbox_to_anchor=(0.035, 0.975),
            frameon=False,
            borderpad=0,
            borderaxespad=0,
            handlelength=1.9,
            handletextpad=0.45,
            labelspacing=0.22,
            fontsize=10.6,
        )
        fig.savefig(
            output,
            format="pdf",
            metadata={
                "Title": "GAMA quantiles of normalized absolute prediction error",
                "Subject": "OEL-DW simultaneous 95% profile empirical likelihood band",
                "CreationDate": None,
                "ModDate": None,
            },
        )


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    plot_error_quantiles(json.loads(args.result.read_text()), args.pdf)
