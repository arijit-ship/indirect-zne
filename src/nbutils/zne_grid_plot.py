import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Union

def zne_grid(
    data: Dict[str, Any],
    models: List[str],
    plot_titles: List[str],
    plot_colors: List[str],
    exact_solution: float,
    extrapol_target: Union[float, List[float]],
    output_dir: str,
    filename_prefix: str = "zne_grid",
    leg_noise_type: str = None,
    ncols: int = 3,
    figsize: Tuple[float, float] = (9, 6),
    show: bool = True,
    sharex: bool = False,
    sharey: bool = True,
    legend: bool = True,
    global_legend: bool = False,
    legend_loc: str = "lower center",
    # --- Font control ---
    title_fontsize: int = 13,
    label_fontsize: int = 12,
    tick_fontsize: int = 11,
    legend_fontsize: int = 10,
    # --- Custom styling parameters ---
    border_width: float = 1.5,
    marker_size: float = 5,
):
    """
    Create a single grid plot of ZNE results.
    """

    nplots = len(models)
    nrows = (nplots + ncols - 1) // ncols

    plt.rcParams.update({
        "font.size": tick_fontsize,
        "axes.labelsize": label_fontsize,
        "axes.titlesize": title_fontsize,
        "legend.fontsize": legend_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2
    })

    fig, axs = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey
    )
    axs = axs.flatten() if nplots > 1 else [axs]
    handles, labels = None, None

    for i, model in enumerate(models):
        ax = axs[i]
        DATA = data[model]
        title = plot_titles[i] if plot_titles and i < len(plot_titles) else model

        # --- Noisy estimation ---
        ax.errorbar(
            x=DATA["redundant"]["sorted_noise_levels"],
            y=DATA["redundant"]["mean"],
            yerr=DATA["redundant"]["std"],
            fmt="o",
            ecolor=plot_colors[0],
            capsize=4,
            label = f"{leg_noise_type} estimation" if leg_noise_type else "Noisy estimation",
            color=plot_colors[0],
            markersize=marker_size,
            markeredgewidth=0.8,
            elinewidth=1,
        )

        # --- ZNE Extrapolated ---
        ax.errorbar(
            x=np.atleast_1d(extrapol_target),
            y=np.atleast_1d(DATA["zne"]["mean"]),
            yerr=np.atleast_1d(DATA["zne"]["std"]),
            fmt="o",
            ecolor=plot_colors[min(2, len(plot_colors)-1)],
            capsize=4,
            label="Richardson\nZNE",
            color=plot_colors[min(2, len(plot_colors)-1)],
            markersize=marker_size,
            markeredgewidth=0.8,
            elinewidth=1,
        )

        # --- Noise-free estimation ---
        ax.errorbar(
            x=[0],
            y=[DATA["noiseoff"]["mean"]],
            yerr=[DATA["noiseoff"]["std"]],
            fmt="*",
            ecolor=plot_colors[min(6, len(plot_colors)-1)],
            capsize=4,
            label="Noise-free\nestimation",
            color=plot_colors[min(6, len(plot_colors)-1)],
            markersize=marker_size + 2,
            markeredgewidth=1,
            elinewidth=1,
        )

        # --- Exact solution ---
        ax.axhline(
            y=exact_solution,
            color=plot_colors[min(5, len(plot_colors)-1)],
            linestyle="--",
            linewidth=1.5,
            label="Exact solution",
        )

        # --- Outer border (spines) ---
        for spine in ax.spines.values():
            spine.set_linewidth(border_width)
            spine.set_color("black")

        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()

        ax.set_title(title, fontsize=title_fontsize, pad=8)
        ax.set_xlabel(r"Noise level ($\alpha_k \lambda$)", fontsize=label_fontsize)
        if i % ncols == 0:
            ax.set_ylabel("Expectation value", fontsize=label_fontsize)

        ax.grid(linestyle="--", alpha=0.6)
        ax.tick_params(width=1, length=4, direction='inout')

        if legend and not global_legend:
            ax.legend(loc="best", fontsize=legend_fontsize, frameon=False)

    # Hide unused axes
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(w_pad=1.4, h_pad=0.8)
    plt.subplots_adjust(top=0.9, bottom=0.15 if global_legend else 0.1)

    # --- Global legend ---
    if legend:
        if global_legend and handles and labels:
            fig.legend(
                handles, labels,
                loc=legend_loc,
                ncol=4,
                frameon=False,
                fontsize=legend_fontsize,
                handletextpad=0.5,
                columnspacing=1.2,
            )

    # --- Save compiled figure ---
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{filename_prefix}.eps")
    plt.savefig(plot_path, format="eps", bbox_inches="tight")
    print(f"✅ ZNE grid saved to: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
