from typing import Dict, List, Tuple, Any, Union
import matplotlib.pyplot as plt
import numpy as np
import os

def compile_zne_subplots(
    data: Dict[str, Any],
    models: List[str],
    plot_titles: Dict[str, str],
    plot_colors: List[str],
    exact_solution: float,
    extrapol_target: Union[float, List[float]],
    output_dir: str,
    filename_prefix: str = "compiled_zne",
    ncols: int = 3,
    figsize: Tuple[float, float] = (9, 6),
    show: bool = True,
):
    """
    Create a compiled subplot figure showing noisy, extrapolated, and exact ZNE results.
    """

    nplots = len(models)
    nrows = (nplots + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axs = axs.flatten() if nplots > 1 else [axs]

    for i, model in enumerate(models):
        ax = axs[i]
        DATA = data[model]

        # Try full key first, fall back to prefix if needed
        title = plot_titles.get(model, plot_titles.get(model.split("-")[0], model))

        # --- Noisy estimation ---
        ax.errorbar(
            x=DATA["redundant"]["sorted_noise_levels"],
            y=DATA["redundant"]["mean"],
            yerr=DATA["redundant"]["std"],
            fmt="o",
            ecolor=plot_colors[0],
            capsize=4,
            label="Noisy estimation",
            color=plot_colors[0],
            markersize=5
        )

        # --- ZNE Extrapolated ---
        ax.errorbar(
            x=np.atleast_1d(extrapol_target),
            y=np.atleast_1d(DATA["zne"]["mean"]),
            yerr=np.atleast_1d(DATA["zne"]["std"]),
            fmt="o",
            ecolor=plot_colors[min(2, len(plot_colors)-1)],
            capsize=4,
            label="Richardson ZNE",
            color=plot_colors[min(2, len(plot_colors)-1)],
            markersize=5
        )

        # --- Noise-free estimation ---
        ax.errorbar(
            x=[0],
            y=[DATA["noiseoff"]["mean"]],
            yerr=[DATA["noiseoff"]["std"]],
            fmt="*",
            ecolor=plot_colors[min(6, len(plot_colors)-1)],
            capsize=4,
            label="Noise-free estimation",
            color=plot_colors[min(6, len(plot_colors)-1)],
            markersize=7
        )

        # --- Exact solution ---
        ax.axhline(
            y=exact_solution,
            color=plot_colors[min(5, len(plot_colors)-1)],
            linestyle="--",
            linewidth=1.5,
            label="Exact solution"
        )

        # --- Titles and labels ---
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(r"Noise level ($\alpha_k \lambda$)", fontsize=12)
        if i % ncols == 0:
            ax.set_ylabel("Expectation value", fontsize=12)

        ax.grid(linestyle="--", alpha=0.6)
        ax.legend(loc="best", fontsize=10, frameon=False)

    # --- Hide unused axes ---
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(w_pad=1.2, h_pad=0.6)
    plt.subplots_adjust(top=0.93)

    # --- Save compiled figure ---
    os.makedirs(output_dir, exist_ok=True)
    plot_file_name = f"{filename_prefix}.eps"
    plt.savefig(os.path.join(output_dir, plot_file_name), format="eps")
    print(f"✅ Compiled figure saved: {os.path.join(output_dir, plot_file_name)}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
