from typing import Dict, List, Tuple, Any, Union, Optional
import matplotlib.pyplot as plt
import os
from pprint import pprint

def plot_zne_result(
    data: Dict[str, Any],
    extrapol_target: Union[float, List[float]],
    exact_solution: float,
    plot_colors: List[str],
    plot_title: str,
    plot_file_name: str,
    output_dir: str,
    figsize: Tuple[float, float] = (4, 6),
    xlabel: str = r"Noise level ($\alpha_k\lambda$)",
    ylabel: str = "Expectation value",
    legend_loc: str = "upper left",
    legend_fontsize: int = 14,
    label_fontsize: int = 16,
    grid_style: Optional[Dict[str, Any]] = None,
    capsize: int = 5,
    save_format: str = "eps",
    show_plot: bool = True,
    print_data: bool = True
) -> plt.Figure:
    """
    Creates a ZNE result plot with noisy, extrapolated, noise-free, and exact solution lines.
    Returns the matplotlib Figure object for storage or later PDF compilation.

    Parameters
    ----------
    data : dict
        Nested dictionary with keys "redundant", "zne", and "noiseoff", each containing
        'sorted_noise_levels', 'mean', and 'std'.
    extrapol_target : float or list
        X-values for the extrapolated point(s).
    exact_solution : float
        Reference exact solution value.
    plot_colors : list
        List of colors; indices [0], [2], [5], [6] are used for noisy, extrapolated,
        exact, and noise-free points.
    plot_title : str
        Plot title.
    plot_file_name : str
        File name to save (include extension if desired).
    output_dir : str
        Directory to save the plot.
    figsize : tuple
        Figure size (width, height).
    xlabel, ylabel : str
        Axis labels.
    legend_loc : str
        Legend location.
    legend_fontsize : int
        Font size for legend text.
    label_fontsize : int
        Font size for axis labels.
    grid_style : dict, optional
        Grid style, e.g., {"linestyle": "--", "alpha": 0.6}.
    capsize : int
        Cap size for error bars.
    save_format : str
        File format for saving (eps, png, etc.).
    show_plot : bool
        Whether to display the plot immediately.
    print_data : bool
        Whether to pretty-print the data dictionary.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot, suitable for saving or adding to PDFs.
    """

    if grid_style is None:
        grid_style = {"linestyle": "--", "alpha": 0.6}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create figure and axes explicitly
    fig, ax = plt.subplots(figsize=figsize)

    # --- Noisy estimation ---
    ax.errorbar(
        x=data["redundant"]["sorted_noise_levels"],
        y=data["redundant"]["mean"],
        yerr=data["redundant"]["std"],
        fmt="o",
        ecolor=plot_colors[0],
        capsize=capsize,
        label="Noisy estimation",
        color=plot_colors[0],
        markersize=5
    )

    # --- Extrapolated ---
    ax.errorbar(
        x=extrapol_target,
        y=data["zne"]["mean"],
        yerr=data["zne"]["std"],
        fmt="D",
        ecolor=plot_colors[2],
        capsize=capsize,
        label="Ric. ZNE value",
        color=plot_colors[2],
        markersize=5
    )

    # --- Noise-free ---
    ax.errorbar(
        x=0,
        y=data["noiseoff"]["mean"],
        yerr=data["noiseoff"]["std"],
        fmt="*",
        ecolor=plot_colors[6],
        capsize=capsize,
        label="Noise-free estimation",
        color=plot_colors[6],
        markersize=7
    )

    # --- Exact solution ---
    ax.axhline(
        y=exact_solution,
        color=plot_colors[5],
        linestyle="--",
        linewidth=1.5,
        label="Exact Solution"
    )

    # --- Labels, grid, legend ---
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(plot_title)
    ax.grid(**grid_style)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False)

    # --- Save ---
    save_path = os.path.join(output_dir, plot_file_name)
    fig.savefig(save_path, format=save_format, bbox_inches='tight')
    print(f"✅ Figure saved as (in '{output_dir}' folder): {plot_file_name}")

    if print_data:
        pprint(data, sort_dicts=False, width=80)

    # --- Show or close ---
    if show_plot:
        # Only works in interactive backends
        plt.show()
    else:
        plt.close(fig)

    return fig