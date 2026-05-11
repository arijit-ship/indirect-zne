from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from pprint import pprint
import colorsys
import matplotlib.colors as mcolors


def get_plotting_data(simulation_data):
    plot_map = {}

    for exp_name, sub_categories in simulation_data.items():
        plot_map[exp_name] = {}
        
        for sub_key, file_list in sub_categories.items():
            zne_files       = [f for f in file_list if f['type'] == 'ZNE']
            noise_off_files = [f for f in file_list if f['type'] == 'noise_off']

            if not zne_files:
                continue
            
            # Use the first file to establish the "Ground Truth" for this category
            first_data = zne_files[0]['data']
            expected_noise = first_data.get('output', {}).get('zne_values', {}).get('others', {}).get('sorted_noise')
            expected_len = len(expected_noise) if expected_noise else 0
            
            all_extrapolated = []
            all_y_curves = []
            
            for f in zne_files:
                fname = f['filename']
                out_vals = f['data'].get('output', {}).get('zne_values', {})
                others = out_vals.get('others', {})
                
                current_noise = others.get('sorted_noise', [])
                y_vals = others.get('sorted_expectation_vals', [])
                ext_val = out_vals.get('extrapolated_value')

                # --- Validation Checks ---
                
                # 1. Check one-to-one correspondence within the file
                if len(current_noise) != len(y_vals):
                    warnings.warn(f"\n[MISMATCH] Internal length mismatch in file: {fname}\n"
                                  f"Noise points: {len(current_noise)}, Expectation points: {len(y_vals)}")
                    continue

                # 2. Check consistency across the entire sub-folder (xy-ric#)
                if len(current_noise) != expected_len:
                    warnings.warn(f"\n[INCONSISTENCY] Sample length differs from category baseline!\n"
                                  f"Location: {exp_name} -> {sub_key}\n"
                                  f"File: {fname}\n"
                                  f"Expected: {expected_len}, Found: {len(current_noise)}")
                    continue

                if ext_val is not None:
                    all_extrapolated.append(ext_val)
                if y_vals:
                    all_y_curves.append(y_vals)

            # --- Noise-off Aggregation ---
            all_noise_off_costs = []

            for f in noise_off_files:
                fname = f['filename']
                costs = f['data'].get('output', {}).get('optimized_minimum_cost')

                if costs is None:
                    warnings.warn(f"\n[MISSING] 'optimized_minimum_cost' not found in noise_off file: {fname}\n"
                                  f"Location: {exp_name} -> {sub_key}")
                    continue

                if not isinstance(costs, list) or len(costs) == 0:
                    warnings.warn(f"\n[EMPTY] 'optimized_minimum_cost' is empty or not a list in noise_off file: {fname}\n"
                                  f"Location: {exp_name} -> {sub_key}")
                    continue

                all_noise_off_costs.extend(costs)

            if all_noise_off_costs:
                noise_off_arr  = np.array(all_noise_off_costs)
                mean_noise_off = float(np.mean(noise_off_arr))
                std_noise_off  = float(np.std(noise_off_arr))
            else:
                mean_noise_off = None
                std_noise_off  = None

            # --- Final Aggregation ---
            if all_extrapolated and all_y_curves:
                curves_array = np.array(all_y_curves)
                
                plot_map[exp_name][sub_key] = {
                    "noise_type":     first_data.get('config', {}).get('noise_profile', {}).get('type'),
                    "noise_prob":     first_data.get('config', {}).get('noise_profile', {}).get('noise_prob'),
                    "exact_sol":      first_data.get('output', {}).get('exact_sol'),
                    "sorted_noise":   expected_noise,
                    "mean_exp_vals":  np.mean(curves_array, axis=0).tolist(),
                    "std_exp_vals":   np.std(curves_array, axis=0).tolist(),
                    "zne_mean":       np.mean(all_extrapolated),
                    "zne_std":        np.std(all_extrapolated),
                    "mean_noise_off": mean_noise_off,
                    "std_noise_off":  std_noise_off,
                }
            else:
                plot_map[exp_name][sub_key] = None

    return plot_map

def plot_single_zne(
    data: Dict[str, Any],
    plot_colors: Dict[str, str],
    plot_file_name: str,
    output_dir: str,
    plot_title: str = None,
    extrapol_target: Optional[Union[float, List[float]]] = None,
    figsize: Tuple[float, float] = (4, 6),
    dpi: int = 150,
    xlabel: str = r"Noise level ($\alpha_k\lambda$)",
    ylabel: str = "Expectation value",
    title_fontsize: int = 14,
    label_fontsize: int = 16,
    legend_fontsize: int = 14,
    show_legend: bool = True,
    legend_loc: str = "upper left",
    legend_outside_plot: bool = False,
    grid_style: Optional[Dict[str, Any]] = None,
    capsize: int = 5,
    save_format: str = "eps",
    show_plot: bool = True,
    print_data: bool = True,
) -> plt.Figure:
    """
    Creates a ZNE result plot from a flat result dictionary.

    The dict is expected to have the following keys:
        - 'noise_type'     : str        — label used in the legend (e.g. 'depolarizing')
        - 'noise_prob'     : list       — per-point noise probabilities (unused visually, printed only)
        - 'exact_sol'      : float      — exact reference solution (horizontal dashed line)
        - 'sorted_noise'   : list       — x-axis noise level values for the noisy points
        - 'mean_exp_vals'  : list       — mean expectation values for the noisy points
        - 'std_exp_vals'   : list       — std deviations for the noisy points
        - 'zne_mean'       : float      — ZNE extrapolated mean (plotted at extrapol_target or x=0)
        - 'zne_std'        : float      — ZNE extrapolated std
        - 'mean_noise_off' : float|None — noise-free mean, plotted at x=0 if present
        - 'std_noise_off'  : float|None — noise-free std, plotted at x=0 if present

    Parameters
    ----------
    data : dict
        Flat result dictionary as described above.
    plot_colors : dict
        Named color dict with the following keys:
            'noisy'      — noisy estimation markers and errorbars
            'zne'        — ZNE extrapolated marker and errorbars
            'exact'      — exact solution horizontal line
            'noise_free' — noise-free estimation marker and errorbars
        Example::

            COLORS = {
                "noisy":      "#1f77b4",
                "zne":        "#ff7f0e",
                "exact":      "#d62728",
                "noise_free": "#9467bd",
            }

    plot_title : str
        Title displayed above the plot.
    plot_file_name : str
        Base file name for the saved figure (without extension; extension is
        derived from save_format).
    output_dir : str
        Directory where the figure is saved (created if it does not exist).
    extrapol_target : float or list, optional
        X-position(s) for the ZNE extrapolated point.
        Defaults to 0 (zero-noise limit) when not provided.
    figsize : tuple of float
        Figure dimensions as (width, height) in inches.
    dpi : int
        Resolution in dots per inch for both rendering and raster save formats.
        Default is 150. Has no effect on vector formats (eps, svg, pdf).
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title_fontsize : int
        Font size for the plot title. Default is 14.
    label_fontsize : int
        Font size for x/y axis labels. Default is 16.
    legend_fontsize : int
        Font size for legend entries. Default is 14.
    show_legend : bool
        If False, the legend is omitted entirely. Default is True.
    legend_loc : str
        Matplotlib legend location string (e.g. 'upper left').
        Ignored when legend_outside_plot is True.
    legend_outside_plot : bool
        If True, places the legend to the right of the axes and adjusts the
        layout so it is not clipped. Overrides legend_loc. Default is False.
    grid_style : dict, optional
        Keyword arguments forwarded to ``ax.grid()``.
        Defaults to ``{"linestyle": "--", "alpha": 0.6}``.
    capsize : int
        Cap size for error bar whiskers.
    save_format : str
        Output format passed to ``fig.savefig()`` (e.g. 'eps', 'png', 'pdf').
    show_plot : bool
        If True, calls ``plt.show()``; if False, closes the figure after saving.
    print_data : bool
        If True, pretty-prints the full data dictionary to stdout.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object, ready for further use or PDF compilation.
    """

    if grid_style is None:
        grid_style = {"linestyle": "--", "alpha": 0.6}

    if extrapol_target is None:
        extrapol_target = 0

    # ------------------------------------------------------------------ #
    #  Unpack flat dict                                                    #
    # ------------------------------------------------------------------ #
    noise_type     = data["noise_type"]
    exact_sol      = data["exact_sol"]
    sorted_noise   = data["sorted_noise"]
    mean_exp_vals  = data["mean_exp_vals"]
    std_exp_vals   = data["std_exp_vals"]
    zne_mean       = data["zne_mean"]
    zne_std        = data["zne_std"]
    mean_noise_off = data.get("mean_noise_off")
    std_noise_off  = data.get("std_noise_off")

    # ------------------------------------------------------------------ #
    #  Build figure                                                        #
    # ------------------------------------------------------------------ #
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")

    # --- Noisy estimation ---
    ax.errorbar(
        x=sorted_noise,
        y=mean_exp_vals,
        yerr=std_exp_vals,
        fmt="o",
        ecolor=plot_colors["noisy"],
        capsize=capsize,
        label=f"{noise_type.capitalize()} estimation",
        color=plot_colors["noisy"],
        markersize=5,
    )

    # --- ZNE extrapolated ---
    ax.errorbar(
        x=extrapol_target,
        y=zne_mean,
        yerr=zne_std,
        fmt="D",
        ecolor=plot_colors["zne"],
        capsize=capsize,
        label="Richardson ZNE",
        color=plot_colors["zne"],
        markersize=5,
    )

    # --- Noise-free estimation (x=0, only if available) ---
    if mean_noise_off is not None:
        ax.errorbar(
            x=0,
            y=mean_noise_off,
            yerr=std_noise_off if std_noise_off is not None else 0,
            fmt="*",
            ecolor=plot_colors["noise_free"],
            capsize=capsize,
            label="Noise-free estimation",
            color=plot_colors["noise_free"],
            markersize=7,
        )

    # --- Exact solution ---
    ax.axhline(
        y=exact_sol,
        color=plot_colors["exact"],
        linestyle="--",
        linewidth=1.5,
        label="Exact solution",
    )

    # --- Cosmetics ---
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(plot_title, fontsize=title_fontsize)
    ax.grid(**grid_style)

    # --- Legend ---
    if show_legend:
        if legend_outside_plot:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0,
                fontsize=legend_fontsize,
                frameon=False,
            )
            fig.tight_layout()
        else:
            ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False)

    # --- Save ---
    base_name = os.path.splitext(plot_file_name)[0]
    save_path = os.path.join(output_dir, f"{base_name}.{save_format}")
    fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches="tight")
    print(f"✅ Figure saved as (in '{output_dir}' folder): {base_name}.{save_format}")

    if print_data:
        pprint(data, sort_dicts=False, width=80)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_multi_zne(
    data_list: List[Dict[str, Any]],
    plot_colors: Dict[str, str],
    plot_file_name: str,
    output_dir: str,
    # --- Panel labels ---
    plot_titles: Optional[List[str]] = None,
    panel_label_y: Optional[float] = None,  # None = top (set_title); float = axes-fraction position (e.g. -0.25 for below xlabel)
    panel_label_fontsize: Optional[int] = None,
    # --- Data ---
    extrapol_target: Optional[Union[float, List[float]]] = None,
    # --- Grid ---
    ncols: int = 3,
    figsize: Tuple[float, float] = (12, 4),
    dpi: int = 150,
    sharex: bool = False,
    sharey: bool = True,
    # --- Axis labels ---
    xlabel: str = r"Noise level ($\alpha_k\lambda$)",
    ylabel: str = "Expectation value",
    # --- Font sizes ---
    title_fontsize: int = 8,
    label_fontsize: int = 8,
    tick_fontsize: int = 8,
    legend_fontsize: int = 8,
    # --- Global legend ---
    show_legend: bool = True,
    global_legend: bool = False,
    legend_loc: str = "lower center",
    legend_bbox: Optional[Tuple[float, float]] = None,
    legend_ncols: Optional[int] = None,
    # --- Subplot spacing (passed directly to subplots_adjust) ---
    subplot_top: Optional[float] = None,
    subplot_bottom: Optional[float] = None,
    subplot_wspace: Optional[float] = None,
    subplot_hspace: Optional[float] = None,
    # --- Per-panel legend (when global_legend=False) ---
    legend_outside_plot: bool = False,
    # --- Figure caption (fig.text at arbitrary position) ---
    figure_title: Optional[str] = None,
    figure_title_x: float = 0.5,
    figure_title_y: float = -0.01,
    figure_title_ha: str = "center",
    figure_title_va: str = "top",
    figure_title_fontsize: int = 12,
    # --- Styling ---
    grid_style: Optional[Dict[str, Any]] = None,
    capsize: int = 4,
    marker_size: float = 5,
    border_width: float = 1.5,
    save_format: str = "eps",
    show_plot: bool = True,
    print_data: bool = False,
) -> plt.Figure:
    """
    Creates a multi-panel grid of ZNE result plots, one panel per entry in
    ``data_list``.

    Each entry in ``data_list`` is a flat dict with the following keys:
        - 'noise_type'     : str        — noise model label (e.g. 'depolarizing')
        - 'noise_prob'     : list       — per-gate noise probabilities (printed only)
        - 'exact_sol'      : float      — exact reference solution (dashed horizontal line)
        - 'sorted_noise'   : list       — x-axis noise scaling factors
        - 'mean_exp_vals'  : list       — mean expectation values for noisy points
        - 'std_exp_vals'   : list       — std deviations for noisy points
        - 'zne_mean'       : float      — Richardson-extrapolated mean
        - 'zne_std'        : float      — Richardson-extrapolated std
        - 'mean_noise_off' : float|None — noise-free mean, plotted at x=0 if present
        - 'std_noise_off'  : float|None — noise-free std, plotted at x=0 if present

    Layout tuning guide
    -------------------
    Legend at top, panel labels at bottom::

        plot_multi_zne(
            ...
            global_legend  = True,
            legend_loc     = "upper center",
            legend_bbox    = (0.5, 1.0),   # x=centre, y=top of figure
            subplot_top    = 0.82,          # shrink subplots down to fit legend
            plot_titles    = ["(a)", "(b)"],
            panel_label_y  = -0.25,         # push label below x-axis label
        )

    Legend at bottom, panel labels at bottom (stacked)::

        plot_multi_zne(
            ...
            global_legend  = True,
            legend_loc     = "lower center",
            legend_bbox    = (0.5, 0.0),
            subplot_bottom = 0.22,
            plot_titles    = ["(a)", "(b)"],
            panel_label_y  = -0.30,
        )

    Parameters
    ----------
    data_list : list of dict
        One flat result dict per subplot panel.
    plot_colors : dict
        Named color dict with keys: 'noisy', 'zne', 'exact', 'noise_free'.
    plot_file_name : str
        Base file name (no extension; extension derived from save_format).
    output_dir : str
        Output directory (created if absent).
    plot_titles : list of str, optional
        Per-panel caption labels, e.g. ["(a)", "(b)"]. Placed below x-axis label.
    panel_label_y : float
        Vertical position of panel labels in axes-fraction coordinates.
        0 = bottom of axes, negative values go below. Default -0.22.
    panel_label_fontsize : int, optional
        Font size for panel labels. Defaults to title_fontsize.
    extrapol_target : float or list, optional
        X-position for ZNE extrapolated point. Defaults to 0.
    ncols : int
        Subplot grid columns. Default 3.
    figsize : tuple of float
        Figure size in inches.
    dpi : int
        Figure resolution. Default 150.
    sharex, sharey : bool
        Share axes across panels.
    xlabel, ylabel : str
        Axis labels.
    title_fontsize : int
        Panel title / label font size. Default 13.
    label_fontsize : int
        Axis label font size. Default 12.
    tick_fontsize : int
        Tick label font size. Default 11.
    legend_fontsize : int
        Legend font size. Default 10.
    show_legend : bool
        Master switch for all legends.
    global_legend : bool
        Single shared figure-level legend instead of per-panel.
    legend_loc : str
        Matplotlib loc string for global legend, e.g. 'upper center'.
    legend_bbox : tuple of float, optional
        Explicit bbox_to_anchor (x, y) in figure coordinates.
        If None, matplotlib places the legend using legend_loc alone.
    legend_ncols : int, optional
        Number of columns in global legend. Defaults to number of items (one row).
    subplot_top : float, optional
        Passed to subplots_adjust(top=). Use to create space for a top legend.
    subplot_bottom : float, optional
        Passed to subplots_adjust(bottom=). Use to create space for a bottom legend.
    subplot_wspace : float, optional
        Horizontal spacing between subplots.
    subplot_hspace : float, optional
        Vertical spacing between subplots.
    legend_outside_plot : bool
        Per-panel mode only: anchor legend to the right of each axes.
    figure_title : str, optional
        Text placed via fig.text() at an arbitrary figure position.
    figure_title_x : float
        X position of figure_title in figure coordinates. Default 0.5.
    figure_title_y : float
        Y position of figure_title in figure coordinates. Default -0.01.
    figure_title_ha : str
        Horizontal alignment. Default 'center'.
    figure_title_va : str
        Vertical alignment. Default 'top'.
    figure_title_fontsize : int
        Font size for figure_title. Default 12.
    grid_style : dict, optional
        kwargs for ax.grid(). Default {"linestyle": "--", "alpha": 0.6}.
    capsize : int
        Errorbar cap size. Default 4.
    marker_size : float
        Marker size. Default 5.
    border_width : float
        Spine line width. Default 1.5.
    save_format : str
        File format: 'eps', 'png', 'pdf', etc.
    show_plot : bool
        Call plt.show() after saving.
    print_data : bool
        Pretty-print each data dict to stdout.

    Returns
    -------
    matplotlib.figure.Figure
    """

    if grid_style is None:
        grid_style = {"linestyle": "--", "alpha": 0.6}
    if extrapol_target is None:
        extrapol_target = 0
    if panel_label_fontsize is None:
        panel_label_fontsize = title_fontsize

    # ------------------------------------------------------------------ #
    #  Grid layout                                                         #
    # ------------------------------------------------------------------ #
    nplots = len(data_list)
    nrows  = (nplots + ncols - 1) // ncols

    plt.rcParams.update({
        "font.size":        tick_fontsize,
        "axes.labelsize":   label_fontsize,
        "axes.titlesize":   title_fontsize,
        "legend.fontsize":  legend_fontsize,
        "xtick.labelsize":  tick_fontsize,
        "ytick.labelsize":  tick_fontsize,
    })

    os.makedirs(output_dir, exist_ok=True)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi,
                             sharex=sharex, sharey=sharey)
    axs = axs.flatten() if nplots > 1 else [axs]

    shared_handles, shared_labels = None, None

    for i, data in enumerate(data_list):
        ax = axs[i]

        # ---- Unpack ----
        noise_type     = data["noise_type"]
        exact_sol      = data["exact_sol"]
        sorted_noise   = data["sorted_noise"]
        mean_exp_vals  = data["mean_exp_vals"]
        std_exp_vals   = data["std_exp_vals"]
        zne_mean       = data["zne_mean"]
        zne_std        = data["zne_std"]
        mean_noise_off = data.get("mean_noise_off")
        std_noise_off  = data.get("std_noise_off")

        if print_data:
            label = plot_titles[i] if plot_titles and i < len(plot_titles) else i
            print(f"\n--- Panel {i}: {label} ---")
            pprint(data, sort_dicts=False, width=80)

        # --- Noisy estimation ---
        ax.errorbar(
            x=sorted_noise,
            y=mean_exp_vals,
            yerr=std_exp_vals,
            fmt="o",
            ecolor=plot_colors["noisy"],
            capsize=capsize,
            #label=f"{noise_type.capitalize()} estimation",
            label=f"Noisy estimation",
            color=plot_colors["noisy"],
            markersize=marker_size,
            markeredgewidth=0.8,
            elinewidth=1,
        )
        # Unmitigated

        ax.axhline(
            mean_exp_vals[0], color=plot_colors["unmitigated"],
            linestyle="--", linewidth=border_width, zorder=4,
            label="Unmitigated"
        )
        ax.axhspan(
                    mean_exp_vals[0] - std_exp_vals[0],
                    mean_exp_vals[0] + std_exp_vals[0],
                    color=plot_colors["unmitigated"], alpha=0.2
                )
        # --- ZNE extrapolated ---
        ax.errorbar(
            x=np.atleast_1d(extrapol_target),
            y=np.atleast_1d(zne_mean),
            yerr=np.atleast_1d(zne_std),
            fmt="D",
            ecolor=plot_colors["zne"],
            capsize=capsize,
            label="Richardson ZNE",
            color=plot_colors["zne"],
            markersize=marker_size,
            markeredgewidth=0.8,
            elinewidth=1,
        )

        # --- Noise-free estimation (x=0, only if available) ---
        if mean_noise_off is not None:

            ax.axhline(
            mean_noise_off, color=plot_colors["noise_free"],
            linestyle="--", linewidth=border_width, zorder=4,
            label="Noise-free estimation"
        )
            if std_noise_off is not None:
                ax.axhspan(
                    mean_noise_off - std_noise_off,
                    mean_noise_off + std_noise_off,
                    color=plot_colors["noise_free"], alpha=0.2
                )
            # ax.errorbar(
            #     x=0,
            #     y=mean_noise_off,
            #     yerr=std_noise_off if std_noise_off is not None else 0,
            #     fmt="*",
            #     ecolor=plot_colors["noise_free"],
            #     capsize=capsize,
            #     label="Noise-free estimation",
            #     color=plot_colors["noise_free"],
            #     markersize=marker_size + 2,
            #     markeredgewidth=1,
            #     elinewidth=1,
            # )

        # --- Exact solution ---
        ax.axhline(
            y=exact_sol,
            color=plot_colors["exact"],
            linestyle="--",
            linewidth=1.5,
            label="Exact solution",
        )

        # --- Axis labels ---
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=label_fontsize)

        # --- Panel label: top (default) or custom vertical position ---
        if plot_titles is not None and i < len(plot_titles):
            if panel_label_y is None:
                ax.set_title(plot_titles[i], fontsize=panel_label_fontsize)
            else:
                ax.text(
                    x=0.5, y=panel_label_y,
                    s=plot_titles[i],
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=panel_label_fontsize,
                )

        ax.grid(**grid_style)
        ax.tick_params(width=1, length=4, direction="inout", labelsize=tick_fontsize)

        for spine in ax.spines.values():
            spine.set_linewidth(border_width)
            spine.set_color("black")

        # Capture handles once for global legend
        if shared_handles is None:
            shared_handles, shared_labels = ax.get_legend_handles_labels()

        # --- Per-panel legend ---
        if show_legend and not global_legend:
            if legend_outside_plot:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1),
                    borderaxespad=0,
                    fontsize=legend_fontsize,
                    frameon=False,
                )
            else:
                ax.legend(loc="best", fontsize=legend_fontsize, frameon=False)

    # Hide unused axes
    for j in range(nplots, len(axs)):
        fig.delaxes(axs[j])

    # ------------------------------------------------------------------ #
    #  Subplot spacing                                                     #
    # ------------------------------------------------------------------ #
    plt.tight_layout(w_pad=1.4, h_pad=0.8)

    adjust_kwargs = {}
    if subplot_top    is not None: adjust_kwargs["top"]    = subplot_top
    if subplot_bottom is not None: adjust_kwargs["bottom"] = subplot_bottom
    if subplot_wspace is not None: adjust_kwargs["wspace"] = subplot_wspace
    if subplot_hspace is not None: adjust_kwargs["hspace"] = subplot_hspace

    # Auto-reserve space for global legend when user hasn't set it explicitly
    if show_legend and global_legend and shared_handles:
        _ncols = legend_ncols if legend_ncols is not None else len(shared_handles)
        _legend_rows = -(-len(shared_handles) // _ncols)  # ceiling division
        if legend_loc in ("lower center", "lower left", "lower right") or (
            legend_bbox is not None and legend_bbox[1] <= 0.15
        ):
            if subplot_bottom is None:
                adjust_kwargs["bottom"] = 0.12 + 0.06 * _legend_rows
        elif legend_loc in ("upper center", "upper left", "upper right") or (
            legend_bbox is not None and legend_bbox[1] >= 0.85
        ):
            if subplot_top is None:
                adjust_kwargs["top"] = 0.94 - 0.06 * _legend_rows

    if adjust_kwargs:
        plt.subplots_adjust(**adjust_kwargs)

    # ------------------------------------------------------------------ #
    #  Global legend                                                       #
    # ------------------------------------------------------------------ #
    if show_legend and global_legend and shared_handles:
        _ncols = legend_ncols if legend_ncols is not None else len(shared_handles)
        legend_kwargs = dict(
            ncol=_ncols,
            frameon=False,
            fontsize=legend_fontsize,
            handletextpad=0.5,
            columnspacing=1.2,
        )
        if legend_bbox is not None:
            fig.legend(
                shared_handles, shared_labels,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox,
                **legend_kwargs,
            )
        else:
            fig.legend(
                shared_handles, shared_labels,
                loc=legend_loc,
                **legend_kwargs,
            )

    # ------------------------------------------------------------------ #
    #  Figure caption / title via fig.text                                #
    # ------------------------------------------------------------------ #
    if figure_title is not None:
        fig.text(
            figure_title_x, figure_title_y,
            figure_title,
            ha=figure_title_ha, va=figure_title_va,
            fontsize=figure_title_fontsize,
        )

    # ------------------------------------------------------------------ #
    #  Save                                                                #
    # ------------------------------------------------------------------ #
    base_name = os.path.splitext(plot_file_name)[0]
    save_path = os.path.join(output_dir, f"{base_name}.{save_format}")
    #fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches="tight")
    fig.savefig(save_path, format=save_format, dpi=dpi)
    print(f"✅ Figure saved as (in '{output_dir}' folder): {base_name}.{save_format}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _derive_series_colors(base_color: str) -> Dict[str, str]:
    """
    Derive a full ``plot_colors``-style dict from a single base color.

    Given a base color (any matplotlib color spec), returns:
      - 'noisy'      : base color at full saturation / slightly darkened
      - 'zne'        : base color lightened (mixed toward white, ~40 %)
      - 'noise_free' : base color desaturated (mixed toward gray, ~35 %)

    All three are returned as hex strings so they work everywhere matplotlib
    accepts a color string.
    """
    import colorsys

    def _to_rgb(c: str) -> Tuple[float, float, float]:
        return mcolors.to_rgb(c)

    def _to_hex(rgb: Tuple[float, float, float]) -> str:
        return mcolors.to_hex(rgb)

    def _lighten(rgb, amount=0.40):
        """Mix rgb toward white by `amount` (0 = no change, 1 = white)."""
        return tuple(v + (1.0 - v) * amount for v in rgb)

    def _darken(rgb, amount=0.20):
        """Mix rgb toward black by `amount` (0 = no change, 1 = black)."""
        return tuple(v * (1.0 - amount) for v in rgb)

    def _desaturate(rgb, amount=0.35):
        """Reduce saturation by `amount` in HLS space."""
        h, l, s = colorsys.rgb_to_hls(*rgb)
        s_new = s * (1.0 - amount)
        return colorsys.hls_to_rgb(h, l, s_new)

    base_rgb = _to_rgb(base_color)
    return {
        "noisy":      _to_hex(_darken(base_rgb, 0.15)),
        "zne":        _to_hex(_lighten(base_rgb, 0.40)),
        "noise_free": _to_hex(_desaturate(base_rgb, 0.35)),
    }


# Default qualitative color cycle used when series_colors is not provided.
_DEFAULT_SERIES_COLORS = [
    "#1f77b4",  # muted blue
    "#d62728",  # brick red
    "#2ca02c",  # cooked asparagus green
    "#ff7f0e",  # safety orange
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#17becf",  # blue-teal
]


def plot_single_zne_imposed(
    data_list: List[Dict[str, Any]],
    plot_colors: Dict[str, str],        # Dict containing "exact"
    plot_file_name: str,
    output_dir: str,
    plot_titles: Optional[List[str]] = None,
    dataset_colors: Optional[List[str]] = None, # NEW: Custom colors per dataset
    extrapol_target: Optional[Union[float, List[float]]] = None,
    figsize: Tuple[float, float] = (7, 5),
    dpi: int = 150,
    xlabel: str = r"Noise level ($\alpha_k\lambda$)",
    ylabel: str = "Expectation value",
    title_fontsize: int = 13,
    label_fontsize: int = 12,
    tick_fontsize: int = 11,
    legend_fontsize: int = 10,
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_bbox: Optional[Tuple[float, float]] = None,
    legend_ncols: Optional[int] = None,
    grid_style: Optional[Dict[str, Any]] = None,
    capsize: int = 4,
    marker_size: float = 6,
    border_width: float = 1.5,
    save_format: str = "eps",
    show_plot: bool = True,
    print_data: bool = False,
) -> plt.Figure:

    # ------------------------------------------------------------------ #
    # Defaults & Setup
    # ------------------------------------------------------------------ #
    if grid_style is None:
        grid_style = {"linestyle": "--", "alpha": 0.6, "color": "gray"}
    if extrapol_target is None:
        extrapol_target = 0

    # Default fallback palette
    DEFAULT_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    # Use user-provided colors or the default palette
    colors_to_use = dataset_colors if dataset_colors is not None else DEFAULT_PALETTE

    markers = {"noisy": "o", "zne": "D", "noise_free": "*"}

    plt.rcParams.update({
        "font.size": tick_fontsize,
        "axes.labelsize": label_fontsize,
        "axes.titlesize": title_fontsize,
        "legend.fontsize": legend_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
    })

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, layout="constrained")
    ax.set_axisbelow(True) # Grid goes behind data

    # ------------------------------------------------------------------ #
    # Plotting Logic
    # ------------------------------------------------------------------ #
    exact_drawn = False
    legend_handles = []

    for i, data in enumerate(data_list):
        # Pick color based on the provided list (wraps around if list is short)
        color = colors_to_use[i % len(colors_to_use)]
        
        label = plot_titles[i] if (plot_titles and i < len(plot_titles)) else f"Series {i}"

        if print_data:
            print(f"\n--- Series {i}: {label} ---")
            pprint(data, sort_dicts=False, width=80)

        # 1. Noisy Data (Handle for legend)
        h_noisy = ax.errorbar(
            data["sorted_noise"],
            data["mean_exp_vals"],
            yerr=data["std_exp_vals"],
            fmt=markers["noisy"],
            color=color,
            ecolor=color,
            capsize=capsize,
            markersize=marker_size,
            label=label,
            zorder=3
        )
        legend_handles.append(h_noisy)

        # 2. ZNE Point
        ax.errorbar(
            np.atleast_1d(extrapol_target),
            np.atleast_1d(data["zne_mean"]),
            yerr=np.atleast_1d(data["zne_std"]),
            fmt=markers["zne"],
            color=color,
            ecolor=color,
            capsize=capsize,
            markersize=marker_size,
            zorder=4
        )

        # 3. Noise-free
        if data.get("mean_noise_off") is not None:
            ax.errorbar(
                0,
                data["mean_noise_off"],
                yerr=data.get("std_noise_off", 0),
                fmt=markers["noise_free"],
                color=color,
                ecolor=color,
                capsize=capsize,
                markersize=marker_size + 2,
                zorder=5
            )

        # 4. Exact Line (Add once)
        if not exact_drawn:
            h_exact = ax.axhline(
                y=data["exact_sol"],
                color=plot_colors.get("exact", "red"),
                linestyle="--",
                linewidth=1.5,
                label="Exact solution",
                zorder=2
            )
            legend_handles.append(h_exact)
            exact_drawn = True

    # ------------------------------------------------------------------ #
    # Styling & Grid
    # ------------------------------------------------------------------ #
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, **grid_style)

    ax.tick_params(width=1, length=4, direction="inout")
    for spine in ax.spines.values():
        spine.set_linewidth(border_width)

    # ------------------------------------------------------------------ #
    # Legend
    # ------------------------------------------------------------------ #
    if show_legend:
        kwargs = dict(frameon=False)
        if legend_ncols:
            kwargs["ncol"] = legend_ncols
        
        if legend_bbox:
            ax.legend(handles=legend_handles, loc=legend_loc, bbox_to_anchor=legend_bbox, **kwargs)
        else:
            ax.legend(handles=legend_handles, loc=legend_loc, **kwargs)

    # ------------------------------------------------------------------ #
    # Save & Show
    # ------------------------------------------------------------------ #
    plt.tight_layout()
    base = os.path.splitext(plot_file_name)[0]
    path = os.path.join(output_dir, f"{base}.{save_format}")
    fig.savefig(path, format=save_format, dpi=dpi)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_zne_mul_var_single(
    data: Dict[str, Any],
    plot_colors: Dict[str, str],
    plot_file_name: str,
    output_dir: str,
    # --- Panel label ---
    plot_title: Optional[str] = None,
    panel_label_y: Optional[float] = None,
    panel_label_fontsize: Optional[int] = None,
    # --- Data ---
    extrapol_target: Optional[float] = None,
    # --- Figure ---
    figsize: Tuple[float, float] = (12, 5),
    dpi: int = 150,
    # --- Axis labels ---
    xlabel: str = None,
    ylabel: str = None,
    # --- Font sizes ---
    title_fontsize: int = 8,
    label_fontsize: int = 8,
    tick_fontsize: int = 8,
    legend_fontsize: int = 8,
    # --- Legend ---
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_bbox: Optional[Tuple[float, float]] = None,
    legend_ncols: int = 1,
    legend_outside_plot: bool = False,
    # --- Figure caption ---
    figure_title: Optional[str] = None,
    figure_title_x: float = 0.5,
    figure_title_y: float = -0.01,
    figure_title_ha: str = "center",
    figure_title_va: str = "top",
    figure_title_fontsize: int = 8,
    # --- Subplot spacing ---
    subplot_top: Optional[float] = None,
    subplot_bottom: Optional[float] = None,
    subplot_left: Optional[float] = None,
    subplot_right: Optional[float] = None,
    # --- Styling ---
    grid_style: Optional[Dict[str, Any]] = None,
    capsize: int = 4,
    marker_size: float = 5,
    border_width: float = 1.5,
    # --- Save ---
    save_format: Union[str, List[str]] = "png",
    show_plot: bool = True,
    print_data: bool = False,
) -> plt.Figure:
    """
    Single-panel plot for one ZNE-mul-var label.

    Args:
        data:           single label dict from processed["ZNE-mul-var"][label]
        plot_colors:    override any of: "zne", "noisy", "exact", "noise_off", "separator"
        plot_file_name: output filename stem (no extension)
        output_dir:     directory to save into

    Expected data keys:
        sorted_noise      : List[List[int]]
        mean_exp_vals     : List[float]
        std_exp_vals      : List[float]
        zne_mean          : float
        zne_std           : float
        exact_sol         : float  (optional)
        mean_noise_off    : float  (optional)
        std_noise_off     : float  (optional)
        tmax              : float  (optional)
        noise_type        : str    (optional)
    """

    # ------------------------------------------------------------------ #
    # Defaults
    # ------------------------------------------------------------------ #
    _colors = {
        "zne":       "#60a5fa",
        "noisy":     "red",
        "exact":     "magenta",
        "noise_off": "green",
        "separator": "#666666",
    }
    _colors.update(plot_colors)

    _grid_style = {"linestyle": "--", "alpha": 0.4}
    if grid_style:
        _grid_style.update(grid_style)

    if print_data:
        print(data)

    # ------------------------------------------------------------------ #
    # Unpack data
    # ------------------------------------------------------------------ #
    noise_vecs     = data["sorted_noise"]
    means          = np.array(data["mean_exp_vals"])
    stds           = np.array(data["std_exp_vals"])
    zne_mean       = data["zne_mean"]
    zne_std        = data["zne_std"]
    noise_off_mean = data.get("mean_noise_off")
    noise_off_std  = data.get("std_noise_off")
    target         = extrapol_target if extrapol_target is not None else data.get("exact_sol")

    # ------------------------------------------------------------------ #
    # X-axis
    # ------------------------------------------------------------------ #
    first_key_len = len(noise_vecs[0])
    zne_x_label   = f"ZNE\n({','.join(['0'] * first_key_len)})"
    noise_labels  = [f"({','.join(map(str, nv))})" for nv in noise_vecs]
    x_labels      = [zne_x_label] + noise_labels

    all_means = np.concatenate([[zne_mean], means])
    all_stds  = np.concatenate([[zne_std],  stds])
    x         = np.arange(len(x_labels))

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=figsize, dpi= dpi, layout="constrained")

    # 1. ZNE extrapolated point
    ax.errorbar(
        x[0], all_means[0], yerr=all_stds[0],
        fmt="o", color=_colors["zne"], ecolor=_colors["zne"],
        elinewidth=border_width, capsize=capsize,
        markersize=marker_size,
        label=f"ZNE: {all_means[0]:.4f} ± {all_stds[0]:.4f}",
        zorder=5
    )

    # 2. Noisy simulation points
    ax.errorbar(
        x[1:], all_means[1:], yerr=all_stds[1:],
        fmt="o", color=_colors["noisy"], ecolor=_colors["noisy"],
        elinewidth=border_width, capsize=capsize,
        markersize=marker_size,
        linestyle="None",
        label="Noisy estimations",
        zorder=3
    )

    # 3. VQE noise-off line + band
    if noise_off_mean is not None:
        ax.axhline(
            noise_off_mean, color=_colors["noise_off"],
            linestyle="--", linewidth=border_width, zorder=4,
            label=f"VQE noise-off: {noise_off_mean:.4f} ± {noise_off_std:.4f}"
        )
        if noise_off_std is not None:
            ax.axhspan(
                noise_off_mean - noise_off_std,
                noise_off_mean + noise_off_std,
                color=_colors["noise_off"], alpha=0.2
            )

    # 4. Exact / target line
    if target is not None:
        ax.axhline(
            target, color=_colors["exact"],
            linestyle="--", linewidth=border_width,
            label=f"Exact solution: {target:.4f}", zorder=4
        )

    # 5. Vertical separator
    ax.axvline(0.5, color=_colors["separator"], linestyle=":", alpha=0.8)

    # ------------------------------------------------------------------ #
    # Ticks & labels
    # ------------------------------------------------------------------ #
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, ha="center")
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(tick_fontsize)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(tick_fontsize)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.grid(**_grid_style)

    for spine in ax.spines.values():
        spine.set_linewidth(border_width)

    # ------------------------------------------------------------------ #
    # Panel title
    # ------------------------------------------------------------------ #
    panel_title = (plot_title)
    title_fs    = panel_label_fontsize or title_fontsize

    if panel_label_y is None:
        ax.set_title(panel_title, fontsize=title_fs, pad=8)
    else:
        ax.annotate(
            panel_title,
            xy=(0.5, panel_label_y), xycoords="axes fraction",
            ha="center", va="top", fontsize=title_fs,
        )

    # ------------------------------------------------------------------ #
    # Legend
    # ------------------------------------------------------------------ #
    if show_legend:
        if legend_outside_plot:
            ax.legend(
                fontsize=legend_fontsize,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
                frameon=False,
                ncol=legend_ncols,
            )
        elif legend_bbox:
            ax.legend(
                fontsize=legend_fontsize,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox,
                ncol=legend_ncols,
                frameon=False,
            )
        else:
            ax.legend(
                fontsize=legend_fontsize,
                loc=legend_loc,
                ncol=legend_ncols,
                frameon=False,
            )

    # ------------------------------------------------------------------ #
    # Figure caption
    # ------------------------------------------------------------------ #
    if figure_title:
        fig.text(
            figure_title_x, figure_title_y,
            figure_title,
            ha=figure_title_ha, va=figure_title_va,
            fontsize=figure_title_fontsize,
        )

    # ------------------------------------------------------------------ #
    # Spacing
    # ------------------------------------------------------------------ #
    spacing = {k: v for k, v in {
        "top":    subplot_top,
        "bottom": subplot_bottom,
        "left":   subplot_left,
        "right":  subplot_right,
    }.items() if v is not None}

    if spacing:
        plt.subplots_adjust(**spacing)
    else:
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)   # default room for rotated x labels

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in (save_format if isinstance(save_format, list) else [save_format]):
        out_path = output_dir / f"{plot_file_name}.{fmt.lstrip('.')}"
        fig.savefig(out_path, dpi=dpi)
        print(f"💾 Saved: {out_path}")

    if show_plot:
        plt.show()

    plt.close(fig)
    return fig

# Multivariate ZNE: plot extrapolated value vs extrapolation degree, with noisy points and optional exact/noise-free references.

def ricmul_plot_zne_vs_degree(
    processed_mul_var: Dict[str, Any],
    plot_colors: Dict[str, str],
    plot_file_name: str,
    output_dir: str,
    # --- Data ---
    exact_sol: Optional[float] = None,
    # --- Figure ---
    figsize: Tuple[float, float] = (7, 5),
    dpi: int = 150,
    # --- Axis labels ---
    xlabel: str = "Extrapolation Order",
    ylabel: str = "ZNE",
    # --- Font sizes ---
    title_fontsize: int = 8,
    label_fontsize: int = 8,
    tick_fontsize: int = 8,
    legend_fontsize: int = 8,
    # --- Legend ---
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_bbox: Optional[Tuple[float, float]] = None,
    legend_ncols: int = 1,
    # --- Figure caption ---
    figure_title: Optional[str] = None,
    figure_title_x: float = 0.5,
    figure_title_y: float = -0.01,
    figure_title_ha: str = "center",
    figure_title_va: str = "top",
    figure_title_fontsize: int = 12,
    # --- Annotations ---
    annotate_cost: bool = True,
    annotation_fontsize: int = 5,
    # --- Styling ---
    grid_style: Optional[Dict[str, Any]] = None,
    capsize: int = 5,
    marker_size: float = 8,
    border_width: float = 1.5,
    save_format: str = "eps",
    show_plot: bool = True,
    print_data: bool = False,
) -> plt.Figure:
    """
    Plots ZNE extrapolated value vs extrapolation degree for multivariate ZNE.

    Each entry in ``processed_mul_var`` is a flat dict with keys:
        - 'degree'         : int         — extrapolation degree
        - 'zne_mean'       : float       — mean extrapolated value
        - 'zne_std'        : float       — std of extrapolated value
        - 'exact_sol'      : float       — exact reference solution
        - 'mean_noise_off' : float|None  — noise-free mean reference
        - 'std_noise_off'  : float|None  — noise-free std reference
        - 'cost_mean'      : float|None  — sampling overhead c = gamma^2
        - 'cost_std'       : float|None  — std of sampling overhead

    Parameters
    ----------
    processed_mul_var : dict
        The ZNE-mul-var processed data, e.g. PROCESSED_SIM_DATA["ZNE-mul-var"].
    plot_colors : dict
        Named color dict with keys: 'zne', 'exact', 'noise_free'.
    plot_file_name : str
        Base file name (no extension; extension derived from save_format).
    output_dir : str
        Output directory (created if absent).
    exact_sol : float, optional
        Exact solution override. If None, taken from first entry's 'exact_sol'.
    figsize : tuple of float
        Figure size in inches. Default (7, 5).
    dpi : int
        Figure resolution. Default 150.
    xlabel : str
        X-axis label. Default 'Extrapolation Degree'.
    ylabel : str
        Y-axis label. Default 'ZNE Extrapolated Energy'.
    title_fontsize : int
        Title font size. Default 13.
    label_fontsize : int
        Axis label font size. Default 12.
    tick_fontsize : int
        Tick label font size. Default 11.
    legend_fontsize : int
        Legend font size. Default 10.
    show_legend : bool
        Master switch for legend. Default True.
    legend_loc : str
        Matplotlib loc string. Default 'best'.
    legend_bbox : tuple of float, optional
        Explicit bbox_to_anchor (x, y) in axes coordinates.
    legend_ncols : int
        Number of legend columns. Default 1.
    figure_title : str, optional
        Text placed via fig.text() at an arbitrary figure position.
    figure_title_x : float
        X position of figure_title in figure coordinates. Default 0.5.
    figure_title_y : float
        Y position of figure_title in figure coordinates. Default -0.01.
    figure_title_ha : str
        Horizontal alignment of figure_title. Default 'center'.
    figure_title_va : str
        Vertical alignment of figure_title. Default 'top'.
    figure_title_fontsize : int
        Font size for figure_title. Default 12.
    annotate_cost : bool
        Annotate sampling overhead c on each point. Default True.
    annotation_fontsize : int
        Font size for cost annotations. Default 8.
    grid_style : dict, optional
        kwargs for ax.grid(). Default {"linestyle": "--", "alpha": 0.4}.
    capsize : int
        Errorbar cap size. Default 5.
    marker_size : float
        Marker size. Default 6.
    border_width : float
        Spine line width. Default 1.5.
    save_format : str
        File format: 'eps', 'png', 'pdf', etc. Default 'eps'.
    show_plot : bool
        Call plt.show() after saving. Default True.
    print_data : bool
        Pretty-print each label's data to stdout. Default False.

    Returns
    -------
    matplotlib.figure.Figure
    """

    # --- Defaults ---
    if grid_style is None:
        grid_style = {"linestyle": "--", "alpha": 0.4}

    # --- Sort labels by degree ---
    labels  = sorted(processed_mul_var.keys(), key=lambda k: processed_mul_var[k]["degree"])
    degrees = [processed_mul_var[k]["degree"]    for k in labels]
    means   = [processed_mul_var[k]["zne_mean"]  for k in labels]
    stds    = [processed_mul_var[k]["zne_std"]   for k in labels]
    costs   = [processed_mul_var[k].get("cost_mean") for k in labels]

    first = processed_mul_var[labels[0]]
    _exact_sol    = exact_sol if exact_sol is not None else first.get("exact_sol")
    _noise_off    = first.get("mean_noise_off")
    _noise_off_std = first.get("std_noise_off")

    _unmitigated_mean = first.get("mean_exp_vals", [None])[0]
    _unmitigate_sd = first.get("std_exp_vals", [None])[0]

    if print_data:
        for k in labels:
            print(f"\n[{k}]")
            for key, val in processed_mul_var[k].items():
                print(f"  {key}: {val}")

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")

    # ZNE mean ± std
    ax.errorbar(
        degrees, means, yerr=stds,
        fmt="o-",
        capsize=capsize,
        markersize=marker_size,
        color=plot_colors.get("zne", "steelblue"),
        label="ZNE (multivariate)",
        zorder=3,
    )

    # Exact solution
    if _exact_sol is not None:
        ax.axhline(
            _exact_sol,
            color=plot_colors.get("exact", "red"),
            linestyle="--",
            label=f"Exact solution: {_exact_sol:.4f}",
        )

    # Unmitigated
    ax.axhline(
        _unmitigated_mean,
        color=plot_colors.get("unmitigated", "orange"),
        linestyle=":",
        label=f"Unmitigated: {_unmitigated_mean:.4f}",
    )
    ax.axhspan(
            _unmitigated_mean - _unmitigate_sd,
             _unmitigated_mean + _unmitigate_sd,
                alpha=0.12,
                color=plot_colors.get("unmitigated", "orange"),
            )

    # Noise-off reference
    if _noise_off is not None:
        ax.axhline(
            _noise_off,
            color=plot_colors.get("noise_off", "green"),
            linestyle=":",
            label=f"Noise-off: {_noise_off:.4f}",
        )
        if _noise_off_std is not None:
            ax.axhspan(
                _noise_off - _noise_off_std,
                _noise_off + _noise_off_std,
                alpha=0.12,
                color=plot_colors.get("noise_off", "green"),
            )

    # Annotate cost
    if annotate_cost:
        for d, m, c in zip(degrees, means, costs):
            if c is not None:
                ax.annotate(
                    f"c={c:.1f}",
                    xy=(d, m),
                    xytext=(5, 8),
                    textcoords="offset points",
                    fontsize=annotation_fontsize,
                    color=plot_colors.get("zne", "steelblue"),
                )

    # --- Axes formatting ---
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(figure_title, fontsize=title_fontsize)
    ax.set_xticks(degrees)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(**grid_style)

    for spine in ax.spines.values():
        spine.set_linewidth(border_width)

    # --- Legend ---
    if show_legend:
        legend_kwargs = dict(fontsize=legend_fontsize, loc=legend_loc, ncols=legend_ncols, frameon=False)
        if legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox
        ax.legend(**legend_kwargs)

    # --- Figure caption ---
    if figure_title is not None:
        fig.text(
            figure_title_x, figure_title_y,
            figure_title,
            ha=figure_title_ha,
            va=figure_title_va,
            fontsize=figure_title_fontsize,
        )

    #plt.tight_layout()

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{plot_file_name}.{save_format}")
    fig.savefig(out_path, format=save_format, dpi=dpi)
    print(f"✅ Saved: {out_path}")

    if show_plot:
        plt.show()
    plt.close(fig)
    return fig

# Univariate ZNE: plot extrapolated value vs extrapolation degree.

def ric_plot_zne_vs_degree(
    DATA: Dict[str, Any],
    plot_colors: Dict[str, str],
    plot_file_name: str,
    output_dir: str,
    # --- Data ---
    exact_sol: Optional[float] = None,
    # --- Figure ---
    figsize: Tuple[float, float] = (7, 5),
    dpi: int = 150,
    # --- Axis labels ---
    xlabel: str = "Extrapolation Order",
    ylabel: str = "ZNE",
    # --- Font sizes ---
    title_fontsize: int = 8,
    label_fontsize: int = 8,
    tick_fontsize: int = 8,
    legend_fontsize: int = 8,
    # --- Legend ---
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_bbox: Optional[Tuple[float, float]] = None,
    legend_ncols: int = 1,
    # --- Figure caption ---
    figure_title: Optional[str] = None,
    figure_title_x: float = 0.5,
    figure_title_y: float = -0.01,
    figure_title_ha: str = "center",
    figure_title_va: str = "top",
    figure_title_fontsize: int = 12,
    # --- Annotations ---
    annotate_values: bool = True,
    annotation_fontsize: int = 8,
    # --- Styling ---
    grid_style: Optional[Dict[str, Any]] = None,
    capsize: int = 5,
    marker_size: float = 5,
    border_width: float = 1.5,
    save_format: str = "eps",
    show_plot: bool = True,
    print_data: bool = False,
) -> plt.Figure:
    """
    Plots ZNE extrapolated value vs extrapolation order (Richardson degree).

    Each entry in ``DATA`` is a dict with keys:
        - 'order'           : int         — extrapolation order (Richardson degree)
        - 'sorted_noise'   : list[int]   — noise levels used
        - 'mean_exp_vals'  : list[float] — mean expectation values at each noise level
        - 'std_exp_vals'   : list[float] — std of expectation values
        - 'zne_mean'       : float       — mean ZNE extrapolated value
        - 'zne_std'        : float       — std of ZNE extrapolated value
        - 'exact_sol'      : float       — exact reference solution
        - 'mean_noise_off' : float|None  — noise-free mean reference
        - 'std_noise_off'  : float|None  — noise-free std reference

    Parameters
    ----------
    DATA : dict
        Raw ZNE results keyed by label (e.g. 'ric2', 'ric3', ...).
    plot_colors : dict
        Named color dict with keys: 'zne', 'exact', 'noise_free'.
    plot_file_name : str
        Base file name (no extension; extension derived from save_format).
    output_dir : str
        Output directory (created if absent).
    exact_sol : float, optional
        Exact solution override. If None, taken from first entry's 'exact_sol'.
    figsize : tuple of float
        Figure size in inches. Default (7, 5).
    dpi : int
        Figure resolution. Default 150.
    xlabel : str
        X-axis label. Default 'Extrapolation Order'.
    ylabel : str
        Y-axis label. Default 'ZNE Extrapolated Energy'.
    title_fontsize : int
        Title font size. Default 13.
    label_fontsize : int
        Axis label font size. Default 12.
    tick_fontsize : int
        Tick label font size. Default 11.
    legend_fontsize : int
        Legend font size. Default 10.
    show_legend : bool
        Master switch for legend. Default True.
    legend_loc : str
        Matplotlib loc string. Default 'best'.
    legend_bbox : tuple of float, optional
        Explicit bbox_to_anchor (x, y) in axes coordinates.
    legend_ncols : int
        Number of legend columns. Default 1.
    figure_title : str, optional
        Text placed via fig.text() at an arbitrary figure position.
    figure_title_x : float
        X position of figure_title in figure coordinates. Default 0.5.
    figure_title_y : float
        Y position of figure_title in figure coordinates. Default -0.01.
    figure_title_ha : str
        Horizontal alignment of figure_title. Default 'center'.
    figure_title_va : str
        Vertical alignment of figure_title. Default 'top'.
    figure_title_fontsize : int
        Font size for figure_title. Default 12.
    annotate_values : bool
        Annotate ZNE mean value on each point. Default True.
    annotation_fontsize : int
        Font size for value annotations. Default 8.
    grid_style : dict, optional
        kwargs for ax.grid(). Default {"linestyle": "--", "alpha": 0.4}.
    capsize : int
        Errorbar cap size. Default 5.
    marker_size : float
        Marker size. Default 6.
    border_width : float
        Spine line width. Default 1.5.
    save_format : str
        File format: 'eps', 'png', 'pdf', etc. Default 'eps'.
    show_plot : bool
        Call plt.show() after saving. Default True.
    print_data : bool
        Pretty-print each label's data to stdout. Default False.

    Returns
    -------
    matplotlib.figure.Figure
    """

    # --- Defaults ---
    if grid_style is None:
        grid_style = {"linestyle": "--", "alpha": 0.4}

    # --- Sort labels by order ---
    labels  = sorted(DATA.keys(), key=lambda k: DATA[k]["order"])
    orders  = [DATA[k]["order"]     for k in labels]
    means   = [DATA[k]["zne_mean"] for k in labels]
    stds    = [DATA[k]["zne_std"]  for k in labels]
    
    first = DATA[labels[0]]
    _exact_sol     = exact_sol if exact_sol is not None else first.get("exact_sol")
    _noise_off     = first.get("mean_noise_off")
    _noise_off_std = first.get("std_noise_off")
    _unmitigated_mean = first.get("mean_exp_vals", [None])[0]
    _unmitigate_sd = first.get("std_exp_vals", [None])[0]

    if print_data:
        for k in labels:
            print(f"\n[{k}]")
            for key, val in DATA[k].items():
                print(f"  {key}: {val}")

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")

    # ZNE mean ± std
    ax.errorbar(
        orders, means, yerr=stds,
        fmt="o-",
        capsize=capsize,
        markersize=marker_size,
        color=plot_colors.get("zne", "steelblue"),
        label="ZNE",
        zorder=3,
    )

    # Exact solution
    if _exact_sol is not None:
        ax.axhline(
            _exact_sol,
            color=plot_colors.get("exact", "red"),
            linestyle="--",
            label=f"Exact solution: {_exact_sol:.4f}",
        )
    # Unmitigated
    ax.axhline(
        _unmitigated_mean,
        color=plot_colors.get("unmitigated", "orange"),
        linestyle=":",
        label=f"Unmitigated: {_unmitigated_mean:.4f}",
    )
    ax.axhspan(
            _unmitigated_mean - _unmitigate_sd,
             _unmitigated_mean + _unmitigate_sd,
                alpha=0.12,
                color=plot_colors.get("unmitigated", "orange"),
            )
    # Noise-off reference
    if _noise_off is not None:
        ax.axhline(
            _noise_off,
            color=plot_colors.get("noise_off", "green"),
            linestyle=":",
            label=f"Noise-off: {_noise_off:.4f}",
        )
        if _noise_off_std is not None:
            ax.axhspan(
                _noise_off - _noise_off_std,
                _noise_off + _noise_off_std,
                alpha=0.12,
                color=plot_colors.get("noise_off", "green"),
            )

    # Annotate ZNE values
    if annotate_values:
        for order, mean in zip(orders, means):
            ax.annotate(
                f"{mean:.4f}",
                xy=(order, mean),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=annotation_fontsize,
                color=plot_colors.get("zne", "steelblue"),
            )

    # --- Axes formatting ---
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xticks(orders)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(**grid_style)

    for spine in ax.spines.values():
        spine.set_linewidth(border_width)

    # --- Legend ---
    if show_legend:
        legend_kwargs = dict(
            fontsize=legend_fontsize,
            loc=legend_loc,
            ncols=legend_ncols,
            frameon=False,
        )
        if legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox
        ax.legend(**legend_kwargs)

    # --- Figure title / caption ---
    if figure_title is not None:
        ax.set_title(figure_title, fontsize=title_fontsize)
        fig.text(
            figure_title_x, figure_title_y,
            figure_title,
            ha=figure_title_ha,
            va=figure_title_va,
            fontsize=figure_title_fontsize,
        )

    #plt.tight_layout()

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{plot_file_name}.{save_format}")
    
    fig.savefig(out_path, format=save_format, dpi=dpi)
    print(f"✅ Saved: {out_path}")

    if show_plot:
        plt.show()
    plt.close(fig)
    return fig