from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import os
from pprint import pprint

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
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

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
        label="Exact Solution",
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

from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint


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
    title_fontsize: int = 13,
    label_fontsize: int = 12,
    tick_fontsize: int = 11,
    legend_fontsize: int = 10,
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
            label=f"{noise_type.capitalize()} estimation",
            color=plot_colors["noisy"],
            markersize=marker_size,
            markeredgewidth=0.8,
            elinewidth=1,
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
            ax.errorbar(
                x=0,
                y=mean_noise_off,
                yerr=std_noise_off if std_noise_off is not None else 0,
                fmt="*",
                ecolor=plot_colors["noise_free"],
                capsize=capsize,
                label="Noise-free estimation",
                color=plot_colors["noise_free"],
                markersize=marker_size + 2,
                markeredgewidth=1,
                elinewidth=1,
            )

        # --- Exact solution ---
        ax.axhline(
            y=exact_sol,
            color=plot_colors["exact"],
            linestyle="--",
            linewidth=1.5,
            label="Exact Solution",
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
    fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches="tight")
    print(f"✅ Figure saved as (in '{output_dir}' folder): {base_name}.{save_format}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig