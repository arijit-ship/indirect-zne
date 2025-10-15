from pprint import pprint
import os
from typing import Any, Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from datetime import datetime

def summarize_zne_results(
    ALL_PROCESSED_DATA,
    selected_models,
    output_dir: str,
    column_labels: list | None = None,
    decimals: int = 5,
    mean_std_symbol: str = "±",
    font_size: int = 10,
    row_scale: float = 1.25,
    color_rows: bool = True,
    save_name: str | None = None,
    save_formats: list = ["eps", "csv"],
    print_data: bool = True,
    figsize: tuple = (8, None)
):
    """
    Create a professional summary table for ZNE results (tight layout, no title).

    Parameters
    ----------
    ALL_PROCESSED_DATA : dict
        Nested dictionary containing processed ZNE data.
    selected_models : list
        List of model names to include.
    column_labels : list, optional
        List of column labels. Default: ["Description", "Simulation", "Mean ± Std"]
    decimals : int
        Number of decimals to show in mean ± std.
    mean_std_symbol : str
        Symbol between mean and std.
    font_size : int
        Font size for table.
    row_scale : float
        Row height scaling.
    color_rows : bool
        Whether to color rows by model.
    output_dir : str
        Directory to save files.
    save_name : str, optional
        Base file name (without extension).
    save_formats : list
        List of formats to save: any of ["eps", "png", "csv", "latex"].
    print_data : bool
        Whether to print the dataframe.
    figsize : tuple
        Figure width and optional height (height auto if None).
    """
    os.makedirs(output_dir, exist_ok=True)

    if column_labels is None:
        column_labels = ["Description", "Simulation", "Mean ± Std"]

    # --- Helper to format mean ± std ---
    def fmt_mean_std(mean, std):
        if mean is None or std is None:
            return "N/A"
        return f"{mean:.{decimals}f} {mean_std_symbol} {std:.{decimals}f}"

    # --- Prepare rows ---
    rows = []
    for model, methods in ALL_PROCESSED_DATA.items():
        if model not in selected_models:
            continue
        for method, stats in methods.items():
            if method == "redundant":
                for lvl, m, s in zip(stats['sorted_noise_levels'], stats['mean'], stats['std']):
                    rows.append([model, f"{method} (noise={lvl})", fmt_mean_std(m, s)])
            elif stats:
                rows.append([model, method, fmt_mean_std(stats['mean'], stats['std'])])
            else:
                rows.append([model, method, "N/A"])
        rows.append(["", "", ""])  # empty row between models

    df = pd.DataFrame(rows, columns=column_labels)

    if print_data:
        print("=== Summary Table ===")
        pprint(df, sort_dicts=False, width=120)

    # --- Save CSV / LaTeX ---
    if save_name:
        for fmt in save_formats:
            path = f"{output_dir}/{save_name}.{fmt}"
            if fmt.lower() == "csv":
                df.to_csv(path, index=False)
            elif fmt.lower() == "latex":
                with open(path, "w") as f:
                    f.write(df.to_latex(index=False))
            elif fmt.lower() in ["eps", "png"]:
                continue
            else:
                raise ValueError(f"Unsupported format: {fmt}")

    # --- Create figure ---
    n_rows = len(df)
    if figsize[1] is None:
        fig_height = max(2, n_rows * 0.35)
    else:
        fig_height = figsize[1]

    fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
    ax.axis('off')

    # Table occupies full figure (tight layout)
    table_bbox = [0, 0, 1, 1]

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=table_bbox
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    table.scale(1, row_scale)

    # --- Color rows by model ---
    if color_rows:
        unique_models = {m.lower().strip() for m in df[column_labels[0]] if m.strip() != ""}
        colors = cm.Pastel1(np.linspace(0, 1, len(unique_models)))
        model_colors = {model: colors[i] for i, model in enumerate(sorted(unique_models))}
        for i, model_name in enumerate(df[column_labels[0]]):
            key = model_name.strip().lower()
            color = model_colors.get(key, "#f0f0f0") if key != "" else "#f0f0f0"
            for j in range(len(df.columns)):
                table[(i + 1, j)].set_facecolor(color)

    # --- Save figure ---
    if save_name:
        for fmt in save_formats:
            if fmt.lower() in ["eps", "png"]:
                path = f"{output_dir}/{save_name}.{fmt}"
                plt.savefig(path, format=fmt.lower(), bbox_inches='tight')
                print(f"✅ Saved figure: {path}")

    plt.show()
    return df, fig