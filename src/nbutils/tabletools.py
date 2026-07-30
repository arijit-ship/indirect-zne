import math
import matplotlib
import matplotlib.pyplot as plt
import os


# ── Helper: convert PROCESSED_SIM_DATA → ordered list ────────────────────────
def build_data_list(processed_sim_data: dict) -> list:
    """
    Converts PROCESSED_SIM_DATA["ZNE-single-var"] into the flat list expected
    by make_zne_table(), ordered from most noise levels (ric7) to fewest (ric2).
    """
    zne_data = processed_sim_data["ZNE-single-var"]
    keys_sorted = sorted(
        zne_data.keys(),
        key=lambda k: len(zne_data[k]["sorted_noise"]),
        reverse=True,
    )
    return [zne_data[k] for k in keys_sorted]


# ── Core table function ───────────────────────────────────────────────────────
def make_zne_table(
    data_list: list,
    fig_width: float,          # final rendered width in inches — required
    dpi: int,                  # output DPI — required
    output_dir: str = "reports",
    filename_stem: str = "zne_table",
    row_height: float = 0.28,  # inches per row; tune to control table height
    fontsize: float = 8,
    save_eps: bool = True,
    save_png: bool = True,
) -> plt.Figure:
    """
    Parameters
    ----------
    data_list : list of dicts
        Each dict must contain:
            sorted_noise   – list of noise levels used
            mean_exp_vals  – list of mean expectation values per noise level
            std_exp_vals   – list of std dev per noise level
            zne_mean       – ZNE extrapolated mean
            zne_std        – ZNE extrapolated std
            mean_noise_off – noise-free estimation mean
            std_noise_off  – noise-free estimation std
        Ordered from most noise levels (ric7) to fewest (ric2).
    fig_width : float
        Final rendered figure width in inches (strictly honoured).
    dpi : int
        Output DPI (strictly honoured). Also sets rcParams for display.
    output_dir : str
        Directory to write output files.
    filename_stem : str
        Base filename without extension.
    row_height : float
        Height of each table row in inches.
    fontsize : float
        Font size for all table text (points).
    save_eps : bool
    save_png : bool

    Returns
    -------
    matplotlib Figure
    """

    # ── Reference entry: most noise levels (first in list) ───────────────────
    ref = data_list[0]

    # ── Build row labels and value strings ────────────────────────────────────
    row_labels = []
    row_values = []

    # Section 1 – Noise-free
    row_labels.append("Noise-free estimation")
    row_values.append(
        f"{ref['mean_noise_off']:.3f} \u00b1 {ref['std_noise_off']:.3f}"
    )

    # Section 2 – Individual noise levels
    for idx, (noise_val, mean_val, std_val) in enumerate(
        zip(ref["sorted_noise"], ref["mean_exp_vals"], ref["std_exp_vals"])
    ):
        label = ("Base noise=" + str(noise_val)) if idx == 0 else ("Boosted noise=" + str(noise_val))
        row_labels.append(label)
        row_values.append(f"{mean_val:.3f} \u00b1 {std_val:.3f}")

    # Section 3 – ZNE values, 2-point → N-point
    for entry in reversed(data_list):
        n_pts = len(entry["sorted_noise"])
        row_labels.append(f"{n_pts}-point Richardson (ric-{n_pts}) ZNE value")
        row_values.append(f"{entry['zne_mean']:.3f} \u00b1 {entry['zne_std']:.3f}")

    # ── Section divider positions ─────────────────────────────────────────────
    n_noise_rows   = len(ref["sorted_noise"])
    section_breaks = [1, 1 + n_noise_rows]

    # ── Figure ────────────────────────────────────────────────────────────────
    n_rows = len(row_labels)
    fig_h  = n_rows * row_height

    matplotlib.rcParams["figure.dpi"] = dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)

    # Remove all automatic padding — preserves exact figsize on save.
    # Do NOT use bbox_inches="tight" in savefig; it resizes the canvas.
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    PAD_L = 0.012
    PAD_R = 0.012

    # ── Draw rows ─────────────────────────────────────────────────────────────
    for i, (label, value) in enumerate(zip(row_labels, row_values)):
        y_center = n_rows - i - 0.5
        bold     = (i == 0)

        ax.text(
            PAD_L, y_center, label,
            va="center", ha="left",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            transform=ax.transData,
        )
        ax.text(
            1 - PAD_R, y_center, value,
            va="center", ha="right",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            transform=ax.transData,
        )

    # ── Horizontal rules ──────────────────────────────────────────────────────
    ax.axhline(n_rows, color="black", lw=1.0)
    ax.axhline(0,      color="black", lw=1.0)
    for sb in section_breaks:
        ax.axhline(n_rows - sb, color="black", lw=0.6)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    if save_eps:
        eps_path = os.path.join(output_dir, f"{filename_stem}.eps")
        fig.savefig(eps_path, format="eps", dpi=dpi)
        print(f"Saved EPS → {eps_path}  |  width={fig.get_figwidth():.4f} in  dpi={dpi}")

    if save_png:
        png_path = os.path.join(output_dir, f"{filename_stem}.png")
        fig.savefig(png_path, format="png", dpi=dpi)
        from PIL import Image
        img = Image.open(png_path)
        print(
            f"Saved PNG → {png_path}  |  "
            f"{img.size[0]}x{img.size[1]} px  "
            f"({img.size[0]/dpi:.4f} x {img.size[1]/dpi:.4f} in at {dpi} dpi)"
        )
    plt.close(fig)
    # ── Generate LaTeX String ────────────────────────────────────────────────
    latex_lines = ["\\hline"]

    # Section 1: Noise-free
    latex_lines.append(
        f"        Noise-free estimation & ${ref['mean_noise_off']:.3f} \\pm {ref['std_noise_off']:.3f} $\\\\"
    )
    latex_lines.append("        \\hline")

    # Section 2: Individual noise levels
    for idx, (noise_val, mean_val, std_val) in enumerate(
        zip(ref["sorted_noise"], ref["mean_exp_vals"], ref["std_exp_vals"])
    ):
        label = f"At base noise $\Lambda_0$" if idx == 0 else f"At $\Lambda_{idx}$"
        latex_lines.append(
            f"        {label} $= {noise_val}$ & ${mean_val:.3f} \\pm {std_val:.3f} $\\\\"
        )
    latex_lines.append("        \\hline")

    # Section 3: ZNE Values (No bold)
    for entry in reversed(data_list):
        n_pts = len(entry["sorted_noise"])
        #label = f"\\text{{{n_pts}-point Richardson (RIC-{n_pts}) ZNE value}}"
        label = f"ZNE of order {n_pts-1}"
        val_str = f"{entry['zne_mean']:.3f} \\pm {entry['zne_std']:.3f}"
        latex_lines.append(f"        {label} & ${val_str}$ \\\\")
    latex_lines.append("        \\hline")

    # Join and print
    latex_output = "\n".join(latex_lines)
    print("\n--- LaTeX Table Code ---")
    print(latex_output)
    print("------------------------\n")

    # Optional: Save to .tex file
    with open(os.path.join(output_dir, f"{filename_stem}.tex"), "w") as f:
        f.write(latex_output)

    return fig

def make_multivatiate_zne_table(
    data: dict,
    fig_width: float,
    dpi: int,
    output_dir: str = "reports",
    filename_stem: str = "zne_table",
    row_height: float = 0.28,
    fontsize: float = 8,
    save_eps: bool = True,
    save_png: bool = True,
) -> plt.Figure:

    # ── Build row labels and value strings ────────────────────────────────────
    row_labels = []
    row_values = []

    # Section 1 – Noise-free
    row_labels.append("Noise-free estimation")
    row_values.append(
        f"{data['mean_noise_off']:.3f} \u00b1 {data['std_noise_off']:.3f}"
    )

    # Section 2 – Individual noise levels
    for idx, (noise_vec, mean_val, std_val) in enumerate(
        zip(data["sorted_noise"], data["mean_exp_vals"], data["std_exp_vals"])
    ):
        noise_str = str(noise_vec)
        label = ("Base noise=" + noise_str) if idx == 0 else ("Boosted noise=" + noise_str)
        row_labels.append(label)
        row_values.append(f"{mean_val:.3f} \u00b1 {std_val:.3f}")

    # Section 3 – Single ZNE value
    n_pts = len(data["sorted_noise"])
    row_labels.append(f"Multi-variate Richardson ZNE value")
    row_values.append(f"{data['zne_mean']:.3f} \u00b1 {data['zne_std']:.3f}")

    # ── Section divider positions ─────────────────────────────────────────────
    n_noise_rows   = len(data["sorted_noise"])
    section_breaks = [1, 1 + n_noise_rows]

    # ── Figure ────────────────────────────────────────────────────────────────
    n_rows = len(row_labels)
    fig_h  = n_rows * row_height

    matplotlib.rcParams["figure.dpi"] = dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    PAD_L = 0.012
    PAD_R = 0.012

   # ── Draw rows ─────────────────────────────────────────────────────────────
    for i, (label, value) in enumerate(zip(row_labels, row_values)):
        y_center = n_rows - i - 0.5
        bold     = (i == 0)

        # Serial number (skip for first and last rows)
        if i > 0 and i < n_rows - 1:
            ax.text(
                PAD_L, y_center, f"{i}.",
                va="center", ha="left",
                fontsize=fontsize,
                fontweight="normal",
                transform=ax.transData,
            )
            label_x = PAD_L + 0.05  # indent label to make room for number
        else:
            label_x = PAD_L

        ax.text(
            label_x, y_center, label,
            va="center", ha="left",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            transform=ax.transData,
        )
        ax.text(
            1 - PAD_R, y_center, value,
            va="center", ha="right",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            transform=ax.transData,
        )

    # ── Horizontal rules ──────────────────────────────────────────────────────
    ax.axhline(n_rows, color="black", lw=1.0)
    ax.axhline(0,      color="black", lw=1.0)
    for sb in section_breaks:
        ax.axhline(n_rows - sb, color="black", lw=0.6)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    if save_eps:
        eps_path = os.path.join(output_dir, f"{filename_stem}.eps")
        fig.savefig(eps_path, format="eps", dpi=dpi)
        print(f"Saved EPS -> {eps_path}  |  width={fig.get_figwidth():.4f} in  dpi={dpi}")

    if save_png:
        png_path = os.path.join(output_dir, f"{filename_stem}.png")
        fig.savefig(png_path, format="png", dpi=dpi)
        from PIL import Image
        img = Image.open(png_path)
        print(
            f"Saved PNG -> {png_path}  |  "
            f"{img.size[0]}x{img.size[1]} px  "
            f"({img.size[0]/dpi:.4f} x {img.size[1]/dpi:.4f} in at {dpi} dpi)"
        )
    plt.close(fig)

    # ── Generate LaTeX String ─────────────────────────────────────────────────
    latex_lines = ["\\hline"]

    # Section 1: Noise-free
    latex_lines.append(
        f"        Noise-free estimation & ${data['mean_noise_off']:.3f} \\pm {data['std_noise_off']:.3f} $\\\\"
    )
    latex_lines.append("        \\hline")

    # Section 2: Individual noise levels
    for idx, (noise_vec, mean_val, std_val) in enumerate(
        zip(data["sorted_noise"], data["mean_exp_vals"], data["std_exp_vals"])
    ):
        noise_str = str(noise_vec)
        label = "At base noise" if idx == 0 else "At boosted noise"
        latex_lines.append(
            f"        {label} $= {noise_str}$ & ${mean_val:.3f} \\pm {std_val:.3f} $\\\\"
        )
    latex_lines.append("        \\hline")

    # Section 3: ZNE value
    label = f"\\text{{Multi-variate Richardson ZNE value}}"
    val_str = f"{data['zne_mean']:.3f} \\pm {data['zne_std']:.3f}"
    latex_lines.append(f"        {label} & ${val_str}$ \\\\")
    latex_lines.append("        \\hline")

    latex_output = "\n".join(latex_lines)
    print("\n--- LaTeX Table Code ---")
    print(latex_output)
    print("------------------------\n")

    with open(os.path.join(output_dir, f"{filename_stem}.tex"), "w") as f:
        f.write(latex_output)

    return fig


def make_zne_order_table(
    mul_var_data: dict,          # PROCESSED_SIM_DATA["ZNE-mul-var"]
    fig_width: float,
    dpi: int,
    output_dir: str = "reports",
    filename_stem: str = "zne_order_table",
    row_height: float = 0.28,
    fontsize: float = 8,
    save_eps: bool = True,
    save_png: bool = True,
) -> plt.Figure:

    # ── Sort entries by degree field ─────────────────────────────────────────
    sorted_keys = sorted(mul_var_data.keys(), key=lambda k: mul_var_data[k]["degree"])

    # ── Build row labels and value strings ───────────────────────────────────
    row_labels = []
    row_values = []

    for key in sorted_keys:
        entry = mul_var_data[key]
        degree = entry["degree"]
        row_labels.append(f"{key} (degree {degree})")
        row_values.append(
            f"{entry['zne_mean']:.3f} \u00b1 {entry['zne_std']:.3f}"
        )

    # ── Figure ───────────────────────────────────────────────────────────────
    n_rows = len(row_labels)
    fig_h  = n_rows * row_height

    matplotlib.rcParams["figure.dpi"] = dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    PAD_L = 0.012
    PAD_R = 0.012

    # ── Draw rows ────────────────────────────────────────────────────────────
    for i, (label, value) in enumerate(zip(row_labels, row_values)):
        y_center = n_rows - i - 0.5

        ax.text(
            PAD_L, y_center, label,
            va="center", ha="left",
            fontsize=fontsize,
            fontweight="normal",
            transform=ax.transData,
        )
        ax.text(
            1 - PAD_R, y_center, value,
            va="center", ha="right",
            fontsize=fontsize,
            fontweight="normal",
            transform=ax.transData,
        )

    # ── Horizontal rules ─────────────────────────────────────────────────────
    ax.axhline(n_rows, color="black", lw=1.0)
    ax.axhline(0,      color="black", lw=1.0)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    if save_eps:
        eps_path = os.path.join(output_dir, f"{filename_stem}.eps")
        fig.savefig(eps_path, format="eps", dpi=dpi)
        print(f"Saved EPS -> {eps_path}  |  width={fig.get_figwidth():.4f} in  dpi={dpi}")

    if save_png:
        png_path = os.path.join(output_dir, f"{filename_stem}.png")
        fig.savefig(png_path, format="png", dpi=dpi)
        from PIL import Image
        img = Image.open(png_path)
        print(
            f"Saved PNG -> {png_path}  |  "
            f"{img.size[0]}x{img.size[1]} px  "
            f"({img.size[0]/dpi:.4f} x {img.size[1]/dpi:.4f} in at {dpi} dpi)"
        )
    plt.close(fig)

    # ── Generate LaTeX String ────────────────────────────────────────────────
    latex_lines = ["\\hline"]

    for key in sorted_keys:
        entry = mul_var_data[key]
        degree = entry["degree"]
        label = f"Order {degree}"
        val_str = f"{entry['zne_mean']:.3f} \\pm {entry['zne_std']:.3f}"
        latex_lines.append(f"        {label} & ${val_str}$ \\\\")

    latex_lines.append("        \\hline")

    latex_output = "\n".join(latex_lines)
    print("\n--- LaTeX Table Code ---")
    print(latex_output)
    print("------------------------\n")

    with open(os.path.join(output_dir, f"{filename_stem}.tex"), "w") as f:
        f.write(latex_output)

    return fig



def to_sci_notation(value: float, sig_figs: int = 3) -> str:
    """
    Convert a float into a LaTeX '$a.bc \\times 10^{n}$' style string,
    suitable for journal tables.

    Examples
    --------
    12.25              -> "$1.23 \\times 10^{1}$"
    165.765625         -> "$1.66 \\times 10^{2}$"
    98482.74544026295  -> "$9.85 \\times 10^{4}$"
    """
    if value == 0:
        return f"$0.00 \\times 10^{{0}}$"

    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10 ** exponent)

    # Guard against rounding pushing mantissa to 10.xx (e.g. 9.999 -> 10.00)
    mantissa_str = f"{mantissa:.{sig_figs - 1}f}"
    if float(mantissa_str) >= 10:
        mantissa = mantissa / 10
        exponent += 1
        mantissa_str = f"{mantissa:.{sig_figs - 1}f}"

    return f"${mantissa_str} \\times 10^{{{exponent}}}$"


def make_zne_order_table_latex(mul_var_data: dict, cost_sig_figs: int = 3) -> str:
    """
    Build the LaTeX table string (as in the boilerplate) directly from
    mul_var_data == PROCESSED_SIM_DATA["ZNE-mul-var"].

    mul_var_data: dict of entries (one per ZNE run), each entry containing
        - 'degree'          : ZNE order
        - 'zne_mean'        : ZNE mitigated estimate (mean)
        - 'zne_std'         : ZNE mitigated estimate (std)
        - 'mean_noise_off'  : noise-free estimation mean (same across entries)
        - 'std_noise_off'   : noise-free estimation std  (same across entries)
        - 'mean_exp_vals'   : list, [0] is the unmitigated mean
        - 'std_exp_vals'    : list, [0] is the unmitigated std
        - 'cost_mean'       : ZNE sampling overhead (c)

    Returns
    -------
    str : full LaTeX table string, ready to print / write to file.
    """
    # ── Sort entries by ZNE order (degree) ──────────────────────────────────
    sorted_keys = sorted(mul_var_data.keys(), key=lambda k: mul_var_data[k]["degree"])

    # ── Pull noise-free / unmitigated values (identical across entries) ─────
    any_entry = mul_var_data[sorted_keys[0]]
    noise_free_mean = any_entry["mean_noise_off"]
    noise_free_std = any_entry["std_noise_off"]
    unmitigated_mean = any_entry["mean_exp_vals"][0]
    unmitigated_std = any_entry["std_exp_vals"][0]

    # ── Header ────────────────────────────────────────────────────────────
    lines = []
    lines.append(r"\begin{table}")
    lines.append(
        r"\caption{\textbf{Multi-variate ZNE in plot Figure "
        r"\ref{fig-plots-zne-diff-orders-multi-variate}. Mean values and "
        r"corresponding standard deviations are computed over 10 independent "
        r"experimental runs.}}"
    )
    lines.append(r"\label{table-multi-var-zne}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{|p{71pt}|p{71pt}|p{71pt}|}")
    lines.append(r"\hline")
    lines.append(
        r"Quantity & Estimated Value (Mean $\pm$ Std. Dev.) & "
        r"ZNE Sampling Overhead $(c)$ \\"
    )
    lines.append(r"\hline")

    # ── Fixed rows: noise-free & unmitigated ─────────────────────────────────
    lines.append(
        f"Noise-free estimation & ${noise_free_mean:.3f} \\pm {noise_free_std:.3f}$ & -- \\\\"
    )
    lines.append(r"\hline")
    lines.append(
        f"Unmitigated & ${unmitigated_mean:.3f} \\pm {unmitigated_std:.3f}$ & -- \\\\"
    )
    lines.append(r"\hline")

    # ── ZNE order rows ────────────────────────────────────────────────────
    for key in sorted_keys:
        entry = mul_var_data[key]
        degree = entry["degree"]
        zne_mean = entry["zne_mean"]
        zne_std = entry["zne_std"]
        cost = entry["cost_mean"]
        cost_str = to_sci_notation(cost, sig_figs=cost_sig_figs)
        lines.append(
            f"ZNE of order {degree} & ${zne_mean:.3f} \\pm {zne_std:.3f}$ & {cost_str} \\\\"
        )
    lines.append(r"\hline")

    # ── Footnote row ──────────────────────────────────────────────────────
    lines.append(
        r"\multicolumn{3}{p{215pt}}{All values are expressed in dimensionless "
        r"units. Due to the large number of data points, we do not include "
        r"them explicitly here. For more detailed data used in this "
        r"multi-variate extrapolation, refer to \cite{indirect-zne-github}.} \\"
    )
    lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick sanity check with (a trimmed version of) the provided data structure
    mul_var_data = {
        "muld2": {"mean_noise_off": -9.65844800304873, "std_noise_off": 0.05286937954035172,
                   "mean_exp_vals": [-7.028104805115262], "std_exp_vals": [0.158371805293024],
                   "zne_mean": -9.365709494803022, "zne_std": 0.09252120925043855,
                   "degree": 2, "cost_mean": 165.76562499999915},
        "muld1": {"mean_noise_off": -9.65844800304873, "std_noise_off": 0.05286937954035172,
                   "mean_exp_vals": [-7.028104805115262], "std_exp_vals": [0.158371805293024],
                   "zne_mean": -8.816096372239517, "zne_std": 0.12436742710641012,
                   "degree": 1, "cost_mean": 12.25},
        "muld5": {"mean_noise_off": -9.65844800304873, "std_noise_off": 0.05286937954035172,
                   "mean_exp_vals": [-7.028104805115261], "std_exp_vals": [0.158371805293024],
                   "zne_mean": -9.513671206824233, "zne_std": 0.06799107480719531,
                   "degree": 5, "cost_mean": 98482.74544026295},
        "muld3": {"mean_noise_off": -9.65844800304873, "std_noise_off": 0.05286937954035172,
                   "mean_exp_vals": [-7.028104805115261], "std_exp_vals": [0.158371805293024],
                   "zne_mean": -9.524398265652781, "zne_std": 0.07641775243479651,
                   "degree": 3, "cost_mean": 1614.4072875976015},
        "muld4": {"mean_noise_off": -9.65844800304873, "std_noise_off": 0.05286937954035172,
                   "mean_exp_vals": [-7.028104805115262], "std_exp_vals": [0.158371805293024],
                   "zne_mean": -9.544936237854959, "zne_std": 0.06975206566945787,
                   "degree": 4, "cost_mean": 12941.062088941126},
    }

    print(make_zne_order_table_latex(mul_var_data))