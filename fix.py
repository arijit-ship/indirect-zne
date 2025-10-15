def compile_zne_subplots(
    data:Dict[str, Any],
    models,
    plot_titles,
    plot_colors,
    exact_solution,
    extrapol_target,
    timestamp,
    output_dir=f"reports/{timestamp}/plots",
    filename_prefix="compiled_zne",
    ncols=3,
    figsize=(9, 6),
    show=True,
):
    """
    Create a compiled subplot figure.
    """

    nplots = len(models)
    nrows = (nplots + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axs = axs.flatten() if nplots > 1 else [axs]

    for i, model in enumerate(models):
        ax = axs[i]
        DATA = data[model]
        prefix = model.split("-")[0]
        title = plot_titles.get(prefix, model)

        # --- Noisy estimation ---
        ax.errorbar(
            x=DATA["redundant"]["sorted_noise_levels"],
            y=DATA["redundant"]["mean"],
            yerr=DATA["redundant"]["std"],
            fmt="o",
            ecolor=plot_colors[0],
            capsize=5,
            label="Noisy estimation",
            color=plot_colors[0],
            markersize=5
        )

        # --- ZNE Extrapolated ---
        ax.errorbar(
            x=extrapol_target,
            y=DATA["zne"]["mean"],
            yerr=DATA["zne"]["std"],
            fmt="o",
            ecolor=plot_colors[2],
            capsize=5,
            label="Ric. ZNE value",
            color=plot_colors[2],
            markersize=5
        )

        # --- Noise-free estimation ---
        ax.errorbar(
            x=0,
            y=DATA["noiseoff"]["mean"],
            yerr=DATA["noiseoff"]["std"],
            fmt="*",
            ecolor=plot_colors[6],
            capsize=5,
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

        # Titles and labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"Noise level ($\alpha_k\lambda$)")
        ax.grid(linestyle="--", alpha=0.6)
        if i % ncols == 0:
            ax.set_ylabel("Expectation value")

        # Legend consistent with single plot
        ax.legend(loc="upper left", fontsize=14, frameon=False)

    # Hide empty axes
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(w_pad=1.2, h_pad=0.3)
    plt.subplots_adjust(top=0.88)

    # Save compiled figure
    plot_file_name = f"{filename_prefix}.eps"
    plt.savefig(f"{output_dir}/{plot_file_name}", format="eps")
    print(f"✅ Compiled figure saved: {output_dir}/{plot_file_name}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig