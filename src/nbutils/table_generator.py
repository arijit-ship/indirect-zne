import os
from pathlib import Path
import pandas as pd

def generate_latex_table(all_data, file_name, output_dir, ansatz_suffix="ric4", caption_ref="graph-noisefree-timevo", table_label="table-noisefree-timeevo-richardson"):
    """
    Generates a LaTeX table using Pandas DataFrame and to_latex() method,
    ensuring the specific styling, caption, and label formatting are met.

    Args:
        all_data (dict): A nested dictionary containing the estimation results.
        file_name (str): The name of the file to save the LaTeX output.
        output_dir (str): The directory where the file should be saved.
        ansatz_suffix (str): The suffix (e.g., 'ric4') used to identify the relevant keys.
        caption_ref (str): The figure label for the table's caption (e.g., 'graph-noisefree-timevo').
        table_label (str): The specific LaTeX label for the table (e.g., 'table-noisefree-timeevo-richardson').

    Returns:
        str: The generated LaTeX table string.
    """
    def fmt_latex(m, s):
        """Formats mean and std dev into LaTeX math mode string: $m \pm s$."""
        # Using three decimal places and the requested math environment
        return f"${m:.3f} \\pm {s:.3f}$"

    # --- 1. Data Restructuring ---

    ansatz_map = {
        "XY-ansatz": "xy",
        "Ising-ansatz": "ising",
        "Heisenberg-ansatz": "heisenberg"
    }
    
    # 1.1 Find the full data keys for the given suffix
    data_keys = {}
    for display_name, prefix in ansatz_map.items():
        found_key = [k for k in all_data if k.startswith(prefix) and ansatz_suffix in k]
        if not found_key:
            print(f"Error: Could not find key for ansatz prefix '{prefix}' and suffix '{ansatz_suffix}'.")
            return f"% ERROR: Missing data for {prefix}_{ansatz_suffix}"
        data_keys[display_name] = found_key[0]
    
    # Get the keys for easier access
    xy_key = data_keys["XY-ansatz"]
    ising_key = data_keys["Ising-ansatz"]
    heis_key = data_keys["Heisenberg-ansatz"]
    
    # 1.2 Prepare Noise-free Row
    data_rows = []
    noise_free_row = ["Noise-free estimation"]
    noise_free_row.append(fmt_latex(all_data[xy_key]['noiseoff']['mean'], all_data[xy_key]['noiseoff']['std']))
    noise_free_row.append(fmt_latex(all_data[ising_key]['noiseoff']['mean'], all_data[ising_key]['noiseoff']['std']))
    noise_free_row.append(fmt_latex(all_data[heis_key]['noiseoff']['mean'], all_data[heis_key]['noiseoff']['std']))
    data_rows.append(noise_free_row)

    # 1.3 Prepare Boosted Noise Rows
    # Use XY data to drive the row labels/levels, as it typically has all of them
    xy_redundant = all_data[xy_key].get('redundant', {})
    noise_levels = xy_redundant.get('sorted_noise_levels', [])

    for i, nl in enumerate(noise_levels):
        row = []
        # Custom label logic: Base noise for the first, Boosted for the rest
        label = f"Base noise={nl}" if i == 0 else f"Boosted noise={nl}"
        row.append(label)
        
        # Populate data for each ansatz
        for display_name, key in data_keys.items():
            means = all_data[key]['redundant']['mean']
            stds = all_data[key]['redundant']['std']
            
            if i < len(means):
                row.append(fmt_latex(means[i], stds[i]))
            else:
                row.append("") # Should not happen if noise_levels is correctly maxed
        
        data_rows.append(row)

    # 1.4 Prepare ZNE Row
    zne_row = ["Richardson ZNE value"]
    zne_row.append(fmt_latex(all_data[xy_key]['zne']['mean'], all_data[xy_key]['zne']['std']))
    zne_row.append(fmt_latex(all_data[ising_key]['zne']['mean'], all_data[ising_key]['zne']['std']))
    zne_row.append(fmt_latex(all_data[heis_key]['zne']['mean'], all_data[heis_key]['zne']['std']))
    data_rows.append(zne_row)

    # --- 2. Create Pandas DataFrame ---
    
    df = pd.DataFrame(data_rows, columns=["", *ansatz_map.keys()])
    # Set the first column as the index for cleaner LaTeX conversion
    df = df.set_index("")

    # --- 3. Generate LaTeX Output with Custom Styling ---

    # The to_latex method generates the tabular environment.
    latex_tabular = df.to_latex(
        header=True, 
        index=True,
        # Escape=False is crucial so that LaTeX math ($...$) strings are not escaped
        escape=False,
        # Alignments: l for the index column, lll for the data columns
        column_format='@{}llll@{}', 
        # booktabs=True gives us \toprule, \midrule, \bottomrule
        # We rely on manual string replacements for custom midrules.
        position='h',
    )
    
    # 3.1 Manually inject custom components and midrules

    # Fix 1: Wrap in the final, desired format
    final_latex_output = r"""\begin{table}[h]
\centering
\caption{Result summary for the plots in Figure \ref{""" + caption_ref + r"""}.}
""" + latex_tabular + r"""\label{""" + table_label + r"""}
\end{table}"""
    
    # Fix 2: Insert the required \midrule after "Noise-free estimation"
    # This splits the Noise-free row from the Boosted rows
    final_latex_output = final_latex_output.replace("Noise-free estimation \\\\", "Noise-free estimation \\\\\n\\midrule")

    # Fix 3: Remove the extra \midrule that booktabs puts before the ZNE row
    # This ensures there is only ONE \midrule separating the Boosted rows from the ZNE row.
    final_latex_output = final_latex_output.replace("\\midrule\n\\midrule", "\\midrule")
    
    # Fix 4: Remove the extra \midrule at the very bottom of the table, as \bottomrule should be last
    final_latex_output = final_latex_output.replace("\\midrule\n\\bottomrule", "\\bottomrule")
    
    # Clean up any residual alignment issues from Pandas inserting spaces after \\
    final_latex_output = final_latex_output.replace(r" \\ ", r" \\")

    # --- 4. File Writing ---

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    full_path = output_path / file_name

    try:
        with open(full_path, 'w') as f:
            f.write(final_latex_output)
        print(f"Successfully wrote LaTeX table to: {full_path}")
    except Exception as e:
        print(f"Error writing file {full_path}: {e}")

    return final_latex_output