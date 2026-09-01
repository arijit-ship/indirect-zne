# Standard library imports
import colorsys
import json
import math
import os
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# DATA PROCESSING
# ============================================================================


def load_simulation_tree(root_path: str) -> Dict[str, Any]:
    """
    Recursively crawls a directory to build a dictionary mapping the file tree.

    This function traverses the provided directory path. Folder names become
    keys in the returned dictionary, and JSON files are loaded as nested
    dictionaries. Non-JSON files are ignored.

    Args:
        root_path: The absolute or relative path to the simulation data
            directory (e.g., 'experiment10[...]').

    Returns:
        A nested dictionary where keys are directory or file names and
        values are either further nested dictionaries (for subfolders)
        or the parsed content of JSON files.

    Raises:
        FileNotFoundError: If the provided root_path does not exist.
        PermissionError: If the script lacks read permissions for the directory.

    Example:
        >>> data = load_simulation_tree("./data/tmax_5")
        >>> print(data['ZNEs']['ric4'].keys())
    """
    tree_dict = {}

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Path not found: {root_path}")

    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)

        if os.path.isdir(item_path):
            # Recursively build the tree for sub-directories (tmax_X, ZNEs, etc.)
            tree_dict[item] = load_simulation_tree(item_path)

        elif item.endswith(".json"):
            with open(item_path, "r", encoding="utf-8") as f:
                try:
                    # Use the filename without extension as the key
                    file_key = os.path.splitext(item)[0]
                    tree_dict[file_key] = json.load(f)
                except json.JSONDecodeError:
                    # Gracefully handle corrupted simulation logs
                    tree_dict[file_key] = {"error": "Invalid JSON format"}

    return tree_dict


def _process_vqe(content: dict) -> dict:
    """
    Processes the VQE folder.

    - Computes mean/std of artifacts.opt_obj.nit and artifacts.opt_obj.fun.
    - Computes mean/std of output.optimized_parameters[0][config.ansatz.layer - 1] -> t_opt.
    - Checks config.nqubits, config.ansatz.layer, output.noise_details.odd_wires, and
      output.exact_sol are consistent across runs.
    - Cross-checks optimized_minimum_cost[0] against artifacts.opt_obj.fun
      per run and reports any mismatches.
    """
    vqe_folder = content.get("VQE", {})

    nit_values     = []
    fun_values     = []
    t_opt_vals     = []
    nqubits_set    = set()
    layer_set      = set()
    odd_wires_set  = set()
    exact_sol_set  = set()
    mismatches     = 0
    checked        = 0

    for k, run in vqe_folder.items():
        if not run or "REDUNDANT" in k:
            continue

        try:
            nit_values.append(run["artifacts"]["opt_obj"]["nit"])
        except (KeyError, TypeError):
            pass

        try:
            fun_val = run["artifacts"]["opt_obj"]["fun"]
            fun_values.append(fun_val)
        except (KeyError, TypeError):
            fun_val = None

        try:
            nqubits_set.add(run["config"]["nqubits"])
        except (KeyError, TypeError):
            pass

        try:
            layer_set.add(run["config"]["ansatz"]["layer"])
        except (KeyError, TypeError):
            pass

        try:
            odd_wires = run["output"]["noise_details"]["odd_wires"]
            odd_wires_set.add(odd_wires)
        except (KeyError, TypeError):
            print(f"⚠️ VQE/{k}: could not extract output.noise_details.odd_wires")

        try:
            exact_sol_set.add(run["output"]["exact_sol"])
        except (KeyError, TypeError):
            print(f"⚠️ VQE/{k}: could not extract output.exact_sol")

        try:
            layer = run["config"]["ansatz"]["layer"]
            t_opt_vals.append(run["output"]["optimized_parameters"][0][layer - 1])
        except (KeyError, IndexError, TypeError):
            print(f"⚠️ VQE/{k}: could not extract optimized_parameters[0][layer-1]")

        try:
            cost_val = run["output"]["optimized_minimum_cost"][0]
        except (KeyError, IndexError, TypeError):
            cost_val = None

        if cost_val is not None and fun_val is not None:
            checked += 1
            if not np.isclose(cost_val, fun_val):
                mismatches += 1
                print(f"❌ VQE/{k}: optimized_minimum_cost ({cost_val}) != artifacts.opt_obj.fun ({fun_val})")

    stats = {
        "nit_mean": float(np.mean(nit_values)) if nit_values else None,
        "nit_std":  float(np.std(nit_values))  if nit_values else None,
        "fun_mean": float(np.mean(fun_values)) if fun_values else None,
        "fun_std":  float(np.std(fun_values))  if fun_values else None,
        "t_opt_mean": float(np.mean(t_opt_vals)) if t_opt_vals else None,
        "t_opt_std":  float(np.std(t_opt_vals))  if t_opt_vals else None,
    }

    if not nqubits_set:
        print("⚠️ VQE: no values found for config.nqubits")
        stats["nqubits"] = None
    elif len(nqubits_set) > 1:
        print(f"❌ VQE: inconsistent config.nqubits across runs: {sorted(nqubits_set)}")
        stats["nqubits"] = None
    else:
        stats["nqubits"] = next(iter(nqubits_set))

    if not layer_set:
        print("⚠️ VQE: no values found for config.ansatz.layer")
        stats["layer"] = None
    elif len(layer_set) > 1:
        print(f"❌ VQE: inconsistent config.ansatz.layer across runs: {sorted(layer_set)}")
        stats["layer"] = None
    else:
        stats["layer"] = next(iter(layer_set))

    if not odd_wires_set:
        print("⚠️ VQE: no values found for output.noise_details.odd_wires")
        stats["odd_site_num"] = None
    elif len(odd_wires_set) > 1:
        print(f"❌ VQE: inconsistent output.noise_details.odd_wires across runs: {sorted(odd_wires_set)}")
        stats["odd_site_num"] = None
    else:
        stats["odd_site_num"] = next(iter(odd_wires_set))

    if not exact_sol_set:
        print("⚠️ VQE: no values found for output.exact_sol")
        stats["exact_sol"] = None
    elif len(exact_sol_set) > 1:
        print(f"❌ VQE: inconsistent output.exact_sol across runs: {sorted(exact_sol_set)}")
        stats["exact_sol"] = None
    else:
        stats["exact_sol"] = next(iter(exact_sol_set))

    if checked == 0:
        print("⚠️ VQE: no runs had both optimized_minimum_cost and artifacts.opt_obj.fun to compare.")
    elif mismatches == 0:
        print(f"✅ VQE: optimized_minimum_cost matches artifacts.opt_obj.fun for all {checked} runs.")
    else:
        print(f"❌ VQE: {mismatches}/{checked} runs have optimized_minimum_cost != artifacts.opt_obj.fun.")

    print(f"✅ VQE: {len(nit_values)} runs aggregated, nit_mean={stats['nit_mean']}, fun_mean={stats['fun_mean']}, t_opt_mean={stats['t_opt_mean']}, odd_site_num={stats['odd_site_num']}, exact_sol={stats['exact_sol']}")
    return stats


def _process_vqe_noiseoff(content: dict) -> dict:
    """
    Processes the VQE_noiseoff folder.

    - Computes mean/std of artifacts.opt_obj.nit and artifacts.opt_obj.fun.
    - Checks config.nqubits and config.ansatz.layer are consistent across runs.
    - Cross-checks optimized_minimum_cost[0] against artifacts.opt_obj.fun
      per run and reports any mismatches.
    - Also returns "mean"/"std" (optimized_minimum_cost) for backward
      compatibility with ZNE processing (mean_noise_off / std_noise_off).
    """
    nf_folder = content.get("VQE_noiseoff", {})

    nit_values = []
    fun_values = []
    cost_values = []
    nqubits_set = set()
    layer_set = set()
    mismatches = 0
    checked = 0

    for k, run in nf_folder.items():
        if not run or "REDUNDANT" in k:
            continue

        try:
            nit_values.append(run["artifacts"]["opt_obj"]["nit"])
        except (KeyError, TypeError):
            pass

        try:
            fun_val = run["artifacts"]["opt_obj"]["fun"]
            fun_values.append(fun_val)
        except (KeyError, TypeError):
            fun_val = None

        try:
            nqubits_set.add(run["config"]["nqubits"])
        except (KeyError, TypeError):
            pass

        try:
            layer_set.add(run["config"]["ansatz"]["layer"])
        except (KeyError, TypeError):
            pass

        try:
            cost_val = run["output"]["optimized_minimum_cost"][0]
            cost_values.append(cost_val)
        except (KeyError, IndexError, TypeError):
            cost_val = None

        if cost_val is not None and fun_val is not None:
            checked += 1
            if not np.isclose(cost_val, fun_val):
                mismatches += 1
                print(f"❌ VQE_noiseoff/{k}: optimized_minimum_cost ({cost_val}) != artifacts.opt_obj.fun ({fun_val})")

    stats = {
        "nit_mean": float(np.mean(nit_values)) if nit_values else None,
        "nit_std": float(np.std(nit_values)) if nit_values else None,
        "fun_mean": float(np.mean(fun_values)) if fun_values else None,
        "fun_std": float(np.std(fun_values)) if fun_values else None,
        # Legacy keys, kept for _process_zne_folder (mean_noise_off / std_noise_off)
        "mean": float(np.mean(cost_values)) if cost_values else None,
        "std": float(np.std(cost_values)) if cost_values else None,
    }

    if not nqubits_set:
        print("⚠️ VQE_noiseoff: no values found for config.nqubits")
        stats["nqubits"] = None
    elif len(nqubits_set) > 1:
        print(f"❌ VQE_noiseoff: inconsistent config.nqubits across runs: {sorted(nqubits_set)}")
        stats["nqubits"] = None
    else:
        stats["nqubits"] = next(iter(nqubits_set))

    if not layer_set:
        print("⚠️ VQE_noiseoff: no values found for config.ansatz.layer")
        stats["layer"] = None
    elif len(layer_set) > 1:
        print(f"❌ VQE_noiseoff: inconsistent config.ansatz.layer across runs: {sorted(layer_set)}")
        stats["layer"] = None
    else:
        stats["layer"] = next(iter(layer_set))

    if checked == 0:
        print("⚠️ VQE_noiseoff: no runs had both optimized_minimum_cost and artifacts.opt_obj.fun to compare.")
    elif mismatches == 0:
        print(f"✅ VQE_noiseoff: optimized_minimum_cost matches artifacts.opt_obj.fun for all {checked} runs.")
    else:
        print(f"❌ VQE_noiseoff: {mismatches}/{checked} runs have optimized_minimum_cost != artifacts.opt_obj.fun.")

    print(
        f"✅ VQE_noiseoff: {len(nit_values)} runs aggregated, nit_mean={stats['nit_mean']}, fun_mean={stats['fun_mean']}"
    )
    return stats


def _extract_mul_var_run(run: dict) -> tuple | None:
    """
    Extracts from ZNE-mul-var runs.

    JSON structure:
        zne_values.sampled data  : [[n1,n2,n3,val], ...]
        zne_values.extrapolated_value
        zne_values.others.eta_coefficients   <- LRE extrapolation coefficients (eta_i),
                                                 NOT Richardson betas. See Eq. (lre-estimator).
        zne_values.degree
    """
    zne_vals = run["output"].get("zne_values", {})
    sampled = zne_vals.get("sampled data", [])
    ext_val = zne_vals.get("extrapolated_value")
    degree = zne_vals.get("degree")

    if not sampled or ext_val is None:
        return None

    noise = [row[:-1] for row in sampled]
    y_vals = [row[-1] for row in sampled]

    order = sorted(range(len(noise)), key=lambda i: sum(noise[i]))
    noise = [noise[i] for i in order]
    y_vals = [y_vals[i] for i in order]

    eta = zne_vals.get("others", {}).get("eta_coefficients") or zne_vals.get("eta_coefficients")

    # Sampling Overhead Cost Calculation (Eqs. eq-lre-sampling-c-bar / eq-lre-sampling-c)
    #   c_eq  = M * Gamma_bar^2,   Gamma_bar = sqrt( sum_i |eta_i|^2 )   -> equal shot allocation
    #   c_opt = Gamma_opt^2,       Gamma_opt = sum_i |eta_i|            -> optimal shot allocation
    cost_eq = None
    cost_opt = None
    if eta is not None:
        gamma_opt = sum(abs(e) for e in eta)
        cost_opt = gamma_opt**2
        M: int = len(eta)
        gamma_bar = np.sqrt(sum(abs(e * e) for e in eta))
        cost_eq = M * (gamma_bar**2)

    return noise, y_vals, ext_val, eta, cost_opt, degree, cost_eq


def _process_zne_folder(zne_folder: dict, zne_key: str, nf_stats: dict) -> dict:
    result = {}

    for label, runs in zne_folder.items():
        ext_vals, all_y_curves = [], []
        all_etas, all_costs_opt, all_costs_eq = [], [], []
        exact_sol_set = set()
        noise_levels = None
        tmax         = None
        noise_type   = None
        degree       = None
        raw_gate_counts = None  # filled from the first REDUNDANT entry found

        for k, run in runs.items():
            if not run:
                continue

            if "REDUNDANT" in k:
                if raw_gate_counts is None:
                    try:
                        noise_details = run["output"]["noise_details"]
                        raw_gate_counts = [
                            {
                                "identity_factors": entry["identity_factors"],
                                "raw_gate_count":   entry["raw_gate_count"],
                            }
                            for entry in noise_details
                        ]
                    except (KeyError, TypeError):
                        print(f"⚠️ ZNE/{zne_key}/{label}/{k}: could not extract output.noise_details")
                continue

            if "output" not in run:
                continue

            extracted = _extract_mul_var_run(run)
            if extracted is None:
                print(f"⚠️ Skipping ZNE/{zne_key}/{label}/{k}: Missing required keys.")
                continue

            noise, y_vals, ext_val, eta, cost_opt, degree, cost_eq = extracted
            tmax       = run["config"]["ansatz"]["ugate"]["time"]["max"]
            noise_type = run["config"]["noise_profile"].get("type", "unknown")

            exact_sol = run["output"].get("exact_sol")
            if exact_sol is not None:
                exact_sol_set.add(exact_sol)
            else:
                print(f"⚠️ ZNE/{zne_key}/{label}/{k}: missing output.exact_sol")

            ext_vals.append(ext_val)
            all_y_curves.append(y_vals)
            if eta is not None:      all_etas.append(eta)
            if cost_opt is not None: all_costs_opt.append(cost_opt)
            if cost_eq  is not None: all_costs_eq.append(cost_eq)
            if noise_levels is None:
                noise_levels = noise

        if ext_vals and all_y_curves:
            y_array = np.array(all_y_curves)

            eta_mean = np.mean(all_etas, axis=0).tolist() if all_etas else None
            eta_std  = np.std(all_etas,  axis=0).tolist() if all_etas else None

            if not exact_sol_set:
                print(f"⚠️ ZNE/{zne_key}/{label}: no output.exact_sol values found.")
                exact_sol = None
            elif len(exact_sol_set) > 1:
                print(f"❌ ZNE/{zne_key}/{label}: inconsistent output.exact_sol across runs: {sorted(exact_sol_set)}")
                exact_sol = None
            else:
                exact_sol = next(iter(exact_sol_set))

            if raw_gate_counts is None:
                print(f"⚠️ ZNE/{zne_key}/{label}: no REDUNDANT entry found for raw_gate_counts.")

            result[label] = {
                    "noise_type":     noise_type,
                    "tmax":           tmax,
                    "exact_sol":      exact_sol,
                    "sorted_noise":   noise_levels,
                    "mean_exp_vals":  np.mean(y_array, axis=0).tolist(),
                    "std_exp_vals":   np.std(y_array, axis=0).tolist(),
                    "zne_mean":       float(np.mean(ext_vals)),
                    "zne_std":        float(np.std(ext_vals)),
                    "mean_noise_off": nf_stats["mean"],
                    "std_noise_off":  nf_stats["std"],
                    "degree":         degree if degree is not None else "N/A",
                    "order":          degree if degree is not None else "N/A",  # Alias
                    # LRE eta_i extrapolation coefficients (not Richardson betas).
                    "eta_mean":       eta_mean,
                    "eta_std":        eta_std,
                    # Backward-compat aliases for any downstream code still reading beta_*.
                    "beta_mean":      eta_mean,
                    "beta_std":       eta_std,
                    # Sampling overhead, c_opt = Gamma_opt^2, Gamma_opt = sum|eta_i|  (Eq. eq-lre-sampling-c)
                    # -> optimal shot allocation: s_i ∝ |eta_i|
                    "cost_opt_mean":  float(np.mean(all_costs_opt)) if all_costs_opt else None,
                    "cost_opt_std":   float(np.std(all_costs_opt))  if all_costs_opt else None,
                    # Sampling overhead, c = M * Gamma^2, Gamma = sqrt(sum|eta_i|^2)  (Eq. eq-lre-sampling-c-bar)
                    # -> equal shot allocation: s_eq = s_tot / M per circuit
                    "cost_eq_mean":   float(np.mean(all_costs_eq)) if all_costs_eq else None,
                    "cost_eq_std":    float(np.std(all_costs_eq))  if all_costs_eq else None,
                    # Backward-compat aliases for the old (ambiguous) key names.
                    "cost_mean":      float(np.mean(all_costs_opt)) if all_costs_opt else None,
                    "cost_std":       float(np.std(all_costs_opt))  if all_costs_opt else None,
                    "cost_bar_mean":  float(np.mean(all_costs_eq)) if all_costs_eq else None,
                    "cost_bar_std":   float(np.std(all_costs_eq))  if all_costs_eq else None,
                    # Gate counts from the REDUNDANT run (all entries in a muld are identical).
                    "raw_gate_counts": raw_gate_counts,
                    }
            print(f"✅ ZNE/{zne_key}/{label}: {len(ext_vals)} runs processed")
        else:
            print(f"❌ Failed to process ZNE/{zne_key}/{label}: No valid runs found")

    return result


def get_processed_data(raw_data: dict) -> dict:
    """
    Main entry point. Expects:

        raw_data/
        └── data/
            ├── VQE/
            ├── VQE_noiseoff/
            └── ZNE/
                └── ZNE-mul-var/
                    ├── muld1/ ... muldN/

    Returns:
        {
            "VQE":          {...},   # nit/fun stats, nqubits/layer consistency, cost==fun check
            "VQE_noiseoff": {...},   # same, plus legacy "mean"/"std" (optimized_minimum_cost)
            "ZNE-mul-var":  { "muld1": {...}, ... },
        }
    """
    processed = {}
    content = raw_data.get("data", raw_data)

    # 1. VQE
    processed["VQE"] = _process_vqe(content)

    # 2. VQE_noiseoff
    nf_stats = _process_vqe_noiseoff(content)
    processed["VQE_noiseoff"] = nf_stats

    # 3. ZNE folders (unchanged)
    zne_root = content.get("ZNE", {})
    if not zne_root:
        print("❌ No ZNE folder found in data.")
        return processed

    ZNE_FOLDERS = [
        "ZNE-mul-var",
    ]

    for zne_key in ZNE_FOLDERS:
        zne_folder = zne_root.get(zne_key, {})
        if not zne_folder:
            print(f"⚠️ No data found for ZNE/{zne_key}, skipping.")
            continue
        print(f"✅ ZNE/{zne_key} labels found: {list(zne_folder.keys())}")
        processed[zne_key] = _process_zne_folder(zne_folder, zne_key, nf_stats)

    return processed


def get_plotting_data(simulation_data):
    plot_map = {}

    for exp_name, sub_categories in simulation_data.items():
        plot_map[exp_name] = {}

        for sub_key, file_list in sub_categories.items():
            zne_files = [f for f in file_list if f["type"] == "ZNE"]
            noise_off_files = [f for f in file_list if f["type"] == "noise_off"]

            if not zne_files:
                continue

            # Use the first file to establish the "Ground Truth" for this category
            first_data = zne_files[0]["data"]
            expected_noise = first_data.get("output", {}).get("zne_values", {}).get("others", {}).get("sorted_noise")
            expected_len = len(expected_noise) if expected_noise else 0

            all_extrapolated = []
            all_y_curves = []

            for f in zne_files:
                fname = f["filename"]
                out_vals = f["data"].get("output", {}).get("zne_values", {})
                others = out_vals.get("others", {})

                current_noise = others.get("sorted_noise", [])
                y_vals = others.get("sorted_expectation_vals", [])
                ext_val = out_vals.get("extrapolated_value")

                # --- Validation Checks ---

                # 1. Check one-to-one correspondence within the file
                if len(current_noise) != len(y_vals):
                    warnings.warn(
                        f"\n[MISMATCH] Internal length mismatch in file: {fname}\n"
                        f"Noise points: {len(current_noise)}, Expectation points: {len(y_vals)}"
                    )
                    continue

                # 2. Check consistency across the entire sub-folder (xy-ric#)
                if len(current_noise) != expected_len:
                    warnings.warn(
                        f"\n[INCONSISTENCY] Sample length differs from category baseline!\n"
                        f"Location: {exp_name} -> {sub_key}\n"
                        f"File: {fname}\n"
                        f"Expected: {expected_len}, Found: {len(current_noise)}"
                    )
                    continue

                if ext_val is not None:
                    all_extrapolated.append(ext_val)
                if y_vals:
                    all_y_curves.append(y_vals)

            # --- Noise-off Aggregation ---
            all_noise_off_costs = []

            for f in noise_off_files:
                fname = f["filename"]
                costs = f["data"].get("output", {}).get("optimized_minimum_cost")

                if costs is None:
                    warnings.warn(
                        f"\n[MISSING] 'optimized_minimum_cost' not found in noise_off file: {fname}\n"
                        f"Location: {exp_name} -> {sub_key}"
                    )
                    continue

                if not isinstance(costs, list) or len(costs) == 0:
                    warnings.warn(
                        f"\n[EMPTY] 'optimized_minimum_cost' is empty or not a list in noise_off file: {fname}\n"
                        f"Location: {exp_name} -> {sub_key}"
                    )
                    continue

                all_noise_off_costs.extend(costs)

            if all_noise_off_costs:
                noise_off_arr = np.array(all_noise_off_costs)
                mean_noise_off = float(np.mean(noise_off_arr))
                std_noise_off = float(np.std(noise_off_arr))
            else:
                mean_noise_off = None
                std_noise_off = None

            # --- Final Aggregation ---
            if all_extrapolated and all_y_curves:
                curves_array = np.array(all_y_curves)

                plot_map[exp_name][sub_key] = {
                    "noise_type": first_data.get("config", {}).get("noise_profile", {}).get("type"),
                    "noise_prob": first_data.get("config", {}).get("noise_profile", {}).get("noise_prob"),
                    "exact_sol": first_data.get("output", {}).get("exact_sol"),
                    "sorted_noise": expected_noise,
                    "mean_exp_vals": np.mean(curves_array, axis=0).tolist(),
                    "std_exp_vals": np.std(curves_array, axis=0).tolist(),
                    "zne_mean": np.mean(all_extrapolated),
                    "zne_std": np.std(all_extrapolated),
                    "mean_noise_off": mean_noise_off,
                    "std_noise_off": std_noise_off,
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
    noise_type = data["noise_type"]
    exact_sol = data["exact_sol"]
    sorted_noise = data["sorted_noise"]
    mean_exp_vals = data["mean_exp_vals"]
    std_exp_vals = data["std_exp_vals"]
    zne_mean = data["zne_mean"]
    zne_std = data["zne_std"]
    mean_noise_off = data.get("mean_noise_off")
    std_noise_off = data.get("std_noise_off")

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
    panel_label_y: Optional[
        float
    ] = None,  # None = top (set_title); float = axes-fraction position (e.g. -0.25 for below xlabel)
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
    nrows = (nplots + ncols - 1) // ncols

    plt.rcParams.update(
        {
            "font.size": tick_fontsize,
            "axes.labelsize": label_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, sharex=sharex, sharey=sharey)
    axs = axs.flatten() if nplots > 1 else [axs]

    shared_handles, shared_labels = None, None

    for i, data in enumerate(data_list):
        ax = axs[i]

        # ---- Unpack ----
        noise_type = data["noise_type"]
        exact_sol = data["exact_sol"]
        sorted_noise = data["sorted_noise"]
        mean_exp_vals = data["mean_exp_vals"]
        std_exp_vals = data["std_exp_vals"]
        zne_mean = data["zne_mean"]
        zne_std = data["zne_std"]
        mean_noise_off = data.get("mean_noise_off")
        std_noise_off = data.get("std_noise_off")

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
            # label=f"{noise_type.capitalize()} estimation",
            label=f"Noisy estimation",
            color=plot_colors["noisy"],
            markersize=marker_size,
            markeredgewidth=0.8,
            elinewidth=1,
        )
        # Unmitigated

        ax.axhline(
            mean_exp_vals[0],
            color=plot_colors["unmitigated"],
            linestyle="--",
            linewidth=border_width,
            zorder=4,
            label="Unmitigated",
        )
        ax.axhspan(
            mean_exp_vals[0] - std_exp_vals[0],
            mean_exp_vals[0] + std_exp_vals[0],
            color=plot_colors["unmitigated"],
            alpha=0.2,
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
                mean_noise_off,
                color=plot_colors["noise_free"],
                linestyle="--",
                linewidth=border_width,
                zorder=4,
                label="Noise-free estimation",
            )
            if std_noise_off is not None:
                ax.axhspan(
                    mean_noise_off - std_noise_off,
                    mean_noise_off + std_noise_off,
                    color=plot_colors["noise_free"],
                    alpha=0.2,
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
                    x=0.5,
                    y=panel_label_y,
                    s=plot_titles[i],
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
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
    if subplot_top is not None:
        adjust_kwargs["top"] = subplot_top
    if subplot_bottom is not None:
        adjust_kwargs["bottom"] = subplot_bottom
    if subplot_wspace is not None:
        adjust_kwargs["wspace"] = subplot_wspace
    if subplot_hspace is not None:
        adjust_kwargs["hspace"] = subplot_hspace

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
                shared_handles,
                shared_labels,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox,
                **legend_kwargs,
            )
        else:
            fig.legend(
                shared_handles,
                shared_labels,
                loc=legend_loc,
                **legend_kwargs,
            )

    # ------------------------------------------------------------------ #
    #  Figure caption / title via fig.text                                #
    # ------------------------------------------------------------------ #
    if figure_title is not None:
        fig.text(
            figure_title_x,
            figure_title_y,
            figure_title,
            ha=figure_title_ha,
            va=figure_title_va,
            fontsize=figure_title_fontsize,
        )

    # ------------------------------------------------------------------ #
    #  Save                                                                #
    # ------------------------------------------------------------------ #
    base_name = os.path.splitext(plot_file_name)[0]
    save_path = os.path.join(output_dir, f"{base_name}.{save_format}")
    # fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches="tight")
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
        "noisy": _to_hex(_darken(base_rgb, 0.15)),
        "zne": _to_hex(_lighten(base_rgb, 0.40)),
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
    plot_colors: Dict[str, str],  # Dict containing "exact"
    plot_file_name: str,
    output_dir: str,
    plot_titles: Optional[List[str]] = None,
    dataset_colors: Optional[List[str]] = None,  # NEW: Custom colors per dataset
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

    plt.rcParams.update(
        {
            "font.size": tick_fontsize,
            "axes.labelsize": label_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, layout="constrained")
    ax.set_axisbelow(True)  # Grid goes behind data

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
            zorder=3,
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
            zorder=4,
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
                zorder=5,
            )

        # 4. Exact Line (Add once)
        if not exact_drawn:
            h_exact = ax.axhline(
                y=data["exact_sol"],
                color=plot_colors.get("exact", "red"),
                linestyle="--",
                linewidth=1.5,
                label="Exact solution",
                zorder=2,
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
        "zne": "#60a5fa",
        "noisy": "red",
        "exact": "magenta",
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
    noise_vecs = data["sorted_noise"]
    means = np.array(data["mean_exp_vals"])
    stds = np.array(data["std_exp_vals"])
    zne_mean = data["zne_mean"]
    zne_std = data["zne_std"]
    noise_off_mean = data.get("mean_noise_off")
    noise_off_std = data.get("std_noise_off")
    target = extrapol_target if extrapol_target is not None else data.get("exact_sol")

    # ------------------------------------------------------------------ #
    # X-axis
    # ------------------------------------------------------------------ #
    first_key_len = len(noise_vecs[0])
    zne_x_label = f"ZNE\n({','.join(['0'] * first_key_len)})"
    noise_labels = [f"({','.join(map(str, nv))})" for nv in noise_vecs]
    x_labels = [zne_x_label] + noise_labels

    all_means = np.concatenate([[zne_mean], means])
    all_stds = np.concatenate([[zne_std], stds])
    x = np.arange(len(x_labels))

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout="constrained")

    # 1. ZNE extrapolated point
    ax.errorbar(
        x[0],
        all_means[0],
        yerr=all_stds[0],
        fmt="o",
        color=_colors["zne"],
        ecolor=_colors["zne"],
        elinewidth=border_width,
        capsize=capsize,
        markersize=marker_size,
        label=f"ZNE: {all_means[0]:.4f} ± {all_stds[0]:.4f}",
        zorder=5,
    )

    # 2. Noisy simulation points
    ax.errorbar(
        x[1:],
        all_means[1:],
        yerr=all_stds[1:],
        fmt="o",
        color=_colors["noisy"],
        ecolor=_colors["noisy"],
        elinewidth=border_width,
        capsize=capsize,
        markersize=marker_size,
        linestyle="None",
        label="Noisy estimations",
        zorder=3,
    )

    # 3. VQE noise-off line + band
    if noise_off_mean is not None:
        ax.axhline(
            noise_off_mean,
            color=_colors["noise_off"],
            linestyle="--",
            linewidth=border_width,
            zorder=4,
            label=f"VQE noise-off: {noise_off_mean:.4f} ± {noise_off_std:.4f}",
        )
        if noise_off_std is not None:
            ax.axhspan(
                noise_off_mean - noise_off_std, noise_off_mean + noise_off_std, color=_colors["noise_off"], alpha=0.2
            )

    # 4. Exact / target line
    if target is not None:
        ax.axhline(
            target,
            color=_colors["exact"],
            linestyle="--",
            linewidth=border_width,
            label=f"Exact solution: {target:.4f}",
            zorder=4,
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
    panel_title = plot_title
    title_fs = panel_label_fontsize or title_fontsize

    if panel_label_y is None:
        ax.set_title(panel_title, fontsize=title_fs, pad=8)
    else:
        ax.annotate(
            panel_title,
            xy=(0.5, panel_label_y),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=title_fs,
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
            figure_title_x,
            figure_title_y,
            figure_title,
            ha=figure_title_ha,
            va=figure_title_va,
            fontsize=figure_title_fontsize,
        )

    # ------------------------------------------------------------------ #
    # Spacing
    # ------------------------------------------------------------------ #
    spacing = {
        k: v
        for k, v in {
            "top": subplot_top,
            "bottom": subplot_bottom,
            "left": subplot_left,
            "right": subplot_right,
        }.items()
        if v is not None
    }

    if spacing:
        plt.subplots_adjust(**spacing)
    else:
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)  # default room for rotated x labels

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in save_format if isinstance(save_format, list) else [save_format]:
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
    # --- ZNE label ---
    zne_label: str = "ZNE (multivariate)",
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
    labels = sorted(processed_mul_var.keys(), key=lambda k: processed_mul_var[k]["degree"])
    degrees = [processed_mul_var[k]["degree"] for k in labels]
    means = [processed_mul_var[k]["zne_mean"] for k in labels]
    stds = [processed_mul_var[k]["zne_std"] for k in labels]
    costs = [processed_mul_var[k].get("cost_mean") for k in labels]

    first = processed_mul_var[labels[0]]
    _exact_sol = exact_sol if exact_sol is not None else first.get("exact_sol")
    _noise_off = first.get("mean_noise_off")
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
        degrees,
        means,
        yerr=stds,
        fmt="o-",
        capsize=capsize,
        markersize=marker_size,
        color=plot_colors.get("zne", "steelblue"),
        label=zne_label,
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
            figure_title_x,
            figure_title_y,
            figure_title,
            ha=figure_title_ha,
            va=figure_title_va,
            fontsize=figure_title_fontsize,
        )

    # plt.tight_layout()

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
    labels = sorted(DATA.keys(), key=lambda k: DATA[k]["order"])
    orders = [DATA[k]["order"] for k in labels]
    means = [DATA[k]["zne_mean"] for k in labels]
    stds = [DATA[k]["zne_std"] for k in labels]

    first = DATA[labels[0]]
    _exact_sol = exact_sol if exact_sol is not None else first.get("exact_sol")
    _noise_off = first.get("mean_noise_off")
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
        orders,
        means,
        yerr=stds,
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
            figure_title_x,
            figure_title_y,
            figure_title,
            ha=figure_title_ha,
            va=figure_title_va,
            fontsize=figure_title_fontsize,
        )

    # plt.tight_layout()

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{plot_file_name}.{save_format}")

    fig.savefig(out_path, format=save_format, dpi=dpi)
    print(f"✅ Saved: {out_path}")

    if show_plot:
        plt.show()
    plt.close(fig)
    return fig


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
    fig_width: float,  # final rendered width in inches — required
    dpi: int,  # output DPI — required
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
    row_values.append(f"{ref['mean_noise_off']:.3f} \u00b1 {ref['std_noise_off']:.3f}")

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
    n_noise_rows = len(ref["sorted_noise"])
    section_breaks = [1, 1 + n_noise_rows]

    # ── Figure ────────────────────────────────────────────────────────────────
    n_rows = len(row_labels)
    fig_h = n_rows * row_height

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
        bold = i == 0

        ax.text(
            PAD_L,
            y_center,
            label,
            va="center",
            ha="left",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            transform=ax.transData,
        )
        ax.text(
            1 - PAD_R,
            y_center,
            value,
            va="center",
            ha="right",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            transform=ax.transData,
        )

    # ── Horizontal rules ──────────────────────────────────────────────────────
    ax.axhline(n_rows, color="black", lw=1.0)
    ax.axhline(0, color="black", lw=1.0)
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
        latex_lines.append(f"        {label} $= {noise_val}$ & ${mean_val:.3f} \\pm {std_val:.3f} $\\\\")
    latex_lines.append("        \\hline")

    # Section 3: ZNE Values (No bold)
    for entry in reversed(data_list):
        n_pts = len(entry["sorted_noise"])
        # label = f"\\text{{{n_pts}-point Richardson (RIC-{n_pts}) ZNE value}}"
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
    row_values.append(f"{data['mean_noise_off']:.3f} \u00b1 {data['std_noise_off']:.3f}")

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
    row_labels.append(f"Multivariate Richardson ZNE value")
    row_values.append(f"{data['zne_mean']:.3f} \u00b1 {data['zne_std']:.3f}")

    # ── Section divider positions ─────────────────────────────────────────────
    n_noise_rows = len(data["sorted_noise"])
    section_breaks = [1, 1 + n_noise_rows]

    # ── Figure ────────────────────────────────────────────────────────────────
    n_rows = len(row_labels)
    fig_h = n_rows * row_height

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
        bold = i == 0

        # Serial number (skip for first and last rows)
        if i > 0 and i < n_rows - 1:
            ax.text(
                PAD_L,
                y_center,
                f"{i}.",
                va="center",
                ha="left",
                fontsize=fontsize,
                fontweight="normal",
                transform=ax.transData,
            )
            label_x = PAD_L + 0.05  # indent label to make room for number
        else:
            label_x = PAD_L

        ax.text(
            label_x,
            y_center,
            label,
            va="center",
            ha="left",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            transform=ax.transData,
        )
        ax.text(
            1 - PAD_R,
            y_center,
            value,
            va="center",
            ha="right",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            transform=ax.transData,
        )

    # ── Horizontal rules ──────────────────────────────────────────────────────
    ax.axhline(n_rows, color="black", lw=1.0)
    ax.axhline(0, color="black", lw=1.0)
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
        latex_lines.append(f"        {label} $= {noise_str}$ & ${mean_val:.3f} \\pm {std_val:.3f} $\\\\")
    latex_lines.append("        \\hline")

    # Section 3: ZNE value
    label = f"\\text{{Multivariate Richardson ZNE value}}"
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
    mul_var_data: dict,  # PROCESSED_SIM_DATA["ZNE-mul-var"]
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
        row_values.append(f"{entry['zne_mean']:.3f} \u00b1 {entry['zne_std']:.3f}")

    # ── Figure ───────────────────────────────────────────────────────────────
    n_rows = len(row_labels)
    fig_h = n_rows * row_height

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
            PAD_L,
            y_center,
            label,
            va="center",
            ha="left",
            fontsize=fontsize,
            fontweight="normal",
            transform=ax.transData,
        )
        ax.text(
            1 - PAD_R,
            y_center,
            value,
            va="center",
            ha="right",
            fontsize=fontsize,
            fontweight="normal",
            transform=ax.transData,
        )

    # ── Horizontal rules ─────────────────────────────────────────────────────
    ax.axhline(n_rows, color="black", lw=1.0)
    ax.axhline(0, color="black", lw=1.0)

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
    mantissa = value / (10**exponent)

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
        - 'cost_eq_mean'    : ZNE sampling overhead under equal shot allocation,
                               c = M * Gamma^2  (Eq. eq-lre-sampling-c-bar)

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
        r"\caption{\textbf{Multivariate ZNE in plot Figure "
        r"\ref{fig-plots-zne-diff-orders-multivariate}. Mean values and "
        r"corresponding standard deviations are computed over 10 independent "
        r"experimental runs.}}"
    )
    lines.append(r"\label{table-multi-var-zne}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{|p{71pt}|p{71pt}|p{71pt}|}")
    lines.append(r"\hline")
    lines.append(r"Quantity & Estimated Value (Mean $\pm$ Std. Dev.) & " r"ZNE Sampling Overhead $(c)$ \\")
    lines.append(r"\hline")

    # ── Fixed rows: noise-free & unmitigated ─────────────────────────────────
    lines.append(f"Noise-free estimation & ${noise_free_mean:.3f} \\pm {noise_free_std:.3f}$ & -- \\\\")
    lines.append(r"\hline")
    lines.append(f"Unmitigated & ${unmitigated_mean:.3f} \\pm {unmitigated_std:.3f}$ & -- \\\\")
    lines.append(r"\hline")

    # ── ZNE order rows ────────────────────────────────────────────────────
    for key in sorted_keys:
        entry = mul_var_data[key]
        degree = entry["degree"]
        zne_mean = entry["zne_mean"]
        zne_std = entry["zne_std"]
        cost = entry["cost_eq_mean"]
        cost_str = to_sci_notation(cost, sig_figs=cost_sig_figs)
        lines.append(f"ZNE of order {degree} & ${zne_mean:.3f} \\pm {zne_std:.3f}$ & {cost_str} \\\\")
    lines.append(r"\hline")

    # ── Footnote row ──────────────────────────────────────────────────────
    lines.append(
        r"\multicolumn{3}{p{215pt}}{All values are expressed in dimensionless "
        r"units. Due to the large number of data points, we do not include "
        r"them explicitly here. For more detailed data used in this "
        r"multivariate extrapolation, refer to \cite{indirect-zne-github}.} \\"
    )
    lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)
