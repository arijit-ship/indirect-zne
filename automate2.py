"""
Automation script for redundant and ZNE runs.
"""

import os
import yaml
import json
import subprocess

# === CONFIGURATION ===
RAW_DATA_FOLDER = "experiments/recent/experiment14[depol-time-evo-noisy_variousorderzne_tmax_20]/data/VQE"  # <-- Set this to your simulation raw data folder
#RAW_DATA_FOLDER = "output/"
model: str = "xy-iss"
ANSATZ_TYPE = model
STATIC_PREFIX = f"AUTOMATE_xy-iss_noisy_time_evo_time_depol_varioustmax_tmax20_ricmul_d5"  # Output file prefix
I_FACTOR =  [[0, 0, 0, 0],
 [1, 1, 1, 1],
 [1, 1, 1, 2],
 [1, 1, 1, 3],
 [1, 1, 1, 4],
 [1, 1, 1, 5],
 [1, 1, 1, 6],
 [1, 1, 2, 1],
 [1, 1, 2, 2],
 [1, 1, 2, 3],
 [1, 1, 2, 4],
 [1, 1, 2, 5],
 [1, 1, 3, 1],
 [1, 1, 3, 2],
 [1, 1, 3, 3],
 [1, 1, 3, 4],
 [1, 1, 4, 1],
 [1, 1, 4, 2],
 [1, 1, 4, 3],
 [1, 1, 5, 1],
 [1, 1, 5, 2],
 [1, 1, 6, 1],
 [1, 2, 1, 1],
 [1, 2, 1, 2],
 [1, 2, 1, 3],
 [1, 2, 1, 4],
 [1, 2, 1, 5],
 [1, 2, 2, 1],
 [1, 2, 2, 2],
 [1, 2, 2, 3],
 [1, 2, 2, 4],
 [1, 2, 3, 1],
 [1, 2, 3, 2],
 [1, 2, 3, 3],
 [1, 2, 4, 1],
 [1, 2, 4, 2],
 [1, 2, 5, 1],
 [1, 3, 1, 1],
 [1, 3, 1, 2],
 [1, 3, 1, 3],
 [1, 3, 1, 4],
 [1, 3, 2, 1],
 [1, 3, 2, 2],
 [1, 3, 2, 3],
 [1, 3, 3, 1],
 [1, 3, 3, 2],
 [1, 3, 4, 1],
 [1, 4, 1, 1],
 [1, 4, 1, 2],
 [1, 4, 1, 3],
 [1, 4, 2, 1],
 [1, 4, 2, 2],
 [1, 4, 3, 1],
 [1, 5, 1, 1],
 [1, 5, 1, 2],
 [1, 5, 2, 1]]
    
##-----------------**--------------------##
CONFIG_PATH = "config_samples/q7_various_tmax_time_depol_1e-3.yml"
RIC_MUL = True # Whether to remove RIC columns from data points


# === LOAD PARAMS FROM RAW DATA FOLDER ===

def load_params_from_folder(folder_path):
    """
    Load optimized_param, noise_value, and t_max from each JSON file
    in the given folder. Files are sorted by name; one file = one run.

    Returns a list of dicts with keys: optimized_param, noise_value, t_max
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Raw data folder not found: '{folder_path}'")

    json_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".json")
    ])

    if not json_files:
        raise ValueError(f"No JSON files found in folder: '{folder_path}'")

    params = []
    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        optimized_param = data["output"]["optimized_parameters"][0]
        noise_value     = data["config"]["noise_profile"]["noise_prob"]
        t_max           = data["config"]["ansatz"]["ugate"]["time"]["max"]

        params.append({
            "optimized_param": optimized_param,
            "noise_value":     noise_value,
            "t_max":           t_max,
        })
        print(f"[+] Loaded params from {filename}")

    return params


# === HELPER FUNCTIONS ===

def set_output_prefix(config, index):
    """Set output filename prefix in the config for the given run index."""
    sample_tag = f"{STATIC_PREFIX}_sample#{index + 1}"
    config["output"]["file_name_prefix"] = sample_tag
    return sample_tag

def set_init_param(config, param):
    """Set the init_param value in the config."""
    config["init_param"]["value"] = param

def set_ansatz_type(config, ansatz_type):
    """Set the ansatz type in the config."""
    config["ansatz"]["ugate"]["type"] = ansatz_type

def set_noise_type(config, noise_type):
    """Set the noise type in the config."""
    config["noise_profile"]["type"] = noise_type

def set_i_factor(config, i_factor):
    """Set the i_factor in the config."""
    config["redundant"]["identity_factors"] = i_factor

def set_noise_value(config, noise_value):
    """Set the noise value in the config."""
    config["noise_profile"]["noise_prob"] = noise_value

def set_t_max_value(config, tmax):
    """Set t_max in time-evolution gate."""
    config["ansatz"]["ugate"]["time"]["max"] = tmax


def run_main():
    """Run the main.py script with the given config."""
    subprocess.run(["python", "main.py", CONFIG_PATH], check=True)

def load_output_json_by_prefix(prefix):
    """Load a single JSON output file by matching its filename prefix."""
    base_path = "output"
    if not os.path.isdir(base_path):
        print(f"[!] Directory '{base_path}' not found.")
        return None

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json") and prefix in file:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        print(f"[+] Loaded {file_path}")
                        return data
                except Exception as e:
                    print(f"[!] Failed to load {file_path}: {e}")
                    return None

    print(f"[!] No output file found for prefix: {prefix}")
    return None


# === MAIN AUTOMATION LOOP ===

def main():
    all_params = load_params_from_folder(RAW_DATA_FOLDER)
    num_runs = len(all_params)
    print(f"[*] Found {num_runs} run(s) in '{RAW_DATA_FOLDER}'")

    for i, run_params in enumerate(all_params):
        print(f"\n=== Running redundant + zne iteration {i + 1}/{num_runs} ===")

        optimized_param = run_params["optimized_param"]
        noise_value     = run_params["noise_value"]
        t_max           = run_params["t_max"]

        # Derive noise_type from config (read once to get base noise type)
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        noise_type = config["noise_profile"]["type"]

        # REDUNDANT RUN
        config["run"] = "redundant"
        prefix = set_output_prefix(config, i)
        set_init_param(config, optimized_param)
        set_ansatz_type(config, ANSATZ_TYPE)
        set_noise_type(config, noise_type)
        set_t_max_value(config, t_max)
        set_noise_value(config, noise_value)
        set_i_factor(config, I_FACTOR)
        config["zne"]["data_points"] = None

        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f)

        run_main()

        # LOAD OUTPUT JSON
        data = load_output_json_by_prefix(prefix)
        if data is None:
            print(f"[!] Skipping zne for sample#{i + 1} due to missing redundant output.")
            continue

        # ZNE RUN
        optimized_param_out = data["config"]["init_param"]["value"]
        data_points = data["output"].get("data_points", None)

        print(f"Data points from output sample#{i+1}: {data_points}")

        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        config["run"] = "zne"
        if not RIC_MUL:
            config["zne"]["data_points"] = data_points
        else:
            print("DEBUG: Datapoints\n")
            print(data_points)
            #cleaned_data_points = [tuple(row) for row in data_points]
            config["zne"]["data_points"] = [[(p[0] + p[3]), p[1], p[2], p[4]] for p in data_points]
            print("AFTER TRANSFORMATION")
            # print   (data_points)
            print(config["zne"]["data_points"])

        set_output_prefix(config, i)  # same prefix
        set_init_param(config, optimized_param_out)

        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        run_main()


if __name__ == "__main__":
    main()