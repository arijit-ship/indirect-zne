"""
Automation for redundant and ZNE runs.
"""

import os
import yaml
import json
import subprocess

# ==========OPTIMIZED PARAMETERS==========
with open("optparam.json", "r", encoding="utf-8") as f:
    ALL_OPTIMIZED_PARAMS = json.load(f)


# === CONFIGURATION ===
model: str = "xy-iss"
OPTIMIZED_PARAM = ALL_OPTIMIZED_PARAMS[f"xy-dephasing-noisy-time-evo"]
ANSATZ_TYPE = model
STATIC_PREFIX = f"AUTOMATE_{model}_noisy_time_evo_dephasing_ric7"  # Output file prefix
I_FACTOR = [
  [0, 0, 0, 0],
  [1, 1, 1, 0],
  [2, 2, 2, 0],
  [3, 3, 3, 0],
  [4, 4, 4, 0],
  [5, 5, 5, 0],
  [6, 6, 6, 0],
]
##-----------------**--------------------##
NOISE_TYPE = "dephasing"
NOISE_VALUE = [0.001, 0.001, 0.001, 0.001]
CONFIG_PATH = "exp.auto.yml"
NUM_RUNS = 10  # or len(OPTIMIZED_PARAM)
RIC_MUL = False  # Whether to remove RIC columns from data points

# === HELPER FUNCTIONS ===

def set_output_prefix(config, index):
    """Set output filename prefix in the config for the given run index."""
    sample_tag = f"{STATIC_PREFIX}_sample#{index + 1}"
    config["output"]["file_name_prefix"] = sample_tag
    return sample_tag  # return it so we can use it to load output

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
    for i in range(NUM_RUNS):
        print(f"\n=== Running redundant + zne iteration {i + 1}/{NUM_RUNS} ===")

        # REDUNDANT RUN
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        config["run"] = "redundant"
        prefix = set_output_prefix(config, i)
        set_init_param(config, OPTIMIZED_PARAM[i])
        set_ansatz_type(config, ANSATZ_TYPE)
        set_noise_type(config, NOISE_TYPE)
        set_noise_value(config, NOISE_VALUE)
        set_i_factor(config, I_FACTOR)
        config["zne"]["data_points"] = None

        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        run_main()

        # LOAD OUTPUT JSON
        data = load_output_json_by_prefix(prefix)
        if data is None:
            print(f"[!] Skipping zne for sample#{i + 1} due to missing redundant output.")
            continue

        # ZNE RUN
        optimized_param = data["config"]["init_param"]["value"]
        data_points = data["output"].get("data_points", None)

        print(f"Data points from output sample#{i+1}: {data_points}")

        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        config["run"] = "zne"
        if not RIC_MUL:
            config["zne"]["data_points"] = data_points
        else:
          cleaned_data_points = [row[:2] + row[4:] for row in data_points]
          config["zne"]["data_points"] = cleaned_data_points

        
        set_output_prefix(config, i)  # same prefix
        set_init_param(config, optimized_param)

        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        run_main()


if __name__ == "__main__":
    main()
