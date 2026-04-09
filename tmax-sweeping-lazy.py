"""
Sweep tmax values for time evolution gates in VQE, by patching the config and calling vqe-mcore.py in a subprocess.
This is not the most resource efficient way, but its simple.
"""

import subprocess
import sys
import os
import yaml
import tempfile
import copy

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return None
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 tmax-sweeping.py <config_file> <tmax1> [tmax2 ...]")
        sys.exit(1)

    config_file = sys.argv[1]
    tmax_values = [float(x) for x in sys.argv[2:]]

    base_config = load_config(config_file)
    if base_config is None:
        sys.exit(1)

    base_prefix = base_config["output"]["file_name_prefix"]

    print(f"Sweeping {len(tmax_values)} tmax values: {tmax_values}")
    print(f"Base prefix: {base_prefix}")

    for tmax in tmax_values:
        print(f"\n>>> Running vqe-mcore.py with tmax={tmax:g}")

        patched = copy.deepcopy(base_config)
        patched["ansatz"]["ugate"]["time"]["max"] = tmax
        patched["output"]["file_name_prefix"] = f"{base_prefix}_tmax{tmax:g}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
            yaml.dump(patched, tmp)
            tmp_path = tmp.name

        try:
            subprocess.run(["python3", "vqe-mcore.py", tmp_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[tmax={tmax:g}] FAILED with exit code {e.returncode}")
        finally:
            os.unlink(tmp_path)

    print("\nAll tmax sweeps completed.")