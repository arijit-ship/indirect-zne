import copy
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime
import yaml

from configValidator import validate_yml_config
from src.modules import get_eigen_min
from src.observable import constructObservable
from src.vqe import IndirectVQE

symbol_count = 25

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return None
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)

def run_single_vqe_worker(args):
    """
    Worker function for the Pool. 
    Unpacks arguments and runs a single VQE iteration.
    """
    run_index, tmax, config_file, batch_timestamp = args
    
    # Reload or deepcopy config to ensure isolation
    config = load_config(config_file)
    
    # --- CRITICAL: Override tmax correctly in the nested dict ---
    config["ansatz"]["ugate"]["time"]["max"] = tmax
    
    # Extract variables (mirrors vqe-mcore.py)
    nqubits = config["nqubits"]
    file_name_prefix = config["output"]["file_name_prefix"]
    observable_def = config["observable"]["def"]
    observable_coefficients = config["observable"]["coefficients"]
    
    target_observable = constructObservable(
        nqubits=nqubits, definition=observable_def, coefficient=observable_coefficients
    )
    exact_cost = get_eigen_min(hamiltonian=target_observable)

    tmax_str = f"{tmax:g}"
    print(f"[tmax={tmax_str} | Run {run_index:03d}] Starting...")
    start_time = time.time()

    vqe_instance = IndirectVQE(
        nqubits=nqubits,
        state=config["state"],
        observable=target_observable,
        vqe_profile=config["vqe"],
        ansatz_profile=config["ansatz"],
        noise_profile=config["noise_profile"],
        identity_factors=[0, 0, 0, 0],
        init_param=config["init_param"]["value"],
    )
    vqe_output = vqe_instance.run_vqe()
    total_run_time = time.time() - start_time

    # Build output data (tmax included in the saved config)
    output_data = {
        "config": config, 
        "output": {
            "exact_sol": exact_cost,
            "initial_cost_history": [vqe_output["initial_cost"]],
            "optimized_minimum_cost": [vqe_output["min_cost"]],
            "optimized_parameters": [vqe_output["optimized_param"]],
            "noise_details": vqe_instance.get_noise_level(),
            "run_time_sec": total_run_time,
        },
        "others": {
            "observable_string": str(target_observable),
            "time_evolution_gate_hamiltonian_string": [str(vqe_instance.get_ugate_hamiltonain())],
            "initial_parameters": vqe_output["init_random_param"],
            "initial_states": [vqe_output["initial_density_matrix"]],
            "final_states": [vqe_output["final_density_matrix"]],
        },
    }

    # Save to timestamped sweep folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output", f"sweep_{batch_timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{file_name_prefix}_tmax{tmax_str}_run{run_index:03d}_VQE.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as fh:
        json.dump(output_data, fh, indent=None, separators=(",", ":"))
    
    print(f"[tmax={tmax_str} | Run {run_index:03d}] Saved in {total_run_time:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 vqe-tmax-sweep.py <config_file> <tmax1> [tmax2...]")
        sys.exit(1)

    config_file = sys.argv[1]
    tmax_values = [float(x) for x in sys.argv[2:]]
    
    base_config = load_config(config_file)
    if not base_config or not validate_yml_config(base_config):
        sys.exit(1)

    # Resource Management: Leave 1 core free
    num_cores = max(1, os.cpu_count() - 1)
    num_iterations = base_config["vqe"]["iteration"]
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate job list
    tasks = [
        (idx, t, config_file, batch_timestamp) 
        for t in tmax_values 
        for idx in range(num_iterations)
    ]

    print(f"Starting sweep: {len(tmax_values)} tmax values, {num_iterations} runs each.")
    print(f"Using {num_cores} cores (1 core reserved).")
    print("=" * symbol_count + " SWEEP START " + "=" * symbol_count)

    start_sweep = time.time()
    
    # Use Pool for efficient management of many tasks
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(run_single_vqe_worker, tasks)

    end_sweep = time.time()
    print("=" * symbol_count + " SWEEP DONE " + "=" * symbol_count)
    print(f"Total time: {end_sweep - start_sweep:.2f} seconds.")