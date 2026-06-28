"""
Multicore VQE.
"""
import json
import os
import sys
import time
import uuid
from ast import Dict
from datetime import date, datetime

from typing import List

import yaml
import h5py

import numpy as np

from configValidator import validate_yml_config
from src.modules import get_eigen_min
from src.observable import constructObservable
from src.vqe import IndirectVQE

import multiprocessing

symbol_count = 25


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return None
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


###########################################################
# Helper functions
###########################################################
def serialize_optimize_result(result) -> dict:
    """Convert scipy OptimizeResult to a JSON-serializable dict.
    
    Compatible with SLSQP, COBYLA, L-BFGS-B and other scipy optimizers.
    Fields absent from a given optimizer are serialized as None.
    """

    # --- hess_inv: L-BFGS-B returns LinearOperator, SLSQP returns ndarray ---
    hess_inv = None
    if hasattr(result, "hess_inv") and result.hess_inv is not None:
        if hasattr(result.hess_inv, "todense"):          # LinearOperator
            hess_inv = np.asarray(result.hess_inv.todense()).tolist()
        elif isinstance(result.hess_inv, np.ndarray):    # ndarray
            hess_inv = result.hess_inv.tolist()

    return {
        "x":        result.x.tolist(),
        "fun":      float(result.fun),
        "success":  bool(result.success),
        "message":  str(result.message),
        # COBYLA does not return nit
        "nit":      int(result.nit)  if hasattr(result, "nit")  else None,
        "nfev":     int(result.nfev) if hasattr(result, "nfev") else None,
        # COBYLA / SLSQP do not return jac
        "jac":      result.jac.tolist() if hasattr(result, "jac") and
                    isinstance(result.jac, np.ndarray) else None,
        "hess_inv": hess_inv,
    }

def generate_experiment_id(prefix: str | None = None) -> str:
    experiment_id = str(uuid.uuid4())

    if prefix is not None:
        return f"{prefix}_{experiment_id}"

    return experiment_id
###########################################################


def run_single_vqe(run_index: int, config_path: str, batch_timestamp: str) -> None:
    """
    Runs a single VQE and writes JSON output immediately after completion.
    Identical JSON structure to initialize_vqe() in main.py.
    Designed to run in an isolated process — no shared state.
    """
    experiment_id = generate_experiment_id()
    timestamp_per_vqe = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = load_config(config_path)

    # Parse config — mirrors main.py exactly
    nqubits: int = config["nqubits"]
    state: str = config["state"]
    file_name_prefix: str = config["output"]["file_name_prefix"]
    circuit_draw_status: bool = config["output"]["draw"]["status"]
    fig_dpi: int = config["output"]["draw"]["fig_dpi"]
    fig_filetype: str = config["output"]["draw"]["type"]
    observable_def: Dict = config["observable"]["def"]
    observable_coefficients: Dict = config["observable"]["coefficients"]
    ansatz: Dict = config["ansatz"]
    noise_profile: Dict = config["noise_profile"]
    vqe_profile: Dict = config["vqe"]
    initialparam: List[float] = config["init_param"]["value"]

    message: str | None = config.get("message")
    history: bool = config.get("history")

    #opt_options = vqe_profile["optimization"]["opt_options"]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the shared batch directory directly
    batch_dir = os.path.join(current_dir, "output", f"{file_name_prefix}_{batch_timestamp}")
    os.makedirs(batch_dir, exist_ok=True)

    # Establish a unified base filename so json, h5, and drawings match perfectly
    file_base_name = f"{file_name_prefix}_run{run_index:03d}_{experiment_id}"

    target_observable = constructObservable(
        nqubits=nqubits, definition=observable_def, coefficient=observable_coefficients
    )
    exact_cost: float = get_eigen_min(hamiltonian=target_observable)

    print(f"[Run {run_index:03d}] Running...")
    start_time = time.time()

    # Pass batch_dir directly to run_dir and pass file_base_name to run_id 
    # to enforce identical naming for the generated .h5 file.
    vqe_instance = IndirectVQE(
        nqubits=nqubits,
        state=state,
        observable=target_observable,
        vqe_profile=vqe_profile,
        ansatz_profile=ansatz,
        noise_profile=noise_profile,
        identity_factors=[0, 0, 0, 0],
        init_param=initialparam,
        run_dir = batch_dir if history else None,
        run_id = f"{file_base_name}_HISTORY",
    )
    vqe_output = vqe_instance.run_vqe()

    total_run_time = time.time() - start_time
    print(f"[Run {run_index:03d}] Done in {total_run_time:.2f} sec")

    # Identical JSON structure to initialize_vqe() in main.py.
    output_data = {
        "meta":{
        "id": experiment_id,
        "datatime": timestamp_per_vqe,
        "message": message,
        },
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
            "lie_trotter_details": vqe_output["lie_trotter_details"],
            
        },
        "artifacts": {
            # nfev + 1 
            "cost_callings_total": vqe_output["cost_callings_total"],
            "opt_obj": serialize_optimize_result(vqe_output["opt_obj"]),
        }
    }

    # Save the json directly into the batch_dir using the unified file_base_name
    output_file = os.path.join(batch_dir, f"{file_base_name}_VQE.json")

    if history:
            h5_path = os.path.join(batch_dir, f"{file_base_name}_HISTORY.h5")
            with h5py.File(h5_path, "a") as hf:
                # Convert the config dict back to a clean YAML string layout
                config_yaml_str = yaml.dump(config, default_flow_style=False)
                
                # Encode it to flat ASCII bytes so standard HDF5 openers don't crash
                hf.attrs["config"] = config_yaml_str.encode("ascii", errors="ignore")

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=None, separators=(",", ":"))

    print(f"[Run {run_index:03d}] Output saved to: {os.path.abspath(output_file)}")

    if circuit_draw_status:
        vqe_instance.drawCircuit(
            prefix=file_base_name,
            dpi=fig_dpi,
            filetype=fig_filetype,
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 vqe-mcore.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    config = load_config(config_file)
    if config is None:
        sys.exit(1)

    num_runs: int = config["vqe"]["iteration"]
    num_cores: int = os.cpu_count()

    # Single timestamp shared across all runs in this batch
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Launching {num_runs} isolated VQE processes ({num_cores} logical cores available)...")
    print("=" * symbol_count + "VQE-MCORE" + "=" * symbol_count)

    start_time = time.time()

    processes = []
    for i in range(num_runs):
        p = multiprocessing.Process(target=run_single_vqe, args=(i, config_file, batch_timestamp))
        p.start()
        processes.append((i, p))
        print(f"[Run {i:03d}] Started (PID: {p.pid})")

    for i, p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"[Run {i:03d}] FAILED (exit code: {p.exitcode})")
        else:
            print(f"[Run {i:03d}] Completed successfully")

    total_time = time.time() - start_time
    print("=" * symbol_count + "Done" + "=" * symbol_count)
    print(f"All {num_runs} runs completed in {total_time:.2f} sec")
    print(f"Output folder: output/{config['output']['file_name_prefix']}_{batch_timestamp}/")
    print(f"Output files:  {config['output']['file_name_prefix']}_run000_*.json ... run{num_runs-1:03d}_*.json")