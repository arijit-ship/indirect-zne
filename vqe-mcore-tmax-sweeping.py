"""
VQE tmax sweep — parallel across all CPU cores.

Loops over a list of tmax values, running `vqe_iteration` independent VQE
processes per tmax value.  All (tmax, run_index) jobs are queued in a
multiprocessing.Pool so that every logical core stays busy.

Usage
-----
    python3 vqe-tmax-sweep.py <config_file> <tmax1> [tmax2 tmax3 ...]

Example
-------
    python3 vqe-tmax-sweep.py config_small.yml 2.0 5.0 10.0 20.0

Output files
------------
    output/<prefix>_tmax<tmax>_run<idx>_VQE.json

The JSON structure inside each file is IDENTICAL to vqe-mcore.py /
main.py:initialize_vqe().  The only difference is the output filename,
which encodes the tmax value so files from different sweeps never collide.
"""

import copy
import json
import multiprocessing
import os
import sys
import time

import yaml

from configValidator import validate_yml_config
from src.modules import get_eigen_min
from src.observable import constructObservable
from src.vqe import IndirectVQE

symbol_count = 25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict | None:
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return None
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Worker — runs in an isolated subprocess (no shared state)
# ---------------------------------------------------------------------------

def run_single_vqe(run_index: int, tmax: float, base_config: dict) -> None:
    """
    Runs one VQE with ansatz.ugate.time.max overridden to `tmax`.
    Writes a single JSON file whose structure is identical to
    vqe-mcore.py / main.py:initialize_vqe().
    """
    # Deep-copy so this process owns its own config dict
    config = copy.deepcopy(base_config)

    # --- Override tmax ---
    config["ansatz"]["ugate"]["time"]["max"] = tmax

    # --- Parse config (mirrors main.py / vqe-mcore.py exactly) ---
    nqubits: int           = config["nqubits"]
    state: str             = config["state"]
    file_name_prefix: str  = config["output"]["file_name_prefix"]
    circuit_draw_status: bool = config["output"]["draw"]["status"]
    fig_dpi: int           = config["output"]["draw"]["fig_dpi"]
    fig_filetype: str      = config["output"]["draw"]["type"]
    observable_def         = config["observable"]["def"]
    observable_coefficients = config["observable"]["coefficients"]
    ansatz                 = config["ansatz"]
    noise_profile          = config["noise_profile"]
    vaqe_profile           = config["vqe"]
    initialparam           = config["init_param"]["value"]

    target_observable = constructObservable(
        nqubits=nqubits,
        definition=observable_def,
        coefficient=observable_coefficients,
    )
    exact_cost: float = get_eigen_min(hamiltonian=target_observable)

    # Format tmax for labels/filenames: use int representation when possible
    tmax_str = f"{tmax:g}"

    print(f"[tmax={tmax_str} | run {run_index:03d}] Starting...")
    start_time = time.time()

    vqe_instance = IndirectVQE(
        nqubits=nqubits,
        state=state,
        observable=target_observable,
        vqe_profile=vaqe_profile,
        ansatz_profile=ansatz,
        noise_profile=noise_profile,
        identity_factors=[0, 0, 0, 0],
        init_param=initialparam,
    )
    vqe_output = vqe_instance.run_vqe()

    total_run_time = time.time() - start_time
    print(f"[tmax={tmax_str} | run {run_index:03d}] Done in {total_run_time:.2f} sec")

    # --- Build output (IDENTICAL JSON structure to vqe-mcore.py) ---
    output_data = {
        "config": config,                          # config already has tmax patched in
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

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir  = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Filename encodes tmax so sweeps never collide
    output_file = os.path.join(
        output_dir,
        f"{file_name_prefix}_tmax{tmax_str}_run{run_index:03d}_VQE.json",
    )

    with open(output_file, "w") as fh:
        json.dump(output_data, fh, indent=None, separators=(",", ":"))

    print(f"[tmax={tmax_str} | run {run_index:03d}] Saved → {os.path.abspath(output_file)}")

    if circuit_draw_status:
        vqe_instance.drawCircuit(
            prefix=f"{file_name_prefix}_tmax{tmax_str}_run{run_index:03d}",
            dpi=fig_dpi,
            filetype=fig_filetype,
        )


# Pool worker shim — Pool.starmap passes a single iterable of args
def _worker(args: tuple) -> None:
    run_index, tmax, base_config = args
    run_single_vqe(run_index, tmax, base_config)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 vqe-tmax-sweep.py <config_file> <tmax1> [tmax2 ...]")
        sys.exit(1)

    config_file = sys.argv[1]
    tmax_values = [float(v) for v in sys.argv[2:]]

    base_config = load_config(config_file)
    if base_config is None:
        sys.exit(1)

    if not validate_yml_config(base_config):
        print("Error: Invalid config file.")
        sys.exit(1)

    num_runs: int  = base_config["vqe"]["iteration"]
    num_cores: int = os.cpu_count()

    # Build the full job list: every (tmax, run_index) combination
    jobs = [
        (run_index, tmax, base_config)
        for tmax in tmax_values
        for run_index in range(num_runs)
    ]

    total_jobs = len(jobs)

    print(f"tmax values   : {tmax_values}")
    print(f"Runs per tmax : {num_runs}")
    print(f"Total jobs    : {total_jobs}")
    print(f"Logical cores : {num_cores}  (pool size)")
    print("=" * symbol_count + "VQE-TMAX-SWEEP" + "=" * symbol_count)

    sweep_start = time.time()

    # Pool size = number of logical cores; jobs are queued automatically
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(_worker, jobs)

    sweep_time = time.time() - sweep_start

    print("=" * symbol_count + "Done" + "=" * symbol_count)
    print(f"All {total_jobs} jobs completed in {sweep_time:.2f} sec")
    print(
        f"Output pattern: output/<prefix>_tmax<value>_run<NNN>_VQE.json"
    )