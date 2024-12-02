import os
import sys
import time
from ast import Dict
from datetime import datetime
from typing import List, Union

import yaml
import json

from src.modules import *

# from src.observable import create_ising_hamiltonian
from src.hamiltonian import create_xy_hamiltonian
from src.vqe import IndirectVQE
from src.zne import ZeroNoiseExtrapolation

# Global symbol count
symbol_count = 25


def load_config(config_path):
    # Check if the config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return None

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def initialize_vqe() -> None:
    """
    Initializes the variational quantum eigensolver.
    """

    vqe_instance = IndirectVQE(
        nqubits=nqubits,
        state=state,
        observable=target_obsevable,
        optimization=optimization,
        ansatz=ansatz,
        identity_factor=[0, 0, 0, 0],
        init_param=initialparam,
    )
    print("=" * symbol_count + "Config" + "=" * symbol_count)
    print(config)
    print("=" * symbol_count + "Optimization" + "=" * symbol_count)
    start_time = time.time()
    initial_cost, exact_cost, min_cost_history, optimized_param = vqe_instance.run_vqe()
    nR, nT, nY = vqe_instance.get_noise_level()
    end_time = time.time()

    runtime = end_time - start_time
    print("=" * symbol_count + "Output" + "=" * symbol_count)
    print(f"Exact sol: {exact_cost}")
    print(f"Initial cost: {initial_cost}")
    print(f"Optimized minimum cost: {min_cost_history}")
    print(f"Optimized parameters: {optimized_param}")
    print(f"Noise level (nR, nT, nY): ({nR}, {nT}, {nY}) ")
    print(f"Run time: {runtime} sec")
    # Generate timestamp for unique file name
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{file_name_prefix}_{timestamp}.json")

    # Prepare the data to be written in JSON format
    output_data = {
        "Config": config,
        "Exact_sol": exact_cost,
        "Initial_cost": initial_cost,
        "Optimized_minimum_cost": min_cost_history,
        "Optimized_parameters": optimized_param,
        "Run_time_sec": runtime,
    }
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=None, separators=(",", ":"))

    # Print the path of the output file
    print("=" * symbol_count + "File path" + "=" * symbol_count)
    print(f"Output saved to: {os.path.abspath(output_file)}")
    if ansatz["draw"]:
        vqe_instance.drawCircuit(time_stamp=timestamp, dpi=fig_dpi)


def run_redundant() -> None:
    global symbol_count
    identity_factors: Union[List[int], List[List[int]]] = config["identity_factors"]

    data_points = []

    print("=" * symbol_count + "Config" + "=" * symbol_count)
    print(config)
    print("=" * symbol_count + "VQE values at different noise levels" + "=" * symbol_count)

    start_time = time.time()
    # Turn off the optimization
    optimization["status"] = False
    i = 1
    for factor in identity_factors:
        start_iteration_time = time.time()
        vqe_instance = IndirectVQE(
            nqubits=nqubits,
            state=state,
            observable=target_obsevable,
            optimization=optimization,
            ansatz=ansatz,
            identity_factor=factor,
            init_param=initialparam,
        )
        initial_cost, exact_cost, min_cost_history, optimized_param = vqe_instance.run_vqe()
        nR, nT, nY = vqe_instance.get_noise_level()
        data_points.append([nR, nT, nY, *initial_cost])
        end_iteration_time = time.time()
        print(f"#{i}")
        print(f"Exact sol: {exact_cost}")
        print(f"Initial cost: {initial_cost}")
        print(f"Optimized minimum cost: {min_cost_history}")
        print(f"Optimized parameters: {optimized_param}")
        print(f"Identity factor: {factor}")
        print(f"Noise level (nR, nT, nY): ({nR}, {nT}, {nY}) ")
        print(f"Time taken: {end_iteration_time-start_iteration_time} sec")
        if i < len(identity_factors):
            print("-" * symbol_count)
        i += 1
    print("=" * symbol_count + "Data points" + "=" * symbol_count)
    print(f"No of data points: {len(data_points)}")
    print(f"Data points: {data_points}")
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime} sec")
    # Generate timestamp for unique file name
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{file_name_prefix}_{timestamp}.json")

    output_data = {
        "Config": config,
        "Symbol_count": symbol_count,
        "Data_points": data_points,
        "Run_time_sec": runtime,
    }

    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=None, separators=(",", ":"))
    print("=" * symbol_count + "File path" + "=" * symbol_count)
    print(f"Output saved to: {os.path.abspath(output_file)}")


def initialize_zne() -> None:

    global symbol_count
    zne_config: Dict = config["zne"]
    extrapolation_method: str = zne_config["method"]
    zne_degree: int = zne_config["degree"]
    

    data_points = zne_config["data_points"]

    print("=" * symbol_count + "Config" + "=" * symbol_count)
    print(config)
    print("=" * symbol_count + "Zero-noise extrapolation result" + "=" * symbol_count)

    start_time = time.time()
    # Turn off the optimization
    optimization["status"] = False   

    zne_instance = ZeroNoiseExtrapolation(
            datapoints=data_points, degree=zne_degree, method=zne_method, sampling_mode=zne_sampling
        )
    zne_value = zne_instance.getZne()
    print(zne_value)
        

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime} sec")
    # Generate timestamp for unique file name
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{file_name_prefix}_{timestamp}.json")

    output_data = {
        "Config": config,
        "ZNE_values": zne_value,
        "Run_time_sec": runtime,
    }

    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=None, separators=(",", ":"))
    print("=" * symbol_count + "File path" + "=" * symbol_count)
    print(f"Output saved to: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    # Check if a config file argument is provided
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <config_file>")
        sys.exit(1)

    # Get the config file path from command-line arguments
    config_file = sys.argv[1]
    config = load_config(config_file)

    if config:
        operation: str = config["run"]
        nqubits: int = config["nqubits"]
        state: str = config["state"]

        observable: Dict = config["observable"]
        observable_hami_coeffi_cn: List[float] = observable["coefficients"]["cn"]
        observable_hami_coeffi_bn: List[float] = observable["coefficients"]["bn"]
        observable_hami_coeffi_r: float = observable["coefficients"]["r"]

        file_name_prefix: str = config["output"]["file_name_prefix"]
        fig_dpi: int = config["output"]["fig_dpi"]

        optimization: Dict = config["vqe"]["optimization"]
        ansatz: Dict = config["vqe"]["ansatz"]

        zne: Dict = config["zne"]
        zne_method: str = zne["method"]
        zne_degrees: List[int] = zne["degree"]
        zne_sampling: str = zne["sampling"]

        initialparam: Union[str, List[float]] = ansatz["init_param"]

        """
        Validate the user input.
        """
        observable_cn_len = len(observable_hami_coeffi_cn)
        observable_bn_len = len(observable_hami_coeffi_bn)

        if observable_cn_len != nqubits - 1 or observable_bn_len != nqubits:
            raise ValueError(
                f"Inconsistent lengths in observable Hamiltonian coeffiecients. "
                f"Expected lengths cn: {nqubits-1} and bn: {nqubits}, but got cn: {observable_cn_len} and bn: {observable_bn_len}."
            )
        target_obsevable = create_xy_hamiltonian(
            nqubits=nqubits, cn=observable_hami_coeffi_cn, bn=observable_hami_coeffi_bn, r=observable_hami_coeffi_r
        )

        if operation == "vqe":
            initialize_vqe()
        elif operation == "redundant":
            run_redundant()
        elif operation == "zne":
            initialize_zne()
