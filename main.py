import os
from sre_parse import State
import sys
import time
from datetime import datetime
from typing import List
import yaml

from src.modules import *
from src.vqe import IndirectVQE
from src.observable import create_ising_hamiltonian


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
    print("==========================Config==========================")
    print(config)
    print("==========================Optimization==========================")
    start_time = time.time()
    cost_value, exact_cost, min_cost_history, optimized_param = vqe_instance.run_vqe()
    nR, nT, nY = vqe_instance.get_noise_level()
    end_time = time.time()

    runtime = end_time - start_time
    print("==========================Output==========================")
    print(f"Exact sol: {exact_cost}")
    print(f"Initial cost: {cost_value}")
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
    output_file = os.path.join(output_dir, f"{file_name_prefix}_{timestamp}.txt")
    with open(output_file, "w") as file:
        file.write(f"Config: {config}\n")
        file.write(f"==========================\n")
        file.write(f"Exact sol: {exact_cost}\n")
        file.write(f"Initial cost: {cost_value}\n")
        file.write(f"Optimized minimum cost: {min_cost_history}\n")
        file.write(f"Optimized parameters: {optimized_param}\n")
        file.write(f"Run time: {runtime} sec")

    # Print the path of the output file
    print("==========================Filepath==========================")
    print(f"Output saved to: {os.path.abspath(output_file)}")
    if ansatz["draw"]:
        vqe_instance.drawCircuit(time_stamp=timestamp, dpi=100)


def initialize_zne() -> None:

    extrapolation_method: str = config["zne"]["extrapolation"]["method"]
    degrees: List[int] = config["zne"]["extrapolation"]["degrees"]
    identity_factors: Union[List[int], List[List[int]]] = config["zne"]["redundant_ansatz"]["identity_factors"]

    data_points = []

    print("==========================Config==========================")
    print(config)
    print("==========================Output==========================")
    start_time = time.time()
    # Turn off the optimization
    optimization["status"] = False
    for factor in identity_factors:

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
        data_points.append((nR, nT, nY, initial_cost))
        print(f"Exact sol: {exact_cost}")
        print(f"Initial cost: {initial_cost}")
        print(f"Optimized minimum cost: {min_cost_history}")
        print(f"Optimized parameters: {optimized_param}")
        print(f"Noise level (nR, nT, nY): ({nR}, {nT}, {nY}) ")
        print("-" * 50)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Data points: {data_points}")


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
        # Parse the configuration
        nqubits = config["vqe"]["nqubits"]
        state = config["vqe"]["state"]
        optimization = config["vqe"]["optimization"]
        ansatz = config["vqe"]["ansatz"]
        initialparam = config["vqe"]["param"]
        file_name_prefix = config["fprefix"]

        observable = config["vqe"]["observable"]
        observable_hami_coeffi_cn: List = observable["coefficients"]["cn"]
        observable_hami_coeffi_bn: List = observable["coefficients"]["bn"]
        observable_hami_coeffi_r: float = observable["coefficients"]["r"]

        observable_cn_len = len(observable_hami_coeffi_cn)
        observable_bn_len = len(observable_hami_coeffi_bn)

        if observable_cn_len != nqubits - 1 or observable_bn_len != nqubits:
            raise ValueError(
                f"Inconsistent lengths in observable Hamiltonian parameters. "
                f"Expected lengths cn: {nqubits-1} and bn: {nqubits}, but got cn: {observable_cn_len} and bn: {observable_bn_len}."
            )
        target_obsevable = create_ising_hamiltonian(
            nqubits=nqubits, cn=observable_hami_coeffi_cn, bn=observable_hami_coeffi_bn
        )

        if operation == "vqe":
            initialize_vqe()
        elif operation == "zne":
            initialize_zne()
