import json
import os
import sys
import time
import uuid
from ast import Dict
from datetime import datetime
from typing import List, Union

import yaml

from configValidator import validate_yml_config
from src.modules import get_eigen_min
from src.observable import constructObservable
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
    initial_costs_history = []
    min_cost_history = []
    all_optimized_param = []
    time_evolution_hamiltonian_string = []

    print("=" * symbol_count + "Config" + "=" * symbol_count)
    print(config)
    print("=" * symbol_count + "VQE running" + "=" * symbol_count)

    start_time = time.time()

    for i in range(vqe_iteration):

        each_start_time = time.time()
        vqe_instance = IndirectVQE(
            nqubits=nqubits,
            state=state,
            observable=target_observable,
            optimization=optimization,
            ansatz=ansatz,
            identity_factor=[0, 0, 0, 0],
            init_param=initialparam,
        )
        vqe_output = vqe_instance.run_vqe()
        each_end_time = time.time()

        each_run_time = each_end_time - each_start_time

        print(f"VQE #{i+1} done with time taken: {each_run_time} sec.")

        # Extracting output
        initial_cost = vqe_output["initial_cost"]
        min_cost = vqe_output["min_cost"]
        optimized_param = vqe_output["optimized_param"]

        initial_costs_history.append(initial_cost)
        min_cost_history.append(min_cost)
        all_optimized_param.append(optimized_param)

    # Hamiltonian in time-evolution gate does NOT change in each iteration,
    # so append the Hamiltonian string outside the lopp.
    time_evolution_hamiltonian_string.append(str(vqe_instance.get_ugate_hamiltonain()))

    end_time = time.time()
    total_run_time = end_time - start_time

    nR, nT, nY, nCz = vqe_instance.get_noise_level()
    noise_level_list = [nR, nT, nY, nCz]

    print("=" * symbol_count + "Output" + "=" * symbol_count)

    print(f"Exact sol: {exact_cost}")
    print(f"Initial costs: {initial_costs_history}")
    print(f"Optimized minimum costs: {min_cost_history}")
    print(f"Optimized parameters: {all_optimized_param}")
    print(f"Noise level (nR, nT, nY, nCz): {noise_level_list} ")
    print(f"Run time: {total_run_time} sec")

    # Generate timestamp for unique file name
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{file_name_prefix}_{timestamp}_VQE.json")

    # Prepare the data to be written in JSON format
    output_data = {
        "config": config,
        "output": {
            "exact_sol": exact_cost,
            "initial_cost_history": initial_costs_history,
            "optimized_minimum_cost": min_cost_history,
            "optimized_parameters": all_optimized_param,
            "noise_level": noise_level_list,
            "run_time_sec": total_run_time,
        },
        "others": {
            "observable_string": str(target_observable),
            "time_evolution_gate_hamiltonian_string": time_evolution_hamiltonian_string,
        },
    }

    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=None, separators=(",", ":"))

    # Print the path of the output file
    print("=" * symbol_count + "File path" + "=" * symbol_count)
    print(f"Output saved to: {os.path.abspath(output_file)}")
    if circuit_draw_status:
        vqe_instance.drawCircuit(prefix=file_name_prefix, dpi=fig_dpi, filetype=fig_filetype)


def run_redundant() -> None:
    """
    Running redundant circuit.
    """
    data_points = []
    vqe_instances = []
    time_evolution_hamiltonian_string = []

    identity_factors: Union[List[int], List[List[int]]] = config["identity_factors"]

    print("=" * symbol_count + "Config" + "=" * symbol_count)
    print(config)
    print("=" * symbol_count + "VQE values at different noise levels" + "=" * symbol_count)

    if optimization["status"]:
        """
        The optimisation status is turned-off by default regardless of what
        user specify in config file.
        """
        print("WARNING! Optimization status on, but it will be ignored.")
        # Turn off the optimization
        optimization["status"] = False

    i = 1  # Cosmatic purpose only.

    start_time = time.time()

    for factor in identity_factors:

        each_start_time = time.time()
        vqe_instance = IndirectVQE(
            nqubits=nqubits,
            state=state,
            observable=target_observable,
            optimization=optimization,
            ansatz=ansatz,
            identity_factor=factor,
            init_param=initialparam,
        )
        vqe_output = vqe_instance.run_vqe()
        each_end_time = time.time()

        each_run_time = each_end_time - each_start_time

        # Extracting output
        initial_cost = vqe_output["initial_cost"]
        min_cost = vqe_output["min_cost"]
        optimized_param = vqe_output["optimized_param"]

        nR, nT, nY, nCz = vqe_instance.get_noise_level()
        noise_level_list = [nR, nT, nY, nCz]
        data_points.append([nR, nT, nY, nCz, initial_cost])

        vqe_instances.append(vqe_instance)

        print(f"#{i}")
        print(f"Exact sol: {exact_cost}")
        print(f"Initial cost: {initial_cost}")
        print(f"Optimized minimum cost: {min_cost}")
        print(f"Optimized parameters: {optimized_param}")
        print(f"Identity factor: {factor}")
        print(f"Noise level (nR, nT, nY, nCz): {noise_level_list} ")
        print(f"Time taken: {each_run_time} sec")

        # Not printing - in the list iteration, cosmatic purpose only.
        if i < len(identity_factors):
            print("-" * symbol_count)
        i += 1

    # Hamiltonian in time-evolution gate does NOT change in each iteration,
    # so append the Hamiltonian string outside the lopp.
    time_evolution_hamiltonian_string.append(str(vqe_instance.get_ugate_hamiltonain()))
    end_time = time.time()
    total_run_time = end_time - start_time

    print("=" * symbol_count + "Data points" + "=" * symbol_count)
    print(f"No of data points: {len(data_points)}")
    print(f"Data points: {data_points}")
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {total_run_time} sec")
    # Generate timestamp for unique file name
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{file_name_prefix}_{timestamp}_redundant.json")

    output_data = {
        "config": config,
        "output": {
            "data_points": data_points,
            "run_time_sec": runtime,
        },
        "others": {
            "observable_string": str(target_observable),
            "time_evolution_gate_hamiltonian_string": time_evolution_hamiltonian_string,
        },
    }

    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=None, separators=(",", ":"))
    print("=" * symbol_count + "File path" + "=" * symbol_count)
    print(f"Output saved to: {os.path.abspath(output_file)}")

    if circuit_draw_status:
        for instance in vqe_instances:
            # Sometimes time-based identifier did not work.
            unique_id: str = uuid.uuid4().hex  # Generate a unique identifier
            instance.drawCircuit(prefix=f"{file_name_prefix}_{unique_id}", dpi=fig_dpi, filetype=fig_filetype)


def initialize_zne() -> None:
    """
    Perfrom extrapolation to zero limit.
    """
    zne_config: Dict = config["zne"]
    zne_degree: int = zne_config["degree"]
    data_points = zne_config["data_points"]

    print("=" * symbol_count + "Config" + "=" * symbol_count)
    print(config)
    print("=" * symbol_count + "Zero-noise extrapolation result" + "=" * symbol_count)

    start_time = time.time()

    zne_instance = ZeroNoiseExtrapolation(
        datapoints=data_points, degree=zne_degree, method=zne_method, sampling_mode=zne_sampling
    )
    zne_value = zne_instance.getZne()
    end_time = time.time()
    runtime = end_time - start_time

    print(f"Exact sol: {exact_cost}")
    print(f"ZNE: {zne_value}")
    print(f"Total runtime: {runtime} sec")
    # Generate timestamp for unique file name
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{file_name_prefix}_{timestamp}_ZNE.json")

    output_data = {
        "config": config,
        "output": {
            "exact_sol": exact_cost,
            "zne_values": zne_value,
            "run_time_sec": runtime,
        },
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

    is_valid_config: bool = validate_yml_config(config)

    if config and is_valid_config:
        operation: str = config["run"]
        nqubits: int = config["nqubits"]
        state: str = config["state"]

        observable: Dict = config["observable"]
        observable_def: Dict = config["observable"]["def"]
        observable_coefficients: Dict = config["observable"]["coefficients"]

        # observable_hami_coeffi_cn: List[float] = observable["coefficients"]["cn"]
        # observable_hami_coeffi_bn: List[float] = observable["coefficients"]["bn"]
        # observable_hami_coeffi_r: float = observable["coefficients"]["r"]

        file_name_prefix: str = config["output"]["file_name_prefix"]
        circuit_draw_status: bool = config["output"]["draw"]["status"]
        fig_dpi: int = config["output"]["draw"]["fig_dpi"]
        fig_filetype: str = config["output"]["draw"]["type"]

        vqe_iteration: int = config["vqe"]["iteration"]
        optimization: Dict = config["vqe"]["optimization"]
        ansatz: Dict = config["vqe"]["ansatz"]

        zne: Dict = config["zne"]
        zne_method: str = zne["method"]
        zne_degrees: List[int] = zne["degree"]
        zne_sampling: str = zne["sampling"]

        initialparam: Union[str, List[float]] = ansatz["init_param"]

        # Create the target observable
        target_observable = constructObservable(
            nqubits=nqubits, definition=observable_def, coefficient=observable_coefficients
        )

        # Exact minimum eigen value of the target observable
        exact_cost: float = get_eigen_min(hamiltonian=target_observable)
        # """
        # Validate the user input.
        # """
        # observable_cn_len = len(observable_hami_coeffi_cn)
        # observable_bn_len = len(observable_hami_coeffi_bn)

        # if observable_cn_len != nqubits - 1 or observable_bn_len != nqubits:
        #     raise ValueError(
        #         f"Inconsistent lengths in observable Hamiltonian coeffiecients. "
        #         f"Expected lengths cn: {nqubits-1} and bn: {nqubits}, "
        #         f"but got cn: {observable_cn_len} and bn: {observable_bn_len}."
        #     )
        # target_observable = create_xy_hamiltonian(
        #     nqubits=nqubits, cn=observable_hami_coeffi_cn, bn=observable_hami_coeffi_bn, r=observable_hami_coeffi_r
        # )

        # exact_cost: float = get_eigen_min(hamiltonian=target_observable)

        if operation.lower() == "vqe":
            initialize_vqe()
        elif operation.lower() == "redundant":
            run_redundant()
        elif operation.lower() == "zne":
            initialize_zne()
        else:
            raise ValueError(f"Invalid run: {operation}. Valid runs are: 'vqe', 'redundant', and 'zne'.")
