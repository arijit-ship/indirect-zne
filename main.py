import os
from sre_parse import State
import sys
import time
from datetime import datetime

import yaml

from src.modules import *
from src.vqe import IndirectVQE


def load_config(config_path):
    # Check if the config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return None

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":
    # Check if a config file argument is provided
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <config_file>")
        sys.exit(1)

    # Get the config file path from command-line arguments
    config_file = sys.argv[1]
    config = load_config(config_file)

    if config:

        # Parse the configuration
        nqubits = config["nqubits"]
        state = config["state"]
        observable = config["observable"]
        optimization = config["optimization"]
        ansatz = config["ansatz"]
        initialparam = config["param"]
        file_name_prefix = config["fprefix"]

        if ansatz["noise"]["status"]:
            nR, nT, nY = noise_param(nqubits=nqubits, noise_factor=ansatz["noise"]["factor"])["params"]
        else:
            nR, nT, nY = None, None, None

        vqe_instance = IndirectVQE(
            nqubits=nqubits,
            state=state,
            observable=observable,
            optimization=optimization,
            ansatz=ansatz,
            init_param=initialparam,
        )
        print("==========================Config==========================")
        print(config)
        print("==========================Optimization==========================")
        start_time = time.time()
        cost_value, exact_cost, min_cost_history, optimized_param = vqe_instance.run_vqe()
        end_time = time.time()
        runtime = end_time-start_time
        print("==========================Output==========================")
        print(f"Exact sol: {exact_cost}")
        print(f"Initial cost: {cost_value}")
        print(f"Optimized minimum cost: {min_cost_history}")
        print(f"Optimized parameters: {optimized_param}")
        print(f"(nR, nT, nY, E): ({nR}, {nT}, {nY}, {cost_value})")
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
            file.write((f"(nR, nT, nY, E): ({nR}, {nT}, {nY}, {cost_value})\n"))
            file.write(f"Run time: {runtime} sec")

        # Print the path of the output file
        print(f"Output saved to: {os.path.abspath(output_file)}")
        if ansatz["draw"]:
            vqe_instance.drawCircuit(time_stamp=timestamp, dpi=100)
       
