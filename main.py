import os
import yaml
import time
from datetime import datetime
from src.modules import *
from src.vqe import *

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the config file
config_path = os.path.join(current_dir, "config.yml")

# Load YAML file
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Parsing config file
nqubits = config["nqubits"]
state = config["state"]
layer = config["layer"]

execution_time = config["etime"]

optimization = config["optimization"]

ansatz_type = config["circuit"]["ansatztype"]
time_unitary = config["circuit"]["time"]
coeffiecients = config["circuit"]["coefficients"]

noise_profile = config["circuit"]["noise"]
noise_factor = config["circuit"]["noise"]["factor"]
init_param = config["circuit"]["param"]
draw_circ = config["circuit"]["draw"]

# Generate timestamp for unique file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok = True)
output_file = os.path.join(output_dir, f"output_{timestamp}.txt")

# Noise parameters
nR, nT, nY = noise_param(nqubits, noise_factor)



# Open file for writing
with open(output_file, "w") as file:

    # Write output to file
    file.write(f"Qubit: {nqubits}\n")
    file.write(f"State: {state}\n")
    file.write(f"Layer: {layer}\n")
    file.write(f"Optimizer: {optimization}\n")
    file.write(f"Ansatz type: {ansatz_type}\n")
    file.write(f"Time: {time_unitary}\n")
    file.write(f"Coefficients: {coeffiecients}\n")
    file.write(f"Noise profile: {noise_profile}\n")
    file.write(f"nR, nT, nY: [{nR}, {nT}, {nY}]\n")
    file.write(f"Initial parameters: {init_param}\n")
    file.write(f"Draw: {draw_circ}\n")
    file.write("-----------------\n")

    
    start_time = time.time()
    vqe_instance = VQE(n = nqubits,
                    state = state,
                    layer = layer,
                    type = ansatz_type,
                    time = time_unitary,
                    optimization = optimization,
                    noise_profile = noise_profile,
                    init_param = init_param,
                    coefficients = coeffiecients,
                    draw = draw_circ
                    )

    cost_value, exact_cost, min_cost_history, optimized_param  = vqe_instance.run_vqe()

    end_time = time.time()
    exe_time = end_time - start_time
  
    file.write(f"Exact sol: {exact_cost}\n")
    file.write(f"Initial cost: {cost_value}\n")
    file.write(f"Optimized minimum costs: {min_cost_history}\n")
    file.write(f"Optimized parameters: {optimized_param}\n")
    file.write(f"Execution time: {exe_time} sec\n") if execution_time else None
    file.write("-----------------\n")
    
    # Print output to console
    print(f"Qubit: {nqubits}")
    print(f"State: {state}")
    print(f"Layer: {layer}")
    print(f"Optimizer: {optimization}")
    print(f"Ansatz type: {ansatz_type}")
    print(f"Time: {time_unitary}")
    print(f"Coefficients: {coeffiecients}")
    print(f"Noise profile: {noise_profile}")
    print(f"nR, nT, nY: [{nR}, {nT}, {nY}]")  # Updated line
    print(f"Parameters: {init_param}")
    print(f"Draw: {draw_circ}")
    print("-----------------")
    print(f"Exact sol: {exact_cost}")
    print(f"Initial cost: {cost_value}")
    print(f"Optimized minimum costs: {min_cost_history}")
    print(f"Optimized parameters: {optimized_param}")
    print(f"Execution time: {exe_time} sec") if execution_time else None
    print("-----------------")

    if draw_circ:
        vqe_instance.drawCircuit(timestamp, 100)


# Print the path of the output file
print(f"Output saved to: {os.path.abspath(output_file)}")
