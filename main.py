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
iteration = config["iteration"]
optimizer = config["optimizer"]
constraint = config["constraint"]
execution_time = config["etime"]

ansatz_type = config["circuit"]["ansatztype"]
ti = config["circuit"]["time"]["ti"]
tf = config["circuit"]["time"]["tf"]
cn = config["circuit"]["coefficients"]["cn"]
bn = config["circuit"]["coefficients"]["bn"]
r = config["circuit"]["coefficients"]["r"]
noise_status = config["circuit"]["noise"]["status"]
noise_val = config["circuit"]["noise"]["value"]
noise_factor = config["circuit"]["noise"]["factor"]
init_param = config["circuit"]["param"]
draw_circ = config["circuit"]["draw"]

# Generate timestamp for unique file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"output_{timestamp}.txt")

# Noise parameters
nR, nT, nY = noise_param(nqubits, noise_factor)


# Open file for writing
with open(output_file, "w") as file:

    # Write output to file
    file.write(f"Qubit: {nqubits}\n")
    file.write(f"State: {state}\n")
    file.write(f"Layer: {layer}\n")
    file.write(f"Iteration: {iteration}\n")
    file.write(f"Optimizer: {optimizer}\n")
    file.write(f"Constraint: {constraint}\n")
    file.write(f"Ansatz type: {ansatz_type}\n")
    file.write(f"Time: [{ti}, {tf}]\n")
    file.write(f"cn, bn, r: [{cn}, {bn}, {r}]\n")
    file.write(f"Noise: [{noise_status}, {noise_val}, {noise_factor}]\n")
    file.write(f"nR, nT, nY: [{nR}, {nT}, {nY}]\n")
    file.write(f"Parameters: {init_param}\n")
    file.write(f"Draw: {draw_circ}\n")
    file.write("-----------------\n")

    
    start_time = time.time()
    vqe_instance = VQE(nqubits,
                        state, 
                        layer, 
                        iteration, 
                        optimizer, 
                        constraint,
                        ansatz_type, 
                        ti, tf, cn, bn, r, 
                        noise_status, noise_val, noise_factor, init_param, draw_circ)
    
    exact_sol, min_cost, param = vqe_instance.run_vqe()

    end_time = time.time()
    exe_time = end_time - start_time
  
    file.write(f"Exact sol: {exact_sol}\n")
    file.write(f"Min cost: {min_cost}\n")
    file.write(f"nR, nT, nY, E: [{nR}, {nT}, {nY}, {min_cost}]\n")
    file.write(f"Parameters: {param}\n")
    file.write(f"Execution time: {exe_time} sec\n") if execution_time else None
    file.write("-----------------\n")
    
    # Print output to console
    print(f"Qubit: {nqubits}")
    print(f"State: {state}")
    print(f"Layer: {layer}")
    print(f"Iteration: {iteration}")
    print(f"Optimizer: {optimizer}")
    print(f"Constraint: {constraint}")
    print(f"Ansatz type: {ansatz_type}")
    print(f"Time: [{ti}, {tf}]")
    print(f"cn, bn, r: [{cn}, {bn}, {r}]")
    print(f"Noise: [{noise_status}, {noise_val}, {noise_factor}]")
    print(f"nR, nT, nY: [{nR}, {nT}, {nY}]")  # Updated line
    print(f"Parameters: {init_param}")
    print(f"Draw: {draw_circ}")
    print("-----------------")
    print(f"Exact sol: {exact_sol}")
    print(f"Min cost: {min_cost}")
    print(f"nR, nT, nY, E: [{nR}, {nT}, {nY}, {min_cost}]")
    print(f"Parameters: {param}")
    print(f"Execution time: {exe_time} sec") if execution_time else None
    print("-----------------")

    if draw_circ:
        vqe_instance.drawCircuit(timestamp, 100)


# Print the path of the output file
print(f"Output saved to: {os.path.abspath(output_file)}")