import os
from sympy import continued_fraction_convergents
import yaml
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
layer = config["layer"]
iteration = config["iteration"]
optimizer = config["optimizer"]
constraint = config["constraint"]

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
output_file = f"output_{timestamp}.txt"


# Open file for writing
with open(output_file, "w") as file:

    # Write output to file
    file.write(f"Qubit: {nqubits}\n")
    file.write(f"Layer: {layer}\n")
    file.write(f"Iteration: {iteration}\n")
    file.write(f"Optimizer: {optimizer}\n")
    file.write(f"Constraint: {constraint}\n")
    

    vqe_instance = VQE(nqubits, 
                        layer, 
                        iteration, 
                        optimizer, 
                        constraint,
                        ansatz_type, 
                        ti, tf, cn, bn, r, 
                        noise_status, noise_val, noise_factor, init_param, draw_circ)
    
    exact_sol, min_cost, param = vqe_instance.run_vqe()

    file.write(f"Exact sol: {exact_sol}\n")
    file.write(f"Min cost: {min_cost}\n")
    file.write(f"Parameters: {param}\n")
    file.write("-----------------\n")
    
    # Print output to console
    print(f"Qubit: {nqubits}")
    print(f"Layer: {layer}")
    print(f"Iteration: {iteration}")
    print(f"Optimizer: {optimizer}")
    print(f"Constraint: {constraint}")
    print(f"Exact sol: {exact_sol}")
    print(f"Min cost: {min_cost}")
    print(f"Parameters: {param}")
    print("-----------------")

    if draw_circ:
        vqe_instance.drawCircuit(timestamp, 100)

# Print the path of the output file
print(f"Output saved to: {os.path.abspath(output_file)}")