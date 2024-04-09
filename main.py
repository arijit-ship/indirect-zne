import os
from src.modules import *
from datetime import datetime

class VQE:
    def __init__(self, n, layer, ti, tf, cn, bn, r):
        self.n = n
        self.layer = layer
        self.ti = ti
        self.tf = tf
        self.cn = cn
        self.bn = bn
        self.r = r
        self.xy_ham = create_xy_hamiltonian(n, cn, bn, r)   # XY spin chain Hamiltonian
        self.ising_ham = create_ising_hamiltonian(n, cn, bn)    # Ising spin chain Hamiltonian to be used as observable
    

    # Cost function with noise less parametric ansatz
    def cost(self, param):
        #state = DensityMatrix(self.n)
        state = QuantumState(self.n)
        ansatz_circuit = parametric_ansatz(self.n, self.layer, self.xy_ham, param)
        #ansatz_circuit = he_ansatz_circuit(self.n, self.layer, param)
        ansatz_circuit.update_quantum_state(state)
        return self.ising_ham.get_expectation_value(state)
    
    def run_vqe(self, iteration):
    
        cost_history = []
        min_cost = []
        optimized_params = []  # List to store optimized parameters
        
        method = "SLSQP"
        options = {"disp": True, "maxiter": 2000}

        for _ in range(iteration):
            init_param = create_param(self.layer, self.ti, self.tf)
            #init_param = np.random.random(2*self.n*(self.layer+1))*1e-1
            constraint = create_time_constraints(self.layer, len(init_param))
            opt = minimize(
                self.cost,
                init_param,
                method=method,
                #constraints=constraint,
                callback=lambda x: cost_history.append(self.cost(x))
            )
            min_cost.append(np.min(cost_history))
        print(len(init_param))
        print("-----")
        optimized_params.append(opt.x)
        return min_cost, optimized_params

# Example usage:
n = 7  # Number of qubits
layer = 20
ti = 0
tf = 10
cn = 1
bn = 1
r = 0
iteration = 30

# Generate timestamp for unique file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"output_{timestamp}.txt"

# Open file for writing
with open(output_file, 'w') as file:
    # Loop over the range
    for i in range(10):
        vqe_instance = VQE(n, layer + i, ti, tf, cn, bn, r)
        min_cost, param = vqe_instance.run_vqe(iteration=iteration)
        # Write output to file
        file.write(f"Layer: {layer + i}:\n")
        file.write(f"Min cost: {min_cost}\n")
        file.write(f"Parameters: {param}\n")
        file.write("===================\n")
        # Print output to console
        print(f"Layer: {layer + i}:")
        print(f"Min cost: {min_cost}")
        print(f"Parameters: {param}")
        print("===================")

# Print the path of the output file
print(f"Output saved to: {os.path.abspath(output_file)}")
