import os

import matplotlib.pyplot as plt
from qulacs import DensityMatrix, QuantumState
from qulacsvis import circuit_drawer
from scipy.optimize import minimize

from src.constraint import *
from src.modules import *

# Global variables
ansatz_circuit = None
constraint = None


class VQE:

    def __init__(
        self,
        n: int,
        state: str,
        layer: int,
        type: str,
        time: dict,
        coefficients: dict,
        optimization: dict,
        noise_profile: dict,
        init_param: list[float],
        draw: bool,
    ) -> None:

        self.n = n
        self.state = state
        self.layer = layer
        self.optimization_status = optimization["status"]
        self.optimizer = optimization["algorithm"]
        self.iteration = optimization["iteration"]
        self.constraint = optimization["constraint"]
        self.type = type
        self.ti = time["min"]
        self.tf = time["max"]
        self.cn = coefficients["cn"]
        self.bn = coefficients["bn"]
        self.r = coefficients["r"]
        self.noise_status = noise_profile["status"]
        self.noise_value = noise_profile["value"]
        self.noise_factor = noise_profile["factor"]
        self.init_param = init_param
        self.draw = draw

        # XY spin chain Hamiltonian in time evolution gate
        self.xy_ham = create_xy_hamiltonian(self.n, self.cn, self.bn, self.r)

        # Ising spin chain Hamiltonian to be used as observable
        self.ising_ham = create_ising_hamiltonian(self.n, self.cn, self.bn)

    def create_ansatz(self, param: list[float]):

        if self.type == "xy":
            if self.noise_status:
                ansatz = create_noisy_ansatz(
                    self.n,
                    self.layer,
                    self.noise_value,
                    self.noise_factor,
                    self.xy_ham,
                    param,
                )
            else:
                ansatz = parametric_ansatz(self.n, self.layer, self.xy_ham, param)

        elif self.type == "ising":
            if self.noise_status:
                ansatz = create_noisy_ansatz(
                    self.n,
                    self.layer,
                    self.noise_value,
                    self.noise_factor,
                    self.ising_ham,
                    param,
                )
            else:
                ansatz = parametric_ansatz(self.n, self.layer, self.ising_ham, param)

        elif self.type == "hardware":
            ansatz = he_ansatz_circuit(self.n, self.layer, param)

        else:
            raise ValueError(f"Unsupported ansatz type: {self.type}")

        return ansatz

    # Cost function
    def cost(self, param: list[float]):

        # Create the quantum state
        if self.state == "DMatrix":
            state = DensityMatrix(self.n)

        elif self.state == "Satevector":
            state = QuantumState(self.n)

        else:
            raise ValueError(f"Unsupported quantum state: {self.state}")

        global ansatz_circuit  # Access the global ansatz_circuit
        ansatz_circuit = self.create_ansatz(param=param)

        ansatz_circuit.update_quantum_state(state)  # type: ignore
        cost = self.ising_ham.get_expectation_value(state)

        return cost

    def run_optimization(self, parameters, constraint):

        cost_history = []
        min_cost = None
        optimized_params = None  # List to store optimized parameters (solutions)
        param_constraint = None

        opt = minimize(
            self.cost,
            parameters,
            method=self.optimizer,
            constraints=constraint,
            callback=lambda x: cost_history.append(self.cost(x)),
        )

        min_cost = np.min(cost_history)

        optimized_params = opt.x.tolist()
        # optimized_params.append(np.array2string(opt.x, separator=', '))

        return min_cost, optimized_params

    def run_vqe(self):

        global constraint
        cost_value = None
        exact_cost = None
        min_cost_history = []
        optimized_param = []

        if isinstance(self.init_param, str):
            if self.init_param.lower() == "random":
                param = create_param(self.layer, self.ti, self.tf)
            else:
                raise ValueError(f"Unsupported initial parameters: {self.init_param}")
        elif isinstance(self.init_param, list):
            param = self.init_param
        else:
            raise ValueError(f"Unsupported initial parameters: {self.init_param}")

        cost_value = self.cost(param)  # type: ignore
        exact_cost = exact_sol(self.ising_ham)

        if not self.optimization_status:
            min_cost_history = None
            optimized_param = None

        else:
            if self.constraint and self.optimizer == "SLSQP":
                constraint = create_time_constraints(self.layer, len(param))

            elif self.optimizer != "SLSQP" and self.constraint:
                raise ValueError(f"Constaint not supported for: {self.optimizer}")

            for _ in range(self.iteration):

                cost, param = self.run_optimization(param, constraint)  # type: ignore
                min_cost_history.append(cost)
                optimized_param.append(param)
                param = create_param(self.layer, self.ti, self.tf)

        return cost_value, exact_cost, min_cost_history, optimized_param

    def drawCircuit(self, time_stamp, dpi):

        global ansatz_circuit
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # Go up one level
        output_dir = os.path.join(parent_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"circuit_{time_stamp}.png")

        circuit_drawer(ansatz_circuit, "mpl")  # type: ignore
        plt.savefig(output_file, dpi=dpi)
        plt.close()
        # Print the path of the output file
        print(f"Circuit fig saved to: {os.path.abspath(output_file)}")
