import os
from typing import Dict, List, Union

import matplotlib.pyplot as plt
from qulacs import DensityMatrix, QuantumState
from qulacsvis import circuit_drawer
from scipy.optimize import minimize

from src.constraint import create_time_constraints
from src.modules import *

# Global variables
ansatz_circuit = None
constraint = None


class IndirectVQE:

    def __init__(
        self,
        nqubits: int,
        state: str,
        observable: Dict,
        optimization: Dict,
        ansatz: Dict,
        init_param: Union[List[float], str],
    ) -> None:

        self.nqubits = nqubits
        self.state = state
        
        # Trget observable (many-body Hamiltonian most likely)
        self.observable_hami_def: str = observable["def"]
        self.observable_hami_coeffi_cn: List = observable["coefficients"]["cn"]
        self.observable_hami_coeffi_bn: List = observable["coefficients"]["bn"]
        self.observable_hami_coeffi_r: float = observable["coefficients"]["r"]

        # Optimization variables
        self.optimization_status: bool = optimization["status"]
        self.optimizer: str = optimization["algorithm"]
        self.iteration: int = optimization["iteration"]
        self.constraint: bool = optimization["constraint"]

        # Ansatz variables
        self.ansatz_draw: bool = ansatz["draw"]
        self.ansatz_layer: int = ansatz["layer"]
        self.ansatz_gateset: int = ansatz["gateset"]
        self.ansatz_evolution: str = ansatz["unitary"]["def"]
        self.ansatz_ti: float = ansatz["unitary"]["time"]["min"]
        self.ansatz_tf: float = ansatz["unitary"]["time"]["max"]
        self.ansatz_coeffi_cn: List = ansatz["unitary"]["coefficients"]["cn"]
        self.ansatz_coeffi_bn: List = ansatz["unitary"]["coefficients"]["bn"]
        self.ansatz_coeffi_r: float = ansatz["unitary"]["coefficients"]["r"]
        self.ansatz_noise_status: bool = ansatz["noise"]["status"]
        self.ansatz_noise_value: float = ansatz["noise"]["value"]
        self.ansatz_noise_factor: int = ansatz["noise"]["factor"]
        self.init_param = init_param

        """
        Validate the different args parsed form the config file and raise an error if inconsistancy found.
        """
        noise_value_len = len(ansatz["noise"]["value"])
        noise_factor_len = len(ansatz["noise"]["factor"])
        unitary_cn_len = len(self.ansatz_coeffi_cn)
        unitary_bn_len = len(self.ansatz_coeffi_bn)

        observable_cn_len = len(self.observable_hami_coeffi_cn)
        observable_bn_len = len(self.observable_hami_coeffi_bn)

        if noise_value_len != 4:
            raise ValueError(f"Unsupported length of noise probability values: {noise_value_len}. Expected length: 4.")
        if noise_factor_len != 4:
            raise ValueError(f"Unsupported length of noise factor: {noise_factor_len}. Expected length: 4.")
        if observable_cn_len != nqubits - 1 or observable_bn_len != nqubits:
            raise ValueError(
                f"Inconsistent lengths in observable Hamiltonian parameters. "
                f"Expected lengths cn: {nqubits-1} and bn: {nqubits}, but got cn: {observable_cn_len} and bn: {observable_bn_len}."
            )
        if unitary_cn_len != nqubits - 1 or unitary_bn_len != nqubits:
            raise ValueError(
                f"Inconsistent lengths in unitary Hamiltonian parameters. "
                f"Expected lengths cn: {nqubits-1} and bn: {nqubits}, but got cn: {unitary_cn_len} and bn: {unitary_bn_len}."
            )

        """
        Create the Hamiltonians. We need to define two types of Hamiltonian. One is the observable observable whose expectation value VQE estimates, and the other one is the unitary gate's Hamiltonian. Based on the agrs provided in the config file, these two Hamiltonian needs to be created.

        **Also, for bogus input, value error should be raised.**
        """
        if self.observable_hami_def.lower() == "ising":
            self.observable_hami = create_ising_hamiltonian(
                self.nqubits,
                self.observable_hami_coeffi_cn,
                self.observable_hami_coeffi_bn,
            )
        else:
            raise ValueError(f"Unsupported observable Hamiltonian: {self.observable_hami_def}")

        if self.ansatz_evolution.lower() == "xy":
            self.unitary_hami = create_xy_hamiltonian(
                self.nqubits,
                self.ansatz_coeffi_cn,
                self.ansatz_coeffi_bn,
                self.ansatz_coeffi_r,
            )
        else:
            raise ValueError(f"Unsupported time-evolution Hamiltonian: {self.ansatz_evolution}")

    def create_ansatz(self, param: List[float]) -> QuantumCircuit:
        """
        Construct the ansatz circuit. There are two possibilities: noise less circuit and noisy circuit. Noisy circuit with noise probability 0 is equivalent to noiseless circuit.
        """

        if self.ansatz_noise_status:
            ansatz = create_noisy_ansatz(
                nqubits=self.nqubits,
                layer=self.ansatz_layer,
                gateset=self.ansatz_gateset,
                unitaryH=self.unitary_hami,
                noise_prob=self.ansatz_noise_value,
                noise_factor=self.ansatz_noise_factor,
                param=param,
            )
        else:
            ansatz = parametric_ansatz(nqubits=self.nqubits, layers=self.ansatz_layer, unitaryH=self.unitary_hami, param=param)
        return ansatz

    def cost_function(self, param: List[float]) -> float:
        """
        Variational quantum eigensolver cost function.
        """

        if self.state.lower() == "dmatrix":
            state = DensityMatrix(self.nqubits)
        elif self.state.lower() == "statevector":
            state = QuantumState(self.nqubits)
        else:
            raise ValueError(f"Unsupported state: {self.state}. Supported states are: 'dmatrix', 'statevector'")

        global ansatz_circuit
        ansatz_circuit = self.create_ansatz(param=param)
        ansatz_circuit.update_quantum_state(state)
        cost = self.observable_hami.get_expectation_value(state)

        return cost

    def run_optimization(self, parameters, constraint):

        cost_history = []
        min_cost = None
        optimized_params = None  # List to store optimized parameters (solutions)
        param_constraint = None

        opt = minimize(
            self.cost_function,
            parameters,
            method=self.optimizer,
            constraints=constraint,
            callback=lambda x: cost_history.append(self.cost_function(x)),
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
                param = create_param(self.ansatz_layer, self.ansatz_ti, self.ansatz_tf)
            else:
                raise ValueError(f"Unsupported initial parameters: {self.init_param}.")
        elif isinstance(self.init_param, list):
            param = self.init_param
        else:
            raise ValueError(f"Unsupported initial parameters: {self.init_param}.")

        cost_value = self.cost_function(param)  # type: ignore
        exact_cost = exact_sol(self.observable_hami)

        if not self.optimization_status:
            min_cost_history = None
            optimized_param = None

        else:
            if self.constraint and self.optimizer == "SLSQP":
                constraint = create_time_constraints(self.layer, len(param))

            elif self.optimizer != "SLSQP" and self.constraint:
                raise ValueError(f"Constaint not supported for: {self.optimizer}")

            for i in range(self.iteration):
                start_time = time.time()
                cost, param = self.run_optimization(param, constraint)  # type: ignore
                end_time = time.time()
                run_time = end_time-start_time
                min_cost_history.append(cost)
                optimized_param.append(param)
                param = create_param(self.ansatz_layer, self.ansatz_ti, self.ansatz_tf)
                print(f"Iteration {i+1} done with time taken: {run_time} sec.")

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
