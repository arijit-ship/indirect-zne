import os
import time
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
from qulacs import DensityMatrix, QuantumState
from qulacsvis import circuit_drawer
from scipy.optimize import minimize

from src.constraint import create_time_constraints
from src.modules import *
from src.createparam import create_param
from src.xy_hamiltonian import create_xy_hamiltonian
from src.ansatz import *

# Global variables
ansatz_circuit = None
constraint = None


class IndirectVQE:

    def __init__(
        self,
        nqubits: int,
        state: str,
        observable: Observable,
        optimization: Dict,
        ansatz: Dict,
        identity_factor: List[int],
        init_param: Union[List[float], str],
    ) -> None:

        self.nqubits = nqubits
        self.state = state

        # Optimization variables
        self.optimization_status: bool = optimization["status"]
        self.optimizer: str = optimization["algorithm"]
        self.iteration: int = optimization["iteration"]
        self.constraint: bool = optimization["constraint"]

        # Ansatz variables
        self.ansatz_draw: bool = ansatz["draw"]
        self.ansatz_type: str = ansatz["type"]
        self.ansatz_layer: int = ansatz["layer"]
        self.ansatz_gateset: int = ansatz["gateset"]
        self.ansatz_ti: float = ansatz["ugate"]["time"]["min"]
        self.ansatz_tf: float = ansatz["ugate"]["time"]["max"]
        self.ansatz_coeffi_cn: List = ansatz["ugate"]["coefficients"]["cn"]
        self.ansatz_coeffi_bn: List = ansatz["ugate"]["coefficients"]["bn"]
        self.ansatz_coeffi_r: float = ansatz["ugate"]["coefficients"]["r"]
        self.ansatz_noise_status: bool = ansatz["noise"]["status"]
        self.ansatz_noise_value: float = ansatz["noise"]["value"]
        self.ansatz_identity_factor: List[int] = identity_factor
        self.init_param = init_param

        # """
        # Validate the different args parsed form the config file and raise an error if inconsistancy found.
        # """
        # noise_value_len = len(ansatz["noise"]["value"])
        # identity_factor_len = len(self.ansatz_identity_factor)
        # ugate_cn_len = len(self.ansatz_coeffi_cn)
        # ugate_bn_len = len(self.ansatz_coeffi_bn)

        # if noise_value_len != 4:
        #     raise ValueError(f"Unsupported length of noise probability values: {noise_value_len}. Expected length: 4.")

        # if identity_factor_len != 3:
        #     raise ValueError(f"Unsupported length of noise factor: {identity_factor_len}. Expected length: 3.")

        # if ugate_cn_len != nqubits - 1 or ugate_bn_len != nqubits:
        #     raise ValueError(
        #         f"Inconsistent lengths in ugate Hamiltonian coefficients. "
        #         f"Expected lengths cn: {nqubits-1} and bn: {nqubits}, but got cn: {ugate_cn_len} and bn: {ugate_bn_len}."
        #     )

        # """
        # Create the Hamiltonians. We need to define two types of Hamiltonian. One is the observable observable whose expectation value VQE estimates, and the other one is the ugate (time-evolution) gate's XY-Hamiltonian. Based on coefficients provided in the config file, these two Hamiltonian needs to be created.

        # **Also, for bogus input, value error should be raised.**
        # """

        # Time-evolution gate's U(t)=exp(-iHt) Hamiltonian. For ZNE purpose H must be XY-Hamiltonian.
        if self.ansatz_type.lower() == "xy":
            self.ugate_hami = create_xy_hamiltonian(
                self.nqubits,
                self.ansatz_coeffi_cn,
                self.ansatz_coeffi_bn,
                self.ansatz_coeffi_r,
            )
        elif self.ansatz_type.lower() == "hardware":
            self.ugate_hami = None
        else:
            raise ValueError(f"Unsupported ansatz type: {self.ansatz_type}.")

        self.observable_hami = observable

    def create_ansatz(self, param: List[float]) -> QuantumCircuit:
        """
        Construct the ansatz circuit. There are two possibilities: noise less circuit and noisy circuit. Noisy circuit with noise probability 0 is equivalent to noiseless circuit.
        """

        if self.ansatz_noise_status:
            ansatz = create_noisy_ansatz(
                nqubits=self.nqubits,
                layer=self.ansatz_layer,
                gateset=self.ansatz_gateset,
                ugateH=self.ugate_hami,
                noise_prob=self.ansatz_noise_value,
                noise_factor=self.ansatz_identity_factor,
                param=param,
            )
        else:
            ansatz = noiseless_ansatz(
                nqubits=self.nqubits, layers=self.ansatz_layer, gateset= self.ansatz_gateset, ugateH=self.ugate_hami, param=param
            )
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
                param = create_param(self.ansatz_layer, self.ansatz_gateset, self.ansatz_ti, self.ansatz_tf)
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
                run_time = end_time - start_time
                min_cost_history.append(cost)
                optimized_param.append(param)
                param = create_param(self.ansatz_layer, self.ansatz_gateset, self.ansatz_ti, self.ansatz_tf)
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

    def get_noise_level(self) -> Tuple[Union[int, None], Union[int, None], Union[int, None]]:
        """
        Returns the noise levels.
        """
        if self.ansatz_noise_status:
            nR, nT, nY = noise_level(nqubits=self.nqubits, identity_factor=self.ansatz_identity_factor)["params"]
        else:
            nR, nT, nY = None, None, None

        return nR, nT, nY
