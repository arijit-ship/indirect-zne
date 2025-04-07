import os
from datetime import datetime
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from qulacs import DensityMatrix, Observable, QuantumCircuit, QuantumState
from qulacsvis import circuit_drawer
from scipy.optimize import minimize

from src.ansatz import create_noisy_ansatz, noiseless_ansatz
from src.constraint import create_time_constraints
from src.createparam import create_param
from src.hamiltonian import (
    create_heisenberg_hamiltonian,
    create_ising_hamiltonian,
    create_xy_hamiltonian,
    create_xy_iss_hamiltonian,
)
from src.modules import noise_level


class IndirectVQE:

    def __init__(
        self,
        nqubits: int,
        state: str,
        observable: Observable,
        optimization: Dict,
        ansatz: Dict,
        identity_factor: List[int],
        init_param: list[float] | str,
    ) -> None:

        self.nqubits = nqubits
        self.state = state

        # Optimization variables
        self.optimization_status: bool = optimization["status"]
        self.optimizer: str = optimization["algorithm"]
        self.constraint: bool = optimization["constraint"]

        # Ansatz variables
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

        # Ansatz
        self.ansatz_circuit = None

        """
        Validate the different args parsed form the config file and raise an error if inconsistancy found.
        """
        noise_value_len = len(ansatz["noise"]["value"])
        identity_factor_len = len(self.ansatz_identity_factor)
        ugate_cn_len = len(self.ansatz_coeffi_cn)
        ugate_bn_len = len(self.ansatz_coeffi_bn)

        if noise_value_len != 4:
            raise ValueError(f"Unsupported length of noise probability values: {noise_value_len}. Expected length: 4.")
        if identity_factor_len != 4:
            raise ValueError(f"Invalid identity factor length: {identity_factor_len}. Expected length: 4.")

        if ugate_cn_len != nqubits - 1 or ugate_bn_len != nqubits:
            raise ValueError(
                f"Inconsistent lengths in ugate Hamiltonian coefficients. "
                f"Expected lengths cn: {nqubits-1} and bn: {nqubits}, "
                f"but got cn: {ugate_cn_len} and bn: {ugate_bn_len}."
            )

        """
        Create the Hamiltonians. We need to define two types of Hamiltonian.
        One is the observable observable whose expectation value VQE estimates,
        and the other one is the ugate (time-evolution) gate's XY-Hamiltonian.
        Based on coefficients provided in the config file, these two Hamiltonian needs to be created.

        **Also, for bogus input, value error should be raised.**
        """

        # Time-evolution gate's i.e. U(t)=exp(-iHt) Hamiltonian H.
        # Ansatz type can be: 'custom', 'xy-iss' (stands for xy-identity scaling supported), 'ising', or 'heisenberg'.
        # For ZNE purpose, type mus be 'xy-iss' which is an XY-Hamiltonian.
        # Coeffiecients are applicable for only 'custom' and are overwritten for others.
        if self.ansatz_type.lower() == "custom":
            self.ugate_hami = create_xy_hamiltonian(
                nqubits=self.nqubits,
                cn=self.ansatz_coeffi_cn,
                bn=self.ansatz_coeffi_bn,
                r=self.ansatz_coeffi_r,
            )

        elif self.ansatz_type.lower() == "xy-iss":
            self.ugate_hami = create_xy_iss_hamiltonian(nqubits=self.nqubits)

        elif self.ansatz_type.lower() == "ising":
            self.ugate_hami = create_ising_hamiltonian(nqubits=self.nqubits)

        elif self.ansatz_type.lower() == "heisenberg":
            self.ugate_hami = create_heisenberg_hamiltonian(
                self.nqubits,
                self.ansatz_coeffi_cn,
            )
        # elif self.ansatz_type.lower() == "hardware":
        #     self.ugate_hami = None
        else:
            raise ValueError(
                f"Unsupported ansatz type: {self.ansatz_type}. "
                f"Expected type: 'custom', 'ising', 'xy-iss', or 'heisenberg'."
            )

        self.observable_hami = observable

    def create_ansatz(self, param: List[float]) -> QuantumCircuit:
        """
        Construct the ansatz circuit. There are two possibilities: noise less circuit and noisy circuit.
        Noisy circuit with noise probability 0 is equivalent to noiseless circuit.
        """

        if self.ansatz_noise_status:
            self.ansatz_circuit = create_noisy_ansatz(
                nqubits=self.nqubits,
                layer=self.ansatz_layer,
                gateset=self.ansatz_gateset,
                ugateH=self.ugate_hami,
                noise_prob=self.ansatz_noise_value,
                noise_factor=self.ansatz_identity_factor,
                param=param,
            )
        else:
            self.ansatz_circuit = noiseless_ansatz(
                nqubits=self.nqubits,
                layers=self.ansatz_layer,
                gateset=self.ansatz_gateset,
                ugateH=self.ugate_hami,
                param=param,
            )
        return self.ansatz_circuit

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

        self.ansatz_circuit = self.create_ansatz(param=param)
        self.ansatz_circuit.update_quantum_state(state)
        cost = self.observable_hami.get_expectation_value(state)

        return cost

    def run_optimization(self, parameters, constraint):

        cost_history = []
        min_cost = None
        optimized_params = None  # List to store optimized parameters (solutions)

        opt = minimize(
            self.cost_function,
            parameters,
            method=self.optimizer,
            constraints=constraint,
            callback=lambda x: cost_history.append(self.cost_function(x)),
        )

        min_cost = np.min(cost_history)

        optimized_params = opt.x.tolist()

        return min_cost, optimized_params

    def run_vqe(self) -> Dict:

        vqe_constraint = None
        isRandom: bool = False
        initial_cost: float = 0
        min_cost: float | None = None
        sol_optimized_param = None

        # Decide the initial param type: random or provided. If provided, validate the length.
        if isinstance(self.init_param, str) and self.init_param.lower() == "random":
            isRandom = True
        elif isinstance(self.init_param, list):
            expected_length = self.ansatz_layer + (self.ansatz_layer * 4 * self.ansatz_gateset)
            if len(self.init_param) == expected_length:
                isRandom = False
            else:
                raise ValueError(
                    f"Invalid initial parameters length: {len(self.init_param)}. Expected: {expected_length}."
                )
        else:
            raise ValueError(f"Unsupported initial parameters: {self.init_param}.")

        # Optimization is off
        if not self.optimization_status:

            if isRandom:
                random_initial_param = create_param(
                    self.ansatz_layer, self.ansatz_gateset, self.ansatz_ti, self.ansatz_tf
                )
                initial_cost = self.cost_function(param=random_initial_param)

            else:
                initial_param = self.init_param
                initial_cost = self.cost_function(param=initial_param)

        # Optimization is on
        else:

            # (1) Create random initial param
            random_initial_param = create_param(self.ansatz_layer, self.ansatz_gateset, self.ansatz_ti, self.ansatz_tf)

            # (2) Calculate the initial cost with random initial param
            initial_cost = self.cost_function(param=random_initial_param)

            # (3) Checking constraint before optimization
            if self.constraint and self.optimizer == "SLSQP":
                vqe_constraint = create_time_constraints(self.layer, len(random_initial_param))

            elif self.optimizer != "SLSQP" and self.constraint:
                raise ValueError(f"Constaint not supported for: {self.optimizer}")

            # (4) Run optimization
            min_cost, sol_optimized_param = self.run_optimization(
                parameters=random_initial_param, constraint=vqe_constraint
            )  # type: ignore

            # for i in range(self.iteration):

            #     # (1) Create random initial param
            #     param = create_param(self.ansatz_layer, self.ansatz_gateset, self.ansatz_ti, self.ansatz_tf)

            #     # (2) Calculate the initial cost with random initial param
            #     initial_costs.append(self.cost_function(param=param))

            #     # (3) Run optimization
            #     start_time = time.time()
            #     cost, sol_optimized_param = self.run_optimization(param, constraint)  # type: ignore
            #     end_time = time.time()

            #     run_time = end_time - start_time
            #     min_cost_history.append(cost)
            #     optimized_param.append(sol_optimized_param)

            #     print(f"Iteration {i+1} done with time taken: {run_time} sec.")

        vqe_result: Dict = {
            "initial_cost": initial_cost,
            "min_cost": min_cost,
            "optimized_param": sol_optimized_param,
        }

        return vqe_result

    def drawCircuit(self, prefix: str, dpi: int, filetype: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # Go up one level
        output_dir = os.path.join(parent_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        if filetype.lower() == "svg":
            output_file = os.path.join(output_dir, f"{prefix}_circuit_{timestamp}.svg")
        elif filetype.lower() == "png":
            output_file = os.path.join(output_dir, f"{prefix}_circuit_{timestamp}.png")
        else:
            raise ValueError(f"Invalid circuit figure file type: {filetype}. Valid types are: SVG, PNG.")

        circuit_drawer(self.ansatz_circuit, "mpl")  # type: ignore
        plt.savefig(output_file, dpi=dpi)
        plt.close()
        # Print the path of the output file
        print(f"Circuit fig saved to: {os.path.abspath(output_file)}")

    def get_noise_level(self) -> Tuple[Union[int, None], Union[int, None], Union[int, None]]:
        """
        Returns the noise levels.
        """
        if self.ansatz_noise_status:
            nR, nT, nY, nCz = noise_level(nqubits=self.nqubits, identity_factor=self.ansatz_identity_factor)["params"]
        else:
            nR, nT, nY, nCz = None, None, None, None

        return nR, nT, nY, nCz

    def get_ugate_hamiltonain(self) -> Observable:
        """
        Returns time-evolution gate Hamiltonian.
        """
        return self.ugate_hami
