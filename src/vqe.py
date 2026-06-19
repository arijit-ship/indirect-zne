import os
from datetime import datetime
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from qulacs import DensityMatrix, Observable, QuantumCircuit, QuantumState
from qulacsvis import circuit_drawer
from scipy.optimize import minimize

from src.ansatz import create_noisy_ansatz, noiseless_ansatz
from src.constraint import create_time_constraints, create_tf_fixed_constraint
from src.createparam import create_param
from src.hamiltonian import (
    create_heisenberg_hamiltonian,
    create_ising_hamiltonian,
    create_xy_hamiltonian,
    create_xy_iss_hamiltonian,
)
from src.modules import calculate_noise_levels


class IndirectVQE:

    def __init__(
        self,
        nqubits: int,
        state: str,
        observable: Observable,
        vqe_profile: Dict,
        ansatz_profile: Dict,
        noise_profile: Dict,
        identity_factors: List[int],
        init_param: list[float] | str,
    ) -> None:

        self.nqubits = nqubits
        self.state = state

        # Optimization variables
        self.optimization_status: bool = vqe_profile["optimization"]["status"]
        self.optimizer: str = vqe_profile["optimization"]["algorithm"]
        self.constraint: bool = vqe_profile["optimization"]["constraint"]

        # Ansatz variables
        self.ansatz_type: str = ansatz_profile["ugate"]["type"]
        self.ansatz_layer: int = ansatz_profile["layer"]
        self.ansatz_gateset: int = ansatz_profile["gateset"]
        self.ansatz_ti: float = ansatz_profile["ugate"]["time"]["min"]
        self.ansatz_tf: float = ansatz_profile["ugate"]["time"]["max"]
        self.ansatz_coeffi_cn: List = ansatz_profile["ugate"]["coefficients"]["cn"]
        self.ansatz_coeffi_bn: List = ansatz_profile["ugate"]["coefficients"]["bn"]
        self.ansatz_coeffi_r: float = ansatz_profile["ugate"]["coefficients"]["r"]
        # Noise profile
        self.noise_profile: dict = noise_profile
        self.ansatz_noise_status: bool = noise_profile["status"]
        self.ansatz_noise_type: str = noise_profile["type"]
        self.ansatz_noise_value: float = noise_profile["noise_prob"]
        self.ansatz_noise_on_init_param: bool = noise_profile["noise_on_init_param"]["status"]
        self.ansatz_identity_factors: List[int] = identity_factors
        self.init_param = init_param

        # Ansatz
        self.ansatz: dict = None
        self.ansatz_circuit: QuantumCircuit = None

        # Lie-trotte
        self.lie_trotter_details: list = []

        """
        Validate the different args parsed form the config file and raise an error if inconsistancy found.
        """
        noise_value_len = len(noise_profile["noise_prob"])
        identity_factor_len = len(self.ansatz_identity_factors)
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

        if self.ansatz_noise_on_init_param:
            raise NotImplementedError("Adding noise to the initial parameters is not implemented yet.")

    def create_ansatz(self, param: List[float]) -> QuantumCircuit:
        """
        Construct the ansatz circuit. There are two possibilities: noise less circuit and noisy circuit.
        Noisy circuit with noise probability 0 is equivalent to noiseless circuit.
        """

        if self.ansatz_noise_status:
            self.ansatz = create_noisy_ansatz(
                nqubits=self.nqubits,
                layers=self.ansatz_layer,
                gateset=self.ansatz_gateset,
                ugateH=self.ugate_hami,
                ansatz_noise_type=self.ansatz_noise_type,
                ansatz_noise_prob=self.ansatz_noise_value,
                param=param,
                identity_factors=self.ansatz_identity_factors,
            )
            # If Lie-troter
            if self.ansatz_noise_type == "time-depol-trotter":
                #print ("TROTTE!!!!\n\n\n\n\n\n")
                self.lie_trotter_details = self.ansatz.get("trotter_details", [])
                #print(self.lie_trotter_details)
            else:
                self.lie_trotter_details = None
        else:
            self.ansatz = noiseless_ansatz(
                nqubits=self.nqubits,
                layers=self.ansatz_layer,
                gateset=self.ansatz_gateset,
                ugateH=self.ugate_hami,
                param=param,
            )
        self.ansatz_circuit = self.ansatz["circuit"]
        
        

        return self.ansatz_circuit

    def _cost_function(self, param: List[float]) -> float:
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

    def cost_function(self, param: List[float], n_shots: int=10000) -> float:
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
        cost, var, std = self._estimate_expectation_shots(self.observable_hami, state, n_shots)

        return cost
    
    def _estimate_expectation_shots(
        self,
        observable_hami,
        state,
        n_shots: int,
    ):
        """
        Shot-based expectation value estimation with variance.
        Splits total n_shots equally between X and Z measurement bases.
        """
        n = state.get_qubit_count()
        n_terms = observable_hami.get_term_count()

        # ----------------------------------------------------------
        # Parse Hamiltonian
        # ----------------------------------------------------------
        xx_terms = []
        zz_terms = []
        x_terms = []
        z_terms = []
        constant_energy = 0.0

        for i in range(n_terms):
            term = observable_hami.get_term(i)
            coeff = term.get_coef().real
            pauli_ids = term.get_pauli_id_list()
            pauli_qubits = term.get_index_list()

            # Handle pure Identity terms (if any)
            if len(pauli_ids) == 0 or all(p == 0 for p in pauli_ids):
                constant_energy += coeff
                continue

            if len(pauli_ids) == 2 and all(p == 1 for p in pauli_ids):
                xx_terms.append((coeff, pauli_qubits[0], pauli_qubits[1]))
            elif len(pauli_ids) == 2 and all(p == 3 for p in pauli_ids):
                zz_terms.append((coeff, pauli_qubits[0], pauli_qubits[1]))
            elif len(pauli_ids) == 1 and pauli_ids[0] == 1:
                x_terms.append((coeff, pauli_qubits[0]))
            elif len(pauli_ids) == 1 and pauli_ids[0] == 3:
                z_terms.append((coeff, pauli_qubits[0]))
            else:
                print(f"[WARNING] Term {i} skipped (unsupported Pauli type)")

        # ----------------------------------------------------------
        # Allocate Shot Budget
        # ----------------------------------------------------------
        # Split shots between the two bases needed
        has_x = bool(xx_terms or x_terms)
        has_z = bool(zz_terms or z_terms)
        
        # Simple equal split strategy
        shots_x = n_shots // 2 if (has_x and has_z) else (n_shots if has_x else 0)
        shots_z = n_shots - shots_x if (has_x and has_z) else (n_shots if has_z else 0)

        # ----------------------------------------------------------
        # X-basis measurement group
        # ----------------------------------------------------------
        x_shot_energies = np.zeros(shots_x)
        if shots_x > 0:
            # Clone state and rotate to X basis (Hadamard)
            x_state = state.copy()
            from qulacs import QuantumCircuit
            rot = QuantumCircuit(n)
            for q in range(n):
                rot.add_H_gate(q)
            rot.update_quantum_state(x_state)

            # Sampling: Qulacs returns an array of integers
            samples = x_state.sampling(shots_x)
            for shot_idx, bitstring in enumerate(samples):
                # Bit extraction: bitstring LSB is qubit 0
                e_shot = 0.0
                
                # Optimization: only extract bits for qubits we actually care about,
                # or pre-unpack up to max needed index.
                bits = [(bitstring >> i) & 1 for i in range(n)]
                xvals = [1 - 2 * b for b in bits]

                for coeff, qi, qj in xx_terms:
                    e_shot += coeff * xvals[qi] * xvals[qj]
                for coeff, qi in x_terms:
                    e_shot += coeff * xvals[qi]

                x_shot_energies[shot_idx] = e_shot

        # ----------------------------------------------------------
        # Z-basis measurement group
        # ----------------------------------------------------------
        z_shot_energies = np.zeros(shots_z)
        if shots_z > 0:
            samples = state.sampling(shots_z)
            for shot_idx, bitstring in enumerate(samples):
                e_shot = 0.0
                bits = [(bitstring >> i) & 1 for i in range(n)]
                zvals = [1 - 2 * b for b in bits]

                for coeff, qi, qj in zz_terms:
                    e_shot += coeff * zvals[qi] * zvals[qj]
                for coeff, qi in z_terms:
                    e_shot += coeff * zvals[qi]

                z_shot_energies[shot_idx] = e_shot

        # ----------------------------------------------------------
        # Mean and Variance Calculations
        # ----------------------------------------------------------
        mean_x = x_shot_energies.mean() if shots_x > 0 else 0.0
        mean_z = z_shot_energies.mean() if shots_z > 0 else 0.0
        
        # Combined mean includes the static identity offset
        mean_energy = mean_x + mean_z + constant_energy

        var_energy = 0.0
        # Var(Sample Mean) = Var(Shot Energies) / N_shots
        if shots_x > 1:
            var_energy += np.var(x_shot_energies, ddof=1) / shots_x
        if shots_z > 1:
            var_energy += np.var(z_shot_energies, ddof=1) / shots_z

        stderr = np.sqrt(var_energy)

        return mean_energy, var_energy, stderr

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

        constraints = None
        vqe_constraint = None
        isRandom: bool = False
        initial_cost: float = 0
        min_cost: float | None = None
        sol_optimized_param = None

        # Storing density matrices
        initial_density_matrix_json = None
        final_density_matrix_json = None
        store_init_param_created = None

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

            store_init_param_created = random_initial_param.tolist()
            # (2) Calculate the initial cost with random initial param
            initial_cost = self.cost_function(param=random_initial_param)

            # (3) Checking constraint before optimization
            if self.constraint and self.optimizer == "SLSQP":

                vqe_constraint = create_time_constraints(self.ansatz_layer, len(random_initial_param))
                
                if self.ansatz_noise_type == "time-depol-trotter":
                    num_time = self.ansatz_layer
                    total_params = len(random_initial_param)
                    tf_index = num_time - 1  # Correct: index of the last time parameter

                    # Standardize constraints for SLSQP
                    vqe_constraint = create_time_constraints(num_time, total_params)
                    t_final_constraints = create_tf_fixed_constraint(tf_index, total_params, self.ansatz_tf)
                    
                    # Passing as a list of LinearConstraint objects is supported in SciPy 1.1.0+
                    constraints = [vqe_constraint, t_final_constraints]
                else:
                    constraints = vqe_constraint

            elif self.optimizer != "SLSQP" and self.constraint:
                raise ValueError(f"Constaint not supported for: {self.optimizer}")

            # (4) Run optimization
            min_cost, sol_optimized_param = self.run_optimization(
                parameters = random_initial_param,
                constraint = constraints
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
        # --- ADD THIS LOGIC HERE ---

        
        if sol_optimized_param is not None:
            # Initialize a fresh DensityMatrix object
            state = DensityMatrix(self.nqubits)
            
            # Re-create the ansatz circuit with the best parameters found
            final_circuit = self.create_ansatz(param=sol_optimized_param)
            
            # Apply the circuit to the state
            final_circuit.update_quantum_state(state)
            
            # Get the actual numerical matrix (numpy array)
            final_density_matrix_json = state.to_json()

            # esetting
            state = 0
            state = DensityMatrix(self.nqubits)
            # Re-create the ansatz circuit with the best parameters found
            initial_circuit = self.create_ansatz(param=store_init_param_created)
            # Apply the circuit to the state
            initial_circuit.update_quantum_state(state)
            # Get the actual numerical matrix (numpy array)
            initial_density_matrix_json = state.to_json()


        vqe_result: Dict = {
            "initial_cost": initial_cost,
            "min_cost": min_cost,
            "init_random_param": store_init_param_created,
            "optimized_param": sol_optimized_param,
            "initial_density_matrix": initial_density_matrix_json,
            "final_density_matrix": final_density_matrix_json,
            "lie_trotter_details": self.lie_trotter_details
        }

        return vqe_result

    def drawCircuit(self, prefix: str, dpi: int, filetype: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        output_dir = os.path.join(parent_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        if filetype.lower() == "svg":
            output_file = os.path.join(output_dir, f"{prefix}_circuit_{timestamp}.svg")
        elif filetype.lower() == "png":
            output_file = os.path.join(output_dir, f"{prefix}_circuit_{timestamp}.png")
        else:
            raise ValueError(f"Invalid circuit figure file type: {filetype}. Valid types are: SVG, PNG.")

        chunks = self.ansatz.get("chunks")
        if chunks is None:
            raise ValueError("chunks not available for this ansatz type (e.g. time-depol-trotter)")
        circuit_drawer(chunks[0], "mpl")  # only this line
        plt.savefig(output_file, dpi=dpi)
        plt.close()
        print(f"Circuit fig saved to: {os.path.abspath(output_file)}")

    def get_noise_level(self) -> Tuple[Union[int, None], Union[int, None], Union[int, None]]:
        """
        Returns the noise levels.
        """

        noise_details = calculate_noise_levels(
            nqubits=self.nqubits, identity_factors=self.ansatz_identity_factors, noise_profile=self.noise_profile
        )

        return noise_details

    def get_ugate_hamiltonain(self) -> Observable:
        """
        Returns time-evolution gate Hamiltonian.
        """
        return self.ugate_hami
