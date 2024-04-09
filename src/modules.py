import numpy as np
from qulacs import QuantumState, QuantumCircuit
from scipy.optimize import minimize
from functools import reduce
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import expm
from openfermion.ops.operators.qubit_operator import QubitOperator
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.gate import DenseMatrix
import random
from qulacs import QuantumState, QuantumCircuit, NoiseSimulator, DensityMatrix
from qulacs.gate import *
from qulacsvis import circuit_drawer
from scipy.optimize import LinearConstraint

def create_ansatz(nqubit, layer, time_evolution_gate, noise_prob, theta):
    
    """
    Args:
        nqubit: int, number of qubit
        time_evolution_gate: qulacs dense matrix gate
        noise_prob: noise probability
        theta: list, params for rotation gates.

    Returns:
        :class:`qulacs.QuantumCircuit`
    """

    circuit = QuantumCircuit(nqubit)
    
    for i in range(layer):

        # Add Rx to first and second qubits
        circuit.add_noise_gate(RX(0, theta[i]),"Depolarizing", noise_prob)
        circuit.add_noise_gate(RX(1, theta[i+1]),"Depolarizing", noise_prob)

        # Add identities with Rx and make redudant circuit
        # First qubit
        circuit.add_noise_gate(RX(0, theta[i]).get_inverse(), "Depolarizing", noise_prob)  # Rx_dagger
        circuit.add_noise_gate(RX(0, theta[i]), "Depolarizing", noise_prob)
    
        # Second qubit
        circuit.add_noise_gate(RX(1, theta[i+1]).get_inverse(), "Depolarizing", noise_prob)   # Rx_dagger
        circuit.add_noise_gate(RX(1, theta[i+1]), "Depolarizing", noise_prob)

        # Add Ry to first and second qubits
        circuit.add_noise_gate(RY(0, theta[i+2]),"Depolarizing", noise_prob)
        circuit.add_noise_gate(RY(1, theta[i+3]),"Depolarizing", noise_prob)

        # Add identities with Ry and make redudant circuit
        # First qubit
        circuit.add_noise_gate(RY(0, theta[i+2]).get_inverse(), "Depolarizing", noise_prob)  # Ry_dagger
        circuit.add_noise_gate(RY(0, theta[i+2]), "Depolarizing", noise_prob)
    
        # Second qubit
        circuit.add_noise_gate(RY(1, theta[i+3]).get_inverse(), "Depolarizing", noise_prob)   # Ry_dagger
        circuit.add_noise_gate(RY(1, theta[i+3]), "Depolarizing", noise_prob)

        # Add CZ gate
        circuit.add_gate(CZ(0,1))

        # Add multi-qubit U gate
        circuit.add_gate(time_evolution_gate)

        # Add depolarizing noise
        for i in range(nqubit):
            circuit.add_noise_gate(Identity(i), "Depolarizing", noise_prob)

        # Add Y gates to all odd qubits
        for i in range(nqubit):
            if (i+1) % 2 != 0:
                circuit.add_noise_gate(Y(i), "Depolarizing", noise_prob)

        # Again add multi-qubit U gate
        circuit.add_gate(time_evolution_gate)

        # Add depolarizing noise
        for i in range(nqubit):
            circuit.add_noise_gate(Identity(i), "Depolarizing", noise_prob)

        # Add Y gates to all odd qubits
        for i in range(nqubit):
            if (i+1) % 2 != 0:
                circuit.add_noise_gate(Y(i), "Depolarizing", noise_prob)

        # Again add multi-qubit U gate
        circuit.add_gate(time_evolution_gate)

        # Add depolarizing noise
        for i in range(nqubit):
            circuit.add_noise_gate(Identity(i), "Depolarizing", noise_prob)
        
    return circuit


def create_param(layer, ti, tf):
    """
    Creates parameter for the citcuit. Parameters are time, and theta: angle for rotation gates.
    
    time: 0 - max time
    theta: 0 - 1

    Args:
        layer: int, number of layer
        ti: float, initial time
        tf: float, final time

    Returns:
        class:`numpy.ndarray`: [
        t1, t2, ... td, theta1, ..., theatd * 4
    ]

    """
    
    param = np.array([])
    
    # Time param
    time = np.random.uniform(ti,tf, layer + 1)
    time = np.sort(time) # Time must be in incresing order
    for i in time:
        param = np.append(param, i)

    # Theta param
    theta = np.random.random(layer*4)*1e-1  # Each layer has 4 rotation gates
    for i in theta:
        param = np.append(param, i)

    return(param)

def create_xy_hamiltonian(nqubit, cn, bn, r):
    """
    Args:
        nqubit: int, number of qubits
        cn: list with length of nqubit, 0 - 1, coupling constant
        bn: list with length of nqubit, 0 - 1, magnetic field
        r: list with length of nqubit, 0 - 1, anisotropy param
    
    Returns:
        qulacs observable
    """
    hami = QubitOperator()
    for i in range(nqubit-1):
        hami += (0.5*cn*(1+r)) * QubitOperator(f"X{i} X{i+1}")
        hami += (0.5*cn*(1-r)) * QubitOperator(f"Y{i} Y{i+1}")
    for j in range(nqubit):
        hami += bn*QubitOperator(f"Z{j}")

    return (create_observable_from_openfermion_text(str(hami)))


def create_ising_hamiltonian(nqubit, cn, bn):
    """"
    Args:
        nqubit: int, number of qubits
        cn: list with length of nqubit, 0 - 1, coupling constant
        bn: list with length of nqubit, 0 - 1, magnetic field
    
    Returns:
        qulacs observable
    """
    hami = QubitOperator()
    for i in range(nqubit-1):
        hami += cn * QubitOperator(f"X{i} X{i+1}")
    for j in range(nqubit):
        hami += bn * QubitOperator(f"Z{j}")
    
    return(create_observable_from_openfermion_text(str(hami)))
    


def create_time_evo_unitary(observable, ti, tf):

    """
    Args:
        observable: qulacs observable
        ti: initial time
        tf: final time
    
    Returns:
        a dense matrix gate U(t) = exp(-iHt)
    """
    # Get the qubit number
    n = observable.get_qubit_count()
    # Converting to a matrix
    H_mat = observable.get_matrix()

    # Converting to array
    H_mat_array = H_mat.toarray()

    # this is exp(-iHt)
    exponent = expm(-1*1j*H_mat_array*(tf-ti))
    return (DenseMatrix([i for i in range(n)], exponent))
    
# def create_time_evo_unitary(observable, ti, tf):
    # """
    # Args:
        # observable: qulacs observable
        # ti: initial time
        # tf: final time
    
    # Returns:
        # a dense matrix gate U(t) = exp(-iHt)
    # """
    # # Get the qubit number
    # n = observable.get_qubit_count()
    # # Converting to a matrix
    # H_mat = observable.get_matrix().toarray()

    # # Diagonalize the Hamiltonian
    # diag, eigen_vecs = np.linalg.eigh(H_mat)

    # # Compute the exponent of diagonalized Hamiltonian
    # exponent_diag = np.diag(np.exp(-1j * (tf - ti) * diag))

    # # Construct the time evolution operator
    # time_evol_op = np.dot(np.dot(eigen_vecs, exponent_diag), eigen_vecs.T.conj())

    # return DenseMatrix([i for i in range(n)], time_evol_op)


def parametric_ansatz(nqubit, layer, hamitolian, param):
    """
    Args:
        nqubit: int, number of qubit
        layer: int, number of layer
        param: class:`numpy.ndarray`, params for rotation gates, time evolution gate: [
        t1, t2, ... td, theta1, ..., theatd * 4]

    Returns:
        :class:`qulacs.QuantumCircuit`
    """
    circuit = QuantumCircuit(nqubit)

    flag = layer + 1 # Tracking angles in param ndarray
    
    for i in range(layer):

        # Rotation gate
        circuit.add_gate(RX(0, param[flag]))
        circuit.add_gate(RX(1, param[flag+1]))

        circuit.add_gate(RY(0, param[flag+2]))
        circuit.add_gate(RY(1, param[flag+3]))

        # CZ gate
        circuit.add_gate(CZ(0,1))

        # Time evolution gate
        ti = param [i]
        tf = param [i+1]
        time_evo_gate = create_time_evo_unitary(hamitolian, ti, tf)
        circuit.add_gate(time_evo_gate)
        
        flag += 4 # Each layer has four angle-params. 
        
    return(circuit)

def he_ansatz_circuit(n_qubit, depth, theta_list):
    """he_ansatz_circuit
    Returns hardware efficient ansatz circuit.

    Args:
        n_qubit (:class:`int`):
            the number of qubit used (equivalent to the number of fermionic modes)
        depth (:class:`int`):
            depth of the circuit.
        theta_list (:class:`numpy.ndarray`):
            rotation angles.
    Returns:
        :class:`qulacs.QuantumCircuit`
    """
    circuit = QuantumCircuit(n_qubit)
    for d in range(depth):
        for i in range(n_qubit):
            circuit.add_gate(merge(RY(i, theta_list[2*i+2*n_qubit*d]), RZ(i, theta_list[2*i+1+2*n_qubit*d])))
        for i in range(n_qubit//2):
            circuit.add_gate(CZ(2*i, 2*i+1))
        for i in range(n_qubit//2-1):
            circuit.add_gate(CZ(2*i+1, 2*i+2))
    for i in range(n_qubit):
        circuit.add_gate(merge(RY(i, theta_list[2*i+2*n_qubit*depth]), RZ(i, theta_list[2*i+1+2*n_qubit*depth])))

    return circuit  


def create_time_constraints(time_params_length, all_params_length) -> LinearConstraint:
    """
    Create constraints for time params to ensure each time parameter is positive
    and differences between consecutive time parameters are non-negative.

    Parameters:
        time_params_length (int): Number of time parameters.
        all_params_length (int): Total number of parameters including theta parameters.

    Returns:
        LinearConstraint: Linear constraint object representing the constraints.
    """
    matrix = np.zeros((2 * time_params_length, all_params_length))  # Initialize matrix

    # Set constraints for each time parameter to be positive
    for i in range(time_params_length):
        matrix[i, i] = 1  # t_i

    # Set constraints for differences between consecutive time parameters to be non-negative
    for i in range(1, time_params_length):
        matrix[time_params_length + (i - 1), i - 1] = -1  # -t_{i-1}
        matrix[time_params_length + (i - 1), i] = 1  # t_i

    return LinearConstraint(matrix, np.zeros(2 * time_params_length), np.inf)


   



