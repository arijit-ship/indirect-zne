"""
This scripts contains all the necessary functions for VQE and ZNE.
"""
import time
import numpy as np
from scipy.linalg import expm
from qulacs.gate import *
from qulacs.observable import create_observable_from_openfermion_text
from qulacs import Observable, QuantumCircuit 
from openfermion.ops.operators.qubit_operator import QubitOperator


# Global variables to store the eigenvalues and eigenvectors
diag = None
eigen_vecs = None

def create_param(layer: int, ti: float, tf: float) -> np.ndarray:
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
    time = np.random.uniform(ti,tf, layer)
    time = np.sort(time) # Time must be in incresing order
    for i in time:
        param = np.append(param, i)

    # Theta param
    theta = np.random.random(layer*4)*1e-1  # Each layer has 4 rotation gates
    for i in theta:
        param = np.append(param, i)

    return(param)

def create_xy_hamiltonian(nqubit: int, cn: float, bn: float, r: float) -> Observable:
    """
    Args:
        nqubit: `int`, number of qubits
        cn: `float`, 0 - 1, coupling constant
        bn: `float`, 0 - 1, magnetic field
        r: `float`, 0 - 1, anisotropy param
    
    Returns:
        qulacs observable
    """
    hami = QubitOperator()
    for i in range(nqubit-1):
        hami += (0.5*cn*(1+r)) * QubitOperator(f"X{i} X{i+1}")
        #hami += (0.5*cn*(1-r)) * QubitOperator(f"Y{i} Y{i+1}")
        hami += (0.5*cn*(1-r)) * QubitOperator(f"Z{i} Z{i+1}")
    for j in range(nqubit):
        hami += bn*QubitOperator(f"Y{j}")

    return (create_observable_from_openfermion_text(str(hami)))


def create_ising_hamiltonian(nqubit: int, cn: float, bn: float) -> Observable:
    """"
    Args:
        nqubit: int, number of qubits
        cn: `float`, 0 - 1, coupling constant
        bn: `float`, 0 - 1, magnetic field
    
    Returns:
        qulacs observable
    """
    hami = QubitOperator()
    for i in range(nqubit-1):
        hami += cn * QubitOperator(f"X{i} X{i+1}")
    for j in range(nqubit):
        hami += bn * QubitOperator(f"Z{j}")
    
    return(create_observable_from_openfermion_text(str(hami)))

def exact_sol(hamiltonian: Observable) -> float:
    """
    Finds the exact minimum eigen value for a given matrix.

    Args:
        hamiltonian: `qulacs_core.Observable`
    
    Returns:
        min_eigenvalue: `float`, minimum eigen value
    """
    eigenvalues, _ = np.linalg.eigh(hamiltonian.get_matrix().toarray())
    min_eigenvalue = np.min(eigenvalues)
    return min_eigenvalue  
    
def create_time_evo_unitary(observable: Observable, ti: float, tf: float):
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
    H_mat = observable.get_matrix().toarray()

    # Compute eigenvalues and eigenvectors only once and reuse them
    global diag, eigen_vecs
    
    if diag is None or eigen_vecs is None:
        diag, eigen_vecs = np.linalg.eigh(H_mat)

    # Compute the exponent of diagonalized Hamiltonian
    exponent_diag = np.diag(np.exp(-1j * (tf - ti) * diag))

    # Construct the time evolution operator
    time_evol_op = np.dot(np.dot(eigen_vecs, exponent_diag), eigen_vecs.T.conj())

    return DenseMatrix([i for i in range(n)], time_evol_op)

def parametric_ansatz(nqubit: int, layer: int, hamitolian: Observable, param: list[float]) -> QuantumCircuit:
    """
    Args:
        nqubit: `int`, number of qubit
        layer: `int`, number of layer
        hamiltonian: `qulacs_core.Observable`, hamiltonian used in time evolution gate i.e. exp(-iHt)
        param: class:`numpy.ndarray`, params for rotation gates, time evolution gate: [
        t1, t2, ... td, theta1, ..., theatd * 4]

    Returns:
        :class:`qulacs.QuantumCircuit`
    """
    circuit = QuantumCircuit(nqubit)

    flag = layer # Tracking angles in param ndarray
    
    for i in range(layer):

        # Rotation gate
        circuit.add_gate(RX(0, param[flag]))
        circuit.add_gate(RX(1, param[flag+1]))

        circuit.add_gate(RY(0, param[flag+2]))
        circuit.add_gate(RY(1, param[flag+3]))

        # CZ gate
        circuit.add_gate(CZ(0,1))

        if i == 0:
            # Time evolution gate
            ti = 0
            tf = param [i]
            time_evo_gate = create_time_evo_unitary(hamitolian, ti, tf)
            circuit.add_gate(time_evo_gate)
        else:
            ti = param[i]
            tf = param[i+1]
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
   
def create_noisy_ansatz(nqubit: int, layer: int , noise_prob: list[float], noise_factor: list[int], hamiltonian: Observable, param: list[float]) -> QuantumCircuit:
    
    """
    Creates noisy redundant ansatz.

    Args:
        nqubit: `int`, number of qubit
        layer: `int`, number of layer
        noise_prob: `float`, noise probability between 0-1
        hamiltonian: `qulacs_core.Observable`, hamiltonian used in time evolution gate i.e. exp(-iHt)
        noise_factor: `list`, noise factor for rotational gates, time evolution unitary gate and Y gate.. Based on this redundant noisy identites are constructed. For example, if value is 1, only one set of identities are introduced.
        param: class:`numpy.ndarray`, params for rotation gates, time evolution gate: [
        t1, t2, ... td, theta1, ..., theatd * 4]

    Returns:
        :class:`QuantumCircuit`
    """

    # Creates redundant circuit
    circuit = create_redundant(nqubit, layer, noise_prob, noise_factor, hamiltonian, param)
  
    return circuit

def add_ygate_odd(circuit: QuantumCircuit, noise_y_prob: float, y_gate_factor: int) -> QuantumCircuit:
    """
    Adds Y gates to odd qubit wires.

    Args:
        circuit: `QuantumCircuit`
        noise_y_prob: `float`,  noise probability for Y gate
        y_gate_factor: `int`, Number of Y_daggar*Y identity gates

    Returns:
        circuit: `QuantumCircuit`
    """
    qubit_count = circuit.get_qubit_count()

    for i in range(qubit_count):
        if (i+1) % 2 != 0:
                    circuit.add_noise_gate(Y(i), "Depolarizing", noise_y_prob)

                    # Add redundant Y gate identities
                    for _ in range(y_gate_factor):
                        circuit.add_noise_gate(Y(i).get_inverse(), "Depolarizing", noise_y_prob)
                        circuit.add_noise_gate(Y(i), "Depolarizing", noise_y_prob)
    return circuit


def create_redundant(nqubit: int, layers: int, noise_prob: list[float], noise_factor: list[int], hamiltonian, param: list[float]) -> QuantumCircuit:

    """
    Creates a noisy circuit with redundant noisy indentities based on a given noise factor.

    Args:
        nqubit: `int`, number of qubit
        layer: `int`, number of layer
        noise_prob: `float`, noise probability between 0-1
        noise_factor: `list`, containts identity scaling factor got rotational gates and time evolution gate
        hamiltonian: `qulacs_core.Observable`, hamiltonian used in time evolution gate i.e. exp(-iHt)
        param: class:`numpy.ndarray`, params for rotation gates, time evolution gate: [
        t1, t2, ... td, theta1, ..., theatd * 4]

    Returns:
        circuit: `QuantumCircuit`
    """
    

    circuit = QuantumCircuit(nqubit)

    flag = layers # Tracking angles in param ndarrsy

    # Noise propabilities
    noise_r_prob = noise_prob [0]
    noise_cz_prob = noise_prob [1]
    noise_u_prob = noise_prob [2]
    noise_y_prob = noise_prob [3]
   
    # Noisy identy factors
    r_gate_factor = noise_factor[0] # Identity sacaling factor for rotational gates
    u_gate_factor = noise_factor[1] # Identity scaling factor for time evolution gates
    y_gate_factor = noise_factor [2] # Identity scaling factor for Y gate
    
    for layer in range(layers):

        # Add Rx to first and second qubits
        circuit.add_noise_gate(RX(0, param[flag]),"Depolarizing", noise_r_prob)
        circuit.add_noise_gate(RX(1, param[flag+1]),"Depolarizing", noise_r_prob)

        # Add identities with Rx and make redudant circuit

        for _ in range(r_gate_factor):

            # First qubit
            circuit.add_noise_gate(RX(0, param[flag]).get_inverse(), "Depolarizing", noise_r_prob)  # Rx_dagger
            circuit.add_noise_gate(RX(0, param[flag]), "Depolarizing", noise_r_prob)
    
            # Second qubit
            circuit.add_noise_gate(RX(1, param[flag+1]).get_inverse(), "Depolarizing", noise_r_prob)   # Rx_dagger
            circuit.add_noise_gate(RX(1, param[flag+1]), "Depolarizing", noise_r_prob)

        # Add Ry to first and second qubits
        circuit.add_noise_gate(RY(0, param[flag+2]),"Depolarizing", noise_r_prob)
        circuit.add_noise_gate(RY(1, param[flag+3]),"Depolarizing", noise_r_prob)

        # Add identities with Ry and make redudant circuit

        for _ in range(r_gate_factor):
            # First qubit
            circuit.add_noise_gate(RY(0, param[flag+2]).get_inverse(), "Depolarizing", noise_r_prob)  # Ry_dagger
            circuit.add_noise_gate(RY(0, param[flag+2]), "Depolarizing", noise_r_prob)
        
            # Second qubit
            circuit.add_noise_gate(RY(1, param[flag+3]).get_inverse(), "Depolarizing", noise_r_prob)   # Ry_dagger
            circuit.add_noise_gate(RY(1, param[flag+3]), "Depolarizing", noise_r_prob)

        # Add CZ gate
        circuit.add_noise_gate(CZ(0,1), "Depolarizing", noise_cz_prob)

        # Add multi-qubit U gate
        if layer == 0:
            ti = 0
            tf = param[layer]
            time_evo_gate =  create_time_evo_unitary(hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)

        else:
            ti = param[layer]
            tf = param[layer+1]
            time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)

        # Add depolarizing noise to time evolution gate.
        for i in range(nqubit):
            circuit.add_noise_gate(Identity(i), "Depolarizing", noise_u_prob)

        # XY spin chain identity 

        for _ in range(u_gate_factor):
            # Add Y gates to all odd qubits
            circuit = add_ygate_odd(circuit, noise_y_prob, y_gate_factor)
            
            # Again add multi-qubit U gate
            if layer == 0:
                ti = 0
                tf = param[i]
                time_evo_gate =  create_time_evo_unitary(hamiltonian, ti, tf)
                circuit.add_gate(time_evo_gate)

            else:
                ti = param[i]
                tf = param[i+1]
                time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
                circuit.add_gate(time_evo_gate)

            # Add depolarizing noise
            for i in range(nqubit):
                circuit.add_noise_gate(Identity(i), "Depolarizing", noise_u_prob)

            # Add Y gates to all odd qubits
            circuit = add_ygate_odd(circuit, noise_y_prob, y_gate_factor)

            # Again add multi-qubit U gate
            if layer == 0:
                ti = 0
                tf = param[i]
                time_evo_gate =  create_time_evo_unitary(hamiltonian, ti, tf)
                circuit.add_gate(time_evo_gate)

            else:
                ti = param[i]
                tf = param[i+1]
                time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
                circuit.add_gate(time_evo_gate)

            # Add depolarizing noise
            for i in range(nqubit):
                circuit.add_noise_gate(Identity(i), "Depolarizing", noise_u_prob)

        
        flag += 4 # Each layer has four angle-params

    return circuit

def noise_param(nqubit: int, noise_factor: list[int]) -> tuple[int, int, int]:
    """
    Finds nR, nT, and nY for a given noise factor.

    Arg:

        nqubit: `int`, number of qubits.    
    
        noise_factor: `list[int, int, int]`, represernts the redundant noisy indities for qubit gates and time evolution gate. For example, `nR = 1` adds one noisy identity (Rx_daggar*Rx) for Rx gate and one noisy identity (Ry_daggar * Ry) for Ry gate in the ciruit.

        layer: `int`, depth of the quantum circuit.

    Returns:

        nR, nT, nY: `int`, each value is proportional to the number of corresponding noisy gates in the circuit.
    """

    nY = 0
    nR = 4
    nT = 1

    r_gate_factor = noise_factor [0]    # For rotational gate
    u_gate_factor = noise_factor [1]    # For time evolution gate
    y_gate_factor = noise_factor [2]    # For Y gate
    

    if r_gate_factor != 0 or u_gate_factor != 0:

        # Count the number of odd qubits
        odd_n = (nqubit // 2) + 1 if nqubit % 2 != 0 else nqubit // 2

        nY += (odd_n + (2 * y_gate_factor*odd_n))
        nR += 8 * r_gate_factor
        nT += 2 * u_gate_factor

    return nR, nT, nY