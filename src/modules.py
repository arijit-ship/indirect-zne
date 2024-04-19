"""
This scripts contains all the necessary functions for VQE and ZNE.
"""
import numpy as np
from scipy.linalg import expm
from qulacs.gate import *
from qulacs.observable import create_observable_from_openfermion_text
from qulacs import QuantumCircuit 
from openfermion.ops.operators.qubit_operator import QubitOperator


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
    time = np.random.uniform(ti,tf, layer)
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
        hami += (0.5*cn*(1-r)) * QubitOperator(f"Y{i} Y{i+1}")
    for j in range(nqubit):
        hami += bn*QubitOperator(f"Z{j}")

    return (create_observable_from_openfermion_text(str(hami)))


def create_ising_hamiltonian(nqubit, cn, bn):
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

def exact_sol(hamiltonian):
    """
    Finds the exact minimum eigen value for a given matrix.

    Args:
        hamiltonian: `qulacs_core.Observable`
    
    Returns:
        `float`, minimum eigen value
    """
    eigenvalues, _ = np.linalg.eigh(hamiltonian.get_matrix().toarray())
    min_eigenvalue = np.min(eigenvalues)
    return min_eigenvalue
    

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
    # H_mat = observable.get_matrix()

    # # Converting to array
    # H_mat_array = H_mat.toarray()

    # # this is exp(-iHt)
    # exponent = expm(-1*1j*H_mat_array*(tf-ti))
    # return (DenseMatrix([i for i in range(n)], exponent))
    
# Global variables to store the eigenvalues and eigenvectors
diag = None
eigen_vecs = None
    
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


def parametric_ansatz(nqubit, layer, hamitolian, param):
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
   

def create_noisy_ansatz2(nqubit, layer, noise_prob, hamiltonian, param):
    
    """
    Indirect controlled noisy ansatz.

    Args:
        nqubit: `int`, number of qubit
        layer: `int`, number of layer
        noise_prob: `float`, noise probability between 0-1
        hamiltonian: `qulacs_core.Observable`, hamiltonian used in time evolution gate i.e. exp(-iHt)
        param: class:`numpy.ndarray`, params for rotation gates, time evolution gate: [
        t1, t2, ... td, theta1, ..., theatd * 4]

    Returns:
        :class:`qulacs.QuantumCircuit`
    """

    circuit = QuantumCircuit(nqubit)
    
    flag = layer # Tracking angles in param ndarrsy
    
    for i in range(layer):

        # Add Rx to first and second qubits
        circuit.add_noise_gate(RX(0, param[flag]),"Depolarizing", noise_prob)
        circuit.add_noise_gate(RX(1, param[flag+1]),"Depolarizing", noise_prob)

        # Add identities with Rx and make redudant circuit

        # First qubit
        circuit.add_noise_gate(RX(0, param[flag]).get_inverse(), "Depolarizing", noise_prob)  # Rx_dagger
        circuit.add_noise_gate(RX(0, param[flag]), "Depolarizing", noise_prob)
    
        # Second qubit
        circuit.add_noise_gate(RX(1, param[flag+1]).get_inverse(), "Depolarizing", noise_prob)   # Rx_dagger
        circuit.add_noise_gate(RX(1, param[flag+1]), "Depolarizing", noise_prob)

        # Add Ry to first and second qubits
        circuit.add_noise_gate(RY(0, param[flag+2]),"Depolarizing", noise_prob)
        circuit.add_noise_gate(RY(1, param[flag+3]),"Depolarizing", noise_prob)

        # Add identities with Ry and make redudant circuit
        # First qubit
        circuit.add_noise_gate(RY(0, param[flag+2]).get_inverse(), "Depolarizing", noise_prob)  # Ry_dagger
        circuit.add_noise_gate(RY(0, param[flag+2]), "Depolarizing", noise_prob)
    
        # Second qubit
        circuit.add_noise_gate(RY(1, param[flag+3]).get_inverse(), "Depolarizing", noise_prob)   # Ry_dagger
        circuit.add_noise_gate(RY(1, param[flag+3]), "Depolarizing", noise_prob)

        # Add CZ gate
        circuit.add_noise_gate(CZ(0,1), "Depolarizing", noise_prob)

        # Add multi-qubit U gate
        if i == 0:
            ti = 0
            tf = param[i]
            time_evo_gate =  create_time_evo_unitary(hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)

        else:
            ti = param[i]
            tf = param[i+1]
            time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)

        # Add depolarizing noise to time evolution gate.
        for i in range(nqubit):
            circuit.add_noise_gate(Identity(i), "Depolarizing", noise_prob)

        # Add Y gates to all odd qubits
        for i in range(nqubit):
            if (i+1) % 2 != 0:
                circuit.add_noise_gate(Y(i), "Depolarizing", noise_prob)
        

        # Again add multi-qubit U gate
        if i == 0:
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
            circuit.add_noise_gate(Identity(i), "Depolarizing", noise_prob)

        # Add Y gates to all odd qubits
        for i in range(nqubit):
            if (i+1) % 2 != 0:
                circuit.add_noise_gate(Y(i), "Depolarizing", noise_prob)

        # Again add multi-qubit U gate
        if i == 0:
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
            circuit.add_noise_gate(Identity(i), "Depolarizing", noise_prob)

        
        flag += 4 # Each layer has four angle-params
        
    return circuit



def create_noisy_ansatz(nqubit, layer, noise_prob, noise_factor, hamiltonian, param):
    
    """
    Creates noisy redundant ansatz.

    Args:
        nqubit: `int`, number of qubit
        layer: `int`, number of layer
        noise_prob: `float`, noise probability between 0-1
        hamiltonian: `qulacs_core.Observable`, hamiltonian used in time evolution gate i.e. exp(-iHt)
        noise_factor: `int`, noise factor for rotational gates, time evolution unitary gate and Y gate.. Based on this redundant noisy identites are constructed. For example, if value is 1, only one set of identities are introduced.
        param: class:`numpy.ndarray`, params for rotation gates, time evolution gate: [
        t1, t2, ... td, theta1, ..., theatd * 4]

    Returns:
        :class:`qulacs.QuantumCircuit`
    """

    circuit = QuantumCircuit(nqubit)

    
    if noise_factor == 0:
        # Creates a noisy standard parametric circuit.
        circuit = create_default_noisy_circuit(nqubit, layer, noise_prob, hamiltonian, param)

    else:
        # Creates redundant circuit
        circuit = create_redundant(nqubit, layer, noise_prob, noise_factor, hamiltonian, param)
    
    return circuit

def create_default_noisy_circuit(nqubit, layer, noise_prob, hamiltonian, param):
    """
    Creates a noisy standard parametric circuit without any redundant identities. This circuit is mainly used to find the optimized parameters for noisy circuit.

    Args:
        nqubit: `int`, number of qubit
        layer: `int`, number of layer
        noise_prob: `float`, noise probability between 0-1
        hamiltonian: `qulacs_core.Observable`, hamiltonian used in time evolution gate i.e. exp(-iHt)
        param: class:`numpy.ndarray`, params for rotation gates, time evolution gate: [
        t1, t2, ... td, theta1, ..., theatd * 4]

    Returns:
        :class:`qulacs.QuantumCircuit`
    """
    circuit = QuantumCircuit(nqubit)

    flag = layer # Tracking angles in param ndarrsy
    
    for i in range(layer):

        # Add Rx to first and second qubits
        circuit.add_noise_gate(RX(0, param[flag]),"Depolarizing", noise_prob)
        circuit.add_noise_gate(RX(1, param[flag+1]),"Depolarizing", noise_prob)

        
        # Add Ry to first and second qubits
        circuit.add_noise_gate(RY(0, param[flag+2]),"Depolarizing", noise_prob)
        circuit.add_noise_gate(RY(1, param[flag+3]),"Depolarizing", noise_prob)

        # Add CZ gate
        circuit.add_noise_gate(CZ(0,1), "Depolarizing", noise_prob)

        # Again add multi-qubit U gate
        if i == 0:
            ti = 0
            tf = param[i]
            time_evo_gate =  create_time_evo_unitary(hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)

        else:
            ti = param[i]
            tf = param[i+1]
            time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)

        flag += 4

    return circuit


def create_redundant(nqubit, layer, noise_prob, noise_factor, hamiltonian, param):
    """
    Creates a noisy circuit with redundant noisy indentities based on a given noise factor.

    Args:
        nqubit: `int`, number of qubit
        layer: `int`, number of layer
        noise_prob: `float`, noise probability between 0-1
        hamiltonian: `qulacs_core.Observable`, hamiltonian used in time evolution gate i.e. exp(-iHt)
        param: class:`numpy.ndarray`, params for rotation gates, time evolution gate: [
        t1, t2, ... td, theta1, ..., theatd * 4]

    Returns:
        :class:`qulacs.QuantumCircuit`
    """
    

    circuit = QuantumCircuit(nqubit)

    flag = layer # Tracking angles in param ndarrsy
    
    for i in range(layer):

        # Add Rx to first and second qubits
        circuit.add_noise_gate(RX(0, param[flag]),"Depolarizing", noise_prob)
        circuit.add_noise_gate(RX(1, param[flag+1]),"Depolarizing", noise_prob)

        # Add identities with Rx and make redudant circuit

        for _ in range(noise_factor):

            # First qubit
            circuit.add_noise_gate(RX(0, param[flag]).get_inverse(), "Depolarizing", noise_prob)  # Rx_dagger
            circuit.add_noise_gate(RX(0, param[flag]), "Depolarizing", noise_prob)
    
            # Second qubit
            circuit.add_noise_gate(RX(1, param[flag+1]).get_inverse(), "Depolarizing", noise_prob)   # Rx_dagger
            circuit.add_noise_gate(RX(1, param[flag+1]), "Depolarizing", noise_prob)

        # Add Ry to first and second qubits
        circuit.add_noise_gate(RY(0, param[flag+2]),"Depolarizing", noise_prob)
        circuit.add_noise_gate(RY(1, param[flag+3]),"Depolarizing", noise_prob)

        # Add identities with Ry and make redudant circuit

        for _ in range(noise_factor):
            # First qubit
            circuit.add_noise_gate(RY(0, param[flag+2]).get_inverse(), "Depolarizing", noise_prob)  # Ry_dagger
            circuit.add_noise_gate(RY(0, param[flag+2]), "Depolarizing", noise_prob)
        
            # Second qubit
            circuit.add_noise_gate(RY(1, param[flag+3]).get_inverse(), "Depolarizing", noise_prob)   # Ry_dagger
            circuit.add_noise_gate(RY(1, param[flag+3]), "Depolarizing", noise_prob)

        # Add CZ gate
        circuit.add_noise_gate(CZ(0,1), "Depolarizing", noise_prob)

        # Add multi-qubit U gate
        if i == 0:
            ti = 0
            tf = param[i]
            time_evo_gate =  create_time_evo_unitary(hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)

        else:
            ti = param[i]
            tf = param[i+1]
            time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)

        # Add depolarizing noise to time evolution gate.
        for i in range(nqubit):
            circuit.add_noise_gate(Identity(i), "Depolarizing", noise_prob)

        # XY spin chain identity 

        for _ in range(noise_factor):
            # Add Y gates to all odd qubits
            for i in range(nqubit):
                if (i+1) % 2 != 0:
                    circuit.add_noise_gate(Y(i), "Depolarizing", noise_prob)
            

            # Again add multi-qubit U gate
            if i == 0:
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
                circuit.add_noise_gate(Identity(i), "Depolarizing", noise_prob)

            # Add Y gates to all odd qubits
            for i in range(nqubit):
                if (i+1) % 2 != 0:
                    circuit.add_noise_gate(Y(i), "Depolarizing", noise_prob)

            # Again add multi-qubit U gate
            if i == 0:
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
                circuit.add_noise_gate(Identity(i), "Depolarizing", noise_prob)

        
        flag += 4 # Each layer has four angle-params

    return circuit
