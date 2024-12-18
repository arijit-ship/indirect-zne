from typing import List

from qulacs import Observable, QuantumCircuit
from qulacs.gate import CZ, RX, RY, RZ, Identity, Y, merge

from .time_evolution_gate import create_time_evo_unitary


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
            circuit.add_gate(
                merge(
                    RY(i, theta_list[2 * i + 2 * n_qubit * d]),
                    RZ(i, theta_list[2 * i + 1 + 2 * n_qubit * d]),
                )
            )
        for i in range(n_qubit // 2):
            circuit.add_gate(CZ(2 * i, 2 * i + 1))
        for i in range(n_qubit // 2 - 1):
            circuit.add_gate(CZ(2 * i + 1, 2 * i + 2))
    for i in range(n_qubit):
        circuit.add_gate(
            merge(
                RY(i, theta_list[2 * i + 2 * n_qubit * depth]),
                RZ(i, theta_list[2 * i + 1 + 2 * n_qubit * depth]),
            )
        )

    return circuit


def noiseless_ansatz(nqubits: int, layers: int, gateset: int, ugateH: Observable, param: list[float]) -> QuantumCircuit:
    """
    Args:
        nqubits (int): Number of qubits.
        layer (int): Depth of the ansatz circuit.
        ugateH (Observable): `qulacs_core.Observable`, Hamiltonian used in time evolution gate i.e. exp(-iHt)
        param (ndarray): Initial params for rotation gates, time evolution gate: [
        t1, t2, ... td, theta1, ..., theatd * 4]

    Returns:
        QuantumCircuit
    """
    circuit = QuantumCircuit(nqubits)

    flag = layers  # Tracking angles in param ndarray

    for layer in range(layers):

        for i in range(gateset):
            # Rotation gate
            circuit.add_gate(RX(0, param[flag + i]))
            circuit.add_gate(RX(1, param[flag + i + 1]))

            circuit.add_gate(RY(0, param[flag + i + 2]))
            circuit.add_gate(RY(1, param[flag + i + 3]))

        # CZ gate
        circuit.add_gate(CZ(0, 1))

        if layer == 0:
            # Time evolution gate
            ti = 0
            tf = param[layer]
            time_evo_gate = create_time_evo_unitary(ugateH, ti, tf)
            circuit.add_gate(time_evo_gate)
        else:
            ti = param[layer]
            tf = param[layer + 1]
            time_evo_gate = create_time_evo_unitary(ugateH, ti, tf)
            circuit.add_gate(time_evo_gate)

        flag += 4 * gateset  # Each layer has 4 * gateset angle-params.

    return circuit


def create_noisy_ansatz(
    nqubits: int,
    layer: int,
    gateset: int,
    ugateH: Observable,
    noise_prob: List[float],
    noise_factor: List[int],
    param: List[float],
) -> QuantumCircuit:
    """
    Creates noisy redundant ansatz.

    Args:
        nqubits (int): Number of qubit.
        layer (int): Depth of the quantum circuit.
        gateset (int): Number of rotatation gate set. Each set contains fours gates which are Rx1, Ry1, Rx2, Ry2.
        ugateH (Onservable): Hamiltonian used in time evolution gate i.e. exp(-iHt).
        noise_prob (List[float]): Probability of applying depolarizing noise. Value is between 0-1.
        noise_factor (List[]), noise factor for rotational gates, time evolution unitary gate and Y gate.
        Based on this redundant noisy identites are constructed.
        For example, if value is 1, only one set of identities are introduced.
        param (ndarray): Initial params for rotation gates, time evolution gate: [
        t1, t2, ... td, theta1, ..., theatd * 4]

    Returns:
        QuantumCircuit
    """

    # Creates redundant circuit
    circuit = create_redundant(nqubits, layer, noise_prob, noise_factor, ugateH, param)

    return circuit


def create_redundant(
    nqubits: int,
    layers: int,
    noise_prob: List[float],
    noise_factor: List[int],
    hamiltonian,
    param: List[float],
) -> QuantumCircuit:
    """
    Creates a noisy circuit with redundant noisy indentities based on a given noise factor.

    Args:
        nqubits (int): Number of qubit.
        layer (int): Depth of the quantum circuit.
        gateset (int): Number of rotatation gate set. Each set contains fours gates which are Rx1, Ry1, Rx2, Ry2.
        ugateH (Onservable): Hamiltonian used in time evolution gate i.e. exp(-iHt).
        noise_prob (List[float]): Probability of applying depolarizing noise. Value is between 0-1.
        noise_factor (List[]): Noise factor/identity for rotational gates, time evolution unitary gate and Y gate.
        Based on this redundant noisy identites are constructed.
        For example, if value is [1, 1, 1] only one set of
        identities are introduced for rotational, unitary and Y gates.
        param (ndarray): Initial params for rotation gates, time evolution gate: [
        t1, t2, ... td, theta1, ..., theatd * 4]

    Returns:
        QuantumCircuit
    """

    circuit = QuantumCircuit(nqubits)

    flag = layers  # Tracking angles in param ndarrsy

    # Noise propabilities
    noise_r_prob = noise_prob[0]
    noise_cz_prob = noise_prob[1]
    noise_u_prob = noise_prob[2]
    noise_y_prob = noise_prob[3]

    # Noisy identy factors
    r_gate_factor = noise_factor[0]  # Identity sacaling factor for rotational gates
    u_gate_factor = noise_factor[1]  # Identity scaling factor for time-evolution gates
    y_gate_factor = noise_factor[2]  # Identity scaling factor for Y gate
    cz_gate_factor = noise_factor[3]

    for layer in range(layers):

        # Add Rx to first and second qubits
        circuit.add_noise_gate(RX(0, param[flag]), "Depolarizing", noise_r_prob)
        circuit.add_noise_gate(RX(1, param[flag + 1]), "Depolarizing", noise_r_prob)

        # Add identities with Rx and make redudant circuit

        for _ in range(r_gate_factor):

            # First qubit
            circuit.add_noise_gate(RX(0, param[flag]).get_inverse(), "Depolarizing", noise_r_prob)  # Rx_dagger
            circuit.add_noise_gate(RX(0, param[flag]), "Depolarizing", noise_r_prob)

            # Second qubit
            circuit.add_noise_gate(RX(1, param[flag + 1]).get_inverse(), "Depolarizing", noise_r_prob)  # Rx_dagger
            circuit.add_noise_gate(RX(1, param[flag + 1]), "Depolarizing", noise_r_prob)

        # Add Ry to first and second qubits
        circuit.add_noise_gate(RY(0, param[flag + 2]), "Depolarizing", noise_r_prob)
        circuit.add_noise_gate(RY(1, param[flag + 3]), "Depolarizing", noise_r_prob)

        # Add identities with Ry and make redudant circuit
        for _ in range(r_gate_factor):
            # First qubit
            circuit.add_noise_gate(RY(0, param[flag + 2]).get_inverse(), "Depolarizing", noise_r_prob)  # Ry_dagger
            circuit.add_noise_gate(RY(0, param[flag + 2]), "Depolarizing", noise_r_prob)

            # Second qubit
            circuit.add_noise_gate(RY(1, param[flag + 3]).get_inverse(), "Depolarizing", noise_r_prob)  # Ry_dagger
            circuit.add_noise_gate(RY(1, param[flag + 3]), "Depolarizing", noise_r_prob)

        # Add CZ gate
        circuit.add_noise_gate(CZ(0, 1), "Depolarizing", noise_cz_prob)

        # Add identites with CZ gates
        for _ in range(cz_gate_factor):
            circuit.add_noise_gate(CZ(0, 1).get_inverse(), "Depolarizing", noise_cz_prob)
            circuit.add_noise_gate(CZ(0, 1), "Depolarizing", noise_cz_prob)

        # Add multi-qubit U gate
        if layer == 0:
            ti = 0
            tf = param[layer]
            time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)

        else:
            ti = param[layer]
            tf = param[layer + 1]
            time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
            circuit.add_gate(time_evo_gate)

        # Add depolarizing noise to time evolution gate.
        for i in range(nqubits):
            circuit.add_noise_gate(Identity(i), "Depolarizing", noise_u_prob)

        # XY spin chain identity

        for _ in range(u_gate_factor):
            # Add Y gates to all odd qubits
            circuit = add_ygate_odd(circuit, noise_y_prob, y_gate_factor)

            # Again add multi-qubit U gate
            if layer == 0:
                ti = 0
                tf = param[layer]
                time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
                circuit.add_gate(time_evo_gate)

            else:
                ti = param[layer]
                tf = param[layer + 1]
                time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
                circuit.add_gate(time_evo_gate)

            # Add depolarizing noise
            for i in range(nqubits):
                circuit.add_noise_gate(Identity(i), "Depolarizing", noise_u_prob)

            # Add Y gates to all odd qubits
            circuit = add_ygate_odd(circuit, noise_y_prob, y_gate_factor)

            # Again add multi-qubit U gate
            if layer == 0:
                ti = 0
                tf = param[layer]
                time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
                circuit.add_gate(time_evo_gate)

            else:
                ti = param[layer]
                tf = param[layer + 1]
                time_evo_gate = create_time_evo_unitary(hamiltonian, ti, tf)
                circuit.add_gate(time_evo_gate)

            # Add depolarizing noise
            for i in range(nqubits):
                circuit.add_noise_gate(Identity(i), "Depolarizing", noise_u_prob)

        flag += 4  # Each layer has four angle-params

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
        if (i + 1) % 2 != 0:
            circuit.add_noise_gate(Y(i), "Depolarizing", noise_y_prob)

            # Add redundant Y gate identities
            for _ in range(y_gate_factor):
                circuit.add_noise_gate(Y(i).get_inverse(), "Depolarizing", noise_y_prob)
                circuit.add_noise_gate(Y(i), "Depolarizing", noise_y_prob)
    return circuit
