"""
This scripts contains supporting functions.
"""

import numpy as np
from qulacs import Observable
from qulacs.gate import *


def get_eigen_min(hamiltonian: Observable) -> float:
    """
    Finds the exact minimum eigen value for a given hamiltonian.

    Args:
        hamiltonian: `qulacs_core.Observable`

    Returns:
        min_eigenvalue: `float`, minimum eigen value
    """
    eigenvalues, _ = np.linalg.eigh(hamiltonian.get_matrix().toarray())
    min_eigenvalue = np.min(eigenvalues)
    return min_eigenvalue


def noise_level(nqubits: int, identity_factor: list[int]) -> tuple[int, int, int]:
    """
    Finds nR, nT, and nY for a given noise factor.

    Arg:

        nqubits: `int`, number of qubits.

        noise_factor: `list[int, int, int]`, represernts the redundant noisy indities for qubit gates and time evolution gate. For example, `nR = 1` adds one noisy identity (Rx_daggar*Rx) for Rx gate and one noisy identity (Ry_daggar * Ry) for Ry gate in the ciruit.

        layer: `int`, depth of the quantum circuit.

    Returns:

        nR, nT, nY: `int`, each value is proportional to the number of corresponding noisy gates in the circuit.
    """

    nY = 0
    nR = 4
    nT = 1
    nCz = 1

    r_gate_factor = identity_factor[0]  # For rotational gate
    u_gate_factor = identity_factor[1]  # For time evolution gate
    y_gate_factor = identity_factor[2]  # For Y gate
    cz_gate_factor = identity_factor[3]  # For CZ gate

    # Count the number of odd qubits
    odd_n = (nqubits // 2) + 1 if nqubits % 2 != 0 else nqubits // 2

    if r_gate_factor > 0:
        nR += 8 * r_gate_factor

    # If there is not U†U identity, then there is no Y gate
    if u_gate_factor > 0:
        # Count the number of odd qubits
        odd_n = (nqubits // 2) + 1 if nqubits % 2 != 0 else nqubits // 2
        # Calculating the values of different noise parameters
        nY += odd_n + (2 * y_gate_factor * odd_n)
        nT += 2 * u_gate_factor

    elif u_gate_factor == 0:
        nY = 0

    # If there is Cz†cz gate
    if cz_gate_factor > 0:
        nCz += 2 * cz_gate_factor

    return {"params": (nR, nT, nY, nCz), "odd_wires": odd_n}
