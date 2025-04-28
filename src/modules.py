import numpy as np
from qulacs import Observable

"""
This scripts contains supporting functions.
"""


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

def calculate_noise_levels(nqubits: int, identity_factors: list[int], noise_profile: dict) -> dict:
    """
    Finds nR, nT, and nY for a given noise factor.

    Arg:

        nqubits: `int`, number of qubits.

        noise_factor: `list[int, int, int]`, represernts the redundant noisy indities for
        qubit gates and time evolution gate.
        For example, `nR = 1` adds one noisy identity (Rx_daggar*Rx) for Rx gate
        and one noisy identity (Ry_daggar * Ry) for Ry gate in the ciruit.

        noise_prob: `list[float]`, represents the probability of noise for each gate. If the probability is 0, then there is no noise.

    Returns:

        `dict`, a dictionary containing the noisy gated related details.
    """

    nY = 0
    nR = 4
    nT = 1
    nCz = 1

    # Noisy identy factors: [R-gates, CZ-gate, U-gate, Y-gate]
    r_gate_factor = identity_factors[0]   # Identity sacaling factor for rotational gates
    cz_gate_factor = identity_factors[1]  # Identity scaling factor for CZ gate
    u_gate_factor = identity_factors[2]  # Identity scaling factor for time-evolution gates
    y_gate_factor = identity_factors[3]  # Identity scaling factor for Y gate

    ansatz_noise_status = noise_profile["status"]
    noise_prob = noise_profile["noise_prob"]

    r_gate_prob = noise_prob[0]  # For rotational gate
    cz_gate_prob = noise_prob[1]  # For cz gate
    u_gate_prob = noise_prob[2]  # For U gate
    y_gate_prob = noise_prob[3]  # For Y gate

    # Noise level initialization
    noise_rot = 0
    noise_CZ = 0
    noise_T = 0
    noise_y = 0

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

    if ansatz_noise_status:
        noise_rot = nR if r_gate_prob != 0 else 0
        noise_CZ = 2 * nCz if cz_gate_prob != 0 else 0
        noise_T = nqubits * nT if u_gate_prob != 0 else 0
        noise_y = nY if y_gate_prob != 0 else 0
            
    return {"identity_factors": identity_factors,
            "noise_level": [noise_rot, noise_CZ, noise_T, noise_y],
            "gates_num": [nR, nCz, nT, nY],
            "odd_wires": odd_n}