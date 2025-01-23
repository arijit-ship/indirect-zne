"""
The Hamiltonian definations.
"""

from typing import List

from openfermion.ops.operators.qubit_operator import QubitOperator
from qulacs import Observable
from qulacs.observable import create_observable_from_openfermion_text


def create_xy_hamiltonian(nqubits: int, cn: List[float], bn: List[float], r: float) -> Observable:
    """
    Creates a one-dimensional custom Hamiltonian (this is NOT any standard familier XY model)that has
    the following mathematical form:

    Mathematical form:

        .. math::
        H_\text{custom} = \sum_{k=1}^{N-1}c_{k}[(1+\gamma)X_{k}X_{k+1}+(1-\gamma)Z_{k}Z_{k+1}] + \sum_{k=1}^{N}b_{k}Z_{k}

    Note: For cn = 0.5, bn = 1, and r = 1 it reduces to transverse-field Ising Hamiltonian.

    Args:
        nqubits (int): The number of qubits.
        cn (List[float]): Coupling coefficients with values between 0.0 and 1.0.
        bn (List[float]): Magnetic fields with values between 0.0 and 1.0.
        r (float): Anisotropy parameter with values between 0.0 and 1.0.

    Returns:
        Observable: Qulacs observable representing the Hamiltonian.
    """
    hami = QubitOperator()

    for i in range(nqubits - 1):
        hami += (cn[i] * (1 + r)) * QubitOperator(f"X{i} X{i+1}")
        hami += (cn[i] * (1 - r)) * QubitOperator(f"Z{i} Z{i+1}")

    for i in range(nqubits):
        hami += bn[i] * QubitOperator(f"Z{i}")

    return create_observable_from_openfermion_text(str(hami))


def create_ising_hamiltonian(nqubits: int) -> Observable:
    """
    Creates an Ising Hamiltonian which is a specific instance of
    custom Hamiltonian (this is NOT any standard familier XY model) with coefficients cn = [0.5], bn = [1], and r = 1.

    Mathematical from:

        ..math::
        H_\text{Ising} = \sum_{k=1}^{N-1}X_k X_{k+1} + \sum_{k=0}^{N} Z_k

    Args:
        nqubits (int): The number of qubits.

    Returns:
        Observable: Qulacs observable representing the Hamiltonian.
    """
    cn_ising = [0.5 for _ in range(nqubits - 1)]  # cn = 0.5
    bn_ising = [1.0 for _ in range(nqubits)]  # bn = 1
    r_ising = 1  # r = 1
    ising_hamiltonian = create_xy_hamiltonian(nqubits=nqubits, cn=cn_ising, bn=bn_ising, r=r_ising)
    return ising_hamiltonian


def create_xy_iss_hamiltonian(nqubits: int) -> Observable:
    """
    Creates an xy-iss Hamiltonian which is a specific instance of
    custom Hamiltonian with coefficients cn = [0.5], bn = [0], and r = 0.

    This model supports identity scaling in exp(-iHt) gate by adding Y-gate.

    Mathematical from:

        ..math::
        H_{XY}=\frac{1}{2}\sum_{k=1}^{N-1} X_{k}X_{k+1} + Z_{k}Z_{k+1}

    Args:
        nqubits (int): The number of qubits.

    Returns:
        Observable: Qulacs observable representing the Hamiltonian.
    """
    cn_xy_iss = [0.5 for _ in range(nqubits - 1)]
    bn_xy_iss = [0 for _ in range(nqubits)]
    r_xy_iss = 0
    xy_iss_hamiltonian = create_xy_hamiltonian(nqubits=nqubits, cn=cn_xy_iss, bn=bn_xy_iss, r=r_xy_iss)
    return xy_iss_hamiltonian


def create_heisenberg_hamiltonian(nqubits: int, cn: List[float]) -> Observable:
    """
    Creates a one-dimensional Heisenberg-Hamiltonian.

    Mathematical Form:

        .. math::
        H_\text{Heisenberg} = \frac{1}{2}\sum_{k=1}^{N-1} X_k X_{k+1} + Y_k Y_{k+1} + Z_k Z_{k+1} .


    Args:
        nqubits (int): The number of qubits.
        cn (List[float]): Coupling coefficients with values between 0.0 and 1.0.
        bn (List[float]): Magnetic fields with values between 0.0 and 1.0.
        r (float): Anisotropy parameter with values between 0.0 and 1.0.

    Returns:
        Observable: Qulacs observable representing the Hamiltonian.
    """
    hami = QubitOperator()

    for i in range(nqubits - 1):
        hami += cn[i] * QubitOperator(f"X{i} X{i+1}")
        hami += cn[i] * QubitOperator(f"Y{i} Y{i+1}")
        hami += cn[i] * QubitOperator(f"Z{i} Z{i+1}")

    return create_observable_from_openfermion_text(str(hami))
