"""
The Hamiltonian definations.
"""

from typing import List

from openfermion.ops.operators.qubit_operator import QubitOperator
from qulacs import Observable
from qulacs.observable import create_observable_from_openfermion_text


def create_xy_hamiltonian(nqubits: int, cn: List[float], bn: List[float], r: float) -> Observable:
    r"""
    Creates a one-dimensional XY-Hamiltonian.

    Note: For cn = 0.5, bn = 1, and r = 1 it reduces to transverse-field Ising Hamiltonian.

    Mathematical Form:

        .. math::
        H = \sum_{i=1}^{n-1} [c_i(1+r) X_i X_{i+1} + c_i(1-r) Z_i Z_{i+1}] + \sum_{j=1}^{n} b_j Z_j


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
    XY-Hamiltonian with coefficients cn = [0.5], bn = [1], and r = 1.

    Args:
        nqubits (int): The number of qubits.

    Returns:
        Observable: Qulacs observable representing the Hamiltonian.
    """
    cn_ising = [0.5 for _ in range(nqubits - 1)]    # cn = 0.5
    bn_ising = [1.0 for _ in range(nqubits)]    # bn = 1
    r_ising = 1 # r = 1
    ising_hamiltonian = create_xy_hamiltonian(nqubits=nqubits, cn=cn_ising, bn=bn_ising, r=r_ising)
    return ising_hamiltonian


def create_xy_iss_hamiltonian(nqubits: int) -> Observable:
    """
    Creates an xy-iss Hamiltonian which is a specific instance of
    XY-Hamiltonian with coefficients cn = [0.5], bn = [0], and r = 0.

    This model supports identity scaling in exp(-iHt) gate by adding Y-gate.

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
