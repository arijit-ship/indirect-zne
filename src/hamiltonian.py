"""
General form of XY-model Hamiltonia.
"""

from typing import List
from openfermion.ops.operators.qubit_operator import QubitOperator
from qulacs import Observable
from qulacs.observable import create_observable_from_openfermion_text


def create_xy_hamiltonian(nqubits: int, cn: List[float], bn: List[float], r: float) -> Observable:
    """
    Creates a one-dimensional XY-Hamiltonian.

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
        # hami += (0.5*cn*(1-r)) * QubitOperator(f"Y{i} Y{i+1}")
        hami += (cn[i] * (1 - r)) * QubitOperator(f"Z{i} Z{i+1}")
    for j in range(nqubits):
        hami += bn[j] * QubitOperator(f"Y{j}")

    return create_observable_from_openfermion_text(str(hami))
