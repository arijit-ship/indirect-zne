from typing import List
from openfermion.ops.operators.qubit_operator import QubitOperator
from qulacs import Observable
from qulacs.observable import create_observable_from_openfermion_text


def create_ising_hamiltonian(nqubits: int, cn: List, bn: List) -> Observable:
    """ "
    Args:
        nqubits: int, number of qubits
        cn: `List`, 0.0 - 1.0, coupling constant
        bn: `List`, 0.0 - 1.0, magnetic field

    Returns:
        qulacs observable
    """
    hami = QubitOperator()
    for i in range(nqubits - 1):
        hami += cn[i] * QubitOperator(f"X{i} X{i+1}")
    for j in range(nqubits):
        hami += bn[i] * QubitOperator(f"Z{j}")

    return create_observable_from_openfermion_text(str(hami))
