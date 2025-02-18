from typing import Dict, List

from src.hamiltonian import create_heisenberg_hamiltonian, create_ising_hamiltonian, create_xy_hamiltonian


def constructObservable(nqubits: int, definition: str, coefficient: Dict[str, float]) -> Dict[str, float]:
    """
    Constructs the target observable (Hamiltonian).

    definition: 'custom', 'ising', 'xy_model-xz-z', or 'heisenberg'.
    'custom' is created using definition: 'xy_model-xz-z' which is an XY-model Hamiltonian.

    ### WARNING! Coefficients are applicable only for def: 'custom', and are overwritten
    ### if def is specified as an in-built: 'ising', 'xy_model-xz-z', or 'heisenberg'.

    Args:
        nqubits (int): Number of qubits in the system.
        definition (str): A string defining the observable type.
        coefficient (dict): A dictionary with keys representing observable components
        and values as their respective coefficients.

    Returns:
        dict: A dictionary representing the constructed Hamiltonian.
    """

    # Validate the coefficient dictionary to ensure all required keys are present
    required_keys = ["cn", "bn", "r"]
    for key in required_keys:
        if key not in coefficient:
            raise ValueError(f"Missing required key '{key}' in coefficient dictionary.")

    observable_cn: List[float] = coefficient["cn"]
    observable_bn: List[float] = coefficient["bn"]
    observable_r: float = coefficient["r"]

    # Validate the user input
    observable_cn_len = len(observable_cn)
    observable_bn_len = len(observable_bn)

    target_observable = None

    if observable_cn_len != nqubits - 1 or observable_bn_len != nqubits:
        raise ValueError(
            f"Inconsistent lengths in observable Hamiltonian coefficients. "
            f"Expected lengths cn: {nqubits-1} and bn: {nqubits}, but got cn: {observable_cn_len} "
            f"and bn: {observable_bn_len}."
        )

    if definition.lower() == "custom":

        target_observable = create_xy_hamiltonian(nqubits=nqubits, cn=observable_cn, bn=observable_bn, r=observable_r)

    elif definition.lower() == "ising":

        target_observable = create_ising_hamiltonian(nqubits=nqubits)

    elif definition.lower() == "heisenberg":

        observable_cn = [1.0 for _ in range(nqubits - 1)]

        target_observable = create_heisenberg_hamiltonian(nqubits=nqubits, cn=observable_cn)

    else:
        raise ValueError(
            f"Invalid target definition: {definition}. "
            f"Valid definitions are: 'custom', 'ising', 'xy_model-xz-z', or 'heisenberg'."
        )

    return target_observable
