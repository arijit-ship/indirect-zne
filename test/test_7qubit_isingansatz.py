import json

import pytest

from src.hamiltonian import create_ising_hamiltonian
from src.vqe import IndirectVQE

# Load 8-qubit results from the JSON from file
with open("test/7qubit_isingansatz_result.json", "r") as file:
    ref_result = json.load(file)

ref_optimized_costs: list[float] = ref_result["output"]["optimized_minimum_cost"]
ref_optimized_param: list[float] = ref_result["output"]["optimized_parameters"]


@pytest.fixture(autouse=True)
def reset_global_diag_cache():
    r"""
    To run tests with different qubit counts in a single test session, global variables must be reset between tests.
    """
    import src.time_evolution_gate  # wherever diag, eigen_vecs are defined

    src.time_evolution_gate.diag = None
    src.time_evolution_gate.eigen_vecs = None


nqubits: int = 7
layer: int = 30

state: str = "dmatrix"

target_observable = create_ising_hamiltonian(nqubits=nqubits)

opt_dtls: dict = {"iteration": 1, "optimization": {"status": False, "algorithm": "SLSQP", "constraint": False}}


ansatz_dtls: dict = {
    "layer": layer,
    "gateset": 1,
    "ugate": {
        "type": "ising",
        "coefficients": {"cn": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "bn": [0, 0, 0, 0, 0, 0, 0], "r": 0},
        "time": {"min": 0.0, "max": 10.0},
    },
}

noise_dtls: dict = {
    "status": True,
    "type": "Depolarizing",
    "noise_prob": [0.001, 0.001, 0, 0],
    "noise_on_init_param": {"status": False, "value": 0.1},
}


def test1():
    index: int = 0
    for init_param in ref_optimized_param:
        vqe_instance = IndirectVQE(
            nqubits=nqubits,
            state=state,
            observable=target_observable,
            vqe_profile=opt_dtls,
            ansatz_profile=ansatz_dtls,
            noise_profile=noise_dtls,
            identity_factors=[0, 0, 0, 0],
            init_param=init_param,
        )
        result = vqe_instance.run_vqe()["initial_cost"]
        assert result == pytest.approx(ref_optimized_costs[index], rel=1e-9, abs=1e-12)
        index += 1
