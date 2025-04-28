import math

from src.hamiltonian import create_ising_hamiltonian
from src.vqe import IndirectVQE

nqubits = 4
layer = 10

state: str = "dmatrix"

# cn1 = [0.5, 0.5, 0.5]
# bn1 = [1.0, 1.0, 1.0, 1.0]
# r1 = 1
target_observable = create_ising_hamiltonian(nqubits=nqubits)

opt_dtls = {
    "iteration": 1,
    "optimization": {
        "status": False,
        "algorithm": "SLSQP",
        "constraint": False
    }
}


ansatz_dtls = {
    "layer": 10,
    "gateset": 1,
    "ugate": {"type": "xy-iss","coefficients": {"cn": [0.5, 0.5, 0.5], "bn": [0, 0, 0, 0], "r": 0}, "time": {"min": 0.0, "max": 10.0}},
}

noise_dtls = {"status": True, "type": "Depolarizing", "noise_prob": [0, 0, 0, 0], "noise_on_init_param": {"status": False, "value": 0.1}}

factors = [
    [0, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 1, 0, 2],
    [1, 1, 1, 3],
    [2, 0, 0, 1],
    [2, 1, 0, 2],
    [2, 1, 1, 1],
    [10, 12, 4, 12],
]

optimized_initial_param = [
    1.5591511877608648,
    2.2334738790806137,
    4.459265413099849,
    4.645646545091418,
    3.405301665951464,
    6.490940184541567,
    6.428435946310779,
    7.114427509946891,
    8.275366768201526,
    9.619467193648417,
    0.6193939275767291,
    -0.46098062621173586,
    0.708028921421339,
    0.005053928758499478,
    0.35995317630140194,
    0.04982106500440027,
    0.8942468046711314,
    0.3265444795696705,
    -0.07857536388522297,
    0.1763751770880597,
    -0.26732824299263386,
    0.2650551650950129,
    -0.5631884629176958,
    0.010218245342612267,
    0.2324631100149391,
    0.014628421317410571,
    0.5550137708703846,
    0.4004351320958063,
    0.41218662403200085,
    -0.08903408463869904,
    -0.37303332181205046,
    -0.3021201038763116,
    0.08459634859071612,
    0.16310264052806578,
    -0.23378403833917324,
    0.7039532063008676,
    -0.4735082635738917,
    1.1041293093045976,
    -0.3492294822344617,
    -0.0991799505295182,
    1.2088996216475842,
    0.8417726679482366,
    -0.21555953648927495,
    0.15911799163457183,
    -0.1634050431323822,
    -0.022839255036782833,
    -0.9570414792626977,
    0.06400285585702489,
    -0.10514778284552612,
    0.1303439397890625,
]

expected_value = -4.758769842654501
tolerance = 1e-7

for factor in factors:
    # vqe_instance = IndirectVQE(
    #     nqubits=nqubits,
    #     state=state,
    #     observable=target_observable,
    #     optimization=opt_dtls,
    #     ansatz=ansatz_dtls,
    #     identity_factor=factor,
    #     init_param=optimized_initial_param,
    # )
    vqe_instance = IndirectVQE(
            nqubits=nqubits,
            state=state,
            observable=target_observable,
            vqe_profile= opt_dtls,
            ansatz_profile= ansatz_dtls,
            noise_profile= noise_dtls,
            identity_factors= factor,
            init_param=optimized_initial_param,
        )
    result = vqe_instance.run_vqe()
    estimation = result["initial_cost"]
    assert math.isclose(
        estimation, expected_value, rel_tol=tolerance, abs_tol=tolerance
    ), f"Result {estimation} is not close to {expected_value}"
