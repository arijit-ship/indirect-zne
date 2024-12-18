import math

from src.hamiltonian import create_xy_hamiltonian
from src.vqe import IndirectVQE

nqubits = 4
layer = 10

state: str = "dmatrix"

cn1 = [0.5, 0.5, 0.5]
bn1 = [1.0, 1.0, 1.0, 1.0]
r1 = 1
target_observable = create_xy_hamiltonian(nqubits=nqubits, cn=cn1, bn=bn1, r=r1)

opt_dtls = {"status": False, "algorithm": "SLSQP", "constraint": False}

ansatz_dtls = {
    "type": "xy_model-xz-y",
    "layer": 10,
    "gateset": 1,
    "ugate": {"coefficients": {"cn": [0.5, 0.5, 0.5], "bn": [0, 0, 0, 0], "r": 0}, "time": {"min": 0.0, "max": 10.0}},
    "noise": {"status": True, "value": [0, 0, 0, 0]},
}

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
    -0.09602606843963848,
    5.337664263159555,
    3.439781498966039,
    4.47275071017785,
    3.136364192020315,
    5.4309009737504335,
    9.362514211689476,
    7.229609254161838,
    10.099956633185844,
    7.537403916293026,
    -0.11062180030359467,
    -0.11384127027395581,
    0.8534285942643706,
    0.11937465634956203,
    -0.18790436857461215,
    -1.25236278928115,
    0.20091144884375606,
    1.1339406209276381,
    0.3710553172113822,
    -0.6177010796762777,
    0.11961752483620948,
    1.3351780952365648,
    0.35200022512843165,
    -0.34028888603767604,
    0.8683461654681884,
    0.9420203656648578,
    -0.1963368119219526,
    -1.094341619660066,
    1.290118424840698,
    1.2924254365459362,
    -1.0429341215203949,
    -1.3757544695880564,
    0.017615587743360108,
    0.3845756484635315,
    0.12336346932440953,
    -1.257989603306994,
    0.27909929614930373,
    0.012174675527444959,
    -1.1450741016075339,
    -0.6093493281528398,
    0.142612826940021,
    -0.054231733977538096,
    -0.9206162503671923,
    -1.1590645288510215,
    -0.5834169471363333,
    -0.02336840191506826,
    0.15797133230346985,
    -0.08088194751133053,
    0.3538622283833272,
    0.30113857101948716,
]

expected_value = -4.758769842654501
tolerance = 1e-6

for factor in factors:
    vqe_instance = IndirectVQE(
        nqubits=nqubits,
        state=state,
        observable=target_observable,
        optimization=opt_dtls,
        ansatz=ansatz_dtls,
        identity_factor=factor,
        init_param=optimized_initial_param,
    )
    result = vqe_instance.run_vqe()
    estimation = result["initial_cost"]
    assert math.isclose(
        estimation, expected_value, rel_tol=tolerance, abs_tol=tolerance
    ), f"Result {estimation} is not close to {expected_value}"
