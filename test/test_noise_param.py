from src.modules import calculate_noise_levels
import numpy as np

# nY = existing odd no + (2 * noise factor * existing odd no)


def test_noise_param1():
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[0, 0, 0, 0],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [0, 0, 0, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (4, 1, 1, 0)
    assert odd_n == 3
    assert noise_level == [0, 0, 0, 0]  # Noise probabilitieds are all zeros.


def test_noise_param2():
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[0, 0, 0, 0],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [1, 0, 0, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (4, 1, 1, 0)
    assert odd_n == 3
    assert noise_level == [4, 0, 0, 0]


def test_noise_param3():
    """
    CZ gate some noise (probability > 0), not no unitary folding. So noise level should be 2 * nCz = 2 * 1 = 2.
    """
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[0, 0, 0, 0],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [1, 1, 0, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (4, 1, 1, 0)
    assert odd_n == 3
    assert noise_level == [4, 2, 0, 0]


def test_noise_param4():
    """
    CZ gate some noise (probability > 0), with 1 unitary folding. So noise level should be 2 * nCz = 2 * 3 = 6.
    """
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[0, 1, 0, 0],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [1, 1, 0, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (4, 3, 1, 0)
    assert odd_n == 3
    assert noise_level == [4, 6, 0, 0]


def test_noise_param5():
    """
    CZ has 1 unitary folding, but noise probability is zero. So noise level should be 0, while nCz = 3.
    """
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[0, 1, 0, 0],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [1, 0, 0, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (4, 3, 1, 0)
    assert odd_n == 3
    assert noise_level == [4, 0, 0, 0]


def test_noise_param6():
    """
    U gate has some noise (probability > 0), but with zero unitary folding.
    So noise level should be nqubits * nT = 5 * 1 = 5.
    """
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[0, 0, 0, 0],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [0, 0, 1, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (4, 1, 1, 0)
    assert odd_n == 3
    assert noise_level == [0, 0, 5, 0]


def test_noise_param7():
    """
    U gate has some noise (probability > 0), with 1 unitary folding.
    So noise level should be nqubits * (nT) = 5 * (1 + 1 + 1) = 15.
    """
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[0, 0, 1, 0],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [0, 0, 1, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (4, 1, 3, 6)
    assert odd_n == 3
    assert noise_level == [0, 0, 15, 0]


def test_noise_param8():
    """
    U gate has some noise (probability > 0), with 1 unitary folding.
    Also the Y gate also has some finite noise (probability > 0) with 1 unitary folding.
    So noise levels are: noise_Ugate = nqubits * (nT) = 5 * (1 + 1 + 1) = 15 and
    noise_Ygate = nY = odd_wire * [( 1 Y gate + 2 * Y gate factor) * 2] = 3 * [(1 + 2*1)*2] = 3 * 6 = 18.
    """
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[0, 0, 1, 1],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [0, 0, 1, 0.1],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (4, 1, 3, 18)
    assert odd_n == 3
    assert noise_level == [0, 0, 15, 18]


def test_noise_param9():
    """
    U gate has some noise (probability > 0), with 1 unitary folding.
    Also the Y gate also has some finite noise (probability > 0) with 1 unitary folding.
    So noise levels are: noise_Ugate = nqubits * (nT) = 5 * (1 + 1 + 1) = 15 and
    noise_Ygate = nY = odd_wire * [( 1 Y gate + 2 * Y gate factor) * 2] = 3 * [(1 + 2*1)*2] = 3 * 6 = 18.
    """

    # A random noise probability for the test.
    noise_prob: float = np.random.uniform(0.0, 1.0)

    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[1, 1, 1, 0],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [noise_prob] * 4,
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert odd_n == 3
    if noise_prob != 0:
        assert (nR, nCz, nT, nY) == (12, 3, 3, 6)
        assert noise_level == [12, 6, 15, 6]
    else:
        assert (nR, nCz, nT, nY) == (12, 3, 3, 6)
        assert noise_level == [0, 0, 0, 0]


def test_noise_param10():
    """
    Although noise status is True, but all noise probabilities are zero. So the noise level should be zero.
    """
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[0, 0, 0, 0],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [0, 0, 0, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (4, 1, 1, 0)
    assert odd_n == 3
    assert noise_level == [0, 0, 0, 0]


def test_noise_param11():
    """
    Although noise status is True, but all noise probabilities are zero.
    But the number of gates will be determined based off the unitary foldings.
    """
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[1, 1, 1, 1],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [0, 0, 0, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (12, 3, 3, 18)
    assert odd_n == 3
    assert noise_level == [0, 0, 0, 0]


def test_noise_param12():
    """
    Although indentity factors (for unitary folding) has non-zero integer values, noise status is False.
    """
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[1, 1, 1, 1],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [0, 0, 0, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (12, 3, 3, 18)
    assert odd_n == 3
    assert noise_level == [0, 0, 0, 0]


def test_noise_param13():
    """
    Identity factor for U gate is zero, but for Y gate is 1; so no folding for Y gate.
    """
    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[1, 1, 0, 1],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [0, 0, 0, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (12, 3, 1, 0)
    assert odd_n == 3
    assert noise_level == [0, 0, 0, 0]


def test_noise_param14():
    r"""
    There are multiple unitary foldings for both U gates and Y gates.

    Let's say the U gate has 5 foldings. That means there should be a total of 11 U gates arranged as follows:

        --U--U†U--U†U--U†U--U†U--U†U--

    Now, suppose each Y gate has 2 foldings. In this setup, each U† gate consists of 10 Y gates,
    and with 3 odd-numbered wires, each U† across the odd wires has the following configuration:

        --Y--Y†Y--Y†Y--U--Y†Y--Y†Y--Y--

    So the total number of Y gates per U† is: 3 wires * 10 Y gates = 30 Y gates.

    Since there are 5 such U† gates, the final total number of Y gates is: 30 * 5 = 150.


    """

    result = calculate_noise_levels(
        nqubits=5,
        identity_factors=[0, 0, 5, 2],
        noise_profile={
            "status": True,
            "type": "Depolarizing",
            "noise_prob": [0, 0, 0, 0],
            "noise_on_init_param": {"status": False, "value": 0.1},
        },
    )

    nR, nCz, nT, nY = result["gates_num"]
    odd_n = result["odd_wires"]
    noise_level = result["noise_level"]
    assert (nR, nCz, nT, nY) == (4, 1, 11, 150)
    assert odd_n == 3
    assert noise_level == [0, 0, 0, 0]
