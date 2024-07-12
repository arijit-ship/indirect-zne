from src.modules import noise_level

# nY = existing odd no + (2 * noise factor * existing odd no)

def test_noise_param1():
    result = noise_level(nqubits= 5, identity_factor=[0, 0, 0])
    nR, nT, nY = result["params"]
    odd_n = result["odd_wires"]
    assert (nR, nT, nY) == (4, 1, 0)
    assert odd_n == 3

def test_noise_param2():
    result = noise_level(nqubits= 7, identity_factor=[0, 0, 0])
    nR, nT, nY = result ["params"]
    assert (nR, nT, nY) == (4, 1, 0)

def test_noise_param3():
    result = noise_level(nqubits= 7, identity_factor=[1, 0, 0])
    nR, nT, nY = result["params"]
    assert (nR, nT, nY) == (12, 1, 0)

def test_noise_param4():
    result  = noise_level(nqubits= 7, identity_factor=[0, 0, 1])
    nR, nT, nY = result["params"]
    assert (nR, nT, nY) == (4, 1, 0)

def test_noise_param5():
    result  = noise_level(nqubits= 7, identity_factor=[0, 1, 0])
    nR, nT, nY = result["params"]
    odd_n = result["odd_wires"]
    assert (nR, nT, nY) == (4, 3, odd_n + (2*0*odd_n))

def test_noise_param6():
    result  = noise_level(nqubits= 7, identity_factor=[0, 1, 1])
    nR, nT, nY = result["params"]
    odd_n = result["odd_wires"]
    assert (nR, nT, nY) == (4, 3, odd_n + (2*1*odd_n)) 
def test_noise_param7():
    result = noise_level(nqubits= 7, identity_factor=[0, 1, 2])
    nR, nT, nY = result["params"]
    odd_n = result["odd_wires"]
    assert (nR, nT, nY) == (4, 3, odd_n + (2*2*odd_n))
    assert odd_n == 4

def test_noise_param8():
    result = noise_level(nqubits= 8, identity_factor=[0, 1, 2])
    nR, nT, nY = result["params"]
    odd_n = result["odd_wires"]
    assert (nR, nT, nY) == (4, 3, odd_n + (2*2*odd_n))

def test_noise_param9():
    result = noise_level(nqubits= 9, identity_factor=[2, 2, 2])
    nR, nT, nY = result["params"]
    odd_n = result["odd_wires"]
    assert (odd_n, nR, nT, nY) == (5, 4 + (2*8), 5, odd_n + (2*2*odd_n))

def test_noise_param10():
    result = noise_level(nqubits= 4, identity_factor=[3, 3, 3])
    nR, nT, nY = result["params"]
    odd_n = result["odd_wires"]
    assert (odd_n, nR, nT, nY) == (2, 4 + (3*8), 1 + (3*2), (2 + (2*3*2)))