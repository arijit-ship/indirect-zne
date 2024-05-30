import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modules import noise_param

def test_noise_param1():
    nR, nT, nY = noise_param(nqubit= 5, noise_factor=[0, 0, 0])
    assert (nR, nT, nY) == (4, 1, 0)
    
def test_noise_param2():
    nR, nT, nY = noise_param(nqubit= 7, noise_factor=[0, 0, 0])
    assert (nR, nT, nY) == (4, 1, 0)

def test_noise_param3():
    nR, nT, nY = noise_param(nqubit= 7, noise_factor=[1, 0, 0])
    assert (nR, nT, nY) == (12, 1, 0)

def test_noise_param4():
    nR, nT, nY = noise_param(nqubit= 7, noise_factor=[0, 0, 1])
    assert (nR, nT, nY) == (4, 1, 0)

def test_noise_param5():
    nR, nT, nY = noise_param(nqubit= 7, noise_factor=[0, 1, 0])
    assert (nR, nT, nY) == (4, 3, 4)

def test_noise_param6():
    nR, nT, nY = noise_param(nqubit= 7, noise_factor=[0, 1, 1])
    assert (nR, nT, nY) == (4, 3, (4 + (4*2)))

def test_noise_param7():
    nR, nT, nY = noise_param(nqubit= 7, noise_factor=[0, 1, 2])
    assert (nR, nT, nY) == (4, 3, (4 + (4*2*2)))

def test_noise_param8():
    nR, nT, nY = noise_param(nqubit= 8, noise_factor=[0, 1, 2])
    assert (nR, nT, nY) == (4, 3, (4 + (4*2*2)))

def test_noise_param9():
    nR, nT, nY = noise_param(nqubit= 9, noise_factor=[2, 2, 2])
    assert (nR, nT, nY) == (4 + (2*8), 5, (5 + (5*2*2)))

def test_noise_param10():
    nR, nT, nY = noise_param(nqubit= 4, noise_factor=[3, 3, 3])
    assert (nR, nT, nY) == (4 + (3*8), 1 + (3*2), (2 + (2*3*2)))