from src.createparam import create_param

ti, tf = 0.0, 10.0


def test_param1():
    result = create_param(layer=1, gateset=1, ti=ti, tf=tf)
    assert len(result) == 5
