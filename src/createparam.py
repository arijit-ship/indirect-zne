import numpy as np


def create_param(layer: int, ti: float, tf: float) -> np.ndarray:
    """
    Creates parameter for the citcuit. Parameters are time, and theta: angle for rotation gates.

    time: 0 - max time
    theta: 0 - 1

    Args:
        layer (int): number of layer
        ti (float): initial time
        tf (float): final time

    Returns:
        numpy.ndarray: An nd-array [
        t1, t2, ... td, theta1, ..., theatd * 4
    ]

    """
    param = np.array([])
    # Time param
    time = np.random.uniform(ti, tf, layer)
    time = np.sort(time)  # Time must be in incresing order
    for i in time:
        param = np.append(param, i)

    # Theta param
    theta = np.random.random(layer * 4) * 1e-1  # Each layer has 4 rotation gates
    for i in theta:
        param = np.append(param, i)

    return param
