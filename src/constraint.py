"""
Scipy SLSQP constraint. It ensures the time params to be in incrementing order.
This code is based on a part of the following repository:
https://github.com/tanan/vqe-by-indirect-ctl
"""

import numpy as np
from scipy.optimize import LinearConstraint


def create_time_constraints(time_params_length, all_params_length) -> LinearConstraint:
    """
    Create constraints for time params to ensure each time parameter is positive
    and differences between consecutive time parameters are non-negative.

    Parameters:
        time_params_length (int): Number of time parameters.
        all_params_length (int): Total number of parameters including theta parameters.

    Returns:
        LinearConstraint: Linear constraint object representing the constraints.
    """
    matrix = np.zeros((2 * time_params_length, all_params_length))  # Initialize matrix

    # Set constraints for each time parameter to be positive
    for i in range(time_params_length):
        matrix[i, i] = 1  # t_i

    # Set constraints for differences between consecutive time parameters to be non-negative
    for i in range(1, time_params_length):
        matrix[time_params_length + (i - 1), i - 1] = -1  # -t_{i-1}
        matrix[time_params_length + (i - 1), i] = 1  # t_i

    return LinearConstraint(matrix, np.zeros(2 * time_params_length), np.inf)  # type: ignore
