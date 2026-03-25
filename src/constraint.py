# """
# Scipy SLSQP constraint. It ensures the time params to be in incrementing order.
# This code is based on a part of the following repository:
# https://github.com/tanan/vqe-by-indirect-ctl
# """

# import numpy as np
# from scipy.optimize import LinearConstraint


# def create_time_constraints(time_params_length, all_params_length) -> LinearConstraint:
#     """
#     Create constraints for time params to ensure each time parameter is positive
#     and differences between consecutive time parameters are non-negative.

#     Parameters:
#         time_params_length (int): Number of time parameters.
#         all_params_length (int): Total number of parameters including theta parameters.

#     Returns:
#         LinearConstraint: Linear constraint object representing the constraints.
#     """
#     matrix = np.zeros((2 * time_params_length, all_params_length))  # Initialize matrix

#     # Set constraints for each time parameter to be positive
#     for i in range(time_params_length):
#         matrix[i, i] = 1  # t_i

#     # Set constraints for differences between consecutive time parameters to be non-negative
#     for i in range(1, time_params_length):
#         matrix[time_params_length + (i - 1), i - 1] = -1  # -t_{i-1}
#         matrix[time_params_length + (i - 1), i] = 1  # t_i

#     return LinearConstraint(matrix, np.zeros(2 * time_params_length), np.inf)  # type: ignore

"""
Scipy SLSQP constraints for indirect time-evolution VQE.

1) Enforces:
   - t_i >= 0
   - t_i >= t_{i-1}  (monotonic time ordering)

2) Optionally enforces:
   - t_f = T_max     (fixed final time)

Based on:
https://github.com/tanan/vqe-by-indirect-ctl
"""

import numpy as np
from scipy.optimize import LinearConstraint
from typing import List


def create_time_constraints(
    time_params_length: int,
    all_params_length: int,
) -> LinearConstraint:
    """
    Enforce:
        t_i >= 0
        t_i - t_{i-1} >= 0

    Parameters
    ----------
    time_params_length : int
        Number of time parameters [t0, t1, ..., t_f]
    all_params_length : int
        Total number of parameters (time + angles)

    Returns
    -------
    LinearConstraint
    """
    # Number of constraints:
    #   time_params_length   (positivity)
    # + time_params_length-1 (ordering)
    rows = 2 * time_params_length - 1
    matrix = np.zeros((rows, all_params_length))

    # --- Positivity: t_i >= 0 ---
    for i in range(time_params_length):
        matrix[i, i] = 1.0

    # --- Ordering: t_i - t_{i-1} >= 0 ---
    for i in range(1, time_params_length):
        row = time_params_length + i - 1
        matrix[row, i - 1] = -1.0
        matrix[row, i] = 1.0

    lower = np.zeros(rows)
    upper = np.full(rows, np.inf)

    return LinearConstraint(matrix, lower, upper)


def create_tf_fixed_constraint(
    tf_index: int,
    all_params_length: int,
    T_max: float,
) -> LinearConstraint:
    """
    Force final time parameter t_f to be exactly T_max.

    Parameters
    ----------
    tf_index : int
        Index of t_f in the parameter vector
    all_params_length : int
        Total number of parameters
    T_max : float
        Fixed final evolution time

    Returns
    -------
    LinearConstraint
    """
    matrix = np.zeros((1, all_params_length))
    matrix[0, tf_index] = 1.0

    return LinearConstraint(matrix, [T_max], [T_max])