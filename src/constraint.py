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


def create_time_constraints_for_COBLYA(
    time_params_length: int,
) -> list[dict]:
    """
    COBYLA-compatible drop-in replacement for create_time_constraints().

    Enforce:
        t_0 >= 0
        t_i - t_{i-1} >= 0  for i = 1, ..., n-1

    COBYLA requires constraints as a list of dicts with:
        {"type": "ineq", "fun": callable}  # fun(x) >= 0

    Parameters
    ----------
    time_params_length : int
        Number of time parameters [t0, t1, ..., t_f].
        Note: all_params_length not needed — lambdas index directly into x.

    Returns
    -------
    list of dict, one scalar constraint per row
    """
    constraints = []

    # t_0 >= 0
    constraints.append({"type": "ineq", "fun": lambda x: x[0]})

    # t_i - t_{i-1} >= 0
    for i in range(1, time_params_length):
        constraints.append({
            "type": "ineq",
            "fun": lambda x, i=i: x[i] - x[i - 1]  # i=i captures loop var
        })

    return constraints

def create_time_constraints_with_mingap(
    time_params_length: int,
    all_params_length: int,
    min_dt: float = 1e-3,
) -> LinearConstraint:
    """
    Enforce:
        t_i >= 0
        t_i - t_{i-1} >= min_dt  (strict ordering, no duplicates)

    Parameters
    ----------
    time_params_length : int
        Number of time parameters [t0, t1, ..., t_f]
    all_params_length : int
        Total number of parameters (time + angles)
    min_dt : float
        Minimum required gap between consecutive time points (default: 1e-3)

    Returns
    -------
    LinearConstraint
    """
    rows = 2 * time_params_length - 1
    matrix = np.zeros((rows, all_params_length))

    # --- Positivity: t_i >= 0 ---
    for i in range(time_params_length):
        matrix[i, i] = 1.0

    # --- Ordering: t_i - t_{i-1} >= min_dt ---
    for i in range(1, time_params_length):
        row = time_params_length + i - 1
        matrix[row, i - 1] = -1.0
        matrix[row, i] = 1.0

    lower = np.zeros(rows)
    # Apply min_dt to only the ordering rows
    lower[time_params_length:] = min_dt
    upper = np.full(rows, np.inf)

    return LinearConstraint(matrix, lower, upper)