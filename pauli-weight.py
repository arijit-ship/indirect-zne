"""
pauli_weight.py
===============
Modular, production-grade computation of the Pauli weight distribution
for an n-qubit density matrix.

The Pauli weight distribution {r_k} is defined as:

    r_k = (1 / 2^n) * sum_{P in P_k} |Tr(P * rho)|^2

where P_k is the set of all n-qubit Pauli operators with exactly k
non-identity single-qubit factors.

References
----------
- Leone et al., "Nonstabilizerness determining the hardness of
  random quantum circuits simulation", PRL 2022.
- Nielsen & Chuang, "Quantum Computation and Quantum Information".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import List, Optional

import numpy as np

# Optional: qulacs is required at runtime but not at import time
# so the module can be imported for testing / type-checking without it.
try:
    from qulacs import DensityMatrix, PauliOperator
    from qulacs import state as qulacs_state
    _QULACS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _QULACS_AVAILABLE = False

__all__ = [
    "PauliWeightResult",
    "compute_pauli_weight_distribution",
    "build_pauli_operators",
    "expectation_value",
    "validate_distribution",
    "summary_table",
    "load_density_matrix_from_json",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PauliWeightResult:
    """
    Container for the Pauli weight distribution of a density matrix.

    Attributes
    ----------
    n_qubits : int
        Number of qubits.
    distribution : List[float]
        r_k values for k = 0, 1, ..., n_qubits.
    is_normalized : bool
        Whether sum(distribution) ≈ 1 within tolerance.
    """
    n_qubits: int
    distribution: List[float]
    is_normalized: bool = field(init=False)
    _tol: float = field(default=1e-6, repr=False)

    def __post_init__(self) -> None:
        self.is_normalized = abs(sum(self.distribution) - 1.0) < self._tol

    @property
    def mean_weight(self) -> float:
        """Expected Pauli weight <k> = sum_k k * r_k."""
        return sum(k * r for k, r in enumerate(self.distribution))

    @property
    def max_weight_index(self) -> int:
        """Index k at which r_k is maximised."""
        return int(np.argmax(self.distribution))

    def __repr__(self) -> str:  # noqa: D105
        lines = [f"PauliWeightResult(n_qubits={self.n_qubits})"]
        for k, r in enumerate(self.distribution):
            bar = "█" * int(r * 40)
            lines.append(f"  r_{k:<2d} = {r:.6f}  {bar}")
        lines.append(f"  sum  = {sum(self.distribution):.6f}")
        lines.append(f"  <k>  = {self.mean_weight:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------

_PAULI_LABELS = ("X", "Y", "Z")


def build_pauli_operators(n_qubits: int, weight: int) -> List[str]:
    """
    Generate all Qulacs Pauli operator strings of a given weight.

    Each string has the form ``"X 0 Z 3"`` — pairs of (label, qubit_index)
    for every non-identity position.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits.
    weight : int
        Number of non-identity single-qubit Pauli factors (0 ≤ weight ≤ n_qubits).

    Returns
    -------
    List[str]
        List of Qulacs-compatible Pauli strings.  Empty string represents
        the all-identity (weight-0) operator.

    Raises
    ------
    ValueError
        If weight is negative or exceeds n_qubits.
    """
    if not (0 <= weight <= n_qubits):
        raise ValueError(
            f"weight must be in [0, {n_qubits}], got {weight}."
        )

    if weight == 0:
        return [""]  # identity operator

    operators: List[str] = []
    for positions in combinations(range(n_qubits), weight):
        for labels in product(_PAULI_LABELS, repeat=weight):
            op_str = " ".join(f"{lbl} {idx}" for lbl, idx in zip(labels, positions))
            operators.append(op_str)

    return operators


def expectation_value(op_str: str, rho: "DensityMatrix") -> complex:
    """
    Compute Tr(P * rho) for a Pauli operator encoded as a Qulacs string.

    Parameters
    ----------
    op_str : str
        Qulacs Pauli string (e.g. ``"X 0 Z 2"``).  Empty string means identity.
    rho : DensityMatrix
        Qulacs density matrix.

    Returns
    -------
    complex
        The expectation value Tr(P * rho).
    """
    if op_str == "":
        return complex(1.0)  # Tr(I * rho) = Tr(rho) = 1

    pauli_op = PauliOperator(op_str, 1.0)
    return complex(pauli_op.get_expectation_value(rho))


def _compute_rk(
    rho: "DensityMatrix",
    k: int,
    norm: float,
) -> float:
    """
    Compute r_k — the k-th element of the Pauli weight distribution.

    Parameters
    ----------
    rho : DensityMatrix
        Qulacs density matrix (must have qubit count consistent with ``norm``).
    k : int
        Pauli weight order.
    norm : float
        Normalisation constant (2^n).

    Returns
    -------
    float
        r_k value.
    """
    n_qubits = rho.get_qubit_count()
    operators = build_pauli_operators(n_qubits, k)

    rk_sum = sum(
        abs(expectation_value(op_str, rho)) ** 2
        for op_str in operators
    )

    return rk_sum / norm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pauli_weight_distribution(
    rho: "DensityMatrix",
    verbose: bool = False,
) -> PauliWeightResult:
    """
    Compute the full Pauli weight distribution {r_k} for a density matrix.

    The distribution satisfies:

        r_k ≥ 0  for all k,
        sum_k r_k = 1.

    Parameters
    ----------
    rho : DensityMatrix
        An n-qubit Qulacs density matrix.
    verbose : bool, optional
        If True, log progress for each k.  Default is False.

    Returns
    -------
    PauliWeightResult
        Dataclass holding the distribution and derived statistics.

    Raises
    ------
    ImportError
        If qulacs is not installed.
    TypeError
        If ``rho`` is not a ``DensityMatrix`` instance.
    """
    if not _QULACS_AVAILABLE:
        raise ImportError(
            "qulacs is required: pip install qulacs"
        )
    if not isinstance(rho, DensityMatrix):
        raise TypeError(
            f"Expected qulacs.DensityMatrix, got {type(rho).__name__}."
        )

    n = rho.get_qubit_count()
    norm = float(2 ** n)
    distribution: List[float] = []

    logger.info("Computing Pauli weight distribution for %d qubits.", n)

    for k in range(n + 1):
        rk = _compute_rk(rho, k, norm)
        distribution.append(rk)

        if verbose:
            n_terms = (3 ** k) * int(
                np.math.comb(n, k)  # type: ignore[attr-defined]
            ) if k > 0 else 1
            logger.info("  r_%d = %.6f  (%d Pauli strings)", k, rk, n_terms)

    return PauliWeightResult(n_qubits=n, distribution=distribution)


def validate_distribution(result: PauliWeightResult, tol: float = 1e-6) -> bool:
    """
    Validate that a PauliWeightResult satisfies normalisation and non-negativity.

    Parameters
    ----------
    result : PauliWeightResult
        Result to validate.
    tol : float, optional
        Tolerance for the normalisation check.  Default is 1e-6.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    total = sum(result.distribution)
    non_negative = all(r >= -tol for r in result.distribution)
    normalised = abs(total - 1.0) < tol

    if not non_negative:
        logger.warning("Distribution contains negative values.")
    if not normalised:
        logger.warning("Distribution not normalised: sum = %.8f", total)

    return non_negative and normalised


def summary_table(result: PauliWeightResult) -> str:
    """
    Return a formatted plain-text summary table of the distribution.

    Parameters
    ----------
    result : PauliWeightResult
        Computed Pauli weight distribution.

    Returns
    -------
    str
        Multi-line string suitable for printing or logging.
    """
    width = 52
    sep = "─" * width
    lines = [
        sep,
        f"  Pauli Weight Distribution  (n = {result.n_qubits} qubits)",
        sep,
        f"  {'k':<4}  {'r_k':>10}  {'%':>7}  Bar",
        sep,
    ]
    for k, r in enumerate(result.distribution):
        bar = "▪" * max(1, int(r * 30))
        lines.append(f"  {k:<4}  {r:>10.6f}  {r*100:>6.2f}%  {bar}")

    lines += [
        sep,
        f"  sum(r_k)     = {sum(result.distribution):.8f}",
        f"  mean weight  = {result.mean_weight:.4f}",
        f"  peak at k    = {result.max_weight_index}",
        f"  normalised   = {result.is_normalized}",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_density_matrix_from_json(path: str, state: str = "final") -> "DensityMatrix":
    """Load a DensityMatrix from a VQE result JSON file."""
    import json
    with open(path) as f:
        data = json.load(f)
    state_json_str = data["others"][f"{state}_states"][0]
    return qulacs_state.from_json(state_json_str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Compute Pauli weight distribution from a VQE JSON file."
    )
    parser.add_argument("path", help="Path to the VQE result JSON file.")
    args = parser.parse_args()

    for state in ("initial", "final"):
        print(f"\n{'='*52}")
        print(f"  {state.upper()} STATE")
        rho = load_density_matrix_from_json(args.path, state=state)
        result = compute_pauli_weight_distribution(rho)
        print(summary_table(result))
        validate_distribution(result)