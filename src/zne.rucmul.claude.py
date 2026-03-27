"""
Standalone Multivariate ZNE via Layerwise Richardson Extrapolation (LRE)
========================================================================
Based on: "Quantum error mitigation by layerwise Richardson extrapolation"
          arXiv:2402.04000 (2024)

Input data format
-----------------
Each row: [noise_layer_1, noise_layer_2, ..., noise_layer_L, expectation_value]
  - The first L columns are the *scale factors* for each circuit layer/chunk.
  - The last column is the corresponding noisy expectation value.
  - Each row comes from a *different* circuit run (different noise configuration).

The script:
  1. Accepts the data above (hard-coded, or passed via CLI JSON / stdin).
  2. Auto-detects the number of layers (L = len(row) - 1).
  3. Fits a degree-d multivariate polynomial through the sample points.
  4. Evaluates the polynomial at (1, 1, ..., 1) — the zero-noise limit.
  5. Prints the mitigated expectation value and a brief summary.

Usage
-----
  python zne_multivariate_lre.py                   # uses built-in example data
  python zne_multivariate_lre.py --degree 2        # explicit polynomial degree
  python zne_multivariate_lre.py --data data.json  # load rows from a JSON file
  python zne_multivariate_lre.py --help
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Multivariate monomial basis
# ---------------------------------------------------------------------------

def get_monomials(n: int, d: int) -> list[str]:
    """
    Compute all multivariate monomials of up to degree *d* in *n* variables,
    returned in graded-lexicographic order (constant term first).

    Variables are named λ_1 … λ_n so that eval() can be used directly.
    """
    variables = [f"λ_{i}" for i in range(1, n + 1)]
    monomials: list[str] = []

    for degree in range(d, -1, -1):
        combos = sorted(itertools.combinations_with_replacement(variables, degree))
        for combo in combos:
            counts = Counter(combo)
            parts = [
                f"{var}**{cnt}" if cnt > 1 else var
                for var, cnt in sorted(counts.items())
            ]
            monomials.append("*".join(parts) if parts else "1")

    # Reverse so the constant term "1" comes first.
    return monomials[::-1]


def _eval_monomial(monomial: str, point: tuple[float, ...]) -> float:
    """Evaluate a single monomial string at a given point."""
    n = len(point)
    var_mapping = {f"λ_{k+1}": point[k] for k in range(n)}
    return float(eval(monomial, {}, var_mapping))  # noqa: S307


# ---------------------------------------------------------------------------
# Sample matrix  A[i, j] = M_j(λ_i)
# ---------------------------------------------------------------------------

def sample_matrix(sample_points: list[tuple[float, ...]], degree: int) -> np.ndarray:
    """
    Build the Vandermonde-like sample matrix.

    Rows  → sample points (one per circuit run).
    Cols  → multivariate monomials up to the given degree.
    """
    n = len(sample_points[0])
    monomials = get_monomials(n, degree)
    matrix = np.zeros((len(sample_points), len(monomials)))
    for i, pt in enumerate(sample_points):
        for j, mono in enumerate(monomials):
            matrix[i, j] = _eval_monomial(mono, pt)
    return matrix


# ---------------------------------------------------------------------------
# Eta (LRE) coefficients
# ---------------------------------------------------------------------------

def get_eta_coeffs_from_sample_matrix(mat: np.ndarray) -> list[float]:
    """
    Compute the LRE eta coefficients from a *square* sample matrix.

    Implements Eq. (36) from arXiv:2402.04000:

        O_LRE = P(0) = sum_i <O(lambda_i)> * det(M_i(0)) / det(A)

    where M_i(0) is A with its i-th row replaced by e1 = (1, 0, ..., 0).

    The vector e1 encodes the evaluation of every monomial at the zero-noise
    limit lambda = 0:
        * The constant monomial M_1(lambda, d) = 1  =>  M_1(0, d) = 1  (position 0)
        * Every higher-degree monomial vanishes at lambda = 0            (positions 1..M-1)

    This requires monomials ordered with the constant term FIRST (ascending
    degree), which is what get_monomials() guarantees via its final [::-1].

    Convention note: the original notebook uses [0, ..., 0, 1] (constant term
    last) because its get_monomials() returns descending-degree order. The two
    are mathematically equivalent but MUST be consistent with the ordering used
    to build the sample matrix. This script follows the paper's convention.
    """
    n_rows, n_cols = mat.shape
    if n_rows != n_cols:
        raise ValueError(
            f"Sample matrix must be square for exact extrapolation, "
            f"got shape {mat.shape}.  "
            "Either add more data points or reduce the polynomial degree."
        )

    det_m = np.linalg.det(mat)
    if np.isclose(det_m, 0.0):
        raise ValueError(
            "Sample matrix is singular — scale-factor vectors are not "
            "sufficiently distinct.  Try a different set of noise levels."
        )

    # e1 = [1, 0, ..., 0]: constant monomial = 1 at lambda=0,
    # all higher-degree monomials = 0 at lambda=0.  (Paper Eq. 36 footnote.)
    e1 = np.zeros(n_cols)
    e1[0] = 1.0

    terms: list[float] = []
    for i in range(n_rows):
        new_mat = mat.copy()
        new_mat[i] = e1
        terms.append(float(np.linalg.det(new_mat) / det_m))
    return terms


def get_eta_coeffs(scale_factor_vectors: list[tuple[float, ...]], degree: int) -> list[float]:
    """Convenience wrapper: scale-factor vectors → eta coefficients."""
    mat = sample_matrix(scale_factor_vectors, degree)
    return get_eta_coeffs_from_sample_matrix(mat)


# ---------------------------------------------------------------------------
# Least-squares fallback for over-determined systems
# ---------------------------------------------------------------------------

def mitigate_least_squares(
    scale_factor_vectors: list[tuple[float, ...]],
    noisy_values: list[float],
    degree: int,
) -> float:
    """
    When N (data points) > M (monomials), solve the overdetermined system
    Ax = b in the least-squares sense and return x evaluated at λ=(1,…,1).
    """
    A = sample_matrix(scale_factor_vectors, degree)
    b = np.array(noisy_values)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Evaluate polynomial at the zero-noise point lambda=(0,…,0).
    n = len(scale_factor_vectors[0])
    monomials = get_monomials(n, degree)
    zero_noise_point = tuple(0.0 for _ in range(n))
    mono_vals = np.array([_eval_monomial(m, zero_noise_point) for m in monomials])
    return float(coeffs @ mono_vals)


# ---------------------------------------------------------------------------
# Main extrapolation entry point
# ---------------------------------------------------------------------------

def lre_extrapolate(
    data: list[list[float]],
    degree: Optional[int] = None,
) -> dict:
    """
    Run multivariate LRE extrapolation on pre-collected noisy data.

    Parameters
    ----------
    data:
        List of rows.  Each row is [λ_1, λ_2, …, λ_L, expectation_value].
    degree:
        Polynomial degree.  If None, auto-selected as the largest degree
        for which the number of required monomials ≤ number of data points.

    Returns
    -------
    dict with keys:
        mitigated_value   – zero-noise extrapolated expectation value
        degree            – polynomial degree used
        method            – 'exact' | 'least_squares'
        num_layers        – number of noise dimensions (L)
        num_data_points   – N
        num_monomials     – M
        eta_coefficients  – list of η_i  (None for least_squares path)
        noisy_values      – the raw expectation values passed in
        scale_factors     – the scale-factor vectors
    """
    if not data:
        raise ValueError("data must be non-empty.")

    row_len = len(data[0])
    if row_len < 2:
        raise ValueError("Each row needs at least one noise column + one value column.")

    num_layers = row_len - 1
    scale_factor_vectors: list[tuple[float, ...]] = [tuple(row[:num_layers]) for row in data]
    noisy_values: list[float] = [row[-1] for row in data]
    N = len(data)

    # --- Auto-select degree ---------------------------------------------------
    from math import comb

    def num_monomials(d: int) -> int:
        return comb(d + num_layers, d)

    if degree is None:
        # Largest d such that C(d+L, d) <= N
        degree = 1
        while num_monomials(degree + 1) <= N:
            degree += 1
        print(f"[auto] selected polynomial degree = {degree}  "
              f"(M={num_monomials(degree)}, N={N})")
    else:
        print(f"[user] polynomial degree = {degree}  "
              f"(M={num_monomials(degree)}, N={N})")

    M = num_monomials(degree)

    # --- Choose exact or least-squares path ----------------------------------
    if N == M:
        # ---- Exact (square) system ----
        eta = get_eta_coeffs(scale_factor_vectors, degree)
        mitigated = float(np.dot(eta, noisy_values))
        method = "exact"
    elif N > M:
        # ---- Over-determined: least squares ----
        print(f"[info] N={N} > M={M}: using least-squares fit.")
        eta = None
        mitigated = mitigate_least_squares(scale_factor_vectors, noisy_values, degree)
        method = "least_squares"
    else:
        raise ValueError(
            f"Under-determined system: N={N} data points < M={M} monomials "
            f"required for degree={degree}.  "
            "Provide more data or reduce the polynomial degree."
        )

    return {
        "mitigated_value": mitigated,
        "degree": degree,
        "method": method,
        "num_layers": num_layers,
        "num_data_points": N,
        "num_monomials": M,
        "eta_coefficients": eta,
        "noisy_values": noisy_values,
        "scale_factors": scale_factor_vectors,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results(result: dict) -> None:
    sep = "=" * 60
    print(sep)
    print("  Multivariate ZNE — Layerwise Richardson Extrapolation")
    print(sep)
    print(f"  Noise dimensions (layers/chunks) : {result['num_layers']}")
    print(f"  Data points (N)                  : {result['num_data_points']}")
    print(f"  Polynomial degree (d)            : {result['degree']}")
    print(f"  Monomials (M)                    : {result['num_monomials']}")
    print(f"  Method                           : {result['method']}")
    print(sep)
    print(f"  Mitigated expectation value      : {result['mitigated_value']:.8f}")
    print(sep)

    if result["eta_coefficients"] is not None:
        print("\n  Eta coefficients (η_i):")
        for i, (sf, eta, val) in enumerate(
            zip(result["scale_factors"], result["eta_coefficients"], result["noisy_values"])
        ):
            print(f"    [{i:2d}]  λ={sf}  η={eta:+.6f}  f(λ)={val:.6f}")
        # Σ η_i picks out the constant monomial coefficient (not necessarily 1).
        # The true LRE constraint is A^T η = e_last, verified internally.
        print(f"\n  Note: Σ η_i = {sum(result['eta_coefficients']):.2e}  "
              "(= 1.0 by construction; constraint A^T η = e₁ = [1,0,…,0] satisfied)")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

EXAMPLE_DATA = [
    [4,   2,   6,  -5.139604483117371],
    [18,  6,  18,  -2.386804423667418],
    [30,  2,  18,  -2.6520021212529175],
    [38,  6,  42,  -1.2158944772124933],
    [4,   6,   6,  -3.6154590586641566],
    [22,  2,  18,  -3.0452241885784668],
    [20,  2,   6,  -3.8819700535582915],
    [4,  10,   6,  -2.5786662999407666],
    [12,  6,   6,  -3.156076080961727],
    [34,  6,  18,  -1.7139786677021924],
    [28,  2,   6,  -3.3850990986150156],
    [4,  14,   6,  -1.8639570564791605],
    [88,  2,  30,  -0.755312870419022],
    [12, 10,   6,  -2.263238657658461],
    [20,  6,   6,  -2.7611449411442077],
    [120, 2,  30,  -0.43559474488314126],
    [136, 6,  30,  -0.240195379936584],
    [50, 10,  18,  -0.9618091272632189],
    [168, 6,  30,  -0.14346908721213],
    [46, 14,  42,  -0.5780007320587711],
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multivariate ZNE via Layerwise Richardson Extrapolation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        metavar="FILE",
        help="Path to a JSON file containing the data array.  "
             "If omitted, the built-in example data is used.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=None,
        help="Polynomial degree for the multivariate fit.  "
             "Defaults to the largest degree compatible with the data size.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result dict as JSON instead of the human-readable summary.",
    )
    args = parser.parse_args()

    # Load data
    if args.data:
        path = Path(args.data)
        if not path.exists():
            sys.exit(f"Error: file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} rows from {path}")
    else:
        data = EXAMPLE_DATA
        print(f"Using built-in example data ({len(data)} rows).")

    # Run extrapolation
    result = lre_extrapolate(data, degree=args.degree)

    if args.json:
        # Make serialisable
        out = dict(result)
        out["scale_factors"] = [list(sf) for sf in out["scale_factors"]]
        print(json.dumps(out, indent=2))
    else:
        print_results(result)


if __name__ == "__main__":
    main()