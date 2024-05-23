"""
We have used the multivariate framework for Richardson extrapolation as discussed in the paper "Quantum error mitigation by layerwise Richardson extrapolation" by Vincent Russo and Andrea Mari (arXiv:2402.04000, 2024).

Parts of the following code are adapted from their notebook, which can be found at the following GitHub repository: https://github.com/unitaryfund/research/blob/main/lre/layerwise_richardson_extrapolation.ipynb.
"""

import numpy as np
from collections import Counter
import itertools


class ZeroNoiseExtrapolation:
    def __init__(self, dataPoints: list[tuple[float]], degree: int):
        """
        Initialize with a list of data points, each represented as a tuple of floats.
        """
        self.dataPoints = dataPoints
        self.degree = degree

        self.NoiseData = [tuple(point[:3]) for point in self.dataPoints]
        self.ExpectationVals = [point[-1] for point in dataPoints]
  
    def getRichardsonZNE(self):

        RichardsonZNEval = 0

        sampleMatrix = sample_matrix(sample_points = self.NoiseData, degree = self.degree) # type: ignore
        detA = np.linalg.det(sampleMatrix)

        matrices = generate_modified_matrices(sampleMatrix) # type: ignore

        for E, matrix in zip(self.ExpectationVals, matrices):
            RichardsonZNEval += E * (np.linalg.det(matrix)/detA)
        
        return RichardsonZNEval
    
    @staticmethod
    def get_monomials(n: int, d: int) -> list[str]:
        """
        Compute monomials of degree `d` in graded lexicographical order.
        """
        variables = [f"λ_{i}" for i in range(1, n + 1)]
        
        monomials = []
        for degree in range(d, -1, -1):
            # Generate combinations for the current degree
            combos = list(itertools.combinations_with_replacement(variables, degree))
            
            # Sort combinations lexicographically
            combos.sort()
            
            # Construct monomials from sorted combinations
            for combo in combos:
                monomial_parts = []
                counts = Counter(combo)
                # Ensure variables are processed in lexicographical order
                for var in sorted(counts.keys()):
                    count = counts[var]
                    if count > 1:
                        monomial_parts.append(f"{var}**{count}")
                    else:
                        monomial_parts.append(var)
                monomial = "*".join(monomial_parts)
                # Handle the case where degree is 0
                if not monomial:
                    monomial = "1"
                monomials.append(monomial)
        # "1" should be the first monomial. Note that order d > c > b > a means vector of monomials = [a, b, c, d].            
        return monomials[::-1]

    @staticmethod
    def sample_matrix(sample_points: list[int], degree: int) -> np.ndarray:
        """Construct a matrix from monomials evaluated at sample points."""
        n = len(sample_points[0])  # type: ignore # Number of variables based on the first sample point
        monomials = get_monomials(n, degree) # type: ignore
        matrix = np.zeros((len(sample_points), len(monomials)))

        for i, point in enumerate(sample_points):
            for j, monomial in enumerate(monomials):
                var_mapping = {f"λ_{k+1}": point[k] for k in range(n)} # type: ignore
                matrix[i, j] = eval(monomial, {}, var_mapping)
        return matrix

    @staticmethod
    def generate_modified_matrices(matrix):
        n = len(matrix)  # Size of the square matrix
        identity_row = [1] + [0] * (n - 1)  # Row to replace with

        modified_matrices = []
        determinants = []
        for i in range(n):
            # Create a copy of the original matrix
            modified_matrix = np.copy(matrix)
            # Replace the i-th row with the identity_row
            modified_matrix[i] = identity_row
            modified_matrices.append(modified_matrix)
            # Calculate the determinant of the modified matrix
            determinant = np.linalg.det(modified_matrix)
            determinants.append(determinant)
        
        return modified_matrices 