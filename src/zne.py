import itertools
import random
import re
from collections import Counter
from typing import List, Tuple, Union

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

"""
Parts of the following class are adapted from their notebook, which can be found at the
following GitHub repository:
https://github.com/unitaryfund/research/blob/main/lre/layerwise_richardson_extrapolation.ipynb.
"""


class ZeroNoiseExtrapolation:

    def __init__(
        self, datapoints: List[Tuple[Union[int, float], ...]], degree: int, method: str, sampling_mode: str
    ) -> None:

        self.datapoints = datapoints
        self.degree = degree
        self.method = method
        self.sampling_mode = sampling_mode

        # Number of independent variables
        self.independent_var_number: int = len(datapoints[0]) - 1

        self.noise_data = [tuple(point[: self.independent_var_number]) for point in self.datapoints]
        self.expectation_vals = [point[-1] for point in datapoints]

    def get_noise_levels(self) -> List[Tuple[int]]:
        """
        Returns a list containing all the noise-level values (independent variable values)
        extracted from the given datapoints.
        """
        return self.noise_data

    def get_expec_vals(self) -> List[float]:
        """
        Returns a list containing all the expectations values (dependent variable values)
        extracted from the given datapoints.
        """
        return self.expectation_vals

    def get_required_points(self) -> int:
        """
        Returns the number of datapoints required in order to perform Richardson extrapolation
        at a given degree and independent variables.
        """
        monomials = self.get_monomials(self.independent_var_number, self.degree)
        return len(monomials)

    def get_independent_var_number(self) -> int:
        """
        Returns the unmber of independent variables.
        """
        return self.independent_var_number

    def sampling(self) -> list:
        """
        This method samples data points from the dataset based on the sampling mode specified
        in the `self.sampling_mode`. The format of `self.sampling_mode` can be:

        - "random-N": Randomly selects `N` items from the dataset.
        - "default-N": Selects the first `N` items from the dataset.
        - "default": Selects all the items from the dataset.

        The method ensures the number of requested items (`N`) does not exceed the dataset size.
        If the format of `self.sampling_mode` is invalid, it raises a ValueError.
        """
        # Handle the case for "default" (all items)
        if self.sampling_mode == "default":
            return self.datapoints

        # Check for "default-N" or "random-N"
        match = re.match(r"(random|default)-(\d+)", self.sampling_mode)
        if not match:
            raise ValueError(
                "Invalid argument format. Use 'random-N', 'default-N', or 'default', where N is an integer."
            )

        mode, num_samples = match.groups()
        num_samples = int(num_samples)

        # Ensure the number of samples doesn't exceed the data size
        if num_samples > len(self.datapoints):
            raise ValueError("Sample size exceeds the size of the dataset.")

        # Handle "random-N" and "default-N"
        if mode == "random":
            return random.sample(self.datapoints, num_samples)
        elif mode == "default":
            return self.datapoints[:num_samples]  # First `num_samples` items

    def mul_RichardsonZNE(self, data) -> float:

        number_of_required_points = self.get_required_points()
        if number_of_required_points < len(data):
            raise ValueError(
                f"Multivariate Richardson error. At degree: {self.degree}, "
                f"Required data points: {number_of_required_points}, but was given: {len(data)}."
            )

        richardson_datapoints = data[:number_of_required_points]
        richardson_noise_vals = [tuple(point[: self.independent_var_number]) for point in richardson_datapoints]
        richardson_expectation_vals = [point[-1] for point in richardson_datapoints]
        RichardsonZNEval = 0
        sampleMatrix = self.sample_matrix(sample_points=richardson_noise_vals, degree=self.degree)  # type: ignore
        detA = np.linalg.det(sampleMatrix)
        if abs(detA) <= 1e-9:
            raise ValueError(f"Determinant of sample matrix is/close to zero. Det: {detA}, Deg: {self.degree}")
        # elif abs(detA) >= 1e+50:
        #     raise ValueError(f"Determinant of sample matrix close to inf. Det: {detA}, Deg: {self.degree}")

        matrices = self.generate_modified_matrices(sampleMatrix)  # type: ignore

        if len(richardson_expectation_vals) != len(matrices):
            raise ValueError("Unmatched length.")

        for E, matrix in zip(richardson_expectation_vals, matrices):
            eta = np.linalg.det(matrix) / detA
            E = np.array(E)
            RichardsonZNEval += E * eta
            eta = 0

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
        monomials = ZeroNoiseExtrapolation.get_monomials(n, degree)  # type: ignore
        matrix = np.zeros((len(sample_points), len(monomials)))

        for i, point in enumerate(sample_points):
            for j, monomial in enumerate(monomials):
                var_mapping = {f"λ_{k+1}": point[k] for k in range(n)}  # type: ignore
                matrix[i, j] = eval(monomial, {}, var_mapping)
        return matrix

    @staticmethod
    def get_eta_coeffs_from_sample_matrix(mat: np.ndarray) -> list[float]:
        """Given a sample matrix compute the eta coefficients."""
        n_rows, n_cols = mat.shape
        if n_rows != n_cols:
            raise ValueError("The matrix must be square.")

        det_m = np.linalg.det(mat)
        if det_m == 0:
            raise ValueError("The matrix is singular.")

        terms = []
        for i in range(n_rows):
            new_mat = mat.copy()
            new_mat[i] = np.array([[0] * (n_cols - 1) + [1]])
            terms.append(np.linalg.det(new_mat) / det_m)

        return terms

    @staticmethod
    def get_eta_coeffs_single_variable(scale_factors: list[float]) -> list[float]:
        """Returns the array of single-variable Richardson extrapolation coefficients associated
        to the input array of scale factors."""

        # Lagrange interpolation formula.
        richardson_coeffs = []
        for factor in scale_factors:
            coeff = 1.0
            for l_prime in scale_factors:
                if l_prime == factor:
                    continue
                coeff *= l_prime / (l_prime - factor)
            richardson_coeffs.append(coeff)

        return richardson_coeffs

    @staticmethod
    def generate_modified_matrices(matrix):
        """
        It generates the Mi(0) matreces for i = 1 to length of sample matrix.
        See this papaer for the detail mathematical formalism:
        "Quantum error mitigation by layerwise Richardson extrapolation"
        by Vincent Russo, Andrea Mari, https://arxiv.org/abs/2402.04000
        """
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

    # Standard single variate Richardson Extrapolation
    def getRichardsonZNE2(self, data):
        """
        Perform single-variable Richardson Extrapolation to estimate the zero-noise value.
        Beta coefficients are calculated using the product formula:
            beta_k = prod_{i != k} (alpha_i / (alpha_k - alpha_i))

        For further datails refer to: https://doi.org/10.48550/arXiv.2210.00921


        Args:
            data (list of lists): Each data point contains independent noise variables followed by the energy value.

        Returns:
            float: Zero-noise extrapolated value.
        """
        # Step 1: Extract total noise and corresponding expectation values
        total_noise = [sum(point[: self.independent_var_number]) for point in data]
        expectation_vals = [point[-1] for point in data]

        # Step 2: Sort data based on total noise in ascending order
        sorted_pairs = sorted(zip(total_noise, expectation_vals), key=lambda pair: pair[0])
        sorted_total_noise, sorted_expectation_vals = zip(*sorted_pairs)

        # Step 3: Compute beta coefficients using the product formula
        n = len(sorted_total_noise)
        betas = []

        for k in range(n):
            alpha_k = sorted_total_noise[k]
            beta_k = 1  # Initialize beta_k product to 1
            for i in range(n):
                if i != k:
                    alpha_i = sorted_total_noise[i]
                    beta_k *= alpha_i / (alpha_k - alpha_i)
            betas.append(beta_k)

        # Step 4: Normalize betas to ensure sum(beta) = 1
        beta_sum = sum(betas)
        betas = [beta / beta_sum for beta in betas]

        # print("Sorted Total Noise:", sorted_total_noise)
        # print("Sorted Expectation Values:", sorted_expectation_vals)
        # print("Betas Coefficients:", betas)

        # Step 5: Compute zero-noise extrapolated value
        zne_val = sum(betas[i] * sorted_expectation_vals[i] for i in range(n))
        # print("Zero-Noise Extrapolated Value:", zne_val)

        richardson_steps_details = {
            "sorted_noise": sorted_total_noise,
            "sorted_expectation_vals": sorted_expectation_vals,
            "beta_coeffiecients": betas,
        }

        return zne_val, richardson_steps_details

    def scikit_linear(self, data) -> float:
        """
        Scikitlearn linear extrapolation.
        """
        # Extract noise levels (nR, nT, nY, nCz) and energy values from datapoints

        # Extract the noise levels
        noise = np.array([point[: self.independent_var_number] for point in data])

        # Extract the energy values
        energy = np.array([point[-1] for point in data])

        # Linear regression model
        model = LinearRegression()

        # Train the model on the data
        model.fit(noise, energy)

        # Zero limit
        zero_limit = []
        for _ in range(self.independent_var_number):
            zero_limit.append(0)

        # Extrapolate the energy value for the noise level (0, 0, 0)
        extrapolated_value = model.predict([zero_limit])[0]

        # Return the predicted energy value at (0, 0, 0)
        return extrapolated_value

    def scikit_poly(self, data) -> float:
        """
        Scikitlearn polynomial extrapolation.
        """
        # Extract noise levels (nR, nT, nY, nCz) and energy values from datapoints

        # Extract the noise levels
        noise = np.array([point[: self.independent_var_number] for point in data])

        # Extract the energy values
        energy = np.array([point[-1] for point in data])

        # Polynomial features based on the degree specified
        poly = PolynomialFeatures(degree=self.degree)

        # Transform the input data into polynomial features
        noise_poly = poly.fit_transform(noise)

        # Step 3: Create and fit the linear regression model
        model = LinearRegression()

        # Fit the model to the polynomial-transformed data
        model.fit(noise_poly, energy)

        # Zero limit
        zero_limit = []
        for _ in range(self.independent_var_number):
            zero_limit.append(0)

        # Step 4: Extrapolate the energy value for the noise level (0, 0, 0)
        # Transform (0, 0, 0, 0) into polynomial features and predict the energy
        # value
        # Transform the (0, 0, 0, 0) noise level into polynomial features
        zero_noise = poly.transform([zero_limit])
        # Predict the energy value at (0, 0, 0)
        extrapolated_value = model.predict(zero_noise)[0]

        # Return the predicted energy value
        return extrapolated_value

    def getZne(self) -> float:

        sampled_data = self.sampling()
        if self.method.lower() == "richardson-mul":
            zne_val = {
                "degree": self.degree,
                "sampling": self.sampling_mode,
                "sampled data": sampled_data,
                "extrapolated_value": self.mul_RichardsonZNE(data=sampled_data),
            }

        elif self.method.lower() == "richardson":
            zne_extrapolated_val, richardson_step_details = self.getRichardsonZNE2(data=sampled_data)
            zne_val = {
                "degree": self.degree,
                "sampling": self.sampling_mode,
                "sampled data": sampled_data,
                "extrapolated_value": zne_extrapolated_val,
                "others": richardson_step_details,
            }

        elif self.method.lower() == "linear":
            zne_val = {
                "degree": self.degree,
                "sampling": self.sampling_mode,
                "sampled data": sampled_data,
                "extrapolated_value": self.scikit_linear(data=sampled_data),
            }

        elif self.method.lower() == "polynomial":
            zne_val = {
                "degree": self.degree,
                "sampling": self.sampling_mode,
                "sampled data": sampled_data,
                "extrapolated_value": self.scikit_poly(data=sampled_data),
            }

        else:
            raise ValueError(
                f"Invalid method: {self.method}. Valid methods are: richardson-m, richardson, linear, and polynomial."
            )

        return zne_val
