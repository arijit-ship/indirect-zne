import itertools
import random
import re
from collections import Counter
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

"""
Parts of the following class are adapted from their notebook, which can be found at the
following GitHub repository:
https://github.com/unitaryfund/research/blob/main/lre/layerwise_richardson_extrapolation.ipynb.

LLM has been used to refactor the code, add docstrings, and implement additional features.
"""


class ZeroNoiseExtrapolation:

    def __init__(
        self, datapoints: List[Tuple[Union[int, float], ...]], degree: int, method: str, sampling_mode: str
    ) -> None:

        self.datapoints = datapoints
        self.degree = degree
        self.method = method
        self.sampling_mode = sampling_mode
 
        # Number of independent variables (noise dimensions / circuit layers).
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
        Returns a list containing all the expectation values (dependent variable values)
        extracted from the given datapoints.
        """
        return self.expectation_vals
 
    def get_required_points(self) -> int:
        """
        Returns the number of datapoints required to perform multivariate Richardson
        extrapolation at the configured degree and number of independent variables.
 
        This equals the number of multivariate monomials M = C(degree + l, degree),
        where l is the number of noise dimensions (circuit layers/chunks).
        """
        monomials = self.get_monomials(self.independent_var_number, self.degree)
        return len(monomials)
 
    def get_independent_var_number(self) -> int:
        """
        Returns the number of independent variables (noise dimensions).
        """
        return self.independent_var_number
 
    def sampling(self) -> list:
        """
        Samples datapoints from the dataset according to `self.sampling_mode`.
 
        Supported formats:
            - ``"default"``   : Returns all datapoints unchanged.
            - ``"default-N"`` : Returns the first N datapoints.
            - ``"random-N"``  : Returns N datapoints chosen uniformly at random
                                without replacement.
 
        Raises:
            ValueError: If the format of ``self.sampling_mode`` is invalid, or if
                        N exceeds the size of the dataset.
        """
        # Return the full dataset when no sub-sampling is requested.
        if self.sampling_mode == "default":
            return self.datapoints
 
        # Parse "default-N" or "random-N".
        match = re.match(r"(random|default)-(\d+)", self.sampling_mode)
        if not match:
            raise ValueError(
                "Invalid argument format. Use 'random-N', 'default-N', or 'default', where N is an integer."
            )
 
        mode, num_samples = match.groups()
        num_samples = int(num_samples)
 
        if num_samples > len(self.datapoints):
            raise ValueError("Sample size exceeds the size of the dataset.")
 
        if mode == "random":
            return random.sample(self.datapoints, num_samples)
        else:  # mode == "default"
            return self.datapoints[:num_samples]
 
    def mul_RichardsonZNE(self, data) -> float:
        """
        Performs multivariate Layerwise Richardson Extrapolation (LRE) to estimate
        the zero-noise expectation value.
 
        Implements Eq. (36) from arXiv:2402.04000:
 
            O_LRE = P(0) = sum_i <O(lambda_i)> * det(M_i(0)) / det(A)
 
        Exactly M = C(degree + l, degree) datapoints are used, where l is the number
        of noise dimensions.  Any additional points supplied in `data` are ignored;
        call with a pre-sampled subset if a different selection is desired.
 
        Args:
            data: List of datapoints, each of the form
                  [noise_1, noise_2, ..., noise_l, expectation_value].
 
        Returns:
            Zero-noise extrapolated expectation value (float).
 
        Raises:
            ValueError: If fewer datapoints are provided than required, or if the
                        resulting sample matrix is singular.
        """
        number_of_required_points = self.get_required_points()

        eta_list = []
 
        # BUG FIX: the guard was inverted — it previously raised when len(data) > required,
        # i.e. on every valid over-supplied call.  The correct check is the opposite:
        # raise only when there are *too few* points to form the square system.
        if number_of_required_points > len(data):
            raise ValueError(
                f"Multivariate Richardson error. At degree: {self.degree}, "
                f"required data points: {number_of_required_points}, but was given: {len(data)}."
            )
 
        # Use exactly the required number of points to form the square system.
        richardson_datapoints = data[:number_of_required_points]
        richardson_noise_vals = [tuple(point[: self.independent_var_number]) for point in richardson_datapoints]
        richardson_expectation_vals = [point[-1] for point in richardson_datapoints]
 
        sample_matrix = self.sample_matrix(sample_points=richardson_noise_vals, degree=self.degree)
        det_a = np.linalg.det(sample_matrix)
 
        if abs(det_a) <= 1e-9:
            raise ValueError(
                f"Determinant of sample matrix is zero or near-zero "
                f"(det = {det_a:.3e}, degree = {self.degree}). "
                "Ensure scale-factor vectors are sufficiently distinct."
            )
 
        # Generate the M_i(0) matrices and accumulate the weighted sum.
        modified_matrices = self.generate_modified_matrices(sample_matrix)
 
        if len(richardson_expectation_vals) != len(modified_matrices):
            raise ValueError(
                f"Length mismatch: {len(richardson_expectation_vals)} expectation values "
                f"vs {len(modified_matrices)} modified matrices."
            )
 
        zne_value = 0.0
        for expectation_val, modified_matrix in zip(richardson_expectation_vals, modified_matrices):
            eta = np.linalg.det(modified_matrix) / det_a
            zne_value += np.array(expectation_val) * eta
            eta_list.append(eta)
        eta_sum = sum(np.abs(eta) for eta in eta_list)
        cost = eta_sum **2
        result = {
            "extrapolated_value": zne_value,
            "richardson_steps_details": {
                "eta_coefficients": eta_list,
                "cost_zne": cost,
                "sample_matrix_determinant": det_a,
            }
        }
        return result
 
    @staticmethod
    def get_monomials(n: int, d: int) -> list[str]:
        """
        Computes all multivariate monomials of degree at most `d` in `n` variables,
        returned in ascending graded-lexicographic order (constant term ``"1"`` first).
 
        Variables are named ``λ_1, …, λ_n`` so the strings can be evaluated directly
        via ``eval()``.
 
        Args:
            n: Number of variables.
            d: Maximum total degree.
 
        Returns:
            List of monomial strings, e.g. ``['1', 'λ_1', 'λ_2', 'λ_1**2', ...]``.
        """
        variables = [f"λ_{i}" for i in range(1, n + 1)]
 
        monomials = []
        for degree in range(d, -1, -1):
            combos = list(itertools.combinations_with_replacement(variables, degree))
            combos.sort()
 
            for combo in combos:
                monomial_parts = []
                counts = Counter(combo)
                for var in sorted(counts.keys()):
                    count = counts[var]
                    if count > 1:
                        monomial_parts.append(f"{var}**{count}")
                    else:
                        monomial_parts.append(var)
                monomial = "*".join(monomial_parts)
                if not monomial:
                    monomial = "1"
                monomials.append(monomial)
 
        # Reverse to ascending order: constant term "1" at index 0, highest-degree
        # monomials last.  This ordering is required for generate_modified_matrices
        # to place the substitution value correctly (see paper Eq. 36 footnote).
        return monomials[::-1]
 
    @staticmethod
    def sample_matrix(sample_points: list[int], degree: int) -> np.ndarray:
        """
        Constructs the Vandermonde-like sample matrix A, where entry A[i, j] is the
        j-th monomial evaluated at the i-th scale-factor vector.
 
        Args:
            sample_points: List of scale-factor vectors (one per circuit run).
            degree: Maximum polynomial degree.
 
        Returns:
            NumPy array of shape (N, M), where N = len(sample_points) and
            M = number of monomials = C(degree + l, degree).
        """
        n = len(sample_points[0])
        monomials = ZeroNoiseExtrapolation.get_monomials(n, degree)
        matrix = np.zeros((len(sample_points), len(monomials)))
 
        for i, point in enumerate(sample_points):
            for j, monomial in enumerate(monomials):
                var_mapping = {f"λ_{k+1}": point[k] for k in range(n)}
                matrix[i, j] = eval(monomial, {}, var_mapping)
 
        return matrix
 
    @staticmethod
    def get_eta_coeffs_from_sample_matrix(mat: np.ndarray) -> list[float]:
        """
        Computes the LRE eta coefficients from a square sample matrix via
        Cramer's rule (Eq. 36, arXiv:2402.04000).
 
        Each coefficient η_i = det(M_i(0)) / det(A), where M_i(0) is the sample
        matrix with row i replaced by e₁ = (1, 0, …, 0).
 
        The vector e₁ encodes monomial evaluations at the zero-noise limit λ = 0:
            - Constant monomial M_1(0) = 1  →  position 0 receives 1.
            - All higher-degree monomials vanish at λ = 0  →  remaining positions are 0.
 
        This convention requires monomials to be ordered with the constant term first
        (ascending degree), which is guaranteed by ``get_monomials()``.
 
        Args:
            mat: Square sample matrix of shape (M, M).
 
        Returns:
            List of M eta coefficients.
 
        Raises:
            ValueError: If the matrix is not square or is singular.
        """
        n_rows, n_cols = mat.shape
        if n_rows != n_cols:
            raise ValueError("The sample matrix must be square.")
 
        det_m = np.linalg.det(mat)
        if np.isclose(det_m, 0.0):
            raise ValueError(
                f"The sample matrix is singular (det ≈ {det_m:.3e}). "
                "Ensure scale-factor vectors are sufficiently distinct."
            )
 
        # e₁ = [1, 0, …, 0]: constant monomial evaluates to 1 at λ = 0;
        # all higher-degree monomials evaluate to 0.  (Paper Eq. 36 footnote.)
        e1 = np.zeros(n_cols)
        e1[0] = 1.0
 
        terms = []
        for i in range(n_rows):
            new_mat = mat.copy()
            new_mat[i] = e1
            terms.append(np.linalg.det(new_mat) / det_m)
 
        return terms
 
    @staticmethod
    def get_eta_coeffs_single_variable(scale_factors: list[float]) -> list[float]:
        """
        Returns the Richardson extrapolation coefficients for the single-variable case
        using the Lagrange interpolation formula:
 
            β_k = ∏_{i ≠ k} α_i / (α_i - α_k)
 
        The coefficients satisfy Σ β_k = 1 by construction; no normalisation is
        applied or needed.
 
        Args:
            scale_factors: List of noise scale factors α_1, …, α_N.
 
        Returns:
            List of N Richardson coefficients.
 
        References:
            https://doi.org/10.48550/arXiv.2210.00921
        """
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
    def generate_modified_matrices(matrix: np.ndarray) -> list[np.ndarray]:
        """
        Generates the sequence of modified matrices M_i(0) for i = 0, …, N-1,
        as required by the Lagrange extrapolation formula (Eq. 36, arXiv:2402.04000).
 
        Each M_i(0) is a copy of the sample matrix with its i-th row replaced by
        e₁ = (1, 0, …, 0), which encodes the evaluation of all monomials at the
        zero-noise limit λ = 0.  The constant monomial evaluates to 1 (position 0);
        all higher-degree monomials evaluate to 0.
 
        Args:
            matrix: Square sample matrix of shape (N, N).
 
        Returns:
            List of N modified matrices, each of shape (N, N).
        """
        n = len(matrix)
 
        # e₁ = [1, 0, …, 0]: constant monomial = 1 at λ = 0, all others = 0.
        # Monomials are ordered ascending (constant term first) by get_monomials(),
        # so the substitution value belongs at index 0.
        identity_row = np.zeros(n)
        identity_row[0] = 1.0
 
        modified_matrices = []
        for i in range(n):
            modified_matrix = np.copy(matrix)
            modified_matrix[i] = identity_row
            modified_matrices.append(modified_matrix)
 
        return modified_matrices

    # Standard single variate Richardson Extrapolation
    def getRichardsonZNE2(self, data) -> Dict:
        """
        Perform single-variable Richardson Extrapolation to estimate the zero-noise value.
        Beta coefficients are calculated using the product formula:
            beta_k = prod_{i != k} (alpha_i / (alpha_k - alpha_i))

        For further datails refer to: https://doi.org/10.48550/arXiv.2210.00921


        Args:
            data (list of lists): Each data point contains independent noise variables followed by the energy value.

        Returns:
            float: Zero-noise extrapolated value.
            dict: Contains some other detals
        """
        # Step 1: Extract total noise and corresponding expectation values
        total_noise = [sum(point[: self.independent_var_number]) for point in data]
        expectation_vals = [point[-1] for point in data]

        # Step 2: Sort data based on total noise in ascending order
        sorted_pairs = sorted(zip(total_noise, expectation_vals), key=lambda pair: pair[0])
        sorted_total_noise, sorted_expectation_vals = map(list, zip(*sorted_pairs))  # Convert tuples to lists

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

        # Compute cost_error_mitigation
        cost_error_mitigation = sum(beta * beta for beta in betas)

        result = {
            "extrapolated_val": zne_val,
            "richardson_steps_details": {
                # For the single variate ZNE, user input degree is discarded, and the extrapolation is computed based off the numbe of data point provided. 
                # Hence, degree is set to len(sorted_total_noise) - 1
                "true_degree": len(sorted_total_noise) - 1,  
                "sorted_noise": sorted_total_noise,
                "sorted_expectation_vals": sorted_expectation_vals,
                "beta_coefficients": betas,
                "cost_richardson_zne": cost_error_mitigation,
            },
        }
        return result

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

        # Sample the data
        sampled_data = self.sampling()

        if self.method.lower() == "richardson-mul":
            result = self.mul_RichardsonZNE(data=sampled_data)
            zne_extrapolated_val = result["extrapolated_value"]
            richardson_steps_details = result["richardson_steps_details"]
            zne_val = {
                "degree": self.degree,
                "sampling": self.sampling_mode,
                "sampled data": sampled_data,
                "extrapolated_value": zne_extrapolated_val,
                "others": richardson_steps_details
            }

        elif self.method.lower() == "richardson":
            result: Dict = self.getRichardsonZNE2(data=sampled_data)
            zne_extrapolated_val: float = result["extrapolated_val"]
            richardson_step_details: dict = result["richardson_steps_details"]
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
                f"Invalid method: {self.method}. Valid methods are: richardson-mul, richardson, linear, and polynomial."
            )

        return zne_val
