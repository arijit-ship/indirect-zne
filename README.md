# Indirect-Control VQE and ZNE Error Mitigation

## Installation

- **Python Version:** `3.11`  
- To install dependencies, run:  
  ```bash
  pip install -r requirements.txt
  ```

## Usage

To run the program, use:  
  ```bash
  python3 main.py <config.yml>
  ```

## Configuration Details

The program uses a YAML configuration file to define its parameters. Below is a detailed description of the configuration categories and their parameters:

| **Category**                       | **Parameter**                    | **Type**                                                                                 | **Description**                                                                                                   |
|-------------------------------------|----------------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **Run Configuration**               | `run`                            | `str`                                                                                    | Specifies the method to run: `"vqe"`, `"redundant"`, or `"zne"`.                                                  |
| **System Configuration**           | `nqubits`                        | `int`                                                                                    | Number of qubits in the system.                                                                                   |
|                                     | `state`                          | `str`                                                                                    | State representation type, e.g., `"dmatrix"`.                                                                     |
| **Target Hamiltonian**             | `observable.def`                 | `str`                                                                                    | Type of Hamiltonian: `"custom"`, `"ising"`, `"xy_model-xz-y"`, or `"heisenberg"`.                                  |
|                                     | `observable.coefficients.cn`     | `list[float]`                                                                            | Coefficients for the custom Hamiltonian (only for `def: custom`).                                                 |
|                                     | `observable.coefficients.bn`     | `list[float]`                                                                            | Additional coefficients for the custom Hamiltonian (only for `def: custom`).                                       |
|                                     | `observable.coefficients.r`      | `float`                                                                                 | A coefficient used in the custom Hamiltonian (only for `def: custom`).                                             |
| **Output Configuration**           | `file_name_prefix`               | `str`                                                                                    | Prefix for the output file name.                                                                                   |
|                                     | `draw.status`                    | `bool`                                                                                   | Whether to draw the circuit diagram (`True`/`False`).                                                              |
|                                     | `draw.fig_dpi`                   | `int`                                                                                    | DPI (dots per inch) setting for the output figure.                                                                 |
|                                     | `draw.type`                      | `str`                                                                                    | File type for the output figure, e.g., `"png"` or `"svg"`.                                                         |
| **VQE Configuration**              | `vqe.iteration`                  | `int`                                                                                    | Number of VQE iterations to perform.                                                                               |
|                                     | `optimization.status`            | `bool`                                                                                   | Whether the optimization should be performed (`True`/`False`).                                                     |
|                                     | `optimization.algorithm`         | `str`                                                                                    | Optimization algorithm to use, e.g., `"SLSQP"`.                                                                    |
|                                     | `optimization.constraint`        | `bool`                                                                                   | Whether to apply constraints during optimization (`True`/`False`).                                                 |
|                                     | `ansatz.type`                    | `str`                                                                                    | Type of ansatz to use, e.g., `"xy_model-xz-y"`.                                                                    |
|                                     | `ansatz.layer`                   | `int`                                                                                    | Number of layers in the ansatz circuit.                                                                            |
|                                     | `ansatz.gateset`                 | `int`                                                                                    | Number of gates in each layer of the ansatz.                                                                       |
|                                     | `ansatz.ugate.coefficients.cn`   | `list[float]`                                                                            | Coefficients for the U gate in the ansatz, e.g., `[0.5, 0.5, 0.5]`.                                                |
|                                     | `ansatz.ugate.coefficients.bn`   | `list[float]`                                                                            | Additional coefficients for the U gate in the ansatz.                                                              |
|                                     | `ansatz.ugate.coefficients.r`    | `float`                                                                                 | The "r" coefficient for the U gate in the ansatz.                                                                  |
|                                     | `ansatz.ugate.time.min`          | `float`                                                                                 | Minimum time for the U gate in the ansatz.                                                                         |
|                                     | `ansatz.ugate.time.max`          | `float`                                                                                 | Maximum time for the U gate in the ansatz.                                                                         |
|                                     | `ansatz.noise.status`            | `bool`                                                                                   | Whether noise is applied to the ansatz circuit (`True`/`False`).                                                   |
|                                     | `ansatz.noise.value`             | `list[float]`                                                                            | Noise values for specific gates: `[r, cz, u, y]` gates.                                                            |
|                                     | `ansatz.init_param`              | `list[float]`                                                                            | Initial parameters for the ansatz circuit (list of floats representing angles and times).                          |
| **Redundant Circuit Configuration**| `identity_factors`               | `list[list[int]]`                                                                        | Identity factors for the gates: `[r, U, Y, CZ]`.                                                                   |
| **Zero Noise Extrapolation (ZNE)** | `method`                         | `str`                                                                                    | Extrapolation method: `"linear"`, `"polynomial"`, `"richardson"`, or `"richardson-mul"`.                           |
|                                     | `degree`                         | `int`                                                                                    | Degree of the regression model (relevant for `"polynomial"` and `"richardson-mul"` methods).                        |
|                                     | `sampling`                       | `str`                                                                                    | Sampling method: `"default"`, `"default-N"`, or `"random-N"`.                                                      |
|                                     | `data_points`                    | `list[list[float]]`                                                                      | Data points for extrapolation (list of lists containing data point sets).                                           |

## Testing

For testing, use `pytest`.  
To run the tests, execute:  
  ```bash
  pytest test/.
  ```

## Linting and Formatting

- Use `flake8` for linting.  
- Use `black` and `isort` for formatting the code.  
  ```bash
  # Run linting
  flake8 .

  # Run formatting
  black .
  isort .
  ```
