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

| **Section**                        | **Key**                           | **Type**                             | **Description**                                                                                                                                                                                                                     |
|-------------------------------------|-----------------------------------|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Run Configuration**               | `run`                             | String                               | Defines the algorithm to run. Options: `'vqe'`, `'redundant'`, `'zne'`.                                                                                                                                                            |
| **System Configuration**            | `nqubits`                         | Integer                              | Specifies the number of qubits in the quantum system.                                                                                                                                                                             |
|                                     | `state`                           | String                               | Defines the state representation. Options: `'dmatrix'` (density matrix) or `'statevector'`.                                                                                                                                      |
| **Observable (Target Hamiltonian)** | `def`                             | String                               | Defines the type of Hamiltonian. Options: `'custom'`, `'ising'`, `'heisenberg'`. **Warning**: Coefficients are overwritten based on the selected type. For `'ising'`, `cn` values are set to `[0.5]`, `bn` values are set to `[1]`, and `r` is set to `1`. For `'heisenberg'`, only `cn` is used. |
|                                     | `coefficients`                    | Object                               | Contains coefficients for the Hamiltonian terms:                                                                                                                                                                                                 |
|                                     |                                   | `cn`                               | List of Floats                        | Defines the coefficients for the interaction term. Example: `[0.5, 0.5, 0.5]`.                                                                                                                                                       |
|                                     |                                   | `bn`                               | List of Floats                        | Defines the coefficients for the coupling terms. Example: `[1.0, 1.0, 1.0, 1.0]`.                                                                                                                                                   |
|                                     |                                   | `r`                                | Float                                | Defines the coefficient for the scaling term. Example: `1`.                                                                                                                                                                          |
| **Output Configuration**            | `file_name_prefix`                | String                               | Specifies the prefix for the output file name.                                                                                                                                                                                      |
|                                     | `draw`                            | Object                               | Configures drawing options for output files:                                                                                                                                                                                                 |
|                                     |                                   | `status`                           | Boolean                              | If `True`, enables figure drawing. Example: `False`.                                                                                                                                                                                   |
|                                     |                                   | `fig_dpi`                          | Integer                              | Specifies the resolution of the output figure. Example: `100`.                                                                                                                                                                        |
|                                     |                                   | `type`                             | String                               | Specifies the type of output file. Example: `"png"`.                                                                                                                                                                                 |
| **VQE Configuration**               | `iteration`                       | Integer                              | Specifies the number of iterations for the VQE algorithm.                                                                                                                                                                         |
|                                     | `optimization`                    | Object                               | Configures optimization settings:                                                                                                                                                                                                      |
|                                     |                                   | `status`                           | Boolean                              | If `True`, enables optimization. Example: `True`.                                                                                                                                                                                       |
|                                     |                                   | `algorithm`                        | String                               | Specifies the optimization algorithm. Options: `"SLSQP"`.                                                                                                                                                                           |
|                                     |                                   | `constraint`                       | Boolean                              | If `True`, enables constraint optimization. Example: `False`.                                                                                                                                                                         |
|                                     | `ansatz`                          | Object                               | Configures the ansatz circuit:                                                                                                                                                                                                         |
|                                     |                                   | `type`                             | String                               | Defines the type of the ansatz. Options: `'custom'`, `'xy-iss'`, `'ising'`, `'heisenberg'`. **Warning**: Must be `'xy-iss'` for ZNE redundant circuits.                                                                                                                                      |
|                                     |                                   | `layer`                            | Integer                              | Specifies the number of layers for the ansatz. Example: `10`.                                                                                                                                                                         |
|                                     |                                   | `gateset`                          | Integer                              | Specifies the gate set used. Example: `1`.                                                                                                                                                                                             |
|                                     |                                   | `ugate`                            | Object                               | Defines the U gate settings:                                                                                                                                                                                                         |
|                                     |                                   |                                   | `coefficients`                      | Object                               | Contains coefficients for the U gate:                                                                                                                                                                                                   |
|                                     |                                   |                                   | `cn`                               | List of Floats                        | Defines the coefficients for the interaction terms in the U gate. Example: `[0.5, 0.5, 0.5]`.                                                                                                                                      |
|                                     |                                   |                                   | `bn`                               | List of Floats                        | Defines the coefficients for the coupling terms in the U gate. Example: `[0, 0, 0, 0]`.                                                                                                                                           |
|                                     |                                   |                                   | `r`                                | Float                                | Defines the coefficient for the scaling term in the U gate. Example: `0`.                                                                                                                                                            |
|                                     |                                   |                                   | `time`                             | Object                               | Defines the time range for the gate:                                                                                                                                                                                                 |
|                                     |                                   |                                   | `min`                              | Float                                | Minimum time. Example: `0.0`.                                                                                                                                                                                                           |
|                                     |                                   |                                   | `max`                              | Float                                | Maximum time. Example: `10.0`.                                                                                                                                                                                                         |
|                                     |                                   | `noise`                            | Object                               | Defines noise parameters:                                                                                                                                                                                                               |
|                                     |                                   |                                   | `status`                           | Boolean                              | If `True`, enables noise. Example: `True`.                                                                                                                                                                                              |
|                                     |                                   |                                   | `value`                            | List of Floats                        | Specifies noise probabilities for different gate types in the order `[r, cz, u, y]`. Example: `[0.001, 0.01, 0.001, 0.01]`.                                                                                                       |
|                                     | `init_param`                      | String                               | Defines the initialization method for parameters. Options: `'random'`, etc. Example: `"random"`.                                                                                                               |
| **Redundant Circuit Configuration** | `identity_factors`                | List of Lists                        | Specifies the identity scaling factors for gates. Example: `[[1, 0, 0, 1], [2, 3, 2, 1], ...]`.                                                                                                                                       |
|                                     | **Warning**                       | String                               | **Warning**: Identity scaling for the U gate is supported only if the ansatz type is `'xy-iss'`. For other types, the identity scaling factor for the U gate must be set to `0`.                                        |
| **Zero Noise Extrapolation (ZNE)**  | `method`                          | String                               | Defines the method for zero-noise extrapolation. Options: `'linear'`, `'polynomial'`, `'richardson'`, `'richardson-mul'`.                                                                                                       |
|                                     | `degree`                          | Integer                              | Sets the degree for polynomial or Richardson methods. Used for regression.                                                                                                                                                         |
|                                     | `sampling`                        | String                               | Specifies the sampling method for extrapolation. Options: `'default'`, `'default-N'`, `'random-N'`, where `N` is an integer.                                                                                                                                 |
|                                     | `data_points`                     | List of Lists                        | Provides the data points for extrapolation. Each entry contains several values, such as time and noise levels. Example: `[[12, 1, 0, 3, -3.3480294367352315], [20, 7, 10, 3, -0.05316450776222178], ...]`.                                   |


## Warnings and Important Notes

1. **Observable (Target Hamiltonian) Coefficients Overwritten**:
   - For the `ising` Hamiltonian (`observable.def: "ising"`), the coefficients are automatically overwritten as follows:
     - `cn`: `[0.5]`
     - `bn`: `[1.0]`
     - `r`: `1`
   - For the `heisenberg` Hamiltonian (`observable.def: "heisenberg"`), only the `cn` coefficient is used, and `bn` and `r` are ignored.

2. **Ansatz Type for Zero Noise Extrapolation (ZNE)**:
   - When using Zero Noise Extrapolation (ZNE) with redundant circuits, the ansatz type must be set to `'xy-iss'` (`vqe.ansatz.type: "xy-iss"`). This is required for identity-scaling the circuit gates (U, Y, and CZ).
   - **Warning**: The identity-scaling for the U gate is only supported if the ansatz type is `'xy_model-xz-z'`. For other ansatz types, the identity scaling for the U gate must be set to `0`.

3. **VQE Ansatz Coefficients Overwritten**:
   - When using the VQE algorithm with certain ansatz types (`'xy-iss'`), the coefficients are automatically overwritten as follows:
     - `cn`: `[0.5]`
     - `bn`: `[0]`
     - `r`: `0`
   - For ansatz types such as `'ising'` or `'heisenberg'`, the coefficients are predefined according to the selected type.

4. **Initialization of Parameters**:
   - The initial parameters for the ansatz are set to `random` by default (`vqe.ansatz.init_param: "random"`). If you need to modify the initialization method, ensure to update this value accordingly.

5. **Sampling Method for Zero Noise Extrapolation**:
   - The `sampling` method should be selected appropriately:
     - `'default'` - All points are sampled.
     - `'default-N'` - The first `N` points are sampled.
     - `'random-N'` - `N` points are sampled randomly.
   - Ensure that the method aligns with the desired sampling strategy for extrapolation.


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
