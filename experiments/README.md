# About

Experimental findings can be found in this directory.

# 1. Table of contents

Experimental findings can be found in this directory.

|Description|Link  |
|--|--|
| Noise-free XY-ansatz experiment: Richardson (2-point linear) & exponential ZNE |[Jupyter notebook](experiment08/experimentBook.xyansatz.ipynb), [Raw data (JSON)](experiment08/experimental%20data/data/noisefree_time_evo_xy_noisefree) |
|Noise-free Ising-ansatz experiment: Richardson (2-point linear) & exponential ZNE|[Jupyter notebook](experiment08/experimentBook.isingansatz.ipynb), [Raw data (JSON)](experiment08/experimental%20data/data/noisefree_time_evo_ising_noisefree)|
|Noise-free Hysenberg-ansatz experiment: Richardson (2-point linear) & exponential ZNE|[Jupyter notebook](experiment08/experimentBook.heisenbergansatz.ipynb), [Raw data (JSON)](experiment08/experimental%20data/data/noisefree_time_evo_heisenberg_noisefree)|
|Noisy XY-ansatz experiment: Richardson (2-point linear)|[Jupyter notebook](experiment07/experimentBook.noisyxy.ipynb), [Raw data (JSON)](experiment07/experimental%20data/data)|
|Further studies- Heisenberg multivariate Richardson ZNE, 3-point nonlinear Richardson ZNE |[Jupyter notebook](experiment09/experimentBook.heisenbergansatz.ipynb), [Raw data (JSON)](experiment09/experimental%20data/data)|
|Code to reproduce experimental data (v0)|[indirect-zne](https://github.com/arijit-ship/indirect-zne/releases/tag/v0)|



# 2. Indirect-Control VQE and ZNE Error Mitigation: How to use the code

Download the code from v0 release.

## 🛠️ Installation

- **Python Version:** `3.11`  
- To install dependencies, run:  
  ```bash
  pip install -r requirements.txt
  ```

## ⚙️ Usage

To run the program, use:  
  ```bash
  python3 main.py <config.yml>
  ```

## 📋 Sample configuration file

The program uses a YAML configuration file to define its parameters. The following is a sample of cofig.yml file.

```yaml
# Run configurations: Choose from 'vqe', 'redundant', and 'zne'.
run: "redundant"

# System configuration
nqubits: 4
# State options: 'dmatrix' and 'statevector'
state: "dmatrix"

# Target Hamiltonian configuration
observable:
  # Definition options: 'custom', 'ising', or 'heisenberg'.
  # 'custom' and 'ising' are created using a Hamiltonian with terms XZ-Z (this is NOT any standard familier XY model, we call it 'Fancy XY-model Hamiltonian').
  def: "ising"
  # WARNING: Coefficients can be overwritten:
  # 'custom': Not overwritten.
  # 'ising': cn, bn, r are overwritten to [0.5], [1], 1.
  # 'heisenberg': Only cn is used (will NOT be overwritten); bn and r are not used.
  coefficients:
    cn: [0.5, 0.5, 0.5]
    bn: [1.0, 1.0, 1.0, 1.0]
    r: 1

# Output configuration
output:
  file_name_prefix: "test_experiment"
  draw:
    status: True
    fig_dpi: 100
    type: "png"

# (1) Variational Quantum Eigensolver (VQE) configuration
vqe:
  iteration: 1

  # Optimization configuration
  optimization:
    status: True
    algorithm: "SLSQP"
    constraint: False

  # Circuit configuration
  ansatz:
    # Type options: 'custom', 'xy-iss', 'ising', and 'heisenberg'.
    # 'custom' and 'ising' are created using a Hamiltonian with terms XZ-Z (this is NOT any standard familier XY model, we call it 'Fancy XY-model Hamiltonian').
    # WARNING: For ZNE redundant circuit, the ansatz type must be 'xy-iss' (stands for XY identity-scaling-supported).
    type: "xy-iss"
    layer: 1
    gateset: 1
    ugate:
      # WARNING: Coefficients can be overwritten:
      # 'custom': Not overwritten.
      # 'xy-iss': cn, bn, r are overwritten to [0.5], [0], 0.
      # 'ising': cn, bn, r are overwritten to [0.5], [1], 1.
      # 'heisenberg': Only cn is used (will NOT be overwritten); bn and r are not used.
      coefficients:
        cn: [0.5, 0.5, 0.5]
        bn: [0, 0, 0, 0]
        r: 0
      time:
        min: 0.0
        max: 10.0
    noise:
      status: True
      value: [0, 0, 0, 0] # Noise probabilities for [r, cz, u, y] gates.

    # Initial parameters for the ansatz: "random" or "List[float]"
    init_param: "random"


# (2) Redundant circuit configuration
  # Identity factors for r, U, Y, and CZ gates
  # WARNING: Identity scaling for the U gate is only possible if vqe.ansatz.type is 'xy-iss'. For other types, the identity factor for the U gate must be set to 0.
identity_factors: [
    [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]
]

# (3) Zero noise extrapolation (ZNE) configuration
zne:
  # Method options: 'linear', 'polynomial', 'richardson', or 'richardson-mul'.
  # 'linear' and 'polynomial' use scikit-learn for regression.
  method: "richardson"

  # Degree is only applicable for 'polynomial' and 'richardson-mul'. For 'richardson', degree is computed based on the number of data points.
  degree: 1

  # Sampling method options: 'default', 'default-N', or 'random-N', where N is an integer.
  # 'default' - samples all points.
  # 'default-N' - samples the first N points.
  # 'random-N' - samples N points randomly.
  sampling: "default-2"
  
  # Data points for extrapolation
  data_points:  [[12, 1, 0, 3, -3.4161775440796838], [20, 7, 10, 3, -0.05166483365312464], [20, 3, 18, 3, -0.24895356624222376], [28, 9, 6, 7, -0.07447160728654578], [12, 5, 34, 3, -0.0010315076068858672], [12, 5, 26, 1, -0.005839864429826224]]
  ```



Below is a detailed description of the configuration categories and their parameters:

| Section                         | Key                          | Type            | Values / Example                  | Description |
|----------------------------------|------------------------------|----------------|----------------------------------|-------------|
| Run configurations              | `run`                        | `String`       | `"vqe"`, `"redundant"`, `"zne"`  | Specifies the type of run to execute. |
| System configuration            | `nqubits`                    | `Integer`      | `7`                              | Number of qubits in the system. |
|                                  | `state`                      | `String`       | `"dmatrix"`, `"statevector"`     | Defines the quantum state representation. |
| Target Hamiltonian configuration | `observable.def`             | `String`       | `"custom"`, `"ising"`, `"heisenberg"` | Specifies the type of Hamiltonian to use. |
|                                  | `observable.coefficients.cn` | `List[Float]`  | `[0.5, 0.5, ...]`                 | Coefficients for Hamiltonian terms. |
|                                  | `observable.coefficients.bn` | `List[Float]`  | `[1.0, 1.0, ...]`                 | Coefficients for Hamiltonian terms. |
|                                  | `observable.coefficients.r`  | `Float`      | `1.0`                              | Coefficients for Hamiltonian terms. |
| Output configuration            | `output.file_name_prefix`    | `String`       | `"noisefree_time_evo_..."`       | Prefix for output file names. |
|                                  | `output.draw.status`         | `Boolean`      | `True` / `False`                 | Enables or disables circuit drawing. |
|                                  | `output.draw.fig_dpi`        | `Integer`      | `100`                            | DPI resolution for output figures. |
|                                  | `output.draw.type`           | `String`       | `"png"`                          | Image format for output figures. |
| VQE configuration               | `vqe.iteration`              | `Integer`      | `30`                             | Number of optimization iterations. |
|                                  | `vqe.optimization.status`    | `Boolean`      | `True` / `False`                 | Enables or disables optimization. |
|                                  | `vqe.optimization.algorithm` | `String`       | `"SLSQP"`                        | Optimization algorithm used. |
|                                  | `vqe.optimization.constraint` | `Boolean`     | `True` / `False`                 | Specifies whether constraints are used. |
|                                  | `vqe.ansatz.type`           | `String`       | `"custom"`, `"xy-iss"`, `"ising"`, `"heisenberg"` | Ansatz type for variational circuits. |
|                                  | `vqe.ansatz.layer`          | `Integer`      | `30`                             | Number of layers in the ansatz circuit. |
|                                  | `vqe.ansatz.gateset`        | `Integer`      | `1`                              | Specifies the gate set used. |
|                                  | `vqe.ansatz.ugate.coefficients.cn` | `List[Float]` | `[0.5, 0.5, ...]`  | Coefficients for time-evolution gate terms. |
|                                  | `vqe.ansatz.ugate.coefficients.bn` | `List[Float]` | `[0, 0, ...]`       | Coefficients for time-evolution gate terms. |
|                                  | `vqe.ansatz.ugate.coefficients.r`  | `Float`      | `0`                              | Coefficients for time-evolution gate terms. |
|                                  | `vqe.ansatz.ugate.time.min`  | `Float`       | `0.0`                            | Initial evolution time. |
|                                  | `vqe.ansatz.ugate.time.max`  | `Float`       | `10.0`                           | Final evolution time. |
|                                  | `vqe.ansatz.noise.status`    | `Boolean`     | `True` / `False`                 | Enables or disables noise. |
|                                  | `vqe.ansatz.noise.value`     | `List[Float]`  | `[0.001, 0.001, 0, 0]`           | Noise probabilities for different gates. |
|                                  | `vqe.ansatz.init_param`      | `List[Float]`  | `[1.1864, -0.9628, ...]`         | Initial parameters for ansatz. |
| Redundant circuit configuration | `identity_factors`           | `List[List[Int]]` | `[[0,0,0,0], [1,0,0,1], [2,0,0,1]]` | Identity factors for different gates. |
| Zero noise extrapolation (ZNE)   | `zne.method`                 | `String`       | `"linear"`, `"polynomial"`, `"richardson"`, `"richardson-mul"` | Extrapolation method. |
|                                  | `zne.degree`                 | `Integer`      | `1`                              | Polynomial degree for extrapolation. |
|                                  | `zne.sampling`               | `String`       | `"default"`, `"default-N"`, `"random-N"` | Sampling strategy. |
|                                  | `zne.data_points`            | `List[List[Float]]` | `[[4,1,-6.7], [12,3,-5.1], [20,3,-4.0]]` | Data points used for extrapolation. |



## ⚠️ Warnings and Important Notes

1. **Observable (Target Hamiltonian) Coefficients Overwritten**:
    - The coefficients are automatically overwritten based on the `observable.def` as follows: 
        - `custom`: Not overwritten.
        - `ising`: cn, bn, r are overwritten to `[0.5]`, `[1]`, `1`.
        - `heisenberg`: Only cn is used (will NOT be overwritten); bn and r are not used.
  
    Note: 'custom' and 'ising' are created using a Hamiltonian with terms XZ-Z (this is NOT any standard familier XY model, we call it 'Fancy XY-model Hamiltonian').
   

2. **Ansatz Type for Zero Noise Extrapolation (ZNE)**:
   - When using Zero Noise Extrapolation (ZNE) with redundant circuits, the ansatz type must be set to `'xy-iss'` (`vqe.ansatz.type: "xy-iss"`). This is required for identity-scaling the circuit gates (U, Y, and CZ).
   - **Warning**: The identity-scaling for the U gate is only supported if the ansatz type is `'xy_model-xz-z'`. For other ansatz types, the identity scaling for the U gate must be set to `0`.

3. **VQE Ansatz Coefficients Overwritten**:
   - The coefficients are automatically overwritten based on the `vqe.ansatz.type` as follows:
      - `custom`: Not overwritten.
      - `xy-iss`: cn, bn, r are overwritten to [0.5], [0], 0.
      - `ising`: cn, bn, r are overwritten to [0.5], [1], 1.
      - `heisenberg`: Only cn is used (will NOT be overwritten); bn and r are not used.

    Note: `custom` and `ising` are created using a Hamiltonian with terms XZ-Z (this is NOT any standard familier XY model, we call it 'Fancy XY-model Hamiltonian'). 

4. **Initialization of Parameters**:
   - The initial parameters for the ansatz are set to `random` by default (`vqe.ansatz.init_param: "random"`). If you need to modify the initialization method, ensure to update this value accordingly.

5. **Sampling Method for Zero Noise Extrapolation**:
   - The `sampling` method should be selected appropriately:
     - `'default'` - All points are sampled.
     - `'default-N'` - The first `N` points are sampled.
     - `'random-N'` - `N` points are sampled randomly.
   - Ensure that the method aligns with the desired sampling strategy for extrapolation.


# 3. Sample config.yml file


