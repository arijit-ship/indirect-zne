# Indirect-ZNE
----

Quantum error mitigation with Zero Noise Extrapolation (ZNE) approach for indirect controlled system.

## Configuration File

| Parameters | Explanation |
|------------|--------------|
| `run` | type: `str`, what should be run |
| `nqubits` | type: `int`, number of qubits |
| `state` | type: `str`, Density matrix or state vector formalism, acceptable values: `dmatrix`, `statevector` |
| `observable.def` | type: `str`, the Hamiltonian model to be used, acceptable values: `ising` |
| `observable.coefficients.cn` | type: `list[float]`, coefficients for the `cn` term |
| `observable.coefficients.bn` | type: `list[float]`, coefficients for the `bn` term |
| `observable.coefficients.r` | type: `float`, coefficient for the `r` term |
| `output.file_name_prefix` | type: `str`, prefix for the output file names |
| `output.fig_dpi` | type: `int`, DPI for output figures |
| `vqe.optimization.status` | type: `bool`, minimizes the cost function with respect to parameters |
| `vqe.optimization.algorithm` | type: `str`, minimization algorithm used by `scipy.optimize.minimize()`, acceptable values: `SLSQP`, `BFGS`, etc. See the SciPy official documentation for more details |
| `vqe.optimization.iteration` | type: `int`, how many times the optimization is performed |
| `vqe.optimization.constraint` | type: `bool`, applies time constraint to parameters, supported only by the `SLSQP` algorithm |
| `vqe.ansatz.draw` | type: `bool`, draws the circuit figure |
| `vqe.ansatz.type` | type: `str`, the ansatz type to be used, acceptable values: `xy`, `ising`, `hardware` |
| `vqe.ansatz.layer` | type: `int`, number of layers in the ansatz |
| `vqe.ansatz.gateset` | type: `int`, type of gateset used |
| `vqe.ansatz.ugate.coefficients.cn` | type: `list[float]`, coefficients for the `cn` term of the unitary gate |
| `vqe.ansatz.ugate.coefficients.bn` | type: `list[float]`, coefficients for the `bn` term of the unitary gate |
| `vqe.ansatz.ugate.coefficients.r` | type: `float`, coefficient for the `r` term of the unitary gate |
| `vqe.ansatz.ugate.time.min` | type: `float`, minimum time for the unitary gate |
| `vqe.ansatz.ugate.time.max` | type: `float`, maximum time for the unitary gate |
| `vqe.ansatz.noise.status` | type: `bool`, turn on/off the noise |
| `vqe.ansatz.noise.value` | type: `list[float]`, noise probabilities for different gates `[r, cz, u, y]` |
| `zne.extrapolation.method` | type: `str`, extrapolation method to use, acceptable value: `Richardson` |
| `zne.extrapolation.degrees` | type: `list[int]`, degrees of the extrapolation method |
| `zne.extrapolation.sampling` | type: `str`, sampling method, acceptable value: `default` |
| `zne.redundant_ansatz.identity_factors` | type: `list[list[int]]`, identity factors for the redundant ansatz |
| `param` | type: `list[float]` or `str` (`random`), initial parameters for the ansatz |

## Example

```yaml
run: "zne"
# System
nqubits: 7
state: "dmatrix"
# Target Hamiltonian
observable:
  def: "ising"
  coefficients:
    cn: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    bn: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    r: 0
# Output
output:
  file_name_prefix: "vqe_vals_q7_l30_n2"
  fig_dpi: 100
# Variational quantum eigensolver config
vqe:
  # Optimization
  optimization:
    status: True
    algorithm: "SLSQP"
    iteration: 5
    constraint: False
  # Circuit config
  ansatz:
    draw: True # Draw the circuit
    type: "xy"
    layer: 30
    gateset: 1
    ugate: 
      coefficients:
        cn: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        bn: [0, 0, 0, 0, 0, 0, 0]
        r: 0
      time:
        min: 0.0
        max: 10.0
    noise:
      status: True
      value: [0.0001, 0.0001, 0.0001, 0.0001] # Noise probabilities for [r, cz, u, y] gates.
# Zero noise extrapolation config
zne:
  extrapolation:
    method: "Richardson"
    degrees: [1, 2, 3, 4, 5]
    sampling: "default"
  redundant_ansatz:
    # Identity factors for the redundant ansatz
    identity_factors: [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0], [2, 1, 0], [2, 1, 1], [2, 2, 0], [2, 2, 1], [2, 2, 2], 
    [3, 0, 0], [3, 1, 0], [3, 1, 1], [3, 2, 0], [3, 2, 1], [3, 2, 2], [3, 3, 0], [3, 3, 1], [3, 3, 2], 
    [3, 3, 3], [4, 0, 0], [4, 1, 0], [4, 1, 1], [4, 2, 0], [4, 2, 1], [4, 2, 2], [4, 3, 0], [4, 3, 1], 
    [4, 3, 2], [4, 3, 3], [4, 4, 0], [4, 4, 1], [4, 4, 2], [4, 4, 3], [4, 4, 4], [5, 0, 0], [5, 1, 0], 
    [5, 1, 1], [5, 2, 0], [5, 2, 1], [5, 2, 2], [5, 3, 0], [5, 3, 1], [5, 3, 2], [5, 3, 3], [5, 4, 0], 
    [5, 4, 1], [5, 4, 2], [5, 4, 3], [5, 4, 4], [5, 5, 0], [5, 5, 1], [5, 5, 2], [5, 5, 3], [5, 5, 4], 
    [5, 5, 5]
]
 # Initial parameters for the ansatz
param:  "random"