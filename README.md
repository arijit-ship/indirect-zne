# indirect-zne
----

Quantum error mitigation with Zero Noise Extrapolation (ZNE) approach for indirect controlled system.

## Configuration file


|Parameters| Explanation  |
|--|--|
| `nqubit` |type: `int`, number of qubits. |
|`state`|type: `str`, Density matrix or state vector formalism, acceptable values: `DMatrix`, `Satevector`|
|`layer`|type: `int`, depth of the circuit. |
|`etime`|type: `bool`, calculates the execution time for eigen solver. |
|`optimization.status`|type: `bool`, minimizes the cost function with respect to parameters. |
|`optimization.algorith`|type:`str`, minimization algorithm used by `scipy.optimize.minimize()`, acceptable values: `SLSQP`, `BFGS` etc. See the SciPy official documentation for more details. |
|`optimization.iteration`|type: `int`, how many times the optimization is performed. |
|`optimization.constraint`|type: `bool`, applies time constraint to parameters, supported by only `SLSQP` algorithm. |
|`circuit.draw`|type:`bool`, draws the circuit figure. |
|`circuit.ansatztype`|type: `str`, the Hamiltonian model to be used in time evolution gate, acceptable values: `xy`,`ising`, `hardware`|
|`circuit.time.min`|type: `float`, lower bound of time duration in the time-evolution gate. |
|`circuit.time.max`|type: `float`, upper bound of time duration in the time-evolution gate. |
|`circuit.coefficients.cn`|type:`float`|
|`circuit.coefficients.bn`|type: `float`|
|`circuit.coefficient.r`|type: `float`|
|`circuit.noise.status`|type: `bool`, turn on/off the noise. |
|`circuit.noise.value`|type: `list[float]`, noise probabilities for rotational gates, time-evolution gate, and Y gate respectively. |
|`circuit.noise.factor`|type: `list[int]`, noise factors for rotational gates, time-evolution gate, and Y gate respectively |
|`param`|type: `list[float]` and `str`, initial parameters to be used. |

