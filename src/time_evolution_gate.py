import numpy as np
from qulacs import Observable
from qulacs.gate import *

# Global variables to store the eigenvalues and eigenvectors
diag = None
eigen_vecs = None


def create_time_evo_unitary(observable: Observable, ti: float, tf: float):
    """
    Args:
        observable: qulacs observable
        ti: initial time
        tf: final time

    Returns:
        a dense matrix gate U(t) = exp(-iHt)
    """
    # Get the qubit number
    n = observable.get_qubit_count()
    # Converting to a matrix
    H_mat = observable.get_matrix().toarray()

    # Compute eigenvalues and eigenvectors only once and reuse them
    global diag, eigen_vecs

    if diag is None or eigen_vecs is None:
        diag, eigen_vecs = np.linalg.eigh(H_mat)

    # Compute the exponent of diagonalized Hamiltonian
    exponent_diag = np.diag(np.exp(-1j * (tf - ti) * diag))

    # Construct the time evolution operator
    time_evol_op = np.dot(np.dot(eigen_vecs, exponent_diag), eigen_vecs.T.conj())

    return DenseMatrix([i for i in range(n)], time_evol_op)
