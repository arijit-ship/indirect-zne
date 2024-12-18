import numpy as np
from qulacs import Observable
from qulacs.gate import DenseMatrix

# Global variables to store the eigenvalues and eigenvectors
diag = None
eigen_vecs = None


def create_time_evo_unitary(hamiltonian: Observable, ti: float, tf: float):
    """
    Args:
        hamiltonian: qulacs obsevable
        ti: initial time
        tf: final time

    Returns:
        a dense matrix gate U(t) = exp(-iHt)
    """
    # Get the qubit number
    n = hamiltonian.get_qubit_count()
    # Converting to a matrix
    H_mat = hamiltonian.get_matrix().toarray()

    # Compute eigenvalues and eigenvectors only once and reuse them
    global diag, eigen_vecs

    if diag is None or eigen_vecs is None:
        # print("gloabl diag not found")
        diag, eigen_vecs = np.linalg.eigh(H_mat)

    # Compute the exponent of diagonalized Hamiltonian
    exponent_diag = np.diag(np.exp(-1j * (tf - ti) * diag))

    # Construct the time evolution operator
    time_evol_op = np.dot(np.dot(eigen_vecs, exponent_diag), eigen_vecs.T.conj())

    return DenseMatrix([i for i in range(n)], time_evol_op)
