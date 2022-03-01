import numpy as np
import scipy


def compute_density_matrix(molecular_orbitals, n_occup_orbitals):

    n_basis_functions = len(molecular_orbitals)
    density_matrix = np.zeros((n_basis_functions, n_basis_functions))

    for i in range(n_occup_orbitals):
        density_matrix = 2.0 * np.outer(molecular_orbitals[i], molecular_orbitals[i])

    return density_matrix


def compute_JK(density_matrix, Vee):

    n_basis_functions = len(density_matrix)
    J = np.zeros((n_basis_functions, n_basis_functions))
    K = np.zeros((n_basis_functions, n_basis_functions))

    def J_element(i, j):
        J_sum = 0.0
        K_sum = 0.0

        for k in range(n_basis_functions):
            for l in range(n_basis_functions):
                density = density_matrix[k, l]
                J = Vee[i, j, k, l]
                K = Vee[i, l, k, j]
                J_sum += density * J
                K_sum += density * K

        return J_sum/2, K_sum/2  # handle double counting

    for i in range(n_basis_functions):
        for j in range(n_basis_functions):
            J[i, j], K[i, j] = J_element(i, j)

    return J, K


def generalized_eig(matrix, overlap):
    """
    Solve generalized eigenvalue problem : F C = e S C

    :param matrix: F square matrix
    :param overlap: S symmetric matrix
    :return: e, C
    """

    overlap_inv = np.linalg.inv(overlap)
    overlap_inv_sqrt = scipy.linalg.sqrtm(overlap_inv)

    matrix_sbasis = np.dot(overlap_inv_sqrt, np.dot(matrix, overlap_inv_sqrt))
    eigenvalues, eigenvectors_sbasis = np.linalg.eigh(matrix_sbasis)

    eigenvectors = np.dot(overlap_inv_sqrt, eigenvectors_sbasis).T

    return eigenvalues, eigenvectors


def scf_cycle(basis_set, S, T, Vne, Vee, tolerance=1e-5, max_scf_steps=20, extra_output=False):
    """
    Solve the SCF cycle

    :param basis_set: basis set used to represent electronic structure
    :param S: overlap matrix
    :param T: kinetic energy matrix
    :param Vne: electron-nuclear interaction matrix
    :param Vee: electron-electron interaction matrix
    :param tolerance: convergence criteria
    :param max_scf_steps: maximum number of SCF steps
    """

    # 1. Initialize density matrix
    n_basis_functions = len(basis_set)
    density_matrix = np.zeros((n_basis_functions, n_basis_functions))

    electronic_energy_ref = 0.0
    for scf_steps in range(max_scf_steps):

        # 2. Compute the 2-electron term add it to the 1-electron term
        J, K = compute_JK(density_matrix, Vee)
        G = 2 * J - K

        # 3. Define Fock matrix
        F = T + Vne + G

        # 4. Solve F C = e S C
        mo_energies, molecular_orbitals = generalized_eig(F, S)

        # 5. Generate new density matrix using MOs
        density_matrix = compute_density_matrix(molecular_orbitals, n_occup_orbitals=1)

        # 6. Compute electronic energy expectation value
        electronic_energy = np.sum(density_matrix * (T + Vne + J - 0.5 * K))

        print('Energy: ', electronic_energy)

        # 7. Check convergence
        if abs(electronic_energy - electronic_energy_ref) < tolerance:
            print('SCF cycle converged after {} steps'.format(scf_steps+1))
            if extra_output:
                return electronic_energy, density_matrix, molecular_orbitals
            return electronic_energy

        electronic_energy_ref = electronic_energy

    raise Exception('Convergence not met')

