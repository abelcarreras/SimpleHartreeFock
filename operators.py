import numpy as np
from basis import gaussian_product, boys


def overlap(electronic_structure):
    """
    Calculate the overlap matrix
    """
    S = [[bf1.get_overlap_with(bf2) for bf1 in electronic_structure] for bf2 in electronic_structure]
    return np.array(S)


def kinetic(molecule):
    """
    Calculate the kinetic energy matrix
    """
    T = [[bf1.get_kinetic_with(bf2) for bf1 in molecule] for bf2 in molecule]
    return np.array(T)


def electron_nuclear(electronic_structure, nuclear_coordinates, nuclear_charges):
    """
    Calculate the electron-nuclear interaction matrix
    """
    nbasis = len(electronic_structure)
    V = np.zeros([nbasis, nbasis])

    for position, charge in zip(nuclear_coordinates, nuclear_charges):
        V += np.array([[bf1.get_potential_with(bf2, position, charge)
                        for bf1 in electronic_structure] for bf2 in electronic_structure])
    return V


def electron_electron(electronic_structure):
    """
    Calculate the electron-electron interaction matrix
    """
    nbasis = len(electronic_structure)

    V_ee = np.zeros([nbasis, nbasis, nbasis, nbasis])

    def matrix_element(i_basisfunc, j_basisfunc, k_basisfunc, l_basisfunc):
        V_ee_element = 0
        for i_primitive, coeff_i in zip(i_basisfunc.primitive_gaussians, i_basisfunc.coefficients):
            for j_primitive, coeff_j in zip(j_basisfunc.primitive_gaussians, j_basisfunc.coefficients):
                for k_primitive, coeff_k in zip(k_basisfunc.primitive_gaussians, k_basisfunc.coefficients):
                    for l_primitive, coeff_l in zip(l_basisfunc.primitive_gaussians, l_basisfunc.coefficients):

                        coeff_ijkl = coeff_i * coeff_j * coeff_k * coeff_l

                        g_ij = gaussian_product(i_primitive, j_primitive)
                        g_kl = gaussian_product(k_primitive, l_primitive)

                        g_ijkl = gaussian_product(g_ij, g_kl)

                        PG = g_ij.coordinates - g_kl.coordinates
                        p = g_ij.alpha * g_kl.alpha
                        exponent = p/g_ijkl.alpha * np.dot(PG, PG)

                        factor = 2 * np.pi**2/p * g_ijkl.integrate**(1/3) * g_ijkl.prefactor**(2/3)

                        V_ee_element += coeff_ijkl * factor * np.exp(exponent) * boys(exponent, 0)

        return V_ee_element

    for i, i_basisfunc in enumerate(electronic_structure):
        for j, j_basisfunc in enumerate(electronic_structure):
            for k, k_basisfunc in enumerate(electronic_structure):
                for l, l_basisfunc in enumerate(electronic_structure):
                    V_ee[i, j, k, l] = matrix_element(i_basisfunc, j_basisfunc, k_basisfunc, l_basisfunc)

    return V_ee


def nuclear_nuclear(nuclear_coordinates, nuclear_charges):
    """
    Calculate the nuclear-nuclear interaction matrix
    """
    nuclear_coordinates = np.array(nuclear_coordinates)
    V_nn = 0
    for i, position in enumerate(nuclear_coordinates):
        for j, position2 in enumerate(nuclear_coordinates):
            if i > j:
                r = np.linalg.norm(position - position2)
                V_nn += nuclear_charges[i] * nuclear_charges[j] / r

    return V_nn


if __name__ == '__main__':
    from basis import PrimitiveGaussian, BasisFunction

    # STO-3G basis set for hydrogen molecule
    H_pg1a = PrimitiveGaussian(alpha=3.4252509140)
    H_pg1b = PrimitiveGaussian(alpha=0.6239137298)
    H_pg1c = PrimitiveGaussian(alpha=0.1688554040)

    H1_1s = BasisFunction([H_pg1a, H_pg1b, H_pg1c],
                          coefficients=[0.1543289673, 0.5353281423, 0.4446345411],
                          coordinates=[0, 0, 0])
    H2_1s = BasisFunction([H_pg1a, H_pg1b, H_pg1c],
                          coefficients=[0.1543289673, 0.5353281423, 0.4446345411],
                          coordinates=[1.6, 0, 0])


    molecule = [H1_1s, H2_1s]
    S = overlap(molecule)
    print(S)

    K = kinetic(molecule)
    print(K)

    V = electron_nuclear(molecule,
                         nuclear_coordinates=[[0, 0, 0],
                                              [1.4, 0, 0]],
                         nuclear_charges=[1, 1])
    print(V)

    print('-----electron-electron---------')

    ee = electron_electron(molecule)

    print(ee)

