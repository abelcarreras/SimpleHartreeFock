import sys
import numpy as np
from basis import gaussian_product, boys


def overlap(electronic_structure):

    S = [[bf1.get_overlap_with(bf2) for bf1 in electronic_structure] for bf2 in electronic_structure]
    return np.array(S)


def kinetic(molecule):

    T = [[bf1.get_kinetic_with(bf2) for bf1 in molecule] for bf2 in molecule]
    return np.array(T)


def electron_nuclear(electronic_structure, nuclear_coordinates, nuclear_charges):

    V = 0
    for position, charge in zip(nuclear_coordinates, nuclear_charges):
        V += np.array([[bf1.get_potential_with(bf2, position, charge)
                        for bf1 in electronic_structure] for bf2 in electronic_structure])
    return np.array(V)


def electron_electron(electronic_structure):
    """
    Calculate the overlap matrix of a molecule.
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

                        pref_ijkl = g_ij.prefactor * g_kl.prefactor

                        term_1 = 2 * np.pi**2 / (g_ij.alpha * g_kl.alpha)
                        term_2 = np.sqrt(np.pi / (g_ij.alpha + g_kl.alpha))

                        alpha_ijkl = (g_ij.alpha * g_kl.alpha)/(g_ij.alpha + g_kl.alpha)
                        PG = g_ij.coordinates - g_kl.coordinates

                        V_ee_element += coeff_ijkl * pref_ijkl * term_1 * term_2 * boys(alpha_ijkl * np.dot(PG, PG), 0)

        return V_ee_element

    for i, i_basisfunc in enumerate(electronic_structure):
        for j, j_basisfunc in enumerate(electronic_structure):
            for k, k_basisfunc in enumerate(electronic_structure):
                for l, l_basisfunc in enumerate(electronic_structure):
                    V_ee[i, j, k, l] = matrix_element(i_basisfunc, j_basisfunc, k_basisfunc, l_basisfunc)

    return V_ee


def nuclear_nuclear(nuclear_coordinates, nuclear_charges):
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


    exit()

    test1_2d = PrimitiveGaussian(alpha=3.4252509140, prefactor=1.0, coordinates=[0.1, 0.2], normalize=True)
    test2_2d = PrimitiveGaussian(alpha=3.4252509140, prefactor=1.0, coordinates=[0.5, 0.0], normalize=True)

    test1 = PrimitiveGaussian(alpha=3.4252509140, prefactor=1.0, coordinates=[0.0], normalize=True)
    test2 = PrimitiveGaussian(alpha=3.4252509140, prefactor=1.0, coordinates=[0.5], normalize=True)

    H1_1s = BasisFunction([test1, test2], [1.0, 0.0], coordinates=[0.1, 0.2])
    H1_2s = BasisFunction([test1, test2], [0.0, 1.0], coordinates=[0.5, 0.0])


    test3 = gaussian_product(test1, test2)
    test3_2d = gaussian_product(test1_2d, test2_2d)

    from matplotlib import pyplot as plt
    from scipy.integrate import quad, dblquad

    x_range = np.linspace(-1.5, 2, 100)
    plt.plot(x_range, [test1([x]) for x in x_range], label='test1')
    plt.plot(x_range, [test2([x]) for x in x_range], label='test2')
    plt.plot(x_range, [test1([x])*test2([x]) for x in x_range], label='test1*test2')

    plt.plot(x_range, [test3([x]) for x in x_range], '--', label='test3')
    a = test1.get_overlap_with(test2)
    test1_2 = gaussian_product(test1, test2)
    test1_2_2d = gaussian_product(test1_2d, test2_2d)


    def f(x):
        return test1([x]) * test2([x])

    def f2(x, y):
        return test1_2d([x, y]) * test2_2d([x, y])


    print(quad(f, -5, 5)[0], test1_2.integrate)
    print(dblquad(f2, -5, 5, lambda x: -5, lambda x: 5)[0], test1_2_2d.integrate, H1_1s.get_overlap_with(H1_2s))

    plt.legend()
    plt.show()
    exit()


    # STO-3G basis set for hydrogen molecule
    H_pg1a = PrimitiveGaussian(alpha=3.4252509140, coeff=0.1543289673)
    H_pg1b = PrimitiveGaussian(alpha=0.6239137298, coeff=0.5353281423)
    H_pg1c = PrimitiveGaussian(alpha=0.1688554040, coeff=0.4446345411)



    H1_1s = BasisFunction([H_pg1a, H_pg1b, H_pg1c], coordinates=[0, 0, 0])
    H2_1s = BasisFunction([H_pg1a, H_pg1b, H_pg1c], coordinates=[1.6, 0, 0])

    molecule = [H1_1s, H2_1s]
    S = overlap(molecule)
    print(S)

    print('--------------')

    T = kinetic(molecule)
    print(T)
    print('-----P---------')

    V = electron_nuclear(molecule,
                         nuclear_coordinates=[[0, 0, 0],
                                              [1.4, 0, 0]],
                         nuclear_charges=[1, 1])

    print(V)


    print('-----electron-electron---------')

    ee = electron_electron(molecule)

    print(ee)

    print('----nuclear nuclear-----')
    Vnn = nuclear_nuclear(nuclear_coordinates=[[0, 0, 0],
                                               [1.4, 0, 0]],
                          nuclear_charges=[1, 1])

    print(Vnn)


    exit()



    # 6-31g basis set for hydrogen molecule
    H_pg1a = PrimitiveGaussian(alpha=0.1873113696E+02, coeff=0.3349460434E-01)
    H_pg1b = PrimitiveGaussian(alpha=0.2825394365E+01, coeff=0.2347269535E+00)
    H_pg1c = PrimitiveGaussian(alpha=0.6401216923E+00, coeff=0.8137573261E+00)
    H_pg2a = PrimitiveGaussian(alpha=0.1612777588E+00, coeff=1.0000000)


    H1_1s = BasisFunction([H_pg1a, H_pg1b, H_pg1c], coordinates=[0, 0, 0])
    H1_2s = BasisFunction([H_pg2a], coordinates=[0, 0, 0])
    H2_1s = BasisFunction([H_pg1a, H_pg1b, H_pg1c], coordinates=[1.4, 0, 0])
    H2_2s = BasisFunction([H_pg2a], coordinates=[1.4, 0, 0])

    molecule = [H1_1s, H1_2s, H2_1s, H2_2s]
    S = overlap(molecule)

    print(S)

    print('--------------')
    T = kinetic(molecule)

    print(T)

    print('--------------')
    V = electron_nuclear(molecule,
                         nuclear_coordinates=[[0, 0, 0],
                                              [1.4, 0, 0]],
                         nuclear_charges=[1, 1])

    print(V)
