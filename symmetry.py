from basis import PrimitiveGaussian, BasisFunction
from operators import overlap, kinetic, electron_nuclear, electron_electron, nuclear_nuclear
from scf import scf_cycle
import numpy as np
import matplotlib.pyplot as plt


# STO-3G basis set primitives for hydrogen molecule
H_pg1a = PrimitiveGaussian(alpha=3.4252509140, normalize=True)
H_pg1b = PrimitiveGaussian(alpha=0.6239137298, normalize=True)
H_pg1c = PrimitiveGaussian(alpha=0.1688554040, normalize=True)

H1_1s = BasisFunction([H_pg1a, H_pg1b, H_pg1c], coefficients=[0.1543289673, 0.5353281423, 0.4446345411])
H2_1s = BasisFunction([H_pg1a, H_pg1b, H_pg1c], coefficients=[0.1543289673, 0.5353281423, 0.4446345411])

r = 0.7 / 0.529  # convert angstrom to bohr

# H2 molecule electronic basis set
H1_1s.set_coordinates([0, 0, 0])
H2_1s.set_coordinates([r, 0, 0])
basis_set = [H1_1s, H2_1s]

# Compute components
S = overlap(basis_set)
T = kinetic(basis_set)
Vne = electron_nuclear(basis_set,
                       nuclear_coordinates=[[0, 0, 0],
                                            [r, 0, 0]],
                       nuclear_charges=[1, 1])

Vee = electron_electron(basis_set)
nuclear_energy = nuclear_nuclear(nuclear_coordinates=[[0, 0, 0],
                                                      [r, 0, 0]],
                                 nuclear_charges=[1, 1])

electronic_energy, density_matrix, mo_orbitals = scf_cycle(basis_set, S, T, Vne, Vee, extra_output=True)


def get_overlap_matrix(basis_set):
    return np.array([[(basis1*basis2).integrate
                      for basis1 in basis_set]
                     for basis2 in basis_set])

def get_overlap_matrix_density(basis_set_1, basis_set_2, density_matrix):
    n = len(basis_set)
    s = np.zeros((n, n, n, n))

    for i, basis1 in enumerate(basis_set_1):
        for j, basis2 in enumerate(basis_set_1):
            for k, basis3 in enumerate(basis_set_2):
                for l, basis4 in enumerate(basis_set_2):
                    dens_prod = density_matrix[i, j] * density_matrix[k ,l]
                    basis_prod = basis1 * basis2 * basis3 * basis4
                    s[i, j, k, l] = basis_prod.integrate * dens_prod

    return np.sum(s)


print('-----------')
print('mo_orbitals')
print(mo_orbitals)
print('SCF energy: ', electronic_energy + nuclear_energy)
print('density matrix')
print(density_matrix)
print('overlap')
print(S)


def rotate_coordinates(coordinates, angle, axis, center):

    coordinates = np.array(coordinates) - np.array(center)

    cos_term = 1 - np.cos(angle)
    rot_matrix = [[axis[0]**2*cos_term + np.cos(angle), axis[0]*axis[1]*cos_term - axis[2]*np.sin(angle), axis[0]*axis[2]*cos_term + axis[1]*np.sin(angle)],
                  [axis[1]*axis[0]*cos_term + axis[2]*np.sin(angle), axis[1]**2*cos_term + np.cos(angle), axis[1]*axis[2]*cos_term + axis[0]*np.sin(angle)],
                  [axis[2]*axis[0]*cos_term + axis[1]*np.sin(angle), axis[1]*axis[2]*cos_term + axis[0]*np.sin(angle), axis[2]**2*cos_term + np.cos(angle)]]

    return np.dot(coordinates, rot_matrix) + np.array(center).tolist()


def rotate_basis_set(axis, angle, basis_set, center=(0, 0, 0)):
    rotated_basis_set = []
    for basis in basis_set:
        rotated_basis = []
        for prim in basis.primitive_gaussians:
            coordinates = rotate_coordinates(prim.coordinates, angle, axis, center)
            rotated_basis.append(PrimitiveGaussian(alpha=prim.alpha, coordinates=coordinates))
        rotated_basis_set.append(BasisFunction(rotated_basis, coefficients=basis.coefficients))

    return rotated_basis_set


self_similarity = get_overlap_matrix_density(basis_set, basis_set, density_matrix)
print('self_similarity', self_similarity)

orbital_1 = H1_1s * mo_orbitals[0][0] + H2_1s * mo_orbitals[0][1]
orbital_2 = H1_1s * mo_orbitals[1][0] + H2_1s * mo_orbitals[1][1]

print('integrate o1*o1', (orbital_1*orbital_1).integrate)
print('integrate o2*o2', (orbital_2*orbital_2).integrate)
print('integrate o1*o2', (orbital_2*orbital_1).integrate)


measure_list = []
for angle in np.linspace(0, 2*np.pi, 100):
    basis_set_r = rotate_basis_set([0, 1, 1], angle, basis_set, center=[r / 2, 0, 0])
    measure_list.append(get_overlap_matrix_density(basis_set, basis_set_r, density_matrix) / self_similarity)

plt.plot(np.linspace(0, 2*np.pi, 100), measure_list)
plt.show()

