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

e_list = []
for r_angs in np.arange(0.2, 3.2, 0.1):
    r = r_angs / 0.529  # convert angstrom to bohr

    # H2 molecule define coordinates in basis set
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

    electronic_energy = scf_cycle(basis_set, S, T, Vne, Vee)
    total_energy = electronic_energy + nuclear_energy

    e_list.append(total_energy)

plt.plot(np.arange(0.2, 3.2, 0.1), e_list)
plt.xlabel('Distance (Angstrom)')
plt.ylabel('Total Energy (Hartree)')
plt.show()
