import numpy as np
from copy import deepcopy
from scipy import special


def boys(x, n):
    if x == 0:
        return 1/(2*n+1)
    else:
        return special.gammainc(n+0.5, x) * special.gamma(n+0.5) * (1/(2*x**(n+0.5)))


def gaussian_product(g_1, g_2):
    """
    Calculate the product of two Gaussian primitives. (s functions only)
    """

    alpha = g_1.alpha + g_2.alpha
    prefactor = g_1.prefactor * g_2.prefactor
    p = g_1.alpha * g_2.alpha

    diff = np.linalg.norm(g_1.coordinates - g_2.coordinates) ** 2
    coordinates = (g_1.coordinates * g_1.alpha + g_2.coordinates * g_2.alpha) / alpha
    K = np.exp(-p/alpha * diff)

    return PrimitiveGaussian(alpha, K*prefactor, coordinates, normalize=False)


class PrimitiveGaussian:
    def __init__(self, alpha, prefactor=1.0, coordinates=(0, 0, 0), normalize=True):
        self.alpha = alpha
        self.prefactor = prefactor
        self.coordinates = np.array(coordinates)
        # self.A = (2 * alpha/np.pi) ** 0.75  # + other terms for l1 l2 l3

        # normalize primitive such that <prim|prim> = 1
        if normalize:
            norm = prefactor * np.sqrt(np.pi/(2*self.alpha))**(len(coordinates))
            self.prefactor = prefactor / np.sqrt(norm)

        self.integrate = self.prefactor * np.sqrt(np.pi / (self.alpha)) ** (len(coordinates))

    def __call__(self, value):
        value = np.array(value)
        return self.prefactor * np.exp(-self.alpha * np.sum((value - self.coordinates)**2))

    def get_overlap_with(self, other):

        return gaussian_product(self, other).integrate

    def get_kinetic_with(self, other):

        g_s = gaussian_product(self, other)
        PG = g_s.coordinates - other.coordinates

        return g_s.integrate * (3 * other.alpha - 2 * other.alpha ** 2 * (3/(2*g_s.alpha) + np.dot(PG, PG)))

    def get_potential_with(self, other, position, charge):

        g_s = gaussian_product(self, other)
        PG = g_s.coordinates - position

        return -charge * g_s.integrate * 2 * (np.pi / g_s.alpha) ** (-1/2) * boys(g_s.alpha * np.dot(PG, PG), 0)


class BasisFunction:
    def __init__(self, primitive_gaussians, coefficients, coordinates=(0, 0, 0)):
        primitive_gaussians = deepcopy(primitive_gaussians)
        for primitive in primitive_gaussians:
            primitive.coordinates = np.array(coordinates)

        self.primitive_gaussians = primitive_gaussians
        self.coefficients = coefficients

    def get_number_of_primitives(self):
        return len(self.primitive_gaussians)

    def set_coordinates(self, coordinates):
        for primitive in self.primitive_gaussians:
            primitive.coordinates = np.array(coordinates)

    def get_overlap_with(self, other):

        s = 0
        for primitive_1, coeff_1 in zip(self.primitive_gaussians, self.coefficients):
            for primitive_2, coeff_2 in zip(other.primitive_gaussians, self.coefficients):
                s += coeff_1 * coeff_2 * primitive_1.get_overlap_with(primitive_2)

        return s

    def get_kinetic_with(self, other):

        t = 0
        for primitive_1, coeff_1 in zip(self.primitive_gaussians, self.coefficients):
            for primitive_2, coeff_2 in zip(other.primitive_gaussians, self.coefficients):
                t += coeff_1 * coeff_2 * primitive_1.get_kinetic_with(primitive_2)

        return t

    def get_potential_with(self, other, position, charge):

        v = 0
        for primitive_1, coeff_1 in zip(self.primitive_gaussians, self.coefficients):
            for primitive_2, coeff_2 in zip(other.primitive_gaussians, self.coefficients):
                v += coeff_1 * coeff_2 * primitive_1.get_potential_with(primitive_2, position, charge)
        return v
