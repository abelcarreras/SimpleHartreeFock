import numpy as np
from copy import deepcopy


def gaussian_product(g_1, g_2):
    """
    Calculate the product of two Gaussian primitives. (s functions only)
    """
    alpha = g_1.alpha + g_2.alpha
    prefactor = g_1.prefactor * g_2.prefactor
    p = g_1.alpha * g_2.alpha

    PG = g_1.coordinates - g_2.coordinates
    coordinates = (g_1.coordinates * g_1.alpha + g_2.coordinates * g_2.alpha) / alpha
    K = np.exp(-p/alpha * np.dot(PG, PG))

    return PrimitiveGaussian(alpha, K*prefactor, coordinates, normalize=False)


class PrimitiveGaussian:
    def __init__(self, alpha, prefactor=1.0, coordinates=(0, 0, 0), normalize=True):
        n_dim = len(coordinates)
        self.alpha = alpha
        self.prefactor = prefactor
        self.coordinates = np.array(coordinates)

        # normalize primitive such that <prim|prim> = 1
        if normalize:
            norm = prefactor * np.sqrt(np.pi/(2*self.alpha))**(len(coordinates))
            self.prefactor = prefactor / np.sqrt(norm)

        self.integrate = self.prefactor * (np.pi / self.alpha)**(n_dim/2)

    def __call__(self, value):
        value = np.array(value)
        return self.prefactor * np.exp(-self.alpha * np.sum((value - self.coordinates)**2))

    def __mul__(self, other):
        return gaussian_product(self, other)


class BasisFunction:
    def __init__(self, primitive_gaussians, coefficients, coordinates=None):
        primitive_gaussians = deepcopy(primitive_gaussians)
        if coordinates is not None:
            for primitive in primitive_gaussians:
                primitive.coordinates = np.array(coordinates)

        self.primitive_gaussians = primitive_gaussians
        self.coefficients = coefficients

    def get_number_of_primitives(self):
        return len(self.primitive_gaussians)

    def set_coordinates(self, coordinates):
        for primitive in self.primitive_gaussians:
            primitive.coordinates = np.array(coordinates)

    @property
    def integrate(self):
        return sum([coef * prim.integrate for coef, prim in zip(self.coefficients, self.primitive_gaussians)])

    def __mul__(self, other):
        if isinstance(other, float):
            return BasisFunction(self.primitive_gaussians, [coef * other for coef in self.coefficients])

        elif isinstance(other, BasisFunction):

            primitive_gaussians = []
            coefficients = []

            for primitive_1, coeff_1 in zip(self.primitive_gaussians, self.coefficients):
                for primitive_2, coeff_2 in zip(other.primitive_gaussians, other.coefficients):
                    primitive_gaussians.append(primitive_1 * primitive_2)
                    coefficients.append(coeff_1 * coeff_2)

            return BasisFunction(primitive_gaussians, coefficients)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return BasisFunction(self.primitive_gaussians + other.primitive_gaussians,
                             self.coefficients + other.coefficients)
