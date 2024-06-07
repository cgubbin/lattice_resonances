import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    pass

import numpy as np
import numpy.typing as npt
from scipy import interpolate
import scipy.special as sp
import toml
from slrs.utils.conversions import nanometre_to_ev


# Particles are modelled as Mie spheres, characterised by a radius and a dielectric function
#
# This means we make two approximations. Firstly to get the mode frequency into the ballpark of the
# monopolar mode we use an artificially shifted dielectric function. At the moment this just entails
# taking the SiC dielectric functoin and shifting the phonon frequencies.
#
# Additionally we do not explicitly model the sphere-SPP interaction. In the experiment SiC pillars
# sit directly on the substrate supporting the SPP and the interaction strength will be some complex
# integral of the two eigenmodes. In our case it is altered by changing the radius of the Mie scatterer.
# Larger particles interact more strongly with the SPP.
#
# Be concious when adjusting these parameters that if the radius becomes too large, the mode frequency will
# begin to red shift from the small particle limit.
@dataclass
class Particle:
    radius_nm: float
    internal_dielectric_function: Callable[[float], complex]

    # Particles are created from tabulated data, however because we want to detune the phonon frequencies
    # to place the mode in the correct place we override the constructor here.
    #
    # It would obviously be better to (perhaps) read in a target frequency from the config file, and construct
    # a suitable dielectric function to match in the constructor.
    def __post_init__(self):
        eps_inf = 6.52
        to = 0.09887739825047723 * (0.925 + 0.015)
        lo = 0.12026467248020425 * (0.925 + 0.015)
        gamma = 0.0004959367937328012
        self.internal_dielectric_function = lambda x: (
            eps_inf
            * (lo**2 - nanometre_to_ev(x) * (nanometre_to_ev(x) + 1j * gamma))
            / (to**2 - nanometre_to_ev(x) * (nanometre_to_ev(x) + 1j * gamma))
        )

    @classmethod
    def from_file(cls, path_to_configuration_file: str) -> None:
        extension = os.path.splitext(path_to_configuration_file)[-1].lower()
        if extension != ".toml":
            raise ValueError(f"expected a `.toml` file, received a `{extension}` file")

        path = Path(".").joinpath(path_to_configuration_file)
        if not os.path.exists(path):
            raise ValueError(f"file {path} not found")

        parsed_configuration = toml.load(path)["Particle"]

        if (
            internal_dielectric_constant := parsed_configuration.get(
                "internal_dielectric_constant"
            )
        ) is not None:
            return cls(
                parsed_configuration["radius_nm"],
                lambda _: internal_dielectric_constant,
            )

        elif (
            tabulated_dielectric_function_file := parsed_configuration.get(
                "internal_dielectric_function_file"
            )
        ) is not None:
            tabulated_dielectric_function = np.loadtxt(
                tabulated_dielectric_function_file
            )
            wavelengths_nm = tabulated_dielectric_function[:, 0] * 1000.0
            real_dielectric_function = tabulated_dielectric_function[:, 1]
            imag_dielectric_function = tabulated_dielectric_function[:, 2]
            real_dielectric_function = interpolate.InterpolatedUnivariateSpline(
                wavelengths_nm, real_dielectric_function
            )
            imag_dielectric_function = interpolate.InterpolatedUnivariateSpline(
                wavelengths_nm, imag_dielectric_function
            )
            return cls(
                parsed_configuration["radius_nm"],
                lambda x: real_dielectric_function(x)
                + 1j * imag_dielectric_function(x),
            )

    @staticmethod
    def psi_n(n: int, z: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        return z * sp.spherical_jn(n, z, derivative=False)

    @staticmethod
    def psi_n_deriv(
        n: int, z: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        return sp.spherical_jn(n, z) + z * sp.spherical_jn(n, z, derivative=True)

    @staticmethod
    def chi_n(n: int, z: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        return z * (sp.spherical_jn(n, z) + 1j * sp.spherical_yn(n, z))

    @staticmethod
    def chi_n_deriv(
        n: int, z: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        return (sp.spherical_jn(n, z) + 1j * sp.spherical_yn(n, z)) + z * (
            sp.spherical_jn(n, z, derivative=True)
            + 1j * sp.spherical_yn(n, z, derivative=True)
        )

    # A1 Mie coefficient
    def a1(
        self, wavelengths_nm: npt.NDArray[np.float64], background_index: float
    ) -> npt.NDArray[np.complex128]:
        x = 2 * np.pi * self.radius_nm * background_index / wavelengths_nm
        m = (
            np.sqrt(self.internal_dielectric_function(wavelengths_nm))
            / background_index
        )
        num = m * self.psi_n(1, m * x) * self.psi_n_deriv(1, x) - self.psi_n(
            1, x
        ) * self.psi_n_deriv(1, m * x)
        denom = m * self.psi_n(1, m * x) * self.chi_n_deriv(1, x) - self.chi_n(
            1, x
        ) * self.psi_n_deriv(1, m * x)
        return num / denom

    # B1 Mie coefficient
    def b1(
        self, wavelengths_nm: npt.NDArray[np.float64], background_index: float
    ) -> npt.NDArray[np.complex128]:
        x = 2 * np.pi * self.radius_nm * background_index / wavelengths_nm
        m = (
            np.sqrt(self.internal_dielectric_function(wavelengths_nm))
            / background_index
        )
        num = self.psi_n(1, m * x) * self.psi_n_deriv(1, x) - m * self.psi_n(
            1, x
        ) * self.psi_n_deriv(1, m * x)
        denom = self.psi_n(1, m * x) * self.chi_n_deriv(1, x) - m * self.chi_n(
            1, x
        ) * self.psi_n_deriv(1, m * x)
        return num / denom

    def scalar_electrical_polarisability(
        self, wavelengths_nm: npt.NDArray[np.float64], background_index: float
    ) -> npt.NDArray[np.complex128]:
        a1 = self.a1(wavelengths_nm, background_index)
        wavevectors_im = 2 * np.pi * background_index / (1e-9 * wavelengths_nm)
        alpha = 6j * np.pi * a1 / wavevectors_im**3
        return alpha

    def scalar_magnetic_polarisability(
        self, wavelengths_nm: npt.NDArray[np.float64], background_index: float
    ) -> npt.NDArray[np.complex128]:
        b1 = self.b1(wavelengths_nm, background_index)
        wavevectors_im = 2 * np.pi * background_index / (1e-9 * wavelengths_nm)
        alpha = 6j * np.pi * b1 / wavevectors_im**3
        return alpha

    def inverse_polarisability_tensor(
        self, wavelengths_nm: npt.NDArray[np.float64], background_index: float
    ) -> npt.NDArray[np.complex128]:
        scalar_electrical_polarisability = self.scalar_electrical_polarisability(
            wavelengths_nm, background_index
        )
        scalar_magnetic_polarisability = self.scalar_magnetic_polarisability(
            wavelengths_nm, background_index
        )

        tensor_polarisability = np.zeros(
            (*wavelengths_nm.shape, 6, 6), dtype=np.complex128
        )
        for ii in range(3):
            tensor_polarisability[..., ii, ii] = scalar_electrical_polarisability
            tensor_polarisability[..., ii + 3, ii + 3] = scalar_magnetic_polarisability

        return np.linalg.inv(tensor_polarisability)
