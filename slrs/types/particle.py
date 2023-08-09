import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slrs.types.calculation import Calculation

import numpy as np
import numpy.typing as npt
from scipy.constants import e, hbar, speed_of_light
from scipy import interpolate
import scipy.special as sp
import toml
from slrs.utils.conversions import nanometre_to_ev

# Mie code -> analytical for now

@dataclass
class Particle:
    radius_nm: float
    internal_dielectric_function: float

    def __post_init__(self):
        # Gold
        # eps_inf = 5.57
        # gamma = 0.018
        # plasma = 9.615
        # eps_inf = 5.3
        # plasma = 9.5
        # gamma = 0.013
        # self.internal_dielectric_function = lambda x: eps_inf - plasma ** 2 / nanometre_to_ev(x) / (nanometre_to_ev(x) + 1j * gamma)
        # SiC
        eps_inf = 6.52
        to = 0.09887739825047723 * (0.925 + 0.015)
        lo = 0.12026467248020425 * (0.925 + 0.015)
        gamma = 0.0004959367937328012
        self.internal_dielectric_function = lambda x: (
            eps_inf * (lo ** 2 - nanometre_to_ev(x) * (nanometre_to_ev(x) + 1j * gamma))
                / (to ** 2 - nanometre_to_ev(x) * (nanometre_to_ev(x) + 1j * gamma))
        )


    @classmethod
    def from_file(cls, path_to_configuration_file: str) -> None:

        extension = os.path.splitext(path_to_configuration_file)[-1].lower()
        if extension != ".toml":
            raise ValueError(f"expected a `.toml` file, received a `{extension}` file")

        path_to_configuration_file = Path(".").joinpath(path_to_configuration_file)
        if not os.path.exists(path_to_configuration_file):
            raise ValueError(f"file {path_to_configuration_file} not found")

        parsed_configuration = toml.load(path_to_configuration_file)['Particle']

        if (internal_dielectric_constant := parsed_configuration.get("internal_dielectric_constant")) is not None:
            return cls(parsed_configuration['radius_nm'], lambda x: internal_dielectric_constant)

        elif (tabulated_dielectric_function_file := parsed_configuration.get("internal_dielectric_function_file")) is not None:

            tabulated_dielectric_function = np.loadtxt(tabulated_dielectric_function_file)
            wavelengths_nm = tabulated_dielectric_function[:, 0] * 1000.0
            real_dielectric_function = tabulated_dielectric_function[:, 1]
            imag_dielectric_function = tabulated_dielectric_function[:, 2]
            real_dielectric_function = interpolate.InterpolatedUnivariateSpline(wavelengths_nm, real_dielectric_function)
            imag_dielectric_function = interpolate.InterpolatedUnivariateSpline(wavelengths_nm, imag_dielectric_function)
            return cls(parsed_configuration['radius_nm'], lambda x: real_dielectric_function(x) + 1j * imag_dielectric_function(x))


    @staticmethod
    def psi_n(
        n: int,
        z: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
	    return z * sp.spherical_jn(n,z, derivative=False)

    @staticmethod
    def psi_n_deriv(
        n: int,
        z: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
        return sp.spherical_jn(n,z) + z * sp.spherical_jn(n,z, derivative=True)

    @staticmethod
    def chi_n(
        n: int,
        z: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
	    return z * (sp.spherical_jn(n, z) + 1j * sp.spherical_yn(n,z))

    @staticmethod
    def chi_n_deriv(
        n: int,
        z: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.complex128]:
	    return (
            (sp.spherical_jn(n,z) + 1j * sp.spherical_yn(n,z))
                + z * (sp.spherical_jn(n, z, derivative=True) + 1j * sp.spherical_yn(n, z, derivative=True))
        )

    def a1(
        self,
        wavelengths_nm: npt.NDArray[np.float64],
        background_index: float
    ) -> npt.NDArray[np.complex128]:
        x = 2 * np.pi * self.radius_nm * background_index / wavelengths_nm
        m = np.sqrt(self.internal_dielectric_function(wavelengths_nm)) / background_index
        num   = m * self.psi_n(1, m * x) * self.psi_n_deriv(1, x) - self.psi_n(1, x) * self.psi_n_deriv(1, m * x)
        denom = m * self.psi_n(1, m * x) * self.chi_n_deriv(1, x) - self.chi_n(1, x) * self.psi_n_deriv(1, m * x)
        return num / denom

    def b1(
        self,
        wavelengths_nm: npt.NDArray[np.float64],
        background_index: float
    ) -> npt.NDArray[np.complex128]:
        x = 2 * np.pi * self.radius_nm * background_index / wavelengths_nm
        m = np.sqrt(self.internal_dielectric_function(wavelengths_nm)) / background_index
        num   = self.psi_n(1, m * x) * self.psi_n_deriv(1, x) - m * self.psi_n(1, x) * self.psi_n_deriv(1, m * x)
        denom = self.psi_n(1, m * x) * self.chi_n_deriv(1, x) - m * self.chi_n(1, x) * self.psi_n_deriv(1, m * x)
        return num / denom

    def scalar_electrical_polarisability(
        self,
        wavelengths_nm: npt.NDArray[np.float64],
        background_index: float
    ) -> npt.NDArray[np.complex128]:
        '''
        - wnm  : vacuum wavelength [nm]
        - eps_NP : complex permittivity at wavelength wmn
        - r    : NP radius [nm]
        - nout : background refractive index

        NOTE:
        - Left out eps_0 in the numerator
        '''
        a1 = self.a1(wavelengths_nm, background_index)
        wavevectors_im = 2 * np.pi * background_index / (1e-9 * wavelengths_nm)
        alpha = 6j * np.pi * a1 / wavevectors_im ** 3
        return alpha

    def scalar_magnetic_polarisability(
        self,
        wavelengths_nm: npt.NDArray[np.float64],
        background_index: float
    ) -> npt.NDArray[np.complex128]:
        '''
        - wnm  : vacuum wavelength [nm]
        - eps_NP : complex permittivity at wavelength wmn
        - r    : NP radius [nm]
        - nout : background refractive index

        NOTE:
        - Left out eps_0 in the numerator
        '''
        b1 = self.b1(wavelengths_nm, background_index)
        wavevectors_im = 2 * np.pi * background_index / (1e-9 * wavelengths_nm)
        alpha = 6j * np.pi * b1 / wavevectors_im ** 3
        return alpha

    def inverse_polarisability_tensor(
        self,
        wavelengths_nm: npt.NDArray[np.float64],
        background_index: float
    ) -> npt.NDArray[np.complex128]:
        scalar_electrical_polarisability = self.scalar_electrical_polarisability(wavelengths_nm, background_index)
        scalar_magnetic_polarisability = self.scalar_magnetic_polarisability(wavelengths_nm, background_index)

        # currently we assume a spherical particle

        tensor_polarisability = np.zeros((*wavelengths_nm.shape, 6, 6), dtype=np.complex128)
        for ii in range(3):
            tensor_polarisability[..., ii, ii] = scalar_electrical_polarisability
            tensor_polarisability[..., ii + 3, ii + 3] = scalar_magnetic_polarisability

        return np.linalg.inv(tensor_polarisability)

# from pathlib import Path
#
# diel_path = "Ag_EC.tab"
# path = Path(".").joinpath(diel_path)
# eps = np.loadtxt(diel_path)
# waves_nm = eps[:, 0] * 1000.0
# eps_re = eps[:, 1]
# eps_im = eps[:, 2]
# eps_re_fun = interpolate.InterpolatedUnivariateSpline(waves_nm, eps_re)
# eps_im_fun = interpolate.InterpolatedUnivariateSpline(waves_nm, eps_im)
# eps = lambda x: eps_re_fun(x) + 1j * eps_im_fun(x)
#
# particle = Particle(50.0, eps)
#
# wavelengths_nm = np.linspace(300.0, 950.0, 501)
# background_index = 1.0
#
# result = particle.polarisability(wavelengths_nm, background_index)
#
#
# import matplotlib.pyplot as plt
#
# plt.plot(wavelengths_nm, np.real(result), 'r-')
# plt.plot(wavelengths_nm, np.imag(result), 'b--')
# plt.show()
