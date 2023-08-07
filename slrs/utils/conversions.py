import numpy as np
import numpy.typing as npt
from scipy.constants import e, hbar, speed_of_light


def wavenumber_to_nanometre(wavenumber_icm: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    wavenumber_icm = np.asarray(wavenumber_icm)
    wavenumber_im = 2 * np.pi / 0.01 * wavenumber_icm
    wavelength = 2 * np.pi / wavenumber_im
    return wavelength / 1e-9

def nanometre_to_wavenumber(wavelength_nm: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    wavelength_m = np.asarray(wavelength_nm) * 1e-9
    return 0.01 / wavelength_m

def ev_to_nanometre(energy_ev: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    energy_joule = np.asarray(energy_ev) * e
    angular_frequency = energy_joule / hbar
    wavevector_im = angular_frequency / speed_of_light
    wavelength_m = 2. * np.pi / wavevector_im
    return wavelength_m / 1e-9

def nanometre_to_ev(wavelength_nm: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    wavelength_m = np.asarray(wavelength_nm) * 1e-9
    wavevector_im = 2 * np.pi / wavelength_m
    angular_frequency = speed_of_light * wavevector_im
    energy_joule = hbar * angular_frequency
    return energy_joule / e

