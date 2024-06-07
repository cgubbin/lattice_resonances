# Example script to scan the number of resonators in an array

from slrs.types.calculation import Calculation
from slrs.types.lattice import n_spp
from slrs.utils.conversions import nanometre_to_wavenumber
import logging
import numpy as np
import toml
from scipy.constants import speed_of_light

logging.basicConfig(level=logging.INFO)

period_nm = 30500.0

for extent in np.arange(5, 10, 5):
    data = toml.load("inputs/input_tm_2d_sic.toml")
    data["Lattice"]["finite_extent"] = [extent, extent]
    data["Lattice"]["lengths"] = [period_nm, period_nm]
    data["Grid"]["maximum_wavevector_im"] = 2 * np.pi / (6000 * 1e-9)
    data["Grid"]["minimum_wavevector_im"] = -2 * np.pi / (6000 * 1e-9)

    probe = data["Field"]["probe"]

    f = open("inputs/input_tm_2d_sic.toml", "w")
    toml.dump(data, f, encoder=toml.TomlNumpyEncoder())
    f.close()

    calculation = Calculation.from_file("inputs/input_tm_2d_sic.toml")
    extinction_tm = calculation.cross_sections()

    _, wavelengths_nm = calculation.grid.generate()
    unique_wavelengths_nm = wavelengths_nm[:, 0]
    effective_index = n_spp(unique_wavelengths_nm)
    wavevectors = 2 * np.pi / (1e-9 * unique_wavelengths_nm) * np.real(effective_index)
    scaled_wavevectors = wavevectors / (
        np.pi / (calculation.lattice.lengths_nm[0] * 1e-9)
    )

    xx, yy = calculation.plot_grid()

    unique_xx = xx[0] * calculation.lattice.background_index
    omega = speed_of_light * unique_xx / calculation.lattice.background_index
    wavelength_nm = speed_of_light / (omega / (2 * np.pi)) / 1e-9
    wavenumber_icm = np.abs(nanometre_to_wavenumber(wavelength_nm))

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # levels = np.linspace(0.0, 200, 100)
    levels = 1000

    fig, ax = plt.subplots(figsize=(12, 12))
    xx = xx / (np.pi / (calculation.lattice.lengths_nm[0] * 1e-9))
    unique_xx = unique_xx / (np.pi / (calculation.lattice.lengths_nm[0] * 1e-9))
    im1 = ax.contourf(xx, yy, extinction_tm, levels=levels, extend="both")

    ax.plot(scaled_wavevectors, nanometre_to_wavenumber(unique_wavelengths_nm))
    # ax.plot(2.0 - scaled_wavevectors, nanometre_to_wavenumber(unique_wavelengths_nm))
    ax.set_ylabel("Wavenumber (1/cm)")
    ax.set_xlabel("Wavevector ($\pi / a$)")
    # ax.set_xlim(-2.0, 2.0)
    ax.plot(unique_xx, wavenumber_icm)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax, orientation="vertical")

    ax.set_ylim(
        data["Grid"]["minimum_wavenumber_icm"],
        data["Grid"]["maximum_wavenumber_icm"],
    )

    plt.savefig(f"results/{period_nm}nm_{extent}x{extent}_{probe}.png")
    # plt.show()

    np.savez(
        f"results/{period_nm}nm_{extent}x{extent}_{probe}", x=xx, y=yy, z=extinction_tm
    )
